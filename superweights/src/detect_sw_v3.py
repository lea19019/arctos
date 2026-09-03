"""Super-weight detector -- FROZEN v3, kept for comparison. Do not edit.

First version written against the paper's stated conditions (both sides
outlier, suppression stopping rule, plausibility bound). On OLMo-1B it
returns 9 candidates of which 1 is causally real -- much better than v2's
138, still over-generating, because it measures down_proj output while the
paper's Figure 4 criterion is persistence in the RESIDUAL STREAM. That is
what v5 fixes. Default output goes to results/v3/.

Lineage: olmo_sw.py (v0) -> v1 -> v2 -> v3 -> detect_sw.py (v5)
(there is no v4: v4 is v1's candidates re-scored on wikitext-2, not a
detector -- see slurm/reablate.sh)

Original docstring follows.

---

Model-agnostic super-weight detector (Yu et al. 2024, section 3.1).

Hooks every layer's `mlp.down_proj`, reads the coordinate off the activation
spikes, then peels: zero what it found, run again, until nothing survives.
Writes JSON that ablate_sw.py consumes. Zeroed weights are restored at exit.

Method recap:  Y[t,j] = sum_k X[t,k] * W[j,k]
    a super activation at Y[t,j] is produced by an outlier product
    X[t,k] * W[j,k]; that (j, k) is the super weight.

## v3 (2026-09-02) — written against the paper, not against our guesses

v1 returned 11 of 21 Table 2 coordinates; v2 fixed localization and returned
20 of 21 but with 1,945 extra candidates and no new causal hits. Reading
Yu et al. properly (rather than just §3.1's formula) turned up four stated
conditions neither version implemented. All four are here:

1. **Both sides must be outliers.** §3.1: "If X_ik *and* W_jk are both
   outliers that are much larger than other values, Y_ij will be dominated
   by their product." v1/v2 only ever checked the output side, so any tiny
   weight sharing a column with a big input spike looked dominant -- the
   "orphaned input spike" false positive. `--top-w` / `--top-x` require each
   side to be an outlier in its own right. This is necessary, not
   sufficient: `coord_check.py` shows all 21 Table 2 coordinates rank 1-6 by
   |W| in their matrix, but plenty of rank-1-6 weights are causally inert.
2. **The loop stops when the spikes are gone.** §3.1: "repeat the above
   process, until the magnitudes of large maximum activations are greatly
   suppressed." v1/v2 stopped on "no candidate passes" instead, so v2 ran
   all 15 rounds on OLMo-1B while max|Y| had already collapsed 419 -> 20 by
   round 2. `--suppression` implements the paper's rule.
3. **A plausibility bound.** §3.1: "Most of the models we have examined have
   no more than three super weights", six at most (Phi-3). `--max-sw` stops
   and warns rather than emitting hundreds.
4. **Per-channel, not per-layer, towering.** v2 gated the layer and then took
   its top-j channels, so ordinary channels rode in behind a towering layer.

The remaining thresholds are still ours and still uncalibrated -- the paper
has no acceptance criterion at all, it reads spikes off a plot. Replacing
them with a null is Phase 0. Detection is *suggestive*; only causal ablation
(ablate_sw.py) confirms a super weight.

Appendix A.5 notes super activations "are typically observed on the first
token of an input sequence"; we record whether that held rather than
enforcing it, since it is an observation about their models, not a rule.

Usage (login node pre-downloads the model; compute nodes have no internet):
    uv run src/detect_sw.py --model allenai/OLMo-1B-0724-hf
    uv run src/detect_sw.py --model huggyllama/llama-13b --top-j 8
"""

import argparse
import datetime
import json
import statistics
from pathlib import Path

import torch
import transformers
from provenance import git_sha
from transformers import AutoModelForCausalLM, AutoTokenizer

TOWERING_FACTOR = 10   # layer's max|Y| vs the median layer -- ours, arbitrary
CHANNEL_FACTOR = 50    # |Y[t,j]| vs the median channel at that token -- ours
TOP_J = 4              # output channels examined per layer
TOP_W = 1000           # |W[j,k]| must be this rank or better within the layer
TOP_X = 10             # |X[t,k]| must be this rank or better at that token
EXPLAIN_FRAC = 0.80    # prefix must explain at least this share of Y[t,j]
OVERSHOOT = 1.25       # ... and at most this much (a prefix over 1 cancels)
MAX_K = 4              # cap on contributors returned for one output channel
MAX_ROUNDS = 15        # safety cap on the peeling loop
SUPPRESSION = 0.10     # stop once max|Y| falls to this share of round 0's
MAX_SW = 8             # stop and warn past this many (paper: at most six)

DEFAULT_PROMPT = "Language modeling is "

# Load in the checkpoint's own precision by default. The Llama and Mistral
# repos are float16 and OLMo is float32; forcing everything to bf16 rounds
# small weights differently from the paper, and the OLMo-1B result turned on
# a weight of 0.0018. CPU gets float32 because fp16 matmul is unsupported there.
DTYPES = {"auto": "auto", "bf16": torch.bfloat16,
          "fp16": torch.float16, "fp32": torch.float32}


def get_layers(model):
    """The decoder layer list, on the architectures we support."""
    try:
        layers = model.model.layers
        _ = layers[0].mlp.down_proj
        return layers
    except AttributeError as e:
        raise SystemExit(
            f"Unsupported architecture ({type(model).__name__}): expected "
            f"model.model.layers[i].mlp.down_proj. Adapt get_layers()."
        ) from e


def make_hook(layer_idx, store):
    """Keep this layer's down_proj input and output for the whole prompt.

    v1 reduced to an argmax inside the hook, which threw away everything
    needed to see a second super weight in the same layer. The tensors are
    small (a few tokens x a few thousand channels) so we keep them whole.
    """
    def hook(module, inp, out):
        store[layer_idx] = (inp[0][0].detach().float().cpu(),
                            out[0].detach().float().cpu())
    return hook


def detection_pass(model, enc):
    """One forward pass with hooks on every down_proj."""
    store = {}
    handles = [
        layer.mlp.down_proj.register_forward_hook(make_hook(i, store))
        for i, layer in enumerate(get_layers(model))
    ]
    with torch.no_grad():
        model(**enc)
    for h in handles:
        h.remove()
    return store


def outlier_cut(v, n):
    """Value of the n-th largest |entry| -- the bar for "is an outlier"."""
    flat = v.abs().flatten()
    n = min(n, flat.numel())
    return flat.topk(n).values[-1].item()


def analyze_layer(layer_idx, X, Y, W, args):
    """Decompose this layer's biggest output spike into the weights making it.

    Returns one record for the layer, carrying every (j, k) candidate that
    satisfies the paper's conditions on BOTH sides of the product.
    """
    # the token carrying the layer's largest |output| -- the super activation
    t = int(Y.abs().max(dim=-1).values.argmax())
    x, y = X[t], Y[t]
    max_y = y.abs().max().item()

    # the bars each side must clear to count as an outlier (paper 3.1)
    Wc = W.float().cpu()
    w_cut = outlier_cut(Wc, args.top_w)
    x_cut = outlier_cut(x, args.top_x)
    # a super activation towers over the OTHER CHANNELS at its own token,
    # not merely over other layers -- v2 gated the layer and let ordinary
    # channels ride in behind it
    y_med = y.abs().median().item()
    y_cut = args.channel_factor * y_med

    candidates, rejected = [], []
    for j in y.abs().topk(min(args.top_j, y.numel())).indices.tolist():
        target = y[j].item()
        if abs(target) < y_cut:
            rejected.append({"j": j, "why": "channel does not tower"})
            continue

        contrib = x * Wc[j]              # signed contributions; these sum to y[j]
        order = contrib.abs().argsort(descending=True).tolist()

        prefix, taken = 0.0, []
        for rank, k in enumerate(order[:args.max_k]):
            prefix += contrib[k].item()
            taken.append({
                "j": j, "k": k, "rank_in_channel": rank,
                "weight": Wc[j, k].item(), "x": x[k].item(),
                "contribution": contrib[k].item(), "y_channel": target,
                "share": contrib[k].item() / target,
                "explained_by_prefix": prefix / target,
                "w_outlier": abs(Wc[j, k].item()) >= w_cut,
                "x_outlier": abs(x[k].item()) >= x_cut,
            })
            if abs(prefix) >= args.explain_frac * abs(target):
                break

        ratio = abs(prefix / target) if target else float("inf")
        if not (args.explain_frac <= ratio <= args.overshoot):
            rejected.append({"j": j, "why": f"prefix explains {ratio:.2f}"})
            continue
        for c in taken:
            if c["w_outlier"] and c["x_outlier"]:
                candidates.append(c)
            else:
                rejected.append({"j": j, "k": c["k"], "why": "not both outliers",
                                 "w_outlier": c["w_outlier"],
                                 "x_outlier": c["x_outlier"]})

    return {"layer": layer_idx, "token": t, "first_token": t == 0,
            "max_y": max_y, "max_x": x.abs().max().item(),
            "w_cut": w_cut, "x_cut": x_cut, "y_cut": y_cut,
            "candidates": candidates, "rejected": rejected}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--revision", default=None,
                    help="pin an exact model commit (recorded in the output)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: results/<model>_found.json)")
    ap.add_argument("--dtype", default="auto", choices=sorted(DTYPES),
                    help="'auto' = the checkpoint's own torch_dtype")
    ap.add_argument("--top-j", type=int, default=TOP_J)
    ap.add_argument("--top-w", type=int, default=TOP_W,
                    help="|W[j,k]| must rank this high in its own matrix")
    ap.add_argument("--top-x", type=int, default=TOP_X,
                    help="|X[t,k]| must rank this high at the spike token")
    ap.add_argument("--explain-frac", type=float, default=EXPLAIN_FRAC)
    ap.add_argument("--overshoot", type=float, default=OVERSHOOT)
    ap.add_argument("--max-k", type=int, default=MAX_K)
    ap.add_argument("--towering-factor", type=float, default=TOWERING_FACTOR)
    ap.add_argument("--channel-factor", type=float, default=CHANNEL_FACTOR)
    ap.add_argument("--suppression", type=float, default=SUPPRESSION,
                    help="stop once max|Y| falls to this share of round 0's")
    ap.add_argument("--max-sw", type=int, default=MAX_SW)
    ap.add_argument("--peel", default="all", choices=["all", "loudest"],
                    help="'loudest' reproduces v1's one-per-round behaviour")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    enc = tokenizer([args.prompt], return_tensors="pt",
                    return_token_type_ids=False).to(device)

    layers = get_layers(model)
    found, seen, rounds_log = [], set(), []
    max_y_round0, stop_reason = None, f"hit --max-rounds ({args.max_rounds})"

    for rnd in range(args.max_rounds):
        store = detection_pass(model, enc)
        records = [analyze_layer(i, X, Y, layers[i].mlp.down_proj.weight, args)
                   for i, (X, Y) in sorted(store.items())]
        median_max_y = statistics.median(r["max_y"] for r in records)

        # the only layer-level gate left: is this spike towering over the rest?
        survivors = [r for r in records
                     if r["max_y"] > args.towering_factor * median_max_y]
        for r in records:
            r["towering"] = r["max_y"] > args.towering_factor * median_max_y

        # the paper's stopping rule: peel "until the magnitudes of large
        # maximum activations are greatly suppressed"
        max_y_now = max(r["max_y"] for r in records)
        if max_y_round0 is None:
            max_y_round0 = max_y_now
        elif max_y_now <= args.suppression * max_y_round0:
            stop_reason = (f"max|Y| suppressed {max_y_round0:.1f} -> "
                           f"{max_y_now:.1f} ({max_y_now / max_y_round0:.3f} "
                           f"of round 0)")
            print(f"Round {rnd}: {stop_reason} -- stopping.")
            break

        n_cand = sum(len(r["candidates"]) for r in survivors)
        print(f"Round {rnd}: {len(survivors)} of {len(records)} layers tower "
              f"(median max|Y| = {median_max_y:.2f}), {n_cand} candidate(s)")
        for r in survivors:
            for c in r["candidates"]:
                print(f"  layer {r['layer']:2d}  W[{c['j']},{c['k']}] = "
                      f"{c['weight']:+.4f}  max|Y|={r['max_y']:.1f}  "
                      f"share={c['share']:+.3f} (prefix {c['explained_by_prefix']:+.3f})")
        rounds_log.append({"round": rnd, "median_max_y": median_max_y,
                           "records": records})

        # candidates not already peeled, loudest layer first
        fresh = [(r, c) for r in sorted(survivors, key=lambda r: -r["max_y"])
                 for c in r["candidates"]
                 if (r["layer"], c["j"], c["k"]) not in seen]
        if not fresh:
            stop_reason = "no new candidate passed"
            print(f"Round {rnd}: nothing new -- stopping.")
            break
        # Plausibility bound (paper: at most six per model). This is a signal
        # that the thresholds are too loose, NOT a reason to lose the round --
        # v3's first run broke here before recording and returned nothing.
        over_budget = len(found) + len(fresh) > args.max_sw
        if over_budget:
            stop_reason = (f"exceeded --max-sw ({args.max_sw}) at round {rnd}; "
                           f"the paper reports at most six per model, so treat "
                           f"these as over-generated")
            print(f"Round {rnd}: {stop_reason}")
        if args.peel == "loudest":
            fresh = fresh[:1]

        for r, c in fresh:
            coord = (r["layer"], c["j"], c["k"])
            seen.add(coord)
            found.append({"layer": r["layer"], "j": c["j"], "k": c["k"],
                          "value": c["weight"], "round": rnd,
                          "share": c["share"], "max_y": r["max_y"]})
            print(f"Round {rnd}: zeroing L{coord[0]}[{coord[1]},{coord[2]}] "
                  f"= {c['weight']:+.4f}")
            with torch.no_grad():
                layers[coord[0]].mlp.down_proj.weight[coord[1], coord[2]] = 0.0

        if over_budget:
            break

    # restore every zeroed weight -- leave the model as we found it
    with torch.no_grad():
        for f in found:
            layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]

    # provenance: enough to know exactly what produced this file
    result = {
        "model": args.model,
        "revision_requested": args.revision,
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "prompt": args.prompt,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_sha": git_sha(),
        "device": device,
        "dtype_requested": args.dtype,
        "dtype": str(model.dtype),
        "detector_version": 3,
        "stop_reason": stop_reason,
        "max_y_round0": max_y_round0,
        "params": {"top_j": args.top_j, "top_w": args.top_w,
                   "top_x": args.top_x, "explain_frac": args.explain_frac,
                   "overshoot": args.overshoot, "max_k": args.max_k,
                   "towering_factor": args.towering_factor,
                   "channel_factor": args.channel_factor,
                   "suppression": args.suppression, "max_sw": args.max_sw,
                   "peel": args.peel, "max_rounds": args.max_rounds},
        "found": found,
        "rounds": rounds_log,
    }
    out = Path(args.out) if args.out else (
        Path("results/v3") / (args.model.replace("/", "_") + "_found.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFound {len(found)} candidate(s). Written to {out}")
    print("Candidates are UNVERIFIED until ablate_sw.py confirms damage.")


if __name__ == "__main__":
    main()
