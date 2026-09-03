"""Super-weight detector -- FROZEN v2, kept for comparison. Do not edit.

Recall-maximising generation: 20 of 21 Table 2 coordinates recovered (the
only miss is OLMo-1B L1[1764,8041], where the paper has a layer typo and no
weight exists), but 1,945 extra candidates and zero new causal hits. Useful
as a candidate generator and as the evidence for why v3 needs the paper's
outlier conditions. Default output goes to results/v2/.

Lineage: olmo_sw.py (v0) -> detect_sw_v1.py -> detect_sw_v2.py -> detect_sw.py

Original docstring follows.

---

Model-agnostic super-weight detector (Yu et al. 2024, section 3.1).

Hooks every layer's `mlp.down_proj`, reads the coordinate off the activation
spikes, then peels: zero what it found, run again, until nothing survives.
Writes JSON that ablate_sw.py consumes. Zeroed weights are restored at exit.

Method recap:  Y[t,j] = sum_k X[t,k] * W[j,k]
    a super activation at Y[t,j] is produced by an outlier product
    X[t,k] * W[j,k]; that (j, k) is the super weight.

## v2 (2026-09-02) — rewritten after the 9-model Table 2 sweep

v1 returned 11 of Yu et al.'s 21 coordinates. The instrumented re-run showed
localization was never the problem: at round 0, twenty of the 21 sat at a
layer whose argmax was that coordinate or a sibling of it in the same layer.
All four miss causes were ours, not the paper's. Each is fixed here:

1. **One argmax per layer (5 misses).** `Y.abs().argmax()` returns a single
   (j, k), so a layer holding two super weights can only ever yield one --
   Llama-13B L2 has both of its coordinates in row 2231, Phi-3 L2 has three
   sharing column 808. Fixed by taking the top `--top-j` output channels.
2. **The dominance band (2 misses).** v1 required X[k]*W[j,k] to explain Y[j]
   to within +-20%, which is only valid when ONE weight feeds that channel.
   Llama-13B scored 0.593 and Llama-2-7B 0.758 -- both real, both rejected.
   Fixed by decomposing Y[t,j] into its per-k contributions and accepting the
   smallest prefix that explains `--explain-frac` of it, so two weights
   feeding one channel are both returned.
3. **Peel order (2 misses).** v1 zeroed only the loudest survivor per round;
   OLMo-7B L7 and L24 passed every check at rounds 0-1 and were never picked.
   Fixed by `--peel all`.
4. **Argmax elsewhere in the layer (1 miss).** Subsumed by fix 1.

The `towering` threshold is still ours and still arbitrary -- the paper has no
acceptance criterion at all, it reads spikes off a plot by eye. Replacing it
with a calibrated null is Phase 0. Detection here remains *suggestive*; only
causal ablation (ablate_sw.py) confirms a super weight.

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

TOWERING_FACTOR = 10   # spike must be >10x the median layer max -- ours, arbitrary
TOP_J = 4              # output channels examined per layer
EXPLAIN_FRAC = 0.80    # contributions to keep, as a share of the output spike
MAX_K = 4              # cap on contributors returned for one output channel
MAX_ROUNDS = 15        # safety cap on the peeling loop

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


def analyze_layer(layer_idx, X, Y, W, args):
    """Decompose this layer's biggest output spike into the weights making it.

    Returns one record for the layer, carrying every (j, k) candidate found.
    """
    # the token carrying the layer's largest |output| -- the super activation
    t = int(Y.abs().max(dim=-1).values.argmax())
    x, y = X[t], Y[t]
    max_y = y.abs().max().item()

    candidates = []
    for j in y.abs().topk(min(args.top_j, y.numel())).indices.tolist():
        # every k's signed contribution to y[j]; these sum to y[j] exactly
        contrib = x * W[j].float().cpu()
        target = y[j].item()
        order = contrib.abs().argsort(descending=True).tolist()

        # smallest prefix of contributors that explains the output channel:
        # one entry when a single weight dominates (the paper's picture),
        # two when two super weights share the channel (Llama-13B L2)
        running = 0.0
        for rank, k in enumerate(order[:args.max_k]):
            running += contrib[k].item()
            candidates.append({
                "j": j, "k": k, "rank_in_channel": rank,
                "weight": W[j, k].item(),
                "contribution": contrib[k].item(),
                "y_channel": target,
                "share": contrib[k].item() / target if target else float("nan"),
                "explained_by_prefix": running / target if target else float("nan"),
            })
            if abs(running) >= args.explain_frac * abs(target):
                break

    return {"layer": layer_idx, "token": t, "max_y": max_y,
            "max_x": x.abs().max().item(), "candidates": candidates}


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
    ap.add_argument("--explain-frac", type=float, default=EXPLAIN_FRAC)
    ap.add_argument("--max-k", type=int, default=MAX_K)
    ap.add_argument("--towering-factor", type=float, default=TOWERING_FACTOR)
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
            print(f"Round {rnd}: nothing new -- stopping.")
            break
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
        "detector_version": 2,
        "params": {"top_j": args.top_j, "explain_frac": args.explain_frac,
                   "max_k": args.max_k, "towering_factor": args.towering_factor,
                   "peel": args.peel, "max_rounds": args.max_rounds},
        "found": found,
        "rounds": rounds_log,
    }
    out = Path(args.out) if args.out else (
        Path("results/v2") / (args.model.replace("/", "_") + "_found.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFound {len(found)} candidate(s). Written to {out}")
    print("Candidates are UNVERIFIED until ablate_sw.py confirms damage.")


if __name__ == "__main__":
    main()
