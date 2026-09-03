"""Super-weight detector v5 — persistence in the residual stream.

Yu et al. 2024 locate super weights by activation spikes. Every earlier
version here read those spikes off `mlp.down_proj` output, which is the
wrong tensor: the paper's Figure 4 plots max LAYER output — the residual
stream — and its claim is that the super activation

    "persists throughout the entire model, at exactly the same magnitude,
     starting after Layer 2"

Measured on OLMo-1B (`activation_profile.py`), that is exactly what happens:

    layer  0:   0.2      layer  3: 309.5
    layer  1:   5.0      ...        ~420   (constant for 13 layers)
    layer  2: 267.7  <-- onset      layer 15: 419.4
                                    layer 16:   7.6  <-- final layer removes it

and it is why v1-v3 all stumbled on the same false positive: OLMo-1B's
LAST layer produces the biggest down_proj spike in the model (419.5, above
the real super weight's 266.9) because it writes almost straight into the
logits. No magnitude rule can separate the two. A persistence rule can —
the last layer's spike creates nothing that survives it.

So v5 inverts the search:

  1. one forward pass, keeping hidden states AND down_proj outputs;
  2. find residual channels (token, channel) that are massive and PERSIST
     over the rest of the depth — the super activations;
  3. each one's ONSET layer is where it first reaches its plateau; the
     super weight is in that layer's down_proj (hidden_states[i] is the
     output of decoder layer i-1);
  4. only there, decompose Y[t,j] into per-k contributions and keep the
     ones where X and W are BOTH outliers (paper 3.1);
  5. peel and repeat until no super activation survives.

Step 4's conditions are inherited from v3; steps 1-3 replace v3's
layer-max heuristic. Thresholds are still ours — the paper has no
acceptance criterion, it reads spikes off a plot — so detection stays
*suggestive* and only ablate_sw.py confirms a super weight.

    uv run src/detect_sw.py --model allenai/OLMo-1B-0724-hf
"""

import argparse
import datetime
import json
from pathlib import Path

import torch
import transformers
from provenance import git_sha
from transformers import AutoModelForCausalLM, AutoTokenizer

MASSIVE_FACTOR = 50    # peak |h| vs the median residual magnitude — ours
MASSIVE_FRAC = 0.10    # ... and vs the largest peak in the model. The median
                       # bar alone is useless: on OLMo-1B it lands at 3.14
                       # while the real super activation peaks at 427, so
                       # ordinary channels peaking at 3-8 sailed through.
                       # Yu et al. describe "a handful" per model, not a tail.
PLATEAU_FRAC = 0.50    # "at the plateau" = within this share of the peak
MIN_PERSIST = 0.50     # must hold the plateau over this share of later layers
TOP_W = 1000           # |W[j,k]| must rank this high within the layer
TOP_X = 10             # |X[t,k]| must rank this high at that token
EXPLAIN_FRAC = 0.80    # prefix must explain at least this share of Y[t,j]
OVERSHOOT = 1.25       # ... and at most this much (a prefix over 1 cancels)
MIN_SHARE = 0.35       # a contributor must carry at least this share of
                       # Y[t,j]. Paper 3.1: "Y_ij will be DOMINATED by their
                       # product... Y_ij ~= X_ik W_jk". A prefix summing to
                       # 0.8 out of six contributors at 0.2 each is not
                       # domination. 0.35 admits Llama-13B's genuine pair
                       # (shares 0.593 + 0.407) and rejects the 0.22-0.29
                       # tail v5 was returning in round 1.
MAX_K = 4              # cap on contributors returned per output channel
MAX_ROUNDS = 10
MAX_SW = 6             # the paper's own maximum (Phi-3), used as a guard

DEFAULT_PROMPT = "Language modeling is "
DTYPES = {"auto": "auto", "bf16": torch.bfloat16,
          "fp16": torch.float16, "fp32": torch.float32}


def get_layers(model):
    try:
        layers = model.model.layers
        _ = layers[0].mlp.down_proj
        return layers
    except AttributeError as e:
        raise SystemExit(
            f"Unsupported architecture ({type(model).__name__}): expected "
            f"model.model.layers[i].mlp.down_proj. Adapt get_layers()."
        ) from e


def forward_pass(model, enc, layers):
    """One pass; returns residual stream and every layer's down_proj in/out."""
    store = {}

    def make_hook(i):
        def hook(module, inp, out):
            store[i] = (inp[0][0].detach().float().cpu(),
                        out[0].detach().float().cpu())
        return hook

    handles = [l.mlp.down_proj.register_forward_hook(make_hook(i))
               for i, l in enumerate(layers)]
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    for h in handles:
        h.remove()
    # (n_layers+1, T, D); index 0 is the embedding output, index i the
    # output of decoder layer i-1
    H = torch.stack([h[0].detach().float().cpu() for h in out.hidden_states])
    return H, store


def find_super_activations(H, args, fixed_bar=None):
    """Residual channels that are massive AND persist to the end of depth.

    The final hidden state is excluded from the persistence window: on
    OLMo-1B the last layer *removes* the super activation (420 -> 7.6), so
    including it would reject the very thing we are looking for.
    """
    n_layers = H.shape[0] - 1
    mag = H.abs()
    last = max(n_layers - 1, 1)          # persistence measured up to here
    med = mag[:last + 1].median().item()
    peak_per_channel, peak_layer = mag[:last + 1].max(dim=0)

    # a super activation is one of the few largest things in the residual
    # stream, so gate on BOTH an absolute-ish bar and a share of the model's
    # own largest peak, whichever is stricter
    # NB: computed once, at round 0, and then held FIXED. A bar relative to
    # the current model's largest peak re-normalises every round, so after
    # peeling the real super weight the next-biggest activation is "massive"
    # by construction and the loop never terminates. The paper's rule is
    # suppression relative to the ORIGINAL magnitudes.
    bar = fixed_bar if fixed_bar is not None else max(
        args.massive_factor * med,
        args.massive_frac * peak_per_channel.max().item())

    found = []
    idx = (peak_per_channel > bar).nonzero()
    for t, j in idx.tolist():
        peak = peak_per_channel[t, j].item()
        bar = args.plateau_frac * peak
        above = [i for i in range(last + 1) if mag[i, t, j].item() >= bar]
        if not above:
            continue
        onset = above[0]
        # of the layers from onset onward, how many hold the plateau?
        window = last - onset + 1
        persist = sum(1 for i in range(onset, last + 1)
                      if mag[i, t, j].item() >= bar)
        if window < 2 or persist / window < args.min_persist:
            continue
        # hidden_states[onset] is the output of decoder layer onset-1
        layer = onset - 1
        if layer < 0:
            continue
        found.append({"token": t, "channel": j, "peak": peak, "bar": bar,
                      "onset": onset,
                      "layer": layer, "persist": persist, "window": window,
                      "persist_frac": persist / window,
                      "median_residual": med})
    found.sort(key=lambda d: -d["peak"])
    return found


def outlier_cut(v, n):
    flat = v.abs().flatten()
    return flat.topk(min(n, flat.numel())).values[-1].item()


def decompose(sa, X, Y, W, args):
    """Which weights in this layer produce the super activation at (t, j)?"""
    t, j = sa["token"], sa["channel"]
    x, y = X[t], Y[t]
    Wc = W.float().cpu()
    w_cut, x_cut = outlier_cut(Wc, args.top_w), outlier_cut(x, args.top_x)

    contrib = x * Wc[j]                   # signed; sums to y[j]
    target = y[j].item()
    order = contrib.abs().argsort(descending=True).tolist()

    prefix, taken = 0.0, []
    for rank, k in enumerate(order[:args.max_k]):
        prefix += contrib[k].item()
        taken.append({
            "j": j, "k": k, "rank_in_channel": rank,
            "weight": Wc[j, k].item(), "x": x[k].item(),
            "contribution": contrib[k].item(), "y_channel": target,
            "share": contrib[k].item() / target if target else float("nan"),
            "explained_by_prefix": prefix / target if target else float("nan"),
            "w_outlier": abs(Wc[j, k].item()) >= w_cut,
            "x_outlier": abs(x[k].item()) >= x_cut,
        })
        if abs(prefix) >= args.explain_frac * abs(target):
            break

    ratio = abs(prefix / target) if target else float("inf")
    if not (args.explain_frac <= ratio <= args.overshoot):
        return [], f"prefix explains {ratio:.2f} of Y[t,j]"
    kept = [c for c in taken if c["w_outlier"] and c["x_outlier"]
            and abs(c["share"]) >= args.min_share]
    if not kept:
        return [], ("no contributor both dominates (share >= "
                    f"{args.min_share}) and has X and W as outliers")
    return kept, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default="auto", choices=sorted(DTYPES))
    ap.add_argument("--massive-factor", type=float, default=MASSIVE_FACTOR)
    ap.add_argument("--massive-frac", type=float, default=MASSIVE_FRAC,
                    help="peak must also be this share of the largest peak")
    ap.add_argument("--plateau-frac", type=float, default=PLATEAU_FRAC)
    ap.add_argument("--min-persist", type=float, default=MIN_PERSIST)
    ap.add_argument("--top-w", type=int, default=TOP_W)
    ap.add_argument("--top-x", type=int, default=TOP_X)
    ap.add_argument("--explain-frac", type=float, default=EXPLAIN_FRAC)
    ap.add_argument("--overshoot", type=float, default=OVERSHOOT)
    ap.add_argument("--min-share", type=float, default=MIN_SHARE)
    ap.add_argument("--max-k", type=int, default=MAX_K)
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--max-sw", type=int, default=MAX_SW)
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
    stop_reason = f"hit --max-rounds ({args.max_rounds})"
    bar = None

    for rnd in range(args.max_rounds):
        H, store = forward_pass(model, enc, layers)
        sas = find_super_activations(H, args, fixed_bar=bar)
        if bar is None and sas:
            bar = sas[0]["bar"]      # freeze round 0's bar for every round
        print(f"\nRound {rnd}: {len(sas)} super activation(s) "
              f"(median residual |h| = "
              f"{sas[0]['median_residual']:.3f})" if sas else
              f"\nRound {rnd}: no super activation survives")

        fresh, round_rec = [], []
        for sa in sas:
            X, Y = store[sa["layer"]]
            cands, why = decompose(sa, X, Y, layers[sa["layer"]].mlp.down_proj.weight,
                                   args)
            round_rec.append({**sa, "candidates": cands, "rejected": why})
            print(f"  h[t{sa['token']},{sa['channel']}] peak {sa['peak']:.1f} "
                  f"onset layer {sa['onset']} (decoder L{sa['layer']}) "
                  f"persists {sa['persist']}/{sa['window']}"
                  + (f"  -- rejected: {why}" if why else ""))
            for c in cands:
                coord = (sa["layer"], c["j"], c["k"])
                print(f"      W[{c['j']},{c['k']}] = {c['weight']:+.4f}  "
                      f"share={c['share']:+.3f}")
                if coord not in seen:
                    fresh.append((sa, c))

        rounds_log.append({"round": rnd, "super_activations": round_rec})

        if not fresh:
            stop_reason = ("no super activation survives" if not sas
                           else "no new candidate passed")
            print(f"Round {rnd}: {stop_reason} -- stopping.")
            break

        over = len(found) + len(fresh) > args.max_sw
        if over:
            stop_reason = (f"exceeded --max-sw ({args.max_sw}) at round {rnd}; "
                           f"treat these as over-generated")
            print(f"Round {rnd}: {stop_reason}")

        for sa, c in fresh:
            coord = (sa["layer"], c["j"], c["k"])
            seen.add(coord)
            found.append({"layer": sa["layer"], "j": c["j"], "k": c["k"],
                          "value": c["weight"], "round": rnd,
                          "share": c["share"], "peak": sa["peak"],
                          "persist_frac": sa["persist_frac"]})
            with torch.no_grad():
                layers[coord[0]].mlp.down_proj.weight[coord[1], coord[2]] = 0.0
        if over:
            break

    with torch.no_grad():
        for f in found:
            layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]

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
        "detector_version": 5,
        "stop_reason": stop_reason,
        "params": vars(args),
        "found": found,
        "rounds": rounds_log,
    }
    out = Path(args.out) if args.out else (
        Path("results/v5") / (args.model.replace("/", "_") + "_found.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFound {len(found)} candidate(s). Written to {out}")
    print("Candidates are UNVERIFIED until ablate_sw.py confirms damage.")


if __name__ == "__main__":
    main()
