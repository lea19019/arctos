"""Model-agnostic super-weight detector (Yu et al. 2024, section 3.1).

Same method as olmo_sw.py, but works on any HF causal LM whose layers have
an mlp.down_proj (Llama, Mistral, OLMo, Qwen, Phi-3, ...), and writes its
findings to a JSON file that ablate_sw.py can consume.

Usage (login node pre-downloads the model; compute nodes have no internet):
    uv run src/detect_sw.py --model allenai/OLMo-1B-0724-hf
    uv run src/detect_sw.py --model meta-llama/Llama-2-7b-hf --out results/llama2_found.json

Method recap:  Y[i,j] ~= X[i,k] * W[j,k]
    X = down_proj input   -> its biggest |value| gives k (and the token)
    Y = down_proj output  -> its biggest |value| gives j (same token, or bust)
    the candidate super weight is W[j, k] of that layer.
One pass only reveals the loudest SW, so we peel: find, zero, run again,
until no layer survives verification. Zeroed weights are restored at exit.

The three verification thresholds are made up (the paper has none at all —
spikes are read off a plot by eye). Replacing them with a calibrated null
is Phase 0 of this track. Detection here is *suggestive*; only causal
ablation (ablate_sw.py) confirms a super weight.
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

# Made-up verification thresholds (see module docstring).
DOMINANCE_BAND = (0.8, 1.2)   # X*W must explain Y within +-20%
TOWERING_FACTOR = 10          # spike must be >10x the median layer max
MAX_ROUNDS = 15               # safety cap on the peeling loop; must exceed the
                              # most super weights any model has (Phi-3: 6)

DEFAULT_PROMPT = "Language modeling is "

# Load in the checkpoint's own precision by default. The Llama and Mistral
# repos are float16 and OLMo/Phi-3 are bfloat16; forcing everything to bf16
# rounds small weights differently from the paper, and the OLMo-1B result
# turned on a weight of 0.0018. CPU gets float32 because fp16 matmul is not
# supported there.
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


def make_hook(layer_idx, records):
    """One hook per layer; the factory freezes layer_idx into each."""
    def hook(module, inp, out):
        X, Y, W = inp[0], out, module.weight
        token_x, k = divmod(X.abs().argmax().item(), X.shape[-1])
        token_y, j = divmod(Y.abs().argmax().item(), Y.shape[-1])
        records.append({
            "layer": layer_idx,
            "max_x": X.abs().max().item(), "token_x": token_x, "k": k,
            "max_y": Y.abs().max().item(), "token_y": token_y, "j": j,
            "sw_value": W[j, k].item(),
            "x_spike": X[0, token_x, k].item(),
            "y_spike": Y[0, token_y, j].item(),
        })
    return hook


def detection_pass(model, enc):
    """One forward pass with hooks on every down_proj; one record per layer."""
    records = []
    handles = [
        layer.mlp.down_proj.register_forward_hook(make_hook(i, records))
        for i, layer in enumerate(get_layers(model))
    ]
    with torch.no_grad():
        model(**enc)
    for h in handles:
        h.remove()
    return records


def dominance(r):
    return (r["x_spike"] * r["sw_value"]) / r["y_spike"]


def criteria(r, median_max_y):
    """Our three home-made checks, reported one by one.

    Kept separate (rather than a single bool) because the interesting
    question when the detector returns nothing is *which* check rejected
    the paper's coordinate -- the paper itself has no checks at all, so
    every rejection here is our addition, not a replication failure of
    Yu et al.
    """
    return {
        "tokens_match": r["token_x"] == r["token_y"],
        "dominant": DOMINANCE_BAND[0] < dominance(r) < DOMINANCE_BAND[1],
        "towering": r["max_y"] > TOWERING_FACTOR * median_max_y,
    }


def passes(r, median_max_y):
    return all(criteria(r, median_max_y).values())


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
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                    help="cap on the peel-and-repeat loop")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    enc = tokenizer([args.prompt], return_tensors="pt",
                    return_token_type_ids=False).to(device)

    found, rounds_log = [], []
    for rnd in range(args.max_rounds):
        records = detection_pass(model, enc)
        median_max_y = statistics.median(r["max_y"] for r in records)
        for r in records:
            r["criteria"] = criteria(r, median_max_y)
            r["dominance"] = dominance(r)
        survivors = [r for r in records if all(r["criteria"].values())]

        print(f"Round {rnd}: {len(survivors)} of {len(records)} layers pass "
              f"(median max|Y| = {median_max_y:.2f})")
        for r in survivors:
            print(f"  layer {r['layer']:2d}  W[{r['j']},{r['k']}]  "
                  f"max|Y|={r['max_y']:.1f}  dominance={dominance(r):.3f}")
        # every layer, not just the survivors: a rejected layer with the
        # paper's coordinate is the thing we most want to see afterwards
        rounds_log.append({"round": rnd, "median_max_y": median_max_y,
                           "survivors": survivors, "records": records})

        if not survivors:
            print(f"Round {rnd}: no candidate passes — stopping.")
            break

        c = max(survivors, key=lambda r: r["max_y"])
        print(f"Round {rnd}: zeroing layer {c['layer']} "
              f"W[{c['j']},{c['k']}] = {c['sw_value']:.4f}")
        found.append({"layer": c["layer"], "j": c["j"], "k": c["k"],
                      "value": c["sw_value"]})
        with torch.no_grad():
            get_layers(model)[c["layer"]].mlp.down_proj.weight[c["j"], c["k"]] = 0.0

    # restore every zeroed weight — leave the model as we found it
    with torch.no_grad():
        for f in found:
            get_layers(model)[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]

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
        "max_rounds": args.max_rounds,
        "thresholds": {"dominance_band": DOMINANCE_BAND,
                       "towering_factor": TOWERING_FACTOR},
        "found": found,
        "rounds": rounds_log,
    }
    out = Path(args.out) if args.out else (
        Path("results") / (args.model.replace("/", "_") + "_found.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFound {len(found)} candidate(s). Written to {out}")
    print("Candidates are UNVERIFIED until ablate_sw.py confirms damage.")


if __name__ == "__main__":
    main()
