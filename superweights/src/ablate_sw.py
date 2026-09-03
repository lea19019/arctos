"""Model-agnostic causal ablation of super-weight candidates.

Takes candidates from a detect_sw.py JSON file, adds the paper's published
coordinates for this model (Yu et al. 2024, Table 2) if we have them, and
judges every candidate the same way: zero that ONE scalar, measure damage
against the intact model, restore. Real super weight = huge damage.

Hardening over the scratch olmo_ablate.py: damage is averaged over SEVERAL
eval texts and prompts (not one), and the exact model revision is recorded,
so a claim like "the weight at the paper's coordinate is ~0" points at an
exact artifact anyone can re-download and check.

Usage:
    uv run src/ablate_sw.py --model allenai/OLMo-1B-0724-hf \
        --candidates results/allenai_OLMo-1B-0724-hf_found.json
"""

import argparse
import datetime
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from provenance import git_sha
from transformers import AutoModelForCausalLM, AutoTokenizer

# Yu et al. 2024, Table 2 — the paper's super-weight directory.
# Every entry: (layer, j, k) for layers[layer].mlp.down_proj.weight[j, k].
# Transcribed from the paper 2026-09-02 (all 9 models, re-read from the PDF
# text layer on 2026-09-02; row/col ranges checked against each config's
# hidden_size/intermediate_size).
TABLE2 = {
    "huggyllama/llama-7b":        [(2, 3968, 7003)],
    "huggyllama/llama-13b":       [(2, 2231, 2278), (2, 2231, 6939)],
    "huggyllama/llama-30b":       [(3, 5633, 12817), (3, 5633, 17439),
                                   (10, 5633, 14386)],
    "meta-llama/Llama-2-7b-hf":   [(1, 2533, 7890)],
    "meta-llama/Llama-2-13b-hf":  [(3, 4743, 7678)],
    "mistralai/Mistral-7B-v0.1":  [(1, 2070, 7310)],
    "allenai/OLMo-1B-0724-hf":    [(1, 1764, 1710), (1, 1764, 8041)],
    "allenai/OLMo-7B-0724-hf":    [(1, 269, 7467), (2, 269, 8275),
                                   (7, 269, 453), (24, 269, 2300)],
    # Phi-3 is the paper's outlier: six coordinates, in two layers, sharing
    # two columns (808 and 2723) across three rows each.
    "microsoft/Phi-3-mini-4k-instruct": [(2, 525, 808), (2, 1693, 808),
                                         (2, 1113, 808), (4, 525, 2723),
                                         (4, 1113, 2723), (4, 1693, 2723)],
}

# Yu et al. Table 1 reports Llama-7B perplexity going 7.08 -> 763.65 (C4) and
# 5.67 -> 1211.11 (Wiki-2) when the super weight is pruned. Our four
# hand-written paragraphs gave x6 on the same model and coordinate -- roughly
# 20-30x under-powered, enough to make a real super weight look like nothing.
# So the default corpus is now wikitext-2-raw-v1 test, their Wiki-2. The
# paragraphs remain available via --eval-corpus paragraphs, since every
# result before 2026-09-02 was measured on them.
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1", "test")
PPL_SEGMENTS = 32     # non-overlapping windows scored; 32 x 2048 = 65k tokens
SEQ_LEN = 2048

EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog. Language models predict "
    "the next token given the tokens that came before. The city of Paris "
    "is the capital of France, and water boils at one hundred degrees "
    "Celsius at sea level.",
    "In 1969, astronauts landed on the Moon for the first time. The mission "
    "was called Apollo 11, and millions of people watched it on television. "
    "Neil Armstrong was the first human to step onto the lunar surface.",
    "To bake bread you need flour, water, yeast, and salt. Mix the "
    "ingredients, knead the dough, let it rise for an hour, and bake it in "
    "a hot oven until the crust turns golden brown.",
    "The stock market fell sharply on Tuesday as investors reacted to news "
    "of rising interest rates. Analysts said the decline reflected broader "
    "concerns about inflation and slowing economic growth.",
]
PROMPTS = [
    "The capital of France is",
    "Summer is hot. Winter is",
    "Two plus two equals",
]

# Same reasoning as detect_sw.py: default to the checkpoint's own precision,
# so a perplexity here is comparable to the paper's rather than to a bf16
# re-rounding of it.
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


def get_weight(model, layer, j, k):
    return get_layers(model)[layer].mlp.down_proj.weight[j, k].item()


def set_weight(model, layer, j, k, value):
    with torch.no_grad():
        get_layers(model)[layer].mlp.down_proj.weight[j, k] = value


def cross_entropy(model, enc):
    """Mean token cross-entropy in nats for one window."""
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return out.loss.item()


def wikitext_windows(tokenizer, device, n_segments, seq_len):
    """The standard protocol: concatenate the test split, cut it into
    non-overlapping windows of seq_len, score each one."""
    from datasets import load_dataset
    path, config, split = WIKITEXT
    ds = load_dataset(path, config, split=split)
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = min(n_segments, ids.numel() // seq_len)
    if n == 0:
        raise SystemExit(f"corpus too short for seq_len={seq_len}")
    return [{"input_ids": ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)}
            for i in range(n)]


def next_token_logprobs(model, enc):
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    return F.log_softmax(logits.float(), dim=-1)


def kl_nats(base_logp, abl_logp):
    """KL(base || ablated). 0 = the ablation changed nothing."""
    return F.kl_div(abl_logp, base_logp, log_target=True, reduction="sum").item()


def greedy_continuation(model, tokenizer, enc, prompt, n_tokens=15):
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()


def measure(model, tokenizer, eval_encs, prompt_encs, base_logps=None):
    """Mean perplexity over EVAL_TEXTS, mean KL over PROMPTS, one sample
    continuation (first prompt). base_logps=None means 'this IS baseline'."""
    # exp(mean loss), not mean of exp -- the standard corpus-perplexity
    # definition, and the one the paper's numbers are on
    losses = [cross_entropy(model, e) for e in eval_encs]
    ppl = math.exp(sum(losses) / len(losses))
    ppls = [math.exp(l) for l in losses]
    logps = [next_token_logprobs(model, p) for p in prompt_encs]
    kls = ([kl_nats(b, a) for b, a in zip(base_logps, logps)]
           if base_logps is not None else [0.0])
    gen = greedy_continuation(model, tokenizer, prompt_encs[0], PROMPTS[0])
    mean = lambda xs: sum(xs) / len(xs)
    return {"ppl": ppl, "ppl_each": ppls,
            "kl": mean(kls), "kl_each": kls,
            "logps": logps, "continuation": gen}


def verdict(ppl_ratio, kl):
    """Eyeball label from made-up cutoffs (display only, not science)."""
    if ppl_ratio > 10 or kl > 1:
        return "CATASTROPHIC"
    if ppl_ratio > 1.5 or kl > 0.1:
        return "damaged"
    return "no effect"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--candidates", default=None,
                    help="JSON from detect_sw.py (its 'found' list is used)")
    ap.add_argument("--no-table2", action="store_true",
                    help="skip the paper's published coordinates")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: results/<model>_ablation.json)")
    ap.add_argument("--dtype", default="auto", choices=sorted(DTYPES),
                    help="'auto' = the checkpoint's own torch_dtype")
    ap.add_argument("--eval-corpus", default="wikitext2",
                    choices=["wikitext2", "paragraphs"],
                    help="wikitext2 = the paper's Wiki-2; paragraphs = the "
                         "four hand-written texts every pre-2026-09-02 "
                         "result was measured on")
    ap.add_argument("--ppl-segments", type=int, default=PPL_SEGMENTS)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()

    # ---- assemble the candidate list: detector finds + paper's directory,
    # de-duplicated on (layer, j, k) so a coordinate is only tested once.
    candidates, seen = [], set()

    def add(name, layer, j, k):
        if (layer, j, k) not in seen:
            seen.add((layer, j, k))
            candidates.append({"name": name, "layer": layer, "j": j, "k": k})

    if args.candidates:
        det = json.loads(Path(args.candidates).read_text())
        for n, f in enumerate(det["found"], 1):
            add(f"found-{n} (L{f['layer']})", f["layer"], f["j"], f["k"])
    if not args.no_table2:
        for n, (layer, j, k) in enumerate(TABLE2.get(args.model, []), 1):
            add(f"table2-{n} (L{layer})", layer, j, k)
    if not candidates:
        raise SystemExit("No candidates: pass --candidates and/or use a "
                         "model that is in TABLE2.")

    # ---- load model, encode the fixed texts once ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # never ask for a window longer than the model's context
    seq_len = min(args.seq_len,
                  getattr(model.config, "max_position_embeddings", args.seq_len))
    if args.eval_corpus == "wikitext2":
        eval_encs = wikitext_windows(tokenizer, device, args.ppl_segments, seq_len)
    else:
        eval_encs = [tokenizer(t, return_tensors="pt",
                               return_token_type_ids=False).to(device)
                     for t in EVAL_TEXTS]
    prompt_encs = [tokenizer(p, return_tensors="pt",
                             return_token_type_ids=False).to(device)
                   for p in PROMPTS]

    # ---- baseline, then one candidate at a time: zero -> measure -> restore
    base = measure(model, tokenizer, eval_encs, prompt_encs)
    results = []
    for c in candidates:
        L, j, k = c["layer"], c["j"], c["k"]
        original = get_weight(model, L, j, k)
        set_weight(model, L, j, k, 0.0)
        m = measure(model, tokenizer, eval_encs, prompt_encs, base["logps"])
        set_weight(model, L, j, k, original)
        assert get_weight(model, L, j, k) == original
        results.append({**c, "weight": original, "ppl": m["ppl"],
                        "ppl_each": m["ppl_each"], "kl": m["kl"],
                        "kl_each": m["kl_each"],
                        "continuation": m["continuation"],
                        "verdict": verdict(m["ppl"] / base["ppl"], m["kl"])})

    # ---- report ----
    print(f"\nmodel: {args.model}  "
          f"revision: {getattr(model.config, '_commit_hash', None)}")
    print(f"perplexity over {len(eval_encs)} x {seq_len}-token windows of "
          f"{args.eval_corpus}; KL over {len(PROMPTS)} prompts\n")
    print(f"{'candidate':<22} {'coordinate':<18} {'weight':>9} "
          f"{'ppl':>9} {'ppl x':>7} {'KL':>7}  verdict")
    print("-" * 88)
    print(f"{'baseline (intact)':<22} {'—':<18} {'—':>9} "
          f"{base['ppl']:>9.2f} {'—':>7} {'—':>7}  —")
    for r in results:
        coord = f"L{r['layer']}[{r['j']},{r['k']}]"
        print(f"{r['name']:<22} {coord:<18} {r['weight']:>9.4f} "
              f"{r['ppl']:>9.2f} {'x' + format(r['ppl'] / base['ppl'], '.1f'):>7} "
              f"{r['kl']:>7.3f}  {r['verdict']}")
    def shorten(s, n=46):
        return s if len(s) <= n else s[:n - 3] + "..."

    print(f"\ncontinuations of {PROMPTS[0]!r} (greedy, truncated; full text in JSON):")
    print(f"  {'baseline':<22} | {shorten(base['continuation'])}")
    for r in results:
        print(f"  {r['name']:<22} | {shorten(r['continuation'])}")

    # ---- write JSON (drop the non-serializable logps from baseline) ----
    out = Path(args.out) if args.out else (
        Path("results") / (args.model.replace("/", "_") + "_ablation.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "revision_requested": args.revision,
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_sha": git_sha(),
        "device": device,
        "dtype_requested": args.dtype,
        "dtype": str(model.dtype),
        "eval_corpus": args.eval_corpus,
        "eval_corpus_spec": (list(WIKITEXT) if args.eval_corpus == "wikitext2"
                             else EVAL_TEXTS),
        "ppl_segments": len(eval_encs), "seq_len": seq_len,
        "prompts": PROMPTS,
        "candidates_file": args.candidates,
        "baseline": {"ppl": base["ppl"], "ppl_each": base["ppl_each"],
                     "continuation": base["continuation"]},
        "results": results,
    }, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
