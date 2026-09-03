"""Model-agnostic causal ablation of super-weight candidates.

Takes candidates from a detect_sw.py JSON file, adds the paper's published
coordinates for this model (Yu et al. 2024, Table 2) if we have them, and
judges every candidate the same way: zero the scalar, measure damage against
the intact model on wikitext-2, restore.

Since 2026-09-02 (experiments/joint_ablation) the unit of ablation is no
longer only one scalar:

  --joint       zero the whole candidate set at once (and, separately, the
                detector's set and Table 2's set), then leave-one-out: every
                candidate restored while the rest stay zeroed. Subramanian
                et al. (COLM 2026, Table 8) ablate jointly; every earlier
                number in this track is individual. Both are reported.
  --null-n N    magnitude-matched null: N random single weights from the
                top-100 |W| of the same matrices, and N random SETS of the
                same per-layer size. The max over draws is the reference for
                "this ratio is extreme" (max-statistic logic).
  --sa-remove   direct super-activation removal (Sun et al. 2024, Table 3):
                at the onset layer, set the detected residual channel to zero
                wherever it exceeds half its detection-prompt peak. Tests
                whether the activation is load-bearing regardless of which
                weights produce it. Control: the same number of
                median-magnitude channels zeroed at the same positions.

Every ratio carries a paired-bootstrap 95% CI over the eval windows
(bootstrap_ci.py); the JSON also records the super-activation magnitude
after each weight ablation, so "did zeroing the weights remove the
activation?" is answered in the same file.

Usage:
    uv run src/ablate_sw.py --model allenai/OLMo-1B-0724-hf \
        --candidates results/v5/allenai_OLMo-1B-0724-hf_found.json \
        --joint --null-n 50 --sa-remove
"""

import argparse
import datetime
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from bootstrap_ci import annotate
from provenance import git_sha
from sw_arch import down_proj, get_layers
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

# Coordinates published only in Yu et al.'s code, not in the paper:
# github.com/mengxiayu/LLMSuperWeight analyze.py SUPER_WEIGHTS_MAP, commit
# 232e4ee (2024-12-03). Subramanian et al. (COLM 2026) Table 8 zero the
# Llama-3-8B triple jointly and report 7.44 -> inf. The three share input
# column 2427 of layer-1 down_proj -- and are exactly what detect_sw.py v5
# returned, unprompted, on Llama-3.1-8B-Instruct (results/modern). We apply
# them to the 3.1 variants too, labelled "repo" so the provenance is visible;
# 3.1 is a different training run from 3.0, so this is a hypothesis, not a
# published coordinate for those checkpoints.
REPO = {
    "meta-llama/Meta-Llama-3-8B":       [(1, 788, 2427), (1, 1384, 2427),
                                         (1, 4062, 2427)],
    "meta-llama/Llama-3.1-8B":          [(1, 788, 2427), (1, 1384, 2427),
                                         (1, 4062, 2427)],
    "meta-llama/Llama-3.1-8B-Instruct": [(1, 788, 2427), (1, 1384, 2427),
                                         (1, 4062, 2427)],
}

# Yu et al. Table 1 reports Llama-7B perplexity going 7.08 -> 763.65 (C4) and
# 5.67 -> 1211.11 (Wiki-2) when the super weight is pruned. Our four
# hand-written paragraphs gave x6 on the same model and coordinate -- roughly
# 20-30x under-powered, enough to make a real super weight look like nothing.
# So the default corpus is wikitext-2-raw-v1 test, their Wiki-2. The
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


# ----------------------------------------------------------------- weights
def W(model, layer):
    return down_proj(get_layers(model)[layer]).weight


def get_weight(model, layer, j, k):
    return W(model, layer)[j, k].item()


def set_weight(model, layer, j, k, value):
    with torch.no_grad():
        W(model, layer)[j, k] = value


class Zeroed:
    """Zero a set of (layer, j, k) coordinates for the duration of a block,
    restoring the exact original values afterwards."""

    def __init__(self, model, coords):
        self.model, self.coords = model, list(coords)

    def __enter__(self):
        self.saved = [get_weight(self.model, *c) for c in self.coords]
        for c in self.coords:
            set_weight(self.model, *c, 0.0)
        return self

    def __exit__(self, *exc):
        for c, v in zip(self.coords, self.saved):
            set_weight(self.model, *c, v)
        for c, v in zip(self.coords, self.saved):
            assert get_weight(self.model, *c) == v, f"restore failed at {c}"


# ------------------------------------------------------------------- eval
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


def measure(model, tokenizer, eval_encs, prompt_encs, base_logps=None,
            with_generation=True):
    """Perplexity over the eval windows, mean KL over PROMPTS, one sample
    continuation (first prompt). base_logps=None means 'this IS baseline'."""
    # exp(mean loss), not mean of exp -- the standard corpus-perplexity
    # definition, and the one the paper's numbers are on
    losses = [cross_entropy(model, e) for e in eval_encs]
    ppl = math.exp(sum(losses) / len(losses))
    ppls = [math.exp(l) for l in losses]
    logps = [next_token_logprobs(model, p) for p in prompt_encs]
    kls = ([kl_nats(b, a) for b, a in zip(base_logps, logps)]
           if base_logps is not None else [0.0])
    gen = (greedy_continuation(model, tokenizer, prompt_encs[0], PROMPTS[0])
           if with_generation else None)
    mean = lambda xs: sum(xs) / len(xs)
    return {"ppl": ppl, "ppl_each": ppls, "loss_each": losses,
            "kl": mean(kls), "kl_each": kls,
            "logps": logps, "continuation": gen}


def verdict(ppl_ratio, kl):
    """Eyeball label from made-up cutoffs (display only, not science).
    The calibrated reference is the magnitude-matched null in the JSON."""
    if ppl_ratio > 10 or kl > 1:
        return "CATASTROPHIC"
    if ppl_ratio > 1.5 or kl > 0.1:
        return "damaged"
    return "no effect"


# --------------------------------------------------- super activations
def super_activations(det):
    """The persistent residual channels the detector found at round 0, one
    entry per (token, channel): where to intervene for --sa-remove."""
    if not det or not det.get("rounds"):
        return []
    seen, out = set(), []
    for sa in det["rounds"][0]["super_activations"]:
        key = (sa["token"], sa["channel"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"token": sa["token"], "channel": sa["channel"],
                    "onset": sa["onset"], "layer": sa["layer"],
                    "peak": sa["peak"], "n_candidates": len(sa["candidates"])})
    return out


def sa_magnitudes(model, layers, enc, sas):
    """|h| at each super activation's (onset, token, channel) on the
    detection prompt: did the ablation remove the activation?"""
    if not sas:
        return []
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    H = out.hidden_states
    return [abs(H[sa["onset"]][0, sa["token"], sa["channel"]].item())
            for sa in sas]


class ChannelIntervention:
    """Forward hook on the decoder layer whose output is hidden_states[onset]:
    for every position where |h[:, t, ch]| > thr, set the given channels to
    `value`. `targets` = list of (onset, trigger_channel, thr, channels_to_zero)."""

    def __init__(self, layers, targets, value=0.0):
        self.layers, self.value = layers, value
        self.by_layer = {}
        for onset, trig, thr, chans in targets:
            self.by_layer.setdefault(onset - 1, []).append((trig, thr, chans))
        self.hits = []

    def _hook(self, specs):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            n_hit = 0
            for trig, thr, chans in specs:
                mask = h[..., trig].abs() > thr             # (B, T)
                n_hit += int(mask.sum().item())
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)
                    for c in chans:
                        h[idx[0], idx[1], c] = self.value
            self.hits.append(n_hit)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    def __enter__(self):
        self.handles = [self.layers[L].register_forward_hook(self._hook(specs))
                        for L, specs in self.by_layer.items()]
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()


def median_channels(model, layers, enc, sa, n, rng):
    """Sun et al.'s control: channels whose |h| at the SA's onset/token sits
    at the median of the residual — n of them, chosen at random."""
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[sa["onset"]][0, sa["token"]].abs()
    order = h.argsort().tolist()
    D = len(order)
    band = order[int(0.4 * D):int(0.6 * D)]
    return rng.sample(band, n)


# ------------------------------------------------------------------ main
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
    ap.add_argument("--joint", action="store_true",
                    help="also zero the whole set at once, and leave-one-out")
    ap.add_argument("--null-n", type=int, default=0,
                    help="magnitude-matched null draws (individual and joint)")
    ap.add_argument("--null-pool", type=int, default=100,
                    help="draw null weights from the top-K |W| of each layer")
    ap.add_argument("--sa-remove", action="store_true",
                    help="zero the super activation channel(s) at onset")
    ap.add_argument("--sa-frac", type=float, default=0.5,
                    help="intervene where |h| > this share of the detection peak")
    ap.add_argument("--sa-null-n", type=int, default=0,
                    help="median-channel control draws for --sa-remove")
    ap.add_argument("--concentration", action="store_true",
                    help="how many weights carry it: zero the top-k contributors "
                         "to each super activation (k = 1..256) at its onset layer")
    ap.add_argument("--conc-ks", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--conc-top-sa", type=int, default=2,
                    help="run the curve for this many super activations (by peak)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-B", type=int, default=2000)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- provenance manifest first, results later
    out = Path(args.out) if args.out else (
        Path("results") / (args.model.replace("/", "_") + "_ablation.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model, "args": vars(args), "git_sha": git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- assemble the candidate list: detector finds + paper's directory,
    # de-duplicated on (layer, j, k) so a coordinate is only tested once.
    candidates, seen, sources = [], set(), {}

    def add(name, layer, j, k, source):
        if (layer, j, k) not in seen:
            seen.add((layer, j, k))
            candidates.append({"name": name, "layer": layer, "j": j, "k": k})
        sources.setdefault((layer, j, k), set()).add(source)

    det = None
    if args.candidates:
        det = json.loads(Path(args.candidates).read_text())
        for n, f in enumerate(det["found"], 1):
            add(f"found-{n} (L{f['layer']})", f["layer"], f["j"], f["k"], "found")
    if not args.no_table2:
        for n, (layer, j, k) in enumerate(TABLE2.get(args.model, []), 1):
            add(f"table2-{n} (L{layer})", layer, j, k, "table2")
        for n, (layer, j, k) in enumerate(REPO.get(args.model, []), 1):
            add(f"repo-{n} (L{layer})", layer, j, k, "repo")
    if not candidates and not (det and det.get("rounds") and
                               (args.sa_remove or args.concentration)):
        raise SystemExit("No candidates: pass --candidates and/or use a "
                         "model that is in TABLE2 (or add --sa-remove / "
                         "--concentration to test the activation anyway).")
    coords_all = [(c["layer"], c["j"], c["k"]) for c in candidates]

    # ---- load model, encode the fixed texts once ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    layers = get_layers(model)
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
    sas = super_activations(det)
    det_prompt = (det or {}).get("prompt", "Language modeling is ")
    det_enc = tokenizer([det_prompt], return_tensors="pt",
                        return_token_type_ids=False).to(device)

    def run(name, coords, extra=None, with_generation=True):
        with Zeroed(model, coords):
            m = measure(model, tokenizer, eval_encs, prompt_encs, base["logps"],
                        with_generation)
            sa_after = sa_magnitudes(model, layers, det_enc, sas)
        rec = {"name": name, "coords": [list(c) for c in coords],
               "ppl": m["ppl"], "ppl_each": m["ppl_each"],
               "loss_each": m["loss_each"], "kl": m["kl"], "kl_each": m["kl_each"],
               "continuation": m["continuation"], "sa_after": sa_after,
               "verdict": verdict(m["ppl"] / base["ppl"], m["kl"])}
        rec.update(extra or {})
        coord = (f"L{coords[0][0]}[{coords[0][1]},{coords[0][2]}]"
                 if len(coords) == 1 else f"{len(coords)} weights")
        print(f"  {name:<30} {coord:<20} ppl {m['ppl']:>10.2f} "
              f"x{m['ppl'] / base['ppl']:<9.2f} KL {m['kl']:.3f}  "
              f"SA-after {['%.0f' % s for s in sa_after]}", flush=True)
        return rec

    # ---- baseline
    base = measure(model, tokenizer, eval_encs, prompt_encs)
    base["sa"] = sa_magnitudes(model, layers, det_enc, sas)
    print(f"\nbaseline ppl {base['ppl']:.2f}; super activations on "
          f"{det_prompt!r}: {[(s['onset'], s['token'], s['channel'], round(s['peak'])) for s in sas]}"
          f" -> now {['%.0f' % s for s in base['sa']]}", flush=True)

    # ---- individual, one candidate at a time
    print("\n[individual]")
    results = []
    for c in candidates:
        L, j, k = c["layer"], c["j"], c["k"]
        rec = run(c["name"], [(L, j, k)],
                  {"layer": L, "j": j, "k": k, "weight": get_weight(model, L, j, k),
                   "kind": "individual", "source": sorted(sources[(L, j, k)])})
        results.append(rec)

    # ---- joint sets and leave-one-out
    if args.joint and coords_all:
        print("\n[joint]")
        sets = {"joint-all": coords_all}
        found = [c for c in coords_all if "found" in sources[c]]
        pub = [c for c in coords_all if sources[c] & {"table2", "repo"}]
        if found and found != coords_all:
            sets["joint-found"] = found
        if pub and pub != coords_all:
            sets["joint-published"] = pub
        for name, cs in sets.items():
            if len(cs) >= 1:
                results.append(run(name, cs, {"kind": "joint", "n": len(cs)}))
        if len(coords_all) > 1:
            print("\n[leave-one-out: everything zeroed EXCEPT the named weight]")
            for c in candidates:
                keep = (c["layer"], c["j"], c["k"])
                rest = [x for x in coords_all if x != keep]
                results.append(run(f"loo-keep {c['name']}", rest,
                                   {"kind": "loo", "kept": list(keep),
                                    "n": len(rest)}))

    # ---- magnitude-matched null
    null = []
    if args.null_n > 0 and coords_all:
        print(f"\n[null: {args.null_n} random top-{args.null_pool} weights, "
              f"individual then joint]")
        pools = {}
        per_layer = {}
        for (L, j, k) in coords_all:
            per_layer[L] = per_layer.get(L, 0) + 1
        for L in per_layer:
            Wl = W(model, L).detach().float()
            flat = Wl.abs().flatten()
            top = flat.topk(args.null_pool + len(coords_all)).indices.tolist()
            D = Wl.shape[1]
            pool = [(L, i // D, i % D) for i in top]
            pool = [p for p in pool if p not in seen][:args.null_pool]
            pools[L] = pool
        pool_all = [p for L in pools for p in pools[L]]
        for n in range(args.null_n):
            c = rng.choice(pool_all)
            null.append(run(f"null-ind-{n}", [c],
                            {"kind": "null-individual", "layer": c[0],
                             "j": c[1], "k": c[2],
                             "weight": get_weight(model, *c),
                             "rank_in_layer": pools[c[0]].index(c) + 1},
                            with_generation=False))
        if len(coords_all) > 1:
            for n in range(args.null_n):
                cs = [x for L, cnt in per_layer.items()
                      for x in rng.sample(pools[L], cnt)]
                null.append(run(f"null-joint-{n}", cs,
                                {"kind": "null-joint", "n": len(cs)},
                                with_generation=False))

    # ---- concentration: zero the top-k contributors to the super activation
    if args.concentration and sas:
        print("\n[concentration: top-k contributors x_k * W[j,k] at the onset layer]")
        ks = [int(k) for k in args.conc_ks.split(",")]
        for sa in sas[:args.conc_top_sa]:
            L, t, j = sa["layer"], sa["token"], sa["channel"]
            captured = {}
            hd = down_proj(layers[L]).register_forward_hook(
                lambda m, i, o: captured.__setitem__("x", i[0][0].detach().float().cpu()))
            with torch.no_grad():
                model(**det_enc)
            hd.remove()
            x = captured["x"][t]
            w = W(model, L).detach().float().cpu()[j]
            contrib = x * w
            order = contrib.abs().argsort(descending=True).tolist()
            total = contrib.sum().item()
            for k in ks:
                if k > len(order):
                    break
                top = order[:k]
                explained = contrib[top].sum().item() / total if total else float("nan")
                results.append(run(f"conc ch{j} top{k}", [(L, j, kk) for kk in top],
                                   {"kind": "concentration", "k": k, "sa": sa,
                                    "explained_share": explained,
                                    "top_k_indices": top[:16]},
                                   with_generation=(k in (1, 8, 64, 256))))

    # ---- direct super-activation removal
    sa_results, sa_null = [], []
    if args.sa_remove and sas:
        print("\n[super-activation removal at onset layer]")

        def run_sa(name, targets, extra):
            with ChannelIntervention(layers, targets, 0.0) as iv:
                m = measure(model, tokenizer, eval_encs, prompt_encs, base["logps"])
                hits = iv.hits
            with ChannelIntervention(layers, targets, 0.0):
                sa_after = sa_magnitudes(model, layers, det_enc, sas)
            rec = {"name": name, "ppl": m["ppl"], "ppl_each": m["ppl_each"],
                   "loss_each": m["loss_each"], "kl": m["kl"],
                   "kl_each": m["kl_each"], "continuation": m["continuation"],
                   "positions_hit_per_forward": (sum(hits) / len(hits)) if hits else 0,
                   "sa_after": sa_after, "sa_frac": args.sa_frac,
                   "verdict": verdict(m["ppl"] / base["ppl"], m["kl"])}
            rec.update(extra)
            print(f"  {name:<30} {'':<20} ppl {m['ppl']:>10.2f} "
                  f"x{m['ppl'] / base['ppl']:<9.2f} KL {m['kl']:.3f}  "
                  f"hits/fwd {rec['positions_hit_per_forward']:.1f}  "
                  f"SA-after {['%.0f' % s for s in sa_after]}", flush=True)
            return rec

        targets = [(sa["onset"], sa["channel"], args.sa_frac * sa["peak"],
                    [sa["channel"]]) for sa in sas]
        for sa, t in zip(sas, targets):
            sa_results.append(run_sa(
                f"sa-zero ch{sa['channel']} @h{sa['onset']}", [t],
                {"kind": "sa-individual", "sa": sa}))
        if len(sas) > 1:
            sa_results.append(run_sa("sa-zero all", targets,
                                     {"kind": "sa-joint", "n": len(sas)}))
        for n in range(args.sa_null_n):
            # same trigger positions, same count, median-magnitude channels
            ctrl = []
            for sa in sas:
                chans = median_channels(model, layers, det_enc, sa, 1, rng)
                ctrl.append((sa["onset"], sa["channel"],
                             args.sa_frac * sa["peak"], chans))
            sa_null.append(run_sa(f"sa-null-{n}", ctrl,
                                  {"kind": "sa-null",
                                   "channels": [c[3] for c in ctrl]}))

    # ---- report ----
    print(f"\nmodel: {args.model}  "
          f"revision: {getattr(model.config, '_commit_hash', None)}")
    print(f"perplexity over {len(eval_encs)} x {seq_len}-token windows of "
          f"{args.eval_corpus}; KL over {len(PROMPTS)} prompts\n")
    print(f"{'candidate':<30} {'coordinate':<18} {'weight':>9} "
          f"{'ppl':>9} {'ppl x':>7} {'KL':>7}  verdict")
    print("-" * 96)
    print(f"{'baseline (intact)':<30} {'—':<18} {'—':>9} "
          f"{base['ppl']:>9.2f} {'—':>7} {'—':>7}  —")
    for r in results:
        if r["kind"] == "individual":
            coord = f"L{r['layer']}[{r['j']},{r['k']}]"
            wt = f"{r['weight']:>9.4f}"
        else:
            coord, wt = f"{len(r['coords'])} weights", f"{'—':>9}"
        print(f"{r['name']:<30} {coord:<18} {wt} "
              f"{r['ppl']:>9.2f} {'x' + format(r['ppl'] / base['ppl'], '.1f'):>7} "
              f"{r['kl']:>7.3f}  {r['verdict']}")

    def shorten(s, n=46):
        return s if not s or len(s) <= n else s[:n - 3] + "..."

    print(f"\ncontinuations of {PROMPTS[0]!r} (greedy, truncated; full text in JSON):")
    print(f"  {'baseline':<30} | {shorten(base['continuation'])}")
    for r in results + sa_results:
        print(f"  {r['name']:<30} | {shorten(r['continuation'])}")

    # ---- write JSON (drop the non-serializable logps from baseline) ----
    doc = {
        "model": args.model,
        "revision_requested": args.revision,
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_sha": git_sha(),
        "slurm_job_id": manifest["slurm_job_id"],
        "device": device,
        "dtype_requested": args.dtype,
        "dtype": str(model.dtype),
        "eval_corpus": args.eval_corpus,
        "eval_corpus_spec": (list(WIKITEXT) if args.eval_corpus == "wikitext2"
                             else EVAL_TEXTS),
        "ppl_segments": len(eval_encs), "seq_len": seq_len,
        "prompts": PROMPTS,
        "candidates_file": args.candidates,
        "detection_prompt": det_prompt,
        "super_activations": sas,
        "params": vars(args),
        "seed": args.seed,
        "baseline": {"ppl": base["ppl"], "ppl_each": base["ppl_each"],
                     "loss_each": base["loss_each"], "sa": base["sa"],
                     "continuation": base["continuation"]},
        "results": results,
        "null": null,
        "sa_results": sa_results,
        "sa_null": sa_null,
    }
    annotate(doc, B=args.bootstrap_B, seed=args.seed)
    out.write_text(json.dumps(doc, indent=2))
    print(f"\nWritten to {out}")
    if null:
        mx = max(null, key=lambda r: r["ratio"])
        print(f"null max ratio: x{mx['ratio']:.2f} ({mx['name']}) over {len(null)} draws")
    for r in results:
        lo, hi = r["ratio_ci95"]
        print(f"  {r['name']:<30} x{r['ratio']:>9.2f} [{lo:.2f}, {hi:.2f}]")
    for r in sa_results:
        lo, hi = r["ratio_ci95"]
        print(f"  {r['name']:<30} x{r['ratio']:>9.2f} [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
