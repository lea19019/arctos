"""Natural-repetition pilot: greedy open-ended continuation, no repetition
penalty, on a healthy model, in several languages. Measures how often the
model loops on its own and where the loop starts.

Design: docs/repetition_experiment_design.md §2 (Arm N1). Metrics:
src/rep_metrics.py (pre-registered L-onset rule, seq-rep-4, rep-r).

    uv run src/gen_loops.py --config configs/loops_pilot/towerbase.yaml

One YAML per run. Every output JSON carries git sha, the resolved config,
library versions, the model revision actually loaded, dtype, device, seed,
and the SLURM job id. Prompts are the first `prefix_tokens` tokens of a
FLORES+ devtest document (consecutive sentences of one URL, in id order), so
the same documents are used for every model and language. Decoding is
greedy with no repetition_penalty / no_repeat_ngram_size / min_new_tokens;
EOS is allowed and counted. Batched generation uses left padding; a
batch-invariance check (bs=1 vs batched, exact token match) is recorded
because greedy argmax at near-ties can depend on batch composition.
"""

import argparse
import datetime
import glob
import json
import math
import os
import random
import sys
import unicodedata
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_metrics as rm  # noqa: E402
from provenance import git_sha  # noqa: E402

DEFAULTS = {
    "revision": None,
    "dtype": "auto",            # the checkpoint's own dtype
    "split": "devtest",
    "prefix_tokens": 50,
    "max_new_tokens": 256,
    "n_prompts": 200,
    "batch_size": 16,
    "batch_invariance_n": 16,
    "seed": 0,
    "add_special_tokens": True,  # tokenizer default BOS handling; recorded
    "out_dir": "results/loops_pilot",
}

SCRIPT_OF = {"Latn": "LATIN", "Cyrl": "CYRILLIC", "Hans": "CJK", "Hang": "HANGUL",
             "Arab": "ARABIC", "Ethi": "ETHIOPIC"}


# ---------------------------------------------------------------- prompts
def flores_file(lang, split):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = sorted(glob.glob(os.path.join(
        hub, "datasets--openlanguagedata--flores_plus", "snapshots", "*", split, f"{lang}.jsonl")))
    if not hits:
        raise SystemExit(f"FLORES+ {split}/{lang}.jsonl not in the HF cache ({hub})")
    return hits[-1]


def documents(lang, split):
    """Consecutive FLORES+ sentences sharing a URL, in id order -> one text."""
    recs = [json.loads(l) for l in open(flores_file(lang, split), encoding="utf-8")]
    recs.sort(key=lambda r: int(r["id"]))
    docs, cur, cur_url = [], [], None
    for r in recs:
        url = r.get("url")
        if url != cur_url and cur:
            docs.append({"url": cur_url, "ids": [c["id"] for c in cur],
                         "text": " ".join(c["text"] for c in cur)})
            cur = []
        cur_url = url
        cur.append(r)
    if cur:
        docs.append({"url": cur_url, "ids": [c["id"] for c in cur],
                     "text": " ".join(c["text"] for c in cur)})
    return docs


def build_prompts(tokenizer, lang, cfg):
    """First prefix_tokens tokens of each document long enough; deterministic order."""
    out = []
    for d in documents(lang, cfg["split"]):
        ids = tokenizer(d["text"], add_special_tokens=False).input_ids
        if len(ids) < cfg["prefix_tokens"]:
            continue
        out.append({"doc_ids": d["ids"], "url": d["url"],
                    "prompt_ids": ids[:cfg["prefix_tokens"]]})
        if len(out) >= cfg["n_prompts"]:
            break
    return out


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate_batch(model, tokenizer, prompt_ids_list, cfg, device):
    bos = [tokenizer.bos_token_id] if (cfg["add_special_tokens"] and tokenizer.bos_token_id is not None
                                       and getattr(tokenizer, "add_bos_token", True)) else []
    seqs = [bos + p for p in prompt_ids_list]
    maxlen = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):                       # left padding
        input_ids[i, maxlen - len(s):] = torch.tensor(s)
        attn[i, maxlen - len(s):] = 1
    out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                         max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                         num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                         pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
    gens = []
    for i in range(len(seqs)):
        g = out[i, maxlen:].tolist()
        ended = tokenizer.eos_token_id in g
        if ended:
            g = g[:g.index(tokenizer.eos_token_id)]
        # strip trailing pad (post-EOS padding)
        while g and g[-1] == pad and pad != tokenizer.eos_token_id:
            g.pop()
        gens.append((g, ended))
    return gens


def script_share(text, expected_script):
    """Crude off-script check: share of alphabetic chars whose Unicode name
    starts with the expected script. Not a language identifier."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    if expected_script == "CJK":
        hit = sum(1 for c in letters if "CJK" in unicodedata.name(c, ""))
    else:
        hit = sum(1 for c in letters if unicodedata.name(c, "").startswith(expected_script))
    return hit / len(letters)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def quantiles(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not xs:
        return None
    xs = sorted(xs)
    return {str(q): xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--languages", default=None, help="comma list overriding the config")
    ap.add_argument("--n-prompts", type=int, default=None)
    args = ap.parse_args()

    cfg = {**DEFAULTS, **yaml.safe_load(open(args.config))}
    if args.languages:
        cfg["languages"] = args.languages.split(",")
    if args.n_prompts:
        cfg["n_prompts"] = args.n_prompts
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg["dtype"]]
    if device == "cpu":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=dtype).to(device).eval()

    slug = cfg["model"].replace("/", "_")
    out_dir = Path(cfg["out_dir"]) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(), "config_file": args.config, "config": cfg,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": sys.version.split()[0], "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(model.dtype),
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tokenizer": {"bos_token": tokenizer.bos_token, "bos_token_id": tokenizer.bos_token_id,
                      "eos_token_id": tokenizer.eos_token_id,
                      "add_bos_token": getattr(tokenizer, "add_bos_token", None),
                      "pad_token_id": tokenizer.pad_token_id,
                      "chat_template_applied": False},
        "decoding": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                     "no_repeat_ngram_size": 0, "max_new_tokens": cfg["max_new_tokens"],
                     "batched_left_padding": True},
    }

    for lang in cfg["languages"]:
        prompts = build_prompts(tokenizer, lang, cfg)
        if not prompts:
            print(f"[{lang}] no documents long enough; skipping")
            continue
        records = []
        for b in range(0, len(prompts), cfg["batch_size"]):
            chunk = prompts[b:b + cfg["batch_size"]]
            gens = generate_batch(model, tokenizer, [p["prompt_ids"] for p in chunk], cfg, device)
            for p, (g, ended) in zip(chunk, gens):
                text = tokenizer.decode(g, skip_special_tokens=True)
                rec = {**p, "gen_ids": g, "gen_text": text,
                       **rm.summarize(g, ended_with_eos=ended),
                       "expected_script_share": script_share(
                           text, SCRIPT_OF.get(lang.split("_")[1], "LATIN"))}
                records.append(rec)
            print(f"[{lang}] {min(b + cfg['batch_size'], len(prompts))}/{len(prompts)}", flush=True)

        # batch-invariance check: regenerate the first n at batch size 1
        n_bi = min(cfg["batch_invariance_n"], len(records))
        exact, first_div = 0, []
        for r in records[:n_bi]:
            g1, _ = generate_batch(model, tokenizer, [r["prompt_ids"]], cfg, device)[0]
            if g1 == r["gen_ids"]:
                exact += 1
            else:
                k = next((i for i, (a, c) in enumerate(zip(g1, r["gen_ids"])) if a != c),
                         min(len(g1), len(r["gen_ids"])))
                first_div.append(k)

        n = len(records)
        loops = sum(r["loop"] for r in records)
        loops_w = sum(r["loop_with_window"] for r in records)
        eos = sum(r["ended_with_eos"] for r in records)
        summary = {
            "language": lang, "n": n,
            "loop_rate": loops / n, "loop_rate_ci95": wilson(loops, n),
            "loop_with_window_rate": loops_w / n, "loop_with_window_ci95": wilson(loops_w, n),
            "eos_rate": eos / n, "eos_rate_ci95": wilson(eos, n),
            "onset_quantiles": quantiles([r["onset"] for r in records if r["onset"] is not None]),
            "seq_rep_4_mean": sum(r["seq_rep_4"] for r in records) / n,
            "rep_r_mean": sum(r["rep_r"] for r in records) / n,
            "mean_gen_tokens": sum(r["n_tokens"] for r in records) / n,
            "expected_script_share_mean": (
                lambda xs: sum(xs) / len(xs) if xs else None)(
                [r["expected_script_share"] for r in records if r["expected_script_share"] is not None]),
            "batch_invariance": {"n_checked": n_bi, "exact_match": exact,
                                 "first_divergence_positions": first_div},
        }
        print(json.dumps(summary, indent=1), flush=True)
        # summary (small, committed) and per-generation records (large, gitignored)
        (out_dir / f"{lang}.json").write_text(json.dumps(
            {"provenance": provenance, "summary": summary,
             "records_file": f"{lang}.records.jsonl"}, ensure_ascii=False, indent=1))
        with open(out_dir / f"{lang}.records.jsonl", "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"written {out_dir / f'{lang}.json'} (+ records.jsonl)", flush=True)


if __name__ == "__main__":
    main()
