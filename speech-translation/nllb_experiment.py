"""NLLB-200 compression baseline: FP16 vs BnB INT8 vs BnB NF4 vs CTranslate2 INT8.

Languages: English, Spanish, French (all supported by XTTS v2).
Metrics: BLEU, chrF++ (word_order=2), XCOMET-XL (WMT25 compression task primary metric).
Data: FLORES+ dev split (100 examples per pair by default).

Compression variants (each reloads the model to keep VRAM clean):
  fp16        — baseline, standard HuggingFace loading
  bnb_int8    — LLM.int8() via bitsandbytes (uniform INT8)
  bnb_nf4     — NF4 4-bit via bitsandbytes (uniform INT4, nf4 quanttype)
  ct2_int8    — CTranslate2 INT8 (fused kernels, fastest at inference)

Inference timing: N_WARMUP examples are translated first (discarded), then the
timed pass covers all N_EXAMPLES. Reported: tok/s (output tokens per second),
ms/sent (mean latency per sentence), peak VRAM (GB).

Run from repo root:
    python speech-translation/nllb_experiment.py \\
        --config   speech-translation/configs/nllb.yaml \\
        --output   speech-translation/results/nllb \\
        [--device  cpu|cuda]          # auto-detect if omitted
        [--n-examples 100]
        [--n-warmup 3]
        [--variants fp16 bnb_int8]

Outputs:
    results/nllb/results.json    — full per-pair per-variant scores
    results/nllb/summary.tsv     — human-readable table
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ── optional imports ─────────────────────────────────────────────────────────

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import ctranslate2
    HAS_CT2 = True
except ImportError:
    HAS_CT2 = False

# ── CUDA probe ────────────────────────────────────────────────────────────────

def _cuda_ok() -> bool:
    """Returns True only when CUDA is available AND compute is not prohibited."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


_CUDA_OK: bool | None = None  # cached after first call


def cuda_ok() -> bool:
    global _CUDA_OK
    if _CUDA_OK is None:
        _CUDA_OK = _cuda_ok()
    return _CUDA_OK

# ── FLORES data ───────────────────────────────────────────────────────────────

def load_pair_jsonl(data_dir: Path, src: str, tgt: str, n: int) -> list[dict]:
    path = data_dir / "nllb" / f"{src}-{tgt}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"FLORES data not found at {path}. "
            "Run: python speech-translation/fetch_flores.py"
        )
    rows = []
    with path.open() as f:
        for line in f:
            if len(rows) >= n:
                break
            rows.append(json.loads(line))
    return rows

# ── HF NLLB loading ───────────────────────────────────────────────────────────

def load_nllb_hf(hf_name: str, variant_cfg: dict, device: str) -> tuple:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    name = variant_cfg["name"]

    # device_map: use "auto" on real GPU, "cpu" otherwise
    device_map = device if not cuda_ok() else "auto"
    if device == "cpu":
        device_map = "cpu"

    if name == "fp16":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_name,
            dtype=torch.float16 if cuda_ok() else torch.float32,
            device_map=device_map,
        )
    elif name == "bnb_int8":
        if not HAS_BNB:
            raise ImportError("bitsandbytes not installed; run: pip install bitsandbytes")
        if not cuda_ok():
            raise RuntimeError("BnB INT8 requires a functional CUDA GPU")
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_name, quantization_config=qcfg, device_map="auto"
        )
    elif name == "bnb_nf4":
        if not HAS_BNB:
            raise ImportError("bitsandbytes not installed; run: pip install bitsandbytes")
        if not cuda_ok():
            raise RuntimeError("BnB NF4 requires a functional CUDA GPU")
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=variant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_name, quantization_config=qcfg, device_map="auto"
        )
    else:
        raise ValueError(f"Unknown HF variant: {name}")

    model.eval()
    return model, tokenizer

# ── translation ───────────────────────────────────────────────────────────────

def _translate_hf_batch(
    model,
    tokenizer,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    num_beams: int,
    max_length: int,
) -> tuple[list[str], int]:
    """Translate texts; return (hypotheses, total_output_tokens)."""
    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    all_hyps: list[str] = []
    total_out_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_id,
                num_beams=num_beams,
                max_length=max_length,
            )

        all_hyps.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
        total_out_tokens += int(out_ids.numel())

    return all_hyps, total_out_tokens


def translate_hf(
    model,
    tokenizer,
    sources: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    num_beams: int,
    max_length: int,
    n_warmup: int,
) -> tuple[list[str], dict[str, float]]:
    """Translate with timed warmup; returns (hyps, timing_dict)."""
    # Warmup: runs but results discarded
    if n_warmup > 0 and sources:
        warm_texts = sources[:min(n_warmup, len(sources))]
        _translate_hf_batch(model, tokenizer, warm_texts, src_lang, tgt_lang,
                             batch_size, num_beams, max_length)
        if cuda_ok():
            torch.cuda.synchronize()

    # Timed pass
    t0 = time.perf_counter()
    hyps, total_out_tokens = _translate_hf_batch(
        model, tokenizer, sources, src_lang, tgt_lang, batch_size, num_beams, max_length
    )
    if cuda_ok():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    timing = {
        "elapsed_s": round(elapsed, 3),
        "tok_per_sec": round(total_out_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "latency_ms": round(elapsed / len(sources) * 1000, 1) if sources else 0.0,
    }
    return hyps, timing

# ── CTranslate2 translation ──────────────────────────────────────────────────

def translate_ct2(
    translator,
    tokenizer,
    sources: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    num_beams: int,
    max_length: int,
    n_warmup: int,
) -> tuple[list[str], dict[str, float]]:
    tokenizer.src_lang = src_lang

    def _run(texts: list[str]) -> tuple[list[str], int]:
        all_hyps, total_tokens = [], 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            src_tokens = [
                tokenizer.convert_ids_to_tokens(tokenizer(t).input_ids)
                for t in batch
            ]
            results = translator.translate_batch(
                src_tokens,
                target_prefix=[[tgt_lang]] * len(batch),
                beam_size=num_beams,
                max_decoding_length=max_length,
            )
            for result in results:
                tokens = result.hypotheses[0]
                if tokens and tokens[0] == tgt_lang:
                    tokens = tokens[1:]
                ids = tokenizer.convert_tokens_to_ids(tokens)
                all_hyps.append(tokenizer.decode(ids, skip_special_tokens=True))
                total_tokens += len(tokens)
        return all_hyps, total_tokens

    # Warmup
    if n_warmup > 0 and sources:
        _run(sources[:min(n_warmup, len(sources))])

    t0 = time.perf_counter()
    hyps, total_out_tokens = _run(sources)
    elapsed = time.perf_counter() - t0

    timing = {
        "elapsed_s": round(elapsed, 3),
        "tok_per_sec": round(total_out_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "latency_ms": round(elapsed / len(sources) * 1000, 1) if sources else 0.0,
    }
    return hyps, timing

# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    sources: list[str],
    hyps: list[str],
    refs: list[str],
    use_xcomet: bool,
) -> dict[str, float]:
    from sacrebleu.metrics import BLEU, CHRF

    bleu_score = float(BLEU(tokenize="intl").corpus_score(hyps, [refs]).score)
    chrfpp = float(CHRF(word_order=2).corpus_score(hyps, [refs]).score)
    scores: dict[str, float] = {"bleu": bleu_score, "chrfpp": chrfpp}

    if use_xcomet:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.eval.metrics import comet_score
            xc_gpus = 1 if cuda_ok() else 0
            sys_score, _ = comet_score(
                sources, hyps, refs, model_name="Unbabel/XCOMET-XL",
                gpus=xc_gpus,
            )
            scores["xcomet_xl"] = sys_score
        except Exception as e:
            print(f"  [warn] XCOMET-XL failed: {e}", file=sys.stderr)
            scores["xcomet_xl"] = float("nan")

    return scores

# ── VRAM helper ───────────────────────────────────────────────────────────────

def peak_vram_gb() -> float:
    if cuda_ok():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_peak_vram() -> None:
    if cuda_ok():
        torch.cuda.reset_peak_memory_stats()

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config",     type=Path, default=Path("speech-translation/configs/nllb.yaml"))
    ap.add_argument("--output",     type=Path, default=Path("speech-translation/results/nllb"))
    ap.add_argument("--data-dir",   type=Path, default=Path("speech-translation/data"))
    ap.add_argument("--n-examples", type=int,  default=None)
    ap.add_argument("--n-warmup",   type=int,  default=3,
                    help="Warmup sentences translated before timing (default 3)")
    ap.add_argument("--device",     choices=["cpu", "cuda", "auto"], default="auto",
                    help="Force device; 'auto' picks CUDA if available")
    ap.add_argument("--variants",   nargs="+", help="Subset of variant names to run")
    ap.add_argument("--no-xcomet",  action="store_true",
                    help="Skip XCOMET-XL (faster; useful for CPU smoke tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    hf_name:    str        = cfg["model"]["hf_name"]
    pairs:      list       = cfg["pairs"]
    variants:   list[dict] = cfg["compression_variants"]
    n_examples: int        = args.n_examples or cfg.get("n_examples", 100)
    batch_size: int        = cfg.get("batch_size", 16)
    num_beams:  int        = cfg.get("num_beams", 4)
    max_length: int        = cfg.get("max_length", 256)
    use_xcomet: bool       = cfg.get("eval", {}).get("xcomet", True) and not args.no_xcomet

    device: str = args.device
    if device == "auto":
        device = "cuda" if cuda_ok() else "cpu"
    print(f"Device: {device}  |  CUDA usable: {cuda_ok()}\n")

    # Filter variants
    if args.variants:
        variants = [v for v in variants if v["name"] in args.variants]

    # Load FLORES data
    pair_data: dict[str, dict] = {}
    for src, tgt in pairs:
        key = f"{src}-{tgt}"
        rows = load_pair_jsonl(args.data_dir, src, tgt, n_examples)
        pair_data[key] = {
            "sources":    [r["source"] for r in rows],
            "references": [r["target"] for r in rows],
        }
    print(f"Loaded {n_examples} examples × {len(pairs)} pairs.\n")

    all_results: dict[str, Any] = {}

    for variant in variants:
        name = variant["name"]
        print(f"{'='*60}")
        print(f"Variant: {name}  |  warmup={args.n_warmup}")
        print(f"{'='*60}")

        reset_peak_vram()
        translator = model = tokenizer = None
        is_ct2 = name == "ct2_int8"

        try:
            if is_ct2:
                if not HAS_CT2:
                    print("  [skip] ctranslate2 not installed\n")
                    continue
                ct2_dir = variant.get("ct2_model_dir", "")
                if not Path(ct2_dir).exists():
                    print(f"  [skip] CT2 model not found at {ct2_dir} — run precache.sh\n")
                    continue
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(hf_name)
                ct2_device = "cuda" if cuda_ok() else "cpu"
                translator = ctranslate2.Translator(ct2_dir, device=ct2_device, inter_threads=1)
                print(f"  CTranslate2 model loaded from {ct2_dir} on {ct2_device}")
            else:
                model, tokenizer = load_nllb_hf(hf_name, variant, device)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  Params: {n_params/1e6:.0f}M  |  Load VRAM: {peak_vram_gb():.2f} GB")
        except (RuntimeError, ImportError) as e:
            print(f"  [skip] {e}\n")
            continue

        variant_results: dict[str, Any] = {"variant": name, "device": device}

        for src, tgt in pairs:
            key = f"{src}-{tgt}"
            sources    = pair_data[key]["sources"]
            references = pair_data[key]["references"]
            print(f"\n  Pair: {key}  ({len(sources)} examples + {args.n_warmup} warmup)")

            reset_peak_vram()

            if is_ct2:
                hyps, timing = translate_ct2(
                    translator, tokenizer, sources, src, tgt,
                    batch_size, num_beams, max_length, args.n_warmup
                )
            else:
                hyps, timing = translate_hf(
                    model, tokenizer, sources, src, tgt,
                    batch_size, num_beams, max_length, args.n_warmup
                )

            metrics = compute_metrics(sources, hyps, references, use_xcomet)
            infer_vram = peak_vram_gb()

            pair_res = {
                "bleu":          round(metrics["bleu"], 2),
                "chrfpp":        round(metrics["chrfpp"], 2),
                "xcomet_xl":     round(metrics.get("xcomet_xl", float("nan")), 4),
                "tok_per_sec":   timing["tok_per_sec"],
                "latency_ms":    timing["latency_ms"],
                "elapsed_s":     timing["elapsed_s"],
                "peak_vram_gb":  round(infer_vram, 2),
                "n_examples":    len(sources),
                "n_warmup":      args.n_warmup,
            }
            variant_results[key] = pair_res

            xc = pair_res["xcomet_xl"]
            print(f"    BLEU={pair_res['bleu']:.1f}  chrF++={pair_res['chrfpp']:.1f}  "
                  f"XCOMET-XL={xc:.4f}")
            print(f"    {timing['tok_per_sec']:.0f} tok/s  |  "
                  f"{timing['latency_ms']:.0f} ms/sent  |  "
                  f"{infer_vram:.2f} GB VRAM")

        all_results[name] = variant_results

        del model, tokenizer, translator
        gc.collect()
        if cuda_ok():
            torch.cuda.empty_cache()
        print()

    # ── Save JSON ────────────────────────────────────────────────────────────
    results_path = args.output / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {results_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"{'Variant':<16} {'Pair':<22} {'BLEU':>6} {'chrF++':>7} {'XCOMET-XL':>10} "
          f"{'tok/s':>7} {'ms/sent':>8} {'VRAM GB':>8}")
    print("-" * 85)
    for vname, vres in all_results.items():
        for key, pres in vres.items():
            if not isinstance(pres, dict):
                continue
            xc = pres.get("xcomet_xl", float("nan"))
            print(f"{vname:<16} {key:<22} {pres['bleu']:>6.1f} {pres['chrfpp']:>7.1f} "
                  f"{xc:>10.4f} {pres['tok_per_sec']:>7.0f} "
                  f"{pres['latency_ms']:>8.0f} {pres['peak_vram_gb']:>8.2f}")

    # ── TSV ──────────────────────────────────────────────────────────────────
    tsv_path = args.output / "summary.tsv"
    with tsv_path.open("w") as f:
        f.write("variant\tpair\tbleu\tchrfpp\txcomet_xl\ttok_per_sec\t"
                "latency_ms\telapsed_s\tpeak_vram_gb\tn_examples\n")
        for vname, vres in all_results.items():
            for key, pres in vres.items():
                if not isinstance(pres, dict):
                    continue
                xc = pres.get("xcomet_xl", float("nan"))
                f.write(
                    f"{vname}\t{key}\t{pres['bleu']:.2f}\t{pres['chrfpp']:.2f}\t"
                    f"{xc:.4f}\t{pres['tok_per_sec']:.1f}\t{pres['latency_ms']:.1f}\t"
                    f"{pres['elapsed_s']:.3f}\t{pres['peak_vram_gb']:.2f}\t"
                    f"{pres['n_examples']}\n"
                )
    print(f"Summary TSV → {tsv_path}")


if __name__ == "__main__":
    main()
