"""Aggregate the phase-two GEM run (results/{model}/q6gem/q6_summary.json).

Focuses on the gem questions, scored with XCOMET-XL where available:
  - GPTQ MT-vs-generic calibration (Δ chrf & comet), by bit-width, per pair
    (en-arz = the low-resource regime the literature flags).
  - KEEP salient/super-weight FP16 recovery at W2/W3 (comet).
  - ALLOC Fisher-mixed vs uniform (exploratory).
  - super-weight KL per model (multilingual super-weight study).

  python scripts/q6gem_collect.py
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["eurollm-9b-instruct", "tower-instruct-7b-v0.2", "aya-expanse-8b",
          "llama-3.1-8b-instruct", "tower-base-7b-v0.1", "tower-plus-9b",
          "gemma-3-12b-it", "bloom-7b1"]
PAIRS = ["cs-de", "en-zh", "en-arz"]


def _f(x, d=1):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "-"


def _chrf(v):
    """Extract chrF++ from a cell that's either a float (old) or {chrf,comet}."""
    return v.get("chrf") if isinstance(v, dict) else v


def _comet(v):
    return v.get("comet") if isinstance(v, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="q6gem", help="results subdir (q6gem | q6extreme)")
    args = ap.parse_args()
    merged = {}
    for m in MODELS:
        p = ROOT / "results" / m / args.subdir / "q6_summary.json"
        if p.exists():
            try:
                merged[m] = json.loads(p.read_text())
            except Exception:
                pass
    if not merged:
        print(f"(no {args.subdir} results yet)")
        return

    print("=== SUPER WEIGHT (causal-KL) — multilingual super-weight study ===")
    for m, d in merged.items():
        c = (d.get("find_super_weights", {}).get("candidates") or [{}])[0]
        print(f"  {m:24} L{c.get('layer')!s:>3}  KL={_f(c.get('ablation_kl'),4)}")

    print("\n=== SHRINK cliff (chrf | comet), cs-de & en-arz, per bit-level ===")
    for m, d in merged.items():
        sh = d.get("shrink", {})
        for lvl, row in sh.items():
            cells = "  ".join(
                f"{p}:{_f(_chrf(row.get(p)))}/{_f(_comet(row.get(p)),3)}" for p in ("cs-de", "en-arz"))
            print(f"  {m:20} {lvl:>12}  {cells}")

    print("\n=== GPTQ: MT-minus-generic calibration (chrf | comet), per pair ===")
    for m, d in merged.items():
        g = d.get("gptq", {})
        for b in sorted(g):
            blk = g[b]
            if isinstance(blk, dict) and "mt_minus_generic" in blk:
                mg = blk["mt_minus_generic"]
                cells = "  ".join(
                    f"{p}:{_f(mg.get(p,{}).get('chrf'))}/{_f(mg.get(p,{}).get('comet'),3)}"
                    for p in PAIRS)
                print(f"  {m:22} {b.upper():>3}  {cells}")

    print("\n=== KEEP: salient-FP16 recovery vs RTN at W2/W3 (comet; cs-de & en-arz) ===")
    for m, d in merged.items():
        keep = d.get("keep", {})
        for bk in sorted(k for k in keep if k.startswith("w")):
            blk = keep[bk]
            if not isinstance(blk, dict):
                continue
            def cm(v, p):
                return _f((blk.get(v, {}).get(p) or {}).get("comet"), 3) if isinstance(blk.get(v, {}).get(p), dict) else _f(blk.get(v, {}).get(p), 3)
            row = "  ".join(f"{v}={cm(v,'cs-de')}/{cm(v,'en-arz')}"
                            for v in ("rtn", "keep_salient_fp16", "rtn_plus_superweight_fp16") if v in blk)
            print(f"  {m:22} {bk.upper()}  {row}")

    print("\n=== ALLOC: Fisher-mixed minus uniform (exploratory) ===")
    for m, d in merged.items():
        a = d.get("alloc", {}).get("mixed_minus_uniform")
        if a:
            print(f"  {m:24} " + "  ".join(f"{p}:{_f(a.get(p,{}).get('comet'),3)}" for p in PAIRS))


if __name__ == "__main__":
    main()
