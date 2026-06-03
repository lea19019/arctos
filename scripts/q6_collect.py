"""Aggregate finished q6 runs into one cross-model table.

Reads every results/{model}/q6/q6_summary.json that exists and prints a compact
find/shrink/keep/prune summary. Safe to run any time — skips models not done.

  python scripts/q6_collect.py            # table to stdout
  python scripts/q6_collect.py --json     # merged JSON to stdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "aya-expanse-8b", "llama-3.1-8b-instruct", "bloom-7b1", "eurollm-9b-instruct",
    "tower-base-7b-v0.1", "tower-instruct-7b-v0.2", "tower-plus-9b", "gemma-3-12b-it",
]


def _g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    merged = {}
    for m in MODELS:
        f = ROOT / "results" / m / "q6" / "q6_summary.json"
        if f.exists():
            try:
                merged[m] = json.loads(f.read_text())
            except Exception:
                pass

    if args.json:
        print(json.dumps(merged, indent=2))
        return

    if not merged:
        print("(no q6_summary.json files yet)")
        return

    # FIND table
    print("FIND — super weight (causal-ranked) + AWQ calibration shift")
    print(f"{'model':26} {'SW layer':>8} {'SW KL':>9} {'AWQ J(mt|src)':>13} {'AWQ ρ':>7}")
    for m, d in merged.items():
        cands = _g(d, "find_super_weights", "candidates", default=[])
        sw_layer = cands[0]["layer"] if cands else None
        sw_kl = cands[0].get("ablation_kl") if cands else None
        jac = _g(d, "find_awq_calibration_shift", "mt_vs_source", "mean_top1pct_jaccard")
        rho = _g(d, "find_awq_calibration_shift", "mt_vs_source", "mean_spearman")
        print(f"{m:26} {str(sw_layer):>8} {('%.3g'%sw_kl) if sw_kl is not None else '-':>9} "
              f"{('%.3f'%jac) if jac is not None else '-':>13} {('%.3f'%rho) if rho is not None else '-':>7}")

    # SHRINK table (cs-de chrF++ by bits)
    print("\nSHRINK — chrF++ on cs-de by bit-width (baseline -> W4 -> W3 -> W2)")
    for m, d in merged.items():
        base = _g(d, "baseline_chrf", "cs-de")
        row = [("base", base)]
        for b in (4, 3, 2):
            row.append((f"W{b}", _g(d, "shrink", f"rtn_w{b}", "cs-de")))
        s = "  ".join(f"{k}={v:.1f}" if isinstance(v, (int, float)) else f"{k}=-" for k, v in row)
        print(f"{m:26} {s}")

    # KEEP table (cs-de chrF++). Handles both schemas: old flat keep[variant]
    # and new nested keep["w{b}"][variant].
    print("\nKEEP — chrF++ on cs-de (protection schemes; higher = better)")
    for m, d in merged.items():
        keep = _g(d, "keep", default={})
        # find the per-bit blocks (new schema) or treat the whole thing as one (old)
        bit_blocks = {k: v for k, v in keep.items() if k.startswith("w") and isinstance(v, dict)}
        if not bit_blocks:
            bit_blocks = {f"w{keep.get('bits','?')}": keep}
        for bkey, block in sorted(bit_blocks.items()):
            parts = []
            for v in ("rtn", "keep_salient_fp16", "rtn_plus_superweight_fp16"):
                val = _g(block, v, "cs-de")
                parts.append(f"{v}={val:.1f}" if isinstance(val, (int, float)) else f"{v}=-")
            # AWQ variant name varies (awq / awq_a0.5)
            awq = next((kk for kk in block if kk.startswith("awq")), None)
            if awq:
                val = _g(block, awq, "cs-de")
                parts.append(f"{awq}={val:.1f}" if isinstance(val, (int, float)) else f"{awq}=-")
            print(f"{m:26} {bkey.upper()}  " + "  ".join(parts))

    # PRUNE stress test
    print("\nPRUNE — super-weight stress (ablate 1 SW vs 1000 largest-|W|), cs-de chrF++")
    for m, d in merged.items():
        st = _g(d, "prune", "superweight_stress", default={})
        one = _g(st, "ablate_1_superweight", "cs-de")
        big = next((v.get("cs-de") for k, v in st.items() if k.startswith("ablate_") and "largest" in k), None)
        f1 = f"{one:.1f}" if isinstance(one, (int, float)) else "-"
        fb = f"{big:.1f}" if isinstance(big, (int, float)) else "-"
        print(f"{m:26} ablate_1_SW={f1}   ablate_1000_largest={fb}")


if __name__ == "__main__":
    main()
