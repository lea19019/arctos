"""Paired-bootstrap 95% CIs for every perplexity ratio in an ablation JSON.

`ablate_sw.py` stores `ppl_each` per 2048-token window for the baseline and
for every ablation, so a CI on each ratio costs no GPU: resample windows with
replacement, recompute exp(mean log-ppl) for both arms on the same draw, and
take the ratio. CLAUDE.md forbids the bare means the earlier tables reported.

    uv run src/bootstrap_ci.py results/v5/*_ablation.json
    uv run src/bootstrap_ci.py --results-dir results/v5
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def ratio_ci(base_each, abl_each, B=2000, seed=0, alpha=0.05):
    """(point, lo, hi) for exp(mean log abl) / exp(mean log base), paired."""
    lb = np.log(np.asarray(base_each, dtype=np.float64))
    la = np.log(np.asarray(abl_each, dtype=np.float64))
    n = len(lb)
    assert len(la) == n, "unpaired arms"
    if not np.isfinite(la).all():
        # fp16 logits overflowed on at least one window: the corpus
        # perplexity is genuinely infinite (Subramanian et al. report the
        # same "-> inf" for Mistral). Report it as such rather than nan.
        return math.inf, math.inf, math.inf
    point = math.exp(la.mean() - lb.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    d = (la[idx].mean(axis=1) - lb[idx].mean(axis=1))
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return point, math.exp(lo), math.exp(hi)


def annotate(doc, B=2000, seed=0):
    """Add ratio / ratio_ci to every result in an ablation document, in place."""
    base = doc["baseline"]["ppl_each"]
    for r in doc["results"]:
        p, lo, hi = ratio_ci(base, r["ppl_each"], B=B, seed=seed)
        r["ratio"], r["ratio_ci95"] = p, [lo, hi]
    for key in ("null", "sa_results", "sa_null"):
        for r in doc.get(key, []) or []:
            if "ppl_each" in r:
                p, lo, hi = ratio_ci(base, r["ppl_each"], B=B, seed=seed)
                r["ratio"], r["ratio_ci95"] = p, [lo, hi]
    return doc


def fmt(r):
    lo, hi = r["ratio_ci95"]
    return f"x{r['ratio']:9.2f} [{lo:9.2f}, {hi:9.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true",
                    help="write ratio + ratio_ci95 back into the JSON")
    args = ap.parse_args()
    files = [Path(f) for f in args.files]
    if args.results_dir:
        files += sorted(Path(args.results_dir).glob("*_ablation.json"))
    for f in files:
        doc = json.loads(f.read_text())
        annotate(doc, B=args.B, seed=args.seed)
        n = len(doc["baseline"]["ppl_each"])
        print(f"\n{f}  (n_windows={n}, B={args.B}, paired bootstrap)")
        for r in doc["results"]:
            coord = (f"L{r['layer']}[{r['j']},{r['k']}]" if "layer" in r
                     else r.get("coords", ""))
            print(f"  {r['name']:<28} {str(coord):<22} {fmt(r)}")
        if doc.get("null"):
            mx = max(doc["null"], key=lambda r: r["ratio"])
            print(f"  null: n={len(doc['null'])} max {fmt(mx)}  ({mx['name']})")
        for r in doc.get("sa_results", []) or []:
            print(f"  {r['name']:<28} {'':<22} {fmt(r)}")
        if args.write:
            f.write_text(json.dumps(doc, indent=2))
            print(f"  (written)")


if __name__ == "__main__":
    main()
