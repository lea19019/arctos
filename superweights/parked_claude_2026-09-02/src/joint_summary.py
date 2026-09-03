"""One table across every model in experiments/joint_ablation.

Reads *_ablation.json (decoder-only) and *_encdec.json (NLLB) from the given
results dirs and classifies each model by the README's criteria:

  jointly critical : joint-all lower CI bound > max of the joint null
  distributed      : jointly critical AND every individual upper CI < 2
  single           : jointly critical AND some leave-one-out drops below 2
  none / inert     : joint-all CI inside the null
  no candidates    : detector returned nothing (ablation never ran)

    uv run src/joint_summary.py results/v6 results/modern_v6 results/bases_v6 \
        results/small_v6 results/multi_v6 results/encdec_v6
"""

import argparse
import glob
import json
from pathlib import Path


def ci(r):
    lo, hi = r["ratio_ci95"]
    if r["ratio"] == float("inf"):
        return "x inf (fp16 overflow)"
    return f"x{r['ratio']:.2f} [{lo:.2f},{hi:.2f}]"


def classify(doc):
    res = doc["results"]
    ind = [r for r in res if r.get("kind") == "individual"]
    joint = next((r for r in res if r["name"] == "joint-all"), None)
    loo = [r for r in res if r.get("kind") == "loo"]
    null = doc.get("null") or []
    null_max = max((r["ratio"] for r in null), default=None)
    null_joint_max = max((r["ratio"] for r in null if r.get("kind") == "null-joint"),
                         default=null_max)
    if not ind:
        return "no candidates", ind, joint, loo, null_max, null_joint_max
    if joint is None:                       # one candidate: joint == individual
        joint = ind[0]
    ref = null_joint_max if null_joint_max is not None else 2.0
    jl = joint["ratio_ci95"][0]
    if jl <= max(ref, 1.0) * 1.0:
        return "inert (joint within null)", ind, joint, loo, null_max, null_joint_max
    if jl < 10:
        return "weak (joint > null, < x10)", ind, joint, loo, null_max, null_joint_max
    if all(r["ratio_ci95"][1] < 2 for r in ind):
        return "DISTRIBUTED", ind, joint, loo, null_max, null_joint_max
    if any(r["ratio_ci95"][1] < 2 for r in loo) or len(ind) == 1:
        return "SINGLE", ind, joint, loo, null_max, null_joint_max
    return "mixed", ind, joint, loo, null_max, null_joint_max


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        for f in sorted(glob.glob(f"{d}/*_ablation.json") + glob.glob(f"{d}/*_encdec.json")):
            doc = json.loads(Path(f).read_text())
            label, ind, joint, loo, nmax, njmax = classify(doc)
            best = max(ind, key=lambda r: r["ratio"]) if ind else None
            sa = next((r for r in (doc.get("sa_results") or [])
                       if r["name"] == "sa-zero all"), None)
            if sa is None and doc.get("sa_results"):
                sa = doc["sa_results"][0]
            sa_null_max = max((r["ratio"] for r in doc.get("sa_null") or []), default=None)
            conc = [r for r in res if r.get("kind") == "concentration"]
            k10 = next((r["k"] for r in conc if r["ratio_ci95"][0] > 10), None)
            rows.append({
                "conc": (f"k>=10x at {k10}" if k10 else (f"none<=k{max(r['k'] for r in conc)}" if conc else "—")),
                "dir": d, "model": doc["model"], "n": len(ind), "label": label,
                "best_ind": ci(best) if best else "—",
                "joint": ci(joint) if joint else "—",
                "null": (f"ind {nmax:.2f} / joint {njmax:.2f}" if nmax else "—"),
                "sa": ci(sa) if sa else "—",
                "sa_null": f"{sa_null_max:.2f}" if sa_null_max else "—",
                "cont": (joint or {}).get("continuation", "") or "",
                "sha": (doc.get("git_sha") or "")[:8],
                "windows": len(doc["baseline"].get("ppl_each") or doc["baseline"].get("loss_each") or []),
            })
    print(f"{'model':<38} {'n':>2} {'best individual':<26} {'joint-all':<30} "
          f"{'null max':<24} {'SA-zero (all)':<28} {'SA null':>7} {'concentration':<16} class")
    print("-" * 206)
    for r in rows:
        print(f"{r['model']:<38} {r['n']:>2} {r['best_ind']:<26} {r['joint']:<30} "
              f"{r['null']:<24} {r['sa']:<28} {r['sa_null']:>7} {r['conc']:<16} {r['label']}")
    print(f"\ncoverage: {len(rows)} model files; windows/sentences per model: "
          f"{sorted(set(r['windows'] for r in rows))}; git: {sorted(set(r['sha'] for r in rows))}")


if __name__ == "__main__":
    main()
