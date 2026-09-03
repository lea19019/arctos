"""How well does a detector run agree with Yu et al. Table 2?

Reads a directory of detect_sw.py outputs and scores them against TABLE2:
how many of the paper's coordinates came back, how many extra candidates
were returned, and -- when the ablation JSON is present -- how many of each
group survive the causal check.

Point it at any generation's results to compare detectors on equal terms:

    uv run src/table2_agreement.py                        # results/   (v2)
    uv run src/table2_agreement.py --results-dir results/v1
"""

import argparse
import json
from pathlib import Path

from ablate_sw import TABLE2
from sw_models import MODELS


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    root = Path(args.results_dir)

    print(f"{'model':<26} {'T2':>3} {'ret':>4} {'extra':>6} {'recovered':>10}  "
          f"missed")
    print("-" * 100)
    tt = tr = te = 0
    versions = set()
    for m in MODELS:
        slug = m.replace("/", "_")
        det = load(root / f"{slug}_found.json")
        if det is None:
            print(f"{m.split('/')[-1][:26]:<26}  (no results in {root})")
            continue
        versions.add(det.get("detector_version", 1))
        found = {(f["layer"], f["j"], f["k"]) for f in det["found"]}
        t2 = set(TABLE2.get(m, []))
        rec, extra = found & t2, found - t2
        tt += len(t2); tr += len(rec); te += len(extra)
        missed = ", ".join(f"L{l}[{j},{k}]" for l, j, k in sorted(t2 - found)) or "-"
        print(f"{m.split('/')[-1][:26]:<26} {len(t2):>3} {len(found):>4} "
              f"{len(extra):>6} {len(rec):>10}  {missed}")
    print("-" * 100)
    print(f"{'TOTAL':<26} {tt:>3} {'':>4} {te:>6} {tr:>10}   "
          f"= {tr}/{tt} of Table 2 recovered  (detector v{sorted(versions)})")

    # If ablations exist, say which group actually survives the causal check:
    # a detector that returns more coordinates is only better if they hold up.
    print()
    print(f"{'model':<26} {'catastrophic candidates (ppl ratio)':<60} source")
    print("-" * 100)
    for m in MODELS:
        slug = m.replace("/", "_")
        abl = load(root / f"{slug}_ablation.json")
        if abl is None:
            continue
        t2 = set(TABLE2.get(m, []))
        base = abl["baseline"]["ppl"]
        hits = [r for r in abl["results"] if r["verdict"] == "CATASTROPHIC"]
        if not hits:
            print(f"{m.split('/')[-1][:26]:<26} {'none':<60}")
            continue
        for r in hits:
            coord = (r["layer"], r["j"], r["k"])
            src = "Table 2" if coord in t2 else "detector only (NOT in Table 2)"
            label = f"L{r['layer']}[{r['j']},{r['k']}]  x{r['ppl'] / base:.0f}  KL {r['kl']:.2f}"
            print(f"{m.split('/')[-1][:26]:<26} {label:<60} {src}")


if __name__ == "__main__":
    main()
