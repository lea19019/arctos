"""Run the full super-weight pipeline for every model in sw_models.MODELS:

    detect_sw.py  ->  results/<model>_found.json
    ablate_sw.py  ->  results/<model>_ablation.json   (detector finds + Table 2)

then print ONE combined summary: every tested coordinate, where it came from
(our detector, the paper's Table 2, or both), and the ablation verdict.

Models are cached first with prefetch_models.py (login node). One model
failing (gated, out of memory) does not stop the others.

    uv run src/run_all.py
"""

import json
import subprocess
import sys
from pathlib import Path

from ablate_sw import TABLE2
from sw_models import MODELS

RESULTS = Path("results")
SRC = Path(__file__).parent


def run_script(script, *args):
    """Run a sibling script as a subprocess; True on success."""
    cmd = [sys.executable, str(SRC / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def main():
    # ---- stage 1: detect + ablate per model ----
    failed = []
    for m in MODELS:
        slug = m.replace("/", "_")
        found_json = RESULTS / f"{slug}_found.json"
        ok = (run_script("detect_sw.py", "--model", m, "--out", str(found_json))
              and run_script("ablate_sw.py", "--model", m,
                             "--candidates", str(found_json)))
        if not ok:
            failed.append(m)
            print(f"!! {m} failed — continuing with the rest")

    # ---- stage 2: combined summary from the JSON files ----
    print("\n" + "=" * 78)
    print("SUMMARY — every coordinate tested, its source, and what ablation says")
    print("=" * 78)

    for m in MODELS:
        slug = m.replace("/", "_")
        ablation_path = RESULTS / f"{slug}_ablation.json"
        print(f"\n### {m}")
        if not ablation_path.exists():
            print("    no results (run failed — see above)")
            continue

        data = json.loads(ablation_path.read_text())

        # Which coordinates came from where?
        found_path = RESULTS / f"{slug}_found.json"
        found_coords = set()
        if found_path.exists():
            det = json.loads(found_path.read_text())
            found_coords = {(f["layer"], f["j"], f["k"]) for f in det["found"]}
        table2_coords = set(TABLE2.get(m, []))

        print(f"    revision: {data['revision_resolved']}   "
              f"baseline ppl: {data['baseline']['ppl']:.2f}")
        print(f"    {'coordinate':<20} {'source':<15} {'weight':>9} "
              f"{'ppl x':>7} {'KL':>7}  verdict")
        base_ppl = data["baseline"]["ppl"]
        real = []
        for r in data["results"]:
            coord = (r["layer"], r["j"], r["k"])
            src = " + ".join(
                s for s, hit in [("found", coord in found_coords),
                                 ("table2", coord in table2_coords)] if hit
            ) or "?"
            coord_s = f"L{r['layer']}[{r['j']},{r['k']}]"
            print(f"    {coord_s:<20} {src:<15} {r['weight']:>9.4f} "
                  f"{'x' + format(r['ppl'] / base_ppl, '.1f'):>7} "
                  f"{r['kl']:>7.3f}  {r['verdict']}")
            if r["verdict"] == "CATASTROPHIC":
                real.append(coord_s)

        # One-line verdict per model: what is real, does Table 2 hold up?
        confirmed = [c for c in table2_coords
                     if any(r["layer"] == c[0] and r["j"] == c[1] and r["k"] == c[2]
                            and r["verdict"] == "CATASTROPHIC"
                            for r in data["results"])]
        print(f"    => real super weights: {', '.join(real) or 'none'}")
        print(f"    => Table 2 for this model: {len(confirmed)} of "
              f"{len(table2_coords)} confirmed by ablation")

    if failed:
        print(f"\nFailed models: {', '.join(failed)}")


if __name__ == "__main__":
    main()
