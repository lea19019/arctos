"""One table across results/loops_pilot/*/<lang>.json (and results/lesion/*.json).

    uv run src/loops_summary.py results/loops_pilot [results/lesion]
"""

import glob
import json
import sys
from pathlib import Path


def pilot_table(root):
    rows = []
    for f in sorted(glob.glob(f"{root}/*/*.json")):
        d = json.load(open(f))
        s, p = d["summary"], d["provenance"]
        bi = s["batch_invariance"]
        rows.append((Path(f).parent.name, s["language"], s["n"],
                     s["loop_rate"], s["loop_rate_ci95"], s["loop_with_window_rate"],
                     s["eos_rate"], s["seq_rep_4_mean"], s["rep_r_mean"],
                     s["onset_quantiles"], s["expected_script_share_mean"],
                     f"{bi['exact_match']}/{bi['n_checked']}", p["git_sha"][:8] if p["git_sha"] else "?"))
    if not rows:
        return
    print(f"{'model':<34} {'lang':<9} {'n':>4} {'loop':>6} {'95% CI':<15} {'loop>=32':>8} "
          f"{'eos':>5} {'srep4':>6} {'rep-r':>6} {'onset p50':>9} {'script':>6} {'bs-inv':>7}  sha")
    for m, lang, n, lr, ci, lw, eos, s4, rr, oq, sc, bi, sha in rows:
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "-"
        p50 = oq["0.5"] if oq else "-"
        sc_s = f"{sc:.2f}" if sc is not None else "-"
        print(f"{m:<34} {lang:<9} {n:>4} {lr:>6.3f} {ci_s:<15} {lw:>8.3f} {eos:>5.2f} "
              f"{s4:>6.3f} {rr:>6.3f} {str(p50):>9} {sc_s:>6} {bi:>7}  {sha}")


def lesion_table(root):
    for f in sorted(glob.glob(f"{root}/*.json")):
        d = json.load(open(f))
        r, c = d["results"], d["provenance"]["coordinate"]
        print(f"\n{d['provenance']['config']['model']}  L{c['layer']}[{c['j']},{c['k']}] = {c['weight']:+.4f}"
              f"   windows={d['provenance']['eval_windows']}x{d['provenance']['seq_len']}  sha={str(d['provenance']['git_sha'])[:8]}")
        print(f"{'condition':<22} {'ppl':>10} {'ratio':>9} {'95% CI':<22} {'|h0[j]|':>9} {'loop':>5} {'srep4':>6}  example")

        def line(x, ratio=None, ci=None):
            ma = abs(x["massive_activation"]["h0_j"])
            g = x["generation"]
            rs = f"x{ratio:.2f}" if ratio is not None else "-"
            cs = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "-"
            print(f"{x['name']:<22} {x['ppl']:>10.2f} {rs:>9} {cs:<22} {ma:>9.1f} {g['loop_rate']:>5.2f} "
                  f"{g['seq_rep_4_mean']:>6.3f}  {g['examples'][0][:50]!r}")
        line(r["baseline"])
        for x in r["doses"]:
            line(x, x["ppl_ratio"], x["ppl_ratio_ci95"])
        if "contribution_mean" in r:
            x = r["contribution_mean"]
            line(x, x["ppl_ratio"], x["ppl_ratio_ci95"])
            cal = x["calibration"]
            print(f"   calibration: mean contribution pos0 {cal['mean_contrib_pos0']:+.3f} (n={cal['n_pos0']}), "
                  f"rest {cal['mean_contrib_rest']:+.5f} (n={cal['n_rest']})")
        if "matched_random" in r:
            mr = r["matched_random"]
            print(f"{'matched-random null':<22} {'':>10} max x{mr['max_ratio']:.3f}  p95 x{mr['p95_ratio']:.3f}  "
                  f"median x{mr['median_ratio']:.3f}  ({mr['n_draws']} draws from top-{mr['pool']} |W|)")


if __name__ == "__main__":
    for root in sys.argv[1:]:
        if "lesion" in root:
            lesion_table(root)
        else:
            pilot_table(root)
