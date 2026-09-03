"""Assertions for slurm/probe.sh -- OLMo-1B, whose answer is known.

Strengthened 2026-09-02: two detector generations passed the old probe while
broken, because it tested one side of a range. Every quantity now has both
bounds, and the causal side uses bootstrap CIs and the null, not the eyeball
verdict.

    python src/probe_check.py results/probe_v6/allenai_OLMo-1B-0724-hf_ablation.json \
                              results/probe_v6/allenai_OLMo-1B-0724-hf_found.json
"""

import json
import sys

d = json.load(open(sys.argv[1]))
det = json.load(open(sys.argv[2]))
fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


n = len(det["found"])
check(1 <= n <= det["params"]["max_sw"],
      f"detector returned {n} candidates (need 1..{det['params']['max_sw']})")
check(det["stop_reason"].startswith("no super activation survives"),
      f"stop reason is the paper's own criterion: {det['stop_reason']!r}")
coords = {(f["layer"], f["j"], f["k"]) for f in det["found"]}
check((1, 1764, 1710) in coords, "detector found L1[1764,1710]")
check((2, 1764, 8041) in coords,
      "detector found L2[1764,8041] (Table 2 layer typo corrected)")

by = {(r["layer"], r["j"], r["k"]): r for r in d["results"]
      if r.get("kind") == "individual"}
r = by.get((1, 1764, 1710))
lo, hi = r["ratio_ci95"]
check(1000 <= lo and hi <= 20000,
      f"L1[1764,1710] x{r['ratio']:.0f} CI [{lo:.0f},{hi:.0f}] within [1000,20000] (v5: x3667)")
r = by.get((2, 1764, 8041))
lo, hi = r["ratio_ci95"]
check(0.95 <= lo and hi <= 1.10,
      f"L2[1764,8041] x{r['ratio']:.3f} CI [{lo:.3f},{hi:.3f}] within [0.95,1.10]")
r = by.get((1, 1764, 8041))
check(r is not None and abs(r["weight"]) < 0.01 and r["ratio_ci95"][1] < 1.05,
      f"paper's L1[1764,8041] holds ~0 (w={r['weight'] if r else float('nan'):.4f}) and does nothing")
joint = [r for r in d["results"] if r["name"] == "joint-all"]
if joint:
    lo, hi = joint[0]["ratio_ci95"]
    check(1000 <= lo and hi <= 20000,
          f"joint-all x{joint[0]['ratio']:.0f} CI [{lo:.0f},{hi:.0f}] within [1000,20000]")
if d.get("null"):
    mx = max(r["ratio"] for r in d["null"])
    check(mx < 1.5, f"null max x{mx:.3f} < 1.5 over {len(d['null'])} draws")
if d.get("sa_results"):
    sa = d["sa_results"][0]
    lo, hi = sa["ratio_ci95"]
    check(lo > 100,
          f"super-activation removal x{sa['ratio']:.0f} CI [{lo:.0f},{hi:.0f}] lower bound > 100")
    check(sa["positions_hit_per_forward"] > 0, "SA intervention actually fired")
print(f"probe info: corpus={d['eval_corpus']} {d['ppl_segments']}x{d['seq_len']} "
      f"dtype={d['dtype']} git={d['git_sha']}")
if fails:
    print(f"\nPROBE FAIL: {len(fails)} check(s) failed")
    sys.exit(1)
print("PROBE PASSED")
