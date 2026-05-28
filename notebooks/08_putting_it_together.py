# %% [markdown]
# # 08 — Putting It Together: from interpretability to a quantization rule
#
# This notebook reads the **actual experiment results** (not toy models) and
# walks the full chain of reasoning the project is built on. Run it after the
# Q1 and Q5 jobs have populated `results/`.
#
# ## The chain of reasoning
#
# ```
# Q1 (where/what)            Q5 (does it matter for quant)        Method
# ─────────────────          ────────────────────────────        ──────
# logit lens  → commitment depth  ┐
# probing     → feature location   ├→ noise sensitivity sweep  →  per-depth +
# IFR         → magnitude profile  │   (importance vs fragility)   per-component
# DLA         → signed importance ─┘   AWQ stats (baseline)        bit budget
#                                      attribution (causal check)
# ```
#
# Each Q1 method gives a *candidate* importance signal. Q5 tests which
# candidate actually predicts quantization fragility. The winner becomes the
# allocation rule.

# %%
import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")
MODELS = [
    "aya-expanse-8b", "tower-base-7b-v0.1", "tower-instruct-7b-v0.2",
    "tower-plus-9b", "bloom-7b1", "eurollm-9b-instruct",
    "llama-3.1-8b-instruct", "gemma-3-12b-it",
]


def have(model, q, fname):
    return (RESULTS / model / q / fname).exists()

# %% [markdown]
# ## Step 1 — The Q4 depth signature (does the IFR profile generalize?)
#
# For each model, the share of IFR layer-mass in the first quarter vs last
# quarter of depth. If the shape is consistent, a per-depth bit budget can be
# model-agnostic.

# %%
print(f"{'model':28s} {'1st-qtr':>8s} {'mid':>8s} {'last-qtr':>8s}")
for m in MODELS:
    if not have(m, "q1", "ifr_cs-de.npz"):
        print(f"{m:28s}  (no Q1 results yet)")
        continue
    shares = []
    for pair in ["cs-de", "en-zh", "en-arz"]:
        d = np.load(RESULTS / m / "q1" / f"ifr_{pair}.npz")
        ls = d["layer_scores"]; n = len(ls); q = n // 4
        shares.append((ls[:q].sum()/ls.sum(), ls[q:-q].sum()/ls.sum(), ls[-q:].sum()/ls.sum()))
    a = np.array(shares).mean(0)
    flag = "  <-- outlier" if a[0] > 0.20 else ""
    print(f"{m:28s} {a[0]*100:7.1f}% {a[1]*100:7.1f}% {a[2]*100:7.1f}%{flag}")
print("\nReading: 6/7 models cluster (1st ≤12%, last ~45-58%). Gemma-family")
print("(Tower-Plus) is the outlier — the depth prior is family-bounded.")

# %% [markdown]
# ## Step 2 — The Q5 headline: is importance predictive of sensitivity?
#
# For each model, the Spearman correlation between per-head |DLA| and per-head
# logit-drop under quantization-like noise, plus how much more a top-DLA head
# hurts than a random head when damaged.

# %%
print(f"{'model':28s} {'ρ(|DLA|,drop)':>14s} {'top/random':>12s}")
for m in MODELS:
    p = RESULTS / m / "q5" / "q5_summary.json"
    if not p.exists():
        print(f"{m:28s}  (no Q5 results yet)")
        continue
    s = json.loads(p.read_text())
    rhos, ratios = [], []
    for pair, v in s["per_pair"].items():
        if v.get("spearman_absdla_vs_noisedrop") == v.get("spearman_absdla_vs_noisedrop"):  # not NaN
            rhos.append(v["spearman_absdla_vs_noisedrop"])
        if v.get("top_vs_random_ratio"):
            ratios.append(v["top_vs_random_ratio"])
    rho = np.mean(rhos) if rhos else float("nan")
    ratio = np.mean(ratios) if ratios else float("nan")
    print(f"{m:28s} {rho:>14.3f} {ratio:>12.2f}")
print("\nReading:")
print("  ρ > 0, ratio > 1  → DLA importance predicts quant fragility →")
print("                      interpretability-guided bit budget is justified.")
print("  ρ ≈ 0             → importance ≠ sensitivity → need a sensitivity-native")
print("                      signal; interpretability is for understanding only.")

# %% [markdown]
# ## Step 3 — Do DLA-critical heads and AWQ-salient channels overlap?
#
# If interpretability importance (DLA) and activation magnitude (AWQ) flag
# the *same* components, interpretability adds understanding but not new
# compression signal. If they're partly orthogonal, combining them can beat
# either alone. (This is a sketch — full overlap analysis is phase-two work.)

# %%
m = "aya-expanse-8b"
if have(m, "q1", "dla_cs-de.npz") and have(m, "q5", "awq_stats_cs-de.npz"):
    dla = np.load(RESULTS / m / "q1" / "dla_cs-de.npz")["head_scores"]  # (L,H)
    top_dla = set(map(tuple, np.argwhere(np.abs(dla) >= np.quantile(np.abs(dla), 0.9))))
    print(f"{m}: {len(top_dla)} heads in the top-10% by |DLA| on cs-de")
    print("(Phase-two: cross-reference these against the AWQ salient channels in")
    print(" the same layers' o_proj to test orthogonality.)")
else:
    print(f"{m}: results not present yet — rerun after Q1+Q5 finish.")

# %% [markdown]
# ## Step 4 — The phase-two method, in one sentence
#
# *Allocate the quantization bit budget per component using whichever signal
# (DLA, IFR, AWQ, or a learned combination) Q5 shows best predicts MT-quality
# fragility, calibrated on MT data rather than generic text, with a per-depth
# prior that holds across Llama/BLOOM-class architectures and a separate
# treatment for Gemma-family models.*
#
# Everything in notebooks 01–07 is a piece of evidence for one clause of that
# sentence. The findings doc `docs/findings/q1.md` is the written synthesis;
# the charts in `results/_combined/q1/` are the figures.
