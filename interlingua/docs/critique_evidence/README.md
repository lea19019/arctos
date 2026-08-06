# Evidence code for `../program_critique.md`

Two of the audit's findings are **measurements, not literature**. This is the code
behind them, moved out of `/tmp` so the claims remain checkable.

**Status: exploratory.** By this repo's own standard (`CLAUDE.md`: "n=24–32 is
exploratory. Nothing at that scale is a paper claim") none of this is publishable
as-is. It is sufficient to decide whether to commit fifteen training runs, which is
what it was written for.

## The Δt sharpness artifact — `program_critique.md` row A3

| File | What it does |
|---|---|
| `premise7_power.py` | Power curve for Δt. Two logistic trajectories on log₁₀(step), sampled at the 60 log-spaced checkpoints, per-seed jitter decomposed into run-level + idiosyncratic components with correlation ρ, exact single-changepoint L2 split, percentile bootstrap over seeds. Sweeps true Δ × jitter × n_seeds. |
| `premise7_part2.py`, `premise7_part3.py` | The width-confound experiment. Sets the **true lag to zero** and varies only the transition widths of the two curves. |
| `premise7_bootstrap_coverage.py` | Actual coverage of a nominal-95% percentile bootstrap CI at n = 5, 10, 20, 50, under normal / t₃ / lognormal seed distributions. |

Headline results reproduced by these: with **true lag identically zero**, the
changepoint detector returns Δt = +0.084 → +0.262 log₁₀ (1.21×–1.83×) at a **93–100%**
false-positive rate, with the sign H1 predicts. A parametric sigmoid midpoint is
unbiased in the same cells (|Δt| ≤ 0.007). Bootstrap coverage at n=5 is **83.4%**
(72.1% skewed), not 95%.

Assumptions that drive the numbers and are **guesses**: transition width w = 0.30 log₁₀,
measurement noise σ_obs = 0.03, and seed jitter σ_seed imported from modular-arithmetic
grokking (arXiv:2603.25009, CV 0.444) because **no published estimate exists for language
pretraining at any scale**. The *ordering* of conclusions is robust across the swept grid;
the exact minimum-detectable-effect values are not.

`ruptures` was not installed; the exact single-changepoint L2 split is implemented
directly and is what PELT/binary-segmentation reduces to at one breakpoint.

## The cross-lingual JSD contrast — rows A8, A8b

| File | What it does |
|---|---|
| `jsd_probe2.py` | Runs the proposal's own §4.4 contrast on **Qwen3-1.7B**: 12 SVA-style prefixes per condition, next-token distributions at the agreement target, averaged per condition as Jian & Manning define P_v(x), JSD in bits, bootstrap over items (2000 resamples). |
| `decomp.py` | Token-level decomposition of the surviving EN–FR signal. |

Headline results: between-class minus within-class = **EN–FR +0.0112 [+0.0050, +0.0247];
EN–TR −0.0049 [−0.0176, +0.0029]; FR–TR −0.0064 [−0.0151, +0.0008]** — wrong sign for both
Turkish pairs, which carry H4. Cross-lingual pedestal 0.64–0.73 bits, so the surviving
EN–FR effect is ~1.7% of it. Decomposition: top 5 tokens carry 52% of the signal, and they
are `,` ` is` ` has` ` are` `:` — **no French token among them**.

`jsd_probe2.py` sets `HF_HUB_OFFLINE=1` and expects Qwen3-1.7B already cached. Qwen3-1.7B
is ~47× the Tier-1 target and instruction-tuned, so it is a **generous upper bound** — if the
measure is null there, it will not find signal at 36M.

**Known limits, stated by the agent that ran it:** single model, single seed, n=12 items per
condition, bootstrap over items only. The sign pattern replicated on a disjoint item set. The
honest weak point is that it measured an **endpoint** and is inferring about a **trajectory**.

## Not included

The tokenizer test behind row A8b (Turkish agreement is never at the next-token position in
mBERT / XLM-R / Qwen3) was run inline and not saved. It is a few lines and takes an hour to
redo; row A8b states the result per verb pair per tokenizer.
