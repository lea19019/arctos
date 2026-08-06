# Reading guide — everything we've done, in order

A path to read and *understand* the whole project: the experiments, the results,
and the papers. Work top to bottom; each item says why it matters.

## 0. Orientation (15 min)
- [`README.md`](../../README.md) — the map.
- [`docs/archive/project_summary.md`](../../docs/archive/project_summary.md) + [`docs/archive/phase1_plan.md`](../../docs/archive/phase1_plan.md) — the thesis spine and original plan.

## 1. Phase one — how translation works inside an LLM (the foundation)
- [`compression/report/arctos-translation-report.pdf`](../report/arctos-translation-report.pdf) — **the full phase-one paper.** Read this first for the science.
- Per-question detail: [`compression/docs/q1_language_emergence.md`](q1_language_emergence.md) (when/where language emerges),
  [`compression/docs/q4_architecture_comparison.md`](q4_architecture_comparison.md) (does the depth signature generalize),
  [`compression/docs/q5_importance_vs_sensitivity.md`](q5_importance_vs_sensitivity.md) (**the pivotal negative:** component importance ⟂ quantization sensitivity).
- Methods, with runnable tutorials: [`notebooks/`](../../notebooks/) (00 overview → 08 synthesis), code in [`compression/src/interp/`](../src/interp/).
- **One-line takeaway:** translation = encode source (early) → think in a shared
  language-neutral pivot (middle) → convert to the target language (late). The
  signature generalizes except Gemma. *Importance does not predict where
  precision matters* — this is what phase two had to deal with.

## 2. Phase two — compression grounded in phase one
Read in this order:
1. [`compression/docs/compression_primer.md`](compression_primer.md) — the **find / keep / shrink / prune** framework + the reading list for super weights, AWQ, Hessian, outliers.
2. [`compression/docs/phase2_synthesis.md`](phase2_synthesis.md) — **the honest consolidated read.** What holds, what doesn't. Start here for conclusions.
3. [`compression/docs/phase2_results.md`](phase2_results.md) — the cross-model tables (chrF++ / XCOMET-XL), gap-to-baseline.
4. [`compression/docs/phase2_method_primer.md`](phase2_method_primer.md) — the proposed method + the literature-gap/novelty map.
5. [`compression/docs/phase2_novel_direction.md`](phase2_novel_direction.md) — the pipeline-aware ("protect endpoints, crush middle") idea — **read knowing it became a negative result** (the experiment is in `q6`).
6. [`compression/docs/q6_compression.md`](q6_compression.md) — the chrF++ sweep writeup.
- Code to actually run it: [`compression/experiments/q6-compression/`](../experiments/q6-compression/) (`README.md` + `notes.md` changelog), [`compression/src/interp/compress.py`](../src/interp/compress.py) (RTN/AWQ/GPTQ/ternary/binary/prune/mixed-precision), collectors in [`compression/scripts/`](../scripts/).

## 3. The literature (papers to read)
- [`compression/docs/annotated_bibliography.md`](annotated_bibliography.md) — **annotated bibliography** (pruning, quantization, recovery) + **3 addenda** with the phase-two papers and **3 deep-research syntheses** (verified claims + votes). The raw agent reports: [`compression/docs/deep_research_raw/`](deep_research_raw/).
- The must-reads it points to (start with these):
  - **Super weights:** Yu et al. 2024 (arXiv:2411.07191); massive activations Sun et al. (arXiv:2402.17762).
  - **Quantizers:** GPTQ (2210.17323), AWQ (2306.00978), SqueezeLLM (2306.07629), LeanQuant (2407.10032), GPTVQ (2402.15319); rotation QuaRot/SpinQuant (2404.00456 / 2405.16406).
  - **MT/multilingual:** Uneven-Impact-of-PTQ-in-MT (2508.20893), Calibrating-Beyond-English (2601.18306), Marchisio "How does quantization affect multilingual" (2407.03211).
  - **Mechanism:** Language-Specific Neurons / LAPE (2402.16438); "Do Llamas Work in English?" (Wendler 2024); calibration-data impact (2311.09755).

## 4. What's next
- [`docs/plans/roadmap.md`](../../docs/plans/roadmap.md) — the ambitious 6-month program (the multi-dimensional sweet-spot study + distillation + pruning + speech↔text + skills plan).
- [`docs/archive/session_handoff_2026_06_03.md`](../../docs/archive/session_handoff_2026_06_03.md) — live state for resuming.

## How to reproduce / poke at results yourself
```bash
uv run pytest -m cpu                       # method unit tests, tiny CPU model
python scripts/q6gem_collect.py            # phase-two gem tables (if results/ present)
python scripts/q6gem_collect.py --subdir q6extreme   # the sub-2-bit cliff
bash scripts/q6_status.sh                   # live SLURM + completion status
```
(`compression/results/` is gitignored — regenerate via the SLURM runners on an A100, or read
the numbers already transcribed into `phase2_results.md` / `phase2_synthesis.md`.)
