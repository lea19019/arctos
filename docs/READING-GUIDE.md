# Reading guide — everything we've done, in order

A path to read and *understand* the whole project: the experiments, the results,
and the papers. Work top to bottom; each item says why it matters.

## 0. Orientation (15 min)
- [`README.md`](../README.md) — the map.
- [`docs/project-summary.md`](project-summary.md) + [`PHASE1-PLAN.md`](../PHASE1-PLAN.md) — the thesis spine and original plan.

## 1. Phase one — how translation works inside an LLM (the foundation)
- [`report/arctos-translation-report.pdf`](../report/arctos-translation-report.pdf) — **the full phase-one paper.** Read this first for the science.
- Per-question detail: [`docs/findings/q1.md`](findings/q1.md) (when/where language emerges),
  [`docs/findings/architecture-comparison.md`](findings/architecture-comparison.md) (does the depth signature generalize),
  [`docs/findings/q5.md`](findings/q5.md) (**the pivotal negative:** component importance ⟂ quantization sensitivity).
- Methods, with runnable tutorials: [`notebooks/`](../notebooks/) (00 overview → 08 synthesis), code in [`src/interp/`](../src/interp/).
- **One-line takeaway:** translation = encode source (early) → think in a shared
  language-neutral pivot (middle) → convert to the target language (late). The
  signature generalizes except Gemma. *Importance does not predict where
  precision matters* — this is what phase two had to deal with.

## 2. Phase two — compression grounded in phase one
Read in this order:
1. [`docs/findings/compression-primer.md`](findings/compression-primer.md) — the **find / keep / shrink / prune** framework + the reading list for super weights, AWQ, Hessian, outliers.
2. [`docs/findings/phase2-synthesis.md`](findings/phase2-synthesis.md) — **the honest consolidated read.** What holds, what doesn't. Start here for conclusions.
3. [`docs/findings/phase2-results.md`](findings/phase2-results.md) — the cross-model tables (chrF++ / XCOMET-XL), gap-to-baseline.
4. [`docs/findings/phase2-method-primer.md`](findings/phase2-method-primer.md) — the proposed method + the literature-gap/novelty map.
5. [`docs/findings/phase2-novel-direction.md`](findings/phase2-novel-direction.md) — the pipeline-aware ("protect endpoints, crush middle") idea — **read knowing it became a negative result** (the experiment is in `q6`).
6. [`docs/findings/q6.md`](findings/q6.md) — the chrF++ sweep writeup.
- Code to actually run it: [`experiments/q6-compression/`](../experiments/q6-compression/) (`README.md` + `notes.md` changelog), [`src/interp/compress.py`](../src/interp/compress.py) (RTN/AWQ/GPTQ/ternary/binary/prune/mixed-precision), collectors in [`scripts/`](../scripts/).

## 3. The literature (papers to read)
- [`docs/research.md`](research.md) — **annotated bibliography** (pruning, quantization, recovery) + **3 addenda** with the phase-two papers and **3 deep-research syntheses** (verified claims + votes). The raw agent reports: [`docs/findings/deep-research-raw/`](findings/deep-research-raw/).
- The must-reads it points to (start with these):
  - **Super weights:** Yu et al. 2024 (arXiv:2411.07191); massive activations Sun et al. (arXiv:2402.17762).
  - **Quantizers:** GPTQ (2210.17323), AWQ (2306.00978), SqueezeLLM (2306.07629), LeanQuant (2407.10032), GPTVQ (2402.15319); rotation QuaRot/SpinQuant (2404.00456 / 2405.16406).
  - **MT/multilingual:** Uneven-Impact-of-PTQ-in-MT (2508.20893), Calibrating-Beyond-English (2601.18306), Marchisio "How does quantization affect multilingual" (2407.03211).
  - **Mechanism:** Language-Specific Neurons / LAPE (2402.16438); "Do Llamas Work in English?" (Wendler 2024); calibration-data impact (2311.09755).

## 4. What's next
- [`docs/ROADMAP.md`](ROADMAP.md) — the ambitious 6-month program (the multi-dimensional sweet-spot study + distillation + pruning + speech↔text + skills plan).
- [`docs/SESSION-HANDOFF.md`](SESSION-HANDOFF.md) — live state for resuming.

## How to reproduce / poke at results yourself
```bash
uv run pytest -m cpu                       # method unit tests, tiny CPU model
python scripts/q6gem_collect.py            # phase-two gem tables (if results/ present)
python scripts/q6gem_collect.py --subdir q6extreme   # the sub-2-bit cliff
bash scripts/q6_status.sh                   # live SLURM + completion status
```
(`results/` is gitignored — regenerate via the SLURM runners on an A100, or read
the numbers already transcribed into `phase2-results.md` / `phase2-synthesis.md`.)
