# Q2 — MT-critical attention heads

> Which heads are MT-critical, and what do they do?

## Methods

- **Head-level activation patching** on (clean, corrupt) MT prompt pairs. Clean = source the model translates correctly; corrupt = source designed so the gold target differs lexically (see `src/data/clean_corrupt.py`).
- **Head ablation** (zero out the head's contribution) as a coarser cross-check on top-ranked heads.
- **Attention-pattern visualization** of top heads to characterize their behavior — source-attender, target-predictor, language-router, induction-like, etc.

Validation: patching × ablation on top-K heads must broadly agree; if they don't, that disagreement is reported.

## Models covered

Aya, omt-llama, Tower.

## Embedded learning

- Q/K/V derivation from scratch — what each tensor is, why softmax temperature matters, what makes attention different from dense routing.
- Multi-head structure as a width-axis decomposition; what gets concatenated, what gets shared.
- Attention-head taxonomies: induction heads (Olsson et al. 2022), name-mover / negative-name-mover (Wang et al. 2023, IOI), copy-suppression (McDougall et al. 2023). Whether MT-critical heads fit any known taxonomy is an open question for this project.

## Expected artifacts

- `results/{model}/q2/head_ranking.csv` — per-head patching effect size with metric.
- `results/{model}/q2/top_heads_attention.png` — attention pattern plots for top ~10 heads.
- `docs/findings/q2.md` — informal characterization of the top heads per model + cross-model comparison.

## Satisfied when

For each model: a ranked list with informal characterizations for the top heads, and an explanation grounded in attention patterns of why patching each top head breaks the translation. Where multiple corrupt-prompt strategies disagree on the ranking, that instability is reported.

## Tests

`tests/interp/test_activation_patching.py` — CPU invariants on self-patch and full-replacement; GPU sign-convention check on a real cs→de pair.

## Working notes

See `notes.md`.
