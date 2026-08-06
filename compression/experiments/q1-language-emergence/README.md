# Q1 — Language emergence and target-language commitment

> Where does language identity get represented as a meaning-bearing feature, and where does target-language generation commit?

## Methods

- **Probing classifiers** for source-language ID and target-language ID across all layers (linear probes; report selectivity per Hewitt & Liang 2019).
- **Logit lens / tuned lens** for the layer at which target-language token mass dominates the next-token distribution at the boundary position.
- **IFR** for cross-checking the layer-flow ranking against the probing/lens story; report agreement and disagreement.

Validation: probing × logit lens × IFR all on the same prompts.

## Models covered

Aya Expanse 8B, omt-llama-8b, TowerInstruct-7B.

## Embedded learning

- Residual stream geometry — what a "layer" is, why probing is meaningful at all.
- Logit-lens math: final RMSNorm + lm_head applied to mid-layer hidden states; why this is exactly the model's "if I had to commit now, what would I say" view.
- "Language identity" as a probeable feature: a linearly-decodable direction, not a single neuron.
- Selectivity (probe accuracy − control accuracy), and why raw probe accuracy alone is misleading.

## Expected artifacts

- `results/{model}/q1/` — per-layer probe (accuracy, control, selectivity), per-layer logit-lens target-mass.
- Plots: per-model layer × probe accuracy; per-model layer × target-language mass.
- `compression/docs/q1_language_emergence.md` — synthesis.

## Satisfied when

For each model and language pair: a chart and a sentence describing (a) the layer at which source-language identity becomes linearly decodable, (b) the layer at which target-language token mass dominates under logit lens, (c) the relationship between (a) and (b). Disagreements between probing and logit lens are reported, not hidden.

## Tests

`tests/interp/test_logit_lens.py`, `tests/interp/test_probing.py`, `tests/interp/test_ifr.py` cover the methods. CPU-tier tests gate the API; GPU-tier tests in `test_logit_lens.py::test_target_language_emergence` and `test_probing.py::test_source_language_id_decodable` are this question's sanity checks on real models.

## Working notes

See `notes.md`.
