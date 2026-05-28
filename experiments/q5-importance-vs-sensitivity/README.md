# Q5 — Importance × quantization sensitivity

> Of the components identified as MT-critical (Q2, Q3), which carry numerically sensitive information, and which carry information that is robust to perturbation?

This is the bridge to phase two.

## Methods

- **Weight-perturbation studies.** Add Gaussian noise of varying magnitude (e.g., σ/‖W‖ ∈ {1e−3, 1e−2, 5e−2}) to MT-critical vs MT-irrelevant components; measure the COMET / chrF++ drop on a held-out MT set.
- **Inverse-Hessian-style sensitivity** on MT calibration data — the diag(H⁻¹) metric that SparseGPT / LeanQuant use, but computed on MT pairs rather than C4.
- **Activation magnitude analysis** on MT-critical components (per AWQ; outliers in activations correlate with quantization sensitivity).

## Models covered

Aya at minimum (continuity with prior paper). The other two if compute permits — multi-model loading is the constraint here, see `PHASE1-PLAN.md` risk register.

## Embedded learning

- Why some weights tolerate quantization and others don't.
- What GPTQ / AWQ / LeanQuant actually measure when they "protect" weights — the inverse-Hessian-diagonal lens vs the activation-magnitude lens are *different* sensitivity definitions.
- The **distinction between component importance and component quantization sensitivity** — these are not the same thing. A head can be MT-critical and quantization-robust, or MT-irrelevant and quantization-fragile. This question is the one that justifies phase two over a generic "quantize the unimportant parts" approach.

## Expected artifacts

- `results/aya/q5/importance_vs_sensitivity.csv` — per-component (importance, sensitivity) pairs for the three sensitivity definitions.
- Updated `docs/phase2-hypotheses.md` with phase-one evidence supporting or killing each candidate.

## Satisfied when

A per-component (importance × sensitivity) map exists for at least Aya, and `docs/phase2-hypotheses.md` has been updated from a seed doc into a hypothesis-shaped writeup the phase-two design draws from.

## Tests

A new method module may be needed for the perturbation harness; tests for it land in `tests/` mirroring its source location, with both CPU shape/math invariants (e.g., zero-noise yields zero metric drop) and a GPU end-to-end test on a real cs→de batch.

## Working notes

See `notes.md`.
