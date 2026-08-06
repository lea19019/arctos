# Phase 2 — candidate directions (seed)

This is a working list of compression-method directions that could become phase two. **None is committed.** Selection happens once Q5 closes; new candidates may emerge from phase-one findings.

For each candidate: what supports it, what kills it, baselines, kernel/deployment risk.

## Candidate A — Task-conditional layer-wise mixed-precision

Use phase-one's per-layer MT-importance findings (from Q1, Q3, Q5) to drive layer-wise bit-width assignment: MT-critical layers stay high-precision; MT-irrelevant layers go lower. Novelty is the *signal* (interpretability-derived, MT-specific), not the *mechanism* (layer-wise mixed precision is well-established and kernel-friendly).

- **Supported by.** Q3 finding that MT-critical work concentrates in identifiable layers; Q5 finding that MT-irrelevant layers are more quantization-tolerant than MT-critical ones.
- **Killed by.** Q5 finding that the MT-critical/MT-irrelevant split does not predict quantization sensitivity (importance and sensitivity decouple); Q4 V0 finding that depth profiles are noise within a single language pair.
- **Baselines.** Uniform 4-bit GPTQ; uniform 4-bit LeanQuant (LeanAya); uniform 4-bit AWQ.
- **Kernel risk.** Low — Marlin / LUT-GEMM support layer-wise mixed precision natively.

## Candidate B — Component-level mixed-precision (head / MLP within layer)

More aggressive A: bit-widths assigned per attention head and per MLP, not just per layer. Bigger potential Pareto win, much harder kernel story.

- **Supported by.** Q2 finding that MT-criticality is concentrated in a small number of heads; Q3 finding that MLP MT-importance varies sharply within a layer.
- **Killed by.** No first-class kernel support for per-head mixed precision; Q5 finding that per-head sensitivity is uncorrelated with per-head importance.
- **Baselines.** Same as A; plus Candidate A as an internal baseline.
- **Kernel risk.** High — likely a stretch goal or future work.

## Candidate C — Task-specific calibration data for existing quantizers

Run GPTQ / AWQ / LeanQuant with MT parallel calibration data (or activations from MT-critical components per Q2/Q3) instead of generic C4. Orthogonal to A and B; lower-risk; addresses the WMT25 calibration gap directly (`compression/docs/annotated_bibliography.md` §1D).

- **Supported by.** Any Q1–Q3 finding that MT-conditional signal differs sharply from C4 signal; the existing literature gap (no surveyed paper ablates MT vs generic calibration for layer pruning).
- **Killed by.** Q5 finding that quantization sensitivity is data-agnostic — i.e., calibration choice doesn't move the per-weight grid much.
- **Baselines.** Same quantizer with generic C4 calibration.
- **Kernel risk.** Zero — same kernels as the underlying quantizer.

## Candidate D — Depth-profile-driven, model-agnostic prior

If Q4 lands on V2/V3 (a depth signature that generalizes across the three models), formulate bit-allocation or pruning candidacy as a function of depth fraction, validated across all three. Strongest claim; cleanest fallback to Candidate A if generalization fails.

- **Supported by.** Q4 V2 (characteristic depth signature) or V3 (depth fraction sufficient for compression decisions).
- **Killed by.** Q4 V1 (only trivial endpoints generalize) or V0 (depth profile differs across models / language pairs); the prior paper's existing within-Aya cs→de vs en→es divergence already cuts against this.
- **Baselines.** Candidate A (per-model interpretability-derived); uniform 4-bit baselines.
- **Kernel risk.** Same as Candidate A.

## Updating this document

When a question closes (`compression/docs/qN.md` is committed), revisit each candidate and add an evidence line: which finding moved which candidate's prospects. By the time Q5 closes, this document either points clearly at one candidate or explicitly lists why none of A–D survived and what the new direction will be.
