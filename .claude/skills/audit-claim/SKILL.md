---
name: audit-claim
description: Adversarially check a written claim against the raw data before it lands in a findings doc, the registry, or a paper. Use before publishing any result, or when reviewing existing claims for drift from their evidence.
---

# Audit a claim against its evidence

A 2026-08 audit of this repo found the measurements sound and several *claims
about them* wrong. The failures were never bad math — they were claims drifting
from their evidence. This skill is that audit, run before publication instead of
months after.

**Adopt a refuting posture.** The job is not to confirm the sentence reads well.
It is to find the reading under which it is false. Default to "not supported"
when uncertain.

## For each claim, answer all eight — in writing

1. **Baseline named?** Is the thing it is compared against *in the sentence*?
   "MT-calibrated GPTQ beats generic-calibrated GPTQ" is supported.
   "MT-calibrated GPTQ recovers the 3-bit cliff" is not — against plain RTN,
   Aya gets much worse.
2. **Baseline measured?** Open the results directory and confirm the baseline
   number exists. "Worse than not quantizing" reached three documents while FP16
   was never scored. *A baseline you did not run is not a baseline.*
3. **Coverage stated?** Cross-model / cross-layer / cross-language claims carry
   their denominator inline: "(7 of 8; Gemma not run)". Two headline results here
   silently rested on 6 of 8 and 7 of 8 — each missing the case most likely to
   complicate them. Check *which* case is missing, not just how many.
4. **Metric substitution flagged at the point of use?** Open the runner and
   confirm the column holds the quantity its header claims. NLLB's "IFR" is a
   different quantity sharing a column with real IFR values. A note in a
   docstring does not count.
5. **Dropped data points named?** If the denominator is not the full design,
   the doc says which cell went missing and why. A published n=17-of-18 statistic
   here has no record of which one.
6. **Config = what ran?** Read the manifest, not the config file. Q5's config
   specifies COMET / n=200 / σ∈{1e-3,1e-2,5e-2}; the run used chrF++ / n=20 /
   σ=0.1. **This is the check most likely to fail and the one people skip.**
7. **Failures explained from the log?** Any "transient" / "flaky" / "one-off"
   attribution needs the traceback that supports it.
8. **Hedged correctly?** "Our hypothesis", not "the mechanism". No `proves`,
   `confirms`, `establishes` unless an intervention was run *and reported*.

## Then check the rigor floor

- Effect reported as `effect [95% CI], n_seeds=N, family=M` — not a bare mean.
- Seeds, not checkpoints/layers/heads, are the unit of independence. If only
  weight init was randomised, effective n is ≈2 regardless of seed count.
- Multiplicity corrected on any scan, family size stated in the caption.
- No bare "no effect" — bounded with an interval.
- Every measure beats its random-init null. Probes, saliency maps and SAEs all
  look interpretable on randomly initialised networks.

## Output

Per claim: **supported** / **overstated** / **unsupported**, with the specific
number and file path that decides it. For anything not "supported", write the
corrected sentence.

If a claim is already published somewhere, fixing it here is not enough — run
`/close-experiment` step 2 and back-annotate every other copy.
