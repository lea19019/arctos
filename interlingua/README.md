# interlingua — does the shared meaning space *grok*?

The direction being taken up next. Where [`../compression/`](../compression/)
asked *what* the shared meaning space looks like in finished production models,
this track asks **when and how it forms during training** — and whether the
"emergence" everyone reports is real phase structure or a measurement artifact.

**Currently docs only.** No code yet; the plan below is what gets built.

## The direction: Tier 1

The umbrella program is [`docs/does_the_interlingua_grok_ringger_2026.pdf`](docs/does_the_interlingua_grok_ringger_2026.pdf)
— *Does the Interlingua Grok? Tracking the Emergence of Cross-Lingual Alignment
via Mechanistic Interpretability Progress Measures* (PI: Eric Ringger, Matrix
Lab, 23 pp). **[`docs/tier1_plan.md`](docs/tier1_plan.md) is the working plan for
Tier 1 of it, and Tier 1 is what's being pursued.** Tiers 2 and 3 are out of
scope.

Tier 1 trains small decoder-only models from scratch under controlled
multilingual conditions and tracks mechanistic and behavioral measures across
training, to test whether mechanistic progress rises *before* cross-lingual
transfer accuracy jumps.

### The move that makes it worth doing

The program's H1 — mechanistic measures lead behavioral ones — is **not safe to
report as stated**, and the plan says so. Schaeffer et al. (NeurIPS 2023) showed
that sharp capability "jumps" are frequently artifacts of *discontinuous
metrics* applied to smoothly improving per-token error. Zero-shot transfer
accuracy is exactly such a metric, so a lag between a smooth mechanistic curve
and a jumpy accuracy curve is **guaranteed by construction** — it would appear
even if nothing interesting happened mechanistically.

So the experiment becomes: *does the lag survive when the behavioral axis is
measured continuously?* Same training runs, no extra cost, and no empty outcome:

| Outcome | Reading |
|---|---|
| Lag survives under continuous metrics | Real phase structure — H1 supported, rigorously enough to build on |
| Lag vanishes | "Multilingual emergence is a mirage" — arguably the more interesting result |
| Lag survives for some constructions/pairs | Local asynchronous grokking — probably the true answer |

That adjudication, plus the statistical protocol around it, is the defensible
novelty. The descriptive two-stage finding is already scooped; *being right
about it* is not.

### Shape of the experiment

- **Model:** 6-layer decoder-only, d_model 512, ~36M params. Cheap — 3–6 GPU-hours per run.
- **Languages:** English, French, Turkish. EN–FR close, EN–TR distant; all three have overt subject-verb agreement, so the JSD construction is identically operationalized.
- **Runs:** 3 configs × 5 seeds = 15. A1 (EN/FR, 2B tok) vs A2 (add TR, fixed total) vs A3 (add TR, fixed per-language) — A1↔A3 is what separates "more languages" from "less data each."
- **Checkpoints:** ~60 **log-spaced**, not uniform — shared concept space forms inside the first 10% of tokens, which uniform spacing would miss entirely. ~130 GB of checkpoints.
- **Measures, in gating order:** behavioral (each phenomenon measured twice, continuous *and* discontinuous — the pair *is* the mirage test) → JSD divergence → representation geometry (debiased CKA + mutual-nearest-neighbor) → crosscoders, conditional on signal.
- **Analysis:** changepoint detection on log(step), bootstrapped over seeds; the claim is that the CI on Δt excludes zero, not that the curves look different. Pre-registered before any trajectory is inspected.

### What to build first

Week 1 is a **positive control**: recover the known Nanda modular-addition
grokking lag through the project's own changepoint pipeline. If the pipeline
can't find a lag that is definitely there, nothing it later reports is
trustworthy. Cheap, and it makes every subsequent number defensible.

Four open questions are parked for the PI at the end of the plan — decoder vs.
encoder, framing, the language set, and whether crosscoders are expected.

## The parked alternative

[`docs/ms_proposal_v2_last_mile.md`](docs/ms_proposal_v2_last_mile.md) —
*The Last Mile of the Shared Meaning Space* — is a **different** project:
diagnosing and repairing decode-out failure for African low-resource languages
on Gemma 4, with a deployed quality-estimation service for ToAll.
[`docs/ms_proposal_v1_shared_meaning_space.md`](docs/ms_proposal_v1_shared_meaning_space.md)
is its earlier draft (frozen encoders, geometric probing).

It is not the direction being taken, but it is kept because it is close to
shippable and it is the applied counterpart to Tier 1: that work asks how the
shared space *forms* in controlled small models, this one asks whether the
*formed* space in a production model can be diagnosed and repaired. It also
connects directly to [`../speech-translation/`](../speech-translation/) — same
NLLB models, same dubbing platform.

## Literature

[`papers/`](papers/) holds 47 papers plus CS 601R course materials. The PDFs are
gitignored (~191 MB); [`papers/README.md`](papers/README.md) indexes them.

The two committed documents there are the substance:

- [`papers/paper_summaries.md`](papers/paper_summaries.md) — per-paper summaries (problem / method / findings / relevance).
- [`papers/literature_sweep_2026_07.md`](papers/literature_sweep_2026_07.md) — positioning report on the 2026 state of the field: what replaced the strong interlingua hypothesis, and where the Platonic Representation Hypothesis is currently under attack.

## Relation to the rest of the repo

`compression/` phase one found that translation is depth-staged — a
language-neutral middle, with target-language emission only in the last quarter.
That is an interlingua claim about a *trained* model. This track asks when that
structure appears during training, and whether the appearance is a phase
transition at all.

Note the tension worth keeping honest: `compression/` Q5 found component
importance uncorrelated with quantization sensitivity — a reminder that a real,
measurable structure need not be the structure that governs the behavior you
care about. Tier 1's mirage test is the same skepticism applied to training
dynamics.
