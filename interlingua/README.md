# interlingua — does the shared meaning space *grok*?

The direction being taken up next. Where [`../compression/`](../compression/)
asked *what* the shared meaning space looks like in finished production models,
this track asks **when and how it forms during training** — and whether the
"emergence" everyone reports is real phase structure or a measurement artifact.

**Currently docs only.** No code yet; the plan below is what gets built.

## The direction: Tier 1

> ⚠️ **Two 2026-08-06 audits sit above this plan and partly supersede it.**
> [`docs/prior_work_map.md`](docs/prior_work_map.md) — what is already occupied, claim by
> claim, with citations. Its verdict: **the surviving-novelty claim in `tier1_plan.md` §7
> does not survive**, and Tier 1 as written is not worth doing. What *is* open is listed in
> its §10. [`docs/program_critique.md`](docs/program_critique.md) — an independent audit of
> the program's *premises* by nine agents (eight adversarial, one steelman), reaching a
> compatible verdict by a different route: *salvageable, but a much smaller project than
> proposed*. Two of its findings are measurements, not literature, with code in
> [`docs/critique_evidence/`](docs/critique_evidence/): the **Δt statistic manufactures a
> positive lag of H1's predicted sign at 93–100% from zero true lag**, and the cross-lingual
> JSD contrast on a real model returns the **wrong sign for both Turkish pairs**. Its
> recommended next action is a few-GPU-hour pilot confirming the behavioral axis moves at all
> in the chosen architecture, before committing the fifteen runs.
> **Read both before proposing anything in this track.**
>
> A third audit of the same date, [`docs/method_landscape.md`](docs/method_landscape.md),
> maps **methods and tooling** and takes no position on novelty or on whether to
> proceed: for each measure, what it costs at ~900 checkpoint-analyses, what it
> requires, which architecture families it supports, whether it has a **calibrated
> null**, and whether its implementation is maintained. Its §8 tabulates thirteen
> claims across `tier1_plan.md`, `CLAUDE.md`, `research_standards.md` and
> `registry.md` that it contradicts; §9 is a ranked reading list; §10 is its own
> verification debt. Unfiltered candidate lists — ~130 papers surfaced but not
> opened — are in [`docs/method_landscape_candidates.md`](docs/method_landscape_candidates.md).
> It corroborates the premise audit independently on the two points where they
> overlap (the Δt pipeline, and the sigmoid-midpoint fix), and separately finds
> that **§3.1's case for dropping the encoder and encoder-decoder arms does not
> hold on either of its two stated reasons.**

The umbrella program is [`docs/does_the_interlingua_grok_ringger_2026.pdf`](docs/does_the_interlingua_grok_ringger_2026.pdf)
— *Does the Interlingua Grok? Tracking the Emergence of Cross-Lingual Alignment
via Mechanistic Interpretability Progress Measures* (PI: Eric Ringger, Matrix
Lab, 23 pp). **[`docs/tier1_plan.md`](docs/tier1_plan.md) is the working plan for
Tier 1 of it, and Tier 1 is what's being pursued.** Tiers 2 and 3 are out of
scope.

Tier 1 trains small models from scratch under controlled multilingual conditions
and tracks mechanistic and behavioral measures across training, to test whether
mechanistic progress rises *before* cross-lingual transfer accuracy jumps.

> **The architecture is an open question, not a decision.** The proposal
> specifies **three arms** at 6L/512d/8h — encoder-only mBERT-like, encoder-only
> XLM-R-like, and encoder-decoder NLLB-like — and **H3a is the comparison
> between two of them**, which makes the architecture a hypothesis under test
> rather than an implementation detail. `tier1_plan.md` §3.1 argues for
> narrowing Tier 1 to decoder-only, but on *tooling* grounds (TransformerLens
> and circuit-tracer are decoder-first; Jian & Manning's JSD measures are
> defined over next-token distributions), and it says outright that **this is a
> PI decision**. §8 lists it as open question #1. Until Ringger answers, nothing
> here should assume a decoder.

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

That adjudication, plus the statistical protocol around it, ~~is the defensible
novelty~~ — **refuted 2026-08-06, see [`docs/prior_work_map.md`](docs/prior_work_map.md).**
The adjudication is largely occupied (Körner et al., EACL 2026, causally; and Du et al.,
NeurIPS 2024, already ran the metric triple across checkpoints on non-English benchmarks,
where emergence *survived*). Both gaps behind the statistical protocol are false as stated.
The descriptive two-stage finding is indeed already scooped — but *being right about it* is
no longer the unclaimed part either. What remains open is in `prior_work_map.md` §10.

### Shape of the experiment

- **Model:** 6 layers, d_model 512, 8 heads, ~36M params. Cheap — 3–6 GPU-hours per run. **Architecture arm unresolved** (see above); the size and shape are fixed by the proposal, the encoder/decoder question is not.
- **Languages:** English, French, Turkish. EN–FR close, EN–TR distant; all three have overt subject-verb agreement, so the JSD construction is identically operationalized.
- **Runs:** 3 configs × 5 seeds = 15. A1 (EN/FR, 2B tok) vs A2 (add TR, fixed total) vs A3 (add TR, fixed per-language) — A1↔A3 is what separates "more languages" from "less data each."
- **Checkpoints:** ~60 **log-spaced**, not uniform — shared concept space forms *early* (Körner et al., EACL 2026; Leino & Tiedemann find PWCCA rising by ~5k steps), which uniform spacing would miss entirely. ~130 GB of checkpoints. *(The "first 10% of tokens" figure previously cited here was attributed to "Dumas et al." — wrong author, and the figure itself is unverified. See `prior_work_map.md` §0.)*
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

[`docs/method_landscape.md`](docs/method_landscape.md) — a map of what exists in
the method and tooling space (nine area surveys, compiled 2026-08-06).
Deliberately **not** a plan and not a recommendation: it chooses nothing and
ranks nothing. Every entry carries a verification flag —
`[verified]` / `[discovery]` / `[UNVERIFIED]` — and nothing flagged short of
`[verified]` may enter a findings doc, the registry, or a paper without the
confirmation pass its flag implies. Where it contradicts `tier1_plan.md`, it
says so and sources the contradiction.

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
