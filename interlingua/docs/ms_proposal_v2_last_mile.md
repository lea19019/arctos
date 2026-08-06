# MS Final Project Proposal (Draft v2)

## The Last Mile of the Shared Meaning Space: Causal Evidence, Speech Translation, and Deployed Quality Signals for African Low-Resource Languages

**Student:** Adrian Castillo · **Advisor:** [advisor] · **Industry collaboration:** ToAll (dubbing platform) · **Estimated effort:** 150+ hours, Fall 2026

*Supersedes `PROPOSAL_DRAFT.md` (v1). Same skeleton — map → signal → mechanism — with three changes: the interpretability instrument is modernized (cross-layer transcoders on Gemma 4 instead of geometric probing of frozen encoders), the mechanism phase is now a single falsifiable hypothesis instead of open-ended exploration, and the project trains real models (CLTs + a speech-translation fine-tune) rather than only frozen-encoder inference.*

---

## Motivation

Multilingual models encode meaning in a partially shared, language-neutral space: middle layers hold concepts, final layers render them into a specific language (Wendler et al. 2024; Semantic Hub Hypothesis, ICLR 2025). The 2025–26 mechanistic literature localizes where this breaks for non-English languages — tokenizer fragmentation, weak late-layer language-identity features, underdeveloped decoding features (Harrasse et al. 2025, arXiv:2511.10840) — and the Translation Barrier Hypothesis (Bafna et al. 2025) reframes low-resource MT failure as a **decode-out** problem: the model understands, but fails to render that understanding into fluent target-language text. None of this work touches African languages; no major lab has published African-language interpretability, period.

Meanwhile ToAll deploys lab-trained MT/TTS into African languages no one on the team can read, with **no automated quality signal**. And the lab holds a data asset almost no research group has: English church-talk audio with human translations of the scripts into dozens of languages — an N-way parallel speech+text corpus, utterance-alignable by ToAll's own script-matching pipeline.

## Central claim (falsifiable)

> **For African low-resource languages, translation failure in multilingual models is concentrated in the final decoding step, not in understanding — and can therefore be partially repaired by targeted intervention on late-layer language-decoding features, without per-language fine-tuning.**

If true: a mechanistic account plus a cheap, deployable fix, demonstrated on languages nobody has studied. If false — understanding itself degrades below some resource floor — that is a novel negative result: the Translation Barrier Hypothesis has a boundary, and we locate it. No empty-handed outcome.

## Research questions

- **RQ1 (map, scoped down from v1):** Across the lab's languages with human-translated parallel data, where do correct translation pairs sit close in the shared space? Calibrated metrics (permutation-null, multi-sample-size). Purpose: *select 6–8 target languages* spanning apparent understanding-intact vs. understanding-degraded regimes.
- **RQ2 (signal → tool, kept from v1):** Does source↔output embedding distance track true translation quality of the lab's NLLB models, per language, validated against our human references and benchmarked vs. AfriCOMET / BLASER 2.0? Deployed as a monitored QE service surfacing provenance-style badges in ToAll.
- **RQ3 (mechanism — the core):** On Gemma 4, using self-trained cross-layer transcoders: when translation into a target language fails, is the shared-space representation intact (decode-out failure) or degraded (understanding failure)? Causal test: strengthen/steer the late-layer language-decoding features and measure quality recovery against human references. Comparison: a full fine-tune on the same language — does fine-tuning mostly move the decode features (hypothesis confirmed mechanistically) or reshape middle layers too?
- **RQ-Stretch (modality):** Gemma 4 12B processes audio in the same residual stream as text. For English-audio → LRL-text translation (a task our corpus supervises directly), does failure localize to hearing (audio→concept) or to the same decode-out step? Does LRL speech inherit a *double* last mile (speech→text pivot per Sternberg et al. 2026, then text→LRL decode)?

## Instruments and models

| Role | Model | Why |
|---|---|---|
| Prototyping substrate | Gemma 3 (+ Gemma Scope 2) | Free pretrained SAEs/CLTs on every layer; debug the full method cheaply before spending on the real target |
| Primary substrate | **Gemma 4 12B** (unified, encoder-free, native audio) | Only open model where speech and text share one residual stream; no interp tooling exists for it yet — we train our own CLTs (first-mover; optional public release) |
| Scale option | Gemma 4 26B-MoE (~3.8B active) | Compute is available; MoE serves cheaply if the fine-tune becomes a provider candidate |
| Study object (quality ground truth) | Lab NLLB models | Used as-is; their per-language output quality is what RQ2 predicts and RQ3 explains |
| QE encoder | AfriE5 / SONAR (frozen) | v1 choice unchanged; benchmarked against AfriCOMET, BLASER 2.0 |

## Data

- **Church-talk corpus (lab):** English talk audio + human-translated scripts, N-way parallel across dozens of languages incl. African LRLs. Aligned at utterance level via ToAll's script matcher. Roles: RQ1/RQ2 evaluation references (cleaner than FLORES-200, whose African splits have documented errors), RQ3 intervention scoring, and **supervised training data for English-speech→LRL-text fine-tuning — a dataset with no public equivalent**.
- **WAXAL (Google Research Africa, 2026, CC-BY-4.0):** ~1,846 hrs ASR + ~565 hrs TTS across 27 African languages — fine-tuning volume to complement our corpus.

## Plan and deliverables

| Phase | Work | Deliverable | ~Hours |
|---|---|---|---|
| 0. Audit & de-risk | Encoder-coverage audit (from v1) **plus**: Gemma 4 audio sanity test on 5 target languages; church-corpus licensing cleared for training + commercial use; Gemma 4 license text verified; compute sizing | Go/no-go memo; coverage table | 15 |
| 1. Map | Batch embedding census over parallel data, calibrated convergence metrics | Language × encoder reliability map; **6–8 target languages selected** | 25 |
| 2. QE tool | Embedding-distance QE scored vs. our references, per-language calibration, deployed service + monitoring, badges via ToAll provenance UI. Side deliverable: **TMX import of the translated scripts into ToAll's translation memory** (~1 week, immediate product win) | Deployed, monitored QE service; TM seeded | 30 |
| 3. Mechanism | Prototype CLT pipeline on Gemma 3/Scope 2 → train CLTs on Gemma 4; locate decode features; steering interventions scored against references; fine-tune (text MT for target languages) + before/after crosscoder diff | Causal verdict on the central claim; intervention-vs-fine-tune cost/quality table | 45 |
| 4. Writeup | Thesis + workshop-paper-shaped artifact | Thesis | 20 |
| Stretch A (modality) | English-audio→LRL-text fine-tune on church corpus; localize failures (hearing vs. decoding) | Cross-modal extension; **fused speech-translation MatrixLab provider candidate** (model + eval + $/segment report; lab decides productionization) | +30–40 |
| Stretch B (tooling) | Clean and release the Gemma 4 CLT suite | Public interp tooling (none exists for Gemma 4) | +10–15 |

## ToAll integration points (all via existing seams)

1. **Translation memory import** (Phase 2 side deliverable): human-origin, high-confidence entries in exactly ToAll's content domain; existing TMX path; RAT exact-tier hits at zero model cost.
2. **QE badges** (RQ2): per-segment quality scores through the existing provenance-badge UI; the app finally knows a good dub from a bad one.
3. **MatrixLab provider candidate** (Stretch A): the speech-translation fine-tune is a *fused* ASR+MT provider (one call replaces two — directly attacks the streaming pipeline's serial-chain bottleneck, mirroring the existing Azure fused path). Delivered as an evaluated candidate with $/segment vs. Azure from the cost ledger; timing/timestamps remain with conventional ASR.

## Explicit non-goals

- No TTS work: Gemma 4 does not produce speech; voices remain lab XTTS + commercial providers. (Flag for the lab, outside this project's scope: XTTS-v2's CPML license is non-commercial by default — verify derivative status before commercial scale-out.)
- No competing MT training: the lab owns NLLB. The fine-tune exists for the RQ3 diff and (Stretch A) as an evaluated *candidate* handed to the lab — productionization is the lab's call.
- No dubbing-pipeline deployment work beyond the three seams above.
- No vanilla SAE feature-cataloguing as an end in itself (direction publicly deprioritized by DeepMind, Mar 2025); dictionaries are instruments for the causal test, with stability controls (fixed/multiple seeds, crosscoders for cross-checkpoint comparison).

## Skills exercised (career development)

Training under real constraints (CLT dictionaries: sparsity/reconstruction trade-offs, dead-feature debugging; LRL fine-tune: data pipelines, mixed precision, divergence debugging); large-scale GPU activation capture; evaluation-harness engineering with calibrated metrics; model serving + monitoring on AWS (QE service; optional GPU endpoint for the provider candidate); cost accounting via ToAll's ledger; hypothesis-driven mechanistic debugging.

## Resources needed

- Lab: church-corpus access **with licensing status for training and production use resolved in Phase 0**; NLLB checkpoints + per-language outputs; GPU allocation for CLT training and one fine-tune at 12B (26B-MoE optional) — confirmed available.
- ToAll: staging access; agreement on QE-badge and TM-import integration points; [manager] as engineering mentor.

## Risks and mitigations

- **Gemma 4 can't hear our languages** → Phase 0 finds out in week one; RQ3 proceeds text-only (thesis intact), the negative result quantifies the LRL modality gap.
- **Steering doesn't recover quality** → the central claim is falsified for those languages = the Translation Barrier Hypothesis has a resource floor; publishable, and the fine-tune diff still explains why.
- **Dictionary instability pollutes measurements** → known 2025–26 failure mode; fixed-seed protocol, multi-seed checks, crosscoders for any cross-model comparison; calibrated nulls for all geometry numbers (Gröger et al. 2026; *Back into Plato's Cave* 2026).
- **Auto-interp labels unreliable off-English** (arXiv:2606.00356: up to 4× label failure, worse on non-Latin scripts) → budget manual validation for target-language features; treat the failure itself as data.
- **Scoop risk** → closest priors: NLLB-200 geometry probe (arXiv:2603.02258 — geometric, correlational, no failure-mode analysis, no interventions) and multilingual CLT tracing (arXiv:2511.10840 — high/mid-resource only, no African languages, no quality-recovery test). Our differentiators: African languages, causal interventions scored against private human references, the speech axis, deployment.

## Positioning (novelty, verified by workflow-based literature sweeps 2026-07)

Closest prior work: Harrasse et al. 2025 (mechanistic failure taxonomy, no African languages, no interventions-to-quality link); Bafna et al. 2025 (Translation Barrier Hypothesis — behavioral, untested causally on LRLs); Mathewson 2026 (NLLB geometry, no failure analysis); Lee et al. NAACL 2025 (cross-modal, stitched models, high-resource); Sternberg et al. 2026 (speech pivots through text — English only). Unclaimed: causal decode-out testing on African languages; any interpretability on Gemma 4 or on unified-audio models; speech-translation supervision from N-way parallel religious-domain data; the intervention↔deployed-QE loop. Complements (does not overlap) the lab's training-dynamics program: that work asks how the shared space *forms* in controlled small models; this project asks whether the *formed* space in a production-scale model can be diagnosed and repaired, and ships the diagnosis.
