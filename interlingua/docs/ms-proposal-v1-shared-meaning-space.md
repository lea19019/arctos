# MS Final Project Proposal (Draft)

## Where Does the Shared Meaning Space Break? Measuring, Explaining, and Deploying Cross-Lingual Representation Quality for African Low-Resource Languages

**Student:** Adrian Castillo · **Advisor:** [advisor] · **Industry collaboration:** ToAll (dubbing platform) · **Estimated effort:** 150+ hours, Fall 2026

---

## Motivation

Multilingual models are claimed to encode meaning in a shared, language-neutral representation space — the modern descendant of the old "interlingua hypothesis," and a machine-testable echo of an older question: whether human languages share a common underlying encoding of meaning. The 2024–2026 literature (Semantic Hub Hypothesis, ICLR 2025; Lee et al., NAACL 2025) shows this sharing is real but partial — and essentially untested for the languages where it matters most. Meanwhile, ToAll is deploying lab-trained NLLB (MT) and XTTS (TTS) models to dub content into African languages that no one on the team can read: the product has **no way to know whether its output is any good.**

These are the same problem. If the shared space is real for a language, distance in that space is a translation-quality signal. Where it isn't, we need to know — and to explain why.

## Research questions

- **RQ1 (map):** Across the lab's ~dozens of African languages with human-translated parallel data, where do frozen multilingual encoders (SONAR, AfriE5, AfroXLM-R) actually place correct translation pairs close together, and where does the shared space break down?
- **RQ2 (signal):** For languages where the space holds, does source↔output embedding distance track true translation quality of the lab's NLLB models (validated against held-out human references, benchmarked against AfriCOMET and BLASER 2.0)?
- **RQ3 (mechanism):** For the failures surfaced by RQ1–RQ2, *why*? Layer-wise convergence analysis, tokenizer fragmentation, language-mean offsets (with interventional correction), and an English-pivot test on NLLB (per Wendler et al. 2024 / Aya-23 2025).
- **RQ-Stretch (modality):** Does representational convergence hold across *modality* as well as language — same content as speech vs. text — in models never trained to align them? (The closest prior work, Lee et al. NAACL 2025, covers almost no African languages and flags this as open.)

## Plan and deliverables

| Phase | Work | Deliverable | ~Hours |
|---|---|---|---|
| 0. Audit | Pretraining-coverage audit of each candidate encoder vs. the lab's language list; data inspection | Coverage table (itself a small novel contribution — none published) | 15 |
| 1. Census | Multi-GPU batch embedding pipeline over all parallel data; per-language convergence map with **calibrated** metrics (permutation-null, multi-sample-size, per Gröger et al. 2026) | The map: language × encoder reliability; queryable results store | 40 |
| 2. QE tool | Score NLLB outputs vs. references; per-language calibration; trained scoring head; length-bias controls; deploy as a service with monitoring; ToAll integration as provenance-style quality badges | Deployed, monitored QE service + benchmark vs. AfriCOMET/BLASER | 40 |
| 3. Interp | 2–3 targeted experiments on Phase 1–2 anomalies (layer-wise geometry, mean-shift intervention, English-pivot test) | Mechanistic account of where/why the signal fails | 35 |
| 4. Writeup | Thesis + (goal) workshop-paper-shaped artifact | Thesis | 20 |
| Stretch | Repeat census across speech×text (2×2: language × modality) using lab audio + speech encoders | Cross-modal convergence study | +30–40 |

## Explicit non-goals (no duplication of lab work)

- **No model training that overlaps the lab's:** NLLB and XTTS are used as-is, as study objects. All encoders are frozen/off-the-shelf.
- Permitted small-scale training only where deployment-motivated and outside lab scope: the QE scoring head (small regressor) and, if needed, distillation/quantization of an encoder to fit ToAll's serving envelope.
- No dubbing-pipeline deployment work (owned by the ToAll team).

## Skills exercised (career development)

Large-scale GPU batch inference pipelines; evaluation-harness engineering; model serving and AWS deployment; production monitoring/calibration; knowledge distillation/quantization (scoped); hypothesis-driven model debugging (interpretability).

## Resources needed

- Lab: parallel speech+text data access (with licensing status for production use clarified); NLLB/XTTS checkpoints; A100/H200 allocation (batch inference only — modest).
- ToAll: agreement on the QE-badge integration point; access to staging environment; [manager] as engineering mentor.

## Risks and mitigations

- **Encoders cover fewer lab languages than hoped** → the coverage audit (Phase 0) finds this in week 1; the map of *non*-coverage is still a result.
- **Distance doesn't track quality for most languages** → that is a publishable negative result (WMT25 abandoned QE for the lowest-resource languages; explaining why is the contribution). The design has no empty-handed outcome.
- **Data alignment quality** → Phase 0 inspection before any pipeline work.

## Positioning (novelty, verified by literature sweep 2026-07)

Closest prior work: Semantic Hub Hypothesis (ICLR 2025); Lee et al. (NAACL 2025, cross-modal but high/mid-resource only); NLLB-200 geometry probing (2026, text-only, word-lists); BLASER 2.0 (engineered shared space — this project asks whether convergence emerges *untrained*). Unclaimed: the per-language link between measured convergence and QE reliability, at this language breadth, with deployment. Full sweep: `papers/LITERATURE_SWEEP_2026-07.md`.
