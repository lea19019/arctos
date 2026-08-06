# Tier 1 — Controlled Small-Scale Training: Working Plan

**Scope:** Tier 1 of "Does the Interlingua Grok?" only. Tiers 2 and 3 are out of scope for this document.
**Compute assumption:** A100-class lab GPUs.
**Status:** draft for discussion with PI.

---

## 1. What this experiment is actually for

The proposal's H1 states that mechanistic progress measures rise smoothly *before* downstream
cross-lingual transfer accuracy jumps. As written, that result is not safe to report.

Schaeffer, Miranda & Koyejo (NeurIPS 2023) showed that sharp capability "jumps" are frequently
artifacts of discontinuous metrics — exact-match accuracy, multiple-choice grade — applied to
smoothly improving per-token error. Zero-shot cross-lingual transfer accuracy is exactly such a
metric. So a lag between a smooth mechanistic curve and a jumpy accuracy curve is **guaranteed by
construction**, and would appear even if nothing mechanistically interesting happened.

The fix is not a caveat. It is the experiment:

> **Does the lag survive when the behavioral axis is measured with a continuous metric?**

This costs nothing extra — the same training runs answer both — and it is non-empty in every outcome:

| Outcome | Reading |
|---|---|
| Lag survives under continuous metrics | Real phase structure. H1 supported, and supported rigorously enough to build on. |
| Lag vanishes under continuous metrics | "Multilingual emergence is a mirage." Arguably the more interesting result, and it constrains the whole field. |
| Lag survives for some constructions/pairs, not others | Local asynchronous grokking — matches the OLMoE pretraining-grokking finding and Jian & Manning's sequential onsets. Probably the true answer. |

**Why this matters upstream.** Phase 2's failure-mode taxonomy (§8.2) and Contribution 9
("progress measures as training-time monitoring tools") both assume progress measures are real
signals rather than metric artifacts. Nobody has validated that assumption. Tier 1 done this way
is the foundation the rest of the program stands on.

### Framing amendment to raise with the PI

The proposal's three-phase story (memorization → circuit formation → cleanup) is in tension with
two published results: Dumas et al. found shared concept space forming within the **first 10% of
tokens**, and "Where to find Grokking in LLM Pretraining" found grokking in real pretraining is
**local and asynchronous** across data groups, not a clean global transition.

Recommendation: defend *per-construction, per-language-pair* transitions rather than a single
global phase transition. Better supported, dodges the strongest counterevidence, and it is what the
JSD measures actually measure anyway.

---

## 2. Scope decisions

### In

- One architecture, one objective, three language configurations, 5 seeds each.
- Behavioral measures (continuous + discontinuous, paired).
- JSD distributional divergence measures.
- Representation geometry (debiased CKA + mutual-nearest-neighbor).
- Changepoint statistics with bootstrap CIs over seeds.
- Positive and negative controls.

### Out, and why

| Dropped | Reason |
|---|---|
| Circuit extraction / edge Jaccard (§4.3) | Three independent 2024–2026 papers document that EAP/EAP-IG circuits have high resampling, rephrasing, and sample-wise variance, and that faithfulness depends on ablation methodology. Jaccard across two languages × two checkpoints compounds all of it. Highest cost, lowest yield. |
| Per-checkpoint independent SAEs (§4.2) | Cross-seed matched-feature rates as low as 1–4% for large TopK dictionaries; feature reproducibility 21–30%. Retraining the dictionary per checkpoint means "fraction cross-lingual" measures dictionary-fitting noise as much as model change. Crosscoder or nothing. |
| NLLB-like encoder-decoder arm | Weakest tooling support (TransformerLens encoder-decoder support is poor; circuit-tracer and Edge Pruning are decoder-oriented), and the least informative of the three arms. |
| Experiment B (curriculum, 4 variants) | Foroutan et al. already established that curriculum order doesn't change the endpoint and that transitions cause forgetting spikes. A "grokking signature" at the switch point is not separable from forgetting/re-learning at this scale. |
| Languages beyond 3 | See §3.2. The H4 contrast needs two distances, not eight languages. |

### Deferred to a Stage 2 that only happens if Stage 1 shows signal

- Crosscoder across checkpoints (following Bayazit et al.), replacing the independent-SAE design.

---

## 3. Experimental configuration

### 3.1 Architecture

**Decoder-only**, not the mBERT-like encoder. Reasons:

1. Jian & Manning's JSD measures are defined over **next-token** distributions. On an MLM you would
   be reconstructing masked-token distributions — workable, but non-standard, and it breaks the
   direct comparison to their exemplar-first baseline.
2. TransformerLens / SAELens / circuit-tracer are decoder-first; encoder support is second-class.
3. Phase 2's targets are decoders.

Counterargument to put to the PI: the classic cross-lingual-structure literature (Chi et al.,
K et al. ICLR 2020) is encoder-based, and mBERT/XLM-R are the course's focal models. Tier 3 can
still cover encoders cross-sectionally. **This is a PI decision, not mine.**

| Parameter | Value |
|---|---|
| Layers | 6 |
| d_model | 512 |
| Heads | 8 |
| FFN | 2048 |
| Sequence length | 512 |
| Vocab | 32k shared BPE, tied embeddings |
| Params | ~36M total, ~19M non-embedding |
| Optimizer | AdamW, weight decay 0.01 (the grokking-analog regularizer — keep it) |

### 3.2 Languages

**English, French, Turkish.**

- EN–FR: typologically close (Indo-European, both SVO, low morphological complexity).
- EN–TR: typologically distant (Turkic, SOV, agglutinative).
- That contrast is the entire H4 test. Eight languages buys nothing extra for 2.7× the cost.

**Why not Chinese or Japanese:** the proposal specifies subject-verb agreement as the shared
construction for both the JSD measures and circuit extraction (§5.2). Neither Chinese nor Japanese
has subject-verb agreement. Including them forces a different construction per language, which
destroys cross-lingual JSD comparability — you would be comparing divergences over different
phenomena. All three chosen languages have overt subject-verb agreement, so the construction is
well-defined and identically operationalized everywhere.

If the PI wants a Sinitic or Japonic language in Tier 1, the construction set must be rebuilt around
something that exists in all of them (argument structure, word-order prediction), and the JSD
measures redefined accordingly. That is a scope increase, not a language swap.

### 3.3 Data

- **Pretraining:** FineWeb2 (EN, FR, TR). Multilingual Wikipedia is dated for pretraining; FineWeb2
  is the current standard and covers all three well.
- **Tokenizer:** shared 32k BPE trained on **equal bytes per language**. Report per-language
  fertility (tokens/byte) as a table in the writeup — tokenizer fertility is a first-order confound
  in any cross-lingual alignment study, and "language-specific features" can be tokenization
  artifacts. This gets handled up front, not deferred.
- **Parallel eval:** FLORES-200 devtest — parallel across all three languages, which is what CKA
  and mutual-nearest-neighbor need (translation-equivalent sentence pairs).
- **Syntactic eval:** Universal Dependencies — EN-EWT, FR-GSD, TR-BOUN. Used for POS probes and
  Hewitt-Manning structural probes.
- **Agreement minimal pairs:** custom-built per language for the JSD measures. This is real
  linguistic work — budget for it explicitly (§6, W2).

### 3.4 Run matrix

| Config | Languages | Total tokens | Tokens per language | Purpose |
|---|---|---|---|---|
| **A1** | EN, FR | 2B | 1B each | Baseline pair |
| **A2** | EN, FR, TR | 2B | 667M each | Fixed *total* budget — more languages, less data each |
| **A3** | EN, FR, TR | 3B | 1B each | Fixed *per-language* budget — capacity pressure isolated |

**This is the fix to Experiment A's confound.** A1 vs A2 confounds "more languages" with "less data
per language" — and Foroutan et al. showed that apparent multilingual effects are frequently
data-quantity artifacts, making this confound first-order. A1 vs A3 holds per-language data
constant, so the only thing that changes is how many languages share the parameters. Report both
comparisons; the honest framing is that fixed capacity means you cannot fully escape the
capacity/data tradeoff, only bound it.

**5 seeds per config. 15 runs total.** Seeds are the line item that never gets cut — see §5.

### 3.5 Compute sizing

At ~2.5e8 FLOPs/token and ~20% MFU on an A100 (small models are overhead-bound; do not assume
peak), expect roughly 100–200k tokens/sec on one A100.

- 2B tokens ≈ **3–6 GPU-hours per run**
- 15 runs ≈ **60–90 GPU-hours total**

The evaluation is right that the proposal's "~2 weeks per run" is off by one to two orders of
magnitude. Base training is cheap. This means the budget goes to **seeds and checkpoint density**,
not model size.

**Day-1 task regardless:** run a 200-step throughput pilot on the actual hardware with the actual
architecture, measure tokens/sec, and derive the token budget from measured wall-clock. Do not plan
against the estimate above.

### 3.6 Checkpoint schedule — log-spaced, not uniform

The proposal specifies every 1000 steps. **That will miss the phenomenon.** Dumas et al. found
shared concept space forming within the first 10% of tokens; uniform spacing puts almost no
resolution there. Pythia's uniform cadence is a known limitation of Pythia, not a model to copy.

At batch = 262k tokens/step, 2B tokens ≈ 7600 steps. Use ~60 log-spaced checkpoints from step 1 to
7600 — which puts roughly half of them below step 760 (the first 10%), where the action is.

**Storage:** weights-only at 36M params fp32 ≈ 144MB per checkpoint. 60 checkpoints × 15 runs
≈ **130GB**. Save full optimizer state only at a handful of steps. Confirm disk allocation in W1 —
this is a real planning item and it is easy to discover too late.

---

## 4. Measures, in build order

Build in this order. Each stage gates the next.

### Stage 1 — Behavioral axis (build first; everything hangs off it)

Every behavioral phenomenon is measured **twice**, once continuous and once discontinuous. The pair
*is* the mirage test.

| Phenomenon | Discontinuous (the proposal's version) | Continuous (the control) |
|---|---|---|
| Language modeling | — | **Bits-per-byte** on FLORES devtest per language |
| Zero-shot POS transfer | Accuracy (train probe on EN, eval FR/TR) | Mean log-prob of gold label; Brier score |
| Syntactic structure | UUAS at threshold | Structural-probe **Spearman correlation** |

**Bits-per-byte, not per-token cross-entropy.** With a shared BPE, fertility differs by language, so
per-token CE is not comparable across languages — it partly measures tokenization. Bits-per-byte is
tokenizer-invariant and is the correct continuous LM metric here.

### Stage 2 — JSD distributional divergence (cheapest real signal, least scooped)

Following Jian & Manning (EACL 2026 Best Paper). Forward passes only, no dictionary training, so it
runs at every checkpoint at full temporal resolution.

- **Cross-lingual within-class JSD:** next-token distributions at the agreement target position,
  same verb class, different languages.
- **Cross-lingual between-class JSD:** same, different verb classes.
- **Within-language item JSD:** individual verbs, same language and class — tracks item-specific
  learning.
- **Exemplar-first baseline:** count-based co-occurrence vectors per language, confirming any
  abstraction-first trajectory is a property of learning dynamics rather than the data distribution.
  Jian & Manning consider this essential; do not skip it.

Prediction under H2: cross-lingual class divergence separates from between-class divergence *before*
within-language item divergence rises.

### Stage 3 — Representation geometry

- **Debiased CKA** between translation-equivalent FLORES sentences, per layer, per checkpoint.
- **Mutual-nearest-neighbor** (the Platonic Representation Hypothesis metric) alongside it.

Two measures, not one, and deliberately one robust + one fragile. CKA is documented as sensitive to
outliers and single-point translations, dominated by high-variance principal components, and
manipulable without functional change; biased CKA is spuriously high in the low-sample/high-dim
regime, which is exactly the per-layer sentence-representation regime. If CKA and MNN disagree,
that disagreement is a finding about the measure and gets reported.

### Stage 4 — Crosscoders (conditional; only if Stages 1–3 show signal)

Crosscoder across checkpoints following Bayazit et al., learning a **joint** feature space, giving a
comparable cross-lingual feature-fraction trajectory. Report cross-seed stability explicitly. If
feature fraction is not stable across seeds, that is a publishable negative methodological result
and it gets written up as one.

---

## 5. Analysis protocol

### Common footing

Mechanistic and behavioral trajectories have different units, scales, and noise. Comparing raw
sigmoid inflection points across them is not a defensible way to establish a lag — inflection
estimates are sensitive to noise and to the assumed parametric form, and different measures'
inflections are not on a common footing.

Instead:

1. Min-max normalize each trajectory over training.
2. **Changepoint detection** (PELT or Bayesian online changepoint) on the log(step) axis.
3. **Bootstrap over the 5 seeds** to get a CI on Δt = t_behavioral − t_mechanistic.
4. The claim is: the CI on Δt excludes zero. Not "the curves look different."

### Controls — build these in week 1, before any real analysis

**Positive control.** Replicate the known Nanda modular-addition grokking lag through *your own*
changepoint pipeline. If the pipeline cannot recover a lag that is definitely there, no multilingual
result it produces is trustworthy. This is a cheap, high-credibility week-1 deliverable and it makes
every later number defensible.

**Negative control.** A measure that should show *no* lag — random-projection similarity between
languages, or shuffled-language-label CKA. Proves the lag is not an artifact of the changepoint
machinery.

**Metric-pair control.** The same behavioral phenomenon under accuracy vs. log-prob (§4 Stage 1).
This is the mirage test proper.

### Pre-registration

Write the analysis plan — measures, changepoint method, seed count, the Δt test — and freeze it
before looking at any trajectory. Cheap, and it is the difference between "we found a lag" and
"we went looking until we found one." Worth a paragraph in the writeup.

---

## 6. Timeline and task breakdown

Weeks are part-time-shaped; compress if full-time.

| Week | Work | Gate |
|---|---|---|
| **W1** | Throughput pilot (200 steps, measured tokens/sec). Tokenizer trained + fertility table. Data pipeline for FineWeb2 EN/FR/TR. Checkpoint storage confirmed. **Positive control: Nanda modular-addition lag recovered through own changepoint pipeline.** | Pipeline recovers the known lag. If not, stop and fix the statistics before spending GPU-hours. |
| **W2** | Agreement minimal pairs built for EN/FR/TR (real linguistic work — do not underestimate). Behavioral eval harness: bits-per-byte, POS probe, structural probe, both metric families. Pre-registration frozen. | Eval harness runs end-to-end on a random-init model without crashing. |
| **W3** | All 15 runs (3 configs × 5 seeds), log-spaced checkpointing. Behavioral measures computed across all checkpoints. | Training curves sane; checkpoints readable; no silent divergence. |
| **W4** | JSD measures + exemplar-first baseline across all checkpoints. | Cross-lingual class divergence shows *any* structured trajectory. If flat everywhere, see risk table. |
| **W5** | Geometry measures (debiased CKA + MNN). Changepoint analysis, bootstrap CIs, Δt test. Negative control. | **Main go/no-go: does Δt exclude zero under the continuous metric?** |
| **W6** | Writeup. Whichever way W5 came out. | — |
| **W7+** | *Conditional:* crosscoder stage, only if W5 produced signal worth localizing. | — |

---

## 7. Risks and what to do about them

| Risk | Response |
|---|---|
| Lag disappears under continuous metrics | **This is a result, not a failure** — see §1. Pivot the writeup to *where* (which layers) convergence happens rather than *when*, using the Platonic framing. Do not chase the lag with a different metric until it reappears. |
| JSD trajectories are flat — no structure at all | Most likely cause is that the minimal pairs are too easy or the model is too small to represent the construction. Diagnose by checking whether the model gets agreement right *at all* at the final checkpoint before concluding anything about dynamics. |
| Seed variance swamps Δt | The honest answer is that the effect is smaller than seed noise, which is itself worth reporting given that the scooping papers used one seed. Adding seeds is cheap (~4 GPU-hours each) — go to 10 before giving up. |
| CKA and MNN disagree | Report it. It is a real methodological finding about a measure the field uses uncritically. |
| Scooped further during the work | The descriptive two-stage finding is already scooped (Inaba EMNLP 2025; Riemenschneider & Frank ACL 2025; copy-first-translate-later). The defensible novelty was never the phenomenon — it is the metric-artifact adjudication and the statistical protocol. That is harder to scoop because it requires someone to care about being right. |
| Turkish tokenizer fertility is much worse than EN/FR | Expected — Turkish is agglutinative. This is why fertility gets reported and why bits-per-byte is the metric. If fertility is extreme, consider a per-language vocabulary allocation and report both. |

---

## 8. Open questions for the PI

1. **Decoder vs. encoder** (§3.1). I recommend decoder for tooling and for JSD-measure compatibility;
   the counterargument is continuity with Chi et al. and the course's focal models.
2. **Framing** (§1): lead the writeup with H1-as-stated, or with the metric-artifact question?
   The experiment is the same either way, so this can be decided after W5.
3. **Language set** (§3.2): confirm that dropping Chinese/Japanese from Tier 1 is acceptable given
   the subject-verb-agreement problem, or fund the construction-set rebuild.
4. **Whether the crosscoder stage is expected**, or whether Stages 1–3 constitute a complete Tier 1
   deliverable on their own. My view: they do.
