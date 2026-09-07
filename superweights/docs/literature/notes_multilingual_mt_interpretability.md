# NOTES — multilingual / MT interpretability, read against a single-scalar super-weight project

Date: 2026-09-05. Index and venues: [`README.md`](README.md).

**Epistemic marking used throughout.** `[read]` = I opened the PDF and the statement
comes from a section I read. `[skim]` = abstract/intro/figures only. `[unopened]` = I
have the title and venue but never opened it; do not cite. `[inferred]` = my reasoning,
not a claim in any paper. Nothing here is cited from memory; every arXiv id and venue
was verified against the ACL Anthology listing or the arXiv `comment`/`journal_ref`
field during this session.

---

## §1 Paper-by-paper (papers actually read)

### 1.1 The three core reads

**Tang et al. 2024, LAPE — ACL 2024, peer-reviewed. `[read]` Causal (ablation).**
Language activation probability entropy over FFN activations; bottom-1% LAPE neurons
are "language-specific". Fig. 2 shows the perturbation matrix is diagonal for LAPE and
not for the three baselines — deactivating language *k*'s neurons raises PPL for *k*
only. Fig. 5: the layer distribution is a skewed **U** — layer 2 holds ~7,000 neurons,
layers 5–47 about 100 each, last four layers >1,000 each (LLaMA-2-70B, 80 layers).
Fig. 6: mean sentence-embedding similarity across languages runs *opposite* to that
distribution — high in the middle, low at the ends. Table 3: ~23K neurons total, and
**English has the fewest** (836 vs 5–6K for others). Table 1: deactivation drops GPT-4-
judged Vicuna scores on the diagonal. Fig. 7a: English is the dominance centre in
LLaMA-2; BLOOM has several centres. Caveat the project should note: zh/ja share ~25%
of their neurons and interfere, so "language-specific" is not clean even here.

**Zhao et al. 2024, MWork/PLND — NeurIPS 2024, peer-reviewed. `[read]` Causal.**
Fig. 1: logit-lens language ratio shifts non-English → English → non-English across
layers. PLND scores neuron importance by the L2 change in the layer output when the
neuron is zeroed, parallelised over the FFN and attention. Table 1: deactivating
language-specific neurons collapses XLSum ROUGE-L (Vicuna avg 26.6 → 0.37; Mistral 25.8
→ 0.21) while an equal number of random neurons does nothing (26.7 / 25.7).
**Correction the project needs:** the paper's own figure is **0.13% of all neurons**
(abstract, and §2.3). I grepped the full text — the phrase "as few as four neurons"
does not occur, and no "four neurons" claim appears anywhere. The project brief's
"as few as four neurons (Zhao 2024)" is **unsupported by this paper** and must be
either re-sourced or restated as 0.13%. `[read]`

**"Attention Sinks in Massively Multilingual NMT" 2026 (arXiv:2605.01229) — arXiv
preprint, NOT peer-reviewed. `[read] fully. Correlational. Treat with caution.**
This paper is the reason the project's framing needs to change, and it is also not
trustworthy as written.

*What it actually measured.* NLLB-200-600M-distilled, **cross-attention only** (decoder
attending to **source** tokens), averaged over 12 decoder layers × 16 heads × all
decoder steps, 1,000 English source sentences translated into Swahili, Kikuyu, Somali,
Luo, plus 200 sentences into German, Turkish, Chinese, Hindi.

*The headline the project has backwards.* Table 1: the sink is **`</s>`, at 78–87%** of
cross-attention mass. **Language tags get only 1.5–2.0%.** Punctuation 2.5–3.4%,
content tokens 9.2–16.5%. The brief's premise — "the NLLB sinks paper says sinks sit on
language tags" — is contradicted by the paper's own main table. The authors say so
explicitly: language tags "despite being the initial motivation for this investigation,
account for only 1.5–2%," and they attribute the `</s>` sink to it being the final
source token in every sentence, i.e. a boundary marker.

*Internal contradictions — three, all material.* (a) §6.2 states "the language tag
absorbed roughly 40–45% of attention in all cases," flatly contradicting Table 1's
1.5–2%. (b) §4.3 validation criterion 1 says filtered content tokens "account for
30–35% of original attention mass," contradicting Table 1's 9.2–16.5%. (c) §5.1 credits
the recovered structure to "removing language tags," when by its own numbers the
filtering is overwhelmingly removing `</s>`. The "Somali paradox" (§6.1) is presented as
a discovery but is a hypothesis with no test. The declared metric is cosine similarity
of the mean attention vector to the *uniform* distribution — an unusual choice, and
"similarity rises after filtering" is close to arithmetically guaranteed once the
dominant spike is removed and the rest renormalised, so the +93% is not obviously a
finding. **Recommendation: cite it only for Table 1's token-type breakdown, always with
the caveat, and never rely on its derived statistics.** `[read]`

### 1.2 Latent language and pivoting

**Wendler et al. 2024 — ACL 2024. `[read] abstract+intro. Correlational (logit lens).**
Llama-2, single-token translation prompts. Three phases: embeddings start far from
output tokens; middle layers already decode the semantically right token but rank its
**English** form above the input-language form; final layers move into an
input-language region. Concept space "lies closer to English."

**Wu et al. 2025, Semantic Hub — ICLR 2025. `[read] abstract+intro. Causal.**
Generalises Wendler across languages *and* modalities (arithmetic, code, image, audio).
The load-bearing part for us: **interventions in the shared space in one data type
predictably affect outputs in another**, so the shared space is used, not vestigial.

**Schut et al. 2025 — arXiv preprint. `[skim].** Extends pivoting to open-ended
multi-token generation (Llama-3.1-70B, Aya-23-35B). Nuance worth keeping: **lexical**
words route through English, **non-lexical** words do not. Steering vectors work better
built in English than in the I/O language.

**Trinley et al. 2025, Aya-23 — arXiv preprint. `[skim].** Balanced-data Aya-23-8B
activates **typologically related** languages during translation rather than one English
pivot, and its code-mixed language-specific neurons concentrate in **final** layers,
diverging from Tang/Kojima's U-shape. Directly relevant: Aya-Expanse-8B is a local model,
so the project cannot assume the LLaMA-family layer profile transfers.

**Bafna et al. 2025, Translation Barrier — IJCNLP-AACL 2025. `[read] abstract.
Correlational + intervention.** Implicit task-solving → translation pipeline; the
**translation (decode-out) stage** dominates final error across 108 language pairs,
worst for low-resource targets. Consequence: an internal-representation measure and an
output-quality measure can diverge, and diverge most exactly where the project's
per-language damage profile would be most interesting.

**Tezuka & Inoue 2025, Transfer Neurons — EMNLP 2025 Main. `[skim].** MLP neurons that
*move* representations between language-specific and shared latent spaces; the paper
argues one function of language-specific neurons is to effect that movement, and that
transfer neurons are critical for reasoning. The most causally-grounded successor to LAPE.

**Kojima et al. 2024 — NAACL 2024. `[skim].** Language-specific neurons overlap **<5%**
between languages, sit in the first and last few layers, and tampering with <1% of
neurons drastically changes the probability of the target language appearing.

**Mondal et al. 2025 — Insights from Negative Results (NAACL 2025 workshop). `[read]
abstract+intro. Negative result.** LAPE / activation-probability neurons + neuron-LoRA
give **<1 absolute point** on XNLI and XQuAD for low-resource languages. The finding is
narrower than the title: it is about *cross-lingual transfer on downstream tasks*, not
about whether the neurons control language identity — Tang and Kojima's generation-side
results are not overturned. The correct use is as a bound on causal over-reading, and
the project should state that scope when citing it.

**Wang et al. 2024, Sharing Matters (arXiv:2406.09265) — arXiv preprint. `[skim].**
Important counterweight: neurons are classed all-shared / partial-shared / specific /
non-activated, and **deactivating the all-shared neurons hurts most**. Language-specific
neurons are not where most of the capability lives.

**Harrasse et al., Cross-Layer Transcoders (arXiv:2511.10840) — preprint, under review.
`[skim].** Trains multilingual models at controlled mixtures. Two things matter here:
language identity is **linearly encoded from early layers**, and decoding relies partly
on **a small set of high-frequency features in the final layers** which can be
intervened on to suppress one language and substitute another.

**Nie et al. 2025, Mechanistic Language Confusion — EMNLP 2025 Findings. `[skim].**
Confusion points localised to **final-layer** transition failures; editing a small
critical-neuron set mitigates confusion while preserving competence.

**Zhong et al., Language Lives in Sparse Dimensions (arXiv:2510.07213) — EACL 2026 Main,
peer-reviewed. `[read] abstract+fig.1. Causal.** This is the closest published thing to
the project's own thesis and the project must position against it explicitly. The claim:
the cross-lingual transition is governed by **a small, sparse set of residual
dimensions occurring at consistent indices from intermediate to final layers**;
identified training-free from as few as 50 sentences; intervening switches the output
language while preserving semantics; **outperforms prior neuron-based approaches at
lower cost**. Note what it is *not*: dimensions of the residual stream, identified by
activation statistics, in decoder-only English-centric LLMs — not a single scalar
*weight* in `mlp.down_proj`, and no MT/encoder-decoder evaluation.

### 1.3 Outlier dimensions in multilingual models

**Rajaee & Pilehvar 2022 — Findings of ACL 2022. `[read] abstract. Correlational.**
The single most important negative prior for this project: **mBERT exhibits _no_
outlier dimension**, while monolingual BERT has a few dimensions with high contribution
to anisotropy. mBERT is still highly anisotropic. Increasing isotropy improves
performance. Two caveats before this is treated as a prediction: it is an *encoder*
at BERT scale, and "outlier dimension" there is defined by contribution to anisotropy
in the embedding space, which is **not** the same construct as a massive activation or
a super weight in a 7B decoder. `[inferred]` But it is exactly the result a reviewer
will raise, and the project should test rather than assume.

**Timkey & van Schijndel 2021 — EMNLP 2021. `[read] abstract. Correlational.**
1–3 rogue dimensions dominate cosine/Euclidean similarity, and there is a **striking
mismatch between the dimensions that dominate similarity measures and those important
to model behaviour**. Standardisation corrects it. This is the null the project must
beat for any claim of the form "this channel is where language lives": show behavioural
consequence, not geometric dominance.

### 1.4 MT internals and hallucination

**Voita et al. 2021 — ACL 2021. `[read] abstract.** LRP variant whose conservation
principle yields the *proportion* of source vs target-prefix influence, not an abstract
importance score. Training-dynamics finding: more data → more source reliance; the
training process is non-monotonic with several stages.

**Ferrando et al. 2022, ALTI — EMNLP 2022. `[read] abstract.** Token-to-token mixing
over the whole attention block (heads + residual + layer-norm), aggregated layer-wise;
more faithful and more robust than gradient methods. **ALTI+ — EMNLP 2022. `[read]
abstract.** Extends to encoder-decoder, giving source *and* target-prefix attributions;
explicitly "can be extended to any encoder-decoder Transformer-based model" — so it
applies to NLLB without adaptation.

**Dale et al. 2023 — ACL 2023. `[read] abstract. Causal-ish (detection + mitigation).**
Hallucinations are translations detached from the source, so low **source contribution**
(measured with ALTI+) detects them; improves detection of the most severe hallucinations
**by a factor of 2** over prior internals-based methods, and mitigates at test time on
par with external-model approaches. Cross-lingual sentence similarity does better still.

**Guerreiro et al. 2023 — TACL 2023. `[read] abstract + grepped taxonomy sections.
Correlational + intervention.** The taxonomy the project needs: hallucinations are
**detached** (content unrelated to source) or **oscillatory** ("inadequate translations
that contain erroneous repetitions of words and phrases"). Detectors: **ALTI+** for
detached, **top n-gram (TNG)** — a lightweight black-box repetition heuristic — for
oscillatory; the two can be applied simultaneously. Empirics over 100+ directions on
M2M and ChatGPT: **over 90% of off-target hallucinations occur when translating out of
English**; detached hallucinations are more frequent in low-resource directions, and
oscillatory ones *less* prevalent there than at mid/high resource.

**HalOmi 2023 — EMNLP 2023. `[skim].** Human-annotated hallucination and omission over
18 directions, sentence and word level, varying resource and script. Crucially it shows
**conclusions drawn from a single language pair largely do not hold at scale** — a
direct warning against the project's existing small-n per-language results.

**Chen et al. 2023, Off-target — Findings of ACL 2023. `[skim].** Off-target arises from
failure to encode a discriminative target-language signal; smaller vocabulary
KL-divergence between two languages predicts higher off-target rate; isolating decoder
vocabularies alone helps (29% → 8% off-target).

**Wu et al. 2021, Language Tags Matter — Findings of ACL 2021. `[skim].** Tag
*placement* (source-side vs target-side) materially changes zero-shot behaviour.

**Qu et al. 2025, Registering — ACL 2025 Main. `[skim].** Inserts artificial
target-language "register" tokens between source and target and masks attention so
target generation attends **only** to the registers. This is an engineered,
deliberately-constructed language-tag sink that beats NLLB-3.3B — evidence that
concentrating target-language identity on a few token positions is functional, not
merely an artifact.

**DecoderLens — Findings of NAACL 2024. `[skim].** The logit-lens analogue for
encoder-decoder models: run the decoder on intermediate *encoder* representations. The
enabling method for layer-wise NLLB analysis, and the reason the project does not need
a decoder-only model to do layer-wise work.

**NLLB-200 tech report (arXiv:2207.04672). `[read] targeted grep, §8.**
Architecturally load-bearing for bridge Q(iii): the model "is trained to accept the
**source language token as prefix for the source sequence** and the **target language
token as the first token input to the decoder**." So NLLB's decoder position 0 *is* the
target language tag. The distilled 600M variant is 12 encoder + 12 decoder layers, FFN
4096, 16 heads, ~256K vocab.

**Repetition In Repetition Out — NeurIPS 2023. `[skim].** Repetition is a
self-reinforcing local attractor; repetition rates in generated text correlate strongly
with repetition rates in *training* data. The baseline explanation for "We. We. We."
that the project must rule out before claiming anything mechanistic.

### 1.5 Training dynamics

**Blevins et al. 2022 — EMNLP 2022. `[skim].** XLM-R checkpoints: in-language ability
early, lower-level skills before higher; **cross-lingual transfer onset differs by
language pair**; the final layer degrades over training while knowledge moves to lower
layers.

**Wang et al. 2024, Probing Emergence — Findings of ACL 2024. `[skim].** BLOOM
checkpoints across steps and scales; cross-lingual **neuron overlap correlates with
transfer performance**, with detected degradation phases mid-training. Confirms public
multilingual checkpoint suites exist (BLOOM), which matters for any onset question.

**Copy First, Translate Later (arXiv:2604.17633) — preprint. `[skim].** Dense
checkpoints of a 1.7B nine-language pretrain; translation develops in two phases —
copying/surface similarity first, generalizing translation second.

### 1.6 Targeted reads from the interlingua copies

**Goldman et al. 2025, ECLeKTic. `[read] abstract+intro.** Closed-book QA built from
facts present in one language's Wikipedia and absent from others, so consistency cannot
come from parallel exposure. 8 LLMs; SOTA models **fail to share knowledge across
languages** even when they know the fact in the source language. Useful as the
knowledge-transfer analogue of the translation-barrier result; not needed for the core
project.

**Jian & Manning, EACL 2026 (`EACL-2026-long-32.pdf`). `[read] abstract.** GPT-2 small,
divergence-based metrics on next-token distributions: **class-level (abstract)
behaviour appears at earlier training steps than item-specific behaviour**, and
different linguistic behaviours emerge **abruptly, in sequence**. Relevant only if the
project takes on an onset/training-dynamics question — it supplies the "abrupt, ordered
emergence" prior and a divergence-metric methodology.

**Shai et al. 2026, Transformers learn factored representations (arXiv:2602.02385).
`[read] abstract+intro.** Transformers represent independent latent factors in
**orthogonal subspaces of the residual stream**, dimension growing linearly rather than
exponentially, with an inductive bias toward factoring even when it costs fidelity.
This is the cleanest theoretical warrant `[inferred]` for expecting "language identity"
to occupy its own low-dimensional residual subspace — which is what would make a
single-channel story coherent. It is a synthetic-data paper; do not present it as
evidence about real LLMs.

---

## §2 What is known about where language identity lives

**In decoder-only LLMs.** Convergent, replicated, and reasonably strong:

1. *A layer-wise trajectory exists.* Input-language → shared/English-scaffolded concept
   space → output-language. Independently reported by Wendler (ACL 2024), Zhao (NeurIPS
   2024), Wu (ICLR 2025), Schut (2025), Tezuka (EMNLP 2025). Evidence is mostly logit
   lens (correlational), but Wu and Tezuka add interventions that propagate as predicted,
   so the shared space is causally used, not decorative.
2. *Language-specific parameters are sparse and sit at the ends.* Tang (bottom-1% LAPE,
   U-shaped over layers), Kojima (<5% overlap, first/last layers), Zhao (0.13% of
   neurons collapses generation). Ablation evidence here is genuinely causal and
   replicated across LLaMA-2, BLOOM, OPT, Mistral, Phi-2, Vicuna.
3. *But "specific" ≠ "where the capability lives."* Wang 2024 finds all-shared neurons
   matter more; Mondal 2025 finds language-specific neurons buy <1 point of
   cross-lingual transfer. The honest summary: these neurons **control which language
   comes out**, and do **not** carry the cross-lingual competence.
4. *The layer profile is training-data dependent.* Trinley finds Aya-23's are in the
   final layers, not U-shaped; Harrasse finds language identity linearly encoded early
   with decoding concentrated in a few late high-frequency features. Do not assume the
   LLaMA profile for Aya, Tower or EuroLLM.
5. *Sparse dimensions, not only neurons.* Zhong (EACL 2026) shows a small set of
   residual **dimensions at consistent indices** across layers governs the transition,
   training-free and beating neuron methods. This is the finest-grained *causal*
   localisation currently published, and it is dimensions — one step from channels.

**In NMT / encoder-decoder models.** Much thinner, and much weaker causally:

6. *Language identity is architecturally placed, not discovered.* NLLB puts the source
   tag at encoder position 0 and the **target tag at decoder position 0** (verified in
   the tech report). Wu 2021 shows placement changes zero-shot behaviour; Qu 2025 shows
   that forcing target generation to attend only to inserted target-language "registers"
   works better than NLLB-3.3B. So concentrating language identity on a few positions is
   functional.
7. *Sinks in NLLB cross-attention are on `</s>`, not the language tag* (arXiv:2605.01229
   Table 1: 78–87% vs 1.5–2%) — and that paper is an unreviewed preprint that
   contradicts itself elsewhere on exactly this number. **Nobody has measured decoder
   self-attention sinks or residual-stream massive activations in NLLB at all.**
   `[inferred]` from the absence of any such paper in this sweep.
8. *Attribution tooling is mature and encoder-decoder-native.* ALTI, ALTI+, LRP-based
   source/target contribution, DecoderLens. Source contribution is a validated detector
   of detached hallucination (Dale, factor-2 improvement).
9. *Geometric NLLB probing exists but is weak.* arXiv:2603.02258 finds a language-neutral
   conceptual store plus per-language offsets, but ρ=0.13 on the phylogenetic
   correlation, Swadesh word lists only, no interventions, unreviewed preprint.

**Strength of causal evidence, summarised.** Strongest: neuron/dimension ablation flips
output language (Tang, Kojima, Zhao, Zhong — multiple models, random-ablation controls).
Moderate: shared-space interventions propagate cross-lingually (Wu). Weak/absent:
anything about massive activations, sinks or single weights in a multilingual or
encoder-decoder model. **The gap the project sits in is real** — I searched explicitly
for super-weight × multilingual and super-weight × translation work and found none.

---

## §3 The bridge — the five candidate questions, evaluated

Standing constraint: claims must be about **single scalar weights / the massive
activation they create**, not neurons. Standing caution (Mondal): localisation is not
load-bearing-ness. Standing caution (Timkey): geometric dominance is not behavioural
importance.

### (i) Is the massive-activation channel/onset the same for every input language, and is its magnitude language-dependent? — **ADOPT, as the foundation.**

This is the right first question because it is cheap, it is a genuine gap, and every
other question here presupposes its answer.

*Measurement.* For each model (NLLB-600M/3.3B, TowerBase-7B, EuroLLM-9B,
Aya-Expanse-8B), run FLORES-200 devtest through N languages. Record, per language: the
`(layer, down_proj row, residual channel)` coordinate of the maximum-magnitude
activation; the layer of first emergence; the magnitude at emergence and at the last
layer; and whether the coordinate set is identical across languages.

*Null.* Coordinate identical across all languages, and magnitude differences across
languages no larger than magnitude differences across **random sentence subsets within
one language** — this within-language resample is the null, and it is the part that is
easy to omit and fatal to omit. Report magnitude as effect size with 95% CI over
sentences, family size = number of languages, multiplicity-corrected.

*What Puccetti predicts.* Puccetti 2022 (in ``,
`[unopened]` by me) is cited in-repo for outlier dimensions being driven by token
frequency. If that transfers, magnitude should track per-language token frequency in
the pretraining mix, so **magnitude should be language-dependent while the coordinate
is not**. That is a sharp, falsifiable prediction and makes this question worth running
even if the coordinate turns out to be shared. Confirm Puccetti's actual claim before
citing it.

*Risk.* Rajaee & Pilehvar's mBERT null says the multilingual case may simply differ.
That is a finding either way.

### (ii) Does super-weight ablation damage languages unequally, and does the profile match per-language PTQ degradation or token frequency? — **ADOPT, sharpened; this is the headline.**

This is the project's strongest claim territory: it is single-weight by construction,
it is MT-native, and no one has done it.

*Sharpening needed.* "Does it damage languages unequally" is nearly guaranteed to be
yes and is therefore not the interesting form. The interesting forms are: (a) **is the
per-language damage ordering stable across models** that share a super-weight coordinate
(TowerBase vs TowerInstruct are the natural pair, since in-repo work already found they
share the coordinate); and (b) **is the damage profile explained by anything** — token
frequency in the pretraining mix, per-language PTQ degradation from
`UnevenPTQ-2025`/`Marchisio-2024` in the sibling folder, resource level, script, or
typological distance.

*Measurement.* chrF++ and COMET on FLORES-200 devtest, en→X and X→en, before/after
zeroing the single scalar (and, where the scalar is inert, the minimal SET). Spearman
the per-language chrF++ drop against each candidate covariate.

*Null.* The damage profile is **rank-uncorrelated** with every covariate, and the
per-language drop is indistinguishable from the profile produced by ablating a
**magnitude-matched random weight in the same `down_proj`** — that control is essential
and is the exact control Subramanian 2026 used in the sibling folder. Also required: an
FP16 no-ablation baseline scored in the same run (the repo's claim-hygiene rule 2).

*Confound that will bite.* Damage is measured in chrF++, which floors near 0. If
ablation collapses every language to chrF++ < 5, there is no profile to correlate —
the measure saturates. Mitigate by also reporting a graded measure that does not floor:
KL between ablated and intact next-token distributions on forced-decode of the
reference, per language. HalOmi's warning applies directly: single-pair conclusions do
not survive scale, so run ≥20 languages, not 4.

### (iii) In NLLB, do encoder and decoder have separate super weights, and is the decoder's on the language-tag position? — **ADOPT, with the premise corrected.**

The premise as written in the brief ("sinks sit on language tags") comes from a paper
whose own Table 1 says the opposite. But the *question* survives the correction and
actually gets better, because the architecture supports it independently: NLLB's target
language token **is** decoder position 0, and position 0 is where decoder-only models
put their sink. So the coincidence is structural, not borrowed from that preprint.

*Two genuinely separate sub-questions, which the brief conflates.*
(a) *Weights:* does the encoder have its own super weight, and the decoder its own?
Yu's detection procedure (one forward pass, max `down_proj` input/output activation)
applies to each stack separately. Nobody has run it on an encoder-decoder model.
(b) *Position:* is the decoder's massive activation carried on the language-tag token
position? Note this must be measured in the **residual stream / decoder self-attention**,
not in cross-attention — cross-attention is over *source* tokens, so it structurally
cannot show a target-tag sink, which is why arXiv:2605.01229 could never have found one.

*Measurement.* Per-position residual-stream norms in both stacks; sink rate in decoder
self-attention by position; then the causal test — ablate the candidate scalar and
measure chrF++ and **language-ID of the output**.

*Null.* Massive activation magnitude at the language-tag position is no larger than at
position 0 of a run where the tag is replaced by an arbitrary token, i.e. the effect is
positional, not tag-identity. This distinguishes "sink because first" from "sink because
language tag" — and the `</s>` result suggests the boundary-marker/positional
explanation is the live one.

*Payoff.* If the decoder's super weight sits on the target-language tag, then a single
scalar is the mechanical implementation of target-language selection, and Qu 2025's
"registers" become an engineered version of the same thing. That is a real result.

### (iv) Is the post-ablation collapse an off-target/oscillatory hallucination detectable by ALTI+/Dale's source contribution? — **ADOPT as an instrument; REJECT as a headline.**

The framing is right and the tooling is exactly correct — Guerreiro's oscillatory
category *is* "erroneous repetitions of words and phrases," TNG detects it, ALTI+
detects the detached kind, both are validated against human annotation (HalOmi), and
ALTI+ is encoder-decoder-native. `[read]`

*Why not a headline.* "Ablation breaks the model and the broken output looks broken" is
close to unfalsifiable. Establishing that post-ablation output is a hallucination by
Guerreiro's criteria is a *characterisation*, not a discovery.

*What makes it worth doing anyway.* It converts the collapse from an anecdote into a
number on a validated scale, and it discriminates two mechanisms: if source contribution
(ALTI+) **collapses**, the ablation severed source conditioning; if source contribution
is **preserved** while output degenerates, the ablation broke the decode-out stage,
which would connect the super weight directly to Bafna's translation barrier. That
contrast is a real finding either way. Use it as the measurement layer under (ii) and
(v), not as its own question.

*Null.* Post-ablation ALTI+ source contribution is within the distribution of
naturally-occurring hallucinations in HalOmi for the same direction.

### (v) Does the repeated token after ablation change with source/target language? — **ADOPT; best cost-to-insight ratio in the list.**

This is the sharpest question here. It is nearly free to run, it is single-weight by
construction, and — unusually — the two outcomes point at genuinely different mechanisms
rather than at "strong effect / weak effect."

*The logic.* If the attractor token is **fixed** across languages (always "We", or
always one token id), the super weight's removal leaves the residual stream pinned to a
fixed direction that decodes to one token — a mechanism claim about a direction. If the
attractor token **changes with the target language** ("We" in English, "der" in German),
the collapse has fallen back on the **target-language unigram prior**, and the super
weight is not encoding a direction but maintaining whatever suppresses that prior.

*Measurement.* Ablate; force each target language; collect the degenerate output;
compute the token-unigram distribution of generations per language and its KL/rank
correlation against (a) that language's corpus unigram distribution and (b) the
generations for every other language. Do it for both the LLM MT models and NLLB.

*Null — and this one is load-bearing.* The attractor token is whatever the model's
**unconditional** most-frequent token is for that language, i.e. exactly the corpus
unigram argmax. If so, the "attractor" is the unigram prior and there is no
direction-specific story to tell. Report the rank correlation with a CI, not the modal
token.

*Design note `[inferred]`.* Repetition In Repetition Out (NeurIPS 2023) says repetition
is a self-reinforcing attractor tied to training repetition rates — so a third
possibility is that the attractor is driven by *training-data* repetition, not the
unigram prior. Distinguish by correlating against both. Also: `Tang` Table 2 shows that
LAPE-deactivated LLaMA-2-70B produces a *code-switched* mixture (Traditional Chinese +
English), not a single repeated token — a qualitatively different failure from the
super-weight collapse. That contrast between neuron-ablation failure mode and
super-weight failure mode is itself publishable and costs one extra experimental arm.

### Rejected framing

**"Super weights are language-specific neurons at finer grain."** Reject. Yu's super
weight is a single scalar in `down_proj` whose removal destroys the model **in every
language including English**; LAPE neurons remove one language and leave others intact.
Fig. 2 in Tang is diagonal; a super weight ablation would be a full row *and* column.
These are different objects and conflating them walks straight into the occupied
territory the project is supposed to avoid.

---

## §4 Open problems sized for one semester

Ordered by ratio of (defensibility × novelty) to cost. Models: NLLB-600M/3.3B,
TowerBase/Instruct-7B, EuroLLM-9B, Aya-Expanse-8B. Metrics: chrF++/COMET on FLORES-200,
KL on forced decode, output language-ID.

### P1. The per-language super-weight damage profile, and what explains it

*Problem.* Zeroing one scalar collapses MT quality; is the collapse uniform across
languages, and is the non-uniformity explained by anything measurable?

*Why open.* No super-weight × multilingual paper exists (searched). The in-repo
EuroLLM result (57.9 → 4.8 chrF++) is n-small and single-model. Per-language *PTQ*
degradation is documented (`UnevenPTQ-2025`, `Marchisio-2024`, sibling folder), so a
ready covariate exists that nobody has connected to single-weight ablation.

*Experiment.* ≥20 FLORES languages spanning resource levels and scripts × 4 model
families × {intact, super-weight zeroed, magnitude-matched random `down_proj` weight
zeroed}. Score chrF++, COMET, and forced-decode KL. Spearman the per-language drop
against per-language PTQ drop, pretraining token share, and typological distance.

*Counts as an answer.* A per-language damage profile with 95% CIs, a stated family size
and multiplicity correction, a demonstrated separation from the random-weight control,
and either a covariate that predicts the ordering (with ρ and CI) or a bounded "no
relationship" interval. Both directions are publishable; the "no relationship" version
must be reported as an interval, never as "no effect."

*Confounds.* chrF++ floors under collapse — carry KL as the graded measure. COMET is
itself a multilingual model with uneven per-language reliability, so a per-language
COMET drop partly measures COMET; report chrF++ as primary. Instruct vs base models
differ in off-target propensity (Marchisio) — do not pool them.

### P2. Does NLLB have super weights at all, and where?

*Problem.* Run Yu's detection on an encoder-decoder MT model, separately per stack, and
test whether the decoder's massive activation sits on the target-language tag position.

*Why open.* Every super-weight result to date is decoder-only. The one NLLB "sinks"
paper measured cross-attention only, is unreviewed, and contradicts itself on the
language-tag number. Nobody has looked at NLLB's residual stream.

*Experiment.* Yu's one-forward-pass detection on encoder and decoder of NLLB-600M and
3.3B. Per-position residual norms and decoder self-attention sink rates. Then ablate the
candidate scalar(s) and measure chrF++ + output language-ID across directions. Control:
replace the language tag with an arbitrary token and re-measure, to separate positional
from tag-identity effects.

*Counts as an answer.* Either a located super weight per stack with an ablation effect
size, or a bounded negative ("no single scalar in NLLB-600M raises FLORES chrF++ loss
above X") — which would itself be a genuinely interesting architecture contrast, since
NLLB-600M is a distilled model and distillation may not reproduce the phenomenon.

*Confounds.* NLLB-600M is *distilled*, so absence of a super weight may reflect
distillation rather than encoder-decoder architecture — this is why 3.3B must also be
run. Encoder and decoder have different depths and roles; do not average across stacks.

### P3. The identity of the post-ablation attractor

*Problem.* Is the degenerate repeated token the target-language unigram prior, a
training-repetition artifact, or a fixed direction?

*Why open.* The "We. We. We." observation is in-repo and unexplained. No paper connects
super-weight ablation to degeneration mechanism, and the standard degeneration account
(NeurIPS 2023) is monolingual.

*Experiment.* Ablate; generate under forced target language across ≥10 languages;
compute generated-token unigram distributions; rank-correlate against corpus unigram
frequency and against cross-language agreement. Add the contrast arm: same protocol
under **LAPE-neuron** ablation, to show the two failure modes differ (Tang's Table 2
shows code-switching, not repetition).

*Counts as an answer.* A correlation with CI between attractor identity and the
language's unigram prior, plus a demonstrated qualitative difference from neuron
ablation. Cheap, self-contained, and it directly supports the "single weights are a
different object from language neurons" positioning the project needs.

*Confounds.* Decoding strategy determines repetition — greedy vs sampling vs repetition
penalty will change the attractor. Fix and report the decoding config (repo rule 6:
config ≠ what ran). Tokenizer differences across models make "the same token" ambiguous;
compare at the string level too.

### P4. Source contribution under ablation: severed conditioning or broken decode-out?

*Problem.* Does super-weight ablation reduce ALTI+ source contribution, or preserve it
while output degenerates?

*Why open.* Bafna's translation-barrier framing and Dale's source-contribution detector
have never been pointed at a parameter-level intervention.

*Experiment.* ALTI+ source contribution and TNG on NLLB (and Tower where ALTI+ applies),
intact vs ablated, across resource levels; compare to the HalOmi-annotated distributions.

*Counts as an answer.* A stated verdict with effect sizes: source contribution collapses
(conditioning severed) or holds (decode-out broken). Either result connects the super
weight to an established MT failure taxonomy.

*Confounds.* ALTI+ was validated on naturally-occurring hallucinations, not on models
with a zeroed parameter — the measure's calibration under intervention is unestablished
and should be stated as a limitation, not assumed away.

### Sizing

P3 is the cheapest and should be run first as a warm-up that produces a real result. P1
is the headline and consumes most of the compute. P2 is the highest-variance and highest
-payoff. P4 is a measurement layer that P1–P3 all benefit from. Attempting all four at
full scale exceeds 125–150 hours; **P1 + P3 is a coherent semester**, with P2 as the
stretch and P4 folded in as instrumentation.

---

## §5 What the proposal wording should change

1. **Delete or re-source "as few as four neurons (Zhao 2024)."** Grepped the full text:
   the paper's claim is **0.13% of all neurons**, and "four neurons" appears nowhere.
2. **Reverse the NLLB sink premise.** Do not write that sinks sit on language tags. The
   only paper making the measurement reports `</s>` at 78–87% and language tags at
   1.5–2.0%, and it is an unreviewed preprint that contradicts its own Table 1 in §6.2.
   Rebuild the NLLB question on the architecture instead — the target language token *is*
   decoder position 0 (NLLB tech report §8) — which is a stronger and verifiable footing.
3. **Position explicitly against `Language Lives in Sparse Dimensions` (EACL 2026).**
   This is peer-reviewed, current, and claims sparse *dimensions* at consistent indices
   causally control output language, beating neuron methods. The project's distinction is
   real but must be stated: a single scalar **weight** in `mlp.down_proj`, whose ablation
   destroys the model in *all* languages, versus residual **dimensions** whose
   manipulation *switches* language while preserving semantics. Different object,
   different effect. Say so, or a reviewer will say it first.
4. **State the Mondal caution at its actual scope.** Mondal shows neuron interventions
   do not improve *cross-lingual transfer on XNLI/XQuAD*. It does not show
   language-specific neurons fail to control output language — Tang, Kojima and Zhao
   show they do. Over-citing Mondal weakens the project's own ablation logic.
5. **Add Rajaee & Pilehvar as a stated risk.** "mBERT exhibits no outlier dimension"
   is the most likely reviewer objection to a multilingual outlier story. Pre-empt it,
   note the construct and scale differences, and treat it as a hypothesis under test.
6. **Add Timkey & van Schijndel as the methodological floor.** Any "this channel is
   where language lives" claim needs behavioural evidence, because rogue dimensions
   dominate similarity measures while mismatching what drives behaviour.
7. **Stop describing per-language results from small n as profiles.** HalOmi shows
   single-pair conclusions do not survive scale. The EuroLLM 57.9 → 4.8 figure is a
   motivating observation, not a per-language finding, and should be labelled as such
   until ≥20 languages are run.
8. **Name the gap positively and precisely.** After searching for super-weight ×
   multilingual and super-weight × translation: **no such work exists.** Every
   super-weight result is decoder-only and English-evaluated; every multilingual
   localisation result is neuron- or dimension-level. The project sits in a real hole —
   say that, with the search stated, rather than claiming general novelty.
9. **Do not assume the LLaMA layer profile.** Trinley (Aya-23, final layers) and
   Harrasse (early linear encoding, late decoding features) both break the U-shape for
   balanced-data models. Aya-Expanse is a target model; state the profile as measured,
   per model.

## Verification debt

- `phenomenon/Puccetti-2022-Outlier-Dimensions-Driven-by-Frequency.pdf` (sibling folder) is
  `[unopened]` by me; the frequency prediction in §3(i) rests on the in-repo
  characterisation of it, not on my reading. Confirm before citing.
- `[unopened]`, found in search listings, never opened, **do not cite**: arXiv:2605.08504
  (*A Single Layer to Explain Them All* — "massive emergence layer"; relevant to onset in
  §3(i) and would fit ``); arXiv:2507.22581
  (*Unveiling the Influence of Amplifying Language-Specific Neurons*); arXiv:2601.16390
  (*Cross-Lingual Activation Steering*); arXiv:2508.17078 (*Linguistic Neuron Overlap
  Patterns*); arXiv:2602.22453 (*Retrieval-Transition Heads*); arXiv:2505.21458 (*Do LLMs
  Need to Think in One Language?*); arXiv:2508.02256 (*Interference Matrix*).
- Venue for `Wang-2024-Sharing-Matters` (arXiv:2406.09265) is unconfirmed — arXiv carries
  no comment or journal_ref. Treat as preprint.
- I did not open the bulk of the NLLB tech report (190 pp) — only §8 via targeted grep.
