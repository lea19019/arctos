# Notes — active problems in speech translation and speech models (2025–2026)

**Written 2026-09-06** by the "Problems" agent of the speech-translation research arm.
Companion notes by sibling agents: *Landscape* and *Interpretability* (same `docs/literature/papers_index.md`
section, different sub-headings).

**Question this file answers.** Where, in speech translation and speech models, is there an
*active, tractable* problem — one the field currently names, with numbers attached — on which
this project's interpretability finding (the "built-in constant": an early-layer FFN neuron
writing a ~1000× massive activation into a few residual channels, persisting to the last layer,
acting as a fixed bias, and used by attention heads as a sink) could give a foothold that a
non-interpretability project could not?

**Reading status legend.** Every claim below is tagged:
- `[read]` — I read the relevant section of the PDF and the number is quoted from it.
- `[abstract]` — I read only the abstract/intro; the number is the authors' own headline claim.
- `[unopened]` — downloaded, indexed, not read. **Nothing load-bearing rests on these.**
- `[search-listing]` — came from a search-result snippet, PDF not consulted. **Confirm before citing.**

**Venue status.** Per the coordinator's instruction, venues are marked:
- `VERIFIED` — I saw the proceedings page or publisher record.
- `author-declared` — from the arXiv `Comments:` field; the authors say it is accepted there. Not
  independently confirmed against proceedings.
- `preprint` — arXiv-only, no venue stated. Treat as unrefereed.

**A caution about dating.** A large share of the most on-point work is from 2026 and is
arXiv-only. The hallucination-mechanism literature in speech is roughly 18 months old and is
moving fast; several of the papers below cite each other within months. This is a fast-moving,
lightly-refereed area — good for finding an open question, bad for assuming anything is settled.

**Not consulted.** `/home/adrian/Repos/arctos/speech-translation/` exists in this repo but the
brief marks it old and untrusted; I did not read it. Whether the lab's actual ToAll pipeline
matches what is in there is unknown to me and is listed in §5.

---

## §1 — Paper by paper

### 1.1 Hallucination and repetition: that it happens, and how much

**`Koenecke-2024-Careless-Whisper.pdf`** — Koenecke, Choi, Mei, Schellmann, Sloane. ACM FAccT
2024 (`10.1145/3630106.3658996`). `VERIFIED` (ACM DL record and FAccT proceedings PDF both seen).
`[read: abstract + results summary]` The canonical harms paper. ~1% of Whisper transcriptions of
aphasia-speaker audio contain an entirely hallucinated phrase or sentence with no counterpart in
the audio; **38% of those hallucinations carry explicit harm** (fabricated violence, invented
authority, false associations). Hallucination rate is higher for speakers with speech impairments
and correlates with longer non-vocal spans. **Observational**, with a controlled speaker-group
comparison. This is the paper that makes the problem a *safety* problem rather than a quality one,
and it is the most citable single fact in the area.

**`Frieske-2024-Hallucinations-Neural-ASR.pdf`** — Frieske & Shi. `preprint` (arXiv 2401.01572;
no venue on the arXiv record; a search snippet asserted Interspeech 2024 but I could not confirm
it — see §5). `[abstract]` Defines ASR hallucination as output "semantically unrelated to the
source utterance, yet still fluent and coherent," and proposes a *perturbation-based* test that
identifies hallucination-prone models without access to training data. Distinguishes models with
similar WER but very different hallucination propensity. **Causal in method** (controlled input
perturbation), observational in claim.

**`Lost-in-Transcription-2025-Hallucination-Speech-Foundation-Models.pdf`** — Atwany, Waheed,
Singh, Choudhury, Raj (CMU / MBZUAI). ACL 2025, `author-declared` ("ACL2025 camera-ready").
`[read: abstract]` Introduces **hallucination error rate (HER)** and applies it to **over 20 ASR
models**. Three results: (1) high WER can mask a low hallucination rate and low WER can conceal
dangerous hallucination — the two metrics are not proxies for each other; (2) synthetic noise
(white noise, pitch shift, time stretch, adversarial) raises HER; (3) **distribution shift
correlates with HER at α = 0.91**. **Observational + controlled perturbation.** This is the paper
to cite for "WER is the wrong instrument," which matters because it is exactly the argument a
mechanistic project needs to justify measuring something other than WER.

**`HALAS-2026-Human-Annotated-ASR-Hallucinations.pdf`** — Barański, Jasiński, Bartolewska,
Witkowski, Kowalczyk (AGH Kraków). Interspeech 2026, `author-declared`. `[read: abstract + intro]`
First **human-annotated** dataset of *naturally occurring* hallucinations, from seven SOTA ASR
models on unprocessed Earnings-22 calls, with span-level labels. Two findings that matter here:
hallucinations occur **even on almost-correctly-transcribed speech (low WER)**, and there is
**strong cross-model vocabulary overlap** in what gets hallucinated. Benchmark result:
character/semantic proxy metrics reach 81% ROC-AUC, but **state-of-the-art detection methods reach
only 53.1% F1**. **Observational.** Critically, the authors' framing is that all prior mitigation
work was evaluated on *non-speech or artificially corrupted* audio — so the entire mitigation
literature below has an ecological-validity problem that HALAS is the first instrument to expose.

**`Radford-2022-Whisper-Robust-Speech-Recognition.pdf`** — Radford et al. `preprint` on arXiv
(no venue stated on the arXiv record; widely cited as ICML 2023, **not verified by me** — §5).
`[read: long-form decoding section, via the HF/whisper.cpp documentation trail]` The origin of the
deployed mitigations: temperature fallback over {0, 0.2, …, 1.0}, a gzip **compression-ratio
threshold of 2.4** above which a segment is discarded and re-decoded hotter, and a log-probability
threshold. These are *symptom* heuristics — the compression ratio is literally a repetition
detector — and their existence is the field's admission that nobody knows the cause.

### 1.2 Hallucination and repetition: mechanism

**`Wu-2025-Beyond-Transcription-Mech-Interp-ASR.pdf`** — Glazer, Segal-Feldman, Segev, Shamsian,
Buchnick, Hetz, Fetaya, Keshet, Navon (aiOla Research). `preprint` (arXiv 2508.15882; no venue
stated). `[read: §4.5 repetition, §4.6 encoder lens, discussion]` **The single most important
paper for this project.** Adapts logit lens, linear probing and activation patching to Whisper and
Qwen2-Audio. On a 102-utterance multilingual repetition-prone set from CommonVoice 16.1 (Japanese
+ English), they patch and ablate cross-attention, self-attention and FFN outputs at each of
Whisper's 32 decoder layers:

- **Patching cross-attention at layer 23 resolves 76% of repetition cases**; layer 18 adds 13%.
- **A single head — layer 18, head 13 — suppresses repetition by 78.1% alone**, out of 640 heads.
- Layer 23 plus that head account for **89%** of corrected examples.
- **"Self-attention and fully-connected interventions had no measurable effect."**

They also run an *encoder lens* (truncate the encoder, decode from an intermediate layer): Whisper
encoder layers 20–27 emit fluent-but-ungrounded language-model-like completions; **from layer 27
the model falls into repetition loops in ~60% of 400 samples, worst at layer 30**, resolving only
at layers 31–32. **Causal** (patching + ablation) for the decoder claim; observational for the
encoder lens. See §3 — that "no measurable effect" line is the strongest single piece of evidence
*against* the most obvious bridge from this project.

**`Wang-2025-Calm-Whisper-Crazy-Heads.pdf`** — Wang, Alhmoud, Alsahly, Alqurishi, Ravanelli (Elm
Company / Concordia / Mila). **Interspeech 2025**, `VERIFIED` (ISCA archive,
`interspeech_2025/wang25b_interspeech.html`). `[read: abstract + intro]` Head-wise masking of the
20 **self-attention** heads in the Whisper-large-v3 decoder finds **3 of 20 heads account for over
75% of hallucinations** on UrbanSound. Fine-tuning only those three on non-speech data gives
**>80% reduction in non-speech hallucination at <0.1% WER cost** on LibriSpeech test-clean/other.
**Causal** (masking, then targeted fine-tuning). Note the apparent tension with Glazer: Wang finds
self-attention heads causal for *non-speech* hallucination; Glazer finds self-attention irrelevant
for *repetition on speech*. These are likely two different failure modes wearing one name — which
is itself a usable research observation (§3).

**`Whisper-2026-Hallucination-SAE-Steering.pdf`** — Aparin, Popov, Sadekova, Yermekova (MISIS /
HSE). `preprint` (arXiv 2606.07473; no venue stated). `[read: abstract + intro]` Trains sparse
autoencoders on **Whisper encoder** activations. Hallucination-related information is **linearly
separable** in both raw activations and SAE latents, concentrated in a **sparse feature subset**,
and **discriminability increases toward deeper encoder layers**. SAE latent-space steering cuts
hallucination rate **72.63% → 14.11%** (Whisper-small) and **86.88% → 27.33%** (large-v3) on a
non-speech test set, at small WER cost, without touching parameters. **Causal** (steering).
Directly relevant: this is the "sparse feature subset in a speech encoder" result, and it is the
nearest existing thing to a superweight-style account — but it is framed as SAE features, not as
weights, and does not look for magnitude outliers.

**`TextMetrics-2026-Whisper-Hallucination-Model-Internals.pdf`** — Jasiński, Barański, et al. (AGH
Kraków). Interspeech 2026, `author-declared`. `[read: abstract]` Companion to HALAS. Compares
text-metric, LLM-judge and **internal decoder-state probing** detection of Whisper-large-v3
hallucination on the human-annotated real-speech data. **Probing the decoder's internal states
wins**, needs no reference transcript, and "hallucination traits are encoded across intermediate
decoding layers"; a late-fusion meta-classifier is best overall. **Observational.** Important
because it establishes that the internals carry a *reference-free* signal on *real* speech, not
just on synthetic non-speech.

**`NullToken-2026-Message-Free-Hallucination-ASR-NMT.pdf`** — Borodin, Kudryavtsev, Viakhirev,
Mkrtchian (Lab260 / MTUCI / ITMO). `preprint` (arXiv 2608.15940; comment says "Submitted to
AAAI-27"). `[read: abstract + intro]` Studies the **reserved null token** (EOT in Whisper, EOS in
NMT, blank in CTC/RNN-T) as an abstention signal, **across both ASR and NMT in one framework** —
a 15-model zoo plus a locked Whisper-small evaluation. Findings: frozen decoder states **linearly
separate non-speech from speech after the very first decoder block**; native null-token scores
distinguish the conditions even when the null token does not win the argmax; raising the
null-token score sharply suppresses fabrication **but also deletes valid speech and shortens
legitimate translations**; and the failure **does not vanish with scale** — non-monotonic across
six Whisper checkpoints even as WER falls. **Causal** (readout edits) with an explicitly stated
suppression/deletion trade-off. The ASR+NMT joint framing is unusual and is the closest published
thing to the cascade question in §3.

**`Dispersion-Attraction-2026-Spectral-Dynamics-Whisper-Hallucination.pdf`** — Viakhirev, Borodin,
Mkrtchian. `preprint` (arXiv 2604.08591). `[read: abstract + intro]` Proposes a "Spectral
Sensitivity Theorem" predicting a phase transition from a dispersive regime to a rank-1 attractor
regime, and validates it on Whisper Tiny → Large-v3-Turbo under adversarial stress: intermediate
models show a **13.4% collapse in cross-attention rank**; large models instead show self-attention
**compressing rank by −2.34%** and hardening the spectral slope, "decoupling the model from
acoustic evidence." **Observational + theoretical**; the theory is the authors' own and unreplicated.
Treat the specific numbers with caution, but the *shape* of the claim — that large Whisper
hallucination is an attractor in the self-attention geometry — is a directly testable rival
hypothesis to the sink account.

**`SpeechLLM-2026-Detecting-Hallucinations-Attention-Maps.pdf`** — Waldendorf (Edinburgh), Awwad
Shiekh Hasan, Tsymbalov (Amazon AGI). **Findings of ACL 2026**, `author-declared`. `[read:
abstract]` Four attention-derived metrics (AudioRatio, AudioConsistency, AudioEntropy,
TextEntropy) feeding a logistic-regression detector on Qwen-2-Audio and Voxtral-3B, across ASR and
speech-to-text translation. Up to **+0.23 PR-AUC** over uncertainty and prior attention baselines
in-domain; **~100 attention heads suffice**, and using fewer heads *improves* out-of-domain
generalisation. Explicitly notes SpeechLLM attention dynamics differ from text LLMs because audio
token sequences are much longer and align differently. **Observational.**

**`Listen-Like-a-Teacher-2025-Whisper-Hallucination-Mitigation.pdf`** — Tripathi, Menon, Gaurav,
Gohil, Wasnik (Sony Research India). `preprint` (arXiv 2511.14219; AAAI-2026 copyright block in
the PDF, acceptance not confirmed). `[read: abstract]` Adaptive Layer Attention groups Whisper
encoder layers into correlated blocks and fuses them; a multi-objective KD stage aligns a
noisy-input student to a clean-input teacher. Reduces hallucination and WER on noisy benchmarks
while preserving clean performance. **Causal** (architecture + training intervention). Notable
for the framing they inherit from Atwany et al.: hallucinations arise from **misaligned internal
representations in both encoder and decoder**.

**`Whisper-CD-2026-Contrastive-Decoding-LongForm.pdf`** — `preprint` (arXiv 2603.06193).
`[read: abstract + intro]` Training-free contrastive decoding: contrast clean-audio logits against
negatives from Gaussian noise, silence, and temporal shift, aggregated by log-sum-exp. **Up to
24.3 pp WER reduction on CORAAL**, 48% faster than beam search. Names the three long-form failure
patterns cleanly: silence-region hallucination, **repetition loops that carry over between
segments**, and content omission. **Causal** (decoding intervention).

**`Xu-2026-LoopGuard-Self-Reinforcing-Attention-Loops.pdf`** — `preprint` (arXiv 2604.10044).
`[read: abstract]` Text-LLM, not speech, but mechanistically the closest analogue: repetition
collapse is driven by **collapsed attention patterns where a subset of heads locks onto a narrow
suffix**, and is *stabilised by KV-cache reuse* because attention-based cache-importance scores
give spuriously high scores to the repeating tokens — cache management amplifies the loop.
LoopBench + a plug-in guard cut loop incidence by **>90 percentage points**. **Causal.** Relevant
as the mechanism-of-repetition baseline any speech claim must beat.

**`AttentionSinks-2026-Internal-Signals-Hallucination-Detection.pdf`** — `preprint` (arXiv
2604.10697). `[abstract]` Text LLMs: sink-score features detect hallucination, and the authors
interpret this as anomalies in attention flow — disproportionate attention to non-informative
tokens — associating with hallucinated output. **Observational.** The text-side precedent for
"sink statistics carry hallucination information," which the project's proposal already flags as a
stretch goal for NLLB.

### 1.3 Outlier / sink / massive-activation structure in speech models

**`Outlier-Reduction-2024-Gated-Attention-PTQ-Speech-Foundation.pdf`** — Wagner, Baumann,
Riedhammer, Bocklet (TH Nürnberg / Intel Labs). **Interspeech 2024**, `author-declared` ("Accepted
at Interspeech 2024"). `[read: §3.4, §3.5]` **The most important find of this sweep for the
project's own phenomenon.** Extends the LM/ViT outlier literature to Whisper and shows it
transfers:

- In the **31st decoder layer of Whisper-large (1.55B)**, an attention head "predominantly
  allocates its probability mass to the transcription token `<|tr|>`, while the same token has
  small values associated with it in V" — so the product is near-zero and the head performs
  **little or no update to the hidden representation**. "Similar patterns can be found across all
  decoder layers and attention heads." *That is an attention sink, in Whisper's own decoder, on a
  task-prompt special token rather than on BOS.*
- Outliers are counted as values beyond **six standard deviations** from the tensor mean, measured
  by kurtosis and ‖·‖∞ averaged over attention-layer outputs.
- In a distilled student's last decoder layer, **hidden dimensions #819 and #1054 alone contribute
  ~15% of all outliers**; gated attention flattens this (top two become #287 and #211 with smaller
  shares) and enables effective **INT8 weight+activation** quantization with lower WER.

**Causal** (the gating intervention). Crucially, the sink observation is presented as an
*incidental* motivating figure, not as the object of study — nobody has followed it up.

**`Cappellazzo-2025-Attention-Sinks-Massive-Activations-AVSR.pdf`** — Anand, Cappellazzo, Petridis,
Pantic (UBC / Imperial). **IEEE ICASSP 2026**, `author-declared`. `[read: abstract, §1, §3.1]`
Self-described **first** study of attention sinks and massive activations in multimodal speech
recognition. On Llama-AVSR (AV-HuBERT video encoder + **Whisper audio encoder** + a Llama decoder,
LoRA-tuned):

- Sinks appear at **BOS and at intermediate low-semantic tokens** across ASR, VSR and AVSR. The
  BOS sink pre-exists in the LLM; **intermediate sinks emerge during fine-tuning**.
- Massive activations "originate in the MLP layers and correspond to **fixed feature indices**
  across all sink tokens," appearing **as early as layer 2**, with magnitudes "up to four orders
  larger than the median."
- Intermediate sink hidden states have **high cosine similarity with the BOS hidden state**; that
  alignment is what propagates the massive activation to them.
- A **decorrelation loss** reducing BOS–token cosine similarity mitigates both, improving WER by
  up to **11.11% (VSR)** and **1.42% (ASR)** — and the gain is concentrated **under high
  audio-visual feature downsampling**, i.e. high token compression.

**Causal.** Read carefully: **the massive activations are found in the Llama decoder, not inside
Whisper.** Whisper here is a frozen feature extractor. So this paper confirms the project's
phenomenon survives contact with speech *input*, in an LLM; it does **not** establish that Whisper
itself has one.

**`OmniLLM-2026-Nature-of-Attention-Sink-Decoding.pdf`** — Yoo, Jang, Chung (KAIST / Oxford VGG).
`preprint` (arXiv 2603.14337). `[read: abstract + intro]` Omni-LLMs over video+audio+text. Two
findings: high sink attention **does not** simply mean head redundancy, and **"the sink value
vector acts as a shared bias added to every token's output, serving as a global signal that
organises the representation as a whole."** Their OutRo method aligns non-sink representations
with the sink and relaxes the causal mask for sink tokens at an early layer; consistent gains on
seven video-QA benchmarks at 1.1× decode overhead. **Causal.** This is an independent
multimodal restatement of the project's "acts as a fixed bias" finding, which is a useful
corroboration to cite.

**`CrossModal-2026-Information-Hubs-AudioVisual-LLMs.pdf`** — KAIST MM lab. `preprint` (arXiv
2605.10815). `[read: abstract]` In audio-visual LLMs, integrated cross-modal information is
**stored in sink tokens**, and a distinct subset ("cross-modal sink tokens") specialises in it.
Proposes a training-free hallucination mitigation that encourages reliance on those tokens.
**Causal.** Together with OmniLLM above, this is a small but coherent 2026 literature saying sink
tokens in multimodal models are *functional carriers*, not junk — directly relevant to the
project's open question "what does the constant do."

**`WnW-2026-Waxing-Waning-KV-Cache-Speech-LLMs.pdf`** — `preprint` (arXiv 2608.22704). `[read:
abstract]` For long-form audio in speech LLMs, **prefill attention concentrates near the audio
start — the authors name it an attention-sink effect** — while decode-time attention is broad, and
the two rankings **overlap weakly**. Their KV policy exploits the split. **Observational
mechanism, causal system.** This is evidence of a *positional* audio-side sink (first audio frame)
distinct from the BOS/prompt sink, which is exactly the "on which frame?" question in §3.

**`Anisotropy-2025-Neural-Representations-of-Speech.pdf`** — `preprint` (arXiv 2506.11096). Full
title: *Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on
Keyword Spotting*. `[read: abstract]` Estimates the anisotropy parameter for wav2vec2 (XLSR-53)
and finds that despite high baseline cosine similarity, wav2vec2 similarity still identifies words
without transcription. **Observational.** Weaker than I hoped: it is a keyword-spotting case
study, not a rogue-dimension audit. Included because it is the only speech-side anisotropy
measurement I found, and anisotropy is the observable that outlier dimensions produce.

**`LayerWise-2026-Probing-wav2vec2-Whisper-AAE.pdf`** — `preprint` (arXiv 2606.23948). `[abstract
— truncated in extraction]` Layer-wise probing of wav2vec 2.0 and Whisper for consonant-cluster
reduction in African American English. **Observational.** Listed as the methodological template
for layer-wise probing across a supervised/self-supervised speech pair.

**`Quantizing-Whisper-2025-Design-Choices.pdf`** — Söhler, Irigoyen, Kirkedal (CBS / Danske Bank /
Jabra). `preprint` (arXiv 2511.08093). `[read: abstract]` Cross-library PTQ sweep on Whisper-small
over PyTorch, Optimum-Quanto, HQQ, bitsandbytes. **Dynamic INT8 (Quanto) gives 57% size reduction
with WER *below* the FP baseline**; static quantization is worse ("likely due to the absence of
efficient low-bit implementations for LayerNorm and Softmax"); nf4/int3 reach 71% compression at
accuracy cost concentrated in acoustically hard conditions. **Observational.** Note what is
missing: no per-language breakdown, and no outlier analysis — it never connects its
static-vs-dynamic result to Wagner's outliers, though that is the obvious explanation.

### 1.4 Low-resource speech translation: what the field names as open

**`Survey-2025-ASR-African-LowResource-Challenges-Future.pdf`** — Imam, Sani, Gete, Ahamed, Ahmad,
Abdulmumin, Yimam, Bello, Muhammad (Bayero / HausaNLP / EthioNLP et al.). `preprint` (arXiv
2505.11690). `[read: abstract]` Survey naming the barriers: **data scarcity, linguistic complexity,
limited compute, acoustic variability, and bias/privacy ethics**; recommends community-driven
collection, self-supervised and multilingual learning, **lightweight architectures**, and
privacy-preserving techniques. **Survey.** Use it for the framing sentence, not for numbers.

**`AfriSwitch-2026-African-CodeSwitched-ASR-Benchmark.pdf`** — Ashungafac, Awobade, Olatunji
(Intron Health). `preprint` (arXiv 2608.26434; no venue stated). `[read: abstract + intro]`
**61.36 hours** of in-the-wild human-transcribed code-switched speech across **16 African
languages/varieties**, with switch-level English span tags, per-utterance Code-Mixing Index and
switch-point counts. Zero-shot benchmarking of five open and commercial multilingual systems:
**best system averages 35.93% WER and no system falls below 24% on any language** — "far above
published monolingual figures for the same languages." Mixing varies along **two largely
independent axes** (how often speakers alternate; how balanced the mixture is), so no single
scalar captures code-switchedness. **Africa-targeted training, not model scale or nominal language
coverage, best predicts performance.** **Observational.** (Note: a search snippet gave 54.41 h /
14 languages; the PDF says 61.36 h / 16 — the PDF wins. See §5.)

**`WAXAL-2026-African-Language-Speech-Corpus.pdf`** — Google Research Africa et al. `preprint`
(arXiv 2602.02734). `[read: abstract]` **24 languages, >100M speakers**: ~**1,250 h** transcribed
ASR speech plus **>235 h** single-speaker phonetically-balanced **TTS** recordings, CC-BY-4.0, on
HF as `google/WaxalNLP`. Includes Swahili, Yoruba, Igbo, Twi/Akan, Xhosa-adjacent varieties.
**Resource.** This materially changes what a one-semester project can afford — it is a free,
permissive, multi-language evaluation set in exactly the lab's languages.

**`WAXAL-NET-2026-Edge-ASR-19-African-Languages.pdf`** — Olufemi et al. (CMU Africa / LyngualLabs /
MBZUAI). `preprint` (arXiv 2606.02375). `[read: abstract]` Fine-tuned compact models reach
**38.0% macro-WER vs 64.9% for the best zero-shot baseline — a 26.9 pp reduction with models
3–40× smaller**. "Domain specialization dominates scale for spontaneous African speech."
**Empirical/observational.** Directly relevant to the compression arm: for these languages, small
specialised beats large generalist, so a compression study here is not academic.

**`Makerere-2025-Benchmarking-ASR-African-Languages.pdf`** — Nahabwe, Kagumire, Musinguzi, Beijuka, Kyagaba,
Nabende (Makerere). **PMLR 302, Deep Learning Indaba 2025** (`VERIFIED` from the PDF's own PMLR
proceedings header). `[read: header + abstract]` Systematic benchmark of ASR models on African
languages. **Observational.** Cited here as an accredited-venue anchor for the African ASR
benchmarking claim, since most of the rest of this subsection is arXiv-only.

**`NaijaS2ST-2026-MultiAccent-Nigerian-S2ST-Benchmark.pdf`** — Maltais, Jeon, Ma, Muhammad,
Abdulmumin, Mukhtar, Abolade, Okepefi, Sewedo, Adelani (Mila / McGill / Google DeepMind / HausaNLP
/ Masakhane). `preprint` (arXiv 2604.16287; comment literally says "Preprint"). `[read: abstract +
intro]` **Igbo, Hausa, Yorùbá and Nigerian Pidgin ↔ English**, ~50 h per language, benchmarked
across **cascaded, end-to-end and AudioLLM** paradigms. Findings: **audio LLMs with few-shot
examples are the strongest for speech-to-**text** translation; cascaded and AudioLLM are
comparable for speech-to-**speech**.** Motivating statistic: of ~2,000 languages in FLORES-101,
**0.85% are African**. **Observational benchmark.** This is the single best-matched external
benchmark to the lab's own language set.

**`Bemba-2025-Low-Resource-African-Speech-Translation.pdf`** — Al Farouq, Wassie, Moslem (Kreasof
AI / AIMS / ADAPT, TCD). **IWSLT 2025**, `VERIFIED` (arXiv journal-ref: *Proceedings of the 22nd
International Conference on Spoken Language Translation*). `[read: abstract]` Cascaded
**Whisper → NLLB-200** Bemba→English with back-translation augmentation filtered by cross-entropy.
**System paper.** Named here because the architecture *is* the lab's architecture, at an accredited
speech-translation venue — a ready-made comparison point.

**`Omnilingual-ASR-2025-1600-Languages.pdf`** — Meta / FAIR et al. `preprint` (arXiv 2511.09690;
no venue stated). `[read: abstract]` **1,600+ languages**, 500+ previously unsupported; a 7B
self-supervised model with an encoder-decoder head and an **"LLM-inspired decoder"** for zero-shot
generalisation, plus **300M variants for low-power devices**. Open-sourced. **Resource.** The
"LLM-inspired decoder" phrasing matters: if the constant is an artifact of LLM-style decoders, this
model is where it should appear in a speech system, and it is small enough at 300M to study on one
A100.

**`Pashto-2026-Zero-Shot-ASR-Script-Failure.pdf`** — Rahman (independent). `preprint` (arXiv
2604.04598; PDF is formatted for IEEE/ACM TASLP; acceptance not stated). `[read: abstract]` The
**language-confusion / off-target** evidence I was asked to look for, and it is stark:
- Zero-shot Whisper WER on Pashto ranges **90%–297%**, with **whisper-medium collapsing to 461%**
  on Common Voice 24 — the author attributes this **explicitly to decoder looping**.
- **No Whisper model produces Pashto-script output in more than 0.8% of utterances**, while
  MMS-1B, SeamlessM4T and OmniASR each exceed **93% script fidelity**.
- "WER alone does not reveal this failure, since a model generating Arabic-script output on Pashto
  audio has not achieved ASR in any interpretable sense."
- Fine-tuned models published at 14% WER degrade to **32.5–59%** out of distribution.
**Observational.** Read speech only. A WER of 461% is a repetition loop with a number attached —
this is the cleanest public instance of "language confusion and repetition loop are the same event."

**`Whisper-2026-Decoder-Inconsistencies-Dravidian-LowResource.pdf`** — Kumar, Tripathi, Wasnik
(Sony Research India). `preprint` (arXiv 2606.09535). `[read: abstract]` Dravidian languages have
longer words, higher vocabulary diversity and **lower word repetition** than Indo-Aryan, producing
sparse token distributions; baseline fine-tuning "reveals **decoder imbalance between
self-attention (linguistic context) and cross-attention (acoustic cues)**." They add
Weighted-Attention (adaptive fusion of the two) and Self-Conditioning; WER falls consistently.
**Causal** (architectural). The self- vs cross-attention *balance* framing is the same axis Glazer
found repetition living on, arrived at independently from a low-resource angle.

**`LLMDecoders-2026-Listen-Fairly-Bias-ASR.pdf`** — `preprint` (arXiv 2604.21276). `[read:
abstract]` **The most surprising paper in this sweep.** Nine models across three decoder
generations (CTC/no-LM, encoder-decoder/implicit-LM, LLM-decoder/explicit-LM), ~43,000 utterances,
five demographic axes, then 12 acoustic degradations × 216 inference runs. Headline numbers:
- **Under chunk masking, Whisper enters catastrophic repetition loops accounting for 86% of
  51,797 insertions**, while **explicit-LLM decoders produce 38× fewer insertions with near-zero
  repetition**.
- But **high-compression audio encoding (Q-former) reintroduces the repetition pathology even in
  LLM decoders**.
- **Silence injection amplifies Whisper's accent bias up to 4.64×** by triggering
  demographic-selective hallucination; Whisper-large-v3 shows a non-monotonic insertion-rate spike
  to **9.62%** on Indian-accented speech.
- Conclusion: "**audio encoder design, not LLM scaling, is the primary lever**."
**Observational at large scale, with controlled degradation.** See §3 and the final summary.

### 1.5 Downloaded, indexed, not read

These are in `papers/` and in the README index, and **nothing above depends on them**:
`Seamless-2023-Expressive-Streaming-Speech-Translation.pdf` `[unopened]`,
`MURMUR-2026-Efficient-LongForm-ASR.pdf` `[abstract]`,
`Phonology-Guided-2024-S2ST-African-Languages.pdf` `[abstract]`,
`CodeSwitch-2026-Benchmarking-Commercial-ASR.pdf` `[unopened]`,
`BENYO-2025-English-Yoruba-Direct-S2ST-Corpus.pdf` `[abstract]`,
`Swahili-2026-Continued-Pretraining-LowResource-ASR.pdf` `[unopened]`.

---

## §2 — Active problems, with evidence

Every number in this table is sourced to a `[read]` or `[abstract]` line in §1.

| # | Problem | Evidence it exists (numbers) | Who is working on it | What is missing |
|---|---|---|---|---|
| P1 | **Whisper hallucinates on non-speech / silence**, fabricating fluent text | 72.63%→14.11% (small) and 86.88%→27.33% (large-v3) hallucination rate before/after steering on a non-speech set (Aparin, preprint); ~1% of aphasia-speaker transcripts fully hallucinated, **38% harmful** (Koenecke, FAccT 2024) | Elm/Concordia (Calm-Whisper, Interspeech 2025); MISIS/HSE (SAE steering); Sony (ALA+KD); Lab260 (null token) | Mitigations are all evaluated on **non-speech or artificially corrupted audio** (HALAS, Interspeech 2026). No account of *why* the model prefers fabrication over EOT. |
| P2 | **Repetition loops in long-form ASR**, carried across segments | Whisper contributes **86% of 51,797 insertions** under chunk masking vs 38× fewer for LLM decoders (arXiv 2604.21276); ~60% of 400 samples loop at encoder layer 27–30 under encoder lens (Glazer, preprint); Pashto WER **461%** attributed to decoder looping (arXiv 2604.04598) | aiOla (mechanism); Whisper-CD, LoopGuard (decoding/KV); OpenAI's own compression-ratio 2.4 + temperature fallback | **The best mechanistic account is one unrefereed preprint.** Cause localised to cross-attention L23 + L18H13 (89% of cases) — but *why those*, and whether it generalises past 102 Japanese/English utterances, is unknown. |
| P3 | **Has the early-layer massive-activation / attention-sink phenomenon been reported in speech models at all?** | **Partially, and thinly.** *Inside Whisper:* Wagner (Interspeech 2024) shows decoder L31 heads dumping probability mass on `<\|tr\|>` with near-zero V (a no-op sink), and dims **#819/#1054 = ~15% of all outliers** — but as an incidental figure. *Outside Whisper:* Cappellazzo (ICASSP 2026) finds massive activations from **layer 2 of the Llama decoder** at **fixed feature indices**, ~4 orders above median, in an AVSR system where Whisper is only a frozen encoder. Multimodal sinks act as **a shared bias on every token** (Yoo, preprint) and as **cross-modal information hubs** (arXiv 2605.10815). WnW (preprint) names a **positional audio-start sink** in prefill. | Wagner et al.; Cappellazzo/Petridis/Pantic; KAIST MM lab | **No study asks whether Whisper's own encoder or decoder builds the constant** — which layer, which neuron, which weights, whether set-to-mean is harmless and set-to-zero catastrophic. No superweight-style causal weight ablation exists for any speech model. My arXiv full-text probe for `"super weight" OR "outlier dimension" OR "outlier channel"` ∧ speech/audio returned nothing, **but the control query also returned nothing, so this is unverified (§5).** |
| P4 | **Repetition/hallucination is worse under high audio-token compression** | Cappellazzo's decorrelation loss helps **most under high downsampling** (up to 11.11% VSR / 1.42% ASR WER gain); Q-former high-compression encoding **reintroduces repetition even in LLM decoders** (arXiv 2604.21276) | Two groups, independently, neither citing the other | Nobody has connected these. **No causal test that token compression *creates* the sink/massive-activation pathology.** |
| P5 | **Language confusion / off-target / script failure in speech models** | **No Whisper model emits Pashto script in >0.8% of utterances**, vs >93% for MMS-1B/SeamlessM4T/OmniASR (arXiv 2604.04598); repetition is triggered by code-switching (Glazer, preprint); best code-switched African ASR **35.93% WER, none below 24%** (AfriSwitch, preprint) | Rahman (Pashto); Intron Health (AfriSwitch); the LLM off-target literature (text-side) | Documented but **not mechanistic**. Whisper's language token sits **adjacent to the sink token** Wagner found — nobody has asked whether they interact. |
| P6 | **Code-switching in African ASR** | 61.36 h / 16 languages; best zero-shot **35.93% WER**; mixing varies on **two independent axes**; Africa-targeted training predicts performance better than scale (AfriSwitch, preprint) | Intron Health; LyngualLabs; CMU Africa | Benchmarks now exist; **no mechanism, no mitigation**. |
| P7 | **Cascade (ASR→MT→TTS) vs end-to-end error propagation in low-resource S2ST** | Audio LLMs + few-shot best for S2**T**T; cascaded ≈ AudioLLM for S2**S**T on Igbo/Hausa/Yorùbá/Pidgin (NaijaS2ST, preprint); IWSLT 2025 Bemba systems are Whisper→NLLB cascades | Mila/McGill/Masakhane/HausaNLP; IWSLT low-resource track | **Nobody attributes a cascade failure to a stage.** When a Whisper→NLLB pipeline loops, is it ASR-side or MT-side? Undecided in the literature. |
| P8 | **Compression/deployment of speech models for low-resource languages** | Dynamic INT8 = 57% smaller at **WER below baseline**; static much worse; nf4/int3 = 71% at accuracy cost (arXiv 2511.08093). Gated attention enables W8A8 INT8 by shrinking outliers (Wagner, Interspeech 2024). Fine-tuned edge models: **38.0% vs 64.9% macro-WER, 3–40× smaller** (WAXAL-NET, preprint) | TH Nürnberg/Intel; CBS/Jabra; CMU Africa | **No per-language quantization breakdown for speech**, though the text-MT literature shows low-resource languages are hit hardest. Static-vs-dynamic gap is never explained by outliers, though that is the obvious cause. |
| P9 | **Evaluation instruments are wrong** | WER and HER **decorrelate**; distribution shift ↔ HER α=0.91 (Atwany, ACL 2025); hallucination occurs at low WER, SOTA detection only **53.1% F1** on real speech (HALAS, Interspeech 2026); script failure **invisible to WER** (arXiv 2604.04598) | CMU/MBZUAI; AGH Kraków | A reference-free internal-state detector works (TextMetrics, Interspeech 2026) but is **Whisper-large-v3, English, one domain**. |
| P10 | **Data scarcity for African languages** | >80% of African languages have **<5 h transcribed speech** (arXiv 2410.23323, quoting Joshi 2020); **0.85%** of FLORES-101 languages are African (NaijaS2ST) | WAXAL (1,250 h ASR + 235 h TTS, 24 langs); Omnilingual (1,600+); Masakhane; Intron | Improving fast. **This is now the *least* differentiated place for a one-semester project to contribute.** |

---

## §3 — The bridge to this project's phenomenon

The project's finding (`docs/proposal_draft_v4.md` §1, §3) is: one early-layer FFN neuron writes a
~1000×-median value into a few residual channels via a handful of `down_proj` weights; it persists
to the last layer; set-to-mean is harmless and set-to-zero is catastrophic (measured: OLMo-1B
×3667 ppl, Mistral-7B ×1430); attention heads sink on the token carrying it; ablating it produces
**loops of function words** ("We. We. We."). Below, each candidate bridge with what is known, what
is not, and the cheapest experiment on the lab's assets.

### 3.1 Does Whisper build the constant, and where?

**Known.** Wagner (Interspeech 2024) shows Whisper-large's decoder has *both* halves of the
signature: a head at layer 31 sinking probability mass onto the `<|tr|>` task token with near-zero
value vectors (a no-op update), and hidden dimensions (#819, #1054) carrying ~15% of all
six-sigma outliers. Cappellazzo (ICASSP 2026) shows the full massive-activation phenomenon —
MLP-origin, layer 2, fixed feature indices, ~4 orders above median — in a **Llama decoder fed by
Whisper**, not in Whisper.

**Not known.** Nobody has run the superweight protocol on Whisper. Specifically: is there a single
early-layer FFN neuron whose `down_proj` fan-out creates a persistent residual-stream constant?
Which layer? Is set-to-mean harmless and set-to-zero catastrophic (the project's decisive test)?
Does the encoder build one too, and if so **on which frame** — WnW's positional audio-start sink
suggests frame 0, which would be a genuine speech-specific analogue of "the first token."

**Cheapest experiment.** Whisper-large-v3 is 1.55B — this fits on one A100 with room to spare, and
the project already has a working detector and causal ablation harness (`notes.md`; run on 9 models
/ 21 coordinates). Load Whisper via HF `transformers`, run the existing magnitude-outlier detector
over `model.encoder.layers[*].fc2` and `model.decoder.layers[*].fc2` (Whisper's `down_proj`
equivalents), plot per-layer residual-stream max-|activation| against median for encoder frames and
decoder positions separately, then run set-to-zero / set-to-mean against **WER on a held-out set**
rather than perplexity. **Cost: ~10–20 A100-hours.** The metric substitution (WER for perplexity)
is the one real design decision and must be validated against a published Whisper WER number
first — the project already learned this lesson the expensive way (`proposal_draft_v4.md` §3: a
hand-written eval set under-estimated Llama-7B's collapse by 30×).

**Risk.** Whisper is encoder-decoder with cross-attention and a fixed 30 s input; the "first token"
that carries the constant in decoder-only LMs corresponds to nothing obvious. It may be the
`<|startoftranscript|>` prompt prefix, it may be encoder frame 0, it may be neither. A null result
is publishable within this project (RQ1's table has no encoder-decoder entry yet) but is a weaker
contribution than a positive one.

### 3.2 Do Whisper's repetition loops coincide with sink loss?

**This is the most attractive-sounding bridge and the evidence is currently against it.**

**Known.** Glazer et al. localised Whisper repetition to **cross-attention** — layer 23 patching
fixes 76% of cases, head 13 of layer 18 fixes 78.1% alone, together 89% — and state that
"**self-attention and fully-connected interventions had no measurable effect**." The project's
constant lives in an FFN and is consumed by *self*-attention sinks. If Glazer is right, the
project's mechanism is in the two components that demonstrably do **not** control Whisper's
repetition.

**But three things keep it alive.** (i) Glazer's interventions were *restorative* (patch clean
activations in / zero out), not *sink-targeted*; "ablating the FFN output at layer L" is not the
same experiment as "removing the constant," which is a rank-one edit to a specific neuron. (ii)
Wang et al. (Interspeech 2025) *did* find self-attention heads causal — 3 of 20 heads, >75% of
hallucinations — for **non-speech** hallucination. So the two papers may simply be studying two
failure modes: repetition-on-speech (cross-attention/grounding) and fabrication-on-silence
(self-attention/prior). Nobody has said this out loud. (iii) Glazer's encoder lens shows loops
appearing at encoder layers 27–30 in ~60% of samples and resolving at 31–32 — a *dynamic in the
encoder*, which neither their decoder patching nor the sink account addresses.

**Cheapest experiment.** Do not try to prove the sink causes repetition. Instead: **measure whether
the sink's attention mass collapses at loop onset.** Take Whisper-large-v3, a loop-inducing set
(silence, chunk-masked audio, and code-switched audio — all three are documented triggers),
instrument the attention to Wagner's `<|tr|>` sink token and the residual-stream magnitude of the
outlier channels at every decoding step, and align the time series to the first repeated token.
Two outcomes, both results: sink mass drops before onset (the project's story extends to speech),
or it does not (Glazer's cross-attention story stands and the project's phenomenon is confined to
self-attention). **Cost: ~15–25 A100-hours**, no training. This is the single best experiment in
this document: it is cheap, pre-registerable, has a real null, and both answers are worth writing.

**Risk.** Low. The main risk is that "sink mass" needs a definition robust to Whisper's short
prompt prefix, and that a correlation at onset does not establish causation — which the write-up
must say plainly ("our hypothesis", not "the mechanism").

### 3.3 Do speech *encoders* build a massive activation, and on which frame?

**Known.** WnW reports prefill attention concentrating near the audio start and names it an
attention-sink effect. Aparin et al. find hallucination linearly separable in Whisper **encoder**
activations, concentrated in a sparse feature subset, increasing with depth. Cappellazzo finds
massive activations at fixed feature indices — but in the LLM, not the speech encoder.

**Not known.** Whether wav2vec2 / HuBERT / the Whisper encoder / Omnilingual's 7B SSL encoder build
a magnitude outlier at a specific frame, and whether it is positional (frame 0) or content-triggered
(first voiced frame, first silence). This is a clean, unclaimed question. Note that these encoders
are **not** autoregressive and have no BOS, so a positive result would be a genuinely new datapoint
about *why* the constant forms — the strongest theoretical contribution available here.

**Cheapest experiment.** Forward passes only, no generation: run Whisper-encoder, wav2vec2-XLSR and
one Omnilingual 300M variant over a few hundred WAXAL utterances in the lab's languages, record
per-layer per-frame residual-stream ‖h‖∞ and the top-k channel indices, and test whether the
argmax frame is positionally fixed. **Cost: ~5–10 A100-hours.** This is the cheapest experiment in
the document.

**Risk.** Moderate: three encoders with three different layer-naming conventions is a fiddly
engineering afternoon, and "we found an outlier channel" without a causal ablation is only an
observation. Pair it with §3.1's set-to-zero/set-to-mean test to make it a result.

### 3.4 Is the cascade's repetition an ASR-side or MT-side failure?

**Known.** The lab ships a cascade (fine-tuned NLLB-200 + in-house TTS); IWSLT 2025's Bemba entry
is the same architecture. Guerreiro et al. (already in `papers/`) find oscillatory hallucination
dominant in NLLB/M2M. Borodin et al. is the only paper treating ASR and NMT abstention in one
frame, and finds the null-token signal in both.

**Not known.** Nobody attributes a cascade loop to a stage. Given the same audio, does Whisper emit
a loop that NLLB faithfully translates, or does Whisper emit clean text that NLLB loops on? These
have different fixes and the field has not measured the split.

**Cheapest experiment.** Build a loop-triggering audio set in the lab's languages (WAXAL gives
Swahili/Yoruba/Igbo/Twi audio free), run Whisper→NLLB, and classify every looped output by whether
the intermediate ASR text already looped. Report the split with a CI. Then, for the MT-side
failures only, check whether NLLB's sink statistics differ — the project's proposal already lists
this as a stretch goal (`proposal_draft_v4.md` §2). **Cost: ~10 A100-hours plus a day of pipeline
plumbing.**

**Risk.** Moderate-to-high on the *interpretability* half: this is primarily an error-attribution
study, and it only becomes an interpretability project if the MT-side half yields something. As a
standalone it is useful engineering with a modest research core. Its real value is that it is the
result the lab would most immediately use.

### 3.5 Does audio-token compression create the pathology?

**Known.** Two independent 2026 results point the same way and neither cites the other:
Cappellazzo's decorrelation loss helps **most under high audio downsampling**, and arXiv 2604.21276
finds high-compression Q-former encoding **reintroduces repetition even in LLM decoders that
otherwise show near-zero repetition**. Wagner shows outlier magnitude governs whether Whisper
survives INT8.

**Not known.** Whether compression *causes* sinks/massive activations, or whether both are
downstream of something else. No causal test exists.

**Cheapest experiment.** Take one AudioLLM (or Llama-AVSR, whose code is public), sweep the audio
downsampling factor, and at each setting measure (a) massive-activation magnitude at the fixed
feature indices, (b) intermediate-sink count, (c) repetition rate. A monotone three-way relation
would be a genuinely new finding connecting two literatures. **Cost: ~30–60 A100-hours**, more if
any fine-tuning is needed.

**Risk.** High for one semester: it needs a working AudioLLM training/fine-tuning setup, which is
the most infrastructure of anything here. Listed because the *idea* is the most novel in this
document, not because it is the most tractable.

---

## §4 — Ranked shortlist

Ranked by (open × tractable × serves Adrian's stated goals: a small real contribution he finds
interesting, interpretability skills, research-engineering skills). Cost is A100-hours of compute
plus rough wall-clock; the semester budget is ~125–150 h of *his* time, which is the binding
constraint, not GPU time.

---

### **S1 — Does Whisper's sink collapse at repetition onset?**

- **Question.** In Whisper-large-v3, does attention mass on the sink token Wagner identified
  (`<|tr|>`/prompt prefix), and the magnitude of its outlier residual channels, change
  systematically in the decoding steps immediately before a repetition loop begins?
- **Why open.** The only mechanistic account of Whisper repetition (Glazer, preprint) localises it
  to cross-attention and reports self-attention as inert — but never measures the sink. Wagner
  documents the sink but never looks at repetition. **The two papers are one measurement apart and
  nobody has taken it.**
- **Experiment.** Three documented loop triggers (silence/non-speech, chunk-masked audio,
  code-switched audio — the last from AfriSwitch or the lab's own languages). Instrument sink
  attention mass and outlier-channel magnitude per decoding step. Align time series to first
  repeated token. Compare against a matched non-looping control set. Pre-register the onset
  window.
- **What counts as an answer.** A signed effect with a 95% CI on sink-mass change in the
  k steps before onset vs matched controls, `effect [95% CI], n_seeds=N, family=M`, corrected for
  the multiplicity of the layer × head scan. **"No change, bounded at ±x"** is a full answer and
  should be pre-committed to as such.
- **Cost.** ~15–25 A100-hours; ~30 h of Adrian's time. No training.
- **Risk.** Low. Main hazard is defining "sink mass" robustly and resisting a causal reading of a
  correlational result.
- **Serves.** Interpretability (patching-adjacent instrumentation on a new architecture);
  research engineering (time-series alignment, pre-registration, multiplicity correction); and it
  is the *interesting* one — it is the project's own "We. We. We." finding, tested in the most
  famous speech failure mode in the field.

---

### **S2 — Run the superweight protocol on Whisper (and one speech encoder)**

- **Question.** Does Whisper build the constant? Which layer, which FFN neuron, which weights? Is
  set-to-mean harmless and set-to-zero catastrophic, as in decoder-only LMs?
- **Why open.** P3 above. Wagner saw the sink incidentally at Interspeech 2024 and never followed
  up; Cappellazzo found massive activations in the *LLM* of an AVSR stack, with Whisper frozen. No
  causal weight ablation exists for any speech model. The project's RQ1 table currently has **no
  encoder-decoder entry at all**, so this fills a hole the proposal already admits.
- **Experiment.** Reuse the existing detector + ablation harness on Whisper-large-v3 encoder and
  decoder `fc2` matrices; substitute WER-on-held-out-set for perplexity and **validate the metric
  against a published Whisper WER number before trusting any collapse ratio**; report the three
  controls the proposal already specifies.
- **What counts as an answer.** The same table row the project produces for every other model:
  coordinates, magnitude rank within the matrix, ×damage on set-to-zero, ×damage on set-to-mean,
  with the encoder and decoder reported separately. Negative ("Whisper has outlier channels but no
  single catastrophic scalar") is a result — three of the models already tested behave that way.
- **Cost.** ~10–20 A100-hours; ~25 h of Adrian's time.
- **Risk.** Low-to-moderate. The architectural mismatch (no "first token" in an encoder; fixed 30 s
  input) is real and may make the question ill-posed rather than answered — which must be reported
  honestly if so.
- **Serves.** Directly extends the existing proposal; maximum reuse of code already written and
  debugged; the most certain-to-produce-a-row option.

---

### **S3 — Whisper→NLLB cascade: attributing the loop to a stage, in the lab's languages**

- **Question.** When the lab's cascade produces a repetition loop on African-language audio, what
  fraction originates in Whisper's transcript versus in NLLB's translation of a clean transcript?
- **Why open.** P7. Cascade error propagation is universally *asserted* (every low-resource ST
  paper says it) and, as far as this sweep found, **never measured by stage for repetition
  specifically**. NaijaS2ST benchmarks paradigms end-to-end; it does not attribute failures.
- **Experiment.** WAXAL (free, CC-BY-4.0, 1,250 h, includes Swahili/Yoruba/Igbo/Twi) plus the
  lab's own Swahili recordings. Run Whisper→NLLB; classify each looped output by whether the
  intermediate ASR text already looped; report the split with a CI per language. Stretch: for the
  MT-side subset, compare NLLB sink statistics against matched non-looping inputs.
- **What counts as an answer.** A per-language ASR-side/MT-side split with CIs and stated coverage
  (which languages, how many utterances, which were dropped and why).
- **Cost.** ~10 A100-hours; ~35 h of Adrian's time, most of it pipeline plumbing.
- **Risk.** Moderate. The interpretability content is concentrated in the stretch half; without it
  this is careful engineering with a modest research core. Also: it uses the lab's fine-tuned NLLB,
  so the result may not generalise beyond the lab — state that.
- **Serves.** Research engineering strongly; the lab would use the answer immediately; interpretability
  only if the stretch half lands. Highest *usefulness*, lowest *novelty*.

---

### **S4 — Where does a speech encoder put its outlier, and on which frame?**

- **Question.** Do the Whisper encoder, wav2vec2-XLSR, and an Omnilingual 300M encoder each build a
  magnitude outlier at a fixed frame position, at fixed channel indices, and does it survive across
  languages?
- **Why open.** P3. Non-autoregressive encoders have no BOS and no "first token," so a positive
  result would be evidence about *why* the constant forms that no decoder-only study can supply.
  The cross-lingual half also feeds the project's RQ3 (is the constant shared across languages?)
  in a modality where nobody has asked.
- **Experiment.** Forward passes only over a few hundred WAXAL utterances across the lab's six
  languages. Record per-layer, per-frame ‖h‖∞ and top-k channel indices. Test positional fixity and
  cross-language channel overlap.
- **What counts as an answer.** For each encoder: the argmax frame's position distribution, the
  channel-index overlap across languages (with a permutation null), and whether outlier magnitude
  exceeds the random-init baseline — **step 0 / random init is the free null and must be reported**.
- **Cost.** ~5–10 A100-hours; ~20 h of Adrian's time. The cheapest item here.
- **Risk.** Moderate. Purely observational unless paired with S2's ablation; three encoders means
  three layer-naming conventions. "We found a channel" is not on its own a result.
- **Serves.** Interpretability and the project's cross-lingual RQ; excellent value per hour;
  natural companion to S2 rather than a standalone project.

---

### **S5 — Does audio-token compression create the sink pathology?**

- **Question.** As audio downsampling increases in an AudioLLM, do massive-activation magnitude,
  intermediate-sink count, and repetition rate rise together?
- **Why open.** P4. Two independent 2026 findings point at compression and neither cites the other;
  no causal test exists. This is the most *novel* question in this document.
- **Experiment.** Sweep the downsampling factor on Llama-AVSR (code public) or a comparable
  AudioLLM; measure the three quantities at each setting.
- **What counts as an answer.** A monotone (or non-monotone, stated) relation across ≥4
  compression settings with CIs, and a stated family size for the layer scan.
- **Cost.** ~30–60 A100-hours; ~50+ h of Adrian's time, dominated by getting an AudioLLM
  fine-tuning loop working.
- **Risk.** **High for one semester.** Most infrastructure of anything here, and the payoff depends
  on a three-way relation actually existing.
- **Serves.** Highest novelty ceiling; worst risk-adjusted fit for a 125–150 h budget. Listed so
  the option is on the record, not recommended as the primary.

---

### **S6 — Is Whisper's language token entangled with its sink token?**

- **Question.** Whisper's prompt prefix is `<|startoftranscript|><|lang|><|transcribe|>`. Wagner
  found the sink on `<|tr|>`. Does the **adjacent language token** modulate sink behaviour, and
  does sink anomaly predict script failure / off-target output?
- **Why open.** P5. Script failure is documented with hard numbers (**<0.8% Pashto-script fidelity
  for every Whisper size**, vs >93% for three other models) and is entirely non-mechanistic. The
  sink token and the language token are neighbours in the same three-token prefix and nobody has
  connected them.
- **Experiment.** Force each of N language tokens on fixed audio; measure sink mass and outlier
  magnitude as a function of the forced language; test whether the sink signature predicts
  off-target/wrong-script output.
- **What counts as an answer.** Sink statistic vs script-fidelity rate across languages, with a CI
  and a permutation null over language assignment.
- **Cost.** ~10–15 A100-hours; ~25 h.
- **Risk.** Moderate-to-high: the effect may be entirely explained by training-data volume per
  language, which must be entered as a covariate or the result is uninterpretable. Also needs a
  non-Latin-script language the lab does not currently ship, so it drifts from lab priorities.
- **Serves.** Interpretability and the project's language-dependence RQ; genuinely novel; but the
  confound is serious.

---

**Recommended combination for one semester: S1 + S2, with S4 as a cheap add-on.** Together they are
~35 A100-hours and ~75 h of Adrian's time, they reuse the detector and ablation harness that
already exist and are already debugged, they produce the encoder-decoder row the current proposal
is missing, and S1 has a genuine pre-registerable null. S3 is the fallback if the lab wants
something it can deploy. S5 and S6 are on the record but should not be the primary.

---

## §5 — Verification debt

Things that must be checked before any of this is cited in a proposal or paper.

1. **The "zero results" gap claim is NOT verified.** I ran an arXiv full-text query
   `(all:"super weight" OR all:"outlier dimension" OR all:"outlier channel") AND (all:speech OR
   all:audio OR all:Whisper)` and got zero results, which would be a striking gap. But when I
   re-ran it three times with a control query (`all:"outlier dimension"` alone, which must have
   hits), **the control also returned empty** — the API was rate-limiting me. The zero result is
   therefore indistinguishable from an API failure. **Re-run both queries from a fresh session
   before claiming this gap anywhere.** The related claim that `all:"massive activations" AND
   (speech OR audio)` returns exactly one paper (Cappellazzo) *was* returned by a live API call in
   a batch where adjacent queries returned 16 results each, so it is better supported — but it
   should still be re-run.
2. **WebSearch budget was exhausted mid-task** (200/200, shared across the session's agents). Areas
   I could not search at all, and which therefore have **no coverage in this file**: simultaneous /
   streaming S2ST latency, prosody and speaker preservation in S2ST, reference-free quality
   estimation for speech translation, TTS-side failures (the lab's African TTS project),
   dialect/accent robustness beyond what arXiv surfaced, and SeamlessM4T's hallucination behaviour
   in any depth. **§2's problem table is incomplete in those five areas** and should not be read as
   a survey of the field.
3. **Venues.** Only three are `VERIFIED` against a proceedings record: Koenecke (ACM FAccT 2024),
   Wang/Calm-Whisper (Interspeech 2025, ISCA archive), Bemba (IWSLT 2025, arXiv journal-ref), plus
   Nahabwe et al. (PMLR 302 header in the PDF). Six more are `author-declared` from arXiv comment
   fields: Wagner (Interspeech 2024), Cappellazzo (ICASSP 2026), Atwany (ACL 2025), HALAS
   (Interspeech 2026), TextMetrics (Interspeech 2026), Waldendorf (Findings of ACL 2026). **All
   should be confirmed against the actual proceedings.** Everything else in §1 is `preprint`.
4. **Radford et al. (Whisper) has no venue on its arXiv record.** It is universally cited as ICML
   2023 (PMLR v202); I did not verify this. Do not print the venue until checked.
5. **Frieske & Shi (2401.01572)**: a search snippet implied Interspeech 2024; the arXiv record
   states no venue. Unresolved.
6. **AfriSwitch size discrepancy.** A search snippet said 54.41 h / 14 languages; the PDF abstract
   says 61.36 h / 16 languages/varieties. I used the PDF. If citing, quote the PDF and note the
   version (v1, 2026-08-26).
7. **Glazer et al. is the load-bearing paper for S1 and S2's framing and it is an unrefereed
   preprint from a company research lab.** Its repetition result rests on **102 utterances** from
   two languages. Before building on "self-attention and FFN interventions had no measurable
   effect," check whether a refereed version exists and read their appendix for the intervention
   details — the strength of that null is what makes S1 interesting, so it deserves scrutiny.
8. **`[abstract]`-only papers used for numbers in §2.** P4's Q-former claim, P2's 86%-of-51,797
   figure and P5's Pashto numbers all come from abstracts I read but whose methods sections I did
   not. Read the methods before any of these enter a proposal.
9. **The lab's actual pipeline is unverified.** I did not read
   `/home/adrian/Repos/arctos/speech-translation/` (brief says untrusted) and have no confirmation
   of which Whisper size, which NLLB checkpoint, which TTS, or which decoding settings the ToAll
   app runs. S3's cost estimate assumes this is discoverable in under a day; confirm before
   committing.
10. **Compute estimates are unbenchmarked.** All A100-hour figures are my own order-of-magnitude
    guesses from model size and forward-pass counts, not measured. Pilot before scheduling.
11. **`[unopened]` files** listed in §1.5 are indexed but unread: Seamless, CodeSwitch commercial
    benchmark, Swahili continued-pretraining. Do not cite them.
