# Internals of speech and multimodal speech–text models

Agent: interpretability arm of the 2026-09-06 speech-translation sweep.
Scope: does the text-side picture (a "constant" — an early-layer FFN neuron writing a
massive activation into the residual stream on one token, which then acts as a fixed bias
and an attention sink — plus a shared cross-lingual/cross-modal "semantic hub") hold on the
speech side, and what tooling exists to look.

Reading convention used throughout:
- **[read]** = PDF opened and the cited section read.
- **[skimmed]** = abstract + intro + the one section the claim rests on.
- **[unopened]** = title/abstract only, from a search listing or the arXiv API. Not citable
  without opening first.
- **causal** = the paper intervenes (patch, ablate, steer, rotate) and reports the effect.
  **observational** = probing, similarity, correlation, visualisation only.
- Venues are as *printed on the PDF or in the arXiv `comment`/`journal_ref` field*. Anything
  without such a stamp is labelled **preprint**.

All papers named below are downloaded into this directory (flat), and listed in `README.md`
under "Speech translation and speech models (added 2026-09-06) → Interpretability".

---

## §1 Paper by paper

### 1.1 Speech–text representation convergence

**Lee, Liu, Sinhamahapatra & Niehues — "How do Multimodal Foundation Models Encode Text and
Speech?"** · `Lee-2025-Multimodal-Encode-Text-and-Speech.pdf` · **NAACL 2025 (Short Papers,
pp. 600–610)** · arXiv 2411.17666 · **[read]** (§2 methodology, §3.1 cross-modal, §3.3).
SVCCA between activations for semantically equivalent FLEURS sentences across **30
languages** (Amharic `amh` and Swahili are both in the set — Appendix C) and two modalities,
on **SeamlessM4T, SONAR, and SALMONN**.
Findings: (1) cross-modal similarity rises through layers *except* for a **dip in the initial
layers**, which they attribute to early audio layers doing acoustics (speaker identity) while
early text layers do syntax; (2) **length adaptation** (Seamless's length adaptor, SONAR's
pooling) is what closes the cross-modal gap, and it works mainly for high/medium-resource
languages — in low-resource languages the gap stays open; (3) **speech shows larger
cross-lingual spread than text**; (4) for models not explicitly trained to be
modality-agnostic, **the modality gap is larger than the language gap** — SONAR, which *is*
trained for it, is the exception, and there cross-modal similarity exceeds the others.
**Observational** (SVCCA only; no interventions).
*Load-bearing caveat for us:* §2 states plainly — "**We do not analyze the audio encoders'
internal representations as they are audio-only**." The flagship convergence paper therefore
says nothing about what happens inside Whisper's or w2v-BERT's encoder stack.

**Wu, Yu, Yogatama, Lu & Kim — "The Semantic Hub Hypothesis"** · `Wu-2025-Semantic-Hub-
Hypothesis.pdf` (already in this directory) · **ICLR 2025** · arXiv 2411.04986 ·
**[skimmed]**. Middle layers hold a shared, dominant-data-type-anchored representation
across languages *and* across modalities (their modality set includes audio in one arm);
cosine similarity of translation pairs >0.8 in middle layers only; **causal** — intervening
in the shared space using the model's dominant data type predictably changes output on other
data types. This is the text-side hypothesis my §5 questions port to speech.

**Nakai, Suresh & Demberg — "Generation-Step-Aware Framework for Cross-Modal Representation
and Control in Multilingual Speech–Text Models"** · `GenStepAware-2026-CrossModal-
Representation-Control-SeamlessM4T.pdf` · **preprint** (arXiv 2601.17387v3, 2026-08) ·
**[skimmed]** (abstract, intro). The closest existing work to "language-selective neurons,
but in a speech–text model". Their framing: text-only work finds language-selective neurons
concentrated in **early and late layers with mid-layers language-agnostic** (Kojima 2024,
Tang 2024, Tan 2024); neuron-level evidence for multimodal models is scarce. Their
contribution is measuring decoder neuron activations **per decoding step** rather than under
full-sequence conditioning, applied to **SeamlessM4T v2**, whose shared decoder gives a
speech-vs-text testbed. Includes control (steering) experiments — **causal**, but I have not
read the results tables. **This is the single most direct prior work on Adrian's interlingua
question in a speech model, and it must be read in full before any of §5 is proposed.**

**Modica, Landin, Farahani, Qian, Skantze & Johansson — "Do Factual Recall Mechanisms Carry
over from Text to Speech in Multimodal Language Models?"** · `FactualRecall-2026-Text-to-
Speech-Multimodal-LMs.pdf` · ***SEM 2026** (15th Joint Conf. on Lexical and Computational
Semantics) · arXiv 2605.22170 · **[skimmed]** (abstract, intro). **Causal Mediation Analysis**
(the ROME/Meng protocol) transplanted to **SpiritLM**. Result: text→text and speech→text
mediation profiles **disagree**, so factual-recall mechanisms are only *partially* shared
across modalities. **Causal.** Explicitly "initial results" — small, and a natural template
to extend.

**Hsu, Zhang, Tian, Zhang & Wu — "Anatomy of the Modality Gap: Dissecting the Internal States
of End-to-End Speech LLMs"** · `AnatomyModalityGap-2026-Internal-States-Speech-LLMs.pdf` ·
**preprint** (arXiv 2603.01502) · **[skimmed]**. Cross-layer **CKA** with speech–text token
alignment on four open-weight speech LLMs (SpeechMMLU, VoiceBench BBH). Speech shows a
**broad cross-layer alignment band** because semantic content is spread over many frames.
Key negative: **statistical calibration at the input layer is insufficient and can hurt** —
so the modality gap is *not* a mere distribution shift. Partly **causal** (they apply the
calibration and measure).

**Xiang, Zhao, Guo & Zou — "Understanding the Modality Gap: An Empirical Study on the
Speech-Text Alignment Mechanism of Large Speech Language Models"** · `ModalityGap-2025-
SpeechText-Alignment-Mechanism-LSLM.pdf` · **EMNLP 2025 (Main)** · arXiv 2510.12116 ·
**[skimmed]**. Deeper layers: speech and text converge **in direction** (cosine) while
**diverging in magnitude** (Euclidean) — a norm story, and the only place in this literature
where a magnitude asymmetry is reported. Spontaneous token-level alignment; they define an
Alignment Path Score, then **intervene** (angle projection, length normalisation) on critical
tokens and improve speech-input correctness. **Causal.**

**Sternberg, Maimon & Adi — "Interleaved Speech Language Models Latently Work In Text"** ·
`Interleaved-2026-Speech-LMs-Latently-Work-In-Text.pdf` · **preprint** (arXiv 2606.22473) ·
**[skimmed]**. **Logit lens** on interleaved speech–text SLMs: speech-derived hidden states
become decodable as the **text transcription**, then as next-word hypotheses, then project
back to speech tokens — "implicit latent transcription" — *without* transcription training.
Ablation of training decisions: both **text-LM initialisation and interleaved data** are
necessary; given both, the phenomenon appears across families, sizes and compute budgets.
This is the semantic-hub result for speech, in the strongest form currently published.
**Observational** (logit lens) plus a training-side ablation.

**Fan et al. (Amazon AGI + UIUC) — "Can We Read the Mind of an Audio LLM? A Verbalizable,
Multilingual Middle-Layer Workspace"** · `AudioLLM-2026-Verbalizable-Multilingual-MiddleLayer-
Workspace.pdf` · **preprint** (arXiv 2608.24958, 2026-08) · **[read]** (abstract). Logit lens
at audio-token positions in base **Qwen3-Omni**. The answer to a spoken question becomes
legible **in words** in middle layers before any token is emitted; the readout is
**language-agnostic** (38% of top-1 readouts are Chinese on English inputs) and
**paralinguistic** (recovers affect/speaker role that the model's own caption discards); the
audio-driven signal turns on ~10% into the network, separates in the **35–80% depth band**,
and **activation patching shows it is causally used**, committed before the last fifth of
layers. Layer-deletion maps entry layers → hearing, interior → retrieval, output layer →
delivery. **Causal.** The authors are careful: "the quantities are controls, not benchmark
scores."

**Jung, Jung, Kim & Chung — "Probing Cross-modal Information Hubs in Audio-Visual LLMs"** ·
`CrossModal-2026-Information-Hubs-AudioVisual-LLMs.pdf` · **ICML 2026** · arXiv 2605.10815 ·
**[read]** (abstract). AVLLMs encode **integrated audio-visual information in sink tokens**,
and a distinct subset — "**cross-modal sink tokens**" — specialises in storing it. They build
a training-free hallucination mitigation on top by encouraging reliance on those tokens.
**Causal** (the mitigation is an intervention). *This is the strongest published link between
the sink phenomenon and cross-modal information in a speech-adjacent model.*

**Other convergence items, [unopened]:** `CascadeEquivalence-2026-Speech-LLMs-vs-Pipelines.pdf`
(when do speech LLMs behave like ASR→LLM cascades; submitted to Interspeech 2026);
`Omnilingual-SONAR-2026-CrossLingual-CrossModal-Embeddings.pdf` (preprint; cross-lingual and
cross-modal sentence embeddings, massively multilingual — a candidate substrate);
`TranslationEnhanced-2026-Speech-Encoder-Pretraining-SpeechLLMs.pdf` (**Interspeech 2026**;
does a translation objective in encoder pretraining change downstream speech LLMs).

### 1.2 Interpretability of ASR / ST models

**Glazer, Segal-Feldman, Segev, Shamsian, Buchnick, Hetz, Fetaya, Keshet & Navon — "Beyond
Transcription: Mechanistic Interpretability in ASR"** · `Wu-2025-Beyond-Transcription-Mech-
Interp-ASR.pdf` (filename is a sibling agent's; authors are aiOla Research) · **preprint**
(arXiv 2508.15882) · **[read]** (abstract, §1). The field-defining "someone should do this"
paper. Adapts **logit lens, linear probing, activation patching** to Whisper and Qwen2-Audio.
Reported: acoustic and semantic attributes are linearly decodable in **encoder** layers with
cleaner separation up top; **hallucination signals live in the decoder's residual stream** and
support real-time quality prediction; specific **encoder–decoder interactions cause repetition
hallucinations**; and **contextual bias arises inside the encoder** and can override acoustic
evidence — which cuts against the assumed encoder=acoustics / decoder=language split.
**Causal** for the repetition and patching parts. Directly relevant to Adrian's repetition-loop
thread.

**Aparin, Sadekova, Rukhovich, Yermekova, Kushnareva, Popov, Kuznetsov & Piontkovskaya —
"AudioSAE"** · `AudioSAE-2026-EACL-Sparse-Autoencoders-Audio-Models.pdf` · **EACL 2026 (main
track), pp. 3221–3254** · arXiv 2602.05027 · **[read]** (abstract, §1). SAEs on **all encoder
layers of Whisper and HuBERT**. >50% feature stability across seeds; features capture acoustic
*and* semantic content, environmental noise, and paralinguistics (laughter, whispering);
erasing a concept needs removal of only 19–27% of features; **feature steering cuts Whisper's
false speech detections by 70%** at negligible WER cost; features correlate with human EEG.
**Causal** (steering, concept erasure). Code and checkpoints released.

**Pluth, Houghton, Zhou & Gurbani — "On the Interpretability of Whisper Encodings Using Sparse
Autoencoders"** · `Whisper-SAE-2026-Interpretability-Whisper-Encodings.pdf` · **preprint**
(arXiv 2605.12225v2) · **[read]** (abstract, §1). SAE on Whisper's encoder shows a **hierarchy
of linguistic information** — phonetic, lexical, syntactic, morphological, semantic — beyond
what transcription requires, plus a **causal feature-steering campaign across the whole
hierarchy**. **Causal.**

**"Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders"** ·
`AudioSAE-2025-Interpretable-Features-Audio-Latent-Spaces.pdf` · **NeurIPS 2025 Mechanistic
Interpretability Workshop** · arXiv 2510.23802 · **[unopened]**. SAEs on audio *codec* latent
spaces — the nearest thing to interpretability of EnCodec-style tokenisers.

**Ma, Lu, Sang, Jiang & Li — "Behind the Scenes: Mechanistic Interpretability of LoRA-adapted
Whisper for Speech Emotion Recognition"** · `BehindScenes-2025-MechInterp-LoRA-Whisper-SER.pdf`
· **ICASSP 2026** · arXiv 2509.08454 · **[read]** (abstract). Layer-contribution probing,
logit lens, SVD and CKA inside the Whisper encoder: **delayed specialisation** (early layers
keep general features, task-specific consolidation happens late) and a **forward-alignment /
backward-differentiation** dynamic between LoRA's A and B matrices. **Observational.**

**"Whisper Has an Internal Word Aligner"** · `Whisper-2025-Internal-Word-Aligner.pdf` ·
**ASRU 2025** · arXiv 2509.09987 · **[unopened]**. Cross-attention alignment structure inside
Whisper — the timestamp/alignment mechanism question.

**Wang, Alhmoud, Alsahly, Alqurishi & Ravanelli — "Calm-Whisper"** · `Wang-2025-Calm-Whisper-
Crazy-Heads.pdf` · **Interspeech 2025** · arXiv 2505.12969 · **[read]** (abstract, §1).
Head-wise masking of Whisper-large-v3's 20 decoder self-attention heads: **3 heads account
for >75% of non-speech hallucinations** on UrbanSound; fine-tuning just those three cuts
non-speech hallucination >80% with <0.1% WER cost. **Causal** (head ablation is the method).
The single cleanest head-level causal result in the Whisper literature.

**Layer-wise probing of SSL speech encoders (background, mostly [unopened]):**
`LayerWise-2026-Probing-wav2vec2-Whisper-AAE.pdf` (**Interspeech 2026**; wav2vec 2.0 and
Whisper for consonant-cluster reduction in African American English);
`insideSSL-2026-Model-Centric-SSL-Speech-Representations.pdf` (**Interspeech 2026** long-paper
track; a cross-layer Generative Compatibility Matrix exposing "stable phonetic cores,
identity volatility, deep-layer semantic pruning");
`Orthogonality-2024-Speaker-Phonetic-SSL-Speech.pdf` (**Interspeech**; speaker and phonetic
information are approximately orthogonal subspaces in SSL speech representations — the
geometry paper closest to a "language subspace" claim);
`Anisotropy-2025-Neural-Representations-of-Speech.pdf` (preprint, arXiv 2506.11096; anisotropy
in speech representations — the property a massive activation would produce, measured without
reference to it);
`ListeningWithAttention-2026-Entropy-Guided-Explainability-Audio.pdf` (**Interspeech 2026**;
entropy-guided attention explainability for audio transformers).
The consensus these encode: **wav2vec 2.0 peaks phonetically in late-middle layers; HuBERT/
WavLM stay high through the final layers; early layers mirror filterbank/F0/formants; upper
layers align to the pretraining objective.**

**Paralinguistics and prosody:** `HeardNotHeeded-2026-Paralinguistic-Encoding-Loss-ALMs.pdf`
(preprint, 2609.00727) and `ProsodicUnderuse-2026-Causal-Account-Audio-LMs.pdf` (preprint,
2608.19211) — both **[unopened]**; the second is billed as a *causal* account of why prosody
is represented but not used. Relevant if the project touches tone languages (Yoruba, Igbo).

### 1.3 Outliers / sinks / registers in audio transformers — the papers that exist

**Anand, Cappellazzo, Petridis & Pantic — "Mitigating Attention Sinks and Massive Activations
in Audio-Visual Speech Recognition with LLMs"** · `Cappellazzo-2025-Attention-Sinks-Massive-
Activations-AVSR.pdf` · **IEEE ICASSP 2026** · arXiv 2510.22603v3 · **[read]** (abstract,
§3.1–3.3). *The* paper for §3. Substrate: **Llama-AVSR** = AV-HuBERT (video) + **Whisper
(audio)** encoders → average-pool downsample → linear projectors → **LLaMA-3.2-3B** decoder,
LoRA fine-tuned.
Findings, all inside the **LLM decoder**:
- Attention sinks at **BOS *and* at intermediate low-semantic tokens**, across ASR, VSR and
  AVSR. The BOS sink pre-exists in the LLM; the **intermediate sinks emerge during
  fine-tuning**.
- Massive activations defined as `|H^l[i,j]| ≥ τ·median(|H^l|)` with **τ = 10³**. The set is
  **non-empty iff `i` is a sink token and `l ∈ {2,…,L−1}`** — never in the first or last
  layer — and the **feature index set is identical across all sink tokens** (e.g. token
  indices {0, 20, 21} share Θ at layer 5).
- Origin traced to the **MLP of layer 2**: inside the GLU, `hW_gate` is large and positive on
  a fixed feature set for sink tokens and negative for non-sink tokens; SiLU keeps only the
  positives; the elementwise product with `hW_up` amplifies; `W_down` writes it into the
  residual stream. **This is the same mechanism as the text-side "constant".**
- Mechanism: intermediate sink hidden states have **high cosine similarity with BOS from layer
  2 onward**, which is why they inherit both the feature set and the attention.
- **Causal test:** rotating a sink token's hidden state toward the nearest non-sink token
  **removes** both the sink and the massive activation; rotating a non-sink token toward BOS
  **creates** them at that position. Then a decorrelation loss (reduce cos-sim to BOS) that
  mitigates intermediate sinks and improves WER under high downsampling.
**Caveat that defines the gap:** every measurement is on the **text LLM decoder**. Whisper is
used only as a frozen feature extractor; the paper never looks inside it.

**Wagner, Baumann, Riedhammer & Bocklet — "Outlier Reduction with Gated Attention for Improved
Post-training Quantization in Large Sequence-to-sequence Speech Foundation Models"** ·
`Outlier-Reduction-2024-Gated-Attention-PTQ-Speech-Foundation.pdf` · **Interspeech 2024** ·
arXiv 2406.11022 · **[read]** (§3.4, §3.5, §4). **The most important under-cited result for
this project.** §3.4 is titled "**Whisper exhibits outlier behavior similar to LMs**":
- On the **pretrained 1.55B Whisper-large** decoder, layer 31, an attention head allocates
  nearly all its probability mass to the **task token `<|tr|>`** (the transcribe/translate
  special token in Whisper's decoder prompt), while that token's **value vector is small**, so
  the head's output update is ≈0. They call it "minimal or no update of the hidden
  representation". **This is a no-op attention sink on a special prompt token, in Whisper's
  own decoder, reported in 2024 and never named as such.** They say "similar patterns can be
  found across all decoder layers and attention heads."
- In distilled students, activation outliers concentrate in **fixed hidden dimensions**:
  dims **#819 and #1054 carry ~15% of all outliers** at the self-attention output projection
  of the last decoder layer (d=1280); gated attention flattens this to ~5% for the top two.
- **The last encoder layer retains a large dynamic range** and, with the final layer norm, has
  to be **left in original precision** for INT8 to work.
- Measured with **kurtosis and ‖·‖∞**, averaged over test sets.
**Observational** for the Whisper-large analysis; **causal-by-architecture** for the gating
(train with/without, measure outliers and WER). They cite Bondarenko's "helping attention
heads do nothing" as the text-side analogue.

**Yoo, Jang & Chung — "On the Nature of Attention Sink that Shapes Decoding Strategy in
Omni-LLMs"** · `OmniLLM-2026-Nature-of-Attention-Sink-Decoding.pdf` · **preprint** (arXiv
2603.14337v2; the PDF says "Preprint") · **[read]** (abstract, §1). First analysis of sinks in
**Omni-LLMs** (video+audio+text). Two findings: high sink attention does **not** simply mean
head redundancy — sink *value* representations do work; and **the sink value vector acts as a
shared bias added to every token's output**, a global organising signal. Their method (OutRo)
aligns non-sink representations with the sink and relaxes the causal mask for sink tokens at an
early layer. **Causal.** The "shared bias" claim is the same functional story as the text-side
constant, now in a model that ingests audio.

**Jung et al., ICML 2026** (above, §1.1) — sink tokens as cross-modal information hubs.

**Quantization papers that touch Whisper outliers without localising them** (all
**[unopened]** except as noted): `Quantizing-Whisper-2025-Design-Choices.pdf` (**SPEAKABLE
workshop at LREC 2026**; arXiv 2511.08093); `EdgeASR-2025-LowBit-Quantization-ASR-Models.pdf`
(preprint 2507.07877); `PTQ-2026-LayerWise-Compensation-EncoderDecoder-ASR.pdf` (preprint
2601.02455 — diagnostic-driven layer-wise compensation for encoder-decoder ASR, i.e. someone
has already measured *which layers* of an ASR encoder-decoder resist quantization).

**Vision precedents, for method transfer:** `Darcet-2024-ICLR-Vision-Transformers-Need-
Registers.pdf` (**ICLR 2024**, arXiv 2309.16588) — high-norm artifact tokens in ViT/DINOv2 and
the register fix; `VisionEncoders-2026-ECCV-Activation-Quantization-Prefixing-Registers.pdf`
(**ECCV 2026**, arXiv 2510.04547) — prefixed registers absorb the outliers that break
activation quantization of vision encoders. `Sun-2024-COLM-Massive-Activations-in-LLMs.pdf`
(**COLM 2024**) is the text-side original.

### 1.4 Cross-lingual structure in speech models

**Shim, De Cristofaro, Hu, Vietti & Plank — "Languages in Whisper-Style Speech Encoders Align
Both Phonetically and Semantically"** · `Whisper-Encoders-2025-Languages-Align-Phonetically-
Semantically.pdf` · **submitted to Interspeech 2026** (so: **preprint**) · arXiv 2505.19606v2 ·
**[read]** (abstract, §1). The key methodological correction in this area. Prior work reported
~80% spoken-translation-retrieval accuracy from Whisper's encoder on FLEURS and read it as
cross-lingual alignment; Shim et al. point out that **cognates, loanwords and proper nouns are
a phonetic shortcut** and build a **pronunciation-controlled challenge set** between
typologically distant languages. Result: retrieval stays **well above chance without phonetic
cues in the final layers of encoders trained with a speech-translation objective**, most
clearly for models additionally trained on translation; ASR-only Whisper-style encoders show
"weak but non-trivial" retrieval. They then **early-exit the encoder** to get
less-language-specific representations and gain ASR accuracy on **unseen low-resource
languages**. **Causal** for the early-exit part. *Any Swahili/Igbo/Xhosa alignment claim we
make must use their control, or it is measuring cognates.*

**"Latent Mechanisms of Language Control in Multilingual Language Models"** ·
`LanguageControl-2026-Latent-Mechanisms-Multilingual-LMs.pdf` · **EMNLP 2026 (main track)** ·
arXiv 2609.00325 · **[unopened]**. Text-only, but the mechanism vocabulary we would port.

**"Multilingual Emotion Neurons in Large Audio-Language Models"** · `EmotionNeurons-2026-
Multilingual-Large-Audio-LMs.pdf` · **preprint** (arXiv 2608.08772) · **[unopened]**. The only
neuron-level *multilingual* study of an audio LM I found.

**`Omnilingual-ASR-2025-1600-Languages.pdf`** (already present, sibling agent's download) —
the 1,600-language ASR model; no internals analysis exists for it that I could find.

**Not found, despite looking (see §3 search log):** any study of **language-specific vs shared
neurons in Whisper, MMS, or Omnilingual ASR**, any **language-ID subspace** analysis of a
speech encoder, and any mechanistic account of **code-switching internals** in a speech model.
Code-switching in speech is currently benchmark work only (`AfriSwitch-2026-…`,
`CodeSwitch-2026-Benchmarking-Commercial-ASR.pdf`, `Yan-2025-CS-FLEURS-…` in this directory).

---

## §2 What is established about shared vs modality/language-specific structure

Stated with the baseline in the sentence, per the repo's claim-hygiene rule.

1. **Convergence through depth is real but partial, and it is measured on decoders and text
   encoders, not on speech encoders.** SVCCA (Lee, NAACL 2025), CKA (Hsu, preprint), cosine
   (Xiang, EMNLP 2025) all agree that speech and text representations grow more similar with
   depth after an initial dip. Lee et al. explicitly exclude the audio encoders from analysis.

2. **The modality gap is larger than the language gap** — in models *not* trained for
   modality-agnostic representations (Lee, NAACL 2025 §3.3). SONAR, which *is* explicitly
   trained to align modalities, inverts this. So "one shared representation" is a training
   objective, not an emergent inevitability.

3. **Speech has larger cross-lingual spread than text** (Lee, NAACL 2025 §3.2). The interlingua
   is weaker on the speech side of the same model.

4. **Direction converges while magnitude diverges.** Xiang et al. (EMNLP 2025) find deep-layer
   speech and text representations increasingly aligned in cosine and increasingly apart in
   Euclidean distance. This is the one published hint that a **norm-level** phenomenon
   separates the modalities — exactly the quantity a massive activation would dominate. Nobody
   has asked whether the magnitude divergence *is* an outlier-dimension effect.

5. **Length adaptation is the operative mechanism for closing the cross-modal gap**, and it
   works mainly for high- and medium-resource languages (Lee, NAACL 2025). For African
   low-resource languages the adaptor does not deliver — which is precisely the lab's setting.

6. **Interleaved speech LMs latently transcribe to text** (Sternberg, preprint): speech-derived
   states are decodable as the *text* of the spoken word in middle layers, without transcription
   supervision, and this requires both text-LM initialisation and interleaved training data.
   The Amazon Qwen3-Omni logit-lens study (preprint) reports the same middle-layer verbalisable
   workspace, adds that it is **language-agnostic** and shows by **activation patching** that
   it is causally used. Together these are the speech instance of the semantic hub — but both
   are preprints and both study *decoder-side* representations of models initialised from text
   LMs.

7. **Mechanism transfer from text to speech is partial, not total.** Causal mediation for
   factual recall gives different profiles for text→text and speech→text in SpiritLM (*SEM
   2026). The encoder/decoder division of labour is also less clean than assumed: contextual
   bias arises *inside* Whisper's encoder and can override acoustic evidence (Glazer,
   preprint).

8. **Cross-lingual alignment in Whisper-style encoders survives a phonetic control, but only
   in the final layers and mainly with a translation objective** (Shim, preprint). Earlier
   claims of ~80% retrieval were partly cognate shortcuts.

9. **Layer-wise division of labour in SSL speech encoders is settled**: early = filterbank-like
   acoustics and speaker identity; middle = phones, tone, prosody; late = objective-aligned
   semantics (wav2vec 2.0 peaks phonetically late-middle, HuBERT/WavLM stay high to the end).

10. **Unestablished and open:** whether any *speech encoder* has language-specific vs shared
    neurons; whether cross-modal interventions propagate (only Wu ICLR 2025 for text+vision,
    Fan preprint for audio patching, Xiang EMNLP 2025 for token-level projection); whether the
    convergence story holds for African low-resource languages at all.

---

## §3 Does the "constant" / sink / register phenomenon exist in speech models?

**This section is the centre of the notes. Short answer: three positive results exist, all of
them in the *text-LLM decoder* of a multimodal model, plus one 2024 Interspeech observation
inside Whisper's own decoder that was never named as a sink. For speech *encoders* —
Whisper's encoder, w2v-BERT 2.0 / SeamlessM4T's Conformer speech encoder, wav2vec 2.0,
HuBERT, WavLM, AST, BEATs — and for neural audio codecs (EnCodec, DAC, SpeechTokenizer),
nothing exists. I could not find a single paper reporting massive activations, high-norm
tokens, register tokens, or a named attention sink in any speech encoder.**

### 3.1 What exists — positive results, ordered by how close they are to the question

| # | Where the phenomenon was found | Model | Venue | Causal? |
|---|---|---|---|---|
| 1 | Sinks at BOS **and** emergent intermediate tokens; massive activations (τ=10³) from **MLP of layer 2**, identical feature indices across sink tokens, never in first/last layer | **LLaMA-3.2-3B decoder** of Llama-AVSR (Whisper + AV-HuBERT as frozen encoders) | ICASSP 2026 | **Yes** — rotation toward/away from BOS creates/destroys both |
| 2 | A decoder head puts ~all probability mass on the **`<\|tr\|>` task token** (= `<\|transcribe\|>`), whose value vector is small → **no-op update**; activation outliers concentrated in **fixed dims #819/#1054** (~15% of all outliers); **last encoder layer must stay in FP** | **Whisper-large (1.55B), decoder layer 31** and distilled students | Interspeech 2024 | Partly — gated attention is a train-time intervention that reduces the outliers |
| 3 | **Sink value vector acts as a shared bias added to every token's output**; sink heads are not merely redundant | Omni-LLM decoders (video+audio+text) | preprint | **Yes** — OutRo intervention |
| 4 | Integrated audio-visual information is stored in sink tokens; a subset are dedicated **cross-modal sink tokens** | Audio-visual LLM decoders | ICML 2026 | **Yes** — training-free mitigation built on it |

Item **2 is the one to build on.** It is the only published observation of sink-like,
no-op-attention behaviour *inside a speech model's own weights* rather than inside a bolted-on
text LLM, it lands on **a special prompt token in the decoder** (`<|transcribe|>`, adjacent to
the `<|startoftranscript|>` and language-tag tokens Adrian asked about), and it names **fixed
hidden dimensions**. What it does **not** do: give a magnitude ratio to the median, look at
`<|startoftranscript|>` or the language tag specifically, vary the language, look at the
encoder's token axis at all, or connect any of it to the massive-activation literature by name.

### 3.2 What does not exist

To the best of an exhaustive search (log in §3.3), **no paper reports**:

- Massive activations, super weights, or a residual-stream "constant" in **Whisper's audio
  encoder**, in **w2v-BERT 2.0 / SeamlessM4T's speech encoder**, in **wav2vec 2.0, HuBERT or
  WavLM**, or in **Omnilingual ASR / MMS**.
- **Register tokens** or an audio analogue of Darcet's ViT registers in **any** audio
  transformer. The arXiv query `abs:"register tokens"` returns 20 recent papers, **all vision**
  (ViTs, diffusion transformers, VLA models, VGGT, face recognition, Mamba). Zero audio.
- **High-norm / artifact tokens** in an audio encoder. `abs:"high-norm tokens"` returns eight
  papers: ViTs, vision Mamba, diffusion transformers, DeepSeek-OCR, and one LLM paper
  ("Demystifying Singular Defects in LLMs"). Zero audio.
- Attention sinks in **AST, BEATs**, or any spectrogram-patch transformer.
- Outlier structure in **neural audio codecs** — `abs:"audio codec" AND abs:"outlier"` returns
  **no results at all**.
- Whether the phenomenon is **shared across languages** in any speech model. Nobody has run the
  cross-lingual version of the sink question on the speech side.

Two supporting negatives worth stating because they are cheap to verify and easy to
misremember: the strings *attention sink*, *massive activation*, *outlier*, *register token*
and *high-norm* appear **zero times** in the full text of **Beyond Transcription** (the ASR
mech-interp survey), **AudioSAE** (EACL 2026), **the Whisper SAE paper**, **insideSSL**
(Interspeech 2026), and **Lee et al.** (NAACL 2025) — i.e. the five papers that would be
expected to mention it. Checked by `pdftotext | grep -i`.

The adjacent literature that gets *near* it and stops: Whisper quantization papers know the
encoder is hard to quantize (`Quantizing-Whisper-2025-Design-Choices.pdf`, `EdgeASR-2025-…`,
`PTQ-2026-LayerWise-Compensation-EncoderDecoder-ASR.pdf`) but none localises an outlier token
or dimension; the anisotropy paper (`Anisotropy-2025-Neural-Representations-of-Speech.pdf`)
measures the geometric symptom a massive activation would produce, without asking what causes
it; Xiang et al. (EMNLP 2025) find speech/text magnitudes diverging in deep layers without
asking which dimensions.

**This is the gap. It is a real, checkable, un-taken gap, and the first measurement costs one
forward pass.**

### 3.3 Search log — every query run

**WebSearch (11 issued before the session's 200-call budget was exhausted by sibling agents;
2 further queries were refused):**
1. `Lee 2025 NAACL "How do multimodal foundation models encode text and speech" modality gap language gap`
2. `attention sinks massive activations Whisper speech encoder audio transformer register tokens`
3. `"semantic hub" hypothesis Wu 2025 ICLR multilingual multimodal shared representation audio`
4. `Whisper mechanistic interpretability language identification head circuit analysis 2025`
5. `wav2vec2 HuBERT layer-wise probing phonetic semantic layers analysis`
6. `SeamlessM4T interpretability layer-wise analysis speech text representation shared encoder`
7. `speech LLM modality gap Qwen2-Audio Qwen2.5-Omni internal representations analysis 2026`
8. `multilingual ASR language-specific neurons Whisper MMS shared neurons cross-lingual`
9. `Whisper quantization outlier activations int8 hard to quantize encoder`
10. `sparse autoencoder Whisper audio speech model features interpretability SAE`
11. `nnsight TransformerLens support Whisper encoder-decoder speech model hooks interpretability library`
*Refused (budget):* `"Beyond Transcription" mechanistic interpretability ASR survey`;
`register tokens high-norm artifacts audio spectrogram transformer AST BEATs`.

**arXiv API (`export.arxiv.org/api/query`, sorted by submittedDate desc), all run and logged:**

*Sink / register / outlier family — the §3 core:*
- `abs:"massive activations"` → 13 hits: LLMs, diffusion transformers, graph LMs, video
  diffusion, quantization. **Exactly one speech hit: 2510.22603 (the AVSR paper).**
- `abs:"massive activation" AND (abs:multimodal OR abs:vision OR abs:audio)` → 5 hits, same
  single speech hit.
- `abs:"attention sink" AND abs:Whisper` → **1 hit**, XLSR-Transducer (2407.04439), which uses
  "attention sink" for a *streaming attention window*, not the phenomenon.
- `abs:"attention sink" AND abs:"speech"` → 3 hits: WnW KV-cache, the AVSR paper, XLSR-Transducer.
- `abs:"attention sink" AND (abs:audio OR abs:acoustic)` → 5 hits: + Omni-LLM sinks, windowed
  sink attention for vocal separation, Metronome (serving).
- `abs:"sink token" AND (abs:audio OR abs:speech OR abs:multimodal)` → 10 hits, all
  vision/omni/AVLLM decoders; the two relevant ones are 2605.10815 and 2603.14337.
- `abs:"attention sink" AND abs:encoder` → 20 hits, **all vision or text**.
- `abs:"attention sink" AND (abs:"speech translation" OR abs:"machine translation" OR abs:multilingual)`
  → 1 hit: 2605.01229, attention sinks in massively multilingual NMT (NLLB) — text.
- `abs:"register tokens"` → 20 hits, **all vision**.
- `abs:registers AND (abs:audio OR abs:"speech")` → 20 hits, none about register tokens
  (matches "social register", "register of instruments", etc.).
- `abs:"high-norm tokens" OR abs:"high norm tokens"` → 8 hits, **all vision or text**.
- `abs:"artifact tokens" OR abs:"outlier tokens"` → 17 hits, **all vision or text**.
- `abs:"outlier" AND abs:Whisper` → **1 hit**: 2406.11022 (Wagner, Interspeech 2024).
- `abs:"activation outliers" AND (abs:speech OR abs:audio)` → 1 hit, PTQ for audio *diffusion*
  transformers (2510.00313).
- `abs:"outlier" AND (abs:wav2vec OR abs:HuBERT OR abs:Conformer)` → 20 hits, **all false
  positives** (conformal prediction, conforming agents).
- `abs:"SmoothQuant" AND (abs:speech OR abs:Whisper OR abs:ASR)` → **no results**.
- `abs:"quantization" AND abs:Whisper` → 20 hits; the relevant ones are 2406.11022, 2511.08093,
  2601.02455, 2507.07877, 2503.09905, 2305.10788.
- `abs:"audio codec" AND abs:"outlier"` → **no results**.
- `abs:"EnCodec" AND (abs:quantization OR abs:outlier OR abs:analysis)` → 20 hits, all codec
  engineering; only 2510.23802 is interpretability.
- `abs:"BEATs" OR abs:"Audio Spectrogram Transformer" AND abs:analysis` → returned unrelated
  recent submissions (the OR binds loosely in the arXiv grammar); **no audio-transformer
  outlier paper surfaced**.
- `abs:"first frame" AND abs:attention AND abs:speech` → **no results**.
- `abs:"no-op" AND abs:attention` → 20 hits, none speech (one FP8 "sink-induced collapse"
  paper, 2606.06521, text).
- `abs:"anisotropy" AND (abs:speech OR abs:audio)` → 2 relevant hits (2506.11096, 2512.10120).
- `abs:"startoftranscript" OR abs:"language token" AND abs:Whisper` → 2 hits, neither about
  internals ("Do Prompts Really Prompt?", 2406.05806, is the closest).
- `abs:"attention" AND abs:"Whisper" AND (abs:"heads" OR abs:"head")` → 17 hits; relevant:
  Calm-Whisper (2505.12969), Whisper Has an Internal Word Aligner (2509.09987),
  Listening with Attention (2606.14647).

*Convergence / interp / cross-lingual family:*
- `(ti|abs:"modality gap") AND (speech|audio) AND representations` → 20 hits; kept 2603.01502,
  2510.12116, 2606.12199, 2601.05543.
- `abs:"speech" AND abs:"text" AND ("representation alignment" OR "cross-modal alignment") AND layer(s)`
- `abs:"semantic hub" OR ("shared representation" AND modalities AND "language model")` → 20
  hits, none speech.
- `abs:"mechanistic interpretability" AND abs:speech`; `abs:"logit lens" AND (speech|audio)`;
  `abs:"probing" AND abs:Whisper AND abs:layer`; `abs:"interpretability" AND abs:"audio language model"`
- `abs:"language-specific neurons" …`; `abs:"language-specific" AND abs:speech AND (neurons|subspace|representations)`
- `abs:SeamlessM4T AND (analysis|representations|interpretability)`
- `abs:"cross-lingual" AND abs:"speech encoder" AND abs:"representation similarity"` → 1 hit
  (2505.19606).
- `abs:"code-switching" AND ("internal representations"|probing|interpretability)` → 20 hits,
  **no speech-model internals**.
- `abs:"factual recall" AND (speech|audio)` → 1 hit (2605.22170).
- `abs:"cross-modal" AND abs:"activation patching"` → 1 hit, unrelated.
- `abs:"speech" AND abs:"steering vector"` → 17 hits; relevant: SALSA (2606.00460), hidden-state
  nudging for CoT in audio LMs (2603.14636), EmoSteer-TTS (2508.03543).
- `abs:"nnsight" OR abs:"TransformerLens" OR abs:"inseq" OR abs:"Captum"` → 20 hits, tooling.
- `abs:"Omni" AND abs:"attention" AND abs:"analysis" AND abs:"audio"` → 2 hits.

**Direct source checks (not search):**
- `grep -i` over the full text of the five main speech-interp PDFs for
  `attention sink|massive activation|outlier|register token|high-norm` → **zero matches in all
  five**.
- TransformerLens `main` branch `loading_from_pretrained.py`: `grep -ic whisper` → **0**.

### 3.4 Honest statement of the gap

Nobody has asked whether **Whisper's encoder builds a fixed high-norm frame**, whether
**SeamlessM4T's Conformer speech encoder does**, whether **Whisper's decoder constant sits on
`<|startoftranscript|>` or on the language tag**, or whether **any of it is shared across
languages**. The one adjacent data point (Wagner, Interspeech 2024) says a Whisper-large
decoder head does a no-op on the **task** token and that outliers concentrate in fixed
dimensions — which makes a positive result likely enough to be worth one afternoon of
measurement, and a negative result publishable as "translation models do not build one."

---

## §4 Tooling

| Tool | Whisper | SeamlessM4T / NLLB | wav2vec2 / HuBERT | Qwen-Omni / speech LLMs | Verdict |
|---|---|---|---|---|---|
| **HF `transformers` hooks** | Yes — `WhisperModel`, `output_hidden_states=True`, `output_attentions=True`, plus `register_forward_hook` on any submodule | Yes (`SeamlessM4Tv2Model`, `M2M100`/NLLB) | Yes | Yes | **The default. Everything below is a convenience layer over this.** |
| **TransformerLens** (3.8.1) | **No.** `grep -ic whisper` on `loading_from_pretrained.py` = 0 | **No** Seamless; T5 supported (encoder-decoder) | **Yes** — `HubertModel`, `HubertForCTC`, `Wav2Vec2Model` are in the conversion path (`convert_hubert_weights`), with a comment that "HuBERT operates on audio frames, not tokens — n_ctx is flexible" | No | **Usable for SSL speech encoders, not for Whisper or Seamless.** Repo pin note in `CLAUDE.md` (3.6.0) predates this. |
| **nnsight / NDIF** (0.6+; NeurIPS/ICLR-published, arXiv 2407.14561) | Yes in principle — it traces **any** PyTorch/HF module by name, preserving HF behaviour | Yes | Yes | Yes | **The right tool for Whisper and Seamless.** No architecture-specific adaptation, so no numerical mismatch. |
| **nnterp** (NeurIPS 2025 MI workshop, arXiv 2511.14465) | Standardises *transformer-LM* naming over nnsight; speech not a target | No | No | No | Not for this. |
| **inseq** (ACL 2023 Demo) | **No.** Docs list `ForConditionalGeneration` and `ForCausalLM` **text** models; no audio/speech attribution | Yes for NLLB/text-side Seamless | No | No | Good for the **text** half of a speech-text comparison only. |
| **captum** | Works on any PyTorch model, but gradient attribution over log-Mel frames is not a standard recipe | Same | Same | Same | Generic; no speech recipes. |
| **DecoderLens** (`DecoderLens-2023-Layerwise-Interpretation-EncoderDecoder.pdf`) | **Yes — already applied to Whisper base/small/medium** (6/12/24 layers) in the paper | Applicable (it is an encoder-decoder method) | n/a | n/a | **Directly reusable**, and the method is exactly "read the encoder's layers through the decoder". |
| **SAE tooling** | AudioSAE (EACL 2026) **releases code and checkpoints** for SAEs on all Whisper *and* HuBERT encoder layers | No | Yes (AudioSAE) | No | The only off-the-shelf speech interp artefact with released weights. |
| **s3prl** | Standard SSL speech probing/benchmark harness (SUPERB) | No | Yes | No | For layer-wise probing baselines. **[unverified — not opened this session]** |
| **circuit-tracer / SAELens** | Decoder-only text only | No | No | No | Out of scope, per repo `CLAUDE.md`. |

**No speech-specific mechanistic-interpretability library exists.** The nearest things are
AudioSAE's released checkpoints and the DecoderLens method. Everything else is HF hooks.

---

## §5 Candidate MS-sized questions

Constraints assumed: one semester, ~125–150 h, BYU ORC single-GPU SLURM (A100/H100), 7–13B
fits; lab assets = fine-tuned NLLB-200, African-language ASR/TTS interest (Xhosa, Igbo, Efik,
Swahili, Twi, Yoruba), and the text-side detector code Adrian already has.

Ordered by ratio of (novelty × feasibility) to cost.

---

### Q1 — Does Whisper's decoder have the constant, and does it sit on the language tag?

**Question.** Whisper's decoder is prompted with a fixed prefix:
`<|startoftranscript|> <|lang|> <|transcribe|>|<|translate|> <|notimestamps|>`. Does one of
those positions carry a massive activation (≥10³ × median, per the Cappellazzo/Sun definition),
is it written by an early-layer MLP `down_proj`, does it persist to the last layer, and **is it
on the same token for every language?**

**Why it is not already done.** Wagner et al. (Interspeech 2024) saw a decoder head do a no-op
on `<|tr|>` and never followed up; no paper applies the massive-activation definition to
Whisper.

**Experiment.**
1. Run `whisper-large-v3` (and `small`, `medium` for a size sweep) over FLEURS with
   `output_hidden_states=True`. Compute, per layer and per decoder-prefix position,
   `max|h| / median|h|` and the argmax feature index. Detector code exists on the text side.
2. Localise the writer: decompose layer *ℓ* into attn-out and MLP-out; if MLP, find the
   `down_proj` rows (the "super weights" step).
3. **Causal:** set-to-zero vs set-to-mean at that position, per the text-side protocol. Score
   WER/BLEU and check for the "We. We. We." repetition signature — Whisper's hallucination
   loops are the famous speech instance, and `Wu-2025-Beyond-Transcription…` already localises
   repetition to encoder–decoder interactions.
4. **Cross-lingual:** repeat for ≥12 FLEURS languages including **Swahili, Yoruba, Igbo,
   Xhosa, Amharic** and 4 high-resource controls. Is it the same position, same feature index,
   same magnitude? Also compare `<|transcribe|>` vs `<|translate|>`.

**Cost.** Whisper-large-v3 is 1.55B — inference-only, single A100, well under an hour per
language for a FLEURS test split. Total **~15–25 GPU-hours**, ~30 h of work. The cheapest
project on this list.

**What counts as an answer.** Either (a) a named position + layer + feature index + magnitude
ratio, with a set-to-zero/set-to-mean asymmetry and a language-invariance statement with CIs
over languages; or (b) a bounded negative — "no decoder position exceeds *k*× median at any
layer, 95% CI [.,.], n = 12 languages" — which is itself a finding, because it would say
encoder-decoder translation models *do not* build one.

---

### Q2 — Does SeamlessM4T's speech encoder build a constant, and is it shared with the text encoder?

**Question.** SeamlessM4T v2 has a **Conformer speech encoder (w2v-BERT 2.0) + length adaptor**
and a **separate NLLB-initialised text encoder**, feeding **one shared decoder**. Three sub-questions:
does the speech encoder build a high-norm frame (and is it the **first** frame, or a
silence/padding frame — Whisper pads every clip to 30 s, so there is always non-speech
material); does the text encoder build one; and **do the two write into the same feature
dimensions of the shared decoder's residual stream?**

**Why it matters for the interlingua question.** If speech and text write their constant to the
*same* dimensions, that is a concrete, mechanistic sense in which the model has one shared
scaffold across modalities — much sharper than an SVCCA curve. If they write to different ones,
that is a mechanistic account of the modality gap Lee et al. measured.

**Experiment.** Same detector, three passes: (a) speech encoder over FLEURS audio; (b) text
encoder over the matched FLoRes text; (c) the shared decoder under both conditions. Report the
feature-index overlap between (a) and (b) at the decoder, against a null of random index sets
of the same size. **Causal:** zero/mean-patch the dimension in the speech pass and measure
speech-to-text BLEU; do the same in the text pass and measure text-to-text BLEU. Cross-lingual
arm as in Q1.

**Cost.** SeamlessM4T-v2-large is 2.3B. **~25–40 GPU-hours**. Requires reading
`GenStepAware-2026-…` first, since it already works on this decoder.

**What counts as an answer.** A feature-index overlap statistic with a null, plus paired
causal deltas. A negative ("the speech and text encoders build constants in disjoint
dimensions") answers the interlingua question in the negative *mechanistically*, which is
publishable.

---

### Q3 — Speech–text convergence for Swahili / Igbo / Xhosa: Qwen-Omni vs SeamlessM4T

**Question.** Lee et al. (NAACL 2025) found length adaptation closes the cross-modal gap only
for high- and medium-resource languages. Their 30-language set includes Amharic and Swahili but
not Igbo, Xhosa, Efik, Twi. **Does the convergence story hold at all for the lab's languages,
and does it differ between an encoder-decoder built for translation (Seamless) and a
decoder-only omni-LLM (Qwen2.5-Omni / Qwen3-Omni)?**

**Experiment.** Replicate Lee et al.'s SVCCA protocol exactly (their §2), extend to the lab's
six languages using FLEURS where available and the lab's own recordings where not. Add the
**pronunciation control from Shim et al.** — otherwise any cross-lingual alignment number is
partly cognates. Add the magnitude/direction split from Xiang et al. (cosine *and* Euclidean).
Then one **causal** arm: the Wu ICLR 2025 intervention — steer in the shared middle-layer space
using the dominant data type (English text) and see whether the Swahili *speech* output moves.

**Cost.** Inference only. Qwen2.5-Omni-7B and SeamlessM4T-v2-large both fit on one A100.
**~30–50 GPU-hours** plus real analysis time; the replication step is the expensive part.

**What counts as an answer.** Per-language SVCCA curves with CIs over utterances, an explicit
statement of coverage (which languages, which had FLEURS data), and a yes/no on whether the
English-text steering transfers to Swahili speech. **Risk:** this is the most "extension of an
existing paper" of the three — it needs the pronunciation control and the causal arm to be
more than a replication.

---

### Q4 (smaller, as a fallback or a first month) — Is the repetition loop the constant failing?

**Question.** Adrian's text-side finding is that removing the constant makes the model loop on
function words. `Wu-2025-Beyond-Transcription…` localises Whisper's repetition hallucinations to
specific encoder–decoder interactions; `Wang-2025-Calm-Whisper…` shows 3 of 20 decoder heads
cause >75% of non-speech hallucination. **Are the "crazy heads" the sink heads?** I.e. do the
three heads Calm-Whisper identifies attend to the decoder-prefix sink position found in Q1?

**Experiment.** One forward pass over UrbanSound + LibriSpeech, per-head attention mass on the
prefix positions, cross-referenced with Calm-Whisper's head list (they release which heads).
**Causal:** mask the prefix position for those heads only.

**Cost.** **~5 GPU-hours.** This is a two-week result and a natural first deliverable that
de-risks Q1.

**What counts as an answer.** A per-head attention-mass table with the family size stated (20
heads × 32 layers = 640; correct for multiplicity), and a masking delta on hallucination rate.

---

**Recommendation.** Q1 → Q4 as the spine (Whisper, cheap, unclaimed, directly extends the
text-side result, and has a real cross-lingual arm), with Q2 as the stretch if Q1 is positive.
Q3 is the safest and the least novel.

---

## §6 Verification debt

Everything here that is not yet safe to cite.

**Marked `[unopened]` above — abstract only, must be opened before citing:**
`CascadeEquivalence-2026-Speech-LLMs-vs-Pipelines.pdf`,
`Omnilingual-SONAR-2026-CrossLingual-CrossModal-Embeddings.pdf`,
`TranslationEnhanced-2026-Speech-Encoder-Pretraining-SpeechLLMs.pdf`,
`AudioSAE-2025-Interpretable-Features-Audio-Latent-Spaces.pdf`,
`Whisper-2025-Internal-Word-Aligner.pdf`,
`LayerWise-2026-Probing-wav2vec2-Whisper-AAE.pdf`,
`insideSSL-2026-Model-Centric-SSL-Speech-Representations.pdf`,
`Orthogonality-2024-Speaker-Phonetic-SSL-Speech.pdf`,
`Anisotropy-2025-Neural-Representations-of-Speech.pdf`,
`ListeningWithAttention-2026-Entropy-Guided-Explainability-Audio.pdf`,
`HeardNotHeeded-2026-Paralinguistic-Encoding-Loss-ALMs.pdf`,
`ProsodicUnderuse-2026-Causal-Account-Audio-LMs.pdf`,
`LanguageControl-2026-Latent-Mechanisms-Multilingual-LMs.pdf`,
`EmotionNeurons-2026-Multilingual-Large-Audio-LMs.pdf`,
`Quantizing-Whisper-2025-Design-Choices.pdf`,
`EdgeASR-2025-LowBit-Quantization-ASR-Models.pdf`,
`PTQ-2026-LayerWise-Compensation-EncoderDecoder-ASR.pdf`,
`Darcet-2024-ICLR-Vision-Transformers-Need-Registers.pdf`,
`VisionEncoders-2026-ECCV-Activation-Quantization-Prefixing-Registers.pdf`,
`Sun-2024-COLM-Massive-Activations-in-LLMs.pdf`,
`NNsight-2024-NDIF-Model-Internals.pdf`, `nnterp-2025-Standardized-Interface-MechInterp.pdf`,
`Inseq-2023-ACL-Interpretability-Toolkit-Sequence-Generation.pdf`.

**Marked `[skimmed]` — abstract + intro only; results tables NOT read:**
`GenStepAware-2026-CrossModal-Representation-Control-SeamlessM4T.pdf` (**highest priority — it
may already answer part of Q2**), `FactualRecall-2026-Text-to-Speech-Multimodal-LMs.pdf`,
`AnatomyModalityGap-2026-Internal-States-Speech-LLMs.pdf`,
`ModalityGap-2025-SpeechText-Alignment-Mechanism-LSLM.pdf`,
`Interleaved-2026-Speech-LMs-Latently-Work-In-Text.pdf`,
`Wu-2025-Semantic-Hub-Hypothesis.pdf`, `AudioSAE-2026-EACL-…` (§1 only),
`Whisper-SAE-2026-…` (§1 only).

**Specific facts to verify before use:**
1. **Venue of `Cappellazzo-2025-Attention-Sinks-Massive-Activations-AVSR.pdf`.** The arXiv
   comment on v3 says "IEEE ICASSP 2026". Confirm against the ICASSP 2026 proceedings — the
   comment is author-supplied.
2. **`Whisper-Encoders-2025-Languages-Align-Phonetically-Semantically.pdf`** says "Submitted to
   Interspeech 2026" — **submitted, not accepted**. Cite as preprint until confirmed.
3. **`CascadeEquivalence-2026-…`** likewise says "submitted for review Interspeech 2026".
4. **`Orthogonality-2024-…`** comment says "Accepted to Interspeech" with no year; the arXiv
   date is 2024-06, so presumably Interspeech 2024. Confirm.
5. **`Wu-2025-Beyond-Transcription-Mech-Interp-ASR.pdf` filename is wrong** — the authors are
   Glazer et al. (aiOla Research), not Wu. A sibling agent named it. It is arXiv 2508.15882
   with **no venue stamp** — preprint. I did not rename it (not my file).
6. **The `<|tr|>` token in Wagner et al. — RESOLVED this session.** Whisper also has a `<|tr|>`
   *Turkish* language tag, so the reading mattered. Figure 1's axis labels read
   `<|sot|>, <|en|>, <|tr|>, <|nots|>` bottom-up, i.e. the standard decoder prompt
   `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`. **`<|tr|>` is the transcribe
   task token, the third prefix position, immediately after the language tag.** Verified by
   `pdftotext -layout`. Anyone rechecking should confirm against the rendered figure, not the
   text layer.
7. **TransformerLens speech support** was verified against the `main` branch of
   `loading_from_pretrained.py` on GitHub, not against a released wheel. Which released version
   first shipped `convert_hubert_weights` is **unverified**; the repo's `CLAUDE.md` pins 3.6.0
   and PyPI latest is 3.8.1.
8. **s3prl** row in §4 is from background knowledge, **not checked this session**.
9. **nnsight's actual behaviour on Whisper** (encoder-decoder, cross-attention, audio input) is
   asserted from its design, **not tested**. Test it before committing to it in a plan.
10. **Amharic/Swahili in Lee et al.** verified by grepping the PDF (`amh`, `Amharic`, `Swahili`
    all present in the Appendix C table). Igbo, Xhosa, Efik, Twi are **not** in the 30.
11. **Whisper's 30-second padding** (relevant to Q2's "is the high-norm frame a silence frame?")
    is background knowledge, not re-verified against `Radford-2022-Whisper-…` this session.
12. **Web search coverage is incomplete.** The session's 200-call WebSearch budget was exhausted
    by sibling agents after my 11th query, so §3's exhaustiveness rests on the arXiv API, which
    indexes abstracts only — a result buried in a paper's body without the term in its abstract
    would be missed. The `grep` over five full PDFs partly compensates. **Semantic Scholar and
    ACL Anthology full-text search were both unavailable (HTTP 429) and should be re-run.**
13. **Not searched at all:** ISCA Archive / Interspeech proceedings full text, ICASSP
    proceedings, and the SIGTYP/SIGMORPHON venues. Interspeech and ICASSP papers are where an
    audio-encoder outlier result would most plausibly hide, and the arXiv API only sees the
    subset that is also on arXiv.
