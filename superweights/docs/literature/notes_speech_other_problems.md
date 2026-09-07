# Open problems in speech translation, 2025–2026 — the ones the field names for itself

Scope note: this survey deliberately **excludes** the massive-activation / attention-sink /
outlier-dimension line entirely. Nothing below depends on it. Sibling notes in this directory
cover that thread.

Reading discipline used here, per the shared brief:
- **[read]** = I opened the PDF and read the relevant section(s).
- **[abstract]** = I read the abstract plus targeted greps for the numbers quoted; I did not
  read the whole paper.
- **[unopened]** = named but not fetched. Nothing in §2 or §3 rests on an `[unopened]` item.
- Venues are as printed on the PDF or as recorded in the ACL Anthology / ISCA Archive volume
  metadata I fetched. arXiv-only items are labelled **(preprint)**.
- 37 PDFs are in `papers/` (flat), indexed in `README.md` under
  "### Other open problems in speech translation".

The single most useful documents here are the two IWSLT findings papers. IWSLT is the venue
where the field writes down, once a year and in public, what it could not do. I read both at
length; most of §2 is downstream of them.

---

## §1 Paper by paper

### 1.1 The organisers' own problem lists

**Adelani et al. (2026), "Speech Translation and Metrics in 2026: Findings of the IWSLT
Campaign", IWSLT 2026 (ACL).** `IWSLT-2026-Findings-Speech-Translation-and-Metrics.pdf`
[read: §1, tracks II, V, VI, VII, VIII, X]

Ten tracks, >30 teams. The 2026 edition is the year the campaign pivoted to **speech
generation and to metrics** — three of the ten tracks are new S2S/voice tracks and one is a
brand-new Quality Estimation track. What the organisers flag, with numbers:

- **African/Celtic S2S (Track VII, new).** Hausa, Igbo, Yorùbá → English, on a newly recorded
  corpus (NaijaS2ST: ~75 h recorded, 5,000/500/500 sentences per language, 3 speakers per
  training sentence, up to 70 speakers per language, held-out speakers in dev/test, English
  side recorded by L1 speakers of the African languages in Northern and Southern Nigerian
  accents). Motivation stated as "linguistic and cultural fidelity": proper nouns, named
  entities, culturally grounded expressions. S2TT results (Table 12), spBLEU / chrF++ /
  SSA-COMET:

  | System | Hausa | Igbo | Yorùbá |
  |---|---|---|---|
  | SeamlessM4T mono-FT (supervised upper bound) | 18.6 / 41.9 / 54.9 | 17.6 / 39.2 / 52.3 | 21.1 / 43.5 / 60.3 |
  | Cascade (OmniASR LLM 1B + NLLB-200) | 17.3 / 42.0 / 54.1 | 11.0 / 33.8 / 42.9 | 17.0 / 38.1 / 50.6 |
  | AURA-ST (frozen encoders + LoRA'd Gemma) | 5.2 / 23.2 / 33.9 | 4.6 / 16.7 / 20.6 | 19.5 / 41.0 / 57.1 |

  Two things jump out. (a) The **cascade is at parity for Hausa (17.3 vs 18.6) and collapses
  for Igbo (11.0 vs 17.6)** — the organisers attribute this to ASR error propagation that is
  "increasingly difficult to recover from in lower-resource settings", but no error
  decomposition is reported. (b) The track received **exactly one submission**. The organisers
  close: "the nuances in tone and diacritic … are much more difficult to detect than in text."
- **Metrics / QE (Track X, new).** 5 teams, 14 systems, plus 6 organiser baselines. En→De and
  En→Zh only. Best segment-level Kendall τb: HydraQE 34.5 (speech-based), TieCal 33.0
  (text-based), Lexilogic 31.4 — against **human inter-annotator agreement of 45.8 (EnDe) and
  46.6 (EnZh)**. System-level SPA is 93–97 and therefore near-useless as a discriminator
  (35.5% of the data has only two systems to rank). Two negative results the organisers state
  plainly: **text-based metrics are competitive with or better than speech-based ones at the
  segment level**, and **swapping human source transcripts for Whisper large-v3 transcripts
  costs at most 1.3 τb points** (e.g. TieCal 47.7 → 46.5). Their explanation: "the test data
  does not expose speech-specific phenomena, such as prosody, speaking style, or speaker
  gender, that would require audio input". Domain matters more than modality: CallCenter is
  the hardest domain (TieCal 20.9, human 42.4) and TVSeries the easiest (TieCal 38.0, human
  47.7).
- **Simultaneous (Track V).** Second consecutive year on **raw unsegmented audio**, no
  pre-segmentation. Two latency regimes (0–2 s, 2–4 s) assigned by LongYAAL. New this year: an
  **Extra Context** sub-track (the ACL paper PDF is given to the system). Findings: all but one
  submission were **cascaded**; higher latency does not buy proportionally better quality for
  the top systems; extra context helps almost everywhere (MLLP-VRAIN +2.75 COMET averaged over
  three directions) and helps specifically on entities and domain vocabulary; the
  YouTube-sourced YODAS test set induces ~1.0 s higher latency than talk/news sets
  (out-of-domain robustness); computation-aware latency is ~0.3 s above the
  computation-unaware figure on average, and the *smallest* system (1B, end-to-end) shows the
  *largest* gap (>0.4 s), contradicting the expectation that small = fast. COMET and chrF/BLEU
  disagree on ranking in En→Zh.
- **Cross-lingual voice cloning (Track VIII, new).** 5 teams, targets Ar/Zh/Fr, reference
  speakers from ACL 2023 talks. Clear **content-consistency ↔ speaker-similarity trade-off**;
  Arabic hardest, French easiest; speaker similarity spans 0.479–0.813. The **prosody-similarity
  metric is saturated at 0.980–0.997 across every system and language** — i.e. the metric does
  not discriminate, which is itself the finding. Organisers' own future work: expand language
  families, require systems to *translate as well as clone* (currently target text is given),
  and add code-switching and terminology consistency.
- **Low-resource (Track II).** 10 pairs. Notable individual results: Mapudungun–Spanish
  0.34–0.82 BLEU ("all submissions struggle to produce meaningful outputs"); Central
  Kurdish–English 21.09 BLEU (LIUM) vs 0.16 (SLC); Fleurs-Badini (new Northern Kurdish variant
  dataset, 15 h 40 m, 45 speakers) fine-tuned Whisper 5.24 BLEU.

**Agostinelli et al. (2025), "Findings of the IWSLT 2025 Evaluation Campaign", IWSLT 2025
(ACL).** `IWSLT-2025-Findings-Evaluation-Campaign.pdf` [read: §1, §2, low-resource track]

Seven tracks, 32 teams. The low-resource section contains the campaign's bluntest statement:
Quechua–Spanish jumped 19.7 → 26.7 BLEU (through added data), but Bemba–English,
Bhojpuri–Hindi, Irish–English and **both Arabic dialects were flat or worse**, and the
organisers write that "we have perhaps reached a performance ceiling of sorts in the current
datasets under the current data-scarce conditions" — while noting this ceiling still "lags
substantially behind" high-resource quality. Their rule of thumb: good ST needs **>50 h of
high-quality translated speech**. Almost every submission was *unconstrained*, i.e. pretrained
multilingual models are the only viable route. On the winning KIT system: TTS-generated
synthetic speech "was found to be ineffective, primarily due to the lack of high-quality
speech data, leading to an under-trained TTS system"; back-translation/forward-translation/
paraphrase synthetic *text* remains effective; dialect-ID filtering of a general Arabic corpus
was the winning trick for LIA on North Levantine.

### 1.2 Simultaneous and streaming

**Papi, Polák, Macháček & Bojar (2025), "How 'Real' is Your Real-Time Simultaneous
Speech-to-Text Translation System?", TACL 2025.**
`Papi-2025-How-Real-Is-Your-RealTime-SimulST.pdf` [read: abstract, §1, §3.2, §4 highlights]

A literature review of **110 SimulST papers**, and the most quotable indictment in the area:
**81.8% of papers use pre-segmented audio**, and of those **97.7% use *gold* segmentation**;
only **20 of 110** either solve concurrent audio segmentation (14) or are segmentation-free
(6); only **2 papers ever** measured what happens when gold segmentation is swapped for
automatic segmentation. Over **65% of papers mix and match** the terms
simultaneous/streaming/online/real-time. They propose a standard taxonomy and terminology and
argue the field's results do not transfer to deployment. This is the paper that made IWSLT move
its simultaneous track to unsegmented audio.

**Xue, Ouyang & Li (2026), "A Practical Evaluation Method for Long-Form Simultaneous
Speech-to-Speech Translation", IWSLT 2026 (ACL).**
`Xue-2026-Practical-Evaluation-LongForm-Simultaneous-S2ST.pdf` [abstract + method]

The S2ST analogue of the above. Existing SimulS2ST evaluation assumes short or pre-segmented
speech; prior long-form protocols (pBAL, StreamAtt/StreamLAAL, LongYAAL) make assumptions that
break for end-to-end systems. They run ASR + forced alignment on the *generated target speech*
to recover token timestamps, then align target sentences to source sentences with a
sentence-embedding aligner, enabling sentence-level YAAL and xCOMET. Headline finding: **current
SimulS2ST systems suffer substantial latency accumulation on long speech** — i.e. the reported
latencies of short-form benchmarks are not what a user experiences in a 40-minute talk.

**Zhang, Yang & Nakamura (2026), "Redefining Machine Simultaneous Interpretation: From
Incremental Translation to Human-Like Strategies", IWSLT 2026 (ACL).**
`Zhang-2026-Redefining-Machine-Simultaneous-Interpretation.pdf` [abstract + §1]

Position + system paper: SimulMT is stuck optimising a quality/latency point on an incremental
translation curve, whereas human interpreters use *strategies* — the salami technique
(minimal information-complete segments), rephrasing, summarisation, selective omission. Frames
"what should the system output under partial input" as a separate open problem from "when
should it emit".

**Zhang et al. (2024), "StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task
Learning", ACL 2024.** `Zhang-2024-StreamSpeech.pdf` [abstract]

The reference direct Simul-S2ST system: policy and translation learned jointly in one
multi-task "All-in-One" model (ASR, ST, synthesis, offline and simultaneous), SOTA on CVSS.
Included as the baseline any streaming-S2ST project must beat or at least cite.

### 1.3 Evaluation: quality estimation without references, and metric validity

This is the most active and most clearly-stated open problem of the two campaigns.

**Han & Duh (2024), "SpeechQE: Estimating the Quality of Direct Speech Translation", EMNLP
2024.** `Han-2024-SpeechQE-Estimating-Quality-Direct-ST.pdf` [abstract + §1]

Defines the task, builds the benchmark, compares cascaded (ASR → text QE) against end-to-end
(Whisper encoder + modality adapter + TowerInstruct-7B with LoRA). End-to-end wins. Their
argument, which the 2026 campaign inherited: **QE for speech must be studied as a separate
problem from text QE**, not as text QE with a transcription step bolted on.

**Züfle, Liu, Zouhar & Niehues (2026), "Why We Need Speech to Evaluate Speech Translation",
arXiv:2605.28227 (preprint).** `Zufle-2026-Why-We-Need-Speech-To-Evaluate-ST.pdf` [read:
abstract, §2, §4.1, conclusion, App. B.3]

The sharpest negative in the area. They meta-evaluate text and speech QE metrics on two
*contrastive* sets that isolate speech-specific phenomena: **MuST-SHE** (1,612 pairs, speaker
gender agreement, en→fr/es/it) and **ContraProST** (15,972 pairs, en→de/es/ja, five
phenomena: sentence stress, intonation, prosodic breaks, emotional prosody, politeness). Text
metrics score exactly **50.0** — chance — by construction. Speech-based metrics score
**26.1, 20.0, 38.2, 51.0, 51.5, 51.1**, i.e. *at or below chance*. They then train SpeechCOMET
(COMETKiwi + SONAR/Whisper speech encoder) and evaluate a fine-tuned SpeechLLM judge: both
match or beat text COMET on *standard* QE (SpeechLLM 40.4 → 47.1 τb with IWSLT fine-tuning;
SPA 64.1 → 88.5) yet **remain at chance on MuST-SHE and show no ContraProST sensitivity**.
Three diagnosed causes: speech-specific features are not preserved by the encoders; models
learn to ignore the audio; the training data contains too few relevant examples.

**Zarzu & Zouhar (2026), "Hurdles of Automatic Metric for Speech Translation Evaluation",
IWSLT 2026 (ACL).** `ZarzuZouhar-2026-Hurdles-Automatic-Metric-ST-Evaluation.pdf` [abstract +
results]

The controlled ablation behind the campaign's headline. Two metric paradigms
(COMET+audio regression; few-shot Phi-4-multimodal prompting), each run text-only,
speech-only, and text+speech. Text-only Phi-4 scores **23.2** τb against speech-only 15.5 and
text+speech 17.6. Their diagnosis is *not* "audio is useless" but "audio is unreliable **on
this data**": noise, audio–transcript misalignment, and evaluation corpora that are technical
and low-prosody.

**Krahn & Fosler-Lussier (2026), "HydraQE", IWSLT 2026 (ACL).** `Krahn-2026-HydraQE-Speech-QE.pdf`
[abstract] — the winning system: Qwen3-ASR-1.7B backbone, sparsemax scalar mix over all layers,
bidirectional re-encoder, three heads (human DA, MetricX-24 pseudo-labels, xCOMET
pseudo-labels), curriculum from synthetic/silver to human. First speech-based system to beat
cascaded text baselines in this setting — though only by 34.5 vs 33.0.

**Dale & Costa-jussà (2024), "BLASER 2.0", Findings of EMNLP 2024.** `Dale-2024-BLASER-2.0.pdf`
[abstract] — SONAR-based, 202 text / 57 speech languages, reference-based and reference-free
variants, usable at sentence level for hallucination detection and corpus filtering. Included
because it is the default speech-QE baseline everywhere — and it lost to text COMETKiwi at
IWSLT 2026 (22.6 vs 32.9 τb).

**Post & Hoang (2025), "Effects of automatic alignment on speech translation metrics", IWSLT
2025 (ACL).** `PostHoang-2025-Effects-Automatic-Alignment-ST-Metrics.pdf` [abstract]

Everyone evaluates ST with gold segmentation because otherwise you must realign outputs to
references; the tool for that (mwerSegmenter) is "noisy and not well understood". Using WMT24's
eleven language tasks they show **automatic realignment has minimal effect on COMET-level
system rankings** and releases a modernised `mweralign` Python module. A useful *negative*: the
segmentation problem is real for systems but mostly *not* a metric-ranking confound.

**Koneru, Huck, Exel & Niehues (2025), "Quality-Aware Decoding", IWSLT 2025 (ACL).**
`Koneru-2025-Quality-Aware-Decoding.pdf` [abstract] — folds QE into decoding rather than
N-best reranking or MBR. The 2026 low-resource track shows the applied version: ADAPT-MTU used
"QE Fusion with COMET-Kiwi" to select from beam candidates for Bhojpuri–Hindi.

**Li et al. (2025), "SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for
Under-Resourced African Languages?", EMNLP 2025.** `Li-2025-SSA-COMET-African-MT-Evaluation.pdf`
[abstract] — SSA-MTE: >73,000 sentence-level human annotations across 14 African language
pairs (News), and SSA-COMET / SSA-COMET-QE trained on it, plus GPT-4o / Claude-3.7 prompting
baselines. This is the metric IWSLT 2026 scores its African S2TT track with; it is a **text**
metric applied to speech output.

### 1.4 Long-form and document-level

**Papi et al. (2026), "MCIF: Multimodal Crosslingual Instruction-Following Benchmark from
Scientific Talks", ICLR 2026** (PDF used is arXiv:2507.19634v2).
`Papi-2026-MCIF-Multimodal-Crosslingual-Instruction-Following.pdf` [abstract]

First multilingual human-annotated benchmark over scientific talks spanning speech, vision and
text, **short- and long-form**, four macro-tasks (recognition, translation, QA,
summarisation). Built precisely because existing benchmarks are English-only, single-modality,
short-form or unannotated. It is the dev/test substrate for the IWSLT simultaneous and
instruction-following tracks.

**Ugan et al. (2026), "Multilingual Long-Form Speech Instruction Following: KIT's Submission to
IWSLT 2026", IWSLT 2026 (ACL).** `Ugan-2026-Multilingual-LongForm-Speech-Instruction-Following.pdf`
[abstract + §1]

States the long-form problem crisply: Whisper natively supports 30 s per pass; newer models
(Phi-4, Qwen2.5-Omni) remove the architectural limit but **lack long-form audio in training**,
"a gap that manifests as significant performance degradation even on basic ASR". Their fixes:
augment short-form data into long-form, temperature-scaled task interleaving, re-ranking.

**Alabi et al. (2025), "AFRIDOC-MT: Document-level MT Corpus for African Languages", EMNLP
2025.** `Adelani-2025-AFRIDOC-MT-Document-Level-African.pdf` [abstract]

Text, not speech, but it is the document-level anchor for the lab's languages: 334 health +
271 IT news documents, English ↔ Amharic/Hausa/Swahili/Yorùbá/Zulu. NLLB-200 is the best NMT
model, GPT-4o the best general LLM; **models trained on sentences fail to generalise to
documents**, and LLMs show under-generation, over-generation, **repetition of words and
phrases**, and off-target translation.

### 1.5 Low-resource: end-to-end vs cascaded, and where the errors come from

**Zevallos et al. (2026), "CATENG Submission … IWSLT 2026", IWSLT 2026 (ACL).**
`Zevallos-2026-CATENG-Cascaded-vs-EndToEnd.pdf` [abstract]

Catalan–English, 15 h task data + large parallel text. Three systems: ConMamba ASR + fine-tuned
NLLB-200; Whisper-v3 + NLLB (**44.7 BLEU / 65.1 chrF**, best); end-to-end SpeechT5 with
augmentation. Conclusion in their own words: cascades beat end-to-end, and **performance is
constrained by ASR quality rather than MT capacity**.

**Ortega et al. (2026), "Team QUESPA … IWSLT 2026", IWSLT 2026 (ACL).**
`Ortega-2026-QUESPA-Quechua-LLM-Prompting.pdf` [abstract; numbers cross-checked against the
2026 Findings low-resource section]

Quechua–Spanish. They benchmarked GPT-5, Gemini 3, Claude, DeepSeek-V3 and Qwen with guided
Spanish prompts: **none beat the previous year's fine-tuned NLLB-200 baseline (19.5 BLEU /
23.5 chrF); the best prompt-based result was 10.8 BLEU (Gemini 3 Flash)**. Failure modes
named: hallucination and dialectal confusion between Quechua variants. They also added SIDON
audio enhancement and a Quechua Collao corpus. The team's own best-ever score for this pair is
26.7 BLEU.

**Dauvet, Ma, Ojo & Adelani (2025), "Reassessing Speech Translation for Low-Resource Languages:
Do LLMs Redefine the State-of-the-Art Against Cascaded Models?", MRL 2025 (workshop at EMNLP).**
`Reassessing-2025-LowResource-ST-Do-LLMs-Redefine-SOTA.pdf` [abstract + setup]

Ten FLEURS low-resource pairs (five Indic, five African), 10–30+ h each, X→English, six
fine-tuning strategies, three paradigms (Whisper+NLLB cascade, SeamlessM4T, audio LLMs GPT-4o
Audio and Gemini 2.0 Flash). Findings: **LLMs struggle on LRLs**; **more AST fine-tuning data
is not always better**; a "2-stage with ASR corrector" recipe (mT5-base correcting ASR output
before MT) gives **54.2% relative WER reduction on African languages and up to 5.8× BLEU**
(2.6× for Indic) with no extra data and no architecture change. This is the single most
directly transferable recipe in this survey for the lab's setting.

**HB et al. (2026), "AURA-ST", IWSLT 2026 (ACL).** `HB-2026-AURA-ST-African-LowResource-E2E.pdf`
[abstract; architecture read from the 2026 Findings §VII.4] — frozen w2v-BERT 2.0 + ResNet34
dual-stream encoders → conv subsampler → **prefix prompt** into a frozen Gemma-4-E2B, LoRA on
the MLP layers only (attention-LoRA "was ineffective due to architectural constraints"). Second
best on Yorùbá→English and beats the cascade there by +2.5 spBLEU / +6.5 SSA-COMET, while
scoring 5.2/4.6 spBLEU on Hausa/Igbo. The variance across three closely-related-resource
languages is unexplained.

**Bär, De Marco & Labaka (2025), "Swiss German Speech Translation and the Curse of
Multidialectality", IWSLT 2025 (ACL).** `Bar-2025-Swiss-German-ST-Curse-Multidialectality.pdf`
[abstract] — ASR pre-training **costs 1.48 BLEU**; joint Swiss + Standard German training
**costs 2.29 BLEU**; training on multiple dialects jointly degrades single-dialect
performance, though small amounts of dialectal variability help the lowest-resource dialects.
A dialect-scale "curse of multilinguality".

### 1.6 Data: pseudo-labelling vs synthetic speech (an unresolved contradiction)

**Mohammadamini, Sini, Tahon & Laurent (2025), "Scaling pseudo-labeling data for end-to-end
low-resource speech translation (the case of Kurdish)", Interspeech 2025.**
`Mohammadamini-2025-Scaling-PseudoLabeling-LowResource-ST.pdf` [abstract + setup]

4,000 h of clean Kurdish audiobook speech collected; 3,200 h pseudo-labelled through
segmentation → fine-tuned Seamless v2 ASR → a new 222k-pair CKB→EN MT model; **20.68 BLEU on
FLEURS**, from a language whose only prior ST resource was FLEURS' 13 h.

**Mohammadamini & Tahon (2026), "LIUM Submission for IWSLT 2026 Low-Resource Track", IWSLT
2026 (ACL).** `Mohammadamini-2026-LIUM-LowResource-Pseudolabel-vs-Synthetic.pdf` [abstract]

Head-to-head of the two augmentation pipelines on the same pair. Pseudo-labelling: **25.73
BLEU dev / 21.09 test**. TTS-synthesised source speech: their stated conclusion is that
"using synthetic speech generation for real-world applications such as spontaneous speech
translation remains particularly challenging and not fruitful."

**Li et al. (2025), "KIT's Low-resource Speech Translation Systems for IWSLT2025", IWSLT 2025
(ACL).** `Li-2025-KIT-LowResource-Synthetic-Data.pdf` [abstract]

The counter-case, and I want to flag the contradiction explicitly because the 2025 Findings
summarises this paper as a negative. KIT's own abstract says: for North Levantine, "a system
trained solely on synthetic data slightly surpasses the cascaded system trained on real data",
and TTS-generated synthetic speech demonstrates "the benefits of synthetic data in improving
both ASR and ST performance **for Bemba**". The Findings' "found to be ineffective" verdict is
scoped to a specific under-trained TTS in the apc setting. **So the honest state of the
literature is: synthetic speech for ST works sometimes, and nobody has characterised when.**

### 1.7 Rare words, named entities, terminology

**Gaido, Rodríguez, Negri, Bentivogli & Turchi (2021), "Is 'moby dick' a Whale or a Bird? Named
Entities and Terminology in Speech Translation", EMNLP 2021.**
`Gaido-2021-Named-Entities-Terminology-in-ST.pdf` [abstract + §1]

Still the reference numbers: on NEuRoparl-ST (en→es/fr/it) ST systems correctly translate
**75–80% of terms**, **65–70% of NEs**, and only **37–40% of person names**. Their framing —
NE/terminology errors produce "blatant (meaningless, hilarious, or even offensive) errors,
which jeopardize users' trust" — is exactly the IWSLT African track's stated motivation, 5
years later.

**Li, Liu & Niehues (2024), "Optimizing Rare Word Accuracy in Direct Speech Translation with a
Retrieval-and-Demonstration Approach", EMNLP 2024.**
`Liu-2024-Rare-Word-Accuracy-Retrieval-Demonstration-ST.pdf` [abstract]

Prepend retrieved examples containing the rare word, in-context-learning style, to a direct ST
model. **+17.6% rare-word accuracy with gold examples, +8.5% with retrieved**; the
speech-to-speech retriever beats text retrievers and is more robust to unseen speakers. The
best-evidenced *fix* for the NE problem, never tested on African languages.

### 1.8 Code-switching, accent, dialect

**Yan et al. (2025), "CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset",
Interspeech 2025.** `Yan-2025-CS-FLEURS-CodeSwitched-Speech-Dataset.pdf` [abstract + §5]

Four test sets, **113 code-switched language pairs over 52 languages**, plus 128 h of
generative-TTS training data over 16 X-English pairs. Whisper-large-v3 **CER is >2× higher on
CS-FLEURS than on monolingual FLEURS**, and **3× higher on distinct-script pairs than
same-script pairs**. Honest about its own construction: LLM-generated code-switched text needed
human validation, which rejected ~10% of sentences for 14 X-English pairs, "reasonable to
expect a higher reject rate for lower resourced languages"; an align-then-swap method gives
more consistent code-mixing index (σ 60% lower) than LLM generation.

**Bafna & Wiesner (2025), "LID Models are Actually Accent Classifiers", Interspeech 2025.**
`Bafna-2025-LID-Models-Are-Accent-Classifiers.pdf` [abstract + §1]

Language-ID models degrade badly on L2-accented speech by predicting *the accent's source
language* (Filipino-accented English → Tagalog/Cebuano). Diagnosis: pooled speaker-ID-style
architectures (ECAPA-TDNN, pooled SSL) treat sequences as exchangeable and key on phoneme
inventory / short phonotactics rather than lexis or syntax. This is upstream of every
multilingual ST pipeline that routes by detected language.

**Blaschke, Winkler, Förster, Wenger-Glemser & Plank (2025), "A Multi-Dialectal Dataset for
German Dialect ASR and Dialect-to-Standard Speech Translation", Interspeech 2025.**
`Blaschke-2025-German-Dialect-ASR-and-Dialect-to-Standard-ST.pdf` [abstract]

Betthupferl: 4 h read speech in Franconian/Bavarian/Alemannic + 0.5 h Standard German, with
**both** dialectal and standard transcriptions. The interesting problem they surface: the model
sometimes normalises grammatical differences and often does not, so "correct output" for
dialect→standard ST is itself underdefined.

### 1.9 Gender, bias, and who the system is for

**Attanasio, Savoldi, Fucci & Hovy (2024), "Twists, Humps, and Pebbles: Multilingual Speech
Recognition Models Exhibit Gender Performance Gaps", EMNLP 2024.**
`Attanasio-2024-Gender-Performance-Gaps-Multilingual-ASR.pdf` [abstract + method]

Two multilingual ASR models, three datasets, 19 languages, eight families. Clear gender WER
gaps whose *direction varies by language and model*, and — the important part — **no
significant correlation with acoustic or lexical features of the test data**. What does
correlate is a **probe**: the easier it is to decode speaker gender from internal states, the
smaller the gap (favouring female speakers).

**Fucci, Gaido, Negri, Bentivogli, Martins & Attanasio (2025), "Different Speech Translation
Models Encode and Translate Speaker Gender Differently", ACL 2025 (short).**
`Savoldi-2025-ST-Models-Encode-Speaker-Gender-Differently.pdf` [abstract + §4.1]

Probing across architectures on MuST-C, en→fr/it/es. Classic encoder-decoder ST **encodes
speaker gender strongly** (probe macro-F1 up to **96.19**); newer speech-encoder + MT +
adapter stacks (Seamless, ZeroSwot) **do not** (≤ **61.80**). Consequence: low gender encoding
→ masculine default in translation, a bias that is *worse in the newer architecture*. An
architecture-level regression that no BLEU/COMET number would surface.

**Conti, Fucci, Gaido, Negri, Wisniewski & Bentivogli (2026), "Voice, Bias, and Coreference:
An Interpretability Study of Gender in Speech Translation", LREC 2026.**
`Gaido-2026-Voice-Bias-Coreference-Gender-in-ST.pdf` [abstract + results]

Mechanism study across en→es/fr/it. Transformer ST gets speaker-referring gender right
**77.1–80.6% for feminine and 91.4–94.4% for masculine** terms; Conformer models are much worse
(**39.2–49.8%** feminine). Models do **not** replicate term-specific training-data
associations — for 85.5–92.1% of feminine outputs the produced form is the *less* prevalent one
in training — they learn a broad masculine prevalence (0.68–0.71 average) that acoustic input
can override. Occluding the top **1–20%** of salient input features flips the predicted gender
in **37–47%** of examples.

**Savoldi et al. (2025), "Mind the Inclusivity Gap: Multilingual Gender-Neutral Translation
Evaluation with mGeNTE", EMNLP 2025.** `Savoldi-2025-mGeNTE-Gender-Neutral-Translation.pdf`
[abstract] — en→es/de/it/el: instruction-following LMs **recognise when neutrality is
appropriate but cannot consistently produce it**. Text MT, included as the target-side
counterpart of the ST gender problem.

**Attanasio, Savoldi, Chechelnitsky, Negri, Carpuat & Martins (2026), "Does Speech Translation
Meet Users' Needs? An English to Portuguese Study Across Demographics", EAMT 2026.**
`Savoldi-2026-Does-ST-Meet-Users-Needs.pdf` [abstract] — the Ouvia project: crowdworkers from
different sociodemographic groups produce spoken requests and self-assess quality,
satisfaction and reliability. A project description, not results yet. Cited because it names
the validity question the metric papers keep deferring.

### 1.10 Expressivity, speaker identity, isochrony

**Ahtasam, Jamaluddin & Nadeem (2026), "Balancing Linguistic Intelligibility and Speaker
Identity in Zero-Shot Cross-Lingual Voice Cloning", IWSLT 2026 (ACL).**
`Ahtasam-2026-Balancing-Intelligibility-Speaker-Identity-Voice-Cloning.pdf` [abstract]

Four SOTA CLVC systems (autoregressive vs diffusion/flow-matching), English source speakers
from ACL-60/60, targets ar/zh/fr/de/ru/ja. The trade-off between content accuracy and speaker
identity is architecture-dependent; **Arabic is consistently the hardest** target under
zero-shot transfer.

**Subramanian et al. (2025), "Length Aware Speech Translation for Video Dubbing", Interspeech
2025.** `Subramanian-2025-Length-Aware-ST-Video-Dubbing.pdf` [abstract]

Rather than constraining duration during translation, generate short/normal/long candidates
(LSST) and let the target speaker's duration model pick; plus Length-Aware Beam Search for
on-device real-time decoding. The concrete state of the art on isochrony for direct ST.

---

## §2 Problem table

"Named by" cites where the field itself states the problem. "Missing" is what nobody has done.

| # | Problem | Evidence (numbers) | Named by | What is missing |
|---|---|---|---|---|
| P1 | **Speech QE metrics are blind to speech.** Adding audio to a reference-free metric does not help; metrics are at chance on gender and prosody contrast sets | Best segment τb 34.5 vs human 45.8/46.6 (IWSLT26); text-only Phi-4 23.2 > text+speech 17.6 (Zarzu); MuST-SHE contrastive accuracy 20.0–51.5 vs 50.0 chance (Züfle) | IWSLT 2026 Track X conclusion; Züfle 2026; Han 2024 | Evaluation data that *contains* prosody/gender/tone contrasts; any evidence at all outside En→De/Zh; encoders that preserve paralinguistics through a QE objective |
| P2 | **Simultaneous ST is evaluated in a setting that does not exist.** Gold pre-segmentation, unbounded audio ignored, latency terminology incoherent | 81.8% pre-segmented, 97.7% of those gold; 20/110 handle unbounded audio; 2/110 ever swapped gold for automatic segmentation; >65% mix terms (Papi TACL 2025) | Papi 2025 (TACL); IWSLT 2025+2026 moved to unsegmented audio | Long-form latency protocols that survive contact with S2ST (Xue 2026 is a first attempt); computation-aware latency reported by default |
| P3 | **Latency accumulates in long-form S2ST**, and small models are not fast | Xue 2026: "substantial latency accumulation on long speech"; IWSLT26: computation-aware latency +0.3 s average but +0.4 s for the *1B end-to-end* system; YODAS +1.0 s vs talks | Xue 2026; IWSLT 2026 Track V | Anything at all for S2ST outside En→X talks; no African/Indic long-form streaming numbers exist |
| P4 | **Long-form is a training-data gap, not an architecture gap** | Whisper 30 s window; Phi-4/Qwen2.5-Omni have no limit but "significant performance degradation even on basic ASR" (Ugan 2026, citing MCIF) | Ugan 2026; MCIF (ICLR 2026) | Long-form audio in training mixes; document-level context for speech (AFRIDOC-MT does this for text only) |
| P5 | **Cascade vs end-to-end for low-resource is unresolved and error attribution is missing** | Cascade at parity for Hausa (17.3 vs 18.6) and collapses for Igbo (11.0 vs 17.6); CATENG: cascade wins, ASR-bound not MT-bound; AURA-ST beats the cascade on Yorùbá (+2.5 spBLEU) but scores 4.6 on Igbo | IWSLT 2026 Track VII; Zevallos 2026; Dauvet 2025 | Nobody has run the oracle-transcript ablation that would say *where* the loss is. The 2-stage ASR-corrector (54.2% rel. WER, 5.8× BLEU) is the one candidate fix and it is untested on the IWSLT African data |
| P6 | **Frontier LLMs lose to a fine-tuned NLLB-200 on genuinely low-resource pairs** | GPT-5/Gemini 3/Claude/DeepSeek/Qwen best 10.8 BLEU vs fine-tuned NLLB-200 19.5 (Quechua–Spanish); audio LLMs lose to cascades on 10 FLEURS LRLs | Ortega 2026; Dauvet 2025 | Whether this holds for the lab's languages, and whether it is a tokenizer, data or prompting failure |
| P7 | **Tone and diacritics.** African-language ST loses information that is lexical in the source and invisible in the transcript | "the nuances in tone and diacritic … are much more difficult to detect than in text" — IWSLT 2026 Track VII conclusion. No measurement accompanies the claim | IWSLT 2026 Track VII | A measurement. Nobody has quantified tone/diacritic error rates in Yorùbá/Igbo ST, nor tested whether audio-conditioned QE helps where tone *is* lexical (the exact case Zarzu/Züfle did not have) |
| P8 | **Synthetic TTS speech for ST training: works sometimes, nobody knows when** | KIT 2025: helps Bemba ASR+ST, synthetic-only beats cascade for apc. LIUM 2026: "not fruitful" for Kurdish. IWSLT 2025 Findings: "ineffective … under-trained TTS". Pseudo-labelling reliably works: 3,200 h → 20.68 BLEU; 21.09 BLEU test | IWSLT 2025 + 2026 Findings; Li 2025; Mohammadamini 2025/2026 | A controlled sweep over TTS quality × amount × language. The confound (TTS quality) is named but never measured |
| P9 | **Named entities, person names, terminology** | 37–40% person-name accuracy in ST (2021, en→es/fr/it, high-resource); retrieval-and-demonstration recovers +17.6%/+8.5% | Gaido 2021; Li/Liu 2024; IWSLT 2026 Track VII motivation; IWSLT 2026 Track V (extra context helps entity translation) | No NE-annotated ST benchmark for any African language; the retrieval fix has never been tried outside en→de |
| P10 | **Code-switching** | Whisper CER >2× on CS-FLEURS vs FLEURS; 3× on distinct-script pairs | Yan 2025; IWSLT 2026 voice-cloning future work | CS data for Nigerian languages specifically; CS-FLEURS' lower-resourced subset is concatenative TTS, not real speech |
| P11 | **Accent and dialect** | LID predicts accent not language; joint dialect training costs 2.29 BLEU (Swiss German); Arabic dialect + Nigerian English accents both flagged | Bafna 2025; Bär 2025; Blaschke 2025; IWSLT NaijaS2ST includes two Nigerian English accent varieties by design | Whether the accented-English side of NaijaS2ST changes anything — the corpus was built to allow this analysis and nobody has done it |
| P12 | **Gender: an architecture-level regression** | Probe F1 96.19 (enc-dec) vs ≤61.80 (adapter stacks) → masculine default; 37–47% of predictions flip under 1–20% feature occlusion; ASR gender WER gaps not explained by acoustics | Fucci 2025 (ACL); Conti 2026 (LREC); Attanasio 2024 | Everything outside Romance targets. Whether the concept transfers to noun-class languages (Bantu) is not even posed |
| P13 | **Expressivity vs intelligibility in S2S** | Speaker similarity 0.479–0.813; content ↔ identity trade-off; **prosody similarity saturated at 0.980–0.997** across all systems | IWSLT 2026 Track VIII | A prosody metric that discriminates. The current one is measuring nothing |
| P14 | **A performance ceiling in low-resource ST** | 4 of the returning IWSLT pairs flat or worse year-on-year; >50 h rule of thumb; Mapudungun 0.34–0.82 BLEU | IWSLT 2025 Findings §7 | Whether the ceiling is data, evaluation, or metric-driven. The organisers say "current datasets under current data-scarce conditions" and leave it there |
| P15 | **Evaluation validity for users, not for leaderboards** | En→Pt study across demographics is a *project description*, no results | Attanasio 2026 (EAMT) | Any human-centred ST evaluation for an African language |

**Two things that are *not* open, and should not be proposed as if they were:**
mwerSegmenter realignment does not meaningfully distort COMET-level system rankings (Post &
Hoang 2025); and swapping human for Whisper source transcripts barely moves QE metrics on
English source speech (IWSLT 2026 Track X, ≤1.3 τb) — so "ASR quality is the QE bottleneck" is
falsified *for English*, and remains open for everything else.

---

## §3 Ranked shortlist — MS-sized projects with the lab's assets

Assets assumed: BYU ORC A100/H100, single-GPU SLURM; a lab-fine-tuned NLLB-200 serving
Xhosa/Igbo/Efik/Swahili/Twi/Yorùbá; in-house African TTS voices (incl. a Swahili recording
project); FLORES/FLEURS/CommonVoice; the newly released IWSLT 2026 **NaijaS2ST** corpus
(Hausa/Igbo/Yorùbá ↔ English, ~75 h, held-out speakers, two Nigerian English accent varieties).
One semester ≈ 125–150 h.

Goals served are abbreviated: **V** value to the lab, **I** interest/novelty, **X**
interpretability skills, **E** research-engineering skills.

---

### 1. Does source audio help quality estimation when tone is lexical? (V I X E)

**Question.** IWSLT 2026 concluded that audio does not help reference-free QE, and Züfle showed
metrics sit at chance on prosody. Both conclusions come from **English source speech, technical
content, low prosody**, En→De/Zh. In Yorùbá and Igbo, tone is lexically contrastive and the
written form often drops diacritics — so the transcript is *provably* lossy in a way the German
case is not. Does audio-conditioned QE beat text-only QE when the ASR transcript cannot in
principle carry the distinction?

**Why open.** Zarzu & Zouhar name their own confound ("the technical, low-prosody-occurrence
nature of the evaluation data"); Züfle names "training data contains too few relevant
examples". Neither can test the tonal case because no such dataset exists. IWSLT 2026's Track X
covers two high-resource pairs. This is the cleanest gap in the whole survey.

**Experiment.** (a) Build a **contrastive** set in the spirit of MuST-SHE/ContraProST: from
NaijaS2ST Yorùbá/Igbo audio, generate minimal pairs where the correct English translation
depends on a tonal or diacritic distinction, verified by native speakers (the lab has the
contacts; ~300–600 pairs is enough). (b) Score each pair with text-only COMETKiwi over a
Whisper/OmniASR transcript, BLASER-2.0-QE, SSA-COMET-QE, and one speech-conditioned model
(SpeechCOMET or a small SpeechLLM). (c) Report pairwise accuracy against the 50.0 chance
baseline, plus segment-level τb on a small human-DA slice.

**Answer looks like.** A table of contrastive pairwise accuracies with bootstrap CIs. A
positive result ("audio-conditioned QE is 68% [61,74] where text-only is at chance") reverses
the campaign's conclusion for a language class. A negative result ("still at chance, n=480")
is publishable as the second independent confirmation and tells the field the encoders, not the
data, are the problem.

**Cost.** Low compute (inference only + one small fine-tune). The cost is annotation
coordination. ~40 h engineering, ~40 h data, ~30 h analysis.

**Risk.** Medium. Annotation throughput is the schedule risk; mitigate by pre-generating
candidate minimal pairs automatically and having annotators *filter* rather than author.

---

### 2. Where does the Igbo cascade lose 6.6 spBLEU? An error-attribution study on NaijaS2ST (V E X)

**Question.** IWSLT 2026 reports cascade at 17.3 for Hausa (parity with the supervised bound)
and 11.0 for Igbo (−6.6). The organisers attribute this to ASR error propagation without
measuring it. Decompose the loss into (i) ASR error, (ii) MT error on gold transcripts, (iii)
error introduced by their interaction.

**Why open.** Nobody has run the ablation; the track received one submission. CATENG asserts
"ASR-bound not MT-bound" for Catalan; Dauvet's 2-stage ASR-corrector implies the same and gives
54.2% relative WER reduction on African languages — untested here.

**Experiment.** Reproduce the OmniASR + NLLB-200 cascade on NaijaS2ST test (the lab already
serves fine-tuned NLLB). Then four conditions per language: real ASR → NLLB; **gold transcript
→ NLLB** (MT ceiling); real ASR → **mT5 corrector** → NLLB (Dauvet's recipe); gold transcript →
lab-fine-tuned NLLB. Stratify by WER decile, by named-entity presence, and by diacritic
density. Report SSA-COMET and chrF++ with CIs, family size stated for the per-language scan.

**Answer looks like.** A decomposition table: "of the 6.6 spBLEU Igbo gap, X points are
recoverable by an ASR corrector, Y points are MT-side, Z points are irreducible". Plus a direct
answer to whether the lab's fine-tuned NLLB is better than stock NLLB-200 in the cascade — which
the lab currently does not know.

**Cost.** Low–medium: inference over 3 × 500 test utterances plus one mT5-base fine-tune.
~50 h.

**Risk.** Low. Every component exists. The main risk is NaijaS2ST access/licensing; check
first.

---

### 3. Tone, diacritics and named entities: an annotated error taxonomy for Nigerian-language ST, plus the retrieval fix (V I E)

**Question.** Gaido 2021 measured 37–40% person-name accuracy for high-resource ST. What are
the equivalent numbers for Hausa/Igbo/Yorùbá → English, and does retrieval-and-demonstration
(Li & Liu, +17.6%/+8.5% on en→de) transfer?

**Why open.** The IWSLT African track's stated motivation is names, local terminology and
culturally grounded expressions, and it publishes no measurement of any of them. YoNER (LREC
2026) provides Yorùbá NER annotations to bootstrap from.

**Experiment.** Annotate NEs and diacritic-bearing tokens on the NaijaS2ST test set
(semi-automatic: NER tagger + native-speaker verification). Compute NE / person-name /
term-level accuracy with coverage reported, for the cascade, the fine-tuned Seamless baseline
and the lab's NLLB pipeline. Then implement retrieval-and-demonstration with a speech-to-speech
retriever over the NaijaS2ST training split and re-measure.

**Answer looks like.** The missing table — per-category accuracy with CIs — plus a
before/after on the retrieval intervention. Either result is a contribution; a null on
retrieval is informative because the method's authors flagged unseen-speaker robustness as its
strength and NaijaS2ST has held-out speakers.

**Cost.** Medium. Annotation is the bottleneck again; the modelling is fine-tuning-scale. ~60 h.

**Risk.** Medium–high on annotation quality. Scope down to person names only if needed.

---

### 4. When does TTS-synthesised speech help ST training? A controlled crossover (V I E)

**Question.** KIT found synthetic speech helps Bemba; LIUM found it does not help Kurdish; the
IWSLT 2025 organisers blamed an under-trained TTS. Nobody has varied TTS quality as an
independent variable. Where is the crossover?

**Why open.** Three papers, three verdicts, one named-but-unmeasured confound. The lab is
unusually well placed: it *owns* African TTS voices of differing maturity (Swahili recordings,
plus the deployed xho/ibo/efi voices), so TTS quality is a knob it can actually turn.

**Experiment.** Fix one target pair (Swahili→English, where FLEURS + CommonVoice give a real
test set). Generate synthetic ST training data from monolingual Swahili text at ≥3 TTS quality
levels (lab voice, MMS-TTS, and a deliberately degraded variant), × 3 data volumes. Train
identical E2E ST models. Also run the pseudo-labelling arm (Mohammadamini's pipeline) as the
comparison the literature says wins. Report ΔBLEU/ΔchrF++ vs a real-data-only baseline, with
seeds as the unit of independence (≥3 seeds varying data order, not just init).

**Answer looks like.** A curve — ΔST quality against a measured TTS quality index (UTMOS or
speaker count) — with the crossover point and a CI. That is a reusable result for every
low-resource ST group, and it settles a live contradiction.

**Cost.** Highest compute of the shortlist: 9–12 training runs. Still single-A100-scale if the
model is Seamless-medium or a Whisper+NLLB adapter. ~70 h wall, mostly unattended.

**Risk.** Medium. Effect sizes may be small; pre-register the satisfied-when and be prepared to
report an interval rather than a point.

---

### 5. Speaking rate and isochrony in the lab's deployed African dubbing pipeline (V E)

**Question.** Can length-aware ST (LSST + length-aware beam search, Subramanian 2025) bring
target-audio duration into line with source duration for Xhosa/Igbo/Efik without losing chrF++?

**Why open for the lab specifically.** A parallel agent's report on the ToAll repo records a
55-minute live broadcast in which MATRIX TTS ran ~40% slower than Azure Neural (9.6–11.5 vs
16.9–17.6 chars/sec), producing **1.9–2.6× audio expansion** and multi-minute drift, and Efik
NLLB output showed decoding degeneration (13.9% of segments with ≥3× repeated tokens).
**[unverified by me — second-hand from `scratchpad/toall_models_report.md`; confirm against
the incident docs before citing.]** Length-aware ST is the published fix and has never been
applied to African languages.

**Experiment.** Add length-class tokens (short/normal/long) to the lab NLLB fine-tune;
implement length-aware beam search; pick the candidate whose predicted TTS duration best
matches source duration using the lab voices' duration statistics. Measure audio-expansion
ratio, drift over a full 45-minute recording, and chrF++/COMET against the unmodified pipeline.

**Answer looks like.** "Expansion ratio 2.3× → 1.3× [CI] at a cost of −0.4 chrF++ [CI]" — a
result the lab can ship.

**Cost.** Low–medium; the duration model is cheap, the NLLB fine-tune is the main job. ~50 h.

**Risk.** Low technically; depends on getting duration statistics for the in-house voices.
Lowest research novelty of the seven — this is engineering with a measurement attached.

---

### 6. Does the accented-English side of NaijaS2ST break anything? (I E)

**Question.** NaijaS2ST deliberately records the English side by L1 speakers of Hausa/Igbo/
Yorùbá, and the test set uses **two** Nigerian English accent varieties (Northern, Southern).
Does En→X ST/ASR degrade on Nigerian-accented English relative to the same sentences read by
other speakers, and does LID misroute it (Bafna's failure mode)?

**Why open.** The corpus was built to allow exactly this analysis and the Findings paper does
not perform it. It is a clean, small, adversarial-evaluation project.

**Experiment.** Run Whisper-large-v3, OmniASR and SeamlessM4T on the accented English test
recordings; compare WER/BLEU against matched non-Nigerian-accented readings of the same
FLORES/NTREX sentences. Run an LID model over the same audio and count Nigerian-English →
non-English predictions. Report gaps with CIs, corrected for multiplicity across the accent ×
model grid.

**Answer looks like.** A bias-audit table. If the gap is large, it is a finding the ToAll app
cares about (its users speak Nigerian English).

**Cost.** Very low — inference only. ~30 h. Good candidate to run *alongside* project 1 or 2.

**Risk.** Low. Main risk is that the gap is small, in which case bound it and say so.

---

### 7. Long-form / document context for African-language ST (I E)

**Question.** AFRIDOC-MT shows sentence-trained models fail on documents for the lab's exact
languages; IWSLT 2026 shows extra context is worth +2.75 COMET in simultaneous ST. Does
document-level context help *speech* translation for Swahili/Yorùbá, and does it fix the
repetition/off-target failures AFRIDOC-MT reports?

**Why open.** No document-level *speech* translation resource or result exists for African
languages. The lab's own broadcast use case is 45-minute continuous speech, which is the
long-form regime Ugan 2026 says the models were never trained for.

**Experiment.** Concatenate FLEURS/NaijaS2ST utterances into pseudo-documents; compare
sentence-level, sliding-context, and full-document prompting for the cascade's MT stage;
measure chrF++/SSA-COMET plus explicit counts of the three AFRIDOC-MT failure modes
(under-generation, repetition, off-target).

**Answer looks like.** A context-window ablation with failure-mode counts, and a statement of
whether long-form degradation for these languages is a context problem or a model problem.

**Cost.** Low–medium, inference-heavy. ~45 h.

**Risk.** Medium — pseudo-documents are a weaker proxy than real long-form recordings; if the
lab can share a de-identified broadcast transcript, use that instead.

---

**Recommendation.** Projects 1 and 2 are the strongest pair: 2 is low-risk, fully specified and
directly useful to the lab; 1 is the one with a genuine claim to novelty at a real venue
(IWSLT 2027's metrics track would take it), and 6 is a cheap add-on to either. If the semester
must produce a single defensible number, do 2 first and treat 1 as the extension.

---

## §4 Verification debt

**Read in full or near-full (sections cited):** IWSLT 2026 Findings (§1–2, Tracks II, V, VI,
VII, VIII, X); IWSLT 2025 Findings (§1–2, low-resource track §7); Züfle 2026 (§2, §4.1,
conclusion, App. B.3); IWSLT 2026 metrics track results and discussion.

**Read at abstract + targeted-grep level only (30 papers).** Every number I attribute to these
came from a sentence I actually extracted and read in context, but I did not read the papers
end to end, so **method caveats, ablation scope and negative sub-results may be missing**. In
particular:
- Papi 2025 (TACL): I read §1, §3.2 and the survey statistics. I did not read their proposed
  taxonomy or §5 recommendations.
- Zhang 2024 (StreamSpeech), Dale 2024 (BLASER 2.0), Han 2024 (SpeechQE), Gaido 2021 (NEs),
  Li & Liu 2024 (rare words), Adelani 2025 (AFRIDOC-MT), Li 2025 (SSA-COMET), Savoldi 2025
  (mGeNTE), Papi 2026 (MCIF): **abstract + intro only**.
- Subramanian 2025, Blaschke 2025, Bafna 2025, Ahtasam 2026, Zevallos 2026, Ortega 2026,
  Krahn 2026, Ugan 2026, Bär 2025, Koneru 2025, Post & Hoang 2025: **abstract + one results
  grep**.

**Named in the brief but NOT covered here, and not cited anywhere above:**
- **Hibiki** (simultaneous S2ST). I could not fetch it or verify a peer-reviewed venue within
  budget. Do not cite it from this document.
- **Seamless / SeamlessExpressive streaming.** `Seamless-2023-Expressive-Streaming-Speech-Translation.pdf`
  is already in `papers/` (added by a sibling agent). I did **not** open it. It is a 2023 Meta
  technical report, i.e. a **preprint**, not a peer-reviewed venue.
- **xCOMET for speech** as a distinct line — it appears only as a *pseudo-label source*
  (HydraQE) and as a quality metric (Xue 2026); I found no dedicated "xCOMET for speech" paper.
- **Dialectal Arabic** is covered only through IWSLT 2025's Tunisian/North Levantine track
  summary; I did not open ALADAN or LIA's system papers, and Arabic dialect ST deserves its own
  pass if it becomes a candidate direction.
- **IWSLT 2024 findings** — not read. Several year-on-year claims (the "performance ceiling")
  rest on the 2025 organisers' summary of 2024 rather than on 2024 itself.

**Claims that need checking before they are reused:**
1. **The synthetic-TTS contradiction (P8).** I am asserting that the IWSLT 2025 Findings'
   "ineffective" verdict conflicts with KIT's own abstract. I read both. But I did not read
   KIT's results tables, so it is possible both statements are true of different experiments
   within the same paper. **Read `Li-2025-KIT-LowResource-Synthetic-Data.pdf` §5 before
   building project 4 on this.**
2. **The IWSLT 2026 low-resource per-language BLEU figures.** `pdftotext` scrambled the table
   layout around Tables 3–7; I quote only Mapudungun (0.34–0.82, Table 6 explicitly labelled),
   Central Kurdish (21.09 / 0.16, Table 7) and Fleurs-Badini (5.24). The Bhojpuri and Irish
   numbers that appeared in my extract are **not** attributed here because I could not confirm
   which table they belong to. Re-read pages 347–348 of the PDF if you need them.
3. **NaijaS2ST availability and licence.** The Findings say the data "is available here" with a
   footnote link I did not follow. Projects 1, 2, 3 and 6 all depend on it. **Check first.**
4. **The ToAll production numbers** in project 5 (40% slower TTS, 1.9–2.6× expansion, 13.9%
   repeated-token segments) are second-hand from a parallel agent's report in this session's
   scratchpad; I did not open the ToAll incident documents myself.
5. **Venue labels.** Every venue above was taken from the ACL Anthology volume `.bib`, the ISCA
   Archive Interspeech 2025 index, or the PDF header. Two exceptions to double-check:
   MCIF's "ICLR 2026 Poster" comes from an OpenReview API record (the PDF itself is the arXiv
   v2 and says "Preprint"), and Züfle 2026 is arXiv-only.
6. **Search coverage.** The web-search budget for this session was exhausted before I began, so
   this survey was assembled from the ACL Anthology (IWSLT 2025/2026 volumes and ~8 author
   pages), the ISCA Interspeech 2025 index, and citation-chasing inside the two Findings
   papers. **ICASSP, SLT/ASRU, NeurIPS/ICML and NAACL 2026 were not swept at all**, and
   arXiv's API was rate-limited throughout. Anything primarily published at those venues is
   systematically absent.
