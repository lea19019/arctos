# ASR and TTS for low-resource and African languages — landscape, problems, MS-sized projects

Compiled 2026-09-06 for the superweights track's speech arm. Scope set by the shared brief
(`scratchpad/speech_brief.md`) with two overrides from the parent: **ignore the massive-activation /
attention-sink phenomenon entirely** (a sibling agent owns it), and **prefer accredited venues**
(ACL/EMNLP/NAACL/TACL, Interspeech, ICASSP, NeurIPS/ICML/ICLR, IWSLT, SLT/ASRU, 2024–2026), labelling
arXiv-only work as preprint.

**Reading status convention.** `[read]` = I opened the PDF and the statement comes from text I read.
`[skim]` = I read the abstract/intro and the specific numbers quoted, not the whole paper.
`[listing]` = title/venue/date from an arXiv search listing only; the PDF was **not** opened.
Every paper cited below is downloaded into this folder unless the row says otherwise.

**Venue convention.** Where a venue is stated I say where it is stated: `printed on PDF` (the venue
appears on the paper itself), `arXiv comment` (the submitter's comment field said so — not verified
against proceedings), or `preprint` (no venue claim found). This matters: arXiv comments say
"Accepted to X" for things that have not appeared, and the brief forbids repeating unverified venues
as fact.

**Duplicate note.** A parallel agent downloaded an overlapping set into this same flat folder while I
worked. I removed my own byte-identical duplicates and cite the surviving filename, which for twelve
papers is the parallel agent's name (e.g. `Alabi-2025-AfriHuBERT.pdf`, `Chen-2024-F5-TTS.pdf`,
`Meyer-2022-BibleTTS.pdf`). Those files are indexed in that agent's README subsection, not mine.

---

## §1 Paper by paper

### 1.1 Massively multilingual ASR backbones

| Paper | File | Venue status | What is actually shown |
|---|---|---|---|
| Omnilingual ASR team, *Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages* (2025) | `Omnilingual-ASR-2025-1600-Languages.pdf` | preprint (arXiv 2511.09690, dated 10 Nov 2025; no venue on PDF) `[read]` | See §2. 1,600+ languages, 500+ never before served; wav2vec 2.0 SSL scaled to 7B; two heads (CTC and "LLM-ASR" encoder–decoder); an in-context zero-shot extension. |
| Pratap et al., *Scaling Speech Technology to 1,000+ Languages* (MMS) | `Pratap-2024-MMS-Scaling-Speech-1000-Languages.pdf` | preprint on this PDF (arXiv 2305.13516); widely cited as JMLR 2024 — **not verified here** `[skim]` | wav2vec 2.0 pre-trained on 1,406 languages; one ASR model for 1,107 languages; LID for 4,000+. Trained largely on religious readings. The reference point every African-ASR paper below compares against. |
| Conneau et al., *FLEURS* | `Conneau-2023-FLEURS-Benchmark.pdf` | preprint on this PDF (arXiv 2205.12446); commonly cited as SLT 2022 — **not verified here** `[skim]` | 102-language n-way parallel read-speech eval set. Its African coverage is exactly Amharic, Fula, Ganda, Hausa, Igbo, Kamba, Lingala, Nyanja, Oromo, Shona, Somali, Swahili, Wolof, Xhosa, Yoruba, Zulu. **No Efik, no Twi/Akan** — verified by grepping the PDF. |
| Radford et al., *Whisper* | `Radford-2022-Whisper-Robust-Speech-Recognition.pdf` | pre-existing in folder `[not opened by me]` | Baseline for everything below. |
| Alabi et al., *AfriHuBERT* | `Alabi-2025-AfriHuBERT.pdf` | Interspeech 2025 per arXiv comment `[skim]` | Continued pretraining of mHuBERT-147 on 10K+ h → 1,226 African languages. **+3.6 F1** on SLID and **−2.1 average WER** on ASR vs mHuBERT-147 on FLEURS; competitive with much larger MMS/XEUS. The "small model, African data" counterpoint to scale. |
| Azunre et al., *DONDO* (2026) | `DONDO-2026-w2v-BERT-African-ASR-Models.pdf` | preprint `[skim]` | 21 monolingual + 5 multilingual w2v-BERT 2.0 models over 27 varieties in Ghana, Sierra Leone, Nigeria, Senegal, Kenya, Zimbabwe. Annealed two-step fine-tuning; **10–13% average WER** for the multilingual families; language conditioning by one-hot prefix frames. Apache-2.0. |
| *Sometin Beta Pass Notin (SBPN)* (2026) | `SBPN-2026-Nigerian-Languages-Knowledge-Distillation-ASR.pdf` | preprint `[skim]` | Yorùbá/Hausa/Igbo/Nigerian Pidgin/Nigerian English foundation ASR, 120M and 600M. Teacher–student distillation from monolingual models conditioned on n-gram LMs, then iterative pseudo-label self-improvement: **−29% relative WER** vs monolingual baselines. |

### 1.2 African-language ASR benchmarks and evaluations

| Paper | File | Venue status | What is actually shown |
|---|---|---|---|
| Nahabwe et al., *Benchmarking ASR Models for African Languages* | `Makerere-2025-Benchmarking-ASR-African-Languages.pdf` (dup: `Makerere-2025-Benchmarking-ASR-African-Languages.pdf`) | **Deep Learning Indaba 2025, PMLR 302 — printed on PDF** `[skim]` | Whisper / XLS-R / MMS / w2v-BERT across 13 African languages, fine-tuned on 1→400 h. MMS and w2v-BERT are more data-efficient in the very-low regime; XLS-R scales better with data; Whisper wins in mid-resource. Also: where external LM decoding helps and where it plateaus. |
| Akera et al. (Sunbird AI), *How much speech data is necessary for ASR in African languages?* | `HowMuchData-2025-ASR-African-Languages-Hours.pdf` | preprint `[skim]` | Kinyarwanda scaling 1→1,400 h with Whisper: **WER < 13% at ~50 h**, **WER < 10% by ~200 h**. Kikuyu error analysis on 270 h: **38.6% of high-error cases are noisy ground-truth transcripts**, not model failure. |
| Diack et al., *WAXAL* | `WAXAL-2026-African-Language-Speech-Corpus.pdf` (dup: `WAXAL-2026-African-Language-Speech-Corpus.pdf`) | preprint (Google Research + 4 African partners) `[skim]` | 24 languages, ~1,250 h transcribed ASR + **235 h single-speaker TTS**, CC-BY-4.0, `google/WaxalNLP`. ASR+TTS for Acholi, Kiswahili, Luganda, Akan, Ewe; TTS for Hausa, Igbo, Yoruba. |
| Olufemi et al., *WAXAL-NET* | `WAXAL-NET-2026-Edge-ASR-19-African-Languages.pdf` | preprint `[skim]` | Fine-tuned compact models beat the best zero-shot foundation baseline on WAXAL by **38.0% vs 64.9% macro WER** — 26.9 pts, at 3–40× smaller. Native-speaker error audit; and **WER misrepresents syllabary-script languages, where CER/WER ratios show much higher character accuracy than headline WER**. |
| Ashungafac et al., *AfriSwitch* | `AfriSwitch-2026-African-CodeSwitched-ASR-Benchmark.pdf` | preprint (Intron Health) `[skim]` | 61.36 h human-transcribed in-the-wild code-switched speech, 16 African languages/varieties, with switch-level English span tags and per-utterance Code-Mixing Index. Five systems zero-shot: **best average 35.93% WER; no system below 24% on any language**. Africa-targeted training, not scale or nominal coverage, predicts performance. |
| Ashungafac et al., *AfriSpeech-MultiBench* | `AfriSpeech-MultiBench-2025-African-Accented-English-ASR.pdf` | IJCNLP-AACL 2025 per arXiv comment `[skim]` | 100+ African English accents, 10+ countries, 7 domains (incl. a **hallucination-robustness** vertical). Open ASR is best on spontaneous speech but degrades on noisy non-native dialogue; multimodal LLMs are accent-robust but weak on domain named entities; "hallucinations still remain a big problem for most SOTA models". |
| *AfriVox-v2* (2026) | `AfriVox-v2-2026-Verticalized-African-ASR-Benchmark.pdf` | preprint `[skim]` | In-the-wild unscripted audio, 10 sectors, targeted number/named-entity tests; benchmarks Sahara-v2, Gemini 3 Flash, **and the Omnilingual CTC models**. |
| Olatunji et al., *AfriSpeech-200* | `Olatunji-2023-AfriSpeech-200.pdf` | TACL 2023 per arXiv comment `[skim]` | 200 h Pan-African **accented English**, 67,577 clips, 2,463 speakers, 120 accents, 13 countries, clinical + general. The standard accent-robustness testbed. |
| *AfriNames* | `AfriNames-2023-Interspeech-ASR-Butcher-African-Names.pdf` | PDF header reads "Submitted to INTERSPEECH"; arXiv comment claims Interspeech 2023 main `[skim]` | ASR systematically mis-transcribes African **named entities**. |
| Elmadany et al., *Voice of a Continent* | `Voice-of-a-Continent-2025-African-Speech-Survey.pdf` | preprint `[skim]` | Maps African speech datasets → **SimbaBench**; releases the Simba model family. Analyses how dataset quality, domain diversity and language-family relationships drive per-language performance. Downloaded by the parallel agent. |
| *ASR for African Low-Resource Languages: A Systematic Literature Review* | `SLR-2025-ASR-African-LowResource-Systematic-Review.pdf` | preprint `[skim]` | PRISMA review, Jan 2020–Jul 2025. **74 datasets, 111 languages, ~11,206 h** in scope; **63.5% of datasets rated high risk of bias**; 18 datasets have <10 h and only 17 exceed 100 h. Explicit finding: **WER dominates, and CER/DER "remain underutilized" partly because most corpora lack phoneme- or diacritic-level annotation.** |
| *ASR for African Low-Resource Languages: Challenges and Future Directions* | `Survey-2025-ASR-African-LowResource-Challenges-Future.pdf` | pre-existing; preprint `[not opened by me]` | Companion survey. |

### 1.3 Transfer, data efficiency, pseudo-labelling

| Paper | File | Venue status | What is actually shown |
|---|---|---|---|
| *Evaluating the Effect of Linguistic Relatedness on Cross-Lingual Transfer in Large Multilingual ASR* (2026) | `Relatedness-2026-CrossLingual-Transfer-Multilingual-ASR.pdf` | preprint `[skim]` | **A negative result worth its weight.** Six factors × two Africa-centric corpora × four large ASR models, sequential adaptation on a related auxiliary language then the target. **In every setting, pre-adaptation on a related language yields no practically meaningful gain once as little as one hour of target data exists.** |
| *DonorRank* (2026) | `DonorRank-2026-Donor-Language-Selection-CrossLingual-ASR.pdf` | preprint `[skim]` | Learning-to-rank for donor-language selection in zero-shot ASR, on VAANI-D (Indic, shared script) and WAXAL (African, heterogeneous). Argues genealogy and lexical similarity alone are unreliable when orthographic standardisation is weak. Sits in direct tension with the paper above — worth pairing. |
| Lin et al., *Pseudo2Real* | `Pseudo2Real-2026-ACL-Findings-Task-Arithmetic-PseudoLabel.pdf` | ACL 2026 Findings per arXiv comment `[skim]` | Fine-tune two models from the same init — one on gold, one on pseudo-labels — and their weight difference is a **correction vector** for pseudo-label bias. **Up to 35% relative WER reduction on AfriSpeech-200 across ten African accents (Whisper TINY).** Cheap, parameter-space, no target labels. |
| *Synthetic Voice Data for ASR in African Languages* | `SyntheticVoice-2025-RANLP-African-ASR-Data.pdf` | RANLP 2025 workshop per arXiv comment `[skim]` | LLM text → TTS voice → ASR training. >2,500 h generated at **<1% of the cost of real collection** (real data quoted at **US$100–150/h**). First systematic assessment of synthetic corpora for African ASR. |
| *A Practitioner's Guide to Building ASR Models for Low-Resource Languages* (Scottish Gaelic) | `Gaelic-2025-Interspeech-Practitioners-Guide-LowResource-ASR.pdf` | Interspeech 2025 per arXiv comment `[skim]` | Recipe-level guidance; useful as a methods template, non-African. |
| *Adapting Whisper for Parameter-efficient Code-Switching ASR via Soft Prompt Tuning* | `SoftPrompt-2025-Interspeech-Whisper-CodeSwitching.pdf` | Interspeech 2025 per arXiv comment `[skim]` | Soft prompts as a cheap code-switching adapter for Whisper. |

### 1.4 Tone, diacritics, orthography, and what WER hides

| Paper | File | Venue status | What is actually shown |
|---|---|---|---|
| Mokgosi, Marivate et al., *Tone-Conditioned Curriculum Learning for Low-Resource Bantu ASR* | `ToneCurriculum-2026-Bantu-ASR.pdf` | preprint `[skim]` | Six Southern Bantu languages. Baseline claim: **foundation models yield zero-shot WER above 100%**. Tone-gated adapters + staged curriculum: **28.41% average WER**, 23.79% on Xitsonga transfer. Architecture interacts with language family — w2v-BERT beats Whisper on Nguni by 3–4 WER points, Whisper wins on Sotho-Tswana. Key linguistic point: **Bantu tone spreads across phrases and standard orthographies omit it**, so the model must recover tonal dependencies that the transcript never encodes. |
| Gogoi et al., *Tone recognition in low-resource languages of North-East India* | `Tone-2025-Interspeech-Recognition-NorthEast-India.pdf` | Interspeech 2025 per arXiv comment `[read: abstract + intro]` | Layer-wise probing of four wav2vec 2.0 models for tone in Angami, Ao, Mizo. **Middle layers (roughly 4–6) carry tone**, regardless of whether pretraining languages were tonal. Tone inventory size and dialect variation move accuracy. This is the closest thing in the literature to an *interpretability* result about tone, and it is on Tibeto-Burman, not Niger-Congo. |
| *SITA: Speaker-Invariant and Tone-Aware Speech Representations* (2026) | `SITA-2026-Tone-Aware-LowResource-Speech-Representations.pdf` | preprint `[listing + title verified]` | Lightweight adaptation recipe for wav2vec-style encoders on Hmong and Mandarin. |
| Rahman, *Script collapse in multilingual ASR* (2026) | `ScriptCollapse-2026-Reference-Free-Metric-Multilingual-ASR.pdf` | preprint `[skim]` | Defines **Script Fidelity Rate (SFR)** — fraction of hypothesis characters in the target script block, **computable with no reference transcript**. Across 100 model–language pairs (7 Whisper sizes, MMS-1B, SeamlessM4T-v2, Gemma 4 E2B) on FLEURS, **21% (95% Wilson CI 14–30%) show script collapse (SFR < 10%)**. Script-aware prompting on Gemma 4 raises mean SFR 71.2 → 97.7 and recovers **5.9 chrF on downstream NLLB translation**. Explicit claim: no peer-reviewed paper previously defined a script-fidelity metric. |
| Ezeani, *Corpus-Based Approaches to Igbo Diacritic Restoration* | `Igbo-2026-Corpus-Based-Diacritic-Restoration.pdf` | PhD thesis, University of Sheffield (270 pp), posted to arXiv 2026 `[skim]` | The reference work on Igbo diacritic restoration: n-gram, classification and embedding approaches. |
| *Diacritic Restoration for Low-Resource Indigenous Languages* (Bribri, Cook Islands Māori) | `Diacritics-2025-LowResource-Indigenous-Bribri-Maori.pdf` | preprint `[skim]` | Character-level fine-tuned LLMs win (they decompose complex characters into UTF-8 bytes); massively multilingual models underperform at this data scale; **reliable performance emerges around 10,000 words**; zero-shot is poor throughout. Includes **tonal** diacritics. |
| *Voices Unheard: NLP Resources and Models for Yorùbá Regional Dialects* | `VoicesUnheard-2024-Yoruba-Regional-Dialects.pdf` | preprint `[listing]` | YORÙLECT corpus, three domains, four regional Yorùbá dialects; text and speech. |
| *ÌròyìnSpeech* | `IroyinSpeech-2024-LREC-COLING-Yoruba-Corpus.pdf` | LREC-COLING 2024 per arXiv comment `[listing]` | Multi-purpose Yorùbá speech corpus. |

### 1.5 Speaker populations: children, accents, dialects

| Paper | File | Venue status | What is actually shown |
|---|---|---|---|
| *An End-to-End Approach for Child Reading Assessment in the Xhosa Language* | `Xhosa-2025-Child-Reading-Assessment-ASR.pdf` | preprint `[skim]` | New Xhosa child-speech dataset built around EGRA (Early Grade Reading Assessment) items; wav2vec 2.0 / HuBERT / Whisper fine-tuned. Performance is dominated by **how much data and how balanced the classes are**; wav2vec 2.0 improves when trained on multiple classes at once even under sample constraints. |
| *An bɛ kalan: Building an ASR Solution for Children's Reading in Bambara* (2026) | `Bambara-2026-Children-Reading-ASR.pdf` | preprint `[skim]` | 55 h raw reading speech from 60 children; public benchmark. Best model: **WER 0.42 → 0.22, CER 0.15 → 0.08**. **Children under 10 are the main residual error source.** Ten classroom trials. |
| *ASR of African American English: Lexical and Contextual Effects* | (not downloaded — non-African, and the sibling agent covers ASR bias) | Interspeech 2025, pp. 3883–3887 per arXiv comment `[listing]` | Included for completeness of the accent/dialect line. |

### 1.6 TTS

| Paper | File | Venue status | What is actually shown |
|---|---|---|---|
| Meyer et al., *BibleTTS* | `Meyer-2022-BibleTTS.pdf` | Interspeech 2022 per arXiv comment `[skim]` | Up to **86 h aligned studio-quality 48 kHz single-speaker** recordings **per language** for **Akuapem Twi, Asante Twi, Chichewa, Ewe, Hausa, Kikuyu, Lingala, Luganda, Luo, Yoruba**. CC-BY-SA. Still the highest-fidelity open African TTS data. |
| Guzmán, Jeon, Beyene, Alabi, Klakow, Adelani, *OpenBibleTTS* (2026) | `OpenBibleTTS-2026-African-Language-TTS.pdf` | preprint `[skim]` | 37 underrepresented languages (**19 African**), systematic architecture comparison (incl. EveryVoice, VITS, F5-TTS-class systems) on in-domain Biblical and **out-of-domain** text, with human listening. Headline: **no single system dominates**; Gemini-TTS wins listener ratings on most languages, but **monolingual EveryVoice models trained on OpenBibleTTS are strongest for intelligibility and preferred in several African languages**, and **open from-scratch systems degrade sharply out of domain**. |
| Edet et al., *Towards Digital Preservation of Efik: TTS for a Low-Resource African Language* (2026) | `Efik-TTS-2026-Digital-Preservation.pdf` | Interspeech 2026 per arXiv comment `[read: abstract + §1–2]` | **The single most directly relevant paper to the lab.** First documented end-to-end Efik TTS. Corpus: **2,632 utterances, ~3 hours, single speaker**, wireless mic, controlled conditions. Four models compared (VITS, MMS-TTS, SpeechT5, Orpheus-TTS), native-speaker MOS / Nat-MOS / A-MOS. **MMS-TTS best at MOS 3.80 ± 0.63** and more stable on long-form, **but tonal errors persisted in all four**. Efik: Lower Cross, tonal, ~1.5M L1 and ~3M L2 speakers. |
| Kagumire, Katumba, Nakatumba-Nabende, Quinn, *Building a Luganda TTS Model From Crowdsourced Data* | `Luganda-2024-AfricaNLP-Crowdsourced-TTS.pdf` | **AfricaNLP workshop at ICLR 2024 — printed on PDF** `[skim]` | Common Voice Luganda → usable TTS by (i) selecting **six female speakers of close intonation** rather than one, (ii) trimming silence + a pretrained speech-enhancement pass, (iii) **filtering by a non-intrusive self-supervised MOS estimator at >3.5**. Result: **MOS 3.55** vs 2.5 for the prior model, and vs 3.13 (one speaker) / 3.22 (two speakers). This is a complete, reproducible recipe for turning noisy crowdsourced audio into a TTS voice. |
| Ogun, Owodunni, Olatunji et al., *AfroTTS / 1000 African Voices* | `Ogun-2024-Afro-TTS-1000-African-Voices.pdf` | preprint `[skim]` | First pan-African **accented-English** TTS: 86 accents, 747 speakers, 9 countries, 1,000 personas; includes a speaker-averaging method to synthesise new personas. Masakhane NLP. Downloaded by the parallel agent. |
| Túbọ̀sún et al., *A Situational Speech Synthesizer for Yorùbá* (2026) | `Yoruba-2026-Situational-Speech-Synthesizer-Tone.pdf` | "Under review at Speech Communication" per arXiv comment; PDF is a manuscript `[skim]` | Rule-based concatenative **diphone** synthesiser deployed on YorubaName.com. 651 diphone units spanning **five tonal variants of every CV combination**; hand-written tonal file-selection rules; three-way nasal disambiguation; contextual rising/falling tones derived from level-tone input. Orthographic contribution: caron and circumflex as single-vowel contour-tone markers. Listener study N=50. A useful reminder that a tone-explicit rule system is a live baseline, not a museum piece. |
| Casanova et al., *XTTS* | `XTTS-2024-Interspeech-Massively-Multilingual-ZeroShot-TTS.pdf` | Interspeech 2024 per arXiv comment `[skim]` | 16-language zero-shot multilingual TTS. **No African language among the 16** (checked the paper's language list framing, not an exhaustive per-code audit — see §6). |
| Chen et al., *F5-TTS* | `Chen-2024-F5-TTS.pdf` | preprint `[skim]` | Flow-matching non-autoregressive TTS; simpler than E2-TTS/VoiceBox, strong zero-shot cloning. The current open default. |
| Liu et al., *Cross-Lingual F5-TTS* | `CrossLingual-F5-TTS-2025-Unseen-Language-Cloning.pdf` | preprint `[skim]` | Removes the dependency on a **transcript of the audio prompt** — the thing that blocks cross-lingual cloning for unseen languages. Forced alignment for word boundaries at training; speaking-rate predictors for duration at inference. Matches F5-TTS while enabling transcript-free cross-lingual cloning. |
| Du et al., *CosyVoice* | `CosyVoice-2024-Supervised-Semantic-Tokens-TTS.pdf` | preprint ("work in progress") `[skim]` | Supervised semantic tokens for scalable multilingual zero-shot TTS. |
| Do, Coler, Dijkstra, Klabbers, *Strategies in Transfer Learning for Low-Resource Speech Synthesis* | `TransferTTS-2023-SSW-LowResource-Speech-Synthesis.pdf` | PDF header reads "Submitted to SSW" `[skim]` | PHOIBLE-based phone mapping vs **phonological-features input** for cross-lingual TTS transfer; five source and six target languages, **Swahili among the targets**. Features input wins, but the effect is language-pair dependent. Also tests Angular Similarity of Phone Frequencies (ASPF) as a source-language selection criterion. |
| *Edge-Based Speech Transcription and Synthesis for Kinyarwanda and Swahili* | `Edge-2025-Kinyarwanda-Swahili-STT-TTS.pdf` | preprint (CMU-Africa) `[skim]` | Whisper + SpeechT5 split across edge and cloud; 9.5%/14% memory compression, max 149 MB, 270 characters in under a minute on a 1.7 GHz CPU with 1 MB/s uplink. Deployment-shaped, not modelling. |

### 1.7 Evaluating synthetic speech

| Paper | File | Venue status | What is actually shown |
|---|---|---|---|
| Huang, Cooper, Toda, *MOS-Bench* | `MOS-Bench-2024-TASLP-Generalization-Speech-Quality-Assessment.pdf` | IEEE TASLP per arXiv comment; PDF is in IEEE journal template `[skim]` | 8 training sets + 17 test sets for subjective speech-quality assessment. Central finding: **existing SSQA models generalise poorly out of domain**, and **pooling multiple training sets beats domain-aware AlignNet — variation in the data matters more than data size.** This is the paper that licenses scepticism about using UTMOS-class predictors on an African language they never saw. |
| Minixhofer, Klejch, Bell, *TTSDS2* | `Minixhofer-2026-TTSDS2-Benchmark.pdf` | preprint `[skim]` | Of **16 compared metrics, TTSDS2 is the only one correlating above Spearman 0.50 with subjective scores in every domain and score type tested**. Releases 11,000+ subjective ratings, a leakage-avoiding multilingual test-set pipeline, and a **14-language** benchmark. Downloaded by the parallel agent. |
| *Cross-lingual MOS-predictor generalisation* (Language Barriers, arXiv 2502.13004); *Investigating Human–Model Discrepancies in SQA via Acoustic and Prosodic Perturbations* (Interspeech 2026 per arXiv comment) | not downloaded | `[listing]` | Named because the second reports that SQA models are **insensitive to prosodic errors despite large subjective drops** — directly relevant to tone. I did not open it; it is verification debt (§6). |

### 1.8 Efik and the lab's other languages, beyond speech

| Paper | File | Venue status | What is shown |
|---|---|---|---|
| *Developing an English-Efik Corpus and Machine Translation System* (2026) | `Efik-2026-AfricaNLP-English-Efik-Corpus-MT.pdf` | AfricaNLP 2026 @ EACL per arXiv comment `[listing]` | Text-side Efik resource. |
| *Ibom NLP: A Step Toward Inclusive NLP for Nigeria's Minority Languages* (2025) | `IbomNLP-2025-IJCNLP-AACL-Nigeria-Minority-Languages.pdf` | IJCNLP-AACL per arXiv comment `[listing]` | Efik/Ibibio/Annang-area resources. |
| Wanjawa et al., *Kencorpus* | `Kencorpus-2022-Kenyan-Language-Corpus.pdf` | *Journal for Language Technology and Computational Linguistics* 36(2), 2023 per arXiv comment `[skim]` | Swahili, Dholuo, Luhya. |
| *AfriVoices-KE* (2026) | `AfriVoices-KE-2026-LREC-Kenyan-Speech-Dataset.pdf` | RTUEL @ LREC 2026 per arXiv comment `[skim]` | ~3,000 h across Dholuo, Kikuyu, Kalenjin, Maasai, Somali; 750 h scripted + 2,250 h spontaneous; 4,777 speakers; smartphone app collection with SNR gating + human review. Documents the failure modes of field collection: unreliable infrastructure, device incompatibility, community trust. |
| Marivate et al., *Swivuriso* | `Swivuriso-2025-African-Next-Voices-South-Africa.pdf` | preprint `[skim]` | 3,000 h, seven South African languages **including isiXhosa**, agriculture / healthcare / general, scripted **and** unscripted. Part of African Next Voices. |
| *Africa-Centric Self-Supervised Pre-Training for Multilingual Speech Representation* | `Africa-Centric-2024-SSL-Pretraining-SubSaharan.pdf` | **AfricaNLP 2024 @ ICLR — printed on PDF** `[skim]` | Sub-Saharan SSL pretraining (the SSA-HuBERT line). |

---

## §2 Omnilingual ASR in detail

All statements here are `[read]` from `Omnilingual-ASR-2025-1600-Languages.pdf` with the section
number given. The PDF carries **no venue** — treat it as a preprint / technical report from FAIR at
Meta dated 10 Nov 2025 (arXiv 2511.09690), correspondence Yu-An Chung and Jean Maillard.

### 2.1 What it claims

- **1,600+ languages** supported, **500+ never before served by any ASR system** (abstract, §6).
- Self-supervised pre-training of wav2vec 2.0 scaled to **7B parameters** on **4.3M hours** of
  unlabeled speech (§5.2.2 explicitly contrasts this with USM's 12M hours of proprietary YouTube audio).
- Two supervised heads on the same encoder (§4.1–4.2): a **CTC** head (linear layer on the SSL
  encoder) and **LLM-ASR**, a Transformer decoder over the speech encoder trained with next-token
  prediction. Encoder sizes 300M / 1B / 3B / 7B; LLM-ASR variants share one decoder size.
- **Language-code conditioning** (§4.5): an optional `<language>` token plus a language+script ID
  embedding, dropped with probability *p* during training so the model works with or without it.
- Model family from **300M for low-power devices to 7B**; all artifacts open-sourced at
  `github.com/facebookresearch/omnilingual-asr`.
- Data (§3.3): existing ASR corpora, **partner-created** data, and a **commissioned** Omnilingual ASR
  Corpus — target 10 h per language from 10 native speakers (1 h each) across ~350–400 languages,
  elicited by a pool of 1,500+ survey-style prompts, with compensated local partnerships. Explicitly
  connected to **African Next Voices** (Maseno / Pretoria / Data Science Nigeria, Gates-funded),
  Mozilla Common Voice's Open Multilingual Speech Fund (**170+ new communities, Common Voice now
  "well over 300 languages"**), and **Lanfrica/NaijaVoices**, which produced data for 11 African
  languages named in §3.3.2 (Bainouk-Gunyaamolo, Balanta-Kentohe, Bube, Fang, Igala, Central Kanuri,
  Karon, Nupe-Nupe-Tako, Upper Guinea Crioulo, Serer, Urhobo).

### 2.2 The zero-shot / in-context language-extension mechanism (§4.3, §4.4, §5.4, §5.5)

This is the paper's actual novelty and it is mechanically simple:

> At training time, instead of one speech–text pair, present **N+1 pairs from the same language**. The
> first N are prepended to the decoder prompt as context; the last is the target, predicted by ordinary
> next-token prediction. (§4.3)

The prompt syntax, verbatim from §4.3:

```
<c> {<cs> g_s(x_i^c) <cs BOS> g_t(y_i^c) <cs EOS> </cs>} × N </c> g_s(x) <BOS> g_t(y) <EOS>
```

where `g_s` is the wav2vec 2.0 speech encoder and `g_t` the text embedding. Because training spans a
large number of languages, the *behaviour* of conditioning on a few in-language examples is
hypothesised to generalise to languages absent from training — which is what "communities can add
their language with a handful of samples" means concretely.

**Measured (§5.4, Table 11).** Hold out 32 languages (half high-resource, half low-resource), average
CER across all eval sets:

| Model | Context | Unseen-language CER |
|---|---|---|
| CTC baseline | 0 | 26.3 |
| LLM-ASR baseline | 0 | 31.0 |
| ZS LLM-ASR, CTC seed | 5 | 19.3 |
| ZS LLM-ASR, CTC seed, frozen encoder | 5 | 26.5 |
| ZS LLM-ASR, w2v2 seed | 5 | 17.6 |
| **ZS LLM-ASR, w2v2 seed** | **10** | **14.4** |

Two ablation findings stated in §5.4: **seeding from CTC hurts** zero-shot generalisation, and
**tuning (not freezing) the speech encoder is crucial**. Zero-shot models slightly degrade on seen
languages, except on FLEURS-102 and CV22 where they *improve* — because context examples "significantly
reduce script and language confusion errors".

**Context selection (§5.5).** Retrieval with **SONAR** speech/text embeddings beats random selection by
**up to 11.2% relative CER**; speech→speech and speech→text retrieval are about equally good;
**wav2vec 2.0 mean-pooled embeddings give no obvious improvement over random**; a max-bigram-diversity
selection gives only a slight gain. `text_sim` (BM25 on the target transcript) and `same_ex` (feeding
the answer) are labelled oracles.

### 2.3 Reported accuracy, and where African languages land

**vs Whisper (§5.2.1, Table 5).** Even the 300M-CTC beats Whisper large-v3 in average CER on
MMS-Lab-63, FLEURS-82 and CV22-76, losing only on MLS-8. 7B-LLM vs Whisper large-v3 on FLEURS-81:
**80% win rate (65/81)** overall, **71% (24/34)** on the world's most-spoken languages. §5.2.1 also
diagnoses why CTC trails LLM-ASR: **CTC fails by script misprediction**, most often in low-resource
settings; language-code conditioning "largely resolves the wrong-script problem".

**By resource bucket (§5.3.1, Tables 8–9).** High = >50 h, mid = 10–50 h, low = <10 h.

| | High (249 langs) | Mid (881) | Low (546) |
|---|---|---|---|
| 7B-CTC mean CER (95% CI) | 3.7 ± 0.7 | 4.4 ± 0.6 | **18.6 ± 1.2** |
| 7B-LLM mean CER (95% CI) | 3.13 ± 0.7 | 3.0 ± 0.3 | **18.0 ± 1.2** |
| 7B-LLM with CER < 10 | 236 | 841 | **195 (36%)** |

**By language grouping (§5.3.2, Table 10, 7B-LLM, no LM fusion).** The two African-dominant groupings
are the worst two in the table:

| Grouping | # langs | Avg CER | CER ≤ 10 | % |
|---|---|---|---|---|
| **Afroasia** (Afro-Asiatic) | 92 | **11.8** | 61 | **66%** |
| **Atlacong** (Atlantic-Congo) | 389 | **9.3** | 280 | **72%** |
| Mesoamer | 159 | 7.8 | 115 | 72% |
| Indoeuro | 209 | 9.1 | 154 | 74% |
| Sinotibe | 65 | 8.2 | 52 | 80% |
| Nilosaha | 56 | 4.4 | 50 | 89% |
| Amazbasi / Amerande | 83 / 67 | 2.0 / 2.0 | 82 / 66 | 99% / 99% |
| **Total** | **1570** | **7.1** | **1237** | **78%** |

Afro-Asiatic is **the only grouping whose average CER exceeds 10**, and Atlantic-Congo — the family
containing Swahili, Yoruba, Igbo, Xhosa, Twi and Efik — is the largest grouping and third-worst by
success rate. The paper states this plainly ("is able to reach a CER below 10 for all groupings
except for Afroasia") and does not investigate why.

**Language-specific fine-tuning (§5.7.5, Tables 20–21).** For 11 languages with 5–10 h, bespoke CTC
models beat the omnilingual baselines and often reach CER < 5 at 300M and 1B. Practical guidance
straight from the paper: **seed from a CTC checkpoint at 300M/1B (5K steps at lr 1e-5) rather than
from w2v2 (30K steps)**; at 3B the w2v2 seed wins. 300M CTC fine-tuning is **~1 h wall-clock on 32
GPUs**. The 7B-LLM is competitive with the bespoke models without any per-language optimisation, but
the small bespoke models still win on most of the 11.

### 2.4 What it does not solve

Read directly out of the paper and its omissions:

1. **The low-resource bucket is still bad and is where African languages sit.** 18.0 CER mean and only
   36% of languages under CER 10 with <10 h (§5.3.1). "1,600 languages supported" and "1,600 languages
   *usable*" are different claims and the paper's own Table 9 separates them.
2. **Zero-shot is not parity.** §4 says so outright: "zero-shot performance cannot yet match that of
   fully trained systems". 14.4 CER on unseen languages is a research result, not a product.
3. **It requires an established writing system.** §3.2: only languages with "a form of writing in
   frequent use, intelligible to the speaker community"; IPA transcriptions and ad-hoc note-taking are
   excluded. Languages with unsettled orthographies — a large share of the African long tail — are out
   of scope by construction, and the paper does not measure orthographic-variation error.
4. **Coverage is not the lab's coverage.** Grepping the paper's full language list: **Akan is present,
   Ibibio is present, Efik is not.** Amharic, Hausa, Igbo, Swahili, Xhosa, Yoruba, Zulu are present.
   So one of the six languages the lab's app ships is outside the 1,600.
5. **No tone analysis anywhere.** The words "tone" and "tonal" do not appear as an object of study.
   For Atlantic-Congo languages where tone is lexical and orthographies under-mark it, CER cannot
   distinguish "wrong word" from "right word, wrong diacritic", and the paper never separates them.
6. **CER-only reporting for the headline claims.** Tables 5, 8, 9, 10 are all CER. CER flatters
   agglutinative and morphologically rich languages relative to WER, and the paper does not report the
   CER/WER gap that `WAXAL-NET` shows to be large for some African scripts.
7. **Read/prompted speech, mostly.** The commissioned corpus is prompt-elicited monologue. AfriSwitch
   (35.93% WER best system on in-the-wild code-switched African speech) and AfriVox-v2 exist precisely
   because benchmark speech is not deployment speech. AfriVox-v2 already benchmarks the Omnilingual CTC
   models; I did not extract its per-model numbers (§6).
8. **Code-switching is not addressed** in the modelling, only implicitly through data.
9. **No TTS.** The system is transcription-only; the other end of the lab's cascade gets nothing.

---

## §3 Problem table

| # | Problem | Evidence, with numbers | Who works on it | What is missing |
|---|---|---|---|---|
| P1 | **The low-resource cliff persists inside "1,600-language" systems** | Omnilingual §5.3.1: 18.0 mean CER and 36% of languages under CER 10 with <10 h, vs ~3 CER / ~95% above 10 h. Atlantic-Congo 72% success, Afro-Asiatic 66% (§5.3.2) | FAIR; Makerere (`Makerere-2025`); Sunbird (`HowMuchData-2025`) | A *per-language, per-family* account of what breaks. The papers report the gap; nobody decomposes it into tone, orthography, morphology and domain. |
| P2 | **Tone is unmodelled and unmeasured** | Bantu tone spreads across phrases and orthographies omit it (`ToneCurriculum-2026` §1); foundation models at zero-shot WER >100% on Southern Bantu; Efik TTS: **all four** neural models keep tonal errors even at the best MOS 3.80 (`Efik-TTS-2026`); SQA models are insensitive to prosodic errors (`[listing]`) | Marivate/Pretoria + TU Dublin; Gogoi et al. (Tibeto-Burman); YorubaName.com (rule-based Yorùbá) | Tone probing on **Niger-Congo** encoders. `Tone-2025-Interspeech` localises tone to wav2vec2 layers 4–6 for three Tibeto-Burman languages — nobody has replicated that on Yorùbá/Igbo/Efik, and nobody has connected the probe to downstream ASR/TTS errors. |
| P3 | **WER is the wrong instrument and everyone knows it** | SLR: WER dominates; CER and DER "remain underutilized" partly for lack of diacritic-level annotation. `WAXAL-NET`: CER/WER ratios show far higher character accuracy than headline WER for syllabary scripts. `ScriptCollapse-2026`: **21% of 100 model–language pairs produce fluent output in the wrong script while WER stays finite** | Rahman (SFR); WAXAL-NET auditors; the SLR authors | A **diacritic/tone error rate** with a reference implementation and a released per-language normaliser. SFR shows a reference-free metric can be defined and shipped in one paper; the tone analogue does not exist. |
| P4 | **Code-switching is the deployment register and the benchmarks are monolingual** | AfriSwitch: best of five systems **35.93% average WER, none below 24% on any of 16 languages**, "far above published monolingual figures". Africa-targeted training beats scale | Intron Health | Anything modelling-side. AfriSwitch is a benchmark paper; the soft-prompt work (`SoftPrompt-2025`) is on non-African pairs. |
| P5 | **Data quality, not data volume, is the binding constraint past ~50 h** | Sunbird: **WER < 13% at 50 h**, <10% by 200 h; and **38.6% of high-error cases on Kikuyu trace to noisy ground-truth transcripts** | Sunbird AI; AfriVoices-KE (SNR gating + human review); SLR (63.5% of datasets high risk of bias) | Automatic transcript-quality triage that a small lab can run. The Luganda TTS recipe does exactly this for TTS (MOS-estimator filtering) and nobody has ported it to ASR transcripts. |
| P6 | **Relatedness-based transfer does not work, and nobody has said why** | `Relatedness-2026`: **no practically meaningful gain from related-language pre-adaptation once ≥1 h of target data exists**, across six factors × two corpora × four models. `DonorRank` argues donor choice is predictable but not from genealogy | Both preprints, 2026 | Reconciliation. Two 2026 preprints reach near-opposite operational conclusions on the same corpus family (WAXAL). An honest replication is a publishable object. |
| P7 | **Pseudo-labelling and synthetic speech are the only affordable scaling paths, and both are under-characterised** | Real data at **US$100–150/h**; synthetic at **<1%** of that (`SyntheticVoice-2025`). Pseudo2Real: **35% relative WER reduction on AfriSpeech-200** from a weight-difference correction vector | NTU (Pseudo2Real); the RANLP synthetic-voice group; SBPN (distillation + self-training, −29% relative) | Whether synthetic-TTS-trained ASR inherits the TTS system's **tone errors**. Given `Efik-TTS-2026` (tonal errors in all four models), a synthetic pipeline for a tonal language may be laundering a systematic error into the ASR training set. Nobody has tested this. |
| P8 | **TTS evaluation for African languages has no trustworthy automatic proxy** | MOS-Bench: SSQA models generalise poorly OOD; pooling diverse data beats domain-aware methods. TTSDS2: **only 1 of 16 metrics** clears Spearman 0.50 across all domains; its benchmark covers **14 languages**. OpenBibleTTS needed human listeners to find that EveryVoice beats Gemini-TTS on intelligibility in several African languages | Nagoya (MOS-Bench); Edinburgh (TTSDS2); Mila/McGill/Saarland (OpenBibleTTS) | Any validation of UTMOS-class predictors **on an African language**. The Luganda paper used an SSL MOS estimator as a *filter* at 3.5 and it worked, but that is one language, one threshold, no reported correlation with the native-speaker MOS they also collected. |
| P9 | **Child, elderly and field speech fall outside every foundation model's distribution** | Bambara: **children under 10 are the main residual error source** after fine-tuning (WER 0.42→0.22). Xhosa EGRA: performance dominated by data amount and class balance | RobotsMali (Bambara); the Xhosa EGRA group; AfriVoices-KE documents field-collection failure modes | Almost everything. Two papers, two languages, both reading-assessment framed. |
| P10 | **The lab's own languages are unevenly served, and one is invisible** | Efik is **absent** from Omnilingual's 1,600+ list and from FLEURS; Twi/Akan is **absent from FLEURS** (present in Omnilingual and in BibleTTS as Akuapem and Asante Twi). The only Efik speech resource in the literature is **3 hours, one speaker** (`Efik-TTS-2026`) | UNICROSS/Calabar/ML Collective (Efik); the ToAll lab itself | Efik ASR at all. There is an Efik TTS baseline and an English–Efik MT corpus; there is no published Efik ASR. |

---

## §4 Ranked shortlist: 5–7 MS-sized projects

Constraints assumed: one semester, ~125–150 h of Adrian's time, BYU ORC single-GPU SLURM (A100/H100),
lab assets = fine-tuned NLLB-200, in-house TTS voices, Swahili recordings, ToAll's six languages
(Xhosa, Igbo, Efik, Swahili, Twi, Yoruba). "Goals served" refers to the three he named: a small real
contribution he finds interesting; interpretability skills; research-engineering skills.

---

### A1 — Where does a speech encoder put tone, and does that predict its errors on Yorùbá and Igbo? **(top pick)**

- **Question.** `Tone-2025-Interspeech` localises tone information to wav2vec 2.0 layers ~4–6 for three
  Tibeto-Burman languages. Does the same hold for Niger-Congo tone in MMS / w2v-BERT 2.0 / Omnilingual's
  wav2vec2 encoder — and does per-utterance tone-probe accuracy predict the utterance's diacritic
  errors downstream?
- **Why open.** The layer-wise tone result exists for exactly three languages, none Niger-Congo, and it
  stops at probe accuracy: it never connects the probe to transcription errors. Bantu/Kwa tone behaves
  differently (phrasal spreading, high/low with downstep) and orthographies under-mark it
  (`ToneCurriculum-2026` §1). Omnilingual, which now covers these languages, contains no tone analysis at all.
- **Experiment.** (i) Build a tone-labelled probe set from Yorùbá and Igbo data whose orthography *does*
  mark tone — ÌròyìnSpeech and NaijaVoices for Yorùbá/Igbo; the tone marks in the transcript give free
  syllable-level labels after forced alignment. (ii) Frozen encoder, linear probe per layer, per tone
  class; compare four encoders (MMS-1B, w2v-BERT 2.0, AfriHuBERT, OmniASR-300M). (iii) **Random-init
  encoder as the null**, per the repo's rigor floor — probes look interpretable on noise. (iv) Correlate
  per-utterance probe margin with the utterance's diacritic error rate from a fine-tuned CTC head.
- **What counts as an answer.** A layer profile with 95% CIs over ≥5 seeds, family size stated for the
  layer scan and Holm-corrected; a stated effect size for the probe-vs-error correlation, with the
  random-init floor plotted. Answer is "yes" only if the probe beats random init *and* the correlation
  survives multiplicity correction. "No effect" gets an interval, never a bare null.
- **Cost.** Frozen-encoder probing is cheap: feature extraction dominates. ~15–20 GPU-hours plus one
  CTC fine-tune per language (~5–10 GPU-hours each at 300M, per Omnilingual §5.7.5's "1 h on 32 GPUs"
  scaled to one GPU). Comfortably inside a semester.
- **Risk.** *Medium-low.* Main risk is label quality — Yorùbá text in the wild is inconsistently
  tone-marked, so the probe set needs filtering, and that filtering is itself a judgement call that must
  be documented. Forced alignment for Igbo may be shaky; mitigate by falling back to syllable-nucleus
  segmentation. Fallback finding if the probe fails: a negative with an interval is still a result, and
  the diacritic-error-rate instrument (below) survives.
- **Goals served.** All three, and it is the only shortlist item that is *primarily* interpretability.

---

### A2 — A tone/diacritic error rate for African ASR, plus a per-language normaliser

- **Question.** How much of the reported WER/CER for Yorùbá, Igbo and Efik ASR is tone-mark error
  rather than lexical error, and does the ranking of systems change when you separate them?
- **Why open.** The SLR states outright that CER and DER are underused "due to the lack of phoneme or
  diacritic-level annotations". `ScriptCollapse-2026` proves the shape of this contribution is
  publishable — define a metric, measure it across models and languages, show WER was misleading —
  and it did so for *script*, leaving *tone* untouched. `WAXAL-NET` shows CER/WER ratios already carry
  signal nobody reports.
- **Experiment.** Define **DER-T** = diacritic/tone-mark edit rate computed after Unicode NFD
  decomposition, with the base-character edits factored out, so a hypothesis that is lexically right
  but tonally wrong scores 0 on lexical error and non-zero on DER-T. Ship a per-language normaliser
  (tone marks, sub-dot characters, common orthographic variants) as a versioned config. Evaluate 4–6
  systems (Whisper large-v3, MMS-1B-all, OmniASR-300M CTC and 7B-LLM if it fits, AfriHuBERT-CTC,
  the lab's own) on FLEURS Yorùbá/Igbo + NaijaVoices + the Efik corpus, reporting WER, CER, DER-T and
  the CER/WER ratio side by side.
- **What counts as an answer.** A table where at least one system's rank changes under DER-T versus WER,
  with bootstrap CIs; plus a released normaliser + metric with tests. If no rank changes, that too is a
  clean answer and bounds the size of the problem.
- **Cost.** Low: inference only, no training. ~10–15 GPU-hours plus engineering. The engineering *is*
  the point.
- **Risk.** *Low* technically; *medium* on novelty — someone may publish a diacritic-error metric first.
  De-risk by combining with A1 (the probe gives the mechanism, DER-T gives the instrument).
- **Goals served.** Research engineering above all — versioned configs, provenance manifests, invariant
  tests for a new measure — which is exactly what `docs/research_standards.md` §20.3 says to build when
  a track first writes code.

---

### A3 — Does Omnilingual's in-context extension actually work for a language it has never seen? An Efik case study

- **Question.** Omnilingual claims communities can add an unserved language with a handful of paired
  samples. **Efik is not in its 1,600+ list.** Given the existing 3-hour single-speaker Efik corpus, how
  far does in-context extension get, and how does it compare against (a) fine-tuning a 300M CTC head on
  the same 3 hours and (b) MMS fine-tuning?
- **Why open.** The zero-shot evaluation in §5.4 holds out **32 languages that were in the training
  distribution's world** — high-resource languages excluded deliberately, retrieval base drawn from
  their own training sets. A genuinely never-collected language with 3 hours of single-speaker audio,
  chosen because a real product needs it, is a different test and it is the test the claim is marketed
  on. It also directly serves the lab: ToAll ships Efik.
- **Experiment.** Re-split the Efik TTS corpus into an ASR-style train/dev/test (it is read speech,
  single speaker — a limitation to state, not hide). Conditions: OmniASR LLM-ASR zero-shot with N ∈
  {1, 3, 5, 10} context examples, random vs SONAR-retrieved selection (§5.5's method); OmniASR-300M CTC
  fine-tuned per §5.7.5's recipe (CTC seed, lr 1e-5, 5K steps); MMS-1B fine-tuned; and a Whisper
  baseline. Report WER, CER **and DER-T** from A2.
- **What counts as an answer.** A curve of CER against context size with CIs, against the fine-tuning
  baseline at the same data budget. Satisfied-when should be written before running: e.g. "in-context
  extension reaches within X CER of same-budget fine-tuning". Either direction is publishable as an
  audit of a widely-repeated claim.
- **Cost.** Moderate. The 7B LLM-ASR at inference on one A100 is the main question mark — the 300M/1B
  variants are the safe path and the paper releases them. ~20–30 GPU-hours.
- **Risk.** *Medium-high.* Three specific risks: (i) the 7B checkpoint may not fit comfortably —
  mitigate by pre-registering the 1B as the headline model; (ii) 3 hours of one speaker is a weak base
  and single-speaker read speech will overstate everything — state it in the abstract, not the appendix;
  (iii) the Efik corpus is "available upon request" style — confirm access **before** committing.
- **Goals served.** Small real contribution (an independent audit of a headline claim on a language the
  system does not cover) + research engineering. Light on interpretability.

---

### A4 — Does synthetic TTS data launder tone errors into ASR? A Swahili↔Yorùbá contrast

- **Question.** Synthetic voice data is the affordable path (<1% of US$100–150/h). But the Efik study
  found tonal errors in **all four** neural TTS systems. If you train ASR on TTS output for a tonal
  language, do the TTS system's tone errors show up as a systematic, measurable bias in the ASR?
- **Why open.** Nobody has asked. `SyntheticVoice-2025` reports aggregate WER gains and does not
  decompose the error; `Efik-TTS-2026` reports TTS tone errors and does not follow them downstream.
  The Swahili/Yorùbá contrast is the control: **Swahili is not tonal in its standard orthography;
  Yorùbá is**, so an effect that appears for Yorùbá and not Swahili is attributable to tone rather than
  to synthetic-data artefacts in general.
- **Experiment.** Generate matched synthetic corpora for Swahili and Yorùbá (MMS-TTS and one
  flow-matching system, e.g. F5-TTS via the cross-lingual transcript-free variant). Fine-tune the same
  ASR base on (real) / (synthetic) / (mixed) at matched hours. Measure WER, CER and **DER-T** on real
  held-out speech. Primary contrast: the DER-T gap between synthetic-trained and real-trained, Yorùbá
  minus Swahili.
- **What counts as an answer.** A difference-in-differences with CIs over ≥5 seeds, seeds varying data
  order and shuffling, not just init. Report the interval whichever way it goes.
- **Cost.** Highest of the shortlist — TTS generation plus 6+ ASR fine-tunes. ~40–60 GPU-hours. Trim by
  dropping to one TTS system and 3 data budgets.
- **Risk.** *Medium.* Needs A2's metric to exist first, so it is naturally a follow-on. Confounds are
  real (speaker diversity differs between the two synthetic sets) and must be controlled by design.
- **Goals served.** The most genuinely novel question on the list, and it uses the lab's Swahili
  recordings as the real-data arm.

---

### A5 — Reconcile the relatedness contradiction on WAXAL

- **Question.** `Relatedness-2026` finds no meaningful gain from related-language pre-adaptation past
  1 h of target data; `DonorRank` builds a ranker that predicts effective donors, partly on the same
  WAXAL corpus. Which holds, under what data budget, and is the disagreement about the *effect* or
  about *how it was measured*?
- **Why open.** Two 2026 preprints, opposite operational advice, overlapping data. A careful
  replication with a pre-registered satisfied-when is exactly the kind of negative-or-reconciling
  result the repo's registry is built to hold.
- **Experiment.** WAXAL (CC-BY-4.0, on HF) with a fixed base model. Grid: target-data budget
  ∈ {15 min, 1 h, 5 h} × donor selected by (genealogy / DonorRank-style features / random) — the random
  arm is the calibrated null both papers need. ≥5 seeds.
- **What counts as an answer.** Effect of donor choice relative to the random-donor arm, with CIs, at
  each budget. The interesting outcome is a budget threshold: relatedness may matter below 1 h and
  vanish above, which would reconcile both papers.
- **Cost.** Moderate-high: it is a grid. ~35–50 GPU-hours; trim by cutting to two budgets.
- **Risk.** *Medium.* Low novelty ceiling — it is a replication — but low failure probability, and it
  produces a reusable adaptation harness for the lab.
- **Goals served.** Research engineering and rigour; less interpretability, less novelty.

---

### A6 — Port the Luganda crowdsourced-TTS recipe to Swahili, with a validated MOS proxy

- **Question.** The Luganda recipe (multi-speaker close-intonation selection + enhancement + MOS-estimator
  filtering at 3.5) took MOS from 2.5 to 3.55. Does it transfer to the lab's own Swahili recordings — and
  **does the MOS estimator it depends on actually correlate with native-speaker judgements in Swahili?**
- **Why open.** MOS-Bench says SSQA models generalise poorly out of domain; TTSDS2 says only 1 of 16
  metrics clears Spearman 0.50 everywhere. The Luganda paper used an SSL MOS estimator as a filter and it
  worked, but reported no correlation between the estimator and the native-speaker MOS it also collected.
  Validating an automatic proxy on an African language is a small, clean, genuinely missing measurement.
- **Experiment.** Two arms. (i) TTS: apply the recipe to the lab's Swahili data; compare to a
  single-speaker baseline and to MMS-TTS Swahili. (ii) Metric validation: collect native-speaker MOS on
  a stratified sample, and report Spearman/PCC of UTMOS-class predictors **and TTSDS2** against it.
- **What counts as an answer.** Arm (ii) is the contribution and it succeeds whatever the correlation
  turns out to be. Report n_raters, per-rater variance, and the CI on the correlation.
- **Cost.** Low-moderate GPU (~15–25 h); the real cost is **human listening-test logistics**.
- **Risk.** *Medium, and it is not technical.* Recruiting native Swahili raters, IRB, and rater
  reliability are the failure modes. Do not start this without the raters lined up.
- **Goals served.** Directly useful to the lab's African TTS project; research engineering; least
  interpretability.

---

### A7 — Code-switched ASR for the ToAll register (stretch / backup)

- **Question.** AfriSwitch shows no system below 24% WER on any of 16 languages. Does cheap adaptation —
  soft prompts (`SoftPrompt-2025`) or Pseudo2Real's weight-difference correction — close any of that gap
  on the lab's languages?
- **Why open.** AfriSwitch is a benchmark; the modelling response does not exist for African pairs.
- **Experiment.** AfriSwitch subsets for Swahili/Yorùbá/Igbo; baseline vs soft-prompt vs Pseudo2Real
  correction vector; report WER overall and **on the English spans specifically**, which AfriSwitch's
  switch-level tags make possible and which no paper has reported.
- **What counts as an answer.** Effect size with CI against the zero-shot baseline; span-level WER as the
  novel measurement.
- **Cost.** Moderate (~25–35 GPU-hours).
- **Risk.** *Medium-high* — depends on AfriSwitch being released, which I have not verified (§6).
- **Goals served.** Contribution + engineering. Listed last because of the availability risk.

**Ranking rationale.** A1 first because it is the only one that is primarily interpretability, has a
free null (random init), is cheap, and produces a mechanism claim. A2 second because it is the
instrument A1, A3 and A4 all want, is low-risk, and is exactly the engineering discipline the repo's
standards demand. A3 third: highest lab relevance, audits a headline claim, but carries a real data-access
risk. A4 is the most novel and most expensive. A5–A7 are solid but either replication-shaped, logistics-
bound, or availability-risked.

---

## §5 Datasets for the lab's eight languages

Rows are what I could verify from documents I opened, not what exists. "—" means *I found no evidence
in the papers I read*, which is not the same as *does not exist*; see §6.

| Language | In FLEURS? | In Omnilingual's list? | Dedicated speech corpora found (size, source) | TTS data | Notes |
|---|---|---|---|---|---|
| **Swahili** | **Yes** (verified by grep of the FLEURS PDF) | **Yes** ("Swahili (individual language)") | Kencorpus (Swahili/Dholuo/Luhya); ALFFA Swahili (via SLR); WAXAL Kiswahili ASR; Digital Umuganda Afrivoice Swahili (cited in Omnilingual's refs); **the lab's own recordings** | WAXAL Kiswahili TTS (part of 235 h single-speaker); MMS-TTS | Best-served of the eight. Not tonal in standard orthography → the natural **control** for any tone experiment. |
| **Hausa** | **Yes** | **Yes** | NaijaVoices (1,800 h across Igbo+Hausa+Yoruba, 5,000+ speakers); ALFFA Hausa; SBPN covers it | **BibleTTS Hausa** (up to 86 h, 48 kHz single speaker, CC-BY-SA); WAXAL Hausa TTS | Omnilingual §6 reports Hausa transcription deployed in Nigerian community clinics. |
| **Yoruba** | **Yes** | **Yes** | NaijaVoices; ÌròyìnSpeech; YORÙLECT (4 regional dialects); SBPN | **BibleTTS Yoruba**; WAXAL Yoruba TTS; the rule-based diphone synthesiser (651 units, 5 tonal variants per CV) | Tone-marked orthography → the best-instrumented tonal language for probe work. |
| **Igbo** | **Yes** | **Yes** | NaijaVoices; SBPN | WAXAL Igbo TTS | Diacritic restoration has a dedicated 270-page thesis (`Igbo-2026`). |
| **Xhosa** | **Yes** | **Yes** | Swivuriso (3,000 h across 7 SA languages **incl. isiXhosa**; per-language hours not extracted); the Xhosa EGRA **child-speech** set (ten words/letters, available on request); NCHLT is used as a transfer target in `ToneCurriculum-2026` | — (none found in what I read) | Nguni: `ToneCurriculum-2026` finds **w2v-BERT beats Whisper on Nguni by 3–4 WER points**. |
| **Amharic** | **Yes** | **Yes** | ALFFA Amharic; the SLR names Ethiopia as the single largest contributor of hours (Amharic, Oromo, Tigrigna); WAXAL partner Addis Ababa University | — | Ge'ez script → **CER/WER divergence and script collapse are live risks** (`WAXAL-NET`, `ScriptCollapse-2026`). |
| **Twi / Akan** | **No** — absent from FLEURS's language list (verified by grep) | **Akan: yes**; "Twi" as such: not found | WAXAL Akan ASR; DONDO covers Ghanaian varieties | **BibleTTS Akuapem Twi and Asante Twi** (up to 86 h each, 48 kHz) — the strongest TTS asset of the eight; WAXAL Akan TTS | The **FLEURS gap matters**: any evaluation that uses FLEURS silently drops Twi. Akuapem vs Asante is a real dialect split that BibleTTS separates and most systems do not. |
| **Efik** | **No** | **No** — grep of the full language list finds **Ibibio** but **not Efik** | **None found for ASR.** | **3 hours, 2,632 utterances, single speaker** (`Efik-TTS-2026`) — the only Efik speech corpus in the literature I read | Lower Cross, tonal, ~1.5M L1 / ~3M L2. Text side: English–Efik MT corpus + Ibom NLP. **The clearest gap on the list, and one the lab already ships a product in.** |

Cross-cutting resources: **AfriSpeech-200** (200 h Pan-African accented *English*, clinical + general)
and **AfriSpeech-MultiBench** / **AfriVox-v2** / **AfriSwitch** for evaluation; **WAXAL** (24 languages,
CC-BY-4.0, `google/WaxalNLP`) is the most permissively licensed multi-language ASR+TTS pair.

---

## §6 Verification debt

Things a reader should not take from this document without checking.

**Search-tooling caveat.** The session's WebSearch budget (200 calls) was **already exhausted** when I
started, and both the arXiv Atom API and the Semantic Scholar API returned HTTP 429 to every request,
including with retry/backoff. All discovery here was done by **WebFetch against arXiv's HTML search
pages**, whose results are summarised by a small model. That summariser **demonstrably erred at least
once**: it attributed *"Towards Digital Preservation of Efik: TTS"* to arXiv 2607.04814, which on
download turned out to be *"Evaluating the Effect of Linguistic Relatedness on Cross-Lingual Transfer"*
(the Efik paper is 2607.04515). **Every arXiv ID I actually downloaded was verified by opening the PDF
and reading its title**; IDs marked `[listing]` were not. Treat all `[listing]` rows as unconfirmed.

1. **Venues.** Only four venue claims in this document are confirmed from the paper itself: Deep
   Learning Indaba 2025 / PMLR 302 (`Makerere-2025`), AfricaNLP@ICLR 2024 (`Luganda-2024`,
   `Africa-Centric-2024`), and the two "Submitted to …" headers (`AfriNames-2023` → Interspeech,
   `TransferTTS-2023` → SSW), which are *submissions*, not acceptances. Everything else labelled
   Interspeech / ACL / TACL / IJCNLP-AACL / LREC / TASLP rests on the **arXiv comment field**. Confirm
   against proceedings before citing in writing.
2. **MMS as JMLR 2024 and FLEURS as SLT 2022.** Both are conventional citations that I did **not**
   verify; the PDFs in this folder carry no venue.
3. **Omnilingual's language list.** My "Efik absent / Akan present / Ibibio present" finding comes from
   grepping the extracted text of the appendix language table. Text extraction from a multi-column
   appendix can drop entries. **Re-check against the released model's language list** in
   `facebookresearch/omnilingual-asr` before building A3 on it.
4. **XTTS's 16 languages.** I asserted "no African language among the 16" from the paper's framing, not
   from an enumerated per-code audit of the list. Check before repeating.
5. **AfriVox-v2's numbers for Omnilingual.** The abstract says it benchmarks "the Omnilingual CTC
   models". I did not extract those numbers. They would directly strengthen or weaken §2.4 item 7.
6. **Per-language hours** in Swivuriso, WAXAL, NaijaVoices and AfriVoices-KE. I have corpus totals, not
   per-language breakdowns. §5's rows say "covers", never "has N hours of", except where a number is
   printed in the abstract I read.
7. **Common Voice per-language coverage.** Not verified. My only source is Omnilingual §3.3.2 saying
   the Open Multilingual Speech Fund brought Common Voice to "well over 300 languages". Per-language
   hours for the eight languages in §5 need to come from the Common Voice release stats, which I could
   not fetch.
8. **Chatterbox has no primary paper** that I could find — arXiv search surfaces only
   *Chatterbox-Flash* (2605.30748, preprint) and downstream users. If the brief's mention of Chatterbox
   matters, treat it as a model release, not a citable method.
9. **Data availability for A3 and A7.** The Efik corpus's release terms and AfriSwitch's actual public
   release were **not** checked. Both projects are gated on this; check first.
10. **Papers named but not downloaded** (all `[listing]`, none opened): *Investigating Human-Model
    Discrepancies in SQA via Acoustic and Prosodic Perturbations* (2606.19951), *Language Barriers:
    cross-lingual MOS* (2502.13004), *ASR of African American English* (2506.06888), *SBPN* appears at
    two different IDs in two different listings (2605.17710 downloaded and title-verified;
    a listing also gave 2609.01287 for a different paper), *Continued Pretraining for Low-Resource
    Swahili ASR* (2603.11378 — a file of that name already existed in the folder from the parallel
    agent; I did not open it), *A Survey of Text and Speech Resources for Hausa and Fongbe*
    (2605.22828, "to appear IEEE SDS 2026").
11. **The duplicate cleanup.** I deleted twelve of **my own** byte-identical downloads where the
    parallel agent's copy of the same PDF already existed under a different name. If that agent's files
    move or are renamed, the twelve filenames cited above break. Verified identical by `md5sum` at the
    time of deletion.
12. **`speech-translation/` in this repo** exists and was **not** consulted, per the brief, which marks
    it old and untrusted.
