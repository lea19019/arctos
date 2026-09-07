# Speech translation and speech models — landscape notes (2026-09-06)

**Scope.** Current (2025–2026) models for ASR, speech translation (S2TT/S2ST), speech LLMs and TTS, biased
toward **open weights that run on one A100/H100** and toward **low-resource / African languages** —
specifically the six the lab ships: **Xhosa (xho), Igbo (ibo), Efik (efi), Swahili (swh), Twi/Akan
(twi/aka), Yoruba (yor)**.

**Evidence discipline.** Per the shared brief: read vs inferred is marked; venues are given only as
printed/verified; **arXiv-only work — including every model technical report — is labelled `preprint`**;
unopened items are marked `[unopened]` and collected in §6. Model metadata (licence, creation date,
language counts, parameter counts) was read from the Hugging Face `/api/models` JSON or the model card on
**2026-09-06** unless stated otherwise. Downloaded PDFs are named inline as `papers/<file>.pdf` and indexed
in `docs/literature/papers_index.md` under "Speech translation and speech models (added 2026-09-06) → Landscape".

**One caveat on method.** The WebSearch budget for this session was exhausted early, so discovery ran on the
arXiv API, the Hugging Face API, ACL Anthology and direct page fetches. A model or paper published outside a
known HF org or venue could have been missed. Where a subagent supplied a quotation, I re-grepped the source
PDF before using it; the four IWSLT 2026 quotations in §4.7 were verified this way.

**Related, and untrusted:** `/home/adrian/Repos/arctos/speech-translation/` exists in this repo. It is old
and, per the brief, not to be relied on. Nothing here was taken from it.

---
## 1. Model table

**How to read this.** Every `license`, `created` date and language count marked ✓ below was read from the
Hugging Face `/api/models` JSON or the model card on 2026-09-06; parameter counts come from cards, papers or
safetensors indices as noted. **"Fits 1 GPU"** means one 80 GB A100/H100 in BF16 for inference — and the
honest answer is that it is almost never the binding constraint. The binding constraints are **licence** and
**language coverage**, and in 2026 those two pull against each other.

African coverage is given against the lab's six: **Xhosa (xho), Igbo (ibo), Efik (efi), Swahili (swh),
Twi/Akan (twi/aka), Yoruba (yor)**.

### 1.1 ASR

| Model | Date | Task(s) | Architecture (one line) | Params | Open? | Languages / African coverage | License | 1 GPU? | HF id |
|---|---|---|---|---|---|---|---|---|---|
| **Omnilingual ASR** (CTC / LLM / W2V / ZS) | 2025-11 | ASR; X→En S2TT (reported, not shipped) | wav2vec 2.0 encoder + linear (CTC) or Transformer decoder (LLM-ASR); ZS variant reads N in-context speech–text pairs | 300M / 1B / 3B / **7B** (encoders 317M–6488M) | ✓ | 1,600+; **swh H, yor H, ibo H, xho M, aka M; efi ✗** (Ibibio L, Anaang L) | **Apache-2.0** (corpus CC-BY-4.0) | ✓ (7B LLM ≈17 GiB) | `facebook/omniASR-{W2V,CTC,LLM}-{300M,1B,3B,7B}`, `facebook/omniASR-LLM-7B-ZS` |
| **Whisper large-v3** | 2023-11 | ASR; X→En S2TT | Encoder–decoder Transformer on log-mel, weakly supervised | 1.54B | ✓ | 99; **only 10 African** (ar, af, am, ha, ln, mg, sn, so, sw, yo) — **no ibo, xho, twi, efi** | apache-2.0 (card) | ✓ | `openai/whisper-large-v3` |
| **Whisper large-v3-turbo** | 2024-10 | ASR; X→En S2TT | Same, decoder cut 32→4 layers | 809M | ✓ | 99, as above | **mit** (card) | ✓ | `openai/whisper-large-v3-turbo` |
| **MMS** (`mms-1b-all`) | 2023-05 | ASR (+ TTS, LID as separate models) | wav2vec 2.0 1B + per-language CTC adapters | 964.8M + adapters | ✓ | 1,125 adapter codes; **xho ✓ ibo ✓ swh ✓ aka ✓ yor ✓, efi ✗, twi ✗** | **cc-by-nc-4.0** | ✓ | `facebook/mms-1b-all` |
| **w2v-BERT 2.0** | 2023-12 | SSL encoder (fine-tune base) | Conformer encoder from SeamlessM4T v2; 4.5M h / 143+ langs unlabelled | **580.5M** | ✓ | 96 tags, **18 African incl. ig, xh, zu, wo, lg, ny, om** that Whisper lacks | **mit** | ✓ | `facebook/w2v-bert-2.0` |
| **AfriHuBERT** | 2024–25 (Interspeech) | SSL encoder | mHuBERT-147 continued on 10K+ h African speech | ~95M (large ~315M, xlarge ~962M, 2026-06) | ✓ | 1,230 langs, 1,226 indigenous African | base card says bare `cc` — **ambiguous, verify**; large/xlarge Apache-2.0 | ✓ | `ajesujoba/AfriHuBERT` |
| **DONDO** | 2026-07 | ASR | w2v-BERT 2.0 fine-tuned, one-hot language identity as prefix frames | 580M-class ×26 | ✓ | 27 African varieties (Ghana, Sierra Leone, Nigeria, Senegal, Kenya, Zimbabwe); multilingual families 10–13% avg WER | **Apache-2.0, commercial use permitted** | ✓ | `KhayaAI/*` |
| **NVIDIA Canary-1B-v2** | 2025-08 | ASR (25) + X→En and En→X AST | FastConformer encoder (32L) + 8L Transformer decoder, unified 16,384-token SPM across 25 langs | 978M | ✓ | **25, all European** (bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk). **Zero African.** | cc-by-4.0 | ✓ | `nvidia/canary-1b-v2` |
| **NVIDIA Parakeet TDT 0.6B v3** | 2025-08 | ASR only (no AST) | FastConformer + token-and-duration transducer; throughput champion (v2: RTFx 3386) | 600M | ✓ | **identical 25 European set** — no African | cc-by-4.0 | ✓ | `nvidia/parakeet-tdt-0.6b-v3` |
| *Granary (dataset, for context)* | 2025-05 | ASR + AST training data | Dual-pass Whisper-large-v3 pseudo-labelling + hallucination filtering; AST pairs from EuroLLM-9B | ~643k h ASR + ~351k h AST ≈ **1M h** | ✓ | **the same 25 European languages — zero African** | cc-by-4.0, but per-subset (YODAS-Granary is CC-BY-3.0); source audio keeps its own licences | — | `nvidia/Granary` (**Interspeech 2025**, arXiv:2505.13404) |

### 1.2 Speech translation (S2TT / S2ST)

| Model | Date | Task(s) | Architecture | Params | Open? | Languages / African | License | 1 GPU? | HF id |
|---|---|---|---|---|---|---|---|---|---|
| **SeamlessM4T v2 Large** | 2023-12 | ASR, S2TT, S2ST, T2TT, T2ST | w2v-BERT 2.0 encoder + NLLB-initialised text decoder + non-autoregressive text-to-unit + vocoder (UnitY2) | 2.3B | ✓ | 101 speech-in / 96 text / **35 speech-out**. **swh full both ways; ibo, yor speech-in→text-out; xho speech-in ONLY; twi/aka, efi, hau absent** | **cc-by-nc-4.0** | ✓ | `facebook/seamless-m4t-v2-large` |
| **SeamlessStreaming** | 2023-12 | Simultaneous S2TT/S2ST | EMMA streaming policy over the Seamless stack | 2.5B | ✓ | ~100 | cc-by-nc-4.0 | ✓ | (seamless_communication repo) |
| **SeamlessExpressive** | 2023-12 | Prosody-preserving S2ST | Expressive vocoder + prosody transfer | n/s | **gated** | limited; **no African languages in mExpresso** | proprietary "Seamless" licence, approval form | ✓ | gated |
| **Hibiki** | 2025-02 | Simultaneous S2ST | **Decoder-only multistream LM**: source and target speech processed synchronously, text and audio tokens emitted jointly | 1B / 2B | ✓ | **fr→en only** | cc-by-4.0 (weights) | ✓ | `kyutai/hibiki-{1b,2b}-pytorch-bf16` |
| **Hibiki-Zero** | 2026-02 | Simultaneous S2ST | Same, trained on **sentence-level pairs only** + GRPO RL for latency/quality | **3B** | ✓ | **fr, es, pt, de → en**; adds a language with **<1,000 h** | ⚠ **UNRESOLVED** — the HF card has **no `license` field** (confirmed via API); the GitHub README says **MIT**; a second reading of the rendered HF page reported **CC-BY-NC-SA-4.0**. **Do not rely on MIT without checking.** | ✓ (8–12 GB) | `kyutai/hibiki-zero-3b-pytorch-bf16` |
| **StreamSpeech** | 2024-06 | Simultaneous S2ST/S2TT | Multi-task CTC + unit decoder, joint ASR/S2TT/S2ST | ~0.5B | ✓ | fr/es/de→en | (repo) | ✓ | (ictnlp/StreamSpeech) |

### 1.3 Speech LLMs / omni models

*None of these is a serious option for the lab's six languages — their language lists are Chinese, English
and European. Listed for completeness and because they are where the field's attention is.*

| Model | Date | Task(s) | Architecture | Params | Open? | Languages / African | License | 1 GPU? | HF id |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-Omni-30B-A3B** | 2025-09 | ASR, AST, audio QA, speech-out | Thinker (MoE LLM, text out) + Talker (speech codec tokens), AuT audio pretraining, multi-codebook MTP | **35B total / ~3B active** | ✓ | text 119; **speech in 19, speech out 10**; **no African** | card `other`, `license_name: apache-2.0` | **borderline** — card gives 78.85 GB BF16 for 15 s A/V, 88.52 GB for 30 s; audio-only fits | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (`-Thinking`, `-Captioner`) |
| **Qwen2.5-Omni-7B** | 2025-03 | same | Thinker–Talker + TMRoPE | **11B actual** | ✓ | **no African** | card `other`, `license_name: apache-2.0`. *The 3B variant is `qwen-research` — non-commercial.* | ✓ (~42 GB) | `Qwen/Qwen2.5-Omni-7B` |
| **Voxtral Mini / Small** | 2025-07 | ASR, AST, audio understanding | Whisper-style audio encoder + Ministral/Mistral-Small LLM | 3B / 24B | ✓ | 8 declared, **no African** | **apache-2.0** | ✓ | `mistralai/Voxtral-{Mini-3B,Small-24B}-2507` |
| **Phi-4-multimodal** | 2025-02 | ASR, AST, vision, text | Shared LLM with **LoRA-routed** speech and vision modalities | 5.6B | ✓ | 24 declared, **no African** | **mit** | ✓ | `microsoft/Phi-4-multimodal-instruct` |
| **Ultravox v0.5–v0.7** | 2025-02 → 2025-12 | Speech-in → text-out LLM | Whisper-large-v3-turbo encoder → trained adapter → **frozen** text LLM, KD-trained against the backbone's logits | adapter ~1.37 GB; LLM loaded from `base_model` | ✓ | **42 declared, including `sw` (Swahili) — the only African language in any speech LLM found, and understanding only** | **mit** *on Fixie's artifacts only*; the base weights keep their own terms (Llama Community / Gemma / Apache) | ✓ for 1B/8B/27B/32B variants; 70B and 355B do not | `fixie-ai/ultravox-v0_5-llama-3_1-8b` |
| **Kimi-Audio-7B-Instruct** | 2025-04 | ASR, audio QA, speech-out | Whisper encoder at 12.5 Hz + discrete semantic tokens → Qwen2.5-7B-init LLM with parallel text/audio heads → flow-matching detokenizer + BigVGAN | **9.77B** (the "7B" is the LLM core only) | ✓ | en/zh | **mit** | ✓ (~19.5 GB) | `moonshotai/Kimi-Audio-7B-Instruct` |
| **Step-Audio 2 mini** | 2025-08 | ASR, S2TT, **S2ST**, audio QA, tool calling, speech-out | End-to-end audio-in/audio-out LLM (Qwen2.5-7B lineage) with retrieval/tool calling | **8.32B** | ✓ | en, zh, yue, ja, ar; **no African** | **apache-2.0** | ✓ (~16.6 GB). *The flagship Step-Audio 2 was never open-weighted; `Step-Audio-Chat` 132B and `-AQAA` 137B do **not** fit one card.* | `stepfun-ai/Step-Audio-2-mini` |
| **MiniCPM-o 2.6 / 4.5** | 2025-01 / 2026-02 | omni (vision+audio+speech-out, full-duplex) | SigLip vision + Whisper audio encoder → Qwen2.5-7B (2.6) or Qwen3-8B (4.5) → `minicpmtts` head (≈CosyVoice2) | 8B / 9B | ✓ | **speech explicitly en/zh only** — the "30+ languages" claim is inherited vision/text | **apache-2.0** (4.5's card drops the old registration-required licence) | ✓ | `openbmb/MiniCPM-o-2_6`, `openbmb/MiniCPM-o-4_5`. *There is no "MiniCPM-o 4".* |
| **Gemma 3n E4B** | 2025-06 | ASR + AST, text out | Per-Layer-Embeddings + MatFormer LLM + **USM-based audio encoder** (one token / 160 ms) | 8B raw / ~4B effective | ✓ | audio language list **not published**; Google names strong AST only for en↔es/fr/it/pt. **No African named.** | **gemma** licence (gated) | ✓ | `google/gemma-3n-E4B-it` |
| **Gemma 4** | 2026-03 | ASR + speech-to-translated-text, text out | E4B: ~300M audio encoder, 30 s cap. **The 12B `gemma4_unified` is encoder-free — raw audio and image patches ingested directly** | E2B / E4B (8B total, 4.5B effective) / 12B | ✓ | "35+ out of the box, 140+ in pretraining"; **no per-language audio list, no African named** | **apache-2.0** — a real licence change from Gemma 3n | ✓ | `google/gemma-4-{E2B,E4B,12B}-it` |
| **Moshi** | 2024-09 | Full-duplex speech dialogue | Multistream LM over Mimi codec | 7B-class | ✓ | English | cc-by-4.0 | ✓ | `kyutai/moshiko-pytorch-bf16` |
| **LLaMA-Omni 2** | 2025-05 | Speech-in/speech-out LLM | Whisper-large-v3 → adaptor → Qwen2.5-Instruct → autoregressive streaming speech decoder + **CosyVoice 2** flow matching and vocoder | 0.5B–32B | ✓ | base English-only, `-Bilingual` en+zh; **no African** | ⚠ **contradiction**: HF card says Apache-2.0, the GitHub repo says academic-research-only, **not for commercial use** | ✓ (all sizes incl. 32B) | `ICTNLP/LLaMA-Omni2-{0.5B…32B}` |
| **Freeze-Omni** | 2024-11 | Full-duplex speech dialogue | Chunk-wise streaming encoder → **frozen** Qwen2-7B-Instruct → AR speech decoder + turn-taking state head | ~7.5B | ✓ | not stated; Qwen2 backbone implies en/zh | **apache-2.0** | ✓ | `VITA-MLLM/Freeze-Omni` |
| **SALMONN / SALMONN-2** | 2023-10 / 2026-07 | ASR, S2TT, audio+music captioning, audio QA | Dual Whisper-large-v2 + BEATs encoders → window-level Q-Former → Vicuna-13B + LoRA. SALMONN-2 swaps in a SPEAR SSL encoder → Qwen3 + LoRA | ~14B / 9B & 30B-A3B | ✓ (SALMONN-omni is **paper only, no weights**) | en | **apache-2.0** | ✓ | `tsinghua-ee/SALMONN`, `marcoyang/SALMONN-2-8B` |
| *GPT-realtime / Gemini native audio* | 2025–26 | full-duplex S2S, tool use | closed | — | ✗ | broad | proprietary API | — | — |

### 1.4 TTS

*The 80 GB constraint is not binding anywhere here — the largest open TTS model in this landscape is ~9.3B.*

| Model | Date | Architecture | Params | Open? | Languages / African | License | HF id |
|---|---|---|---|---|---|---|---|
| **OpenBibleTTS** (VITS / EveryVoice / F5-TTS arms) | 2026-06 | Per-language monolingual models trained from scratch on a 3,469 h Bible corpus | 18.2M / 36.3M / 335.8M | ✓ | **37 langs, 19 African**: swh 96.38 h, ibo 94.46 h, yor 89.91 h, Twi-Asante 78.87 h, Twi-Akuapem 71.20 h; **xho ✗, efi ✗** | **CC-BY-SA-4.0** — best-licensed real African coverage | `multilingual-tts/{VITS,F5-TTS,EveryVoice}-OpenBible-*`, dataset `multilingual-tts/open-bible` |
| **MMS-TTS** | 2023-09 | VITS, single-speaker, 16 kHz, per-language checkpoint | 36.3M each | ✓ | 1,108 TTS checkpoints but **swh ✓ yor ✓ aka ✓ hau ✓; ibo ✗ xho ✗ zul ✗ efi ✗ (LID-only)** | **cc-by-nc-4.0** | `facebook/mms-tts-<iso639-3>` |
| **XTTS-v2** | 2023-12 (frozen) | GPT-2-style + VQ-VAE + HiFiGAN, zero-shot cloning | ~460M (inferred from file sizes; never published) | ✓ | 17, **none African** | **Coqui Public Model License — non-commercial**; Coqui defunct, fork `idiap/coqui-ai-TTS` (MPL-2.0 code) | `coqui/XTTS-v2` |
| **F5-TTS** | 2024-10 | DiT + ConvNeXt, flow matching | 336M | ✓ | en/zh base, none African | cc-by-nc-4.0 | `SWivid/F5-TTS` |
| **CosyVoice 2 / 3** | 2024-12 / 2025-12 | LLM + flow matching + HiFiGAN | 0.5B | ✓ | **9 (unchanged v2→v3), none African**; the paper's 1.5B was never released | **apache-2.0** | `FunAudioLLM/CosyVoice2-0.5B`, `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| **Chatterbox** | 2025-05 (multilingual 2026) | Llama backbone + S3Gen | 0.5B | ✓ | **23 incl. Swahili** | **MIT** | `ResembleAI/chatterbox` (weights `t3_mtl23ls_v3.safetensors`) |
| **Kokoro** | 2024-12 | StyleTTS2 + ISTFTNet | **82M** | ✓ | 8, **none African**; card warns non-English support is "absent or thin due to weak G2P" | **apache-2.0** | `hexgrad/Kokoro-82M` |
| **Sesame CSM** | 2025-03 | Llama + Mimi RVQ decoder | 1.55B | ✓ | English | apache-2.0 | `sesame/csm-1b` |
| **Dia / Dia2** | 2025-04 / 2025-11 | SoundStorm-style over DAC; Dia2 streaming over Mimi | 1.61B / 1.08B–1.92B | ✓ | English | apache-2.0 | `nari-labs/Dia-1.6B`, `nari-labs/Dia2-{1B,2B}` |
| **MaskGCT** | 2024-10 | Masked generative codec transformer, non-autoregressive | ~1.58B across 4 components | ✓ | 6, none African | code MIT, **weights cc-by-nc-4.0** | `amphion/MaskGCT` |
| **Orpheus** | 2025-03 | Llama-3.2-3B + SNAC codec | 3.78B | ✓ | en + 7, none African | apache-2.0 (some pretrained ckpts under llama3.2 licence) | `canopylabs/orpheus-3b-0.1-ft` |
| **Fish S2-Pro** | 2026-03 | Slow-AR 4B + Fast-AR 400M over 10-codebook RVQ | 4.56B | ✓ | ~80 incl. **Yoruba, Swahili, Shona, Amharic, Afrikaans** | **Fish Audio Research License — commercial prohibited** | `fishaudio/s2-pro` |
| **Higgs TTS 3** | 2026-06 | audio LLM | 4.65B | ✓ | **~18 African langs with published WER tiers** (only source with Xhosa) | Boson Research & Non-Commercial | `bosonai/higgs-tts-3-4b` |
| **OmniVoice** | 2026-03 | Diffusion LM over codec | 613M | ✓ | claims **600+** incl. am/yo/ig/sw/ha/zu — **publishes zero per-language quality numbers, and loses to 18M monolingual models on intelligibility** | code apache-2.0, **weights cc-by-nc** | `k2-fsa/OmniVoice` |
| **VoxCPM2** | 2026-04 | — | 2.29B | ✓ | Swahili only of the six | **apache-2.0** | `openbmb/VoxCPM2` |
| **MOSS-TTS v1.5** | 2026-05 | — | ~8B / 1.7B | ✓ | 30+, Swahili | **apache-2.0** | `OpenMOSS-Team/MOSS-TTS-v1.5` |
| **Efik TTS** (venue unverified) | 2026 | VITS / MMS-TTS / SpeechT5 / Orpheus compared; MMS-TTS won | — | corpus only | **Efik**, 2,632 utts / 3 h, single speaker; MOS 3.80 ± 0.63; "tonal errors persisted" | — | — |

**Naming warning, verified repeatedly:** repo names systematically understate size — CSM-"1B" = 1.55B,
Orpheus-"3B" = 3.78B, Higgs-"3B" = 5.77B, Llasa-"3B" = 4.01B, VibeVoice-"1.5B" = 2.70B,
VibeVoice-"7B" = 9.34B.
## 2. Per-task state of the art in 2026, in plain language

### 2.1 ASR — coverage is solved on paper, quality is not, and the licence just changed

**What is solved.** Massively multilingual ASR is no longer a research question for *coverage*. Omnilingual
ASR (§3) claims 1,600+ languages and beats MMS, USM and Whisper on their own benchmarks with a single
model. Whisper large-v3 (1.54B, MIT, 99 languages) remains the default English/high-resource workhorse, and
`large-v3-turbo` (809M, decoder cut from 32 layers to 4) made it cheap. Below ~30 languages, ASR is a
commodity.

**What is not solved.**
1. **The long tail is measured, not fixed.** Omnilingual ASR's own low-resource bucket (<10 h training
   data, 546 languages) sits at CER 18.0, with 36% of those languages under CER 10. Coverage of a language
   means it appears in a list, not that it works.
2. **Which base model to fine-tune is now an answered question, and the answer is not "the biggest."**
   Makerere's benchmark (arXiv:2512.10968, **Deep Learning Indaba 2025 / PMLR 302**) fine-tunes Whisper,
   XLS-R, MMS and w2v-BERT 2.0 on **13 African languages at 1→400 h** and states: *"MMS and W2v-BERT are
   more data efficient in very low-resource regimes, XLS-R scales more effectively as additional data
   becomes available, and Whisper demonstrates advantages in mid-resource conditions."* That is a
   three-way regime split, and it is the single most practically useful ASR result for this lab.
   → `papers/Makerere-2025-Benchmarking-ASR-African-Languages.pdf`
3. **Whisper's African coverage is 10 of 99 languages** — ar, af, am, ha, ln, mg, sn, so, sw, yo. It has
   **no Igbo, isiZulu, isiXhosa, Wolof, Akan/Twi, Kinyarwanda, Luganda, Chichewa, Oromo, Tigrinya, Ewe,
   Bambara, Dholuo, Kikuyu**. Any pipeline built on Whisper is structurally blind to most of the continent.
4. **Hallucination and repetition are named, patched and unexplained.** See §4.8.
5. **Metrics mismeasure tonal languages.** arXiv:2602.04716 finds Yoruba WER 0.788 against a Feature Error
   Rate of 0.151 on the same system. Four of the lab's six languages are tonal.

**The licence change is the real 2026 event.** MMS is CC-BY-NC and SeamlessM4T is CC-BY-NC; **Omnilingual
ASR is Apache-2.0 and its corpus is CC BY 4.0**. w2v-BERT 2.0 (580M, MIT) is likewise commercially clean.
For the first time, a lab shipping a product can use the frontier multilingual speech stack rather than
working around it.

**Also worth knowing:** Kyutai and NVIDIA occupy the fast/streaming end; Voxtral and Phi-4-multimodal put
ASR inside an LLM; the `mms-1b-all` adapter set (1,125 language codes) remains the cheapest per-language
fine-tune. See the table in §1.

### 2.2 Speech-to-text translation (S2TT) — the cascade won, and the field says so in print

**What is solved.** X→English S2TT for the ~100 languages Seamless and Whisper cover is good and cheap.
SeamlessM4T v2 (2.3B, CC-BY-NC-4.0) remains the reference end-to-end system; Omnilingual ASR's S2TT variant
beats Whisper large-v2 on 74 of 81 FLEURS directions but still loses to SeamlessM4T v1 Large by ~0.5 BLEU.

**What is not solved, in the organisers' own words.** IWSLT 2025 and 2026 (§4) report that cascaded systems
beat end-to-end ones in the simultaneous, instruction-following-long, Indic and Catalan tracks; that
low-resource ST has hit a **data ceiling** ("we have perhaps reached a performance ceiling of sorts in the
current datasets"); and that it takes **>50 hours of high-quality translated speech** to get a decent
system. Irish–English topped out at 2.4 BLEU in 2026.

**For African languages specifically**, IWSLT 2026's new African track (§4.3) gives the first real numbers:
a fine-tuned SeamlessM4T upper bound at 17.6–21.1 spBLEU, and **OmniASR-LLM-1B → NLLB-200 as the published
cascaded baseline** at 11.0–17.3. Error propagation from ASR into MT is named as the reason the cascade
loses more on Igbo than on Hausa. This is the lab's own architecture appearing as a shared-task baseline.

### 2.3 Speech-to-speech translation (S2ST) — narrow, and that is the story

**What is solved.** Simultaneous, high-fidelity, voice-preserving S2ST works — for a handful of European
pairs. Hibiki (Labiausse et al., **ICML 2025**, arXiv:2502.03382) is a decoder-only multistream LM that
processes source and target speech synchronously and emits text and audio tokens jointly, French→English,
weights CC-BY-4.0 (`kyutai/hibiki-{1b,2b}-pytorch-bf16`). **Hibiki-Zero** (Labiausse et al., **ICML 2026**,
arXiv:2602.11072) removes the word-alignment requirement — sentence-level pairs plus GRPO reinforcement
learning for the latency/quality trade-off — covers fr/es/pt/de→en, is **3B under MIT** at
`kyutai/hibiki-zero-3b-pytorch-bf16`, runs in 8–12 GB, and reports **adaptation to a new language with
under 1,000 hours of speech**. That last number is the one to remember.
StreamSpeech (**ACL 2024**, arXiv:2406.03049) remains the multi-task simultaneous baseline.
→ `papers/Labiausse-2025-Hibiki-Simultaneous-S2ST.pdf`,
  `papers/Labiausse-2026-Hibiki-Zero-S2ST-Without-Aligned-Data.pdf`

**What is not solved.** Almost everything, for the lab's languages. SeamlessM4T v2 produces **speech output
in only 35 languages**; of the six, **only Swahili gets speech out**. Xhosa is *speech input only* — no text
or speech target at all. Igbo and Yoruba are speech-in → text-out. Twi/Akan and Efik are absent entirely.
CoVoST 2 and mExpresso contain no African languages. SeamlessExpressive is gated behind a proprietary
licence. **There is no open S2ST system that speaks Igbo, Yoruba, Xhosa, Twi or Efik.** IWSLT 2026's African
S2S sub-track scores generated *English* speech, so even the new benchmark evaluates only the into-English
direction; English→African is released but explicitly "exploratory" and unranked.

The gap between "Hibiki-Zero needs <1,000 h to add a language" and "no open model speaks Igbo" is the
opening in this task.

### 2.4 Speech LLMs / omni models — capable in demos, thin where it counts

**What is solved.** Open omni models now do ASR, translation, audio QA and speech-out in one stack, on one
GPU. See §1 for the roster.

**What is not solved**, per IWSLT 2026's instruction-following track: "long-form processing is still a major
limitation of current models"; "summarization remains the most challenging task"; and on a surprise quality-
estimation task, "most current systems struggle to generalize and, notably, to strictly adhere to the
instruction, indicating a big room for improvement in the instruction following abilities of SpeechLLMs."
The independent *Hearing to Translate* study (arXiv:2512.16378, **preprint**; 6 SpeechLLMs vs 16
alternatives, 13 pairs, 9 acoustic conditions) concludes "cascaded systems remain the most reliable
solution overall."

**And for African languages, the omni models are simply not in the race.** Their language lists are
Chinese/English/European. The open stack that actually serves African languages in 2026 is **Omnilingual
ASR (Apache-2.0) for recognition, w2v-BERT 2.0 (MIT) as the fine-tuning base, SeamlessM4T v2 (CC-BY-NC) if
speech output is required** — and none of the omni models.

### 2.5 TTS — quality is solved in English; coverage is a licence problem, and tone is unsolved

**What is solved.** Zero-shot voice cloning and human-parity naturalness in English/Chinese are done, at
small scale: Kokoro is 82M under Apache-2.0, CosyVoice 2/3 are 0.5B under Apache-2.0, Chatterbox is 0.5B
under MIT. **The 80 GB constraint is not binding anywhere in TTS** — the largest open model in this
landscape is ~9.3B.

**What is not solved.**
1. **Coverage is not quality.** OpenBibleTTS (arXiv:2606.09553, **preprint**; Guzmán, Alabi, Klakow,
   Adelani et al., McGill/Mila/Saarland) is the paper that demonstrates this. **3,469.01 h, 1,121,956
   utterances, 37 underrepresented languages, 19 African, CC-BY-SA-4.0**, with per-language models released
   in three architectures (EveryVoice 18.2M, VITS 36.3M, F5-TTS 335.8M). Its finding: "no single system
   dominates across languages and metrics: Gemini-TTS achieves the highest listener ratings on most
   evaluated languages, but monolingual EveryVoice models trained on OpenBibleTTS remain strongest for
   intelligibility." Mean WER 16.95% for the 18.2M EveryVoice models — beating far larger pretrained
   multilingual systems, including OmniVoice, which claims 600+ languages and publishes no per-language
   quality numbers. **A 18M-parameter monolingual model beats a 600-language one on the language in
   question.** → `papers/OpenBibleTTS-2026-African-Language-TTS.pdf`
2. **Tone is the residual failure, and it is exactly the lab's problem.** Both 2026 African TTS papers
   report it. The Efik paper (arXiv:2607.04515; **venue unverified** — the PDF's front matter claims Interspeech 2026, but Semantic Scholar records no venue, so treat it as a preprint) built the first end-to-end Efik TTS
   from **2,632 utterances / 3 hours, single speaker**, compared VITS, MMS-TTS, SpeechT5 and Orpheus-TTS,
   found MMS-TTS best (MOS 3.80 ± 0.63, most stable long-form) — and reports that "tonal errors persisted."
   → `papers/Efik-TTS-2026-Digital-Preservation.pdf`
3. **Zero-shot cross-lingual cloning transfers timbre, not phonology.** The recurring blocker is not speaker
   similarity but that most systems **require a transcript of the reference audio**, which you do not have
   for an unseen language, and that duration/prosody models are language-conditioned. Cross-Lingual F5-TTS
   (arXiv:2509.14579, **preprint**) attacks exactly this with forced alignment at training time plus
   multi-level speaking-rate predictors "particularly for unseen languages".
   → `papers/CrossLingual-F5-TTS-2025-Unseen-Language-Cloning.pdf`
4. **TTS metrics do not survive contact with new languages.** TTSDS2 (arXiv:2506.19441, **preprint**;
   Minixhofer, Klejch, Bell, Edinburgh) collects 11,000+ subjective ratings over 14 languages and reports
   mean Spearman ρ against human MOS: **TTSDS2 0.67, speaker-similarity (X-Vector) 0.60, DNSMOS ~0.36,
   UTMOS ~0.26, WER ~−0.23** — it is "the only one out of 16 compared metrics to correlate above 0.50 for
   every domain." **None of its 14 languages is African**, so UTMOS/UTMOSv2 numbers on Swahili or Yoruba
   are uncalibrated. OpenBibleTTS's own design is the safer pattern: **ASR-based WER as the primary metric
   (transcribed with Omnilingual ASR) plus native-speaker MOS**, UTMOSv2 secondary — and even then WER and
   MOS correlate in only 7 of 10 languages. → `papers/Minixhofer-2026-TTSDS2-Benchmark.pdf`
5. **The licence trap.** MMS-TTS is CC-BY-NC and its coverage is patchier than "1,100 languages" implies:
   verified against Meta's own coverage table, **Igbo, Xhosa, Zulu, Wolof, Lingala are ASR-only (no TTS)**
   and **Efik is LID-only**. XTTS-v2 is frozen at 2023-12 under a non-commercial licence with Coqui defunct
   (maintained fork `idiap/coqui-ai-TTS`, MPL-2.0 code, CPML weights unchanged). Meanwhile the biggest new
   isiXhosa corpus forbids TTS use outright (§5.2).

**Net for the lab's six languages:** Swahili, Yoruba and Twi have MMS-TTS (NC) plus OpenBibleTTS (CC-BY-SA,
commercially usable with share-alike) — Swahili additionally Chatterbox (MIT), VoxCPM2 and MOSS-TTS v1.5
(Apache-2.0). Igbo has no MMS-TTS but has OpenBibleTTS (94.46 h). **Xhosa has neither**, only non-commercial
options, and its best corpus bans TTS. **Efik has no model anywhere** except the 3-hour Interspeech 2026
system.

### 2.6 Evaluation — the field promoted it to a shared task, then failed it

IWSLT 2026 renamed its own findings paper *"Speech Translation and Metrics in 2026"* and created a Metrics
track. The result: "current automatic metrics for speech translation quality estimation still fall
considerably short of human judgment at the segment level, despite performing well at ranking systems"
(best 34.5 vs human agreement 45.8/46.6). Text-only metrics matched speech-aware ones, which the organisers
read as the *test data* failing to expose prosody at all. On call-centre audio, text metrics could not even
rank the human reference above the system. See §4.1.

For African languages the metric stack is **SSA-COMET** (EMNLP 2025) with spBLEU and chrF++; for S2ST,
**BLASER 2.0** and, in IWSLT 2026's African track, **CER measured by transcribing with OmniASR-LLM-1B**.
## 3. Omnilingual ASR in detail

**Source read:** `papers/Omnilingual-ASR-2025-1600-Languages.pdf` — Omnilingual ASR team et al.,
*Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages*, arXiv:2511.09690v1,
dated 12 Nov 2025, 53 pp. **arXiv-only preprint** (no venue printed on the PDF as of this reading).
Sections cited below are that PDF's own numbering. Model/dataset metadata read from the HF API and
the GitHub README, both fetched 2026-09-06.

### 3.1 What was actually released

Thirteen repos under `facebook/` on the Hub, all **Apache-2.0** (read from the HF API `cardData.license`
field of `facebook/omniASR-LLM-7B`; created 2025-11-27):

| Family | Sizes | HF ids |
|---|---|---|
| SSL encoders (wav2vec 2.0) | 300M / 1B / 3B / 7B | `facebook/omniASR-W2V-{300M,1B,3B,7B}` |
| CTC ASR (encoder + linear layer) | 300M / 1B / 3B / 7B | `facebook/omniASR-CTC-{300M,1B,3B,7B}` |
| LLM-ASR (encoder + Transformer decoder) | 300M / 1B / 3B / 7B | `facebook/omniASR-LLM-{300M,1B,3B,7B}` |
| Zero-shot LLM-ASR | 7B | `facebook/omniASR-LLM-7B-ZS` |

Plus the dataset `facebook/omnilingual-asr-corpus` — **CC-BY-4.0**, **348 languages** in its card metadata
(created 2025-10-30). §1 of the paper describes it as "an average of 10 hours of transcribed speech per
language; for many languages, this represents the first ASR corpus ever built."

Exact encoder configs (paper Table 4): 317M (24L/1024d), 965M (48L/1280d), 3046M (60L/2048d),
6488M (128L/2048d) — all 16 heads. The GitHub README's own inference table gives BF16 VRAM on an A100,
batch 1, 30 s audio: CTC 300M ≈ 2 GiB → CTC 7B ≈ 15 GiB; LLM 300M ≈ 5 GiB → LLM 7B ≈ 17 GiB.
**Everything, including the 7B, fits comfortably on one 80 GB card**, and the 300M runs on a laptop.

### 3.2 What is genuinely new

1. **Extensibility as the design goal, not coverage.** §4.3: instead of one speech–text pair per training
   step, the decoder is shown *N+1* pairs from the same language, the first *N* prepended to the prompt
   inside `<c>…</c>` / `<cs>…</cs>` special tokens, and trained to predict the last one's transcript with
   ordinary next-token prediction. This is in-context learning transplanted onto a speech encoder–decoder,
   and it is what lets a community add an unseen language from a handful of paired examples rather than
   from a fine-tuning run.
2. **Speech SSL scaled past the 2B ceiling.** §4.1.2 states plainly that the largest previously reported
   speech SSL models — Google USM and Meta XLS-R — were both ≈2B, and that whether 2B is the effective
   limit "remains an open question." They take wav2vec 2.0 to 7B on 4.3M hours across 1,600+ languages
   and attribute their win over USM largely to "encoder size scaling" (§5.2.2) — despite pre-training on
   **less than half** USM's unlabelled audio (4.3M vs 12M h) and with no multi-stage pipeline.
3. **Optional language conditioning** (§4.5): a `<language>` token plus a language-and-script embedding,
   dropped at random during training with probability *p* so the model works with or without it. Motivated
   by script ambiguity (their example is Urdu) and confusion between close relatives.
4. **The training corpus** (Table 3): 120,710 h / 1,690 languages, assembled from open datasets (15,000 h),
   internal+licensed (150,100 h — note this exceeds the total, so Table 3's rows are not a clean partition
   of the 120,710 h figure; I did not resolve this discrepancy), **African Next Voices (7,200 h, 13 langs)**,
   the Open Multilingual Speech Fund (1,940 h, 177 langs), **Lanfrica/NaijaVoices (110 h, 11 langs)**, and
   their own commissioned Omnilingual ASR Corpus (3,350 h, 348 langs).

### 3.3 The numbers, stated with their comparison

- **vs MMS** (Table 7, CER unless noted): 7B-LLM-ASR beats MMS on all three sets — MMS-Lab-1143 1.9 vs 2.1;
  FLEURS-102 6.2 (6.1 with LM) vs 6.3–6.4; MLS-8 8.0 WER vs 8.7–9.0. MMS's numbers are with n-gram LM
  decoding and per-language adapters; Omnilingual's are a single model.
- **vs USM** (Table 6, FLEURS-102 CER): 6.2 (6.1 + LM) vs USM-M's 6.5.
- **vs Whisper for S2TT** (Table 14, BLEU, X→English): OmniASR-LLM-7B gets 37.1 on CoVoST2-21 and 23.5 on
  FLEURS-81, against Whisper large-v2's 29.1 / 17.9; it wins on 74 of 81 FLEURS directions. But it **loses
  to SeamlessM4T v1 Large** (34.1 / 24.0 / 21.4) on FLEURS-81 by 0.5 BLEU and FLEURS-101 by 0.6 — the paper
  notes Seamless initialised its decoder from NLLB while theirs was trained from scratch.
- **Resource buckets** (Tables 8–9, 7B-LLM, no LM fusion): high-resource (>50 h, 249 langs) CER 3.13 ± 0.7;
  mid (10–50 h, 881 langs) 3.0 ± 0.3; **low (<10 h, 546 langs) CER 18.0 ± 1.2**, with only 195/546 = 36%
  of low-resource languages reaching CER ≤ 10 (7B-CTC: 3.7 / 4.4 / 18.6, and 184/546 = 34%).
  *Bookkeeping wart:* §5.3.1's prose says the buckets are "249, 881, and 549 languages" while Tables 8–10
  head the low column **546**. Small, but it is the kind of thing to check before quoting.
- **Zero-shot on 32 held-out languages** (Table 11): CTC baseline 26.3 CER, plain LLM-ASR 31.0, zero-shot
  LLM-ASR with 10 context examples **14.4**. Context selection matters: SONAR-embedding retrieval beats
  random by up to 11.2% relative (Table 12), and the `same_ex` oracle (feeding the answer as context)
  drops CER to 11.6/16.4/9.8 — so the decoder demonstrably reads the context.

### 3.4 African coverage — for this lab's six languages specifically

Appendix A lists every supported language as `code_Script` with a resource bucket (H/M/L). Grepped from
the PDF:

| Lab language | ISO 639-3 | In Omnilingual ASR? | Bucket |
|---|---|---|---|
| Swahili | swh | **yes** (`swh_Latn`, "Swahili (individual language)") | H |
| Yoruba | yor | **yes** (`yor_Latn`) | H |
| Igbo | ibo | **yes** (`ibo_Latn`) | H |
| Xhosa | xho | **yes** (`xho_Latn`) | M |
| Twi | twi | **no** — but Akan (`aka_Latn`) is present | M |
| **Efik** | efi | **no** | — |

Efik is absent. Its closest well-known relatives in the Cross River group *are* present but in the worst
bucket: **Ibibio (`ibb_Latn`, L)** and **Anaang (L)**. Obolo (`ann_Latn`, M) is also there. So Efik is
exactly the case the zero-shot model is advertised for, and the lab has a natural experiment available:
does `omniASR-LLM-7B-ZS` with a few Efik examples beat fine-tuning `omniASR-W2V-*` on the same data?

At the family level (Table 10, 7B-LLM, no LM), the two big African groupings do middling-to-well:
**Atlantic-Congo ("Atlacong"): 389 languages, avg CER 9.3, 72% under CER 10** — the largest grouping in
the paper and the one containing Swahili, Yoruba, Igbo, Xhosa, Akan. **Nilo-Saharan: 56 langs, CER 4.4,
89%.** The single worst grouping in the whole table is **Afroasiatic: 92 langs, CER 11.8, 66%** — the only
group that fails to reach CER ≤ 10 on average. Overall: 7.1 CER over 1,570 languages, 78% at CER ≤ 10.

### 3.5 What it does **not** do

- **No speech output, and no X→Y translation.** It is ASR plus X→English S2TT (§5.6) and nothing else.
  For the lab's actual product — English/other → African-language *speech* — Omnilingual ASR contributes
  the input half only. It does not replace NLLB or the TTS voices.
- **The S2TT model is not one of the released checkpoints.** §5.6 describes S2TT models built by putting a
  1.2B decoder on OmniASR-W2V-{1B,3B,7B}, but no S2TT checkpoint appears among the 13 `facebook/omniASR-*`
  repos. That capability is reported, not shipped. [needs confirmation — I checked the HF `facebook/`
  listing, not every branch of the GitHub repo]
- **It does not solve the long tail; it measures it.** CER 18.0 on the <10 h bucket, and only 36% of those
  languages under CER 10, is the paper's own headline caveat. §1: "zero-shot performance cannot yet match
  that of fully trained systems."
- **Zero-shot costs something on seen languages.** §5.4: "zero-shot models somewhat degrade accuracy on
  some datasets of seen languages" — which is why they ship the ZS model separately. The exceptions are
  FLEURS-102 and CV22, where context examples *reduce script/language confusion*.
- **Diversity of context does not help; similarity does.** §5.5: the max-bigram-diversity selection barely
  beats random, and mean-pooled wav2vec 2.0 embeddings do not beat random at all. Only semantic (SONAR)
  retrieval helps. Their reading: "the model may struggle to effectively learn from context examples that
  are not directly related to the target."
- **CER, not WER, is the headline metric** across 1,600 languages. Reasonable given orthographic variety,
  but it is a softer number than WER, and cross-paper comparison needs care.
- **Domain skew is acknowledged.** §5.7.1: the (0.0, 0.0) uniform-upsampling setting wins the language-based
  protocols precisely because those are "largely determined by the broad language coverage of MMS-lab" —
  i.e. read-aloud religious text. They deliberately chose (0.5, 0.25) instead to protect robustness on
  Babel and CommonVoice. This is an honest admission that the 1,600-language claim rests heavily on a
  narrow audio domain.
- **No interpretability, no analysis of failure modes.** There is no repetition/hallucination analysis
  anywhere in the 53 pages — relevant to the superweights arm, which is looking for whether speech models
  build the same "constant" and loop the same way.
## 4. What the IWSLT organisers themselves name as open

Two sources, both **ACL Anthology only — neither has an arXiv preprint, so do not cite an arXiv id for
either**:

- Abdulmumin et al. (51 authors), *Findings of the IWSLT 2025 Evaluation Campaign*, **IWSLT 2025**
  (co-located with ACL 2025, Vienna, 31 Jul – 1 Aug 2025), pp. 412–481, `2025.iwslt-1.44`.
  → `papers/IWSLT-2025-Findings-Evaluation-Campaign.pdf` (70 pp.)
- *Speech Translation and Metrics in 2026: Findings of the IWSLT Campaign*, **IWSLT 2026** (co-located with
  ACL 2026, San Diego, 3–4 Jul 2026), pp. 336–422, `2026.iwslt-1.39`.
  → `papers/IWSLT-2026-Findings-Speech-Translation-and-Metrics.pdf` (87 pp.)

**IWSLT 2026 has already happened.** The next edition is the 24th, Kyoto, 17–22 Aug 2027, co-located with
ACL; tasks are posted in January and are not yet up.

**Caution on both papers: their abstracts disagree with their own introductions.** The 2025 abstract lists
a dubbing track and a speech-to-speech track; neither ran. Cite the intros.

### 4.1 Evaluation is the problem they promoted to a track of its own

The 2026 edition's title is the finding. It created a first-edition **Speech Translation Metrics track** for
reference-free quality estimation, and the track intro says QE for ST "remains underexplored," that
"evaluation of speech-to-text translation still relies heavily on text-based evaluation approaches, which
often overlook speech-specific phenomena … and make unrealistic assumptions about access to gold
segmentation and error-free source transcriptions during evaluation," and that "comparable speech-native
resources remain limited."

The measured result is worse than the framing: **"current automatic metrics for speech translation quality
estimation still fall considerably short of human judgment at the segment level, despite performing well at
ranking systems."** Best segment-level submission scored 34.5 against human inter-annotator agreement of
45.8 (en–de) / 46.6 (en–zh). Two sharper sub-findings:

- **Text-only metrics did as well as speech-aware ones**, which the organisers read as the test data failing
  to expose prosody, speaking style or speaker gender at all.
- **Replacing gold source transcripts with Whisper-large-v3 transcripts moved segment-level metric
  performance by ≤1.3 points** — so on English source speech, ASR quality is not the bottleneck. That is a
  useful negative: it says the cascade's weak link is elsewhere.
- Metric quality **collapses by domain**: on call-centre audio the best segment-level correlation is ~20.9
  (human agreement 42.4), and text metrics there fail even to rank the human reference above the system.

IWSLT 2025 found human evaluation itself compromised by segmentation (App. A.1): informed test-subset
selection lost to random selection, because force-aligned segments make system outputs non-comparable.

The **2026 call for tasks** names the same thing among its open topics: "new applications scenarios (e.g.
meetings, subtitling, dubbing), specific aspects (e.g. names, accents), different styles, multilinguality,
discourse and summarization, multimodal and multi-party speech translation, **automatic evaluation metrics
for speech translation**." Corroborating from the text side, WMT 2025's two shared-task titles are the
argument: *"Linguistic diversity is challenging and references still help"* and *"Time to stop evaluating on
easy test sets."*

### 4.2 Low-resource ST has hit a data ceiling, and 2026 confirmed it

IWSLT 2025 §7.4, verbatim: performance "stagnated, remaining in exactly the same levels (if not worse) for
Bemba-English, Bhojpuri-Hindi, Irish-English, and the two Arabic dialects … This might suggest that we have
perhaps reached a performance ceiling of sorts in the current datasets under the current data-scarce
conditions … this 'ceiling' performance nevertheless still lags substantially behind the translation quality
we observe for high-resource pairs." Their working threshold for decent ST: **>50 hours of high-quality
translated speech**. Almost all submissions were *unconstrained*, which they call "a clear indication that
pre-trained multilingual systems seem to be the best option."

2026 did not break the ceiling: **Irish–English topped out at 2.4 BLEU** and **Mapudungun–Spanish at 0.82
BLEU**, with the organisers writing "all submissions struggle to produce meaningful outputs."

The 2025 low-resource pairs were apc-eng, aeb-eng, **bem-eng**, **fon-fra**, gle-eng, bho-hin, est-eng,
mlt-eng, mar-hin, que-spa. The 2026 pairs added **Hausa–English, Igbo–English and Yorùbá–English** alongside
Irish, Bhojpuri, Mapudungun, Bemba, Quechua, Central Kurdish and Catalan.

### 4.3 The 2026 African/Celtic S2S track — read this one closely, it is this lab's benchmark

Track VII of the 2026 findings is a purpose-built **Hausa / Igbo / Yorùbá → English** speech-translation
track, with a speech-to-text and a speech-to-speech sub-track. Details read from the PDF:

- **NaijaS2ST**, newly recorded: ~75 h total, ~60–70 h train per language pair, 5,000 / 500 / 500 unique
  sentences per split, up to 70 speakers per language, held-out speakers in dev/test, each training sentence
  recorded by three different speakers, 48 kHz wav. Source text from **MAFAND**, **NTREX** and **SSA-MT**;
  test sets curated from Voice of America news, deliberately balanced between European-context and
  African-context articles. English recordings capture Northern and Southern **Nigerian English accents**.
  Reverse directions (English → Hausa/Igbo/Yorùbá) are released but "exploratory," not officially ranked.
- **Metrics:** S2TT is ranked by **SSA-COMET**, with spBLEU and chrF++ for analysis. S2S is scored by **CER
  after transcribing the generated English speech with OmniASR-LLM-1B** — i.e. Omnilingual ASR has already
  been adopted as evaluation infrastructure, not just as a system.
- **Baselines and the one submission (Table 12, XX→English):**

  | System | Hausa spBLEU | Igbo spBLEU | Yorùbá spBLEU |
  |---|---|---|---|
  | SeamlessM4T Mono FT (supervised upper bound, per-pair fine-tune) | **18.6** | **17.6** | **21.1** |
  | Cascaded: **OmniASR-LLM-1B → NLLB-200** | 17.3 | 11.0 | 17.0 |
  | AURA-ST (KIT-RBG-AI, the only submission) | 5.2 | 4.6 | **19.5** |

  Read the middle row: **the lab's own architecture — a big multilingual ASR feeding NLLB-200 — is the
  published cascaded baseline for this track**, and it is within 1.3 spBLEU of the fine-tuned Seamless upper
  bound on Hausa while losing 6.6 on Igbo. The organisers' reading: "recognition errors propagate through
  the translation pipeline and become increasingly difficult to recover from in lower-resource settings."
  Hausa is not supported by SeamlessM4T at all, so each Seamless baseline was fine-tuned per language.
- AURA-ST is worth knowing as a recipe: frozen w2v-BERT 2.0 + frozen ResNet34 dual-stream encoders → conv
  subsampler → **prefix prompt into a frozen Gemma-4-E2B**, no cross-attention, **LoRA rank 16 on the MLP
  layers only (gate/up/down proj)** because "standard attention-based LoRA injection was ineffective due to
  architectural constraints in Gemma's custom projection modules." It beats the cascade on Yorùbá by +2.5
  spBLEU / +6.5 SSA-COMET with no per-language supervised fine-tuning.
- The track's own conclusion: "the continuing difficulty of speech translation for African languages, as the
  nuances in tone and diacritic … are much more difficult to detect than in text … substantial performance
  disparities remain across languages," and parameter-efficient speech-to-LLM adaptation "is a promising
  direction … but additional advances in multilingual representation learning and low-resource adaptation
  are needed."

### 4.4 Cascaded still beats end-to-end, repeatedly and in writing

- 2026 simultaneous track: "with the exception of a single end-to-end submission, all participating teams
  adopted cascaded approaches."
- 2026 instruction-following LONG track: "cascaded systems [are robust] to long-form input … In contrast,
  the end-to-end systems degrade substantially."
- 2025 Indic track: "cascaded architectures generally outperform[ed] end-to-end approaches."
- 2026 Catalan–English participants: "cascaded systems continue to outperform end-to-end speech translation,
  with performance primarily being constrained by ASR quality over MT."
- Independent corroboration: *Hearing to Translate* (arXiv:2512.16378, **preprint**, 6 SpeechLLMs vs 16
  alternatives over 13 pairs and 9 acoustic conditions) concludes "cascaded systems remain the most reliable
  solution overall." → `papers/Hearing-to-Translate-2025-Speech-Modality-Integration-LLMs.pdf`

This matters for a lab that ships a cascade: the field's own evidence says the cascade is not the legacy
option, it is the current default, and the interesting question is where its error propagation lives.

### 4.5 Speech-LLM capability is thinner than the marketing

2026 instruction-following conclusions: "long-form processing is still a major limitation of current
models"; "summarization remains the most challenging task"; and on the surprise QE task, "most current
systems struggle to generalize and, notably, to strictly adhere to the instruction, indicating a big room
for improvement in the instruction following abilities of SpeechLLMs." In 2025 the *constrained*-condition
systems won the IF track.

### 4.6 Latency and robustness

"The quality-latency tradeoff remains generally observable across systems, [but] higher latency does not
always yield proportional translation quality improvements," and computation-aware latency "metrics are
generally not available for all submissions, therefore we cannot draw firm conclusions." On out-of-domain
YouTube-sourced YODAS audio, ~1.0 s of extra latency across all pairs: "the performance degradation and
latency spikes observed on the YouTube-sourced YODAS dataset underscore the ongoing challenge of handling
out-of-domain conditions."

### 4.7 Compression — the track that is closest to this repo's other arm

- **2025: one participant.** 4-bit + QLoRA + layer pruning got to 5.0B/4.1B params and 9.7/8.8 GB —
  "notable but insufficient to meet the most relaxed size requirements defined by Bin1 (i.e., a maximum of
  4 GB)."
- **2026: three submissions**, and the conclusion is quotable: "aggressive quantization can reduce model
  size by 4 times while retaining performance close to, and in some cases exceeding, that of the
  full-precision models. **Quantization has emerged as the to-go approach for all submission, while pruning
  has not been equally investigated.** … compressing for very low-resource settings, where a 4GB system is
  still too big, is a challenging problem yet to be addressed for speech translation."
- And an uncontrolled observation the organisers flag themselves: en–zh 4-bit NF4 *beat* full precision
  (hypothesised as regularisation) while en–de degraded consistently, which "might uncover another source of
  exacerbation of the performance gap between high-resource and lower-resource languages" — but "this
  finding has to be confirmed with a broader investigation." This is an open, cheap, named question sitting
  in a shared task, and it is adjacent to `compression/experiments/replication-uneven-ptq/` in this repo.

### 4.8 Hallucination and repetition loops — named, patched, barely explained

- Koenecke, Choi, Mei, Schellmann, Sloane, *Careless Whisper: Speech-to-Text Hallucination Harms*, **FAccT
  2024** (arXiv:2402.08021): ~1% of transcriptions contain entirely fabricated phrases, **38% of
  hallucinations carry explicit harms**, disproportionately affecting speakers with aphasia and correlated
  with long non-vocal durations. → `papers/Koenecke-2024-Careless-Whisper.pdf`
- OpenAI's own `whisper-large-v3-turbo` card says the model "is prone to generating repetitive texts, which
  can be mitigated to some degree by beam search and temperature scheduling but not perfectly." A decoding
  patch, in the vendor's words, with no account of the cause.
- **Calm-Whisper** (arXiv:2505.12969, **preprint**) is the closest thing to a mechanistic account found:
  **3 of 20 decoder self-attention heads account for ~75% of non-speech hallucinations**, and fine-tuning
  only those cuts them ~80%. → `papers/Wang-2025-Calm-Whisper-Crazy-Heads.pdf`
- IWSLT participants treat it as engineering debt, not science: the 2026 subtitling track's FBK system
  post-processes out "consecutive n-gram repetitions — **a known hallucination pattern of both ASR and MT
  models**"; another team applies length-based filtering "to suppress hallucinations."

**This is the seam for the superweights arm.** The field names ASR repetition loops as a known failure,
patches them at decode time, and has exactly one paper localising them to specific heads. Whether speech
encoders/decoders build a massive-activation "constant" and whether the loop is the same phenomenon the
text-side work found is, as far as this survey went, unasked.
## 5. Datasets for the lab's six languages

Scope: Xhosa (xho), Igbo (ibo), Efik (efi), Swahili (swh), Twi/Akan (twi/aka), Yoruba (yor).
**Headline: five of the six are reasonably served; Efik is absent from essentially every open corpus.
And the three largest recent corpora each carry a licence restriction that bites a shipping product.**

### 5.1 The one-screen answer

| | Xhosa | Igbo | Efik | Swahili | Twi/Akan | Yoruba |
|---|---|---|---|---|---|---|
| Common Voice 26.0 validated h | 0.02 | 2.80 | — | **392.2** | 0.36 (`tw`) | 5.89 |
| FLEURS (n-way parallel, S2TT-capable) | ✓ `xh_za` | ✓ `ig_ng` | ✗ | ✓ `sw_ke` | ✗ | ✓ `yo_ng` |
| MMS ASR adapter (`mms-1b-all`) | ✓ | ✓ | ✗ | ✓ | ✓ as `aka` | ✓ |
| MMS-TTS | ✗ | ✗ | ✗ | ✓ | ✓ as `aka` | ✓ |
| Omnilingual ASR (§3) | ✓ (M) | ✓ (H) | ✗ (Ibibio/Anaang L) | ✓ (H) | ✓ as `aka` (M) | ✓ (H) |
| SeamlessM4T v2 | speech **in only** | Sp,Tx in → Tx out | ✗ | full S2ST both ways | ✗ | Sp,Tx in → Tx out |
| Largest open speech corpus | ANV-SA 500 h (**no-TTS clause**) | NaijaVoices ~600 h (NC) | ~1 h sample | AfriVoice ~1,000 h (BY 4.0) | GhanaNLP 500 h (NC, pseudo-labels) | NaijaVoices ~600 h (NC) |
| IWSLT S2TT benchmark | ✗ | ✓ 2026 | ✗ | ✗ | ✗ | ✓ 2026 |
| FLORES+ text (CC BY-SA 4.0) | ✓ | ✓ | **✗** | ✓ | ✓ (Akuapem+Asante) | ✓ |

### 5.2 The corpora, with licences stated

**Nigerian languages**
- **NaijaVoices** — arXiv:2505.20564, **Interspeech 2025**. 1,800 h, **~600 h each of Igbo, Hausa, Yoruba**;
  5,000+ speakers; ~1.92M instances. HF `naijavoices/naijavoices-dataset` (446 GB; compressed 84 GB), gated
  behind registration. **CC BY-NC-SA 4.0 — non-commercial.** Reported relative WER improvements over
  baselines: Whisper 75.86%, MMS 52.06%, XLSR 42.33%.
  → `papers/NaijaVoices-2025-Igbo-Hausa-Yoruba-Speech-Dataset.pdf`
- Also feeds Omnilingual ASR's training mix as "Lanfrica/NaijaVoices, 110 h, 11 langs" (§3.2) — a small
  slice of it.

**Xhosa**
- **Swivuriso / South African African Next Voices** — arXiv:2512.02201 (**preprint**), DOI
  10.5281/zenodo.17776289. HF `dsfsi-anv/za-african-next-voices`, 459k rows, 3,000+ h over 7 SA languages,
  **isiXhosa 500 h**, ASR-labelled, domains agriculture/health/finance/sports/transport/culture/society.
  **CC BY 4.0 *but the card explicitly prohibits use for TTS, voice cloning, or voice synthesis*.** For a
  lab with an African TTS project this is the single most important licence flag in this document: the best
  Xhosa corpus in existence is off-limits for exactly the use the lab wants.
  → `papers/Swivuriso-2025-African-Next-Voices-South-Africa.pdf`
- **NCHLT** (SADiLaR): ~56 h isiXhosa main corpus, **CC BY 3.0**; HF mirror `danielshaps/nchlt_speech_xho`.
  The auxiliary multilingual set `dsfsi-anv/multilingual-nchlt-dataset` is ~1,420 h over 11 languages
  (CC BY 4.0) but its isiXhosa slice is small (7,173 utts).
- **OpenSLR SLR32** (van Niekerk et al., **Interspeech 2017**): multi-speaker TTS for Afrikaans/Sesotho/
  Setswana/**isiXhosa**, **CC BY-SA 4.0** — commercially clean, and the realistic Xhosa TTS starting point
  given the ANV clause.

**Swahili**
- **Digital Umuganda AfriVoice Swahili** — HF `DigitalUmuganda/Afrivoice_Swahili`, **CC BY 4.0**. Reported
  ~3,218 h (3,097 h transcribed); an independent ingest counted ~1,000 h / 180,527 clips after a 1–30 s
  filter, locale `sw_KE`, evenly split across agriculture/education/finance/government/health.
  **The largest commercially clean open Swahili corpus found.** [hours not reconciled — cite the range]
- **Common Voice `sw`: 392.2 h validated** at CV 26.0 (2026-06-12), CC0 — by far the best CV coverage of the
  six. Note CV 17.0 (2024) had *more* (399.5 h); Swahili growth has stalled.
- **Kencorpus / KenSpeech** — arXiv:2208.12081, **CC BY 4.0**: Swahili 27 h31m (26h32 read + 59m
  spontaneous), 5,816 clips, 26 speakers.
- **ALFFA (OpenSLR SLR25)**: Swahili broadcast news, **MIT**. Commonly quoted as ~10 h; **hours unverified**.
- **BABEL Swahili** LDC2017S05 (~350 h) and **BABEL Igbo** LDC2019S16 (~207 h) — LDC, paid.

**Twi / Akan**
- **BibleTTS** — arXiv:2207.03546, **Interspeech 2022**, OpenSLR SLR129, **CC BY-SA 4.0 (commercially
  friendly)**. Studio 48 kHz, single speaker per language: **Asante Twi 74.9 h, Akuapem Twi 67.1 h,
  Yoruba 33.3 h** (plus Ewe, Hausa, Lingala). Of the six it covers only Twi and Yoruba — but it is the
  cleanest-licensed high-fidelity African TTS corpus that exists.
  → `papers/Meyer-2022-BibleTTS.pdf`
- **GhanaNLP** `ghanaopenai/twi-health-asr-gemini-500hrs`: **500 h health-domain Twi ASR, CC BY-NC 4.0**,
  but the transcripts are **Gemini pseudo-labels over YouTube**, not gold. Useful for pre-training, not for
  evaluation.
- **WAXAL** (Google + Gates) — arXiv:2602.02734 (**preprint**), `google/WaxalNLP`. ~1,250 h transcribed ASR
  over 19 languages + 180 h+ single-speaker TTS over 17. Per-provider licences (**CC BY 4.0** for Univ. of
  Ghana; **CC BY-SA 4.0** for Makerere / Digital Umuganda / others). Configs touching the six: `aka_asr`
  (~69.5 h), `twi_tts` (~3.0 h), `fat_tts` (~2.9 h), `ibo_tts`, `yor_tts` (~8.1 h), `swa_tts` (~4.2 h).
  **No Xhosa, no Efik.** → `papers/WAXAL-2026-African-Language-Speech-Corpus.pdf`
- **CMU Wilderness** (Black, **Interspeech 2019**): 699 languages, but of the six only Akan/Akuapem
  (`AK1BSG`, ~14.5 h). Alignments released; audio must be pulled from bible.is yourself.

**Accented-English corpora (not indigenous-language speech — read the caveat)**
- **AfriSpeech-200** — Olatunji et al., **TACL** (arXiv:2310.00274), **CC BY-NC-SA 4.0**, HF `intronhealth/afrispeech-200`. 200.9 h,
  67,577 clips, 2,463 speakers, 120 accents, 13 countries. **This is African-*accented English*.** All six
  appear only as *accent labels*: Yoruba ≈44.9 h, Igbo ≈25.8 h, Swahili ≈15.5 h, Twi ≈4 h, **Efik ≈0.7 h**.
  Valuable for the Nigerian-English half of a dubbing pipeline; useless as Igbo or Yoruba speech.
  → `papers/Olatunji-2023-AfriSpeech-200.pdf`
- **AfriSpeech-Dialog** — arXiv:2502.03945, **NAACL 2025**. 6 h of two-speaker conversations, 11 accents,
  98 speakers; ASR + diarization + summarization. Same accented-English caveat.
  → `papers/AfriSpeech-Dialog-2025-Benchmark.pdf`
- **AfriVox-Transcribe** (HF `intronhealth/afrivox-transcribe`, **gated**) is the aggregate that *is*
  indigenous-language: 79.5 h over 20 African languages from 9 sources — Xhosa 6.68 h, Yoruba 5.77 h,
  Swahili 5.45 h, Igbo 2.88 h, Akan 1.47 h, Twi 0.75 h, **Efik absent**. Mixed inherited licences, so
  effectively NC.

### 5.3 Efik — the actual gap, and the actual opportunity

Efik is absent from FLEURS, **FLORES+**, NTREX-128, MAFAND-MT, MMS ASR adapters, MMS-TTS, Common Voice,
BABEL, CMU Wilderness, BibleTTS, WAXAL, African Next Voices, SeamlessM4T, Omnilingual ASR, and IWSLT.
What exists:

- `scubavoice/ibibio-efik-speech-corpus-sample` (2026-06): a **1 h sample** of Ibibio + Efik, ASR-labelled
  with English glosses and per-clip consent metadata, under a custom **"Scuba Sample Voice Data License"** —
  evaluation/non-commercial, no redistribution. The card advertises a full **~200 h Ibibio+Efik corpus,
  500+ speakers**, licensable commercially. This is the only plausible route to real Efik speech at scale.
- `offiongbassey/efik_speech_dataset`: 5,875 clips, 16 kHz, audio+text+speaker_id, **Apache-2.0**, ~1 GB
  (likely single-digit hours, no card text). `offiongbassey/efik_audio_dataset`: 2,632 clips at 22.05 kHz,
  TTS-oriented, **no licence declared**.
- A ~3 h Efik corpus is reported at Interspeech 2026 [unopened — surfaced in search, not confirmed].
- Omnilingual ASR does carry Efik's close Cross River relatives **Ibibio (`ibb_Latn`, Low)** and **Anaang
  (Low)**, plus Obolo (`ann_Latn`, Mid).

**The research opening this creates** is concrete and small enough for one semester: Efik is a language for
which (a) the lab already ships a translation product, (b) no ASR system claims support, (c) a close
relative *is* supported by a 7B model that was explicitly built to be extended from a handful of examples,
and (d) an ~200 h commercial corpus exists if it is ever needed. Comparing `omniASR-LLM-7B-ZS` few-shot,
Ibibio-adapter transfer, and a small fine-tune of `omniASR-W2V-300M` on the same few hours is a clean
three-arm experiment with a real baseline.

### 5.4 Benchmarks and evaluation resources

- **FLEURS** (Conneau et al., **IEEE SLT 2022**; arXiv:2205.12446), `google/fleurs`, **CC BY 4.0**, 102 languages,
  n-way parallel over FLoRes-101 sentences (~10–12 h/lang). Of the six: `xh_za` (4,953 utts), `ig_ng`
  (4,221), `sw_ke` (3,768), `yo_ng` (3,548); **Efik and Twi/Akan absent**. Because it is n-way parallel, it
  is a free S2TT test set for any pair among {xho, ibo, swh, yor} × English.
  → `papers/Conneau-2023-FLEURS-Benchmark.pdf`
- **FLEURS-R** (arXiv:2408.06227, **Interspeech 2024**): same 102 languages, audio restored with Miipher for
  generation tasks. **FLEURS-SLU** (arXiv:2501.06117): 692 h topical classification + 944 h listening MCQA.
- **IWSLT 2026 African/Celtic track / NaijaS2ST** — see §4.3. **The first real S2TT benchmark for Igbo and
  Yoruba**, ~75 h newly recorded, with published baselines including OmniASR→NLLB-200. Licence not stated on
  the track page.
- **SSA-COMET** (arXiv:2506.04557, **EMNLP 2025**) is now IWSLT's official metric for the African track,
  succeeding **AfriCOMET/AfriMTE** (arXiv:2311.09828, **NAACL 2024**, 13 languages). SSA-MTE has ~73k
  annotations over 14 African language pairs.
  → `papers/SSA-COMET-2025-MT-Evaluation-African-Languages.pdf`, `papers/Wang-2024-AfriMTE-AfriCOMET.pdf`
- **Tone warning, and it applies to four of the six.** arXiv:2602.04716 (**AfricaNLP 2026 workshop**) argues WER
  mischaracterises ASR for tonal languages and proposes Feature/Tone Error Rate; its Yoruba baseline is
  WER 0.788, CER 0.305, **FER 0.151**. Xhosa, Igbo, Twi, Yoruba and Efik are all tonal, and Omnilingual ASR
  reports **CER**, not WER, across 1,600 languages. Any claim comparing the two metrics needs this paper.
  → `papers/Linguistically-Informed-2026-Multilingual-ASR-African-Tone.pdf`
- **AfriHuBERT** (arXiv:2409.20201, **Interspeech**, `ajesujoba/AfriHuBERT`): mHuBERT-147 continued on
  10K+ h, 16 → 1,226 African languages; +3.6 F1 language ID, −2.1 WER on FLEURS. The natural small
  self-supervised baseline against `omniASR-W2V-300M`. → `papers/Alabi-2025-AfriHuBERT.pdf`

### 5.5 Licence flags that matter for a shipping product

**Non-commercial / restricted:** NaijaVoices (BY-NC-SA) · AfriSpeech-200 / -Dialog / AfriVox (BY-NC-SA) ·
**MMS weights and MMS-TTS (CC-BY-NC)** · **SeamlessM4T, SeamlessStreaming (CC-BY-NC-4.0)** ·
SeamlessExpressive (gated, proprietary Seamless licence) · GhanaNLP Twi 500 h (BY-NC) ·
Scuba Efik (custom, evaluation only) · BABEL (LDC, paid).
**CC BY 4.0 but no-TTS-use:** `dsfsi-anv/za-african-next-voices` — the only 500 h of Xhosa.
**Commercially clean:** **Omnilingual ASR weights (Apache-2.0)** and its corpus (CC BY 4.0) ·
FLEURS / FLEURS-R (BY 4.0) · BibleTTS (BY-SA 4.0) · WAXAL (BY 4.0 / BY-SA 4.0) · NCHLT (BY 3.0 / BY 4.0) ·
OpenSLR SLR32 & SLR86 (BY-SA 4.0) · ALFFA (MIT) · Kencorpus (BY 4.0) ·
Digital Umuganda AfriVoice Swahili (BY 4.0) · Common Voice (CC0) · Hibiki-Zero (MIT).

That contrast is the practical headline of this section: **the Apache-2.0 licence on Omnilingual ASR makes
it the first frontier-scale multilingual speech model a product lab can actually ship**, where MMS and
Seamless — the two systems it beats — cannot be.

### 5.6 Traps found while checking

1. **Common Voice locale `xhe` is Khetrani (a Lahnda variety of Pakistan), not Xhosa.** Xhosa is `xh`
   (0.02 h validated). Easy to grab by mistake; `xhe` has 9.62 h.
2. **HF mirrors of Common Voice stop at `mozilla-foundation/common_voice_17_0`.** CV 18–26 exist only as
   direct downloads from commonvoice.mozilla.org.
3. **"MSLR" could not be verified to exist** — likely a conflation with MLS (Multilingual LibriSpeech) or
   OpenSLR. Do not cite it.
4. **African Next Voices headline numbers do not reconcile**: Gates Africa says 9,000 h / 18 languages /
   3 countries; the Lanfrica aggregator says ~18,000 h / 24 languages / 7 countries. Cite per-dataset
   figures only. The Nigeria portion (`DataScienceNigeria/african_voice_ng`) is currently an **empty
   placeholder**; the Kenya portion (HF org `Anv-ke`) is gated, has **no Swahili**, and its unscripted
   transcripts have a known mojibake bug (UTF-8 read as Latin-1).
5. **CoVoST 2 and mExpresso/SeamlessExpressive contain no African languages at all.** VoxPopuli is European
   Parliament only. VoxLingua107 has Swahili and Yoruba only.
## 6. Verification debt

Everything in this section was **not** opened. Confirm before citing any of it.

### 6.1 Claims I could not resolve, and that matter

1. **Hibiki-Zero's licence.** The HF card for `kyutai/hibiki-zero-3b-pytorch-bf16` has **no `license`
   field** (confirmed via the API). The GitHub README (`kyutai-labs/hibiki-zero`) says **MIT**. A second
   reading of the rendered HF page reported **CC-BY-NC-SA-4.0**. These cannot all be true. **Do not treat
   Hibiki-Zero as MIT-licensed without checking the repo's LICENSE file directly.** This matters because it
   is otherwise the most attractive open S2ST model in this document.
2. **Omnilingual ASR's S2TT checkpoint.** §5.6 of the paper reports S2TT models built on OmniASR-W2V
   encoders, but no S2TT repo appears among the 13 `facebook/omniASR-*` models. I checked the HF `facebook/`
   listing, **not** every branch/tag of the GitHub repo. The GitHub README also mentions "v2 versions" and
   "unlimited audio length variants" that do not correspond to any HF repo I found; third-party ONNX
   conversions dated **2026-02-05** are tagged `-v2`, implying a v2 release exists somewhere I did not open.
3. **Omnilingual ASR Table 3's arithmetic.** The row "LTPP, internal & licensed data — 150,100 h" exceeds
   the stated total of 120,710 h. I did not resolve whether the rows are a partition, an overlap, or a
   typo. Do not sum that table.
4. **AfriHuBERT's licence.** `ajesujoba/AfriHuBERT` carries the bare string `cc` — not a valid SPDX
   identifier. The June-2026 `-large` and `-xlarge` variants are Apache-2.0 and are cited only to a
   dissertation. `[unopened]`
5. **Gemma 3n's audio language list and audio-encoder size.** Not published on the card; the widely quoted
   ~680M encoder figure is not stated there. Gemma 4's launch date is also ambiguous — HF `createdAt` says
   2026-03-02, one secondary source says July 2026. `[unopened]`
6. **XTTS-v2's parameter count** (~467M main + ~52.6M DVAE) is **inferred from file sizes**; Coqui never
   published it. `coqui.ai` and the canonical CPML URL now 404.

### 6.2 Papers cited from listings/abstracts only, never opened

- `2501.11378` Whisper hallucinations induced by non-speech audio · `2511.14219` Listen Like a Teacher ·
  `2603.06193` Whisper-CD (long-form repetition loops) · `2604.10736` BlasBench (Whisper >100% WER on Irish
  via insertion hallucination).
- `2605.28227` SpeechCOMET / *Why We Need Speech to Evaluate Speech Translation* · `2509.17349` latency-metric
  meta-evaluation · `2606.08748` HydraQE.
- Surveys with open-challenge sections: `2312.01053` (end-to-end S2TT survey) · `2505.15957` (holistic
  evaluation of large audio-language models) · `2502.19548` (LLMs meet speech) · `2605.21008` (audio
  reasoning) · `2511.01299` (general auditory intelligence) · `2505.11690` (African speech survey).
- `2603.11378` (w2v-BERT 2.0 Swahili, 3.24% CV WER) · `2507.17578` (synthetic Hausa) · `2404.02000`
  (SSA-HuBERT) · `2511.23370` (SSA-HuBERT Base/Large/XL) · `2606.02375` (WAXAL-NET) · `2602.09785` (Bambara
  benchmark) · `2604.08448` (AfriVoices-KE) · `2605.03590` (AfriVox-v2) · `2604.00688` (OmniVoice) ·
  `2608.11650` (Confucius4-TTS) · `2605.05611` (X-Voice) · `2607.18317` (rule-based diphone Yoruba
  synthesizer) · `2409.09305` (UTMOSv2) · `2406.18009` (E2-TTS) · `2505.17589` (CosyVoice 3) ·
  `2409.00750` (MaskGCT, ICLR 2025).

### 6.3 Things reported to exist that I could not confirm

- **"AutoRank" is not an IWSLT metric.** It appears in neither IWSLT findings paper and arXiv full-text
  search returns nothing. The premise appears to be mistaken. IWSLT's actual stack is COMET (primary),
  BLEURT, BLEU, TER, chrF/chrF++, characTER, SubER (subtitling), spBLEU + chrF++ + SSA-COMET (African
  track), BLASER (S2S).
- **"MSLR"** as a Bible-derived corpus could not be verified to exist — likely a conflation with MLS
  (Multilingual LibriSpeech) or OpenSLR. Do not cite it.
- **Neither IWSLT findings paper has an arXiv preprint.** Both are ACL Anthology only (`2025.iwslt-1.44`,
  `2026.iwslt-1.39`). Do not attach an arXiv id to either.
- **Both IWSLT abstracts contradict their own introductions.** The 2025 abstract lists a dubbing track and a
  speech-to-speech track; neither ran. Cite the intros.
- **African Next Voices headline numbers do not reconcile** (9,000 h / 18 langs / 3 countries vs ~18,000 h /
  24 langs / 7 countries). Cite per-dataset figures only. The Nigeria portion is an empty HF placeholder;
  the Kenya portion is gated and has no Swahili.
- **A ~3 h Efik corpus reported at Interspeech 2026** — I have the paper (`papers/Efik-TTS-2026-Digital-
  Preservation.pdf`) but did not verify the venue independently of the PDF's own front matter.
- **Fish Audio "S2.1-Pro"**, `2603.08823`, and the 81.88% EmergentTTS-Eval figure: no such repo on the
  fishaudio HF org. **MiniMax Speech, Inworld TTS, Parakeet-TTS**: no first-party open-weight repo found —
  treat as closed. **Lelapa AI's Vulavula** is a closed API; **Intron's Sahara** has no public weights.
- **`microsoft/VibeVoice-ASR-*` cards** — HF returned 429. `microsoft/VibeVoice-7B` now 401s; Microsoft
  removed the VibeVoice TTS code from GitHub on 2025-09-05 citing misuse. Community mirrors carry the
  weights under MIT.
- **Per-language CER/WER/UTMOSv2 numbers for the OpenBibleTTS releases** are not on the HF cards; they live
  at `github.com/davidguzmanr/open-bible-models`. `[unopened]` Also, the `F5-TTS-OpenBible-*` cards are
  tagged CC-BY-SA-4.0 and claim "trained from scratch" while also tagging `base_model: SWivid/F5-TTS`, whose
  weights are CC-BY-NC. **If commercial use matters, prefer the VITS or EveryVoice arms.**
- **`ALFFA` Swahili hours** — the commonly quoted ~10 h is not stated on the OpenSLR page.
- **Whisper's own licence is inconsistent across OpenAI's cards**: `openai/whisper-large-v3` is tagged
  `apache-2.0`, `openai/whisper-large-v3-turbo` is tagged `mit`. Both were read from the API today.

### 6.4 Where this survey did not look

- **No interpretability literature on speech models.** Whether speech encoders/decoders build the
  massive-activation "constant" the text-side arm found, and whether Whisper's repetition loops are the same
  phenomenon, is not covered here beyond noting that Calm-Whisper (§4.8) is the only mechanistic account
  found. That is the interpretability arm's territory.
- **No streaming/latency engineering** beyond what IWSLT reports.
- **No commercial API pricing or throughput benchmarking.**
- **Model quantization for speech** is touched only where IWSLT's compression track touches it (§4.7).
