# Compression Methods for NLLB and XTTS: Deep Research Report

**Date:** 2026-06-19 | **Arctos Project (Adrian, BYU)**
**Purpose:** Survey of compression methods (PTQ, pruning, distillation) for NLLB encoder-decoder MT models and XTTS v2 TTS models, as context for designing a custom LRL-preserving quantization method.

---

## Context

The lab uses NLLB-200 (encoder-decoder multilingual MT) and XTTS v2 (TTS: GPT autoregressive + HiFi-GAN vocoder). The goal is a custom quantization method that preserves **low-resource language (LRL)** quality under compression — informed by interpretability. This report is the literature foundation for that method.

Existing Arctos compression work targets **decoder-only** LLMs (Aya Expanse 8B, Qwen3, Llama-3). This report covers what changes when moving to enc-dec or TTS architectures.

Also relevant: **Meta Omnilingual MT (March 2026)** — 1,600-language MT system (8× NLLB-200's 200 languages). Two architectures: OMT-LLaMA (decoder-only, LLaMA3 + RAG) and OMT-NLLB (enc-dec, OmniSONAR encoder). 1B–8B models match/exceed 70B LLM baselines. Evaluation: BLASER 3, OmniTOX, BOUQuET/Met-BOUQuET (freely available). The lab is not using this yet, but it is the forward direction.

---

## 1. NLLB Compression: State of the Art (2024–2026)

### 1.1 PTQ Methods

**The dominant production path is CTranslate2 INT8, not AWQ or GPTQ.**

- `OpenNMT/nllb-200-3.3B-ct2-int8` (HuggingFace): uniform linear-layer INT8, `int8_float16` on GPU. No activation-aware scaling, no Hessian reconstruction, no differential treatment of encoder vs decoder vs cross-attention. 2–4× memory reduction, 2–8× CPU speedup. **No published BLEU/COMET quality delta for this checkpoint.** This is a documented literature gap.
- **AWQ and GPTQ: not natively supported, no published results for any NLLB variant.** Both were designed for decoder-only architectures and assume unidirectional KV caching. None of AutoAWQ, AutoGPTQ, or LeanQuant have published enc-dec support as of June 2026.
- **bitsandbytes:** applies to any `nn.Linear` via HF `load_in_8bit`/`load_in_4bit`; community reports of use with `nllb-200-distilled-1.3B` exist but no peer-reviewed quality evaluation.
- **GGUF:** not available for NLLB — llama.cpp is decoder-only only.

### 1.2 Key Architectural Differences from Decoder-Only PTQ

NLLB has three distinct attention types with fundamentally different activation statistics:

| Attention type | Keys/Values source | Calibration needed |
|---|---|---|
| Encoder self-attention | Encoder hidden states | Source-side text |
| Decoder self-attention (causal) | Decoder hidden states | Target-side text |
| **Cross-attention** | **Encoder outputs (fixed after encoder pass)** | **Bilingual parallel pairs** |

**The cross-attention problem:** CAR-SAM (CVPR 2026, arXiv:2605.16901) on SAM's cross-attention shows 4-bit PTQ causes "attention dissipation" (attention weights collapse to near-uniform) and "reconstruction oscillation" (correcting one branch drives error into the other). Finding transfers structurally to NLLB: **cross-attention is the dominant quantization degradation source.** Aespa (NeurIPS 2024, arXiv:2402.08958) makes the same point: layer-wise reconstruction (GPTQ-style) fails when layers are tightly coupled.

**Calibration data asymmetry:** For NLLB, generic text calibration (C4, Pile) is inappropriate — cross-attention activations literally do not exist without bilingual parallel pairs in the calibration set. This is even stronger motivation for MT-conditional calibration than in Arctos's Q6 decoder-only work.

### 1.3 Layer Pruning

**NASH (EMNLP 2023, arXiv:2310.10054) — most applicable general framework:**
- Width pruning (head + FFN neuron removal) works better for the **encoder** — removing encoder layers causes larger quality drops than pruning channels
- **Decoder depth pruning dominates speedup** — autoregressive KV cache grows with depth
- **Cross-attention layers are preserved last** — empirically the highest-sensitivity component
- Pruning is a **2D optimization problem**: encoder depth × decoder depth, not 1D like decoder-only (ShortGPT/SLEB/GeLaCo)

| Dimension | Decoder-only (SLEB/GeLaCo) | Encoder-decoder (NASH/NLLB) |
|---|---|---|
| Pruning targets | Decoder blocks only | Encoder + decoder + cross-attention (3 targets) |
| Asymmetry | None | Encoder computed once; decoder runs N times autoregressively |
| Cross-attention | Does not exist | Protected preferentially — highest sensitivity |
| Calibration | Monolingual text sufficient | Bilingual pairs required |

**MoE expert pruning of NLLB-200-54B (Meta/Naver, arXiv:2212.09811, ACL 2023):**
Up to 80% of experts removable per language pair without fine-tuning, negligible BLEU loss. Collapses 54.5B → single-GPU for a specific language pair. **Does NOT transfer to NLLB-3.3B (dense, no MoE structure).**

**AfriNLLB (arXiv:2602.09373, AfricaNLP 2026):** iterative pruning + distillation for African language pairs on NLLB-600M. Distillation component confirmed; pruning details uncertain.

### 1.4 Knowledge Distillation

**Meta's original NLLB distillation (arXiv:2207.04672):**
- Teacher: NLLB-200-54B MoE; Students: 1.3B and 615M dense
- Method: **online token-level KD** (teacher forward pass → soft targets during student training), NOT offline sequence-level KD
- Results: 1.3B student +0.5 chrF++, 615M +0.3 chrF++ over same-size baseline without distillation
- `nllb-200-distilled-600M` and `nllb-200-distilled-1.3B` on HuggingFace are these checkpoints

**Contrast with GKD (Arctos's method):** GKD generates student rollouts and uses teacher to score them (on-policy). Meta's NLLB distillation uses forward-pass soft targets (off-policy). GKD-style on-policy distillation **has never been applied to NLLB** — genuinely novel contribution.

**Multi-hypothesis distillation for multilingual NMT (arXiv:2507.21568, July 2025):** sequence-level n-best-list distillation targeting LRL pairs in multilingual NMT including NLLB variants. Medium confidence — check before citing.

---

## 2. XTTS Compression: State of the Art (2024–2026)

### 2.1 Architecture (arXiv:2406.04904, Interspeech 2024)

1. **GPT-2-style autoregressive module** — takes text tokens + speaker conditioning, generates discrete audio tokens (VQ-VAE, 8192-code codebook truncated to 1024). Causal decoder-only transformer — architecturally the same family as GPT-2.
2. **HiFi-GAN vocoder** — converts discrete audio token sequences to raw waveforms. Convolutional, single-pass, fast.
3. **Speaker Perceiver** — generates speaker conditioning from reference audio mel-spectrogram.

### 2.2 Quantization

**No published academic paper on XTTS quantization exists.** GitHub issue coqui-ai/TTS #3819 (July 2024) requested quantization support → closed as "won't fix."

Community workarounds: bitsandbytes INT8 on XTTS-GPT via HF `load_in_8bit` (no published quality evaluation). GGUF not available (llama.cpp does not support XTTS).

Why XTTS quantization is harder than text LLM quantization:
- The GPT module generates discrete VQ-VAE audio tokens. Errors that would be tolerable in text (slightly wrong next-token probabilities) produce audible artifacts — clicks, pitch jumps, speaker identity shifts — because audio tokens index into a perceptually-sensitive codebook.
- Speaker conditioning embedding injected as prefix/cross-attention creates input distribution mismatch if GPT is quantized while embedding stays at full precision.

### 2.3 Pruning and Distillation

None published for XTTS or XTTS-v2. Completely open area.

**Closest published work — SPADE (KAIST/42dot, arXiv:2509.20802, 2026):**
WER-based layer importance scoring → structured depth pruning → multi-level KD (token + sequence level), applied to a generic LLM-TTS autoregressive system. Results: 1.7× faster, 20% VRAM reduction, near-parity perceptual quality. Does not touch vocoder. Natural template for XTTS compression.

### 2.4 TTS Compression More Broadly

| TTS type | Compression approach | Vocoder treatment |
|---|---|---|
| Non-autoregressive (FastSpeech2, VITS, StyleTTS2) | Standard transformer INT8 via ONNX/TFLite; layer dropping acoustic model | Left at full precision (already fast, very sensitive) |
| Autoregressive LLM-TTS (XTTS, Parler-TTS, VoiceCraft) | Same challenges as LLM PTQ + perceptual sensitivity of audio token prediction | Downstream of bottleneck; more robust to quantization |

---

## 3. Open Research Gaps (Contribution Space)

**For NLLB:**
1. No published BLEU/COMET vs bit-width table for NLLB-200-3.3B under any PTQ method
2. No AWQ, GPTQ, or LeanQuant application to any NLLB variant
3. No cross-attention sensitivity analysis (outlier channels, salient channels, super-weights) for NLLB
4. No mixed-precision PTQ for enc-dec MT (e.g., cross-attention at INT8, self-attention at INT4)
5. No NASH-style layer pruning of NLLB-3.3B dense model

**For XTTS:**
6. No published quantization benchmark for XTTS (any method, any bit-width) — field starts from zero
7. No WER/MOS vs compression tradeoff table
8. No pruning or distillation applied to XTTS

**For LRL specifically:**
9. No study of LRL vs HRL calibration data effect on salient channel sets for NLLB
10. No mixed-precision scheme for NLLB that allocates bits based on LRL language sensitivity

---

## 4. Transfer from Arctos Existing Tools

| Arctos tool (`src/interp/`) | NLLB encoder | NLLB decoder + cross-attn | XTTS-GPT |
|---|---|---|---|
| Super-weight causal-KL | Adapt: measure encoder output KL under ablation with LRL inputs | Adapt: separate per attention type | Direct transfer (decoder-only) |
| AWQ salient channels (`salient_channels.py`) | Applicable; run separate LRL vs HRL calibration split | Applicable; bilingual pairs required | Direct transfer |
| Fisher diagonal (`hessian_diag.py`) | Applicable; bilingual calibration | Applicable | Direct transfer |
| Wanda mask (`compress.py`) | Applicable to FFN; LRL calibration | Applicable | Direct transfer |
| SLEB/depth pruning | Needs 2D extension (encoder × decoder) | Cross-attention protected last | Direct to GPT layers |
| GKD distillation | Novel for NLLB — no prior | Novel for NLLB | Novel for XTTS |

---

## 5. Priority Reading List

1. **arXiv:2310.03686** — DecoderLens (NAACL 2024): NLLB-600M encoder layer importance for LRL; the first experiment to run
2. **arXiv:2605.01229** — Attention sinks in NLLB-200 (2026): cross-attention 83–91% mass on sink tokens; most actionable finding
3. **arXiv:2212.09811** — Memory-efficient NLLB (Naver, ACL 2023): MoE expert pruning; decoder LRL fragility
4. **arXiv:2310.10054** — NASH (EMNLP 2023): encoder-width + decoder-depth pruning; cross-attention preservation
5. **arXiv:2402.08958** — Aespa (NeurIPS 2024): attention-wise reconstruction; cross-layer coupling problem for GPTQ
6. **arXiv:2605.16901** — CAR-SAM (CVPR 2026): cross-attention PTQ failure modes (vision analogy)
7. **arXiv:2406.04904** — XTTS v2 paper (Interspeech 2024): architecture ground truth
8. **arXiv:2509.20802** — SPADE (KAIST/42dot, 2026): only structured pruning + KD paper for LLM-TTS
9. **arXiv:2207.04672** — NLLB original paper (Meta 2022): online distillation methodology

---

*Research conducted 2026-06-19 via 6-angle web search + adversarial verification (54 agents, 78 claims collected, 8/12 verified claims survived 3-vote check, 4 refuted).*
