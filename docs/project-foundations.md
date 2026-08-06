# Project Foundations: Compression for NLLB and XTTS

> Personal reference for understanding the project before the advisor meeting.
> Covers: what quantization is, the two target architectures, the compression landscape,
> how interpretability fits, and a reading list.

---

## 1. What Quantization Is

A neural network is a massive collection of floating-point numbers called weights. By default
they live in **float16** — 16 bits per number. Quantization means storing them with fewer bits:
**INT8** (8 bits), **INT4** (4 bits), INT3, etc. The math is simple: find the range of values in
a weight matrix, divide that range into N buckets, and store which bucket each value falls into.
At inference you multiply by a scale factor to recover an approximation of the original value.

**Why do this?**
- **Memory:** INT4 is 4× smaller than float16. A 3.3B-parameter model at float16 is ~6.6GB;
  at INT4 it is ~1.65GB.
- **Speed:** Integer arithmetic is faster than float arithmetic on most hardware, especially
  on budget GPUs like the T4.

**The cost:** some information is lost. float16 can represent ~65,000 distinct values; INT4 can
only represent 16. That precision loss is what degrades model quality — especially at very low
bit-widths (2–3 bit) and especially for low-resource languages (LRL), which have narrower
activation patterns and less redundancy to absorb the noise.

**Post-training quantization (PTQ)** = quantize an already-trained model without any retraining.
Cheap and fast. Reasonably good at 4-bit; gets hard below 4-bit. This is the approach here —
not training from scratch in low precision.

---

## 2. What "Custom Quantization" Means

**Uniform PTQ:** every weight matrix gets the same bit-width (e.g., everything → INT4). Fast to
implement, naive in practice.

**Mixed-precision PTQ:** different layers get different bit-widths. The most sensitive layers
get INT8; the least sensitive get INT4 or INT2. Done right, you capture most of the memory
savings while protecting quality where it actually matters.

**The hard problem:** which layers get which bits? That is what sensitivity analysis is for.
Standard tools:

| Tool | What it measures |
|---|---|
| Fisher diagonal / Hessian | Loss curvature — how much does the loss change if this weight changes? |
| AWQ activation magnitude | Which input channels produce large activations that amplify weight errors? |
| Wanda | Activation magnitude × weight magnitude — best proxy for pruning sensitivity |
| Super-weight detection (causal KL) | Individual scalar weights whose ablation alone collapses output |

**The custom part in this project:** instead of running these sensitivity tools on generic English
text, run them on LRL data (Bengali, Zulu, Kinyarwanda, Kekchi). A layer that is not sensitive
for English may be critical for Zulu. LRL-specific calibration finds LRL-specific protection
targets. That is the core idea.

**One sentence:** sensitivity-based mixed-precision, where sensitivity is measured on LRL data
instead of generic text.

---

## 3. The Two Target Architectures

### 3.1 NLLB-200-distilled-1.3B

An **encoder-decoder transformer** — fundamentally different from the decoder-only LLMs
(Aya, Llama, Qwen) in the Arctos phase-one work.

```
Source sentence (English)
        ↓
  ┌─────────────────────────────────────┐
  │  Encoder — 24 transformer layers    │
  │  Reads entire source at once        │
  │  (bidirectional, not autoregressive)│
  └──────────────┬──────────────────────┘
                 │ encoder output (fixed)
        ↓
  ┌─────────────────────────────────────┐
  │  Decoder — 24 transformer layers    │
  │  Generates target one token at once │
  │  (autoregressive)                   │
  │                                     │
  │  Each layer has:                    │
  │   - decoder self-attention          │
  │   - cross-attention ← encoder output│
  └──────────────┬──────────────────────┘
                 │
        ↓
Target sentence (Zulu / Kinyarwanda / ...)
```

**Model sizes:**

| Variant | FP16 | INT8 | INT4 |
|---|---|---|---|
| distilled-600M | ~1.2 GB | ~0.6 GB | ~0.3 GB |
| **distilled-1.3B** (lab uses this) | **~2.6 GB** | **~1.3 GB** | **~0.65 GB** |
| 3.3B | ~6.6 GB | ~3.3 GB | ~1.65 GB |

**What makes this harder to quantize than decoder-only:**
There are three distinct attention types with fundamentally different activation statistics:

| Attention | Attends to | Calibration needed |
|---|---|---|
| Encoder self-attention | Source tokens | Source-side text |
| Decoder self-attention | Previously generated tokens | Target-side text |
| **Cross-attention** | **Encoder output** | **Bilingual parallel pairs** |

Cross-attention is where PTQ failure concentrates. It couples two separately-calibrated
components (encoder and decoder), so quantizing it carelessly introduces errors that compound
through the decoder's generation process. AWQ and GPTQ were designed for decoder-only
architectures and do not handle this — they have no native enc-dec support.

**The LRL-specific vulnerability:** Language tag tokens like `__zul_Latn__` function as
**attention sinks** — they absorb 83–91% of all cross-attention mass in NLLB (arXiv:2605.01229).
These are not just routing hints; they carry an outsized structural load. Quantizing the
embedding layer aggressively corrupts the routing signal and LRL quality collapses first, before
high-resource language quality degrades. They must stay at 16-bit.

### 3.2 XTTS v2

A **two-part pipeline** — not a single transformer:

```
Text input (Zulu sentence)  +  Reference audio clip (speaker)
        ↓                              ↓
        ↓              [Speaker Perceiver — generates speaker embedding]
        ↓                              ↓
  ┌─────────────────────────────────────────────────────┐
  │  GPT-2-style autoregressive transformer             │
  │  ~350–450M parameters, decoder-only                 │
  │  Generates discrete audio tokens (indices 0–1023)   │
  │  conditioned on speaker embedding                   │
  └──────────────────────────┬──────────────────────────┘
                             │ sequence of audio token indices
        ↓
  ┌─────────────────────────────────────────────────────┐
  │  HiFi-GAN vocoder — ~50M parameters, convolutional  │
  │  Converts audio token sequence → raw waveform       │
  │  Single-pass, fast, local processing                │
  └──────────────────────────┬──────────────────────────┘
                             │
        ↓
Audio output (dubbed Zulu speech)
```

**Total checkpoint: ~1.87 GB** (the released checkpoint is already somewhat mixed-precision).

**VQ-VAE codebook:** a lookup table mapping integer indices (0–1023) to audio vectors. This is
separate from the GPT weights. One wrong index lookup produces an audible artifact — a click,
pitch jump, or speaker shift — not a subtle quality degradation. **Codebook must stay at
full precision (16-bit), always.**

**What this means for compression:**
- GPT component: decoder-only transformer → all existing Arctos tools transfer directly.
  This is the main compression and research target.
- HiFi-GAN vocoder: convolutional, no long-range dependencies, robust. Expected to tolerate
  INT4 with minor audible artifacts only.
- VQ-VAE codebook: 16-bit, non-negotiable.
- Speaker Perceiver: small, keep at INT8.

**The XTTS-specific challenge:** audio token prediction is more perceptually sensitive than
text token prediction. A slightly wrong probability for the next text token might produce a
synonym. A slightly wrong probability for the next audio token might produce the wrong phoneme
for a click consonant in Zulu. Quality metrics for TTS are also harder — CER (character error
rate via ASR back-transcription), UTMOS (neural MOS predictor), and speaker similarity scores,
all of which are noisier than COMET.

---

## 4. Compression Landscape for These Models

### NLLB (encoder-decoder)

| Method | Status | Notes |
|---|---|---|
| **CTranslate2 INT8** | Production, works | Uniform INT8; no published quality delta table — running this and measuring chrF++ is itself a contribution |
| **bitsandbytes INT8 / INT4** | Works via HuggingFace | Accessible, per-layer; no published NLLB results |
| AWQ | No enc-dec support | Designed for decoder-only; skip |
| GPTQ | No enc-dec support | Designed for decoder-only; skip |
| GGUF (llama.cpp) | No enc-dec support | Decoder-only only; skip |
| **NASH structured pruning** | Research (EMNLP 2023) | Most applicable pruning framework: encoder-width + decoder-depth as a 2D problem |
| **Distillation (online KD)** | Meta's own method | Used to create the distilled-1.3B checkpoint; can be extended with LRL data |

**The accessible path:** CTranslate2 (baseline, no code needed) + bitsandbytes (custom
mixed-precision on top of standard HF loading). Your custom method is implemented as a
post-load weight-replacement loop that swaps specific layers to lower precision based on
the sensitivity map from the interpretability diagnostics.

### XTTS

| Method | Status | Notes |
|---|---|---|
| **bitsandbytes INT8 / INT4 on GPT** | Technically works, untested | Community reports exist, no quality evaluation |
| Standard PyTorch INT8 on HiFi-GAN | Works | Convolutional, standard quantization fine |
| **SPADE-style pruning + KD** | Research template (KAIST, 2026) | WER-based layer importance → depth pruning → multi-level KD; applied to generic LLM-TTS, not XTTS specifically |
| GGUF | No | llama.cpp doesn't support XTTS |

**Zero published baselines exist for XTTS quantization.** Any result is the first result.

---

## 5. Does Compression for XTTS Make Sense?

**Yes, for the deployment goal.** The target is moving the dubbing app from A100 (~$3–4/hr)
to T4 (~$0.50/hr). NLLB and XTTS are deployed as separate API services — each on its own
server — so the question is whether each model fits and runs at real-time latency on its
target GPU independently.

| Component | FP16 size | Compressed size | Target |
|---|---|---|---|
| NLLB-3.3B | ~6.6 GB | ~1.65 GB (INT4) | T4 (16 GB) |
| XTTS v2 | ~1.87 GB | ~1.0 GB (INT8) | T4 (16 GB) |

NLLB-3.3B at FP16 fits in a T4's 16 GB, but T4 FP16 throughput (~8 TFLOPS) is 10× slower
than A100 (~78 TFLOPS) — not viable for real-time latency. With CTranslate2 INT8 (fused
kernels), the T4 runs at 65 TOPS INT8, making latency competitive while halving memory.

**The scope caveat:** XTTS compression is more engineering-heavy and quality metrics are noisier.
If the project is time-constrained at 150 hours, the sensible prioritization is:

> NLLB = research core (6–8 weeks).
> XTTS = deployment validation (3–4 weeks).
> If XTTS hits unexpected blockers, document what was found and stay on NLLB.

---

## 6. How Interpretability Fits In

Interpretability here is a **diagnostic tool, not the method itself.** The deliverable is a
compression scheme (per-layer bit-width decisions), not a mechanistic theory.

The interpretability tools generate a **protection map** — which components are LRL-critical
and therefore need higher precision. Then the compression work applies that map and measures
whether it actually improves LRL quality retention vs uniform quantization.

| Interpretability tool | What it tells you | How it maps to bit allocation |
|---|---|---|
| DecoderLens (arXiv:2310.03686) | How many encoder layers LRL needs to resolve | Minimum protection depth → INT8 for early encoder |
| AWQ salient channel A/B split | Which channels activate specifically for LRL vs HRL | LRL-specific channels get higher precision |
| Attention sink analysis (arXiv:2605.01229) | Which cross-attention heads do content-routing vs just absorb sink mass | Content-routing heads → INT8; sink-serving → INT4 |
| Super-weight detection (causal KL) | Individual scalar weights whose ablation collapses output | Those scalars → 16-bit regardless of layer assignment |
| Wanda mask on XTTS-GPT | Layer-level sensitivity using activation × weight magnitude with LRL calibration | Sensitive XTTS-GPT layers → INT8 |

**If LRL-calibrated sensitivity identifies different protection targets than HRL-calibrated
sensitivity, and protecting those targets measurably improves LRL quality at the same
compression ratio, you have a result.** If it does not, that is a clean null result. Both
are honest contributions — the field has no baseline for either model.

---

## 7. The Advisor Pitch (Plain Language)

> "The dubbing app runs NLLB for machine translation and XTTS for speech synthesis. Both need
> to move from A100 to T4, which is a 6–8× cost reduction. No published work exists on
> quantizing either model — for NLLB there is no published quality table under any PTQ method;
> for XTTS the quantization issue was closed as won't fix. My approach is to use interpretability
> diagnostics — DecoderLens, attention sink analysis, salient channel analysis with LRL
> calibration — to identify which components are specifically critical for low-resource language
> quality, and use that to build a mixed-precision scheme rather than quantizing everything
> uniformly. The research question is whether LRL-aware precision allocation retains more LRL
> quality than uniform quantization at the same compression ratio. I validate on the actual
> deployed models and measure end-to-end latency and cost on a T4."

**Likely advisor question:** Is NLLB + XTTS too much for 150 hours?

**Answer:** NLLB is the research core; XTTS is the deployment validation. If XTTS hits blockers,
NLLB stands alone as the primary result. Having both models is the ambition, not the floor.

---

## 8. Papers to Read Before Starting

### Must-Read First (Before Any Experiments)

| Paper | What it gives you | arXiv |
|---|---|---|
| XTTS v2 (Interspeech 2024) | Ground truth on XTTS architecture — read before touching the model | 2406.04904 |
| DecoderLens (NAACL Findings 2024) | How to measure which encoder layers LRL needs — the first NLLB experiment to run | 2310.03686 |
| Attention sinks in NLLB-200 (2026) | Cross-attention 83–91% mass on sink tokens; language tag protection | 2605.01229 |
| NLLB original (Meta, 2022) | Architecture spec, distillation methodology, LRL evaluation setup | 2207.04672 |

### High Priority (Read Before Experiments, Week 1–2)

| Paper | What it gives you | arXiv |
|---|---|---|
| Memory-efficient NLLB / MoE pruning (ACL 2023) | Decoder LRL fragility (2× quality hit vs HRL); expert specificity | 2212.09811 |
| NASH pruning (EMNLP 2023) | Encoder-width + decoder-depth as 2D pruning; cross-attention protected last | 2310.10054 |
| SPADE — LLM-TTS compression (KAIST, 2026) | Only published structured pruning + KD paper for autoregressive TTS; direct XTTS template | 2509.20802 |
| Uneven Impact of PTQ in MT (arXiv:2508.20893) | The paper already replicated in Arctos — LRL + 2-bit nexus, calibration language effects | 2508.20893 |

### Useful Background (Read When Relevant)

| Paper | What it gives you | arXiv |
|---|---|---|
| Aespa (NeurIPS 2024) | Why GPTQ-style layer-wise reconstruction fails in coupled architectures (cross-attention problem) | 2402.08958 |
| CAR-SAM (CVPR 2026) | Cross-attention PTQ failure modes (vision model, but transfers structurally to NLLB) | 2605.16901 |
| AWQ (MLSys 2024) | Activation-weighted quantization — the tool you adapt for LRL-calibrated salient channels | 2306.00978 |
| SmoothQuant (ICML 2023) | How outlier activations break naive quantization and the standard fix | 2211.10438 |
| AfriNLLB (AfricaNLP 2026) | Iterative pruning + distillation for African language pairs on NLLB-600M | 2602.09373 |
| How Does Quantization Affect Multilingual LLMs? | Human eval study showing 1.7% automatic drop = 16% human drop for Japanese | 2407.03211 |
| Calibrating Beyond English (2026) | Calibration language alignment — the closest existing paper to the LRL calibration idea | 2601.18306 |

### For XTTS Quality Evaluation

| Tool / Paper | What it gives you |
|---|---|
| UTMOS (Interspeech 2022) | Neural MOS predictor — automated naturalness score for TTS, no human raters needed |
| SpeechBrain speaker verification | Speaker similarity metric for checking voice identity is preserved after compression |

---

## 9. Quick Reference: Model Sizes on T4

T4 = 16GB VRAM. Each model is a separate API service on its own server.

| Component | FP16 | INT8 | INT4 |
|---|---|---|---|
| NLLB-distilled-1.3B | 2.6 GB | 1.3 GB | 0.65 GB |
| NLLB-3.3B | 6.6 GB | 3.3 GB | 1.65 GB |
| XTTS v2 (full) | ~1.87 GB | ~1.0 GB | — |

---

*Written 2026-06-19. Update after advisor meeting with scope decisions.*
