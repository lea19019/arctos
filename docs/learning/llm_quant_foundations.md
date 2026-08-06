# Quantization Foundations: All Three Tracks

> Covers the three model tracks in the project, in the same style as
> `learning/project_foundations.md`. Read that file first for what quantization is
> and why we do it — this document assumes that background and adds:
> (1) decoder-only LLM architecture and compression, (2) deeper NLLB
> architecture with more on the cross-attention problem, (3) deeper XTTS
> architecture, (4) how quantized models actually execute on the GPU,
> (5) papers with clickable links.

---

## 1. Decoder-Only LLMs (the main research track)

### 1.1 What a Decoder-Only LLM Is

The models in our main research track — Aya Expanse, Llama 3, EuroLLM,
TowerInstruct, BLOOM, Gemma, Qwen — all share the same basic architecture.
It generates text one token at a time. At each step it reads everything
produced so far and predicts the next word (token).

```
Prompt: "Translate to Zulu: Hello, how are you?"
              ↓
  ┌─────────────────────────────────────────────┐
  │  Token embedding lookup                     │
  │  Each token → a vector of size d_model      │
  │  (4096 for 7B models; 8192 for larger)      │
  └───────────────────┬─────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────┐
  │  Transformer block × L layers (32 for 7B)   │
  │                                             │
  │  Each block:                                │
  │  ┌────────────────────────────────────────┐ │
  │  │ Causal self-attention                  │ │
  │  │  Each token attends to all PAST tokens │ │
  │  │  (cannot see the future — "causal")    │ │
  │  │  4 weight matrices: Q, K, V, O         │ │
  │  └──────────────────┬─────────────────────┘ │
  │  ┌──────────────────▼─────────────────────┐ │
  │  │ Feed-forward MLP (gate / up / down)    │ │
  │  │  Expand d_model → 4×d_model → shrink  │ │
  │  │  3 weight matrices per layer           │ │
  │  └────────────────────────────────────────┘ │
  └───────────────────┬─────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────┐
  │  LM head — linear → vocabulary probabilities│
  │  Sample the next token                      │
  │  Append to context → repeat                 │
  └─────────────────────────────────────────────┘
```

**The weights that quantization touches:**
Each of the L layers has 7 weight matrices (Q, K, V, O projections in
attention, plus gate/up/down in the MLP). A 7B model with 32 layers =
~224 large matrices. Quantizing all of these is the entire compression
problem — embeddings and the LM head are usually left at 16-bit.

**The KV cache:** as generation proceeds, past Key and Value vectors are
cached to avoid recomputing them. At small batch sizes this is minor. At
batch=64 with long sequences, the KV cache can exceed the model weights in
memory — it becomes the dominant cost for high-throughput serving.

### 1.2 Model Families and Sizes

The models we study in order of multilingual capability for MT:

| Model | Params | FP16 | INT8 | INT4 | Languages / focus |
|---|---|---|---|---|---|
| Aya Expanse | 8B | ~16 GB | ~8 GB | ~4 GB | 23 languages, instruction-tuned for MT |
| EuroLLM | 9B | ~18 GB | ~9 GB | ~4.5 GB | European languages |
| TowerInstruct | 7B | ~14 GB | ~7 GB | ~3.5 GB | MT-specific instruction tuning on Llama-2 |
| BLOOM | 7B | ~14 GB | ~7 GB | ~3.5 GB | 46 languages, African included |
| Llama 3.1 | 8B | ~16 GB | ~8 GB | ~4 GB | Primarily English; multilingual limited |
| Gemma 2 | 9B | ~18 GB | ~9 GB | ~4.5 GB | Primarily English |
| Qwen 2.5 | 7B | ~14 GB | ~7 GB | ~3.5 GB | Chinese + English strong |

T4 has 16 GB VRAM. At FP16, a 7B model barely fits; nothing larger does.
At INT4, every model above drops under 5 GB — the whole research question
is whether that INT4 compression preserves translation quality, especially
for low-resource languages.

### 1.3 Why Multilingual MT Makes Quantization Harder

A monolingual English model can absorb quantization noise roughly uniformly —
English is dense, well-trained, redundant. MT models are asymmetric:

- **High-resource languages** (French, German, Spanish): dense representations,
  lots of training data, lots of redundancy. Quantization noise gets absorbed.
- **Low-resource languages** (Zulu, Kinyarwanda, Bengali, Kekchi): narrower
  activation patterns, less training data, less redundancy to buffer errors.

The same bit-width applied uniformly does more damage where it can least be
absorbed. The key empirical finding: Marchisio et al. 2024 showed that a 1.7%
automatic metric drop corresponds to a 16% human-rated quality drop —
automatic metrics grossly understate the damage at low-resource languages.

### 1.4 Compression Landscape for Decoder-Only MT

| Method | Status | Notes |
|---|---|---|
| **GPTQ** | Production-ready | Hessian-weighted rounding; best quality at 3-bit; Marlin kernel on A100 |
| **AWQ** | Production, fastest kernel | Per-channel activation scaling; near-GPTQ at 4-bit; Marlin-AWQ is fastest W4 on A100+vLLM |
| **LeanQuant** | WMT25 quality winner | Loss-error-aware non-uniform grid; best 4-bit quality on record; slow kernel |
| **bitsandbytes NF4** | Easy, slow at inference | Used for QLoRA training; not for deployment |
| **HQQ** | Calibration-free | 50× faster than GPTQ; zero calibration; useful for rapid iteration |
| AQLM / QuIP# | Research tier, 2-bit | Best sub-3-bit quality; but kernel-bound on A100+vLLM |
| Wanda / SparseGPT | Training-free pruning | Unstructured sparsity; composable with INT4 quantization |
| GeLaCo + GKD | WMT25 high-compression anchor | 75% depth pruning + distillation; collapses quality at extreme ratios |

---

## 2. NLLB-200 (encoder-decoder MT)

### 2.1 What NLLB Is

NLLB-200 (No Language Left Behind) is Meta's 200-language MT model. The lab
uses `nllb-200-distilled-1.3B`. It is an **encoder-decoder transformer** —
a fundamentally different architecture from the decoder-only models above.

```
Source sentence: "Hello, how are you?"  (in any of 200 languages)
                            ↓
  ┌─────────────────────────────────────────────────────┐
  │  ENCODER — 24 transformer layers                    │
  │                                                     │
  │  Reads the ENTIRE source sentence at once.          │
  │  Not autoregressive — bidirectional attention.      │
  │  Each token attends to every other source token.    │
  │                                                     │
  │  Output: one fixed-size vector per source token.    │
  │  This encoder output does NOT change during         │
  │  generation — it is computed once and reused.       │
  └───────────────────────┬─────────────────────────────┘
                          │ encoder output (fixed, shape: seq_len × d_model)
                          │
  ┌───────────────────────▼─────────────────────────────┐
  │  DECODER — 24 transformer layers                    │
  │                                                     │
  │  Generates target one token at a time (like GPT).   │
  │  Each decoder layer has THREE sub-layers:           │
  │                                                     │
  │  ┌──────────────────────────────────────────────┐   │
  │  │ 1. Decoder self-attention (causal)           │   │
  │  │    Attends to previously generated tokens    │   │
  │  └─────────────────────┬────────────────────────┘   │
  │  ┌─────────────────────▼────────────────────────┐   │
  │  │ 2. Cross-attention  ← encoder output         │   │
  │  │    Queries come from the decoder              │   │
  │  │    Keys and Values come from the encoder      │   │
  │  │    This is where source meaning is read       │   │
  │  └─────────────────────┬────────────────────────┘   │
  │  ┌─────────────────────▼────────────────────────┐   │
  │  │ 3. Feed-forward MLP                          │   │
  │  └──────────────────────────────────────────────┘   │
  └───────────────────────┬─────────────────────────────┘
                          ↓
         Target sentence: "Sawubona, unjani?"  (Zulu)
```

### 2.2 Model Sizes

| Variant | FP16 | INT8 | INT4 |
|---|---|---|---|
| distilled-600M | ~1.2 GB | ~0.6 GB | ~0.3 GB |
| **distilled-1.3B** (lab uses this) | **~2.6 GB** | **~1.3 GB** | **~0.65 GB** |
| 3.3B (best quality) | ~6.6 GB | ~3.3 GB | ~1.65 GB |

### 2.3 The Three Attention Types (Why This Is Harder Than Decoder-Only)

A decoder-only LLM has one attention type. NLLB has three, and they have
fundamentally different activation statistics that require different calibration:

| Attention type | What it attends to | What calibration data you need |
|---|---|---|
| Encoder self-attention | All source tokens (bidirectional) | LRL source text |
| Decoder self-attention | Previously generated target tokens | LRL target text |
| **Cross-attention** | **Encoder output (fixed)** | **Bilingual parallel pairs** |

The cross-attention case is the hard one. Its Keys and Values come from the
encoder; its Queries come from the decoder. If you calibrate using only English
monolingual text, you never exercise cross-attention with realistic LRL encoder
outputs — your sensitivity analysis is measuring the wrong thing.

**AWQ and GPTQ cannot handle this.** Both tools assume a single stream of
activations flowing through a causal decoder. Neither has a code path for
"encoder output is fixed and feeds into cross-attention at every decoder
layer." As of June 2026, no published paper applies AWQ or GPTQ to any NLLB
variant.

### 2.4 The Cross-Attention Quantization Problem in Detail

Cross-attention couples two separately-optimized halves of the model. When
you quantize it naively:

- **Attention dissipation:** after quantization, attention weights collapse
  toward uniform — the model can no longer focus on specific source tokens.
  Translation becomes vague averaging over the source.
- **Reconstruction oscillation:** fixing quantization error in the Query
  projection drives error into the Key/Value projections and vice versa.
  Layer-wise reconstruction (the GPTQ approach) fails because the error
  isn't contained within one layer.

CAR-SAM (CVPR 2026, vision model with the same cross-attention structure)
documented both failure modes. Aespa (NeurIPS 2024) formalized why
GPTQ-style layer-wise reconstruction fails in any tightly-coupled
architecture — the cross-attention in NLLB is exactly this case.

### 2.5 Language Tags as Attention Sinks

NLLB uses special tokens — `__zul_Latn__`, `__kin_Latn__`, `__ben_Beng__` —
to route the decoder toward the right target language. They appear as the
first decoder input token.

Research (arXiv:2605.01229) found that these language tag tokens absorb
**83–91% of all cross-attention mass** in NLLB. They function as
**attention sinks** — the model dumps most of its cross-attention capacity
onto them, not onto the source content tokens.

This has two implications:

1. **The language tag embedding must stay at FP16.** Quantizing the token
   embedding matrix aggressively corrupts the routing signal. The model
   cannot route to the right language and LRL quality collapses first —
   before high-resource language quality drops at all.

2. **There are two distinct classes of cross-attention head:**
   - *Sink-serving heads:* almost entirely attend to the language tag.
     Structurally load-bearing but carry little content.
   - *Content-routing heads:* distribute attention across the source tokens.
     These are the quality-critical ones for LRL.

   If you can identify which heads are which (via attention pattern analysis
   on LRL calibration data), you get a fine-grained protection map: protect
   content-routing heads at INT8, allow sink-serving heads to go to INT4.

### 2.6 How DecoderLens Guides Bit Allocation

DecoderLens (arXiv:2310.03686) is an interpretability technique that probes
each encoder layer's usefulness directly:

**How it works:** run inference while substituting encoder layer ℓ's output
in place of the final encoder output. Measure translation quality (chrF++)
as a function of ℓ. If quality is near-zero until ℓ=18, the first 17 encoder
layers aren't contributing much to LRL translation.

**How it maps to bit allocation:**

```
Layer 1 ──── Layer 8 ──── Layer 16 ──── Layer 24
    ↓                         ↓               ↓
LRL quality                LRL quality   LRL quality
~zero                      ~starts       ~full

→ INT4 aggressive      → INT8 protect  → INT8 protect
```

For HRL (English-French), the quality curve rises earlier — the encoder
resolves those languages faster. This means HRL can be quantized more
aggressively in early layers than LRL. DecoderLens gives you the exact
depth profile to drive that decision.

### 2.7 Compression Landscape for NLLB

| Method | Status | Notes |
|---|---|---|
| **CTranslate2 INT8** | Production, works | Uniform INT8; ~2-4× memory reduction; no published quality table — measuring this is itself a contribution |
| **bitsandbytes INT8 / INT4** | Works via HuggingFace | Per-layer dtype via `load_in_8bit`/`load_in_4bit`; no published quality results |
| AWQ | No enc-dec support | Decoder-only only; skip |
| GPTQ | No enc-dec support | Decoder-only only; skip |
| GGUF (llama.cpp) | Not available | Decoder-only only; skip |
| **NASH pruning** | Research (EMNLP 2023) | Only published enc-dec pruning framework: encoder-width × decoder-depth as 2D problem |
| **Online KD** | Meta's own method | Used to create distilled-1.3B and 600M checkpoints; can be extended with LRL data |
| **GKD on-policy distillation** | Never applied to NLLB | Our decoder-only tool; novel if adapted to enc-dec |

**The accessible path for a custom method:**
CTranslate2 gives the uniform INT8 baseline cheaply (no code needed).
Then bitsandbytes enables custom mixed-precision on top of standard HuggingFace
loading — implement as a post-load weight-replacement loop that reads a
sensitivity map (from AWQ salient channels or Fisher diagonal on LRL calibration
data) and swaps specific layers to lower precision.

---

## 3. XTTS v2 (TTS)

### 3.1 What XTTS Is

XTTS v2 is a two-stage text-to-speech system that converts text and a reference
audio clip into synthetic speech that sounds like the reference speaker.

```
Text input: "Sawubona, unjani?"  +  Reference audio: [3-sec clip of target speaker]
        ↓                                      ↓
        ↓                    ┌─────────────────────────────────┐
        ↓                    │  Speaker Perceiver              │
        ↓                    │  Reads the reference mel-       │
        ↓                    │  spectrogram; outputs a fixed   │
        ↓                    │  speaker embedding vector       │
        ↓                    └────────────────┬────────────────┘
        ↓                                     │ speaker embedding
        ↓                                     ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  GPT-2-style autoregressive transformer                     │
  │  ~350–450M parameters — decoder-only (same family as Llama) │
  │                                                             │
  │  Input: text tokens + speaker embedding (as prefix tokens)  │
  │  Output: a sequence of discrete AUDIO TOKEN indices (0–1023)│
  │                                                             │
  │  This is NOT outputting audio samples directly.             │
  │  It is generating integers that are codebook indices.       │
  └────────────────────────────┬────────────────────────────────┘
                               │ sequence of integers, e.g. [482, 71, 903, ...]
                               ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  VQ-VAE Codebook (lookup table, NOT a neural network)       │
  │  1024 entries, each a 512-dim audio feature vector          │
  │  Maps each integer index → its corresponding audio vector   │
  └────────────────────────────┬────────────────────────────────┘
                               │ sequence of audio feature vectors
                               ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  HiFi-GAN Vocoder — ~50M parameters, convolutional          │
  │  Converts audio feature sequence → raw waveform samples     │
  │  Single forward pass; no autoregression; fast               │
  └────────────────────────────┬────────────────────────────────┘
                               ↓
               Audio output: WAV file (Zulu speech)
```

**Total checkpoint: ~1.87 GB** (already somewhat mixed-precision internally).

### 3.2 The VQ-VAE Codebook

The codebook is the pivot between the language model half and the audio half.
It is a simple lookup table — 1024 rows, each a 512-dimensional float vector
representing one unit of audio.

**Why it cannot be quantized:**
The GPT generates token indices (integers). An index is either right or wrong
— there is no "slightly wrong 348" that produces a graceful degradation. If
quantization corrupts the codebook entry at index 348 even slightly, every
time the GPT produces token 348, the vocoder receives a corrupted audio vector.
This produces audible artifacts immediately: clicks, pitch jumps, sudden
changes in timbre. Not subtle quality degradation — a discrete audible error.

**The rule: codebook stays at full FP16, always.**

### 3.3 Why Audio Token Prediction Is More Fragile Than Text Token Prediction

Quantization introduces small errors in the next-token probability distribution.
For text tokens, small errors are graceful: a slightly wrong probability might
favor "happy" over "glad" — a synonym. For audio tokens, small errors are
perceptually catastrophic:

- Token 482 might be a mid-frequency vowel formant
- Token 481 might be a click consonant specific to Zulu
- Token 483 might be silence

One wrong index lookup produces the wrong phoneme entirely. Languages like Zulu
and Kinyarwanda have phonemes (click consonants, tonal distinctions) that map
to specific narrow regions of the codebook. Quantization errors push the
probability mass in ways that collapse these distinctions first.

### 3.4 What This Means for Compression — By Component

| Component | Size | Compression tolerance | Target precision | Reason |
|---|---|---|---|---|
| **GPT-2 transformer** | ~350-450 MB | Moderate | **INT4** | Decoder-only → all Arctos decoder tools transfer directly |
| **HiFi-GAN vocoder** | ~50 MB | High | **INT4** | Convolutional; single-pass; locally operates; no long-range dependencies |
| **Speaker Perceiver** | ~small | Moderate | **INT8** | Speaker identity is perceptually salient; small so cheap to protect |
| **VQ-VAE codebook** | ~8 MB | None | **FP16 always** | Lookup table; one wrong entry = audible artifact |

The GPT component is where the interesting compression and research work lives.
HiFi-GAN is expected to work fine at INT4 — it's similar to compressing a CNN
image model. The codebook is a constant-size non-negotiable.

### 3.5 Quality Metrics for XTTS (Harder Than Text MT)

COMET and chrF++ do not apply to audio. The TTS quality evaluation uses three
signals, all noisier than translation metrics:

- **CER (Character Error Rate):** run the compressed XTTS output through Whisper
  (ASR), transcribe it back to text, compute edit distance against the input.
  Catches unintelligible synthesis but misses naturalness issues.
- **UTMOS:** a neural MOS (Mean Opinion Score) predictor trained on human
  naturalness ratings. Gives a 1–5 automated naturalness score without human
  raters. Catches prosody collapse, robotic artifacts.
- **Speaker similarity:** run both reference audio and synthesized audio through
  a speaker verification model (SpeechBrain x-vectors). Checks whether the
  compressed model still sounds like the target speaker.

### 3.6 Compression Landscape for XTTS

| Method | Status | Notes |
|---|---|---|
| **bitsandbytes INT8/INT4 on GPT** | Technically works, untested | Community reports; no published quality evaluation |
| PyTorch INT8 on HiFi-GAN | Should work | Convolutional; standard quantization; not yet measured |
| **SPADE-style pruning + KD** | Research template (KAIST, 2026) | WER-based layer importance → depth pruning → multi-level KD; not applied to XTTS specifically |
| GGUF | Not available | llama.cpp doesn't support XTTS |

**Zero published baselines exist for XTTS quantization.** The GitHub issue
requesting quantization support was closed as "won't fix." Any measured result
is the first published result.

---

## 4. How Quantized Models Actually Execute on GPU

`learning/project_foundations.md` explains what quantization is conceptually. This
section answers the follow-up: given that a model has some layers at INT4
and others at INT8, what is the GPU literally computing?

### 4.1 The Memory Bandwidth Bottleneck

LLM inference at small batch sizes is **memory-bandwidth-bound**, not
compute-bound. At batch=1, generating one token means loading all model
weights from VRAM into registers, doing one forward pass, and discarding
most intermediate results.

For a 7B model at FP16: ~14 GB of weight reads per token. On an A100
(2 TB/s memory bandwidth), that's ~7 ms of pure memory load — often longer
than the actual matrix multiply. The math isn't the bottleneck; the data
movement is.

**Consequence:** reducing weight dtype from FP16 to INT4 cuts memory reads
by 4×. That is where most of the inference speedup at small batch sizes
comes from — not faster arithmetic.

### 4.2 Weight-Only Quantization (W4A16, W3A16, W2A16)

The most common production format. Weights are stored at INT4 (or packed
3/2-bit). Activations — the per-token vectors flowing through the network —
remain at FP16. Here is what the GPU actually executes per layer:

```
1. Load INT4 weight block from VRAM    ← 4× less data than FP16
2. Dequantize on-chip:
      w_fp16 = w_int4 × scale          ← cheap multiply in CUDA registers
3. Run the matmul in FP16:
      out = w_fp16 @ x_fp16            ← standard FP16 GEMM
4. Repeat for next tile
```

The matmul is still in FP16. No INT4 arithmetic happens. The GPU is not
doing integer multiplication — it is dequantizing to FP16 first. The speedup
is purely from loading fewer bytes from VRAM.

The **Marlin kernel** pipelines steps 1 and 2 so that the next tile is
dequantizing while the current tile is computing — it hides the dequantize
latency entirely. This is why AWQ-Marlin achieves ~2.5-3.5× speedup over FP16.

### 4.3 Weight + Activation Quantization (W8A8)

Here both weights and activations are quantized. The GPU runs the matmul
on INT8 tensor cores with INT32 accumulation:

```
Per layer, per token (online):
   x_int8 = round(x_fp16 / scale_x)   ← quantize activation
   matmul: INT8 @ INT8 → INT32         ← runs on INT8 tensor cores
   out_fp16 = result × scale_x × scale_w ← dequantize output
```

**What you gain:** both memory AND compute. INT8 tensor cores on A100 deliver
roughly 2× the throughput of FP16. This helps even at large batch sizes where
the GPU is compute-bound — unlike W4A16 which only helps when you're
memory-bandwidth-bound.

**The activation outlier problem:** LLM activations have a handful of channels
with values 10-100× larger than the rest. One outlier channel blows the per-token
INT8 scale and maps all normal values to near-zero. Three solutions:

| Approach | How it works |
|---|---|
| **LLM.int8()** | Detect outlier channels and route them through a separate FP16 path. Mostly kills the throughput gain — mainly useful for memory savings at INT8. |
| **SmoothQuant** | Offline: divide outlier activation channels by a scale factor; multiply corresponding weight row by the inverse. Smooth activations → INT8-quantizable. Zero runtime cost. |
| **QuaRot / SpinQuant** | Apply a random Hadamard rotation to all weight matrices (fused offline, zero runtime cost). This spreads activation energy evenly across all channels so no single one dominates. Enables W4A4. |

### 4.4 Mixed Precision in Practice

"Some layers at INT8, others at INT4" is simpler than it sounds. Each weight
tensor is its own memory allocation with its own dtype. The dispatch code just
calls the appropriate kernel per layer. Memory savings are additive.

There is no "you can't multiply INT4 by FP16" problem because weight-only
quantization always dequantizes to FP16 before the matmul anyway. Mixed
precision at layer granularity is essentially free to implement.

**Keeping individual scalar weights in FP16 (super-weight preservation):**
Store a sparse FP16 matrix alongside the INT4 weight tensor. At inference:

```
out = dequant_matmul(W_int4, x)  +  sparse_matmul(W_fp16_sparse, x)
```

If there are only ~10 protected scalars, the sparse term costs nearly nothing.
This is how SqueezeLLM implements its "dense + sparse" decomposition, and
exactly what our `compress.py` does for super-weight FP16 preservation.

### 4.5 What Hardware Actually Supports

| GPU | FP16 | INT8 GEMM | INT4 GEMM | FP8 | Notes |
|---|---|---|---|---|---|
| **A100** | ✅ | ✅ ~2× FP16 | ✗ (software only) | ✗ | Our training GPU |
| **T4** | ✅ | ✅ ~4× FP16 | ✅ native | ✗ | Target deployment GPU (16 GB) |
| **H100** | ✅ | ✅ | ✅ | ✅ | Not in scope |

**Key for the project:** A100 does not have native INT4 tensor cores, so
W4A16 (memory savings only, matmul stays FP16) is the primary lever on A100.
T4 has native INT4 tensor cores, meaning W4A4 quantization (if activation
quantization is solved) could give real compute speedup — not just memory
savings — on the actual deployment target.

### 4.6 The Three Common Library Formats

| Library | Storage | Inference kernel | Best use |
|---|---|---|---|
| **bitsandbytes NF4** | 4-bit NF4 (non-uniform) | Slow BnB custom kernel | Training (QLoRA); not deployment |
| **GPTQ** | INT4, group=128, per-channel scale | Marlin kernel (fast) or ExLlamaV2 | Quality-focused deployment |
| **AWQ** | INT4, group=128, activation-scaled | Marlin-AWQ (fastest) | Throughput-focused deployment |

---

## 5. AWQ in Depth

AWQ is both the most-deployed W4 format in production and the direct basis of
the salient-channel calibration experiments in our `salient_channels.py`.

### The Core Mechanism

AWQ observes that weight quantization error is not uniform across input channels.
Channels that receive large activations amplify weight error into output error:

```
Output error from quantizing column j ≈ (quantization error on w_j) × ‖x_j‖
```

Channels with large `‖x_j‖` are "salient" — their quantization error matters most.

**What AWQ does:**
1. Run ~128 calibration samples through the model (forward pass only, no gradients).
   Compute per-channel `h(x) = mean(‖x_j‖)`.
2. Find the top 1% of channels by `h(x)`.
3. Before quantization, scale those channels up by `s = h(x)^0.5`. After
   quantization, scale the corresponding output dimension back down by `s`.
   The quantizer now sees a more uniform value range on salient channels and
   allocates its 16 buckets more fairly.

**What AWQ does NOT do:** it does not change bit-width per channel. It is a
rounding improvement, not mixed precision. The effective quality improvement is
approximately equivalent to giving salient channels ~1 extra bit.

**AWQ vs GPTQ:**

| Bit-width | Quality comparison | Notes |
|---|---|---|
| 4-bit | Nearly identical | AWQ better on instruction-tuned models (GPTQ can overfit calibration) |
| 3-bit | GPTQ slightly better | OBS reconstruction is more principled at 3-bit |
| 2-bit | GPTQ clearly better | AWQ's no-gradient approach shows weakness at extreme compression |

**Why AWQ is the production default:** `autoawq` quantizes a 7B model in ~5 minutes
vs ~4 hours for GPTQ. The Marlin-AWQ kernel is the fastest W4 path on A100+vLLM.
Most deployed quantized models on HuggingFace are AWQ format.

**Our extension in `salient_channels.py`:** we compute AWQ salient-channel detection
under three calibration regimes — MT-formatted prompt / raw LRL source / raw LRL target —
and measure Jaccard overlap and Spearman rank correlation of the top-1% salient sets.
If MT calibration picks different channels than generic calibration, MT-conditional AWQ
is a real research lever. The Q6 canary: MT calibration shifts the salient set most
at the lowest bit-widths, where it matters most.

---

## 6. Papers to Read — With Links

### Must-Read First (before any experiments)

| Paper | What it gives you | Link |
|---|---|---|
| XTTS v2 (Interspeech 2024) | Ground truth on XTTS architecture — read before touching the model | https://arxiv.org/abs/2406.04904 |
| DecoderLens (NAACL 2024) | How to measure which encoder layers LRL needs — the first NLLB experiment to run | https://arxiv.org/abs/2310.03686 |
| Attention sinks in NLLB-200 (2026) | 83-91% cross-attention mass on language tag tokens; content-routing vs sink-serving heads | https://arxiv.org/abs/2605.01229 |
| NLLB original (Meta 2022) | Architecture spec, online KD methodology, LRL evaluation setup | https://arxiv.org/abs/2207.04672 |
| The Super Weight in LLMs (Yu 2024) | Single scalar weight can collapse a model; causal-KL detection; FP16 preservation | https://arxiv.org/abs/2411.07191 |
| Uneven Impact of PTQ in MT (2025) | The paper we replicated; 55-lang PTQ, LRL + 2-bit nexus, calibration effects | https://arxiv.org/abs/2508.20893 |
| GPTQ (Frantar 2023) | Hessian-weighted rounding; the math is OBS / second-order Taylor | https://arxiv.org/abs/2210.17323 |
| AWQ (Lin 2024) | Activation-magnitude channel scaling; production W4 default | https://arxiv.org/abs/2306.00978 |

---

### NLLB-Specific Papers

| Paper | What it gives you | Link |
|---|---|---|
| Memory-efficient NLLB / MoE pruning (ACL 2023) | Decoder LRL fragility (2× quality hit vs HRL); MoE expert specificity | https://arxiv.org/abs/2212.09811 |
| NASH pruning (EMNLP 2023) | Encoder-width + decoder-depth as 2D problem; cross-attention protected last | https://arxiv.org/abs/2310.10054 |
| Aespa (NeurIPS 2024) | Why GPTQ-style layer-wise reconstruction fails in coupled (enc-dec) architectures | https://arxiv.org/abs/2402.08958 |
| CAR-SAM (CVPR 2026) | Cross-attention PTQ failure modes in vision model — transfers structurally to NLLB | https://arxiv.org/abs/2605.16901 |
| AfriNLLB (AfricaNLP 2026) | Iterative pruning + distillation for African language pairs on NLLB-600M | https://arxiv.org/abs/2602.09373 |

---

### XTTS-Specific Papers

| Paper | What it gives you | Link |
|---|---|---|
| SPADE (KAIST/42dot, 2026) | Only published structured pruning + KD paper for autoregressive LLM-TTS; WER importance → depth prune → multi-level KD | https://arxiv.org/abs/2509.20802 |
| UTMOS (Interspeech 2022) | Neural MOS predictor — automated TTS naturalness score, no human raters | https://arxiv.org/abs/2204.02152 |

---

### MT / Multilingual Quantization

| Paper | What it gives you | Link |
|---|---|---|
| Uneven Impact of PTQ in MT (2025) | 55-lang MT PTQ; calibration helps at 2-bit LRL; no GPTQ tested → our gap | https://arxiv.org/abs/2508.20893 |
| Calibrating Beyond English (Chimoto 2026) | Multilingual GPTQ with language-matched calibration — perplexity only, no MT quality | https://arxiv.org/abs/2601.18306 |
| How Quantization Affects Multilingual LLMs (Marchisio 2024) | 1.7% automatic drop = 16% human drop; metrics understate LRL damage | https://arxiv.org/abs/2407.03211 |
| Impact of Calibration Data in PTQ + Pruning (Williams & Aletras 2024) | Calibration matters MORE for pruning than quantization — critical framing | https://arxiv.org/abs/2311.09755 |
| Calibration diminishing effect on modern LLMs (2024) | Standard-bit modern LLMs are robust to calibration choice — explains AWQ null at W4 | https://arxiv.org/abs/2405.20835 |
| WMT25 Model Compression Shared Task (Gaido 2025) | The benchmark we target; current Pareto frontier | https://aclanthology.org/2025.wmt-1.25/ |

---

### Quantization Methods

| Paper | Key idea | Link |
|---|---|---|
| GPTQ | Inverse-Hessian weighted rounding per-layer | https://arxiv.org/abs/2210.17323 |
| AWQ | Activation-magnitude channel scaling | https://arxiv.org/abs/2306.00978 |
| LeanQuant | Loss-error-aware non-uniform grid; WMT25 quality winner | https://arxiv.org/abs/2407.10032 |
| SqueezeLLM | Hessian k-means codebook + dense-and-sparse FP16 outlier split | https://arxiv.org/abs/2306.07629 |
| OmniQuant | Learnable clipping + equivalent transformation; W4A4 support | https://arxiv.org/abs/2308.13137 |
| SmoothQuant | Migrate activation outliers to weights offline → W8A8 works | https://arxiv.org/abs/2211.10438 |
| QuIP | Random orthogonal transform → incoherence → better quantization (origin of the rotation idea) | https://arxiv.org/abs/2307.13304 |
| QuaRot | Randomized Hadamard rotation; eliminates activation outliers; W4A4KV4 | https://arxiv.org/abs/2404.00456 |
| SpinQuant | Like QuaRot but learned rotations; better on Llama-3 class | https://arxiv.org/abs/2405.16406 |
| AQLM | Additive multi-codebook quantization; Pareto-optimal at 2-bit | https://arxiv.org/abs/2401.06118 |
| QuIP# | Hadamard incoherence + E₈-lattice codebook; 2-3 bit | https://arxiv.org/abs/2402.04396 |
| HQQ | Half-quadratic splitting; zero calibration; ~50× faster than GPTQ | https://mobiusml.github.io/hqq_blog/ |
| CoopQ | Why isolated per-layer mixed-precision allocation fails below 4-bit | https://arxiv.org/abs/2509.15455 |
| HAWQ-V2 | Hessian-trace mixed-precision allocation; foundational prior for our Fisher allocator | https://arxiv.org/abs/1911.03852 |

---

### Super Weights / Outliers

| Paper | Key idea | Link |
|---|---|---|
| The Super Weight in LLMs (Yu 2024) | Causal-KL super-weight detection; FP16 preservation recovers quality | https://arxiv.org/abs/2411.07191 |
| Massive Activations (Sun 2024) | Activation-side view; fixed-value tokens acting as structural bias | https://arxiv.org/abs/2402.17762 |
| LLM.int8() (Dettmers 2022) | Emergent outlier features at scale; why naive W8A8 fails | https://arxiv.org/abs/2208.07339 |

---

### Extreme Low-Bit (Ternary and Binary)

| Paper | Key idea | Link |
|---|---|---|
| BiLLM | 1-bit PTQ; salient/non-salient split; closest prior for our super-weight preservation (English only) | https://arxiv.org/abs/2402.04291 |
| PB-LLM | Partial binarization; magnitude-salient weights at higher precision | https://arxiv.org/abs/2310.00034 |
| PTQTP (2025) | PTQ to trit-planes (ternary); better than binary | https://arxiv.org/abs/2509.16989 |
| PT2-LLM (2025) | Post-training ternarization; ternary fits unimodal weight distribution | https://arxiv.org/abs/2510.03267 |
| BitNet b1.58 | 1.58-bit trained from scratch; NOT PTQ | https://arxiv.org/abs/2402.17764 |

---

### Multilingual Mechanism (Interpretability Anchors)

| Paper | Key idea | Link |
|---|---|---|
| LAPE / Language-Specific Neurons (Tang 2024) | Language neurons in top+bottom layers; shared language-neutral middle | https://arxiv.org/abs/2402.16438 |
| Do Llamas Work in English? (Wendler 2024) | Models translate via an intermediate English-like latent space | https://arxiv.org/abs/2402.10588 |
| Middle-layer cross-lingual alignment (2025) | Middle layers form the strongest shared cross-lingual space | https://arxiv.org/abs/2502.14830 |

---

### Recovery / Healing After Quantization or Pruning

| Paper | Key idea | Link |
|---|---|---|
| QLoRA | Train LoRA adapters through 4-bit NF4 frozen base; standard healing recipe | https://arxiv.org/abs/2305.14314 |
| LoftQ | Joint quantized-base + LoRA init; better than QLoRA at 2-3 bit | https://arxiv.org/abs/2310.08659 |
| BitDistiller | Self-distillation QAT at 3-bit; beats GPTQ/AWQ at 3-bit | https://arxiv.org/abs/2402.10631 |
| GKD | On-policy distillation; WMT25 Vicomtech used this | https://arxiv.org/abs/2306.13649 |
| ALMA | Large parallel data HURTS; monolingual CPT → small parallel SFT is the right recipe | https://arxiv.org/abs/2309.11674 |
| Tower / TowerInstruct | Mixed monolingual+bilingual CPT; best MT recovery recipe | https://arxiv.org/abs/2402.17733 |

---

### KV Cache

| Paper | Key idea | Link |
|---|---|---|
| KIVI (2024) | INT2 KV cache; zero fine-tuning; ~3× KV memory reduction at large batch | https://arxiv.org/abs/2402.02750 |
| KVQuant (2024) | Sub-4-bit KV with outlier handling; enables 1M+ context | https://arxiv.org/abs/2401.18079 |

---

*Written 2026-06-20. Update after experiments run.*
