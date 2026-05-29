# Q4 — The MT footprint across architectures (the shared-depth hypothesis)

> Synthesis across Q1 (where language emerges), Q3 (layer importance), and
> Q5 (sensitivity), over 8 models spanning 2022→2025, four lineages, two
> normalizations, two positional encodings, and — with NLLB — both
> decoder-only and encoder-decoder architectures.

## The hypothesis, restated

PHASE1-PLAN posed three claim strengths:
- **V1** (trivial): first ~25% of layers MT-irrelevant, last 1–2 protected,
  middle is where MT happens. Task-agnostic; not novel.
- **V2** (novel if true): a characteristic depth signature — source
  understanding → language-agnostic semantics → target commitment — at
  similar relative depths across architectures.
- **V3** (highest payoff): the signature is consistent enough to drive
  bit-allocation / pruning from depth fraction alone, without per-model
  interpretability.

## What the 8 models show

### V2 — SUPPORTED for the IFR depth signature

The IFR layer-importance profile (share of contribution magnitude by depth
quarter) is remarkably consistent:

| model | family | first-Q | mid | last-Q |
|-------|--------|---------|-----|--------|
| Aya 8B | Cohere | 7-8% | 35-39% | 53-58% |
| TowerBase 7B | Llama-2 | 5-6% | 45-52% | 43-49% |
| TowerInstruct 7B | Llama-2 + SFT | 5-6% | 44-52% | 42-51% |
| BLOOM 7B1 | ALiBi+LayerNorm | ~12% | ~43% | ~45% |
| EuroLLM 9B | Llama-3 | ~3% | ~45% | ~53% |
| Llama-3.1 8B | Llama-3 | ~5-8% | ~45-50% | ~45-50% |
| **NLLB 3.3B** | **enc-dec** | **~11%** | **~44%** | **~45%** |
| Tower-Plus 9B | **Gemma 2** | **~30%** | ~52% | **~18%** |

Six decoder-only models **and the NLLB encoder-decoder** all land in the
same band: first quarter ≤12%, last quarter ~45-58%, top layers in the
final ~20% of depth. That this holds across RoPE/ALiBi, RMSNorm/LayerNorm,
2022→2024, four lineages, AND across the decoder-only ↔ encoder-decoder
divide is strong evidence for V2: there is a characteristic MT depth
signature that generalizes.

**The one boundary: Gemma-family.** Tower-Plus (Gemma 2 base) inverts the
profile (~30% first-quarter, ~18% last-quarter). Gemma's extra pre/post-
feedforward layernorms + logit softcapping rescale residual magnitudes at
every layer. So V2 generalizes across a wide architecture space *except*
Gemma-family — a boundary the method must respect.

### Source-language emergence (Q1 probing)

Source-language ID is decodable from every layer at 0.40-0.55 selectivity
in all models. Peak location splits by training recipe — embedding (L0) for
heavily multilingually-trained models (TowerBase/Instruct, BLOOM,
Llama-3.1), mid-network for general-LM-style (Aya L17, EuroLLM L9). The
split is real but small in magnitude; the practical takeaway is uniform:
embedding-layer features carry language information everywhere.

### V3 — NOT SUPPORTED

V3 requires that the depth signature predict where precision matters, so
bit allocation could skip per-model interpretability. **Q5 falsifies this.**
Importance (IFR/DLA, last-quarter-concentrated) does *not* predict
quantization sensitivity (mean ρ ≈ −0.05 across both the per-head logit and
per-layer chrF++ experiments). The depth signature tells you where MT
*computation concentrates*, not where *precision is fragile* — and per-layer
chrF++ fragility varies model-to-model with no universal depth rule.

## Landing on the claim strengths

- **V1: trivially true** (first quarter is light everywhere).
- **V2: supported and stronger than expected** — the depth signature
  generalizes across lineage, normalization, positional encoding,
  generation, and even the decoder-only ↔ encoder-decoder divide; the
  sole exception is Gemma-family.
- **V3: falsified** — depth-fraction / importance does not determine
  quantization sensitivity. Bit allocation needs a sensitivity-native
  signal, measured per model.

## Honest falsifying evidence (required by PHASE1-PLAN)

- The prior paper's own result (IFR layer rankings differing between cs→de
  and en→es within Aya) reappears here as per-pair variance — the depth
  *signature* generalizes but exact per-layer rankings do not.
- Gemma-family breaks V2's magnitude profile outright.
- Q5's null is the clean falsification of V3, reported in full in
  `docs/findings/q5.md` rather than papered over.

## Bearing on the phase-two method

The defensible design that follows from Q1–Q5:
1. Use the **depth signature** (V2) as a coarse, model-agnostic prior for
   *where to look*, valid for Llama/BLOOM/Cohere/enc-dec but NOT Gemma-family.
2. Allocate bits with a **sensitivity-native signal** (per-component noise
   probing / Hessian / AWQ on MT calibration data) — because importance
   does not predict sensitivity (V3 false).
3. Treat Gemma-family models as a separate regime.
