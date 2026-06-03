# Phase-two novel direction: pipeline-aware quantization for MT

The translation-task-specific compression method, grounded in phase-one
interpretability + phase-two findings + the multilingual-mechanism literature.
This is the contribution phase one was built to enable.

## The mechanism (what we and the literature agree on)
Translation in a decoder LLM factorizes by depth into a **language-specific →
language-neutral → language-specific** sandwich:
- **Early layers (bottom):** encode the source into a shared semantic space.
  Language-*specific* input processing. Also where the **super weights** live
  (phase two: EuroLLM L9, Llama L1, Aya L2, BLOOM L0) — the few scalars that
  create the massive activations carrying meaning forward.
- **Middle layers:** a **shared, language-neutral / pivot** representation —
  generic semantics, the same across language pairs (phase-one pivot finding;
  corroborated by LAPE arXiv:2402.16438, middle-layer alignment arXiv:2502.14830,
  shared-neuron compression arXiv:2506.01629).
- **Late layers (top):** **convert** the neutral meaning into target-language
  tokens/script — the late DLA "predictor" heads, the pivot→target crossover,
  the logit-lens target mass (phase one). Language-*specific* output projection.

**The compression blueprint that follows:** the *language-specific* work — the
part translation can't lose — is concentrated at the **two endpoints** (early
super-weights + late conversion circuit), both small. The **middle is
language-neutral, shared, and over-provisioned for any single pair** → it can be
quantized far more aggressively than uniform quantization assumes.

## The proposal: protect the endpoints, crush the middle
> Quantize the language-neutral middle aggressively (down to 2-bit / ternary),
> while preserving the two language-specific endpoints in high precision:
> **(1) the early-layer super weights** (so meaning extraction survives) and
> **(2) the late conversion circuit** (so target-language selection survives).
> Both endpoints are small, so the bit cost is low but they are exactly the
> components translation depends on.

Why it's novel and MT-specific (not generic depth-mixed-precision):
- The protected sets are **mechanistically derived from how translation works**
  (super-weight detection + DLA/pivot conversion-circuit localization), not from
  generic sensitivity. Generic methods protect "outlier-heavy" or "high-norm"
  layers; we protect the **language-conversion machinery**.
- It directly exploits the **language-neutral middle** — a property unique to a
  *cross-lingual* task. For a single-language task there is no neutral pivot to
  exploit.
- It confronts the Q5 null correctly: Q5 found *generic* importance ⟂ *generic*
  sensitivity. We ask whether the **MT-specific endpoints** predict **MT-quality**
  fragility. The decisive test below answers it.

## The decisive experiment (the `pipeline` stage)
Split layers into **ends** (first quarter + last quarter = language-specific) and
**middle** (middle half = language-neutral). At matched bit budget:
- **crush_middle:** middle at low bit, ends FP16  ← the proposal.
- **crush_ends:** ends at low bit, middle FP16  ← control (opposite allocation).
- **uniform_low:** everything at low bit  ← floor.
- **+super-weights:** crush_middle plus explicit early super-weight FP16.

**Hypothesis:** crush_middle ≫ crush_ends at equal budget, and crush_middle
approaches FP16 baseline even as the middle goes to 2-bit/ternary — because the
language-neutral middle is over-provisioned while the endpoints are critical.
Sweep the middle's bit-width (3 → 2 → ternary → binary) to measure *how
aggressively the middle can be crushed* before MT quality drops.

If true: a translation-grounded, healing-free, kernel-friendly mixed-precision
scheme that beats uniform and generic mixed-precision at equal size. If false
(crush_ends ≈ crush_middle): honest negative — depth/stage doesn't localize MT
fragility, and phase one's value stays "understanding only." Either way it's a
clean, publishable result and the right experiment.

## Combine with the healing-free base quantizer
The endpoint-protection is a **task-specific allocation layer** on top of a base
quantizer. The base should be the strongest healing-free PTQ (the separate
deep-research is mapping rotation / codebook / LeanQuant). Stack: rotation/GPTQ-MT
base + endpoint protection (super-weights + conversion circuit) + aggressive
middle. The MT-specific part is the allocation; the base is off-the-shelf.

## The ambitious version (a compression *architecture*)
Because the middle is language-neutral and shared while the endpoints are thin
and language-specific: store **one aggressively-quantized shared trunk +
small per-language-pair high-precision conversion tips.** Serving N pairs =
one low-bit trunk + N small FP16 tips. A multilingual-MT compression architecture
derived from the latent-language structure, not a bit-allocation tweak. (Related
in spirit to NeuronMoE's language-neuron expert allocation, arXiv:2603.05046, but
for quantization + the early/late endpoint split.)

## Honest caveats
- Q5 is a headwind; the `pipeline` experiment is designed to test, not assume.
- "Conversion circuit" first operationalized as last-quarter layers; a finer
  version uses the specific DLA target-predictor heads (phase one) — test both.
- Mixed precision below 4-bit is known-hard with isolated per-layer metrics
  (CoopQ arXiv:2509.15455); the stage-level (not per-layer) split sidesteps that.
- Metrics understate low-bit damage (Marchisio EMNLP'24) — use chrF++ for
  collapse, XCOMET-XL for the quality gap, spot-check translations.

## Literature anchors
- **Tang et al. 2024, Language-Specific Neurons (LAPE)** — arXiv:2402.16438
  (language neurons in top+bottom; shared middle). The key mechanistic support.
- Middle-layer cross-lingual alignment — arXiv:2502.14830.
- Language-specific→shared neurons / compression — arXiv:2506.01629.
- Cross-layer transcoders, multilingual — arXiv:2511.10840.
- Mixed-precision <4-bit needs inter-layer view — CoopQ arXiv:2509.15455;
  channel-wise MP arXiv:2410.13056.
- Phase one (pivot, depth pipeline, DLA heads): `report/`, `docs/findings/q1.md`,
  `architecture-comparison.md`. Super weights: `phase2-method-primer.md`.
