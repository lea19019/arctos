# Phase-two synthesis — what holds, what doesn't

Consolidated honest read across all phase-two experiments (gem, extreme, keep,
pipeline, calib, alloc) + two deep-research passes. chrF++/XCOMET-XL, n=24–32,
greedy, generic prompt → **directional**. Supersedes scattered claims in
earlier docs where they conflict.

## ✅ The robust result: MT-conditional GPTQ
Calibrating **GPTQ on translation data** (vs generic text) recovers much of the
3-bit cliff; generic-calibrated GPTQ is *worse than no quantization*. W3 cs-de,
all 6 models positive (MT − generic: +20..+32 chrF, +0.13..+0.52 COMET).
EuroLLM absolute: RTN 0.587 / GPTQ-MT **0.797** / GPTQ-generic 0.273 (COMET).
**Why robust + novel:** GPTQ reconstructs *on* the calibration activations (so
domain matters), unlike AWQ (scale only). The closest prior work tested AWQ/BnB/
GGUF/AutoRound but **not GPTQ** (2508.20893) or measured only perplexity
(Chimoto 2601.18306). This is the contribution.

## ✅ Secondary: salient-channel FP16 preservation recovers W3
Keeping top-1% salient channels in FP16 lifts W3 (Gemma 12.7→48.4, Llama
32→42, EuroLLM 28→40 chrF). Independent of GPTQ-MT; cheap; kernel-friendly.
Super-weight strength is multilingual-model-varying (EuroLLM KL 3.28 ≫ Gemma ~0)
— an unstudied phenomenon (all prior super-weight work is English-only).

## ❌ The depth-pipeline does NOT give a compression rule (honest negative)
The marquee "protect the language-specific endpoints, crush the language-neutral
middle" hypothesis **fails as a robust rule.** At matched budget (W3, half
FP16/half W3), crush_middle vs crush_ends is a **wash**: 3 models favor middle
(tower-plus/instruct/base), 3 favor ends (eurollm/llama/bloom), aya ties. The
translation *pipeline* is real (phase one), but **stage/depth does not localize
quantization fragility** — this is the Q5 null (importance ⟂ sensitivity)
reconfirmed one level up, at stage granularity. Report as a negative; it
*strengthens* the GPTQ story by ruling out the obvious depth-based alternative.

## ❌ Other confirmed negatives
- **Fisher mixed-precision < uniform** (proxy signal; isolated per-layer metrics
  fail sub-4-bit, cf. CoopQ 2509.15455).
- **MT calibration doesn't help AWQ** (only GPTQ) — AWQ uses calibration only
  for a per-channel scale.
- **Sub-2-bit (ternary/binary) collapses for all**; salient preservation cannot
  rescue it. Consistent with: sub-2-bit needs from-scratch QAT (BitNet), not PTQ.

## Reframed goal (from the healing-free deep-research, wco17ovot)
**No healing-free PTQ reaches FP16 at 3-bit** (Llama-2-7B WikiText2: FP16 5.47 →
best pure-PTQ ~5.83; GPTQ 8.37; 2-bit far for all). So "reach baseline at 3-bit,
no healing" is **not achievable by any known method** — our contribution is *not*
"close the gap to zero." It is: **the best healing-free MT-specific option at a
given size.** Strongest pure-PTQ bases to build on: codebook/VQ (GPTVQ) and
loss-aware grid (LeanQuant); rotation (QuaRot/SpinQuant) targets W4A4, not W3
weight-only. None has been evaluated on MT / WMT25 / XCOMET-XL → still open.

## The defensible thesis (post-consolidation)
**"MT-conditional GPTQ + salient/super-weight preservation: the best
healing-free 3-bit compression for translation, and a demonstration that the
translation depth-pipeline explains MT but does not localize quantization
fragility (so bit allocation needs a sensitivity-native, MT-calibrated signal,
not a depth prior)."** Positive contribution + a clean, well-supported negative.

## Caveats (carry forward)
n small, greedy, generic prompt; XCOMET-XL unreliable on degenerate sub-2-bit
output (scores garbage ~0.2–0.65); automatic metrics understate low-bit damage
~10× vs human (Marchisio EMNLP'24). Decisive version: larger n, chat templates,
human/stress check at extreme bits.

## Pointers
GPTQ gem + tables: `phase2_results.md`. Method/novelty + reading list:
`compression/docs/phase2_method_primer.md`, `compression/docs/compression_primer.md`. Pipeline direction (now a
negative): `phase2_novel_direction.md`. Deep-research raw:
`compression/docs/deep_research_raw/`. Replication of 2508.20893 (separate effort):
`docs/archive/replication_uneven_ptq_brief.md`.
