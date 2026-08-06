# Arctos — advisor briefing

A talking-document for explaining the project. Honest about what's solid vs
preliminary.

## One-paragraph pitch
Arctos studies **how machine translation happens inside open multilingual LLMs**
and uses that understanding to build a **compression method for translation
models**. Phase one (done) mapped the internal mechanism of translation across 8
models. Phase two (in progress) turns that into a post-training compression
method aimed at the **WMT25 Model Compression Shared Task**. The headline phase-two
finding: **for low-bit translation, the quantizer must be calibrated on
translation data — calibrating GPTQ on generic text is *worse than not
quantizing at all*, while MT-calibrated GPTQ recovers most of the quality at
3-bit.** This is a regime the closest prior papers left untested.

## The problem
LLM translation models are big and slow; compressing them (fewer bits per
weight) is the goal of the WMT25 compression task (evaluated with XCOMET-XL).
The leaderboard has a quality-preserving corner and an extreme-compression
corner, with a **gap in the middle** (moderate compression / moderate quality)
and — importantly — the methods used calibrate on **generic web text**, not
translation data. That gap is where we aim.

## Phase one (complete): understand before compressing
Using interpretability methods (logit lens, probing, Information Flow Routes,
direct logit attribution, attribution patching, a language-pivot analysis) on 8
models, we showed translation is a **depth-staged pipeline**: understand the
source early → process in a language-neutral/pivot space in the middle → commit
to the target language only in the final layers. This generalizes across
architectures, with **Gemma-family the consistent exception**. A key *negative*
result (Q5): **where the MT computation concentrates does NOT predict where
quantization precision matters** — so you can't allocate bits by
"importance." That negative result is what forced phase two to use
*sensitivity-native* signals instead.

## Phase two (in progress): the compression method
We built a sandbox with four operations — **find** (locate the precision-fragile
weights), **keep** (protect them), **shrink** (quantize the rest), **prune**
(remove redundant weights) — and tested several "levers," all measured by
translation quality (chrF++ and XCOMET-XL, the WMT25 metric), at bit-widths from
4 down to **ternary (1.58-bit) and binary (1-bit)**.

### The headline result (the contribution)
**MT-conditional GPTQ.** GPTQ is a quantizer that minimizes reconstruction error
on a small *calibration* set, so it's very sensitive to what that set is. We
compared calibrating on **translation data** vs **generic text**. At 3-bit, on
EuroLLM (XCOMET-XL, 0–1):
- No calibration (round-to-nearest): **0.587**
- GPTQ + **MT** calibration: **0.797** (recovers the cliff)
- GPTQ + **generic** calibration: **0.273** (worse than doing nothing)

Same direction on Llama-3.1 and Tower-Instruct. **Why it's novel:** the closest
MT-quantization paper (arXiv:2508.20893) tested four quantizers but *not* GPTQ;
the one multilingual-GPTQ paper (Chimoto et al., EACL 2026) measured only
*perplexity*, never translation quality. So "MT-quality-conditional GPTQ at the
low-bit cliff" is an open, defensible contribution.

### Supporting findings
- **Salient-weight protection** independently recovers the 3-bit cliff: keeping
  ~1% of weights (the "salient channels" / "super weights") in full precision
  lifts quality a lot (e.g. Gemma 12.7→48.4 chrF++ at 3-bit).
- **Super weights are multilingual-model-varying** — a handful of scalar weights
  whose removal is catastrophic; strong in EuroLLM/Tower, near-absent in Gemma.
  Prior super-weight work is *English-only* — nobody has mapped this across
  multilingual MT models, which is a second open gap.
- **Honest negatives** (these matter to an advisor): MT calibration does *not*
  help AWQ-style quantization (only GPTQ); a Fisher/Hessian mixed-precision
  allocator *underperformed* uniform precision; and automatic metrics
  (incl. COMET) understate damage at very low bits, so we treat sub-2-bit
  numbers cautiously.

## Why this is a thesis, not just engineering
1. It's **grounded** (phase-one interpretability explains *why* late layers and
   a few weights are fragile).
2. The headline claim is **novel and verified against the literature** (we ran a
   structured deep-research pass to confirm the gap).
3. It targets a **real benchmark** (WMT25) with the **official metric**
   (XCOMET-XL).
4. We are **independently replicating** the closest prior paper (a preprint)
   before building on it — good scientific hygiene.

## Status / caveats (be upfront)
- The gem is confirmed on **3 of 8 models** so far (full sweep running); numbers
  are **n=24–32 sentences, greedy decoding, generic prompt** — directional, not
  final. The decisive version needs larger n + each model's chat template + a
  small human/stress check at extreme bits.
- Cluster has intermittent transient GPU faults (handled by resubmitting; not a
  result issue).

## Questions worth asking the advisor
1. Is "MT-conditional GPTQ + salient preservation, evaluated on XCOMET-XL,
   focused on the low-bit cliff and low-resource pairs" a strong enough single
   contribution, or should we also push the recovery/healing angle?
2. How much does the **mechanistic explanation** (phase one) need to be tied in
   for it to count as a contribution vs. a pure empirical quantization paper?
3. Scope of evaluation: how many language pairs / models for a credible claim?

## Pointers (in the repo)
- Phase-one report: `compression/report/arctos-translation-report.pdf`
- Method primer (the gem + literature gaps): `docs/findings/phase2-method-primer.md`
- Framework + reading list: `docs/findings/compression-primer.md`
- Phase-two results: `docs/findings/q6.md`; collectors `compression/scripts/q6gem_collect.py`
- Replication plan for the closest prior paper: `docs/replication-uneven-ptq-mt-brief.md`
