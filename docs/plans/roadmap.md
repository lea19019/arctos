# Roadmap — the sweet spot of compression for translation

Where this project is going: from "one quantization result" to an **ambitious,
multi-dimensional study of compression *specifically for machine translation*,**
plus the ML/engineering skills to own it end-to-end. ~6-month horizon, officially
starting in the fall but live now.

## The thesis question (advisor-shaped)
> **Where is the sweet spot of compression for the translation task, and how does
> it move across the dimensions we can vary?** Map the quality↔size↔speed
> frontier for MT across compression *methods* and their *knobs*, and identify
> what is genuinely MT-specific.

This generalizes the current finding (MT-conditional GPTQ) into a **frontier
study**, not a single method.

## Dimensions to vary (the experimental matrix)
A proper study sweeps a grid and reports Pareto frontiers, not point results:
- **Compression method:** quantization (GPTQ / AWQ / GPTVQ / LeanQuant /
  rotation), **pruning** (Wanda / SparseGPT / structured layer-drop), 
  **distillation** (seq-KD / on-policy GKD / DistiLLM), and **combinations**
  (prune→quant, distill→quant).
- **Bit-scale / ratio:** 8→4→3→2→1.58(ternary)→1(binary) for quant; 25/50/75 %
  for prune; student sizes for distill.
- **What gets compressed:** which weight types (attn vs MLP, q/k/v/o vs
  gate/up/down), which depth stages (early/middle/late — the translation
  pipeline), per-channel vs per-tensor, salient vs bulk.
- **Calibration / data:** MT-parallel vs generic vs language-matched vs
  self-generated; size and domain.
- **Model & language:** the 8-model set × language pairs × resource tiers ×
  scripts (high-resource same-script → low-resource divergent-script).
- **Eval axis:** XCOMET-XL (primary, WMT25), MetricX-24, chrF++, COMET-KIWI
  (reference-free), + on-disk GB + tokens/sec + a human/stress check.

Output: per-method **Pareto plots** (quality vs size, quality vs tok/s) and a
clear answer to "what's the best compression per budget *for translation*."

## Anchored by what we already know
- **Use:** MT-conditional GPTQ (the win); salient/super-weight FP16 preservation;
  the strongest healing-free bases (GPTVQ, LeanQuant) from the deep-research.
- **Guardrails (don't repeat):** depth/stage does NOT localize quant fragility
  (Q5 + pipeline negative); Fisher mixed-precision < uniform; no healing-free PTQ
  hits FP16 at 3-bit; automatic metrics understate low-bit damage.
- **Open & MT-specific to push:** does *distillation* (allowed if we drop the
  no-healing constraint) reach baseline where PTQ can't? Does MT-specific
  calibration help *pruning* and *codebook VQ* (not just GPTQ)?

## Statistical rigor (a first-class goal)
Make the measurement trustworthy — the part to *settle down*:
- Report **confidence intervals / bootstrap** on COMET deltas; significance
  tests (paired bootstrap, the WMT-standard) before claiming a method "wins."
- Adequate **n** (hundreds–thousands of sentences), proper **decoding** (beam +
  chat templates), variance across seeds.
- Multiple metrics + at least a small **human eval** at aggressive settings
  (metrics lie at low bit). Pre-register the comparison grid to avoid cherry-pick.

## Data axis
- Lab MT data (large) → in-domain calibration + evaluation beyond FLORES.
- **Speech ↔ text:** extend to speech translation / ASR+MT cascades or direct
  speech-to-text MT — a richer modality to compress and a differentiator.

## Skills + engineering track (full-stack ML)
Run *alongside* the science so the 6 months build durable ability:
- **Train & test many models** (the gap to close): fine-tune / distill / heal
  on MT data — not just analyze pretrained ones. Each method above is a training
  rep.
- **Stats/probability for ML eval:** the rigor section above is the curriculum.
- **Cloud / data at scale (AWS day-job synergy):** data pipelines for large MT/
  speech corpora; train/eval on cloud GPUs; **deploy** a compressed MT model
  (quantized + served, tokens/sec measured) — the WMT25 tok/s axis doubles as a
  deployment exercise. Target: an end-to-end "compress → serve → benchmark" path.

## Suggested phasing (6 months, flexible)
1. **Now–month 1:** harden the GPTQ-MT result (proper n + templates + XCOMET-XL +
   significance); finish the 2508.20893 replication. Lock the evaluation harness.
2. **Months 1–3:** the quant × prune × distill grid on the 8-model set; Pareto
   frontiers; the calibration-data study across methods.
3. **Months 3–4:** distillation + recovery (lift the no-healing constraint as a
   separate axis) — can we reach baseline at 3-bit *with* healing, and how cheap?
4. **Months 4–5:** speech↔text extension; cloud training + deployment path.
5. **Months 5–6:** write-up (WMT/EMNLP-shaped), human eval, polish, thesis chapter.

## Honest framing for an advisor / paper
A frontier study + a clean positive (MT-conditional GPTQ) + clean negatives
(depth doesn't localize fragility; no healing-free FP16 at 3-bit) is a *strong*
contribution — it tells the field both what to do and what not to bother with for
translation compression.

*(Captured 2026-06-03 from the post-advisor discussion. Living document — revise
as the study runs.)*
