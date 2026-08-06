# Phase-two method primer — the gem, grounded in the literature

Primer #1 (`compression/docs/compression_primer.md`) gave the find/keep/shrink/prune framework
and reading list. This primer names the **defensible novel contribution** for an
MT-specific compression method, justified by a verified deep-research pass
(2026-06-02; raw synthesis logged below) and our own q6 results.

> Reference, not template: *The Uneven Impact of PTQ in Machine Translation*
> (arXiv:2508.20893, Aug 2025) is the closest prior work and our main empirical
> anchor — but it is a **preprint, not peer-reviewed**, so we cite it carefully
> and are independently replicating it (`docs/archive/replication_uneven_ptq_brief.md`).

---

## The gem (verified primary gap, 3-0 adversarial vote)

**A multilingual / MT-conditional salient-element preservation scheme: locate
the super weights + super-activations + salient channels in multilingual MT
models, rank them by causal KL on MT data, and preserve them in FP16 to recover
the low-bit (2–3 bit) weight-quantization cliff — evaluated with XCOMET-XL.**

Why this is unoccupied (two documented gaps intersect):
1. **No multilingual/MT super-weight study exists.** All super-weight / massive-
   activation work (Yu et al. 2411.07191; Apple; Sun et al. 2402.17762) is on
   **English-centric models** (Llama, Mistral, OLMo, Phi-3) with **English
   benchmarks**. No Aya/Tower/BLOOM/EuroLLM/NLLB; no translation-task analysis.
   Our finding that super-weight strength is **model-varying** (EuroLLM ablation
   KL 3.28, TowerInstruct 1.25, Gemma ≈ 0) is exactly the un-studied phenomenon.
2. **Salient-FP16 recovery has not been shown for aggressive low-bit weight
   quant in MT.** The "preserve super weights → RTN matches SOTA" result is for
   English **W8A8 activation** quant. Our Gemma **12.7 → 48.4** chrF++ recovery
   at 3-bit (top-1% salient channels FP16) is precisely the missing cell.

Differentiators vs everything found: **multilingual** (not English), **MT-quality
metric** (XCOMET-XL, not perplexity/zero-shot), **causal-KL ranking on MT data**
(not activation-spike detection), **low-bit weight** (not W8A8).

## Secondary novelty (weaker prior, target low-bit only)

**MT-quality-conditional GPTQ at 2–3 bit.** The anchor (2508.20893) tested
AWQ/BnB/GGUF/AutoRound but **not GPTQ**; the one multilingual-GPTQ paper
(Chimoto et al., EACL 2026, arXiv:2601.18306) used **perplexity only**, never
XCOMET/MetricX/chrF. So {GPTQ × MT-quality × low-resource} is open. **Caveat:**
the prior on calibration helping *quantization* is weak (4-bit gains are small;
our AWQ null was 9/24) — pursue only at 2–3 bit and expect the win, if any, on
**low-resource / divergent-script** pairs. Our canary already hints at it
(GPTQ-MT − generic on en-arz: chrF++ +4.5 at W4, +6.6 at W3, bloom-7b1, n=4).

## Extreme low-bit (<2-bit): ternary 1.58 + binary 1 (deep-research 2)

We now sweep the full cliff **4 → 3 → 2 → 1.58 (ternary, BitNet-style absmean)
→ 1 (binary, XNOR-style)**. The sub-2-bit regime is where "preserve the salient
minority" matters most — but it is **also where the closest prior art lives, so
we must differentiate hard:**

- **BiLLM (arXiv:2402.04291)** — first 1-bit PTQ; *splits salient vs non-salient*
  and binarizes each separately. **PB-LLM (arXiv:2310.00034)** — keeps salient
  (magnitude/Hessian) higher-precision, binarizes the rest. **ARB-LLM** —
  adaptive 1-bit. These are *structurally our "keep salient, quantize rest"
  idea at 1-bit.* So at extreme bits our mechanism is **not novel**; our
  contribution is the **setting**: multilingual MT, **super-weight causal-KL**
  saliency (vs their Hessian-magnitude), and **XCOMET-XL** MT-quality eval — none
  of which they do.
- **PTQTP (2509.16989), PT2-LLM (2510.03267)** — recent post-training
  ternarization; ternary fits the unimodal LLM weight distribution and beats
  binary, so **1.58-bit is the sub-2 sweet spot** to feature.
- **BitNet b1.58 (Ma 2024) / b1.58-2B4T (2025)** — 1.58-bit but **trained from
  scratch / QAT**, not PTQ → not a drop-in; our ternary is PTQ RTN absmean.
- **VPTQ (EMNLP'24), AQLM, QuIP#** — 2–3-bit vector/codebook quant (stronger but
  kernel-bound).

**Reframed contribution at extreme bits:** not "salient preservation works
sub-2-bit" (BiLLM showed that) but **"in multilingual MT, super-weight + salient
preservation recovers the 1.58/1-bit collapse, and the amount it recovers is
model-varying (tracking our causal-KL super-weight strength), measured by
XCOMET-XL"** — plus the cross-family super-weight map nobody has built. Target
**ternary (1.58)** as the headline low-bit; report binary as the floor.

## What is NOT the gem (don't over-claim)

- **MT-conditional calibration as a general quant lever.** Verified: calibration
  helps **pruning ≫ quantization** (Williams & Aletras ACL 2024; corroborated by
  our 19/24 pruning vs 9/24 quant). So frame calibration as a **pruning** lever,
  not a quant one.
- **Fisher/Hessian mixed-precision allocation.** Our naive 2/4-bit Fisher split
  *underperformed* uniform 3-bit in the canary (cs-de chrF++ −15.8). HAWQ-style
  prior art exists; our task-conditional twist is unconfirmed-novel AND not yet
  working. Keep as exploratory, not headline.

---

## Method skeleton (thesis-chapter shape; experiments, not prose, are the point)

**Claim 1 — Super weights are multilingual-model-varying and MT-quality-critical.**
- Exp: causal-KL super-weight detection across all 8 families (done in q6 find).
  Report location/count/strength + ablation chrF++/XCOMET drop per model.
- Baselines: activation-spike ranking (show causal-KL finds different/better).

**Claim 2 — Salient-element FP16 preservation recovers the low-bit MT cliff.**
- Exp: at W2 and W3, RTN vs {super-weight-FP16, salient-channel-FP16,
  both} — chrF++ AND XCOMET-XL, per pair. (q6 keep, extend to W2.)
- Baselines: AWQ, GPTQ at same bits; the WMT25 anchors.
- Headline target: the Gemma/Tower/EuroLLM cliff recovery, low-resource en-arz.

**Claim 3 — MT-conditional GPTQ helps at low-bit for low-resource (or honest null).**
- Exp: GPTQ with MT-parallel vs generic-XNLI Hessians at W2/W3, per pair,
  XCOMET-XL. (q6 gptq, extend to W2.) If null, report it — it sharpens Claim 2.

**Ablations:** calibration domain (MT vs generic vs self-calib), #channels kept,
causal-KL vs spike vs Fisher ranking, per-model vs per-script protection.

**Evaluation:** XCOMET-XL (WMT25 primary) + MetricX-24 + chrF++ + on-disk GB +
tok/s on A100. **Critical caveat:** automatic metrics understate low-bit MT
damage ~10× vs human eval (Marchisio et al. EMNLP 2024, arXiv:2407.03211) — add
a small human/stress-test check at 2-bit before any strong claim.

---

## Competitor map (flag before claiming priority — fast-moving area)

| Work | What it did | Why we're still novel |
|---|---|---|
| 2508.20893 (Aug 2025, preprint) | PTQ MT, 55 langs; lang-matched calib helps @2-bit low-resource; AWQ/BnB/GGUF/AutoRound | no GPTQ, no super-weight/salient, no mechanism |
| Chimoto 2601.18306 (EACL 2026) | multilingual/lang-matched calib for GPTQ+AWQ | **perplexity only**, no MT quality, no super-weight |
| Yu 2411.07191 / Apple / Sun 2402.17762 | super weights / massive activations, FP16 preservation | **English-only models + benchmarks**, W8A8 not low-bit weight, no MT |
| Williams & Aletras 2311.09755 (ACL'24) | calibration impact: pruning ≫ quant | not MT, not super-weight |
| HAWQ/HAWQ-V2/Q-BERT | Hessian mixed precision | CNN/BERT, not task-conditional, not LLM-MT (lever 3 unconfirmed in our search) |

**Open (our deep-research did not find coverage — needs a dedicated check):**
task-conditional Hessian/Fisher mixed precision at LLM-MT scale (lever 3); the
WMT25 submitted-system XCOMET/MetricX numbers + the 4–6 GB Pareto gap (lever 6).
Re-verify these before relying on them.

## Caveats carried from the deep research
- Automatic MT metrics understate low-bit damage ~10× vs human eval.
- Perplexity gains (Chimoto) may not transfer to XCOMET — our AWQ null suggests
  they don't for quant.
- "Modern LLMs robust to calibration" (2405.20835) is English/standard-bit only;
  calibration regains importance at 2-bit / multilingually.
- This area moves every cycle — re-check arXiv before claiming priority.

## Citations
2508.20893 · 2601.18306 · 2311.09755 · 2405.20835 · 2407.03211 · 2410.17170 ·
2411.07191 · 2402.17762 · 2306.07629 (SqueezeLLM) · HAWQ-V2 1911.03852.
Full verified synthesis (claims + votes + evidence): deep-research run
`wf_36650cc2-1b3` (2026-06-02). Reading list + mechanics: `compression/docs/compression_primer.md`.
