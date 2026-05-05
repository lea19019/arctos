# Contribution-Guided Compression of Decoder-Only LLMs for Machine Translation: An Annotated Bibliography and Synthesis

This document surveys methods relevant to a four-stage compression pipeline — (1) score component contribution, (2) structurally prune, (3) recovery fine-tune, (4) quantize — evaluated on MT via the WMT 2025 Model Compression Shared Task ([Gaido et al., 2025](https://aclanthology.org/2025.wmt-1.25/)). The anchor setup compresses **Aya Expanse 8B** and evaluates on XCOMET-XL, MetricX-24, on-disk size, and tokens/sec on A100. The user's prior pipeline used **Information Flow Routes** (IFR; [Ferrando & Voita, 2024](https://arxiv.org/abs/2403.00824)) for layer scoring and **GPTQ** ([Frantar et al., 2023](https://arxiv.org/abs/2210.17323)) for quantization.

The three winning WMT25 reference points:
- **LeanAya** (LeanQuant only, no pruning) — COMET 53.2, 7.2 GB — near-baseline quality but very slow (2.9 tok/s at b=1).
- **Vicomtech / GeLaCo 0.75 + GKD + Q4** — 2.9 GB, 81% smaller, but COMET 31.1 (baseline 55.3) — extreme compression with catastrophic quality collapse.
- **TCD-Kreasof (24 layers)** — 12.0 GB, COMET 39.9 — iterative leave-one-out pruning + FT on 100k News Commentary.

---

# SECTION 1 — Methods for Measuring Component Contribution to Enable Pruning

## 1A. Activation-based layer/block redundancy

### SLEB (Song et al., ICML 2024) — [arXiv:2402.09025](https://arxiv.org/abs/2402.09025)
- **Core idea.** Iteratively remove the single block whose deletion least increases calibration perplexity on C4, re-evaluating after each removal. Unlike ShortGPT, removed blocks need not be contiguous.
- **What it gives over IFR+GPTQ.** A **damage-from-removal** signal rather than average-flow: it directly measures the downstream loss harm of pruning each candidate, closer to what actually matters for quality. Also composes cleanly with 4-bit AWQ with "no discernible impact" on PPL — explicitly composable with quantization.
- **Reported results.** Llama-2-70B at 20% block removal: preserves WikiText-2/C4 PPL and zero-shot accuracy better than Wanda-2:4 and SliceGPT-25%. Largest end-to-end latency/throughput win among compared structured methods. No MT numbers.
- **Compatibility.** Training-free; stacks with AWQ per paper. No ordering conflicts reported. Composable with any recovery FT.
- **Calibration.** C4; standard 128×2048 regime. Generic domain.
- **Compute cost.** O(L²) forward passes (≈32² on Aya-8B) — comparable to leave-one-out and cheaper than evolutionary search over merge operations.
- **7B+ / generation.** Yes on ≥7B decoder-only (up to 70B); no MT/COMET evaluation.

### LLM-Streamline (Chen et al., 2024) — [arXiv:2403.19135](https://arxiv.org/abs/2403.19135)
- **Core idea.** Identify the most redundant contiguous block via input-output cosine similarity (ShortGPT-style), then **replace** that block with a lightweight MLP trained via MSE to mimic its hidden-state mapping.
- **What it gives.** A cheap **replacement-based healing** step separate from LoRA — the MLP is trained on tens of thousands of SlimPajama samples and reportedly outperforms LoRA healing for this structural pattern. Explicitly reports the classification-vs-generation gap (92% classification retained vs 68% generation at 25%), which is the regime the user cares about.
- **Reported results.** Llama-2-7B, 25% pruning: ~92% classification retention / ~68% generation. Outperforms ShortGPT, LaCo, LLM-Pruner at matched ratio. No MT.
- **Compatibility.** The MLP replacement is a small new component; should be quantized alongside the base. Authors don't test downstream quantization.
- **Calibration/data.** Tens of thousands of SlimPajama mixture samples for the replacement MLP; no big fine-tune.
- **Compute cost.** One forward pass for similarity + small MLP training.
- **7B+ / generation.** Yes decoder-only up to 13B; generation gap explicitly reported.

### "The Unreasonable Ineffectiveness of the Deeper Layers" (Gromov et al., ICLR 2025) — [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)
- **Core idea.** Pick ℓ* minimizing the **angular distance** d(x^(ℓ), x^(ℓ+n)) = (1/π) arccos(⟨x^(ℓ), x^(ℓ+n)⟩/‖x^(ℓ)‖‖x^(ℓ+n)‖) between last-token hidden states over C4, drop that contiguous block, heal with QLoRA.
- **What it gives.** A clean phase-transition characterization: Llama-2 tolerates ~45–55% removal before a sharp quality cliff, but **Qwen tolerates only ~20%, Mistral ~35%, Phi-2 ~25%**. Directly warns that Aya Expanse (multilingual, arguably denser) may sit closer to Qwen's curve than Llama-2's — suggesting a hard ceiling well below the GeLaCo 0.75 setting.
- **Reported results.** Llama-2-7B/13B/70B, Qwen-7B/14B, Mistral-7B, Phi-2. Above the cliff, near-random MMLU/BoolQ. No MT.
- **Compatibility.** Healing is QLoRA (4-bit NF4 + LoRA), so pruning is already composed with 4-bit quantization. Final quantization could swap NF4 for GPTQ/AWQ/LeanQuant.
- **Calibration.** C4, ~few hundred samples; healing on ~150–200M C4 tokens.
- **Compute.** Single forward pass for the similarity matrix; cheap.
- **7B+ / generation.** Yes multiple ≥7B; no MT — evaluation is MMLU + BoolQ + C4 loss.

### Shortened LLaMA (Kim et al., ICLR-W 2024) — [arXiv:2402.02834](https://arxiv.org/abs/2402.02834)
- **Core idea.** Compare three block-importance signals — magnitude, Taylor (∂L/∂W · W), and PPL-when-removed — on Llama/Vicuna and ablate retraining recipes (LoRA vs continued pretraining on SlimPajama).
- **What it gives.** **The single most useful empirical result for the user's pipeline:** at severe pruning ratios (>~35%), **LoRA healing is insufficient**; full continued pretraining on a large corpus (SlimPajama) is required. Also: the **PPL-when-removed criterion beats Taylor and magnitude for generation quality** — a direct endorsement of leave-one-out style signals over gradient-only saliency for generation tasks.
- **Reported results.** 20–27% block removal; at 20% Llama-7B, modest drops on 7 commonsense tasks. No MT.
- **Compatibility.** Healing recipes explicit. Recommends all-linear LoRA; explicitly documents where LoRA fails.
- **Calibration.** 10 BookCorpus samples for signals; healing uses 50K refined-Alpaca (LoRA) or SlimPajama subset (CPT).
- **Compute.** PPL criterion is O(L) forwards — same as SLEB's inner loop.
- **7B+ / generation.** Yes ≥7B decoder-only. PPL proxy only, no MT.

### ShortGPT (Men et al., 2024) — [arXiv:2403.03853](https://arxiv.org/abs/2403.03853)
- **Core idea.** Block Influence: BI_i = 1 − E[cos(X_{i,t}, X_{i+1,t})]; drop layers with lowest BI.
- **What it gives.** The cheapest possible contribution signal (single forward pass, no gradients, no calibration sweep). Empirically flags deep (late-but-not-final) layers — matching IFR, and arguably making IFR's extra mechanism unnecessary if the signal content is the same. Authors stack with AWQ.
- **Reported results.** Llama-2-7B/13B, Baichuan2-7B/13B; ~25–28% layer removal retains most classification performance. **Explicitly notes** severe degradation on HumanEval/generation. No MT.
- **Compatibility.** Training-free; AWQ-composable.
- **Calibration.** Tiny; any generic text.
- **7B+ / generation.** Yes; generation impact explicitly flagged as more severe than classification.

### LaCo (Yang et al., EMNLP-F 2024) — [arXiv:2402.11187](https://arxiv.org/abs/2402.11187)
- **Core idea.** Instead of deleting, **merge** rear layers into an earlier layer via weight-delta aggregation: W_l ← W_l + Σ(W_{l+k} − W_l), while a cosine-similarity "Reserving Rate" threshold caps output drift.
- **What it gives.** A **mass-preserving structural operation**: parameter count per remaining layer unchanged, so subsequent GPTQ/LeanQuant calibration protocols transfer trivially. The foundation GeLaCo later put inside an evolutionary search.
- **Reported results.** 25–30% pruning retains >80% of OpenCompass average; beats LLMPruner and SliceGPT. No MT.
- **Compatibility.** Training-free; quantization-composable (merged layer is just a new dense weight).
- **Calibration.** Small similarity-check set.
- **7B+ / generation.** Yes ≥7B; XSum summarization only, no MT.

### Synthesis (1A — activation-based depth)
The frontier is **angular/cosine redundancy for ranking + PPL-when-removed for verification**, as used by SLEB and advocated by Shortened LLaMA. All six methods converge on the same empirical structural claim: **deep (late-but-not-final) layers are most removable; early layers and the final 1–4 blocks are critical**. This directly contradicts IFR's "middle layers lowest" reading — and it is exactly the kind of task-agnostic blunt-signal convergence the user flagged. The two most promising for this pipeline are **(i) SLEB's iterative PPL-when-removed** (true damage-from-removal signal at ~O(L²) cost) and **(ii) Shortened LLaMA's ablation methodology** as a recipe for choosing signal + healing jointly. The clear literature gap: **no depth-pruning paper reports MT/COMET/BLEU numbers**; all evaluation is PPL + multiple-choice QA, so the user's WMT25 setting is in fact under-covered by the canonical literature and constitutes genuinely novel evaluation territory.

## 1B. Attribution-based scoring

### EAP-IG ("Have Faith in Faithfulness"; Hanna, Pezzelle, Belinkov, COLM 2024) — [arXiv:2403.17806](https://arxiv.org/abs/2403.17806)
- **Core idea.** Edge Attribution Patching with Integrated Gradients: IE_IG(e) = (z'_u − z_u) · (1/m) Σ_k ∂L/∂z_v at z_v(α_k) interpolating clean→corrupt. Mitigates zero-gradient saturation of vanilla EAP.
- **What it gives over IFR+GPTQ.** A **task-conditional edge-level importance** with principled counterfactuals. Unlike IFR (average flow over generic text), EAP-IG is *designed* to differ between tasks — so calibrating on MT pairs (correct-translation vs. paraphrase-corrupted) should produce an MT-conditional ranking.
- **Reported results.** Paper reports faithfulness of circuits, not PPL/MMLU; validated up to GPT-2 medium / Pythia-2.8B in the original paper, with extensions to CodeLlama-13B. **No compression numbers, no MT.**
- **Compatibility.** Edge-level signal needs aggregation to heads/layers for structured pruning. Open engineering step.
- **Calibration.** Task-specific paired clean/corrupt prompts; MT could use (correct translation, lexically-similar distractor).
- **Compute cost.** 2 forwards + m backwards (typically 5–10).
- **7B+ / generation.** Borderline (2.8–13B range); not validated on ≥7B decoder-only generation tasks for pruning.

### Edge Attribution Patching (EAP; Syed, Rager, Conmy, 2023) — [arXiv:2310.10348](https://arxiv.org/abs/2310.10348)
- **Core idea.** First-order Taylor approximation of activation patching: IE(e) ≈ (z'_u − z_u) · ∂L/∂z_v. All edges scored from 2 forwards + 1 backward.
- **What it gives.** Cheapest task-conditional attribution at edge granularity.
- **Reported results.** Circuit-discovery AUC on IOI/greater-than/docstring; **not used for pruning in the original paper**. No PPL, no MMLU, no MT.
- **Compatibility.** Same as EAP-IG — needs aggregation to a structured granularity.
- **Calibration.** Paired prompts per task.
- **Compute.** 2F+1B — cheapest task-conditional option in this section.
- **7B+ / generation.** GPT-2 small in paper; follow-ups go larger.

### Attribution-guided Pruning (Hatefi et al., 2025) — [arXiv:2506.13727](https://arxiv.org/abs/2506.13727)
- **Core idea.** Uses **Layer-wise Relevance Propagation (LRP)** as the attribution signal for LLM compression, unifying circuit discovery, pruning, and targeted behavior removal.
- **What it gives.** The **only explicit bridge from attribution → LLM compression** found in this survey. Demonstrates that attribution-based importance can actually drive pruning decisions, not just interpretability.
- **Reported results.** [UNVERIFIED in this pass — user should fetch the paper directly for numbers.] The paper itself notes that Wanda-style saliency cannot attribute at attention-head granularity, which LRP can.
- **Compatibility.** In principle stacks with recovery FT and quantization; details depend on paper specifics not verified here.
- **Calibration.** LRP needs forward activations + attribution targets.
- **7B+ / generation.** To verify.

### Synthesis (1B — attribution)
**Attribution-for-pruning is largely an open gap.** Both EAP and EAP-IG are interpretability tools that have not been systematically validated as pruning saliency at 7B+ with generation evaluation. The only concrete bridge is Hatefi et al. 2025 (LRP-based), which is recent and not MT-evaluated. The most promising adaptation path for this pipeline: **EAP-IG with MT clean/corrupt prompt pairs as the task conditioning, aggregated head- and layer-wise**. If it works, it would be the first genuinely **task-conditional** structural-pruning signal for MT — a direct answer to the user's "IFR is task-agnostic" complaint. The risk is that nobody has shown this kind of attribution transfers to structured importance on ≥8B decoder-only models for generation. Treat as a research experiment, not a drop-in.

## 1C. Gradient / Hessian / Taylor importance

### Sheared-LLaMA (Xia et al., ICLR 2024) — [arXiv:2310.06694](https://arxiv.org/abs/2310.06694)
- **Core idea.** Learn differentiable masks for layers, heads, MLP-intermediate, and hidden dim via augmented-Lagrangian constraints; continue pretraining with dynamic batch loading.
- **What it gives.** The most granular structured signal on the list (**4 mask types jointly**), and the only method here that explicitly targets a pre-specified compressed architecture. Gradient-driven, so task/data conditional through the training loss.
- **Reported results.** LLaMA2-7B → 1.3B/2.7B using only 3% of from-scratch compute; beats TinyLlama/OpenLLaMA at scale. No MT.
- **Compatibility.** Training-heavy (~50B tokens continued pretraining), so overlaps with recovery FT rather than composing cleanly with a separate healing stage.
- **Calibration.** RedPajama, dynamic reweighting across 7 domains.
- **Compute.** Full training run — order of magnitude above all other Section 1 methods. Likely too expensive for WMT-scale iteration.
- **7B+ / generation.** Yes ≥7B source; instruction tuning but no MT.

### LLM-Pruner (Ma et al., NeurIPS 2023) — [arXiv:2305.11627](https://arxiv.org/abs/2305.11627)
- **Core idea.** Dependency-graph-based coupled-structure identification scored with Taylor importance |w · ∂L/∂w|, plus LoRA recovery (~3h, 50K Alpaca).
- **What it gives over IFR+GPTQ.** Head- and channel-level granularity with a **gradient signal** — can in principle be made task-conditional by computing gradients on MT data instead of BookCorpus, something the user's IFR pipeline cannot do at this granularity.
- **Reported results.** ~20% params on LLaMA-7B; modest PPL increase, recoverable with LoRA. Only qualitative generation; no MT.
- **Compatibility.** LoRA recovery stage is native; subsequent quantization (GPTQ/LeanQuant) is standard.
- **Calibration.** 10 BookCorpus samples (tiny); 50K Alpaca for recovery.
- **Compute.** 1 forward + 1 backward over calibration.
- **7B+ / generation.** Yes ≥7B decoder-only.

### FLAP (An et al., AAAI 2024) — [arXiv:2312.11983](https://arxiv.org/abs/2312.11983)
- **Core idea.** Retraining-free structured pruning via fluctuation metric: WIF_j = ‖W_*,j‖² · Var(X_j). Adaptive per-module sparsity via cross-layer standardization + bias compensation.
- **What it gives.** Structured (heads + MLP channels) retraining-free pruning with a cheap activation-statistics signal. Bias compensation partially substitutes for a full healing step.
- **Reported results.** Beats LLM-Pruner and Wanda-structured at 20% sparsity on Llama-7B at zero retraining cost. No MT.
- **Compatibility.** Trivial; no retraining required before quantization.
- **Calibration.** WikiText-2; small.
- **Compute.** Single forward pass.
- **7B+ / generation.** Yes up to 65B; PPL + zero-shot only.

### SparseGPT (Frantar & Alistarh, ICML 2023) — [arXiv:2301.00774](https://arxiv.org/abs/2301.00774)
- **Core idea.** Per-layer OBS-style weighted reconstruction: s_ij = w_ij²/[H⁻¹]_jj, with closed-form δ = −w_ij H⁻¹[:,j] / [H⁻¹]_jj. Simultaneously masks and updates weights.
- **What it gives.** Principled second-order saliency with no backward pass; one of the few scalable Hessian-aware methods. But **primarily unstructured / 2:4 semi-structured** — limited direct use for the depth-pruning regime the user is in.
- **Reported results.** 50–60% unstructured on OPT-175B/BLOOM-176B with near-zero PPL change. No MT.
- **Compatibility.** 2:4 sparsity + 4-bit quantization stacks in principle. Structured adaptation is not native.
- **Calibration.** 128×2048 C4; fast.
- **Compute.** Forward only; inverse Hessian per layer.

### Wanda (Sun et al., ICLR 2024) — [arXiv:2306.11695](https://arxiv.org/abs/2306.11695)
- **Core idea.** S_ij = |W_ij| · ‖X_j‖₂ per output row; no retraining, no weight update.
- **What it gives.** The simplest scalable saliency — cheap enough to recompute under a task-specific calibration set and check whether the ranking shifts for MT.
- **Reported results.** 50% sparsity on LLaMA-7B, WikiText-2 PPL ~7.26 vs dense 5.68. Competitive with SparseGPT.
- **Compatibility.** 2:4/4:8 semi-structured meshes with NVIDIA sparse tensor cores; can stack with 4-bit quant.
- **Calibration.** 128×2048 C4.
- **Compute.** Single forward pass.

### OWL (Yin et al., 2024) — [arXiv:2310.05175](https://arxiv.org/abs/2310.05175)
- **Core idea.** Layerwise Outlier Distribution (LOD) allocates per-layer sparsity ratios inversely proportional to outlier density; per-weight scoring still delegated to Wanda/SparseGPT.
- **What it gives.** A **non-uniform per-layer sparsity budget** — the exact counterpoint to IFR's uniform middle-layer-lowest flag. At 70% sparsity on Llama-7B, OWL+Wanda lowers PPL from ~85 (uniform Wanda) to ~10.
- **Compatibility.** Drop-in allocator on top of any weight saliency.
- **Calibration.** 128 C4 samples.
- **Compute.** Forward + outlier statistics.
- **7B+ / generation.** Yes up to 70B; no MT.

### SliceGPT (Ashkboos et al., ICLR 2024) — [arXiv:2401.15024](https://arxiv.org/abs/2401.15024)
- **Core idea.** Use RMSNorm's computational invariance: apply PCA rotation Q to weights, drop the lowest-variance hidden dimensions. Width pruning.
- **What it gives.** The **only width-pruning method here that composes directly with rotation-based quantization** — QuaRot explicitly builds on SliceGPT's invariance framework. Offers a complementary compression axis to layer drop.
- **Reported results.** Up to 25% param reduction on Llama-2-70B/OPT-66B/Phi-2 retaining 99/99/90% zero-shot performance. No MT.
- **Compatibility.** Most natural composition with QuaRot/SpinQuant downstream.
- **Calibration.** WikiText-2; small.
- **7B+ / generation.** Yes up to 70B; no MT.

### Synthesis (1C — gradient/Hessian)
Most promising for this pipeline: **(i) LLM-Pruner with MT-specific calibration gradients** (straightforward generalization of Taylor importance to MT parallel data — the user's natural "task-aware IFR replacement" with backward passes); **(ii) OWL as a layer-budget allocator** stacked on whatever per-weight signal they pick (it would directly counter the "IFR always flags middle layers" monoculture by making the pruning ratio itself outlier-sensitive); and **(iii) SliceGPT** specifically if combining with rotation-based quantization. Sheared-LLaMA is likely too expensive for an iteration cycle. Literature gap: **almost nobody has combined gradient-based structural saliency with task-specific (MT) calibration data and reported COMET** — an obvious empirical opportunity.

## 1D. Calibration-aware and task-aware scoring
No paper in the surveyed set specifically computes layer contribution using **MT parallel data** as calibration. The closest signals are general calibration-data-matters findings scattered through the PTQ literature and the implicit task-conditionality of EAP/EAP-IG. **Flag this as a live gap** — calibrating SparseGPT/Wanda/LLM-Pruner with MT source-target pairs, and comparing the resulting layer rankings against generic-C4 rankings, is a well-defined unreported experiment. The user's intuition that "generic calibration is why middle layers always lose" is consistent with the evidence: every method in Section 1A that uses generic calibration converges on deep-layer pruning, while the only task-conditional methods (EAP/EAP-IG) have not been tested at structural granularity. **UNVERIFIED claim flag:** I did not locate a paper that ablates MT vs generic calibration for layer pruning; this should be treated as an open experimental slot.

## 1E. Feature-level signals for structural decisions
**Explicit literature gap.** Sparse Autoencoders (Anthropic's Towards/Scaling Monosemanticity; Gemma Scope), transcoders, and crosscoders are extensively developed for **interpretability**, but no surveyed paper uses feature density, feature count, or feature-utilization per layer/head as a pruning saliency. This is a promising but unproven direction — dead-feature counts could flag underutilized capacity more semantically than activation-norm methods, but the engineering cost of training layer-wise SAEs on Aya-8B is substantial, and there is no prior to suggest it beats cheaper signals. **Flag as research territory, not drop-in.**

## 1F. Evolutionary / search-based scoring

### GeLaCo (Ponce, Etchegoyhen, Del Ser, 2025) — [arXiv:2507.10059](https://arxiv.org/abs/2507.10059); WMT25 paper at [aclanthology.org/2025.wmt-1.77](https://aclanthology.org/2025.wmt-1.77/)
- **Core idea.** Evolutionary search over LaCo-style layer-collapse operations, ranked by **average module-wise cosine similarity** between original and compressed hidden states on calibration data. Supports Pareto-frontier search over compression ratio × quality.
- **What it gives.** Global search over merge configurations rather than greedy single-layer decisions. Gives Pareto points that greedy SLEB or Gromov cannot reach.
- **Reported results (per WMT25 system paper, COMET-da × 100 scale).** Aya Expanse 8B baseline 0.8476 (cs-de). 0.25+SFT → 0.8304; 0.50+SFT → 0.7964; 0.75+SFT → 0.6578; **0.75+GKD → 0.7799**; 0.75+GKD+Q4 → 0.7756 (2.79 GiB, 81% smaller). On the organizers' harder XCOMET-XL, 0.75+GKD+Q4 is 31.1 — the cliff WMT25 flagged.
- **Failure mode.** GKD-trained models **catastrophically lose in-context-learning ability** — few-shot COMET on cs-de drops from 0.78 to 0.37. And at 0.75 for ja-zh, translation quality collapses even under GKD.
- **Compatibility.** Explicit composition: SFT or GKD healing, then bnb 4-/8-bit quantization.
- **Calibration.** Only **96 ParaCrawl sentences** (16×3 languages); SFT/GKD on ~3M translation instructions.
- **Compute.** Evolution: 10,000 steps × fitness evaluations. Cheaper than retraining each candidate but not cheap.
- **7B+ / generation.** Yes on Aya Expanse 8B with MT evaluation — **the only method in Section 1 with full MT numbers**.

### Synthesis (1F — evolutionary)
GeLaCo is the WMT25 high-compression anchor and the only surveyed method with actual COMET numbers on this exact setting. The **collapse at 0.75 is attributable to (a) signal-agnostic fitness** — average cosine similarity does not distinguish task-critical from task-redundant deviation — **and (b) the merge operator itself averaging away language-specific specialization**. GeLaCo's 96-sentence calibration is strikingly small, suggesting the bottleneck is not calibration size but the fitness metric's coarseness. **Most promising direction:** replace GeLaCo's fitness with a **task-conditional contribution signal** (EAP-IG edge-loss or LLM-Pruner Taylor over MT parallel data) — this would turn GeLaCo from "signal-agnostic evolutionary" into "evolutionary over task-aware damage", likely extending the Pareto frontier at 0.50–0.75 where quality currently collapses.

---

# SECTION 2 — Post-GPTQ Quantization Methods (2024–2026)

## 2A. Weight-only PTQ

### LeanQuant (Zhang & Shrivastava, ICLR 2025) — [arXiv:2407.10032](https://arxiv.org/abs/2407.10032)
- **Core idea.** Replace GPTQ's uniform min-max affine grid with a **loss-error-aware grid** learned via weighted k-means-style placement: grid points minimize Σᵢ diag(H⁻¹)ᵢ^(-p) · (quant(wᵢ,G) − wᵢ)². Weights with small inverse-Hessian diagonal (high loss sensitivity) pull grid levels toward them. Generalizes to affine and non-uniform quantization.
- **What it gives over GPTQ.** **Directly addresses GPTQ's main failure mode**: a few sensitive weight rows dominate task loss because the uniform grid doesn't allocate resolution proportional to loss impact. At 4-bit the advantage is modest; gains widen sharply at 3-bit and 2-bit.
- **Reported results.** Llama-3.1-405B quantized in ~21–24h on 2×RTX 8000; 70B in ~4h on single 24GB GPU. Beats GPTQ/AWQ/OmniQuant/SqueezeLLM at matched bit-widths on Llama-2/3 + Mistral-7B WikiText-2/PTB/C4 PPL and ArcE/PiQA/StoryCloze/WinoGrande.
- **WMT25 evidence.** LeanAya submission achieved COMET 53.2 / MetricX 5.36 at 7.2 GB — **within 2.1 COMET of baseline** — the best quality-preserving compression at the competition. Caveat: tokens/sec was very low (2.9 at b=1, 21.3 at b=16), likely due to non-optimal kernel deployment rather than the method itself.
- **Compatibility.** Output weights remain compatible with **Marlin/LUT-GEMM kernels** — no custom kernel required. This is a real advantage over QuIP#/AQLM.
- **Calibration.** ~128×2048 C4-class; standard.
- **Already-pruned models.** Not explicitly tested by authors.
- **Bit-widths.** W4, W3, W2. Weight-only.

### AWQ (Lin et al., MLSys 2024) — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- **Core idea.** Protect ~1% salient weight channels identified via activation magnitude statistics; apply learned per-channel scaling to suppress their quantization error. No backprop/reconstruction.
- **What it gives over GPTQ.** Better out-of-domain / instruction-tuned generalization (no overfitting to calibration set) and faster kernels. Marlin-AWQ is often the fastest 4-bit path on A100+vLLM.
- **Reported results.** Competitive with GPTQ at 4-bit, stronger on instruction-tuned models.
- **Compatibility.** First-class vLLM + Marlin-AWQ kernel; TensorRT-LLM support.
- **Calibration.** ~128 sequences, Pile-class; no gradients.
- **Pruned.** Not tested.
- **Bit-widths.** W4, W3.

### OmniQuant (Shao et al., ICLR 2024) — [arXiv:2308.13137](https://arxiv.org/abs/2308.13137)
- **Core idea.** Block-wise differentiable PTQ with learnable weight clipping (LWC) + learnable equivalent transformation (LET) that shifts activation outliers into weights.
- **What it gives.** QAT-like quality at PTQ cost; supports W4A4, W2A16, W3A16 where GPTQ collapses.
- **Reported results.** Llama-2 family in 1–16h on single A100-40G. W3A16g128 ≈ 6.03 WikiText2 PPL on Llama-2-7B.
- **Compatibility.** No native vLLM kernel — deploy via MLC-LLM. Partial friction.
- **Calibration.** 128 samples.
- **Bit-widths.** W4A4/W6A6/W4A16/W3A16/W2A16.

### SqueezeLLM (Kim et al., ICML 2024) — [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
- **Core idea.** Hessian-weighted k-means non-uniform codebook + dense-and-sparse decomposition that keeps ~0.05% outliers in FP16.
- **What it gives.** Best 3-bit PPL among uniform-grid alternatives, narrowing the FP16 gap by up to 2.1× vs GPTQ/AWQ.
- **Compatibility.** Custom CUDA kernels only; no first-class vLLM/Marlin/TensorRT-LLM — **likely blocker for WMT-style tokens/sec evaluation**.

### QuIP# (Tseng et al., ICML 2024) — [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)
- **Core idea.** Randomized Hadamard Transform for incoherence + E₈-lattice vector-quantized codebooks + blockwise fine-tuning. Enables 2–3-bit at near-FP16 quality.
- **Compatibility.** Custom CUDA only, no vLLM/TensorRT-LLM. Research-tier for this pipeline.

### AQLM (Egiazarian et al., ICML 2024) — [arXiv:2401.06118](https://arxiv.org/abs/2401.06118)
- **Core idea.** Multi-codebook additive quantization from ANN search adapted to LLM weights, jointly optimized at block granularity.
- **Reported results (verified from abstract).** Llama-2-7B 2-bit WikiText2 = 6.93 (1.29 better than prior best); 13B 2-bit = 5.70; 70B 2-bit = 3.94. Pareto-optimal below 3 bits.
- **Compatibility.** vLLM support experimental; no Marlin. Calibration: ~1 day for 7B, 10–14 days for 70B on single A100.

### HQQ (Badri & Shaji, 2023 Mobius Labs; [blog](https://mobiusml.github.io/hqq_blog/); [github.com/mobiusml/hqq](https://github.com/mobiusml/hqq))
- **Core idea.** Half-quadratic splitting of a sparsity-promoting (lp<1) objective; closed-form zero-point updates. **No calibration data required.**
- **What it gives over GPTQ.** **~50× faster quantization** (70B in <5 min vs ~4h for GPTQ) and calibration-free — valuable for rapid iteration on pruned-model variants.
- **Compatibility.** vLLM-compatible via HF Transformers integration; no Marlin. No formal arXiv paper — cite blog.

### Synthesis (2A — weight-only)
**LeanQuant is the frontier for quality-preserving 4-bit-and-below PTQ on Aya-class models** — it is the WMT25 evidence-backed winner and its grid-learning concept directly addresses why GPTQ falters on outlier-heavy sensitive rows. AWQ (Marlin-AWQ) is the throughput-optimal drop-in if LeanQuant kernels remain unavailable on the target stack. AQLM and QuIP# are the 2-bit frontier but currently kernel-bound; not yet production-ready for A100+vLLM tokens/sec metrics.

## 2B. Rotation-based methods

### QuaRot (Ashkboos et al., NeurIPS 2024) — [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- **Core idea.** Apply randomized Hadamard rotations (R1, R2 offline-fused into weights; R3 online for KV; R4 online for down-proj) that exploit SliceGPT's computational-invariance theorem to eliminate activation outliers, enabling W4A4KV4.
- **What it gives over GPTQ.** End-to-end 4-bit including activations and KV cache — not just weights. Llama-2-7B: ≤0.63 PPL loss at W4A4KV4; Llama-2-70B: ≤0.47 loss. Up to 3.33× prefill speedup on 70B.
- **Rotation × pruning interaction — CRITICAL.** QuaRot is methodologically *built on* SliceGPT (structural width pruning), so rotations are **compatible by construction** with SliceGPT-style pruning. **However, no paper systematically evaluates QuaRot or SpinQuant on top of a separately layer-pruned model** (ShortGPT/SLEB/GeLaCo output), and arXiv:2502.15779 documents QuaRot failing at W2A4KV4, suggesting rotation benefits saturate when combined with aggressive additional compression. **Treat rotation × depth-pruning composition as an open-empirical question**, with width-pruning (SliceGPT) compatibility as the only solid footing.
- **Compatibility.** Weights quantized via GPTQ underneath; W6/W8 calibration-free. No first-class vLLM kernel upstream — AMD Quark / LLMC toolkit.
- **Bit-widths.** W4A4KV4, W4A8, W6A6, W8A8.
- **Pruned.** Not explicitly tested.

### SpinQuant (Liu et al., ICLR 2025) — [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
- **Core idea.** Like QuaRot but **learns** rotations via Cayley SGD on the Stiefel manifold on a small validation set. Observation: random rotations vary by up to 13pp zero-shot accuracy — learning matters.
- **What it gives over QuaRot.** Llama-2-7B W4A4KV4: closes 30.2% of the gap to FP; Llama-3-8B: 34.1% closure. Particularly helpful on harder-to-quantize Llama-3 class (Aya Expanse is Llama-3-based, so relevant).
- **Compatibility.** Fast-Hadamard-transform kernel; ExecuTorch mobile export. vLLM upstream integration not first-class.
- **Calibration.** Hundreds of samples, WikiText2/C4; plus GPTQ for weights.

### Synthesis (2B — rotation)
For Aya Expanse (Llama-3 lineage), **SpinQuant > QuaRot** on pure quality grounds, but **neither has a first-class A100+vLLM kernel** as of early 2026 — a real blocker for the WMT25 tokens/sec metric. For this pipeline, rotation-based quantization is best treated as a **later upgrade** once a pruned+healed model is stable, not as the first-choice quantizer.

## 2C. Weight + activation quantization
The major names (SmoothQuant, Atom, QServe, ZeroQuant family) are one generation behind the rotation-based approaches for sub-4-bit activation quantization. QuaRot and SpinQuant largely subsume this territory for new work. SmoothQuant remains useful as a baseline and for W8A8 where outlier shifting alone is sufficient. Not elaborated here because the WMT25 evaluation dimensions (size, tokens/sec at batch) are better served by weight-only + KV-cache quant than by aggressive activation quant at this scale.

## 2D. Extreme low-bit

### BitNet b1.58 (Ma et al., 2024) — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **Post-hoc viability: NO.** Original paper trains from scratch with BitLinear + STE. Post-hoc ternary conversions of FP16 Llamas (Falcon3-1.58bit, Llama3-8B-1.58) exist as community follow-ups and **underperform native training** substantially per the BitNet 2B4T report ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)).
- **Kernel.** `bitnet.cpp` (CPU); no vLLM.
- **Not a drop-in for this pipeline.**

### BiLLM (Huang et al., ICML 2024) — [arXiv:2402.04291](https://arxiv.org/abs/2402.04291)
- **Core idea.** Hessian-salience split: salient weights get 2-bit binary-residual approximation; non-salient get optimal segmented binarization. ~1.08 average bits via block-wise GPTQ-style reconstruction.
- **Post-hoc.** Yes.
- **Results.** LLaMA2-70B WikiText2 = 8.41 at 1.08 bits. **Subsequent study ([arXiv:2404.14047](https://arxiv.org/abs/2404.14047)) shows BiLLM often collapses on Llama-3 / recent models** — a relevant negative result for Aya-class targets.

### OneBit (Xu et al., NeurIPS 2024) — [arXiv:2402.11295](https://arxiv.org/abs/2402.11295)
- **Core idea.** Sign matrix + two FP16 value vectors per linear layer (Sign-Value-Independent Decomposition); QAT with teacher distillation over ~13.5B tokens.
- **Post-hoc.** Not pure PTQ — requires QAT with 13.5B training tokens, but starts from a pretrained model.
- **Results.** ≥81–83% of FP16 performance on Llama-7B/13B; beats 2-bit GPTQ.

### Synthesis (2D — extreme)
**For this pipeline, extreme low-bit is not yet viable as a drop-in.** BitNet b1.58 is from-scratch; BiLLM collapses on Llama-3-class models; OneBit requires substantial QAT. The practical low-bit frontier for a **pruned + healed** Aya-8B is **2-bit weight via AQLM or QuIP#**, both kernel-bound on A100+vLLM. Recommendation: stay at 4-bit via LeanQuant/AWQ for the near-term pipeline; reserve sub-3-bit for future iteration if LeanQuant kernels improve.

## 2E. KV-cache quantization

### KIVI (Liu et al., ICML 2024) — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **Core idea.** Tuning-free asymmetric 2-bit KV quantization: keys per-channel (outlier-aware), values per-token (streaming-friendly); FP16 residual window on recent tokens.
- **What it gives for MT throughput.** On Llama-2-7B: 2.6× KV-memory reduction → up to **~3.47× batch-inference throughput** at larger batch sizes. Plug-and-play, zero fine-tuning.
- **Compatibility.** HF Transformers integration; not default in vLLM mainline (check current PRs).
- **For WMT25 tokens/sec at batch 64/512, KIVI is the throughput win of the KV literature.**

### KVQuant (Hooper et al., NeurIPS 2024) — [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- **Core idea.** Sub-4-bit KV PTQ via per-channel key quant, pre-RoPE key quant, non-uniform per-layer sensitivity, per-vector dense-and-sparse outlier isolation, Q-Norm for 2-bit.
- **Results.** <0.1 PPL degradation at 3-bit; enables 1M–10M context on 1–8×A100.
- **Trade-off vs KIVI.** KVQuant wins at small-batch long-context; KIVI wins at large-batch throughput.

### Synthesis (2E)
**KIVI** is the clear choice for WMT batch-inference tokens/sec on A100. Composes orthogonally with any weight quantization (LeanQuant/AWQ/GPTQ) since it touches only KV cache. Adds the batch-size-dependent speedup that pure weight quant alone does not deliver (the WMT25 leaderboard shows weight-only methods often *lose* throughput at small batch while gaining disk size; KV-cache quant is the complement).

## 2F. Quantization-aware fine-tuning / QAT-lite

### EfficientQAT (Chen et al., ACL 2025) — [arXiv:2407.11062](https://arxiv.org/abs/2407.11062)
- **Core idea.** Two-phase QAT: Block-AP trains all params + step sizes + zero-points with block-wise reconstruction; E2E-QP end-to-end trains only quantization step sizes with frozen quantized backbone.
- **Results.** **2-bit Llama-2-70B in 41h on single A100-80G**, avg accuracy 69.48 vs FP16 72.41 — <3 points drop. INT2-70B beats FP16-13B (69.48 vs 67.81) with less memory (19.2 vs 24.2 GB). Outperforms AQLM, QuIP#, OmniQuant, AutoRound at 2-bit.
- **Compatibility.** Works with GPTQ-compatible int2/int3/int4 kernels (ExLlamaV2-style); vLLM path not first-class upstream.

### BitDistiller (Du et al., ACL 2024) — [arXiv:2402.10631](https://arxiv.org/abs/2402.10631)
- **Core idea.** Self-distillation QAT with **Confidence-Aware KL (CAKLD)** interpolating forward/reverse KL between FP16 teacher and same-model quantized student.
- **Results.** At 3-bit, beats GPTQ and AWQ on Llama-2-7B/13B WikiText2 + math/code reasoning.
- **Use case.** Recovery phase that is *simultaneously* quantization-aware — joins stages 3 and 4 of the user's pipeline.

### QLoRA (Dettmers et al., NeurIPS 2023) — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Core idea.** Freeze base at 4-bit NF4 + train LoRA adapters in BF16 through quantized weights. Double quantization + paged optimizers.
- **Key for this pipeline.** **The Gromov et al. healing recipe** — 4-bit NF4 healing on single A100. Directly usable as prune → heal step.

### LoftQ (Li et al., ICLR 2024) — [arXiv:2310.08659](https://arxiv.org/abs/2310.08659)
- **Core idea.** Jointly find Q and (A,B) such that Q+AB ≈ W at initialization; alternating SVD + quantize. Fixes QLoRA's init gap at 2–3 bit.
- **Results.** At 4-bit, +1.1 Rouge-1 XSum, +0.8 CNN/DM over QLoRA. At 2-bit NF: +8% MNLI, +10% SQuADv1.1.
- **Relevance.** The pragmatic sub-4-bit LoRA-init recipe.

### Synthesis (2F)
**Consensus across BitDistiller/EfficientQAT/LoftQ/OneBit and the empirical LLaMA-3 study ([arXiv:2404.14047](https://arxiv.org/abs/2404.14047)): at 3-bit, QAT-lite meaningfully outperforms pure PTQ (GPTQ/AWQ/OmniQuant), typically 0.2–1.0 PPL on WikiText2 for 7B-class, and the gap widens sharply at 2-bit.** For the user's pipeline, QAT-lite is the natural bridge if LeanQuant at 4-bit is insufficient — specifically **BitDistiller or LoftQ + healing combined into a single stage** to merge steps 3 and 4.

## 2G. Hardware/kernel landscape on A100 + vLLM
Production-ready: **GPTQ (Marlin/Machete), AWQ (Marlin-AWQ ~10.9× vs naive), FP8 W8A8, INT8 W8A8, HQQ (HF backend)**. Research-tier, no first-class vLLM A100 path: **QuIP#, AQLM, QuaRot, SpinQuant, SqueezeLLM, OmniQuant, LeanQuant** (LeanQuant outputs are Marlin/LUT-GEMM compatible in principle, but the WMT25 LeanAya 2.9 tok/s number suggests the deployed path was not Marlin — this is the most important open engineering question for this pipeline). **KIVI and KVQuant need separate KV-path integration.**

---

# SECTION 3 — Recovery / Healing Strategies for Pruned Models

## 3A. LoRA-based recovery and variants

### DoRA (Liu et al., ICML 2024 Oral) — [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- **Core idea.** Decompose pretrained weight into magnitude vector + direction matrix; apply LoRA only to direction, train magnitude separately.
- **What it gives over LoRA.** Closes most of the capacity gap to full FT at zero inference overhead (merges cleanly). Update patterns resemble full FT more than LoRA does.
- **Relevance.** At severe pruning where LoRA underfits (per Shortened LLaMA), DoRA is the first PEFT method likely to close the gap without going to full CPT.
- **Results.** Consistent gains over LoRA on commonsense reasoning, visual instruction tuning, image/video-text.
- **Pruned bases tested.** Not in original paper; supported in HF PEFT.

### PiSSA (Meng et al., NeurIPS 2024) — [arXiv:2404.02948](https://arxiv.org/abs/2404.02948) — *verification confidence medium*
- **Core idea.** Initialize A, B from top singular vectors of W rather than Gaussian/zero; residual tail is frozen.
- **Relevance.** Faster convergence and better performance at limited FT budget — directly addresses the pruned-model few-epoch recovery regime.

### rsLoRA (Kalajdzievski, 2023) — [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)
- **Core idea.** Scale LoRA by α/√r (rank-stabilized) instead of α/r — stable training at higher ranks.
- **Relevance.** Aggressive pruning needs more recovery capacity; rsLoRA makes high-rank (r≥32) LoRA actually train well.

### LoRA+ (Hayou et al., ICML 2024) — [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)
- **Core idea.** Different learning rates for A and B (typically ηB ≈ 16×ηA) per NTK analysis — B initialized at zero needs faster updates.
- **Relevance.** 1–2% accuracy gain + up to 2× FT speedup over vanilla LoRA; orthogonal to DoRA/PiSSA — composable.

### LoRA (Hu et al., ICLR 2022) — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- The baseline; all variants above are drop-in replacements.

### LoRA layer-targeting for pruned models
- **["Reassessing Layer Pruning in LLMs", arXiv:2411.15558, 2024]:** After layer pruning, **direct FT of the last 3 surviving transformer blocks + lm_head beats LoRA over all layers** on pruned Llama-3.1. Dolly-15k > Alpaca-cleaned for SFT data.
- **QLoRA / Raschka practice:** target **all linear sub-layers** (Q,K,V,O + up/gate/down) for best recovery, not just Q,V.
- **LoRAShear (below):** dependency-graph-aware selection.

### Synthesis (3A)
Best practice for post-pruning recovery in 2024–2026: **DoRA (or PiSSA for faster convergence) on all linear layers of the surviving last-third of the model**, using rsLoRA-style √r scaling at r≥32, with LoRA+ asymmetric LRs. For severe pruning (>35%), expect LoRA-family alone to be insufficient — Shortened LLaMA empirically recommends CPT on a SlimPajama-class corpus.

## 3B. Knowledge distillation

### GKD (Agarwal et al., ICLR 2024) — [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **Core idea.** On-policy distillation from **student self-generated sequences** with teacher feedback; flexible divergences (forward KL, reverse KL, generalized JSD).
- **What it gives for this pipeline.** **This is the method Vicomtech used at WMT25.** Solves train-inference mismatch (exposure bias) + student-capacity mismatch via reverse/skew divergences — both acute after structural pruning.
- **Results.** Beats SeqKD/ImitKD/supervised KD on XSum, WMT translation, GSM8K with T5-small/base/large students. On-policy + JSD/RKL optimal.
- **WMT25 failure observed.** Vicomtech reports GKD-trained compressed models **catastrophically lose ICL** — few-shot COMET on cs-de 0.78 → 0.37. The recovery is real for zero-shot MT but destroys in-context capability. Important design constraint.

### DistiLLM (Ko et al., ICML 2024) — [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)
- **Core idea.** Skew-KL loss (interpolates teacher/student distribution) + adaptive off-policy SGO scheduler with replay buffer.
- **What it gives over GKD.** **Up to 4.3× speedup** over GKD at matched or better quality. Tested on GPT-2/OPT/OpenLLaMA2/LLaMA2 students with LoRA.

### MiniLLM (Gu et al., ICLR 2024) — [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **Core idea.** Reverse KL (mode-seeking) via policy-gradient — prevents student overestimating teacher's low-probability regions.
- **Relevance.** Same problem class as GKD; slightly different optimizer.

### Synthesis (3B)
**On-policy logit distillation with reverse/skew KL dominates 2024–2026 for LLM compression recovery.** For this pipeline: **DistiLLM > GKD on compute, GKD ≈ DistiLLM on quality**, both >> logit matching (forward KL) or hidden-state distillation for generation tasks. The WMT25 GKD→ICL-collapse finding is a warning that on-policy KD over-specializes; mitigations worth testing are (a) mixing a small fraction of instruction data during KD, (b) reduced-epoch KD, (c) DoRA/LoRA-only updates during KD to bound capacity drift.

## 3C. Data-efficient / data-free healing
No paper in the surveyed set rigorously studies parallel-MT-sparse recovery regimes. Tangential evidence: LLM-Streamline's MLP-replacement works with tens of thousands of samples; Gromov et al. heal with ~150–200M C4 tokens on a single A100. **Recommendation under the WMT25 data budget:** TCD-Kreasof's 100K News Commentary scale is on the low end of what the literature validates; Shortened LLaMA's negative result on LoRA-alone at severe ratios suggests the same budget combined with full CPT on a larger monolingual + parallel mix (TowerBase-style) would likely outperform pure parallel FT.

## 3D. Healing recipes from depth-pruning literature
Summarized across Section 1A:
- **Gromov et al.:** QLoRA r=64 on all linear sub-layers, ~5000 steps seq-4096 on C4, single A100 — ~150–200M C4 tokens. Protocol of record for angular-distance drop.
- **Shortened LLaMA:** LoRA r=8 on 50K refined-Alpaca **for mild pruning**; CPT on SlimPajama subset **for severe pruning**. Explicit "LoRA fails at >35%" warning.
- **LLM-Streamline:** MSE-trained replacement MLP on tens of thousands of SlimPajama samples — cheaper than LoRA healing, reportedly better.
- **LaCo, SLEB, ShortGPT:** No healing in original papers.

## 3E. Joint / interleaved procedures

### LoRAPrune (Zhang et al., 2023) — [arXiv:2305.18403](https://arxiv.org/abs/2305.18403)
- **Core idea.** Use LoRA **weights and gradients** (not full-model gradients) as the structured pruning importance criterion; pruning and LoRA recovery unified; LoRA merges cleanly into pruned base.
- **Results.** On Llama-7B/13B/30B/65B at 50% compression: PPL reductions of 4.81/3.46 WikiText2/PTB vs LLM-Pruner (v5 numbers); 52.6% memory reduction.
- **What it gives.** A **joint prune-and-heal** formulation that cuts the user's 3-stage pipeline into 2, at the cost of not being task-conditional on MT.

### LoRAShear (Chen et al., 2023) — [arXiv:2310.18356](https://arxiv.org/abs/2310.18356)
- **Core idea.** Dependency-graph on LoRA modules + Lora Half-Space Projected Gradient for progressive structured pruning + dynamic multi-stage FT on pretraining-data *and* instruction data.
- **Results.** LLaMA-v1 on 1 A100: 20% pruning → 1% perf drop; 50% pruning → 82% preserved. Outperforms LLM-Pruner and LoRAPrune.
- **What it gives.** **The most sophisticated joint pipeline** — explicitly recognizes that structured pruning loses both general and domain knowledge and addresses each with a separate FT phase.

## 3F. MT-specific recovery signals

### ALMA (Xu et al., ICLR 2024) — [arXiv:2309.11674](https://arxiv.org/abs/2309.11674)
- **Core idea.** 2-stage MT FT: (1) continued pretraining on **non-English monolingual** data; (2) SFT on a **small** high-quality parallel set.
- **Key finding for this pipeline.** **Large parallel corpora HURT** LLM MT quality; small high-quality parallel data after monolingual CPT is the right recipe. This directly challenges the TCD-Kreasof "100K News Commentary" design — suggests a pruned Aya Expanse would benefit more from a monolingual multilingual CPT phase followed by a parallel SFT phase than from 100K parallel sentences alone.
- **Results.** +12 BLEU/COMET over zero-shot LLaMA-2 on 10 WMT'21/'22 directions; surpasses NLLB-54B and GPT-3.5-davinci-003 with ~1B monolingual tokens + ~18h training.

### ALMA-R / CPO (Xu et al., 2024) — [arXiv:2401.08417](https://arxiv.org/abs/2401.08417)
- **Core idea.** Replace stage-2 SFT with **Contrastive Preference Optimization** on (reference, GPT-4, ALMA) triplets scored with COMET/KIWI-XXL.
- **Relevance.** A recovery signal that goes **beyond the reference ceiling** — potentially important for pruned models whose reference-imitation ceiling is lower than FP16's.

### Tower / TowerInstruct (Alves et al., COLM 2024) — [arXiv:2402.17733](https://arxiv.org/abs/2402.17733)
- **Core idea.** Llama-2 CPT on 20B-token **mixed monolingual + bilingual** data → TowerBase; SFT on TowerBlocks (MT + APE + NER + paraphrasing) → TowerInstruct.
- **Key finding.** **Mixed monolingual + bilingual CPT beats monolingual-only (ALMA) and parallel-only.** This is the recipe most directly applicable to recovering a pruned multilingual model — the exact profile of Aya Expanse.

### X-ALMA (Xu et al., ICLR 2025) — [arXiv:2410.03115](https://arxiv.org/abs/2410.03115)
- **Core idea.** 50-language extension of ALMA via plug-and-play language-specific modules; 5-stage training; Adaptive Rejection Preference Optimization.
- **Relevance.** Directly relevant if the pruned model shows the "curse of multilinguality" (GeLaCo's 0.75 ja-zh collapse looks exactly like this) — language-grouped LoRA modules could recover affected pairs without disturbing others.

### Synthesis (Section 3)
Most data-efficient recipe for a 50% layer-pruned 8B decoder-only with limited MT parallel data: **TowerInstruct-style mixed monolingual + bilingual CPT (O(1B) tokens) followed by small high-quality parallel SFT and optionally CPO** — ALMA's negative result ("large parallel corpora hurt") should redirect the naive parallel-only instinct. GKD and DistiLLM remain the distillation options of record but carry the ICL-collapse risk documented in the Vicomtech WMT25 paper. LoRAShear is the most integrated joint-procedure option. **The literature gap:** no paper directly benchmarks monolingual-CPT vs parallel-SFT vs mixed-CPT as MT-recovery for a structurally pruned decoder-only LLM — this is exactly the ablation TCD-Kreasof et al. did not perform.

---

# CROSS-SECTION SYNTHESIS — Proposed Pipeline Variants

The WMT25 landscape has a clear shape: **LeanAya** holds the quality-preserving corner (COMET 53.2, 7.2 GB) and is beaten on compression by **GeLaCo 0.75 + GKD + Q4** (COMET 31.1, 2.9 GB), but the quality gap between them is catastrophic. Nothing on the leaderboard occupies the **moderate-compression-and-moderate-quality** middle — roughly the 4–6 GB at COMET 45–52 region — and the pruning signals actually used (Gromov-angular for GeLaCo, iterative-LOO for TCD-Kreasof) are both **task-agnostic**. This is where the user should aim. Three variants follow, each occupying a distinct Pareto region.

## Variant A: "Task-conditional safe middle" — aim for 4.5 GB, COMET 50+

- **Contribution scoring:** SLEB-style **PPL-when-removed**, but computed on an **MT parallel calibration set** (source-target pairs, ~2–4K WMT past-test-set sentences in the three task language pairs). The key deviation from WMT25 precedent is calibrating on MT rather than generic C4 — directly addressing the user's gripe that generic calibration is why middle layers always lose. Cheap: O(L²) ≈ 1000 forward passes on Aya-8B.
- **Pruning granularity:** Whole transformer layers, non-contiguous, up to **30% removal (~10 of 32 layers)** — below Gromov's Llama-2-family cliff and below Qwen/Mistral's empirical bound of ~20–35%.
- **Recovery:** Two-phase à la TowerInstruct — (1) ~500M token mixed monolingual+bilingual CPT on the three language pairs' families, with DoRA r=64 on all linear sub-layers of surviving last-third + direct FT on lm_head per arXiv:2411.15558; (2) ~100K high-quality parallel SFT (TCD-Kreasof's News Commentary + ALMA's small-parallel insight). Optionally add CPO preference phase.
- **Quantization:** LeanQuant W4 (LeanAya precedent) + **KIVI 2-bit KV** for batch throughput. LeanQuant-Marlin kernel path if engineering allows; else AWQ-Marlin as fallback.
- **Pareto estimate:** ~4.5 GB disk, COMET roughly **48–52** on cs-de (between GeLaCo 0.50+SFT q4 at 3.7 GB/COMET 31.0 and LeanAya at 7.2/53.2). Tokens/sec expected to match AWQ-Marlin baseline at b≥16 (≥100 tok/s) with KIVI bringing b=64 throughput higher still.

## Variant B: "Attribution-guided aggressive" — aim for 3.2 GB, COMET 40+

- **Contribution scoring:** **EAP-IG with MT clean/corrupt prompt pairs** (source→correct-target vs source→lexically-plausible-wrong-target), aggregated to head and layer granularity. The gamble is that task-conditional attribution identifies different layers than generic flow/cosine — worth testing because every other signal in the literature flags the same deep-layers regardless of task. Budget: 2 forwards + 5–10 backwards per layer-pair on a few hundred MT prompts.
- **Pruning granularity:** Mix of whole layers (40% removal, ~13 layers) and head-level pruning within retained layers, ranked by EAP-IG head-level aggregation. This is more aggressive than Variant A; falls in GeLaCo's 0.50 regime but with a hopefully smarter selection.
- **Recovery:** DistiLLM on-policy distillation (4.3× faster than GKD) from FP16 Aya as teacher, on mixed multilingual CPT + parallel data (~500M tokens), with DoRA all-linear on surviving last-half. Adds small SFT phase to mitigate GKD's ICL-collapse failure mode Vicomtech documented.
- **Quantization:** AWQ W4 (robustness beats LeanQuant under distribution shift) + KIVI KV. Rotation quant (SpinQuant) deferred — the layer-pruning changes residual geometry and we haven't validated rotations compose.
- **Pareto estimate:** ~3.2 GB, COMET **40–46** on cs-de (above GeLaCo 0.75+GKD+q4's 31.1, below LeanAya's 53.2). The research bet is that task-conditional attribution picks better layers than generic angular distance; if it doesn't, collapses to GeLaCo 0.50-style numbers.

## Variant C: "Joint prune-heal-quantize" — aim for 5 GB, COMET 52+

- **Contribution scoring:** **LoRAShear-style dependency-graph + LoRA-weight-based importance**, calibrated on MT parallel data, at 25% compression (roughly 8 of 32 layers equivalent via head+channel+layer combined). Produces pruning that is inherently LoRA-aware.
- **Pruning granularity:** Structured mix — whole layers at the tail + attention heads + MLP channels, ~25% aggregate parameter reduction.
- **Recovery:** LoRAShear's two-phase recovery (pretraining subset + instruction subset) adapted to MT via Tower-style mixed monolingual+bilingual CPT and parallel SFT.
- **Quantization:** BitDistiller-style self-distillation QAT at W3 on the pruned+healed model — merges steps 3 and 4 of the pipeline. Teacher is the same-model pruned+healed FP16 checkpoint; student is W3 quantized. Known to beat GPTQ/AWQ at 3-bit.
- **Pareto estimate:** ~5.0 GB, COMET **52–55** on cs-de — aiming to **match LeanAya quality at ~70% its size** while also fixing the tokens/sec problem (BitDistiller-W3 on AWQ-compatible kernels runs at AWQ speeds, unlike LeanAya's 2.9 tok/s). The main risk: BitDistiller at W3 may still degrade relative to W4 LeanQuant.

## Positioning against WMT25

On a quality(COMET cs-de)-vs-size scatter, the three variants would populate:
- **Variant A** near (4.5, 50) — fills a genuine Pareto gap between Vicomtech-0.25-q4 (4.5, 41.2) and LeanAya (7.2, 53.2).
- **Variant B** near (3.2, 43) — dominates Vicomtech-0.50-q4 (3.7, 31.0) on both axes if the EAP-IG hypothesis holds.
- **Variant C** near (5.0, 53) — roughly Pareto-matches LeanAya at 30% smaller footprint and, critically, with working throughput kernels.

The common throughline across all three: **task-conditional calibration on MT parallel data + on-policy distillation + Marlin-compatible 4-bit or BitDistiller 3-bit + KIVI KV**. The single biggest departure from WMT25 submissions is using MT data — not generic C4 — for the contribution-scoring step itself. That is the most under-explored lever in the current literature, and the one most directly aligned with the user's "IFR's average flow is too blunt" diagnosis.

---

*Verification caveats.* All arXiv IDs in this document were verified via search unless explicitly flagged. PiSSA, rsLoRA, LoRA+, and original LoRA were verified only via secondary citations in this pass; specific numerical claims for PB-LLM, OneBit, BitDistiller, and the exact WikiText-2 PPLs cited for GPTQ/AWQ/SqueezeLLM/QuIP#/HQQ/SpinQuant were not confirmed directly from tables — marked UNVERIFIED where relevant. No system description paper exists for the WMT25 LeanAya submission (authors did not submit one); method inferred from Gaido et al. 2025 §3 and the LeanQuant paper. The WMT25 Findings paper reports XCOMET-XL (0–100), while Vicomtech's own system paper reports wmt22-comet-da (0–1); the two are not directly comparable and both are cited in their native scales above.