# Phase-two primer: find / keep / shrink / prune

Reading list + the conceptual framework behind the `q6-compression`
experiments. This is the bridge from phase one (what/how MT works) to phase
two (a compression method grounded in it).

The central phase-one result that shapes everything here is **Q5: component
*importance* (IFR/DLA, per-head/per-layer) does not predict quantization
*sensitivity*** (ρ≈0). The resolution is *not* "interpretability is useless
for compression" — it is **"we were measuring at the wrong granularity."**
Sensitivity in LLMs is real and extremely concentrated, but it lives at the
**per-weight / per-channel** level, and it is found with **sensitivity-native**
signals, not with task-importance signals. That is what `find` does.

---

## Is `find` an interpretability method?

Partly — but it is a *different question* from phase one, so it uses
*different instruments*. Worth being precise, because this is the crux.

| | phase one (Q1–Q4) | phase two `find` |
|---|---|---|
| **Question** | *Where does the MT computation happen?* (task importance) | *Where does numerical precision / removal actually hurt?* (sensitivity / saliency) |
| **Instruments** | logit lens, probing, IFR, DLA, attribution patching | activation-spike detection, activation magnitude (AWQ), loss-curvature (Hessian/Fisher), |W|·‖X‖ (Wanda) |
| **Granularity** | layer / head / MLP | **scalar weight / input channel** |
| **Q5 verdict** | does **not** predict sensitivity | **is** the sensitivity signal |

Two of the four `find` methods are genuinely mechanistic-interpretability
(super-weight / massive-activation detection is a mechanistic finding about how
the residual stream carries a few load-bearing scalars). The other two
(AWQ activation magnitude, Hessian/Fisher curvature) are **quantization
saliency**, borrowed from the PTQ literature, not interpretability. So: the
`find` stage is *interpretability-adjacent* but its job is to locate
**precision-fragile structure**, which Q5 proved is a different object than the
**MT-important structure** phase one mapped.

---

## The four `find` experiments (what they are, what they output)

All implemented in `compression/src/interp/`, all run on **MT calibration data** (the
under-explored lever — `compression/docs/annotated_bibliography.md` §1D), all summarized by `q6` into
`compression/results/{model}/q6/q6_summary.json`.

### 1. Super-weight / super-activation detection — `super_weights.py`
- **What:** a single forward pass locates the handful (sometimes one) of
  scalar weights in early-layer `mlp.down_proj` whose ablation collapses the
  model. They are found via the enormous activation spikes ("super
  activations") they create, then confirmed by zeroing the one scalar and
  measuring the next-token KL / top-1 prob drop.
- **Granularity:** individual scalar weights `(layer, out_dim, in_dim)`.
- **Read:** **Yu, Bai, Jaiswal et al. (2024), "The Super Weight in Large
  Language Models"** — [arXiv:2411.07191](https://arxiv.org/abs/2411.07191)
  (Apple). The single most important paper for this idea. Pair with
  **Sun, Chen, Wang et al. (2024), "Massive Activations in Large Language
  Models"** — [arXiv:2402.17762](https://arxiv.org/abs/2402.17762) (the
  activation-side view of the same phenomenon; ties to attention sinks).

### 2. AWQ salient channels + MT-vs-generic calibration — `salient_channels.py`
- **What:** AWQ marks ~1% of *input channels* as salient by activation
  magnitude (q99 |x|). We compute this salience under three calibration
  regimes — MT prompt / raw source / raw target — and measure how much the
  top-1% set moves (Jaccard) and how the per-channel ranking shifts
  (Spearman). If MT calibration picks a different salient set, MT-conditional
  AWQ is a real, kernel-free phase-two lever.
- **Granularity:** input channels (columns of each weight matrix).
- **Read:** **Lin et al. (2024), "AWQ: Activation-aware Weight Quantization"**
  — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978). Background on *why*
  a few channels dominate: **Dettmers et al. (2022), "LLM.int8()"** —
  [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) (emergent **outlier
  features**, the original "small fraction does most of the work" result).

### 3. Hessian / Fisher-diagonal sensitivity — `hessian_diag.py`
- **What:** a weight's effect on the *loss* is governed by loss curvature.
  We compute the empirical **Fisher diagonal** `E[(∂L/∂w)²]` of the MT
  target-NLL (one backward pass per calibration example, accumulate g²) per
  module and per layer — the label-aware, MT-conditional sensitivity signal.
  We also expose the **GPTQ-style activation second moment** `E[x_j²]` per
  channel (the activation-only Hessian diagonal) so it lines up with AWQ.
- **Granularity:** per weight (summarized per module / per layer) and per
  channel.
- **Read:** **Frantar et al. (2023), "GPTQ"** —
  [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) (inverse-Hessian
  weighted rounding). **Kim et al. (2024), "SqueezeLLM"** —
  [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) (Hessian-weighted
  k-means + dense-and-sparse outlier split). **Zhang & Shrivastava (2025),
  "LeanQuant"** — [arXiv:2407.10032](https://arxiv.org/abs/2407.10032)
  (loss-error-aware grid; the WMT25 quality winner). Classical roots:
  LeCun et al. *Optimal Brain Damage*; Hassibi & Stork *Optimal Brain Surgeon*.

### 4. Wanda saliency — `compress.py::wanda_mask`
- **What:** the pruning-side saliency `S_ij = |W_ij|·‖X_j‖` — magnitude
  *times* input-activation norm, per output row. Used to choose which weights
  to zero in the `prune` stage.
- **Read:** **Sun et al. (2024), "A Simple and Effective Pruning Approach
  (Wanda)"** — [arXiv:2306.11695](https://arxiv.org/abs/2306.11695).
  Complement: **Frantar & Alistarh (2023), "SparseGPT"** —
  [arXiv:2301.00774](https://arxiv.org/abs/2301.00774) (Hessian-aware
  pruning); **Yin et al. (2024), "OWL"** —
  [arXiv:2310.05175](https://arxiv.org/abs/2310.05175) (non-uniform per-layer
  sparsity budget by outlier density).

---

## keep / shrink / prune — what they mean here

These are the three *actions* a compression method takes on the structure
`find` located. All evaluated in `q6` by **chrF++ on generated translations**
(not target-token logit — the fix for Q5's weak metric), using the faithful
operations in `compression/src/interp/compress.py`, not Gaussian noise.

### SHRINK — quantize the bulk
- **What:** round-to-nearest INT-k (`absmax_quantize`), per-output-channel
  scales, group size 128. chrF++ vs bits ∈ {4,3,2}. The honest base case;
  the quantization *error clips outliers*, which is exactly the structure the
  Q5 Gaussian-noise proxy could not see.
- **In context:** establishes each model's quality cliff. Expect 4-bit ≈
  lossless, 2-bit to fall off — and the *shape* of the fall to differ by
  model (Gemma-family is the Q4 outlier; watch it here too).

### KEEP — protect the fragile minority
- **What:** at the hardest bit-width, compare four ways of protecting the
  `find`-located structure:
  - `rtn` — no protection (baseline),
  - `awq` — per-channel scaling (`quantize_awq_weight`, α=0.5),
  - `keep_salient_fp16` — leave the top-1% AWQ channels in FP16, quantize
    the rest (`keep_cols`),
  - `rtn_plus_superweight_fp16` — quantize everything but restore the handful
    of detected super-weight scalars.
- **In context:** this is the "keep those more stable and shrink the rest"
  idea made concrete. The super-weight variant tests the Apple result's most
  practical claim: **preserving a few scalars recovers most of the quality at
  large block sizes.** If true here, it is nearly free (a handful of FP16
  scalars) and kernel-friendly.

### PRUNE — remove the redundant
- **What:** magnitude vs Wanda at sparsity ∈ {0.25, 0.5}, chrF++.
- **Plus the super-weight stress test:** ablate the **1** detected super weight
  vs ablate the **1000 largest-magnitude** weights in the same matrix. The
  Apple paper's headline: removing the 1 is catastrophic, removing the 1000
  biggest is survivable — i.e. **magnitude is a bad saliency**, the per-weight
  echo of Arctos's Q5 (magnitude/importance ≠ sensitivity).

---

## How this answers the project's open questions

1. **Does MT calibration matter for the salient set?** (`find` #2) — directly
   tests the `compression/docs/annotated_bibliography.md` §1D gap; result is a per-model Jaccard/Spearman.
2. **Is there a sensitivity-native signal that *does* concentrate?** (`find`
   #1, #3) — super weights and Fisher say yes, at per-weight granularity,
   reconciling Q5.
3. **Does protecting it pay off?** (`keep`) — chrF++ deltas quantify it.
4. **Is magnitude a valid saliency?** (`prune` stress test) — expected: no.

## Where the results live
`compression/results/{model}/q6/q6_summary.json` (+ `fisher.npz`). Runner:
`compression/experiments/q6-compression/experiment.py`; SLURM:
`compression/experiments/q6-compression/slurm/` (`submit_all.sh` for the sweep).
Library: `compression/src/interp/{super_weights,salient_channels,hessian_diag,compress}.py`.
Validated on CPU/bloom-560m before the A100 sweep.
