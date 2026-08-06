# %% [markdown]
# # 07 — AWQ-style Activation Statistics (the baseline we compete with)
#
# **Question it answers:** for each weight matrix, which input channels see
# the largest activations? Those channels' weights matter most for output
# fidelity and should be protected during quantization.
#
# ## Theory
#
# A linear layer computes `y = W x`. When we quantize `W → Ŵ`, the output
# error is `(Ŵ − W) x`. The error in output dimension i is
# `Σ_j (Ŵ−W)_ij x_j` — so an error in weight column j is amplified by the
# magnitude of input `x_j`. **Columns paired with large-magnitude activations
# dominate the output error.**
#
# AWQ (Lin et al. 2023, "Activation-aware Weight Quantization") turns this
# into a method: find the ~1% of weight channels with the largest paired
# activations (measured on a calibration set) and protect them (keep higher
# precision, or scale them to be quantization-friendly). It recovers most of
# the quality lost to naive quantization, using *only activation statistics*
# — no gradients, no interpretability.
#
# `compression/src/interp/activation_stats.py` collects, per Linear module, the
# per-input-channel statistics over a calibration set:
# - `max_abs[j]`  = max over tokens of |x_j|
# - `mean_abs[j]` = mean over tokens of |x_j|
# - `q99_abs[j]`  = 99th percentile of |x_j|  (robust "salient channel" signal)
#
# ## Why it's in this project
#
# AWQ is the **strong baseline our interpretability-guided method must
# beat or complement.** Two honest possibilities:
# 1. AWQ's activation-magnitude signal already captures everything our DLA /
#    IFR signals would — then interpretability adds understanding but not
#    compression gains.
# 2. MT-critical components (DLA/IFR) are *partly orthogonal* to
#    high-activation channels (AWQ) — then combining them beats either alone,
#    especially with MT-specific calibration data instead of generic C4.
#
# Q5 measures which. AWQ stats computed on **MT calibration data** (not C4)
# are also a contribution on their own — "MT-specific AWQ."

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.activation_stats import collect_activation_stats
from src.models._hooked import GPT2_PATHS, HookedModel

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
model = HookedModel(hf, tok, GPT2_PATHS)

# %% [markdown]
# ## Mechanic: collect per-channel activation stats over a calibration set

# %%
calib = [
    "The capital of France is Paris.",
    "Cats sit quietly on the warm mat.",
    "Water boils at one hundred degrees.",
    "She translated the document into German.",
]
stats = collect_activation_stats(model, calib)
print(f"profiled {len(stats.stats_by_module)} Linear modules over {stats.n_examples} examples\n")

# Look at one MLP input projection: the salient-channel distribution
name = "blocks.0.mlp.c_fc"
s = stats.stats_by_module[name]
print(f"{name}: in_features={s.in_features}, tokens seen={s.n_tokens}")
print(f"  max_abs : min={s.max_abs.min():.2f}  median={s.max_abs.median():.2f}  max={s.max_abs.max():.2f}")
print(f"  q99_abs : min={s.q99_abs.min():.2f}  median={s.q99_abs.median():.2f}  max={s.q99_abs.max():.2f}")

# %% [markdown]
# ## The "salient channels" — the few that AWQ would protect
#
# The whole AWQ insight is that activation magnitude is *heavy-tailed*: a
# small fraction of channels are far larger than the rest. Those are the ones
# worth protecting. Let's see the tail.

# %%
q99 = s.q99_abs
order = q99.argsort(descending=True)
ratio = (q99 > 3 * q99.median()).float().mean().item()
print(f"channels with q99 > 3x median: {ratio*100:.1f}%  (the heavy tail AWQ targets)")
print("\ntop-8 salient input channels (index : q99 |activation|):")
for idx in order[:8].tolist():
    print(f"  ch {idx:4d} : {q99[idx]:.2f}")

# %% [markdown]
# ## In the real pipeline
#
# The Q5 runner saves per-module `q99_abs` vectors to `awq_stats_{pair}.npz`
# for every model, computed on the MT calibration prompts. Phase two will:
# 1. compare the AWQ salient-channel map against the DLA/IFR component
#    importance — are they the same components or different?
# 2. build the bit-budget from whichever signal (or combination) best
#    predicts the Q5 noise sensitivity.
#
# This is where "interpretability-guided" meets "the standard quantization
# toolkit" — and where the paper's contribution gets decided.
