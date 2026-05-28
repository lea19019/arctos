# %% [markdown]
# # 06 — Quantization-Noise Sensitivity (Q5: the bridge to the method)
#
# **Question it answers:** if we degrade a component's weights the way
# quantization would, how much does translation quality drop? And does that
# drop correlate with the importance our interpretability methods predicted?
#
# This is **the load-bearing experiment** for the whole project. Everything
# before it is "this component looks important." This is "this component is
# fragile under compression" — and whether those two are the same thing.
#
# ## Theory
#
# Quantization replaces a weight `w` (16-bit) with a low-precision
# approximation `ŵ` (4 or 8-bit). The error `ŵ − w` behaves like added noise
# whose scale is set by the quantization step. We *simulate* this directly:
# add Gaussian noise to a component's weights, scaled to a chosen relative
# magnitude σ:
#
# ```
# W_perturbed = W + σ · ||W||₂ · ε ,   ε ~ N(0, I / numel(W))
# ```
#
# so the perturbation has relative norm σ. Sweep σ ∈ {0.01 … 0.5}, and for
# each, measure the drop in the gold-target-token logit. Steeper drop = more
# sensitive = needs more bits.
#
# ### The experiment design (importance vs sensitivity)
#
# We form three groups of heads using the model's own DLA ranking:
# - **top-DLA**: predicted critical
# - **bottom-DLA**: predicted unimportant
# - **random**: control
#
# Then the headline test: **Spearman correlation between per-head |DLA| and
# per-head logit-drop.** Positive ρ ⇒ interpretability importance predicts
# quantization sensitivity ⇒ the guided bit-budget is justified.
#
# ## Why a context manager matters
#
# We perturb weights *in place* and must restore them exactly between
# measurements, or the noise accumulates. `src/interp/sensitivity.py` uses a
# context manager that clones the original, adds noise, yields, then copies
# the original back — guaranteeing each measurement is independent.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.sensitivity import head_sensitivity, mlp_sensitivity, _mean_target_logit
from src.models._hooked import BLOOM_PATHS, HookedModel

# Use bloom-560m: small, on CPU, and its o_proj is a real nn.Linear so
# head_sensitivity works (gpt2's Conv1D attention isn't supported by the
# head-slice path).
tok = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
hf = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", dtype=torch.float32).eval()
model = HookedModel(hf, tok, BLOOM_PATHS)

examples = [
    ("The capital of France is", tok(" Paris", add_special_tokens=False)["input_ids"]),
    ("Two plus two equals", tok(" four", add_special_tokens=False)["input_ids"]),
]

# %% [markdown]
# ## Mechanic 1: the sensitivity curve for one head
#
# As σ grows, the gold-token logit drops. The *slope* is the sensitivity.

# %%
r = head_sensitivity(model, examples, layer=12, head=5, sigmas=(0.01, 0.05, 0.1, 0.2, 0.5))
print(f"component {r.component}")
print("sigma : logit drop")
for s, d in zip(r.sigmas, r.logit_drops):
    print(f"  {s:.2f} : {d:+.4f}")

# %% [markdown]
# ## Mechanic 2: weights are restored exactly (no accumulation)
#
# Critical correctness check — measure baseline, run a noisy sweep, measure
# baseline again. They must match.

# %%
b1 = _mean_target_logit(model, examples[0][0], examples[0][1])
_ = head_sensitivity(model, examples, layer=12, head=5, sigmas=(0.5, 0.5, 0.5))
b2 = _mean_target_logit(model, examples[0][0], examples[0][1])
print(f"baseline before sweep: {b1:.5f}")
print(f"baseline after  sweep: {b2:.5f}")
print(f"restored exactly: {abs(b1 - b2) < 1e-5}")

# %% [markdown]
# ## Mechanic 3: compare a few heads — sensitivity varies
#
# Different heads have very different fragility. That variation is exactly
# what a per-component bit budget exploits.

# %%
print("head      drop@σ=0.5")
for (l, h) in [(0, 0), (6, 3), (12, 5), (18, 10), (23, 15)]:
    rr = head_sensitivity(model, examples, layer=l, head=h, sigmas=(0.5,))
    print(f"  L{l}.H{h:2d}    {rr.logit_drops[0]:+.4f}")

# %% [markdown]
# ## The real Q5 experiment (running on the cluster now)
#
# `experiments/q5-importance-vs-sensitivity/experiment.py` does this at scale
# per model:
# 1. rank heads by |DLA| from that model's Q1 results,
# 2. noise-sweep top-DLA / bottom-DLA / random groups (12 heads each),
# 3. report **Spearman(|DLA|, drop@σ=0.5)** and the group-mean drops,
# 4. attribution-patch (causal cross-check) and AWQ stats alongside.
#
# Read `results/{model}/q5/q5_summary.json` for each model's headline:
# - `spearman_absdla_vs_noisedrop`: is importance predictive of sensitivity?
# - `top_vs_random_ratio`: how much more does damaging a top-DLA head hurt vs
#   a random head?
#
# **The interpretation fork:**
# - ρ > 0 and ratio > 1 → DLA-guided bit allocation is justified. Build the
#   method around protecting high-DLA components.
# - ρ ≈ 0 → importance and sensitivity are *different* axes (the prior paper's
#   warning). The method then needs a sensitivity-native signal (Hessian /
#   AWQ), and interpretability is for understanding, not allocation. Either
#   outcome is a real result.
