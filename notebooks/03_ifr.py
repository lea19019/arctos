# %% [markdown]
# # 03 — Information Flow Routes (IFR)
#
# **Question it answers:** how much does each component (attention head, MLP)
# *contribute in magnitude* to the residual stream at the position we care
# about?
#
# ## Theory
#
# Recall the residual stream is an additive sum of component outputs:
#
# ```
# resid_post[L-1] = embed + Σ_ℓ attn_out[ℓ] + Σ_ℓ mlp_out[ℓ]
# ```
#
# and `attn_out[ℓ] = Σ_h (z[ℓ,h] @ W_O[ℓ,h])` — the layer's attention output
# is itself a sum over heads, where `z[ℓ,h]` is head h's pre-output-projection
# activation and `W_O[ℓ,h]` is its slice of the output projection.
#
# IFR (Ferrando & Voita, 2024) measures each component's **contribution
# magnitude**: take the L1 norm (sum of absolute values) of each component's
# output vector at the target position, then **L1-normalize across all
# components** so they sum to 1 per token. Average over a calibration set.
#
# ```
# contrib(head ℓ,h) = || z[ℓ,h] @ W_O[ℓ,h] ||₁
# contrib(mlp ℓ)    = || mlp_out[ℓ] ||₁
# contrib(embed)    = || embed ||₁
# IFR(c) = mean over examples of  contrib(c) / Σ_c' contrib(c')
# ```
#
# This is a **magnitude** measure — always non-negative. It tells you which
# components are *loud* (have big outputs), not which ones push the
# prediction in a useful direction (that's DLA, notebook 04).
#
# ## Why it matters for quantization
#
# Low-IFR components contribute little to the residual, so quantizing them
# aggressively costs little. High-IFR components are doing heavy lifting and
# need protection. The *depth profile* of IFR (which layers dominate) is a
# candidate for a model-agnostic bit-budget prior — that's the Q4 question.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.ifr import ifr
from src.models._hooked import GPT2_PATHS, HookedModel

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
model = HookedModel(hf, tok, GPT2_PATHS)

# %% [markdown]
# ## The core invariant: contributions L1-normalize to 1
#
# Per example, the head + MLP + embed contributions sum to 1 by construction.
# Averaged over examples it stays ~1. This is what makes IFR scores
# comparable across components and models.

# %%
prompts = ["The capital of France is", "Cats sit on the mat.", "Water boils at one hundred"]
scores = ifr(model, prompts, target_position="last")
total = scores.layer_scores.sum().item() + scores.embed_score
print(f"Σ layer_scores + embed = {total:.4f}   (should be ~1.0)")
print(f"layer_scores shape {tuple(scores.layer_scores.shape)}, "
      f"head_scores {tuple(scores.head_scores.shape)}, mlp {tuple(scores.mlp_scores.shape)}")

# %% [markdown]
# ## The depth profile: where does contribution magnitude concentrate?

# %%
ls = scores.layer_scores
print("layer : IFR contribution (attn+mlp)")
for layer in range(model.cfg.n_layers):
    print(f"  {layer:2d}  : {ls[layer]:.4f}  {'#' * int(ls[layer] * 200)}")
print(f"\nembed: {scores.embed_score:.4f}")
print(f"first-quarter share : {ls[:model.cfg.n_layers//4].sum() / ls.sum():.3f}")
print(f"last-quarter share  : {ls[-model.cfg.n_layers//4:].sum() / ls.sum():.3f}")

# %% [markdown]
# ## Per-head heatmap (the most-contributing heads)

# %%
hs = scores.head_scores  # (L, H)
flat = hs.flatten()
topk = flat.topk(5)
print("top-5 heads by IFR magnitude:")
for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f"  L{i // hs.shape[1]}.H{i % hs.shape[1]}: {v:.4f}")

# %% [markdown]
# ## What the real results showed (the Q4 headline)
#
# Across 6 of 7 models — Aya (Cohere), TowerBase/TowerInstruct (Llama-2),
# BLOOM (ALiBi+LayerNorm), EuroLLM (Llama-3), Llama-3.1 — the IFR depth
# profile is consistent: first quarter ≤12% of mass, middle half ~40-50%,
# last quarter dominant ~45-58%, top-5 layers always in the last ~20% of
# depth. This holds across two normalizations, two positional encodings, and
# 2022→2024 generations. **That generalization is the core evidence that a
# per-depth bit budget could be model-agnostic.**
#
# **Tower-Plus (Gemma 2 base) is the lone exception** — ~30% first-quarter
# mass, ~18% last-quarter. Gemma's extra pre/post-feedforward layernorms +
# logit softcapping rescale residual magnitudes, breaking the pattern. So the
# depth prior is *family-bounded*, not universal — a method using it must be
# tested on Gemma-derived models separately.
#
# **Caveat:** IFR is magnitude-only, so late layers (where the residual norm
# has grown) are naturally favored. The *shape* across layers is the robust
# signal, not the absolute argmax. DLA (next) gives the signed, direction-aware
# complement.
