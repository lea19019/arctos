# %% [markdown]
# # 05 — Attribution Patching (causal)
#
# **Question it answers:** if I *causally* intervene on component c — swap its
# activation from a "corrupt" run into a "clean" run — how much does the
# output change? This is the gold-standard circuit-discovery question, made
# cheap.
#
# ## Theory
#
# **Activation patching** (Vig 2020; Meng 2022, ROME) is the causal workhorse
# of interpretability:
#
# 1. Run the model on a **clean** prompt (source it translates correctly).
# 2. Run it on a **corrupt** prompt (source altered so the gold target
#    differs).
# 3. Re-run clean, but **overwrite** component c's activation with the value
#    from the corrupt run. Measure how much the clean output degrades.
#
# Large degradation ⇒ c causally matters. But doing this for every head means
# one forward pass *per component* — thousands of passes.
#
# **Attribution patching** (Nanda 2023; Syed et al. 2024) approximates all of
# them with a first-order Taylor expansion. With `L` = the metric (gold-target
# logit) and `a_c` = clean activation:
#
# ```
# patch_effect(c) ≈ ∂L/∂a_c · (a_c^corrupt − a_c^clean)
# ```
#
# One forward + one backward on the clean run gives `∂L/∂a_c` for **every**
# component simultaneously. Then it's just a dot product with the
# (corrupt − clean) difference. ~100× faster than real patching, and it
# agrees well in practice.
#
# ## Why it matters
#
# Correlational methods (IFR, DLA) say "this head looks important." Patching
# says "if you damage this head, the translation breaks." That causal claim is
# what justifies protecting a component during quantization. Q5 (noise) is the
# even-more-direct version, but patching localizes the *circuit*.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.attribution_patching import attribution_patch
from src.models._hooked import GPT2_PATHS, HookedModel

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
model = HookedModel(hf, tok, GPT2_PATHS)

# %% [markdown]
# ## Mechanic: clean vs corrupt, same length
#
# Attribution patching needs clean & corrupt prompts of equal token length
# (so activations align position-by-position). Classic minimal pair: same
# template, one swapped entity that changes the answer.
#
# - clean:   "The Eiffel Tower is in the city of"  → Paris
# - corrupt: "The Colosseum is in the city of"     → Rome
#
# We attribute the **clean** target (" Paris"). A head with large effect is
# one that moved information about *which* landmark into the prediction.

# %%
clean = "The Eiffel Tower is in the city of"
corrupt = "The Colosseum is in the city of"
cl = tok(clean, return_tensors="pt").input_ids
co = tok(corrupt, return_tensors="pt").input_ids
print(f"clean len {cl.shape[-1]}, corrupt len {co.shape[-1]} "
      f"({'aligned' if cl.shape == co.shape else 'MISMATCH — would be skipped'})")

target_ids = tok(" Paris", add_special_tokens=False)["input_ids"]
attr = attribution_patch(model, [(clean, corrupt, target_ids)])
print(f"head_effects {tuple(attr.head_effects.shape)}, mlp_effects {tuple(attr.mlp_effects.shape)}")

H = attr.head_effects.shape[1]
flat = attr.head_effects.flatten()
print("\ntop heads by |causal effect|:")
for i in flat.abs().topk(5).indices.tolist():
    print(f"  L{i//H}.H{i%H:2d}: {flat[i]:+.3f}")

# %% [markdown]
# ## Cross-check against DLA on the same example
#
# DLA (correlational) and attribution patching (causal) should *broadly*
# agree on which heads matter — when they disagree, that's scientifically
# interesting (a head whose output correlates with the answer but isn't
# causally responsible, or vice versa). The validation matrix in
# PHASE1-PLAN.md requires reporting such disagreements, not hiding them.

# %%
from src.interp.dla import dla

dla_scores = dla(model, [(clean, target_ids)])
dla_flat = dla_scores.head_scores.flatten()
print("head      DLA(signed)   patch(causal)")
for i in flat.abs().topk(6).indices.tolist():
    print(f"  L{i//H}.H{i%H:2d}    {dla_flat[i]:+.3f}        {flat[i]:+.3f}")

# %% [markdown]
# ## In the real pipeline (Q2 + Q5)
#
# The Q5 runner builds clean/corrupt pairs with `compression/src/data/clean_corrupt.py`
# (LEXICAL_SUB: swap one content word in the source for an in-distribution
# word from another sentence) and runs attribution patching over the
# calibration set, saving a per-(layer, head) causal map to
# `attribution_{pair}.npz`. That map is the causal cross-check on the DLA
# rankings — and feeds the same "which heads to protect" decision.
#
# **Gotcha worth knowing:** clean/corrupt must tokenize to the same length.
# Source-level word swaps usually preserve length but not always; the runner
# skips mismatched pairs rather than misaligning positions. If too many get
# skipped, the corrupt generator needs length-aware substitution.
