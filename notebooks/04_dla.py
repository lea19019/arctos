# %% [markdown]
# # 04 — Direct Logit Attribution (DLA)
#
# **Question it answers:** does component c push the prediction *toward* or
# *away from* the gold target token — and by how much, in logit units?
#
# ## Theory
#
# IFR tells you a component is *loud*. DLA tells you a component is *useful*
# (or harmful). The difference is sign.
#
# The final logit for token `t` is a linear readout of the final residual:
#
# ```
# logit[t] = (LN_final(resid_post[L-1]) @ W_U)[:, t]
# ```
#
# Because the residual is an additive sum of component outputs `c`, and the
# unembed is linear, we can ask how much *each component* contributes to that
# one logit. The only nonlinearity is the final LayerNorm/RMSNorm. We
# **linearize** it at the operating point: a norm just rescales each feature
# by a fixed factor `s` (computed from the actual final residual), so for a
# component output vector `c`:
#
# ```
# DLA(c → t) ≈ (c ⊙ s) · W_U[:, t]
# ```
#
# Per head, `c = z[ℓ,h] @ W_O[ℓ,h]`. The result is **signed**: positive means
# component c writes in a direction that increases the target token's logit
# (a "predictor"); negative means it decreases it (a "suppressor").
#
# ## Why it matters for quantization
#
# DLA is the sharpest "which heads matter for THIS output" signal we have.
# A head with large positive DLA is directly responsible for producing the
# right target token — protect it. A head with near-zero DLA contributes
# nothing to the prediction — quantize it hard. This is exactly the
# per-component bit-allocation signal the method needs.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.dla import dla
from src.models._hooked import GPT2_PATHS, HookedModel

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
model = HookedModel(hf, tok, GPT2_PATHS)

# %% [markdown]
# ## Mechanic: signed per-head contributions to the target logit
#
# DLA takes `(prompt, target_token_ids)` pairs. We attribute the logit of
# " Paris" after the Eiffel-Tower prompt.

# %%
prompt = "The Eiffel Tower is located in the city of"
target_ids = tok(" Paris", add_special_tokens=False)["input_ids"]
scores = dla(model, [(prompt, target_ids)])

print(f"head_scores shape {tuple(scores.head_scores.shape)}  (signed)")
print(f"layer_attn = head sum?  {torch.allclose(scores.head_scores.sum(-1), scores.layer_attn, atol=1e-3)}")

flat = scores.head_scores.flatten()
H = scores.head_scores.shape[1]
print("\nmost POSITIVE heads (push toward 'Paris'):")
for v, i in zip(flat.topk(4).values.tolist(), flat.topk(4).indices.tolist()):
    print(f"  L{i//H}.H{i%H}: {v:+.3f}")
print("most NEGATIVE heads (push away):")
bot = (-flat).topk(4)
for i in bot.indices.tolist():
    print(f"  L{i//H}.H{i%H}: {flat[i]:+.3f}")

# %% [markdown]
# ## IFR vs DLA on the same heads — the sign is the whole point
#
# A head can be *loud* (high IFR) but *neutral or harmful* (≈0 or negative
# DLA). Let's put them side by side: rank by IFR, then show each head's DLA.

# %%
from src.interp.ifr import ifr

ifr_scores = ifr(model, [prompt], target_position="last")
ifr_flat = ifr_scores.head_scores.flatten()
print("rank-by-IFR :  IFR(magnitude)   DLA(signed)")
for i in ifr_flat.topk(6).indices.tolist():
    print(f"  L{i//H}.H{i%H:2d} :   {ifr_flat[i]:.4f}        {flat[i]:+.3f}")
print("\nNote how some high-magnitude (IFR) heads have small or negative DLA —")
print("they're active but not pushing toward the answer. That's why we need both.")

# %% [markdown]
# ## What the real Aya results showed
#
# DLA found **recurring cross-pair heads**: L30.H8 is a strong "target
# predictor" (+0.72 on cs→de, +0.44 on en→arz); L29.H16 and L26.H5 are
# systematic "suppressors" (negative across multiple pairs). These appear
# regardless of language pair — candidate MT-circuit components.
#
# The cross-model DLA depth curve (`results/_combined/q1/dla_layer_combined.png`)
# shows Aya accumulating positive DLA across its last ~10 layers, while Tower
# models stay flat until the final 1-2 layers then jump — the same
# "distributed vs deferred commitment" split the logit lens showed, now
# confirmed by a second, signed method.
#
# **LayerNorm caveat:** for BLOOM (LayerNorm, not RMSNorm) the linearization
# omits the centering term, so absolute DLA values are biased — use BLOOM's
# DLA *rankings*, not its magnitudes, when comparing to RMSNorm models.
#
# **The Q5 connection:** in notebook 06 we take these DLA-top heads and
# actually perturb them with quantization-like noise to test whether
# "high DLA" really means "fragile under quantization." That's the
# experiment that turns this correlational signal into a quantization rule.
