# %% [markdown]
# # 01 — Logit Lens
#
# **Question it answers:** at each layer, if the model had to commit to an
# output token *right now* from its current hidden state, what would it say?
#
# ## Theory
#
# A decoder transformer keeps a running vector per token position called the
# **residual stream**, `resid ∈ ℝ^d_model`. Each block reads it, computes
# something, and *adds* the result back:
#
# ```
# resid_post[ℓ] = resid_post[ℓ-1] + attn_out[ℓ] + mlp_out[ℓ]
# ```
#
# At the very end, the model converts the final residual to a probability
# distribution over the vocabulary with two operations:
#
# ```
# logits = LayerNorm_final(resid_post[L-1]) @ W_U     # W_U: (d_model, vocab)
# probs  = softmax(logits)
# ```
#
# The **logit lens** (nostalgebraist, 2020) asks: what if we apply that same
# final operation to an *intermediate* residual `resid_post[ℓ]`? Because the
# residual stream is additive and the unembed `W_U` is linear, this is a
# meaningful question — it reads out "the model's current best guess at
# layer ℓ." Early layers usually give garbage; the prediction sharpens as
# you go deeper. Where it sharpens tells you *where the model decides*.
#
# ## Why it matters for MT and quantization
#
# For translation we track the probability mass the lens puts on the **gold
# target-language tokens** at each layer. The layer where that mass starts
# climbing is where the model "commits" to producing the target language.
# Those late commitment layers are prime suspects for being
# quantization-sensitive: break them and the model loses the target token.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.logit_lens import logit_lens
from src.models._hooked import GPT2_PATHS, HookedModel

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
model = HookedModel(hf, tok, GPT2_PATHS)

# %% [markdown]
# ## The core invariant: the last-layer lens == the model's real logits
#
# This is the sanity check every logit-lens implementation must pass. If we
# decode from the *final* layer, we must reproduce the model's actual output
# exactly (it's literally the same computation). Let's verify.

# %%
prompt = "The Eiffel Tower is located in the city of"
result = logit_lens(model, prompt)
with torch.no_grad():
    real_logits = model(model.to_tokens(prompt))[0, -1]
diff = (real_logits.float() - result.layer_logits[-1].float()).abs().max().item()
print(f"max |lens[last] - real logits| = {diff:.2e}   (should be ~0)")
print(f"layer_logits shape = {tuple(result.layer_logits.shape)}  (n_layers, vocab)")

# %% [markdown]
# ## Watch the prediction sharpen across depth
#
# Decode the top token at each layer. Early layers predict nonsense; the
# right answer ("Paris") emerges only in the last few layers.

# %%
for layer in range(model.cfg.n_layers):
    top_id = result.layer_logits[layer].argmax().item()
    top_tok = tok.decode([top_id])
    bar = "#" * int(result.layer_logits[layer].softmax(-1).max() * 40)
    print(f"  layer {layer:2d}: {top_tok!r:15s} {bar}")

# %% [markdown]
# ## Tracking target-token mass (the MT use)
#
# In the real Q1 experiment we pass `target_tokens` = the gold translation's
# first few token ids, and read `target_token_mass[ℓ]` — the probability the
# lens assigns to those tokens at each layer. Averaged over 200 examples,
# this is the "when does the model commit to the target language" curve.
#
# Here's the mechanic on a single example (English continuation, since gpt2
# isn't a translator — the *shape* is what to notice):

# %%
target_ids = tok(" Paris", add_special_tokens=False)["input_ids"]
res = logit_lens(model, prompt, target_tokens=target_ids)
print(f"target token: {tok.decode(target_ids)!r}  ids={target_ids}")
print("\nlayer : P(target) under the lens")
for layer in range(model.cfg.n_layers):
    mass = res.target_token_mass[layer].item()
    print(f"  {layer:2d}  : {mass:.4f}  {'#' * int(mass * 60)}")

# %% [markdown]
# ## What the real Aya results showed
#
# On Aya-8B, en→zh: target mass is ~0 for the first ~18 layers, then climbs
# from 0.003 (L18) to 0.091 (L31). cs→de and en→arz stay near zero (Aya
# distributes probability over many German/Arabic paraphrases rather than
# the exact gold prefix). The cross-model chart
# `results/_combined/q1/lens_combined.png` shows Aya commits in a late
# *band* of layers, while Tower models defer commitment to the final 1-2
# layers — a real architectural fingerprint.
#
# **Limitation:** the vanilla logit lens is noisy in early/mid layers because
# the residual there isn't yet in the "coordinate system" the unembed
# expects. The *tuned lens* (notebook 09, optional) trains a per-layer affine
# correction to clean this up. For commitment-layer detection the vanilla
# lens is enough.
