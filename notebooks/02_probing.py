# %% [markdown]
# # 02 — Probing Classifiers
#
# **Question it answers:** is some feature (e.g. "the source language is
# Czech") *linearly decodable* from the residual stream at layer ℓ?
#
# ## Theory
#
# Train a small classifier (usually just linear: `softmax(W·resid + b)`) to
# predict a label from the hidden state at one layer. If it succeeds, the
# information is "there" and linearly accessible. Run one probe per layer →
# you get a depth profile of *where* a feature becomes readable.
#
# ### The trap: probe accuracy alone is meaningless
#
# A probe with enough capacity can fit almost anything — high accuracy might
# mean "the model represents this feature" OR "the probe memorized the
# training set." Hewitt & Liang (2019) fix this with a **control task**:
# train an identical probe to predict *random* labels (a fixed random label
# per input). A probe that's truly reading a feature will do well on the real
# task and poorly on the random one. The gap is what matters:
#
# ```
# selectivity = accuracy(real labels) − accuracy(random control labels)
# ```
#
# Report **selectivity**, never raw accuracy. High accuracy + high control
# accuracy = expressive probe, not a represented feature. High accuracy + low
# control accuracy = the feature is genuinely there.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.probing import probe_layers, train_probe
from src.models._hooked import GPT2_PATHS, HookedModel

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
model = HookedModel(hf, tok, GPT2_PATHS)

# %% [markdown]
# ## Mechanic 1: a probe on a trivially separable feature reaches ~100%
#
# Sanity check the probe trainer itself. Make a 2-class dataset where class 0
# lives at +e₀ and class 1 at −e₀ — a linear probe must nail it.

# %%
torch.manual_seed(0)
n, d = 64, 16
x = torch.zeros(n, d)
y = torch.zeros(n, dtype=torch.long)
x[: n // 2, 0] = 1.0
x[n // 2 :, 0] = -1.0
y[n // 2 :] = 1
x += 0.01 * torch.randn(n, d)
probe = train_probe(x, y, n_classes=2, epochs=200)
acc = (probe(x).argmax(-1) == y).float().mean().item()
print(f"probe accuracy on a linearly separable feature: {acc:.3f}  (expect ~1.0)")

# %% [markdown]
# ## Mechanic 2: per-layer probing with the control task
#
# `probe_layers` caches `resid_post` at the last token of each prompt, then
# for every layer trains (a) a real probe and (b) a control probe on shuffled
# labels, and returns accuracy / control_accuracy / selectivity.
#
# Toy example: "is this sentence about animals or cities?" — gpt2 won't be
# great at it, but you'll see the selectivity machinery working.

# %%
examples = [
    ("The cat sat on the warm windowsill", 0),
    ("A dog barked loudly at the mailman", 0),
    ("Birds migrate south for the winter", 0),
    ("The horse galloped across the field", 0),
    ("Paris is the capital of France", 1),
    ("Tokyo has the busiest train station", 1),
    ("London sits on the river Thames", 1),
    ("Cairo lies beside the great river Nile", 1),
] * 3  # repeat for a slightly bigger train/test split

results = probe_layers(model, examples, n_classes=2, layers=[0, 4, 8, 11])
print("layer :  acc  ctrl  selectivity")
for r in results:
    print(f"  {r.layer:2d}  : {r.accuracy:.2f}  {r.control_accuracy:.2f}   {r.selectivity:+.2f}")

# %% [markdown]
# ## What the real Q1 results showed
#
# We ran two probes per model:
#
# - **target-language ID** at the end of the MT prompt. This is *leaky* — the
#   prompt literally says "into German", so accuracy ≈ 1.0 at every layer.
#   Useful only as a "where is this most cleanly encoded" signal via
#   selectivity, not as evidence of computation.
# - **source-language ID** at the end of *raw source text* (no instruction).
#   Clean. Czech vs English, 400 balanced examples.
#
# Across 7 models, source-ID selectivity sits at 0.40–0.55 at *every* layer
# (it's an easy binary task), peaking at L0 for some models (TowerBase,
# TowerInstruct, Llama-3.1, BLOOM — embedding already separates the
# languages) and mid-network for others (Aya L17, EuroLLM L9). The peak
# location is a fingerprint of training; the magnitude is uniform.
#
# **Quantization takeaway:** language identity is decodable from the
# embedding in most models, so the embedding matrix carries real
# task-relevant information — quantize it carefully regardless of model.
