# %% [markdown]
# # Arctos — Overview: interpretability-guided quantization for translation
#
# **Run these notebooks in VS Code** (each `# %%` is a runnable cell) or
# convert with `jupytext --to notebook notebooks/00_overview.py`. They run
# on a tiny CPU model by default so you can step through the mechanics
# without a GPU.
#
# ## The thesis in one paragraph
#
# Large language models translate well but are expensive. **Quantization**
# (storing weights in 4 or 8 bits instead of 16) makes them smaller and
# faster, but naive quantization hurts translation quality. The standard
# quantizers (GPTQ, AWQ) decide *which weights to protect* using generic
# signals computed on generic text (C4, WikiText). Our bet: if we first
# *understand* how a model carries out translation — which layers, which
# attention heads, which MLPs do the MT-critical work — we can allocate
# the bit budget more intelligently, protecting MT-critical components and
# aggressively compressing the rest. This is **interpretability-guided,
# task-specific quantization.**
#
# ## Why "understand first" is the whole point
#
# The prior BYU paper (Castillo & Richardson) pruned layers from Aya using
# Information Flow Routes, then recovered quality with fine-tuning and
# quantized with GPTQ. It worked, but it skipped the question *why* those
# layers were prunable. Two findings from it motivate us:
#
# 1. IFR (an importance signal) disagreed with an iterative ablation
#    heuristic at the most aggressive compression — and the disagreement
#    was **language-pair specific within a single model.** Something about
#    late-layer importance wasn't captured by averaged information flow.
# 2. Both signals concentrated pruning in the middle of the network — the
#    same convergence the broad depth-pruning literature reports on
#    *generic* data. So is the "middle is prunable" finding about MT, or
#    just task-agnostic redundancy?
#
# Phase one (these notebooks' subject) answers "how is MT done inside the
# model" with four interpretability methods. Phase two designs the
# quantization method from those answers.

# %% [markdown]
# ## The five investigative questions (phase one)
#
# | Q  | Question | Methods |
# |----|----------|---------|
# | Q1 | Where does language identity emerge, where does target generation commit? | logit lens, probing, IFR, DLA |
# | Q2 | Which attention heads are MT-critical, and what do they do? | activation/attribution patching |
# | Q3 | Which MLPs and layers carry the cross-lingual mapping? | layer patching, IFR |
# | Q4 | How does the MT footprint differ across architectures? (shared-depth hypothesis) | synthesis across Q1-Q3 |
# | Q5 | Of MT-critical components, which are quantization-sensitive? | noise injection, AWQ stats, Hessian |
#
# Q5 is the bridge: it tests whether the components our methods flag as
# *important* are the same ones that are *fragile under quantization*. If
# yes, interpretability-guided quantization is principled. If no — that's
# itself a finding (importance ≠ sensitivity).

# %% [markdown]
# ## The model set (chosen for the quantization story)
#
# We need the method to generalize across generations and architectures,
# so the set spans 2022→2025, multiple positional encodings, normalizations,
# and training intents:
#
# | Model | Year | Arch quirk | Training intent |
# |-------|------|-----------|-----------------|
# | Aya Expanse 8B | 2024 | Cohere, RoPE, RMSNorm | general multilingual LM |
# | TowerBase 7B | 2024 | Llama-2 | + bilingual CPT (no SFT) |
# | TowerInstruct 7B | 2024 | Llama-2 | + MT-task SFT |
# | Tower-Plus 9B | 2025 | **Gemma 2** (extra layernorms, softcap) | CPT + MT-SFT |
# | BLOOM 7B1 | 2022 | **ALiBi, LayerNorm** | multilingual pretraining |
# | EuroLLM 9B | 2024 | Llama-3 | European multilingual specialist |
# | Llama-3.1 8B | 2024 | Llama-3, GQA | general LM |
# | Gemma-3-12B | 2025 | Gemma 3 | **baseline only** (Google QAT — we don't compete) |
#
# The TowerBase→TowerInstruct pair is a controlled ablation: same base,
# the only difference is MT-SFT, so it isolates what SFT changes.

# %% [markdown]
# ## The four core methods, and what each one *is*
#
# All four read the **residual stream** — the running sum that flows down
# a transformer. At each layer, attention and MLP blocks *add* their output
# to this stream: `resid_post[L] = embed + Σ attn_out[ℓ] + Σ mlp_out[ℓ]`.
# Interpretability is largely the art of reading and intervening on this sum.
#
# | Method | Question it answers | Signed? | Causal? | File |
# |--------|--------------------|---------|---------|------|
# | **Logit lens** | "If we decoded from layer ℓ now, what token?" | — | no | `01_logit_lens.py` |
# | **Probing** | "Is feature X linearly decodable at layer ℓ?" | — | no | `02_probing.py` |
# | **IFR** | "How much does component c contribute (magnitude)?" | no (≥0) | no | `03_ifr.py` |
# | **DLA** | "Does component c push toward/away the target token?" | yes (±) | no | `04_dla.py` |
# | **Attribution patching** | "If I corrupt component c, how much does output change?" | yes (±) | **yes** | `05_attribution_patching.py` |
# | **Noise sensitivity** | "If I quantize component c, how much quality drops?" | — | yes | `06_noise_sensitivity.py` |
# | **AWQ stats** | "Which weight columns see large activations?" | — | — | `07_awq_activation_stats.py` |
#
# Correlational methods (lens, probing, IFR, DLA) are cheap and tell you
# *where* things are. Causal methods (patching, noise) are the ground truth
# but expensive. The research strategy: use cheap methods to find candidates,
# use causal methods to verify, and use noise sensitivity (Q5) to connect it
# all to the actual quantization decision.

# %%
# Quick environment check — confirms the repo imports work and shows you
# the tiny model the tutorials use by default.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models._hooked import GPT2_PATHS, HookedModel

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
model = HookedModel(hf, tok, GPT2_PATHS)
print(f"loaded gpt2 via HookedModel: {model.cfg.n_layers} layers, "
      f"{model.cfg.n_heads} heads, d_model={model.cfg.d_model}")
print("\nThe HookedModel wrapper gives every method a uniform interface")
print("(run_with_cache, W_U, W_O, ln_final) across all 8 architectures.")
print("Next: open 01_logit_lens.py")
