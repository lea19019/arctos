"""The models the super-weight sweep runs on (shared by prefetch_models.py
and run_all.py). Comment out what you can't access or fit.

All of these appear in Yu et al. 2024 Table 2 (see TABLE2 in ablate_sw.py),
so every run doubles as a check of the paper's directory.
"""

MODELS = [
    "huggyllama/llama-7b",          # the original LLaMA-1 7B
    "huggyllama/llama-13b",       # ~26 GB in bf16 — enable on a big GPU
    "huggyllama/llama-30b",
    "meta-llama/Llama-2-7b-hf",     # gated: needs `hf auth login` first
    "meta-llama/Llama-2-13b-hf",  # gated + large
    "mistralai/Mistral-7B-v0.1",
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
    "microsoft/Phi-3-mini-4k-instruct"
]


# Models NOT in Yu et al. Table 2 — i.e. with no published answer key.
#
# ⚠️ Read this before believing a negative here. Every threshold in
# detect_sw.py was tuned against Table 2, where the right answer was known.
# On these models that safety net is gone, so the two outcomes are NOT
# symmetric: a catastrophic ablation is trustworthy (the causal test does
# not depend on our thresholds), but "nothing found" is uninterpretable —
# it could equally be our tuning missing a real super weight. Distinguishing
# those two needs Phase 0's calibrated null.
#
# Chosen to answer specific questions rather than to survey:
#   Llama-3.1-8B  — the lineage. Llama-1-7B ablates x181, Llama-2-7B only
#                   x1.55; does the effect come back two generations later?
#   Qwen3-8B      — a 2025 model from a family the paper never touched.
#   TowerBase-7B  — a Llama-2-7B fine-tune. This repo's q6 sweep reports a
#                   super weight at L1[2533,7890] with ablation KL 0.957 —
#                   the exact coordinate where our Llama-2-7B base measures
#                   only x1.55. If the fine-tune is load-bearing where its
#                   base is not, that is a finding about Llama-2, not noise.
#                   (q6 numbers are n=24–32 leads, different protocol, and
#                   are not comparable as measured — see
#                   docs/prior_experiments_and_ideas.md §2.)
MODERN = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "Unbabel/TowerBase-7B-v0.1",
]


# ---- experiments/joint_ablation extensions (2026-09-02) ----------------
# Base counterparts of the two post-trained MODERN models. Yu et al. 3.1
# say instruct tuning does not move the coordinates (Mistral-Instruct,
# Llama-2-chat); Subramanian's Llama-3-8B (base) vs Llama-3.1-8B-Instruct
# contrast is confounded by method (paper coordinates vs top-magnitude), so
# this is the clean base/instruct pair. Meta-Llama-3-8B itself is gated for
# this account (403 on 2026-09-02).
BASES = [
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen3-8B-Base",
]

# RQ2, the <=1B half (OLMo-1B is already in MODELS). No answer key.
SMALL = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "bigscience/bloom-560m",
    "Qwen/Qwen3-1.7B",
]

# RQ3 gate: the multilingual models the q6 sweep reported on
# (docs/prior_experiments_and_ideas.md section 2), re-detected here. Only
# EuroLLM-9B-Instruct is cached (not the base). Gemma-3-12b-it loads as
# Gemma3ForConditionalGeneration; sw_arch handles language_model.layers.
MULTI = [
    "utter-project/EuroLLM-9B-Instruct",
    "Unbabel/TowerInstruct-7B-v0.2",
    "CohereForAI/aya-expanse-8b",
    "bigscience/bloom-7b1",
    "google/gemma-3-12b-it",
]
