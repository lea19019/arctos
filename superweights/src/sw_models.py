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
