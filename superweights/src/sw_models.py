"""The models the super-weight sweep runs on (shared by prefetch_models.py
and run_all.py). Comment out what you can't access or fit.

All of these appear in Yu et al. 2024 Table 2 (see TABLE2 in ablate_sw.py),
so every run doubles as a check of the paper's directory.
"""

MODELS = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
    "mistralai/Mistral-7B-v0.1",
    "huggyllama/llama-7b",          # the original LLaMA-1 7B
    "meta-llama/Llama-2-7b-hf",     # gated: needs `hf auth login` first
    # "huggyllama/llama-13b",       # ~26 GB in bf16 — enable on a big GPU
    # "meta-llama/Llama-2-13b-hf",  # gated + large
]
