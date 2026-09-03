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
