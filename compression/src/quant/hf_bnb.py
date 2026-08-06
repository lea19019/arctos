"""bitsandbytes NF4 (4-bit) quantization — paper's "BnB" method.

The paper uses the bitsandbytes/Transformers integration with NormalFloat4
and *nested* (double) quantization enabled. bitsandbytes is calibration-free
and quantizes at load time, so there is no on-disk quantized artifact: we just
load the base model under a BitsAndBytesConfig. 4-bit only (bnb has no 2-bit).
"""

from __future__ import annotations

import torch

SUPPORTED_BITS = (4,)


def load_model(base_dir: str, bits: int):
    """Load ``base_dir`` as a bnb-NF4 4-bit model. Returns (model, tokenizer)."""
    if bits != 4:
        raise ValueError(f"bitsandbytes supports only 4-bit, got {bits}-bit.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # nested quantization (paper)
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        quantization_config=quant_config,
        device_map="auto",  # spread across job GPUs (large models); =cuda:0 on 1 GPU
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer
