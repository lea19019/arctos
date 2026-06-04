"""AWQ (Activation-aware Weight Quantization) via AutoAWQ — paper's "AWQ".

The paper uses AutoAWQ with zero-points and group size 128, 4-bit only.
AutoAWQ quantizes once to disk; we then reload the saved model through the
standard Transformers AWQ inference path so it shares ``hf_generate``.

Calibration: the paper doesn't specify AWQ's calibration set (AutoAWQ's
default is an online pileval sample, which is unavailable on offline compute
nodes). We therefore pass an explicit ``calib_texts`` list (generic WikiText,
matching the GGUF imatrix corpus) so the run is offline-safe and the
calibration source is documented rather than hidden.

Import ``_compat`` before ``awq`` (AutoAWQ 0.2.9 vs transformers 4.57 shim).
"""

from __future__ import annotations

import os
from typing import Sequence

from . import _compat

SUPPORTED_BITS = (4,)

_QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def quantize_to_disk(
    base_dir: str,
    bits: int,
    artifact_dir: str,
    calib_texts: Sequence[str],
) -> str:
    """Quantize ``base_dir`` with AWQ-4bit and save to ``artifact_dir``."""
    if bits != 4:
        raise ValueError(f"AWQ supports only 4-bit, got {bits}-bit.")
    _compat.apply_all()
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    os.makedirs(artifact_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    # device_map="auto" spreads the model across all GPUs in the job — required
    # for 32B/70B (single-GPU "cuda" OOMs loading ~64-140 GB of weights). For
    # 1-GPU jobs this is equivalent to loading on cuda:0.
    model = AutoAWQForCausalLM.from_pretrained(base_dir, device_map="auto")
    model.quantize(
        tokenizer,
        quant_config=_QUANT_CONFIG,
        calib_data=list(calib_texts),
    )
    model.save_quantized(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)
    return artifact_dir


def load_model(artifact_dir: str, bits: int):
    """Load an AWQ-quantized model for inference. Returns (model, tokenizer)."""
    import torch

    _compat.apply_all()  # transformers imports awq kernels when loading AWQ weights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer
