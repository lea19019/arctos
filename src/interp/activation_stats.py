"""Per-channel activation magnitude statistics (AWQ-style).

For every nn.Linear (and Conv1D for GPT-2) inside the model's transformer
blocks, capture per-input-channel magnitude statistics across the
calibration set:

  max_abs[c] = max over examples and positions of |x[..., c]|
  mean_abs[c] = mean over examples and positions of |x[..., c]|
  q99_abs[c] = 99th percentile over examples and positions of |x[..., c]|

This is the input signal AWQ (Lin et al. 2024) uses to identify "salient"
weight columns to protect at quantization time. For our quantization
method, this is the baseline we compare against: AWQ argues 1% of weight
columns (those paired with high-magnitude inputs) carry most of the
quantization sensitivity.

Cite:
- Lin et al. (2024), "AWQ: Activation-aware Weight Quantization for
  On-Device LLM Compression and Acceleration",
  https://arxiv.org/abs/2306.00978

Implementation: hooks every Linear (and Conv1D) module inside the
transformer blocks. Skips the embedding and lm_head — those have
their own quantization-sensitivity story.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn


@dataclass
class ActivationStats:
    """Per-channel input-activation magnitudes for one Linear module.

    Each tensor has shape (in_features,) — one statistic per input channel.
    """

    name: str                       # e.g. "blocks.5.self_attn.q_proj"
    in_features: int
    max_abs: torch.Tensor           # (in_features,)
    mean_abs: torch.Tensor          # (in_features,)
    q99_abs: torch.Tensor           # (in_features,)
    n_tokens: int                   # how many position-tokens contributed


@dataclass
class ActivationStatsResult:
    """All Linear modules' stats, keyed by dotted module name."""

    stats_by_module: dict[str, ActivationStats] = field(default_factory=dict)
    n_examples: int = 0


def _is_target_linear(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    # GPT-2 / Cohere etc. sometimes use Conv1D from transformers
    try:
        from transformers.pytorch_utils import Conv1D
        if isinstance(module, Conv1D):
            return True
    except ImportError:
        pass
    return False


def collect_activation_stats(
    model: Any,
    prompts: Iterable[str],
    *,
    quantile: float = 0.99,
    max_positions_per_example: int | None = None,
) -> ActivationStatsResult:
    """Run the model over `prompts` and collect per-channel input stats.

    Args:
        model: a HookedModel.
        prompts: iterable of text prompts (typically the MT calibration set).
        quantile: percentile to compute (default 0.99 for "high but robust").
        max_positions_per_example: cap positions per prompt to keep memory
            bounded on long prompts; default = all positions.
    """
    blocks = model.arch.get_blocks(model.hf_model)
    # Collect target submodules by name relative to the block root.
    targets: dict[str, nn.Module] = {}
    for li, block in enumerate(blocks):
        for sub_name, sub in block.named_modules():
            if _is_target_linear(sub) and sub_name:  # skip the block root itself
                targets[f"blocks.{li}.{sub_name}"] = sub

    # Accumulators per module — we keep running max, sum-abs (for mean),
    # and a sampled set of values for the quantile.
    state: dict[str, dict[str, torch.Tensor | int]] = {}
    sampled_per_mod: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
    QUANTILE_SAMPLE_CAP = 32_000  # cap reservoir per module per channel

    handles = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs, output):
            x = inputs[0]
            if x.dim() == 3:
                # (B, T, C)
                x_flat = x.reshape(-1, x.shape[-1])
            elif x.dim() == 2:
                x_flat = x
            else:
                return
            if max_positions_per_example is not None and x_flat.shape[0] > max_positions_per_example:
                x_flat = x_flat[:max_positions_per_example]
            x_abs = x_flat.detach().to(torch.float32).abs()  # (N, C)
            st = state.setdefault(name, {
                "max_abs": torch.zeros(x_abs.shape[-1]),
                "sum_abs": torch.zeros(x_abs.shape[-1], dtype=torch.float64),
                "n_tokens": 0,
            })
            st["max_abs"] = torch.maximum(st["max_abs"], x_abs.amax(dim=0).cpu())
            st["sum_abs"] += x_abs.sum(dim=0).cpu().double()
            st["n_tokens"] += int(x_abs.shape[0])
            # Reservoir: keep the running per-channel quantile by appending a
            # sample (cheap). We collapse later.
            if sum(t.shape[0] for t in sampled_per_mod[name]) < QUANTILE_SAMPLE_CAP:
                sampled_per_mod[name].append(x_abs.cpu())
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_hook(make_hook(name)))

    n_examples = 0
    try:
        with torch.no_grad():
            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                model.hf_model(input_ids=tokens)
                n_examples += 1
    finally:
        for h in handles:
            h.remove()

    result = ActivationStatsResult(n_examples=n_examples)
    for name, st in state.items():
        if st["n_tokens"] == 0:
            continue
        mean_abs = (st["sum_abs"] / st["n_tokens"]).float()
        if sampled_per_mod[name]:
            cat = torch.cat(sampled_per_mod[name], dim=0)
            q = torch.quantile(cat, quantile, dim=0)
        else:
            q = torch.zeros_like(st["max_abs"])
        result.stats_by_module[name] = ActivationStats(
            name=name,
            in_features=int(st["max_abs"].shape[0]),
            max_abs=st["max_abs"],
            mean_abs=mean_abs,
            q99_abs=q,
            n_tokens=int(st["n_tokens"]),
        )
    return result
