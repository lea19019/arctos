"""Quantization-noise sensitivity testing (Q5 in PHASE1-PLAN).

For each candidate component (a single attention head, an MLP block, or
a full layer), apply Gaussian weight noise of varying magnitude and
measure the drop in target-token logits over a calibration set.

This is the empirical bridge from interpretability findings (DLA, IFR,
attribution patching) to the quantization decision. If interpretability
says "head L30.H8 is critical for MT," noise-sensitivity testing
verifies it by showing that adding noise to L30.H8 drops translation
quality more than adding noise to a random head.

Two metric variants:
- `logit_drop`: mean drop in gold-target-token logit (cheap; one forward
  pass per example per noise level). Used for ranking components.
- `quality_drop` (optional, more expensive): actually generate
  translations and compute chrF++/COMET drop. Not implemented yet —
  hooks the same scaffold via a `generate_fn` parameter.

Noise model: `W_perturbed = W + sigma * ||W||_2 * eps`  where
eps ~ N(0, I/sqrt(numel(W))) so the perturbation has relative norm σ.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import torch
import torch.nn as nn


@dataclass
class NoiseSensitivityResult:
    """Sensitivity curve for one component.

    sensitivity[i] = mean baseline_target_logit - mean noisy_target_logit
                     across the calibration set at noise level sigmas[i].
    """

    component: str            # human label, e.g. "L30.H8" or "L17.mlp"
    sigmas: list[float]       # noise levels evaluated
    logit_drops: list[float]  # per-sigma mean target-token logit drop
    n_examples: int           # calibration set size


@contextmanager
def _perturb_weight(weight: torch.Tensor, mask_slice: slice | None,
                    sigma: float, generator: torch.Generator):
    """Temporarily add Gaussian noise to a weight tensor (in-place, then restore).

    `mask_slice`: if not None, only the rows in this slice (along dim 0) are
    perturbed. Used for per-head perturbation of o_proj where the per-head
    weights are concatenated along the input dimension.
    """
    if mask_slice is None:
        target = weight
    else:
        target = weight[mask_slice]
    orig = target.detach().clone()
    norm = float(orig.norm().item())
    if norm == 0:
        yield
        return
    noise = torch.empty_like(target).normal_(generator=generator)
    noise = noise * (sigma * norm / math.sqrt(noise.numel()))
    with torch.no_grad():
        target.copy_(orig + noise)
    try:
        yield
    finally:
        with torch.no_grad():
            target.copy_(orig)


def _slice_for_head(d_head: int, head: int, weight_layout: Literal["row", "col"]) -> slice:
    """For o_proj (input is concat-of-heads), the per-head input slice."""
    start, stop = head * d_head, (head + 1) * d_head
    if weight_layout == "row":
        # nn.Linear weight is (out, in) — head h occupies columns [h*d_head:(h+1)*d_head]
        # Use slicing on weight[:, slice] — but contextmanager wants slice along dim 0.
        # We'll handle this by transposing externally; here we return the slice as-is.
        return slice(start, stop)
    raise NotImplementedError


def _mean_target_logit(model, prompt: str, target_token_ids: Sequence[int]) -> float:
    """Average logit across target_token_ids at the last position of the prompt."""
    with torch.no_grad():
        tokens = model.to_tokens(prompt)
        logits = model(tokens)  # (B, T, V)
    last = logits[0, -1]
    if not target_token_ids:
        return 0.0
    idx = torch.as_tensor(list(target_token_ids), device=last.device, dtype=torch.long)
    return float(last.float().index_select(0, idx).mean().item())


def _measure_baseline(model, examples: list[tuple[str, Sequence[int]]]) -> list[float]:
    return [_mean_target_logit(model, p, t) for p, t in examples]


def _measure_with_noise(model, examples: list[tuple[str, Sequence[int]]],
                        weight: torch.Tensor, mask_slice: slice | None,
                        sigma: float, seed: int) -> list[float]:
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    with _perturb_weight(weight, mask_slice, sigma, generator):
        return [_mean_target_logit(model, p, t) for p, t in examples]


def head_sensitivity(
    model: Any,
    examples: list[tuple[str, Sequence[int]]],
    *,
    layer: int,
    head: int,
    sigmas: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.5),
    seed: int = 0,
) -> NoiseSensitivityResult:
    """Sensitivity of one attention head: perturb its output projection columns.

    For a Llama-class self_attn.o_proj with weight shape (d_model, n_heads*d_head),
    the input channels for head h are columns [h*d_head:(h+1)*d_head] of the
    weight. We perturb those columns only.
    """
    block = model.arch.get_blocks(model.hf_model)[layer]
    proj = model.arch.get_block_attn_proj(block)
    W = proj.weight  # nn.Linear: (d_model, n_heads*d_head)
    d_head = model.cfg.d_head
    n_heads = model.cfg.n_heads
    # For column-slice we transpose: operate on W.T (shape n_heads*d_head, d_model)
    # and pass slice [h*d_head:(h+1)*d_head] along dim 0.
    if W.shape != (model.cfg.d_model, n_heads * d_head):
        raise NotImplementedError(
            f"head_sensitivity assumes nn.Linear o_proj with shape (d_model, n_heads*d_head); "
            f"got {tuple(W.shape)}. (GPT-2 Conv1D layout not supported here yet.)"
        )
    W_T = W.t()  # view: (n_heads*d_head, d_model) — operations on W_T affect W
    head_slice = slice(head * d_head, (head + 1) * d_head)

    baseline = _measure_baseline(model, examples)
    drops = []
    for sigma in sigmas:
        noisy = _measure_with_noise(model, examples, W_T, head_slice, sigma, seed)
        drops.append(float(sum(b - n for b, n in zip(baseline, noisy)) / len(examples)))
    return NoiseSensitivityResult(
        component=f"L{layer}.H{head}",
        sigmas=list(sigmas),
        logit_drops=drops,
        n_examples=len(examples),
    )


def mlp_sensitivity(
    model: Any,
    examples: list[tuple[str, Sequence[int]]],
    *,
    layer: int,
    sigmas: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.5),
    seed: int = 0,
) -> NoiseSensitivityResult:
    """Sensitivity of one MLP block: perturb its output (down) projection."""
    block = model.arch.get_blocks(model.hf_model)[layer]
    mlp = model.arch.get_block_mlp(block)
    # Find the "output" linear of the MLP. For Llama-class it's `down_proj`.
    # For GPT-2 it's `c_proj`. Fall back to last nn.Linear in the MLP.
    if hasattr(mlp, "down_proj"):
        out_lin = mlp.down_proj
    elif hasattr(mlp, "c_proj"):
        out_lin = mlp.c_proj
    else:
        candidates = [c for c in mlp.modules() if isinstance(c, nn.Linear)]
        if not candidates:
            raise ValueError(f"layer {layer}: could not find output linear in mlp")
        out_lin = candidates[-1]
    W = out_lin.weight
    baseline = _measure_baseline(model, examples)
    drops = []
    for sigma in sigmas:
        noisy = _measure_with_noise(model, examples, W, None, sigma, seed)
        drops.append(float(sum(b - n for b, n in zip(baseline, noisy)) / len(examples)))
    return NoiseSensitivityResult(
        component=f"L{layer}.mlp",
        sigmas=list(sigmas),
        logit_drops=drops,
        n_examples=len(examples),
    )


def sensitivity_sweep(
    model: Any,
    examples: list[tuple[str, Sequence[int]]],
    *,
    heads: Sequence[tuple[int, int]] = (),
    mlps: Sequence[int] = (),
    sigmas: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.5),
    seed: int = 0,
) -> list[NoiseSensitivityResult]:
    """Run sensitivity for a list of (layer, head) tuples and/or MLP layers."""
    out: list[NoiseSensitivityResult] = []
    for layer, head in heads:
        out.append(head_sensitivity(
            model, examples, layer=layer, head=head, sigmas=sigmas, seed=seed
        ))
    for layer in mlps:
        out.append(mlp_sensitivity(
            model, examples, layer=layer, sigmas=sigmas, seed=seed
        ))
    return out
