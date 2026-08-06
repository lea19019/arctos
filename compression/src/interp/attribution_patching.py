"""Attribution patching (Nanda 2023).

Gradient-based approximation of activation patching. For a metric L
(e.g., gold-target-token logit) computed from the *clean* forward, and
an activation `act_c` at component c:

  patch_effect[c] ≈ (∂L/∂act_c) · (corrupt_act_c - clean_act_c)

One forward + one backward on the clean run gives the gradient at every
hooked activation. We then compute the inner product with the (corrupt −
clean) activation difference per component to get the per-component
attribution. Cost: O(1) backward passes instead of O(n_components)
forward passes — ~100× faster than full activation patching at the head
granularity.

Reference:
- Nanda, "Attribution Patching: Activation Patching At Industrial Scale",
  https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
- Syed, Rager, Conmy (2024), "Attribution Patching Outperforms
  Automated Circuit Discovery", https://arxiv.org/abs/2310.10348

For our setup, the hooked sites are:
- per-(layer, head) attention z (pre-W_O): contribution per head
- per-layer mlp_out: contribution per MLP block

Clean = normal MT prompt; corrupt = same prompt with source replaced
(per `src/data/clean_corrupt.py` — LEXICAL_SUB or LANG_ID_SWAP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn


@dataclass
class AttributionResult:
    """Per-component attribution effect on the metric, averaged over examples."""

    head_effects: torch.Tensor  # (L, H) — per-head attribution
    mlp_effects: torch.Tensor   # (L,)   — per-MLP attribution
    n_examples: int


def _capture_acts_with_grad(
    model: Any,
    tokens: torch.Tensor,
    target_token_ids: Sequence[int],
    *,
    keep_grad: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    """Forward pass capturing per-layer z (pre-W_O) and mlp_out tensors.

    If keep_grad: returned tensors require grad and we set up the model
    forward to retain grad on them. Metric (mean target-token logit at
    last position) is returned as a scalar tensor with grad attached.

    Returns: (z_by_layer, mlp_by_layer, metric)
    """
    blocks = model.arch.get_blocks(model.hf_model)
    z_by_layer: dict[int, torch.Tensor] = {}
    mlp_by_layer: dict[int, torch.Tensor] = {}
    handles = []

    def make_attn_pre_hook(i: int):
        def hook(_module, inputs):
            x = inputs[0]
            # Reshape to (B, T, H, d_head) and wrap as leaf for gradient
            B, T, _ = x.shape
            z = x.reshape(B, T, model.cfg.n_heads, model.cfg.d_head)
            if keep_grad:
                z = z.clone()
                z.requires_grad_(True)
                z.retain_grad()
            z_by_layer[i] = z
            # Pass the (possibly cloned/grad-leaf) tensor through the projection by
            # collapsing back to (B, T, H*d_head). If keep_grad we use the leaf z.
            return (z.reshape(B, T, -1),) + tuple(inputs[1:])
        return hook

    def make_mlp_hook(i: int):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if keep_grad:
                out = out.clone()
                out.requires_grad_(True)
                out.retain_grad()
            mlp_by_layer[i] = out
            if isinstance(output, tuple):
                return (out,) + tuple(output[1:])
            return out
        return hook

    for i, block in enumerate(blocks):
        proj = model.arch.get_block_attn_proj(block)
        handles.append(proj.register_forward_pre_hook(make_attn_pre_hook(i)))
        mlp = model.arch.get_block_mlp(block)
        handles.append(mlp.register_forward_hook(make_mlp_hook(i)))

    try:
        if keep_grad:
            out = model.hf_model(input_ids=tokens)
        else:
            with torch.no_grad():
                out = model.hf_model(input_ids=tokens)
        logits = out.logits if hasattr(out, "logits") else out[0]
        last = logits[0, -1]
        if target_token_ids:
            idx = torch.as_tensor(list(target_token_ids), device=last.device, dtype=torch.long)
            metric = last.float().index_select(0, idx).mean()
        else:
            metric = last.sum() * 0.0  # degenerate, no-op
    finally:
        for h in handles:
            h.remove()
    return z_by_layer, mlp_by_layer, metric


def attribution_patch(
    model: Any,
    clean_corrupt_pairs: list[tuple[str, str, Sequence[int]]],
) -> AttributionResult:
    """Attribution patching across a list of (clean_prompt, corrupt_prompt, target_ids).

    Returns averaged per-(layer, head) and per-(layer, mlp) attribution.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_acc = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    mlp_acc = torch.zeros(n_layers, dtype=torch.float64)
    n_examples = 0

    for clean_p, corrupt_p, target_ids in clean_corrupt_pairs:
        clean_tokens = model.to_tokens(clean_p)
        corrupt_tokens = model.to_tokens(corrupt_p)
        if clean_tokens.shape[-1] != corrupt_tokens.shape[-1]:
            # Position alignment matters for attribution patching; skip mismatches.
            # The clean/corrupt generators in src/data/clean_corrupt.py should
            # produce same-length pairs at the source-text level, but tokenization
            # can still drift. For phase one we skip rather than error.
            continue

        # Corrupt pass: no grad, save activations for the difference.
        z_corrupt, mlp_corrupt, _ = _capture_acts_with_grad(
            model, corrupt_tokens, target_ids, keep_grad=False
        )

        # Clean pass with grad.
        for p in model.parameters():
            p.requires_grad_(False)  # we don't need param gradients
        z_clean, mlp_clean, metric = _capture_acts_with_grad(
            model, clean_tokens, target_ids, keep_grad=True
        )
        metric.backward()

        # Per-component attribution at the last position.
        last = clean_tokens.shape[-1] - 1
        for ell in range(n_layers):
            z_c = z_clean[ell][0, last]    # (H, d_head)
            z_g = z_clean[ell].grad        # (B, T, H, d_head)
            z_x = z_corrupt[ell][0, last]  # (H, d_head)
            if z_g is None:
                continue
            # grad · (corrupt - clean) summed over d_head per head
            per_head = (z_g[0, last] * (z_x - z_c)).sum(dim=-1)  # (H,)
            head_acc[ell] += per_head.detach().float().cpu().double()

            m_c = mlp_clean[ell][0, last]
            m_g = mlp_clean[ell].grad
            m_x = mlp_corrupt[ell][0, last]
            if m_g is None:
                continue
            per_mlp = (m_g[0, last] * (m_x - m_c)).sum()
            mlp_acc[ell] += float(per_mlp.detach().float().cpu())
        n_examples += 1

    if n_examples == 0:
        raise ValueError("attribution_patch got no usable clean/corrupt pairs (all length-mismatched?)")

    return AttributionResult(
        head_effects=(head_acc / n_examples).float(),
        mlp_effects=(mlp_acc / n_examples).float(),
        n_examples=n_examples,
    )
