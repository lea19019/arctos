"""Information Flow Routes (IFR).

Decomposes the residual stream into per-token, per-layer contributions from
attention heads and MLPs; the L1-normalized contribution magnitudes give a
"flow score" per component, averaged over a calibration set. Used as a
layer / head / MLP importance ranking.

Cite:
- Ferrando & Voita (2024), "Information Flow Routes: Automatically
  Interpreting Language Models at Scale", https://arxiv.org/abs/2403.00824.

Familiar ground from the prior paper. **Implemented from scratch in this
repo** — no port of the prior paper's code (per `docs/project-summary.md`).
The from-scratch implementation forces alignment with the canonical paper's
math and lets us answer Q1/Q3 cleanly without inheriting prior-code
assumptions.

This implementation is the magnitude-based variant: per-component output
contribution norms at the target position(s), L1-normalized per token, then
averaged. It captures per-(layer, head) and per-(layer, MLP) importance
for the final prediction at the chosen positions; it does NOT do full
attention-rollout attribution back to source positions (a future extension
if Q1/Q3 needs it).

Architecture notes:
- All three target models are decoder-only Llama-class. Per-head decomposition
  uses TransformerLens's `blocks.{ℓ}.attn.hook_z` (per-head pre-W_O output);
  the per-head contribution is `z[h] @ W_O[ℓ, h]`.
- The prior paper's IFR result on Aya for cs→de showed somewhat different
  layer rankings than en→es. Document this when comparing pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass
class IFRScores:
    """Per-component IFR scores aggregated over a calibration set.

    Attributes:
        layer_scores: per-layer importance (attn + mlp summed), shape (L,).
        head_scores: per-head importance, shape (L, H).
        mlp_scores: per-MLP importance, shape (L,).
        embed_score: input-embedding importance (scalar, the "layer -1" mass).
        n_examples: number of calibration examples averaged.
    """

    layer_scores: torch.Tensor
    head_scores: torch.Tensor
    mlp_scores: torch.Tensor
    embed_score: float
    n_examples: int


def _resolve_positions(target_position: str | int, n_tokens: int) -> Sequence[int]:
    if isinstance(target_position, int):
        return [target_position if target_position >= 0 else n_tokens + target_position]
    if target_position == "last":
        return [n_tokens - 1]
    raise ValueError(
        f"Unsupported target_position={target_position!r}; pass 'last' or an int. "
        "'all_target' requires the runner to slice the target span and pass int positions."
    )


def ifr(
    model: Any,
    examples: Iterable[str],
    *,
    target_position: str | int = "last",
) -> IFRScores:
    """Compute IFR scores over a calibration set.

    Args:
        model: a `HookedModel`.
        examples: iterable of prompts (source-target concatenated, MT setting).
        target_position: which output position to attribute.
            - "last": final token only.
            - int: specific position (negative indexes from end).

    Returns:
        IFRScores with averaged per-component importance.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_acc = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    mlp_acc = torch.zeros(n_layers, dtype=torch.float64)
    embed_acc = 0.0
    n_examples = 0

    W_O = model.W_O  # (L, H, d_head, D), on model.device

    for prompt in examples:
        tokens = model.to_tokens(prompt)
        _, cached = model.run_with_cache(tokens, capture=("attn_z", "mlp_out", "embed"))
        positions = _resolve_positions(target_position, tokens.shape[-1])

        head_ex = torch.zeros(n_layers, n_heads, dtype=torch.float64)
        mlp_ex = torch.zeros(n_layers, dtype=torch.float64)
        embed_ex = 0.0

        for pos in positions:
            embed_ex += float(cached.embed[pos].detach().to(torch.float32).abs().sum().cpu())
            for ell in range(n_layers):
                z = cached.attn_z[ell, pos]  # (H, d_head)
                head_contrib = torch.einsum("hd,hdm->hm", z, W_O[ell])
                head_ex[ell] += head_contrib.detach().to(torch.float32).abs().sum(dim=-1).cpu().double()
                mlp_ex[ell] += float(cached.mlp_out[ell, pos].detach().to(torch.float32).abs().sum().cpu())

        total = head_ex.sum().item() + mlp_ex.sum().item() + embed_ex
        if total <= 0:
            continue
        head_acc += head_ex / total
        mlp_acc += mlp_ex / total
        embed_acc += embed_ex / total
        n_examples += 1

    if n_examples == 0:
        raise ValueError("ifr() got no examples (or all examples had zero contribution).")

    head_scores = (head_acc / n_examples).float()
    mlp_scores = (mlp_acc / n_examples).float()
    embed_score = float(embed_acc / n_examples)
    layer_scores = head_scores.sum(dim=-1) + mlp_scores

    return IFRScores(
        layer_scores=layer_scores,
        head_scores=head_scores,
        mlp_scores=mlp_scores,
        embed_score=embed_score,
        n_examples=n_examples,
    )
