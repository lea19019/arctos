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
        model: a HookedTransformer.
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

    # Pre-fetch per-layer W_O: shape (n_heads, d_head, d_model).
    W_O = torch.stack([model.blocks[ell].attn.W_O for ell in range(n_layers)], dim=0)
    # W_O on the model's device; we'll move per-example contributions to CPU.

    def names_filter(name: str) -> bool:
        return name.endswith("attn.hook_z") or name.endswith("hook_mlp_out") or name == "hook_embed"

    for prompt in examples:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)
        positions = _resolve_positions(target_position, tokens.shape[-1])

        # Per-example accumulators, summed over the requested positions before
        # L1-normalizing (so per-component contributions and the embed share a
        # single normalizing constant per example).
        head_ex = torch.zeros(n_layers, n_heads, dtype=torch.float64)
        mlp_ex = torch.zeros(n_layers, dtype=torch.float64)
        embed_ex = 0.0

        for pos in positions:
            embed = cache["hook_embed"][0, pos]  # (d_model,)
            embed_ex += float(embed.detach().to(torch.float32).abs().sum().cpu())
            for ell in range(n_layers):
                z = cache[f"blocks.{ell}.attn.hook_z"][0, pos]  # (n_heads, d_head)
                # per-head contribution: (n_heads, d_model)
                head_contrib = torch.einsum("hd,hdm->hm", z, W_O[ell])
                head_ex[ell] += head_contrib.detach().to(torch.float32).abs().sum(dim=-1).cpu().double()
                mlp = cache[f"blocks.{ell}.hook_mlp_out"][0, pos]  # (d_model,)
                mlp_ex[ell] += float(mlp.detach().to(torch.float32).abs().sum().cpu())

        # L1-normalize this example across all components, then accumulate.
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
