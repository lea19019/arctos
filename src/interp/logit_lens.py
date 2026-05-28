"""Logit lens / tuned lens.

Layer-by-layer "what would the model predict if we decoded from this hidden
state?" — apply the final RMSNorm + lm_head to the residual stream at each
layer. Cheap diagnostic for where the model commits to a target-language
token.

Cite:
- Logit lens: nostalgebraist (2020), "interpreting GPT: the logit lens"
  (LessWrong; no canonical paper).
- Tuned lens: Belrose et al. (2023), "Eliciting Latent Predictions from
  Transformers with the Tuned Lens", https://arxiv.org/abs/2303.08112.

Architecture notes:
- All three target models (Aya, omt-llama, Tower) are decoder-only with
  RMSNorm + lm_head. The "decode-from-layer-ℓ" operation is unambiguous:
  RMSNorm(resid_post[ℓ]) @ W_U.
- Aya's Cohere attention scaling does not affect the lens itself, but the
  residual stream's magnitude profile may differ from a Llama-2 model — be
  alert when comparing logit-lens outputs across the three.
- Tuned lens requires training a small affine probe per layer on a held-out
  set; for phase one, start with vanilla logit lens and only add the tuned
  variant if vanilla noise dominates the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass
class LogitLensResult:
    """Result of applying the logit lens to one prompt across all layers.

    Attributes:
        prompt_tokens: input token ids, shape (T,).
        layer_logits: per-layer logits over the vocabulary at the chosen
            position, shape (L, V). L = number of layers.
        target_token_ids: optional, the target-language token ids being
            tracked (e.g., the gold target prefix). Shape (K,).
        target_token_mass: per-layer probability mass on `target_token_ids`,
            shape (L, K). Only present if `target_token_ids` was provided.
    """

    prompt_tokens: torch.Tensor
    layer_logits: torch.Tensor
    target_token_ids: torch.Tensor | None
    target_token_mass: torch.Tensor | None


def _decode_resid(model: Any, resid: torch.Tensor) -> torch.Tensor:
    """Apply final LN + unembed to one residual vector / batch.

    `resid` shape: (..., D). Returns (..., V). Uses the model's own ln_final
    and W_U/b_U so the last-layer lens output equals the model's logits.
    """
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if getattr(model, "b_U", None) is not None:
        logits = logits + model.b_U
    return logits


def logit_lens(
    model: Any,
    prompt: str,
    *,
    target_tokens: Sequence[int] | None = None,
    position: int | None = None,
) -> LogitLensResult:
    """Apply logit lens across all layers for one prompt.

    Args:
        model: a HookedTransformer.
        prompt: input text.
        target_tokens: token ids to track per-layer probability mass for.
            Typical use: gold target-language token prefix.
        position: position to read from. Defaults to last token (-1).

    Returns:
        LogitLensResult with `layer_logits` shape (L, V).
    """
    device = next(model.parameters()).device
    tokens = model.to_tokens(prompt)  # (1, T)
    pos = position if position is not None else tokens.shape[-1] - 1

    names_filter = lambda name: name.endswith("hook_resid_post")  # noqa: E731
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)

    n_layers = model.cfg.n_layers
    # Stack resid_post across layers at the chosen position: (L, D).
    resid_stack = torch.stack(
        [cache[f"blocks.{ell}.hook_resid_post"][0, pos] for ell in range(n_layers)],
        dim=0,
    )
    layer_logits = _decode_resid(model, resid_stack)  # (L, V)

    target_ids_tensor: torch.Tensor | None = None
    target_mass: torch.Tensor | None = None
    if target_tokens is not None and len(target_tokens) > 0:
        target_ids_tensor = torch.as_tensor(list(target_tokens), device=device, dtype=torch.long)
        probs = layer_logits.softmax(dim=-1)  # (L, V)
        target_mass = probs.index_select(dim=-1, index=target_ids_tensor)  # (L, K)

    return LogitLensResult(
        prompt_tokens=tokens[0].cpu(),
        layer_logits=layer_logits.cpu(),
        target_token_ids=target_ids_tensor.cpu() if target_ids_tensor is not None else None,
        target_token_mass=target_mass.cpu() if target_mass is not None else None,
    )


def tuned_lens(
    model: Any,
    prompt: str,
    *,
    probes: Any,  # trained tuned-lens probes; type stub
    target_tokens: Sequence[int] | None = None,
    position: int | None = None,
) -> LogitLensResult:
    """Tuned-lens variant.

    Differs from `logit_lens` only in that each layer's residual is first
    passed through a trained affine probe before RMSNorm + unembed. Probes
    are trained per-model on held-out text; see `scripts/train_tuned_lens.py`
    when implemented.

    TODO: implement after vanilla logit lens — only if vanilla noise dominates.
    """
    raise NotImplementedError
