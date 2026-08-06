"""Tuned lens (Belrose et al. 2023) — a trained, de-biased logit lens.

The vanilla logit lens (notebook 01) applies the final norm + unembed to a
mid-layer residual directly. It's noisy in early/mid layers because those
residuals aren't yet in the "output basis" the unembed expects. The tuned
lens fixes this by learning, per layer ℓ, an affine map A_ℓ (a translator)
applied before the unembed:

    tuned_logits_ℓ = Unembed( LN_final( A_ℓ · resid_ℓ + b_ℓ ) )

A_ℓ is trained so that tuned_logits_ℓ matches the model's *final* output
distribution (minimize KL(model_final || tuned_ℓ)). Intuitively: "given the
residual at layer ℓ, what will the model end up predicting?" — a smoother,
less noisy read of the trajectory than the raw lens.

Cite: Belrose et al. (2023), "Eliciting Latent Predictions from Transformers
with the Tuned Lens", https://arxiv.org/abs/2303.08112.

Cost note: A_ℓ is d_model × d_model per layer. For d_model=4096 × 32 layers
that's ~500M params — trainable but not free. We train on CPU-cached
residuals against cached final logits, a few hundred steps. For phase one
this is a *diagnostic refinement*, not on the critical path: the
commitment-layer finding already holds under the vanilla lens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass
class TunedLensProbes:
    """Trained per-layer affine maps. A[ℓ]: (d, d), b[ℓ]: (d,)."""

    A: torch.Tensor  # (L, d, d)
    b: torch.Tensor  # (L, d)


def train_tuned_lens(
    model: Any,
    texts: Iterable[str],
    *,
    steps: int = 300,
    lr: float = 1e-3,
    max_positions_per_text: int = 16,
    device: str = "cpu",
) -> TunedLensProbes:
    """Fit per-layer affine translators against the model's final logits.

    Caches (resid_post[ℓ], final_logits) over a calibration set, then trains
    A_ℓ initialized to identity to minimize KL(final || tuned_ℓ).
    """
    n_layers, d = model.cfg.n_layers, model.cfg.d_model
    resid_by_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    final_logits_list: list[torch.Tensor] = []

    for text in texts:
        tokens = model.to_tokens(text)
        logits, cached = model.run_with_cache(tokens, capture=("resid_post",))
        T = tokens.shape[-1]
        positions = list(range(max(0, T - max_positions_per_text), T))
        with torch.no_grad():
            fl = logits[0, positions].float().cpu()  # (P, V)
        final_logits_list.append(fl)
        for ell in range(n_layers):
            resid_by_layer[ell].append(cached.resid_post[ell, positions].float().cpu())

    final_logits = torch.cat(final_logits_list, dim=0).to(device)  # (N, V)
    target = final_logits.log_softmax(dim=-1)
    resid = [torch.cat(r, dim=0).to(device) for r in resid_by_layer]  # list of (N, d)

    W_U = model.W_U.detach().float().to(device)  # (d, V)
    b_U = model.b_U.detach().float().to(device) if model.b_U is not None else None

    A = torch.eye(d, device=device).unsqueeze(0).repeat(n_layers, 1, 1).clone().requires_grad_(True)
    b = torch.zeros(n_layers, d, device=device, requires_grad=True)
    opt = torch.optim.Adam([A, b], lr=lr)

    ln_final = model.ln_final
    for step in range(steps):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for ell in range(n_layers):
            x = resid[ell] @ A[ell].T + b[ell]            # affine translate
            # apply the model's final norm (on the model's device, then back)
            normed = ln_final(x.to(model.dtype).to(model.device)).float().to(device)
            logits = normed @ W_U + (b_U if b_U is not None else 0.0)
            loss = loss + F.kl_div(logits.log_softmax(-1), target, log_target=True,
                                   reduction="batchmean")
        loss.backward()
        opt.step()
    return TunedLensProbes(A=A.detach(), b=b.detach())
