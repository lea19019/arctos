"""Direct logit attribution (DLA).

For each example, decomposes the *target-token logit at the final position*
into per-(layer, head), per-(layer, MLP), and embedding contributions.
Complements IFR's magnitude-based view: DLA tells you which components
push *toward* the gold target token, not just which are active.

Math sketch:
  resid_post[L-1, p] = embed[p] + Σ_ℓ (attn_out[ℓ, p] + mlp_out[ℓ, p])
  logit[T] = (ln_final(resid_post[L-1, p]) @ W_U)[T]

We linearize ln_final at the operating-point residual: RMSNorm's
scaling factor at the final residual is `s = gamma / sqrt(mean(r²) + eps)`,
so for a small additive perturbation `Δr`, the change in logit[T] is
approximately `(Δr * s) @ W_U[:, T]`. We use this linearization to
decompose `logit[T]` into per-component contributions: for component c,

  dla_c = (component_c[p] * s) @ W_U[:, T]

where `s` is a per-feature scale vector (broadcasting elementwise).
This is the standard TransformerLens-style DLA decomposition.

Per-head: attn_out is decomposed via z_h @ W_O[ℓ, h].

Outputs are averaged across the calibration set. Gold target token =
first token of the target sentence (we sum across the prefix when
target_tokens has multiple ids, matching how the lens reports mass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass
class DLAScores:
    """Per-component DLA, averaged across a calibration set.

    All scores are in *logit units* of the target token. Positive = pushes
    toward the target; negative = pushes away.

    Attributes:
        layer_attn: per-layer sum of head contributions, shape (L,).
        layer_mlp:  per-layer MLP contribution, shape (L,).
        head_scores: per-(layer, head) contribution, shape (L, H).
        embed_score: scalar embedding contribution.
        n_examples: number of examples averaged.
    """

    layer_attn: torch.Tensor
    layer_mlp: torch.Tensor
    head_scores: torch.Tensor
    embed_score: float
    n_examples: int


def _rms_norm_scale(model: Any, resid_final_at_p: torch.Tensor) -> torch.Tensor:
    """Per-feature normalization-linearization scale at the final residual.

    For RMSNorm (Llama, Gemma): y = x * (gamma / sqrt(mean(x^2) + eps))
        s = gamma / sqrt(mean(x^2) + eps)

    For LayerNorm (GPT-2, BLOOM): y = (x - mean) * gamma / sqrt(var + eps) + beta
    The linearization for a small perturbation Δx is approximately
        Δy ≈ (Δx - mean(Δx)) * gamma / sqrt(var + eps)
    The centering term (Δx - mean(Δx)) is omitted in the magnitude-only
    DLA we compute here — it would require per-component recentering at
    sum time. For LayerNorm models the DLA is therefore *approximate*;
    the rank ordering is still informative but absolute numbers are
    biased relative to RMSNorm models. For BLOOM/GPT-2 use this as a
    diagnostic, not a quantitative comparison vs Llama-class models.
    """
    norm = model.arch.get_ln_final(model.hf_model)
    eps = getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", 1e-6)
    gamma = getattr(norm, "weight", None)
    x = resid_final_at_p.to(torch.float32)
    # Detect LayerNorm (has elementwise_affine + bias) vs RMSNorm.
    if isinstance(norm, torch.nn.LayerNorm):
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        s = (1.0 / torch.sqrt(var + eps))
    else:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        s = (1.0 / rms)
    if gamma is not None:
        s = s * gamma.to(torch.float32)
    return s.squeeze(0) if s.dim() > 1 and s.shape[0] == 1 else s


def dla(
    model: Any,
    examples: Iterable[tuple[str, Sequence[int]]],
    *,
    target_position: str | int = "last",
) -> DLAScores:
    """Compute DLA scores over a calibration set of (prompt, target_token_ids).

    For each example we sum the DLA contribution to each gold-prefix token
    individually, then average across prefix tokens — so the per-component
    score is the *mean direct contribution* to a single target-token logit.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.device

    head_acc = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    mlp_acc = torch.zeros(n_layers, dtype=torch.float64)
    embed_acc = 0.0
    n_examples = 0

    W_O = model.W_O  # (L, H, d_head, D)
    W_U = model.W_U  # (D, V)

    for prompt, target_ids in examples:
        if not target_ids:
            continue
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cached = model.run_with_cache(
                tokens, capture=("resid_post", "attn_z", "mlp_out", "embed")
            )
        # Position to attribute (default last token)
        T = tokens.shape[-1]
        if isinstance(target_position, int):
            pos = target_position if target_position >= 0 else T + target_position
        else:
            pos = T - 1

        resid_final = cached.resid_post[-1, pos]  # (D,)
        scale = _rms_norm_scale(model, resid_final)  # (D,)
        W_U_eff = (scale.to(W_U.dtype).unsqueeze(-1) * W_U.to(torch.float32))  # (D, V)
        # Target slice (V,T_K) — we average across target prefix tokens.
        tgt_ids = torch.as_tensor(list(target_ids), device=device, dtype=torch.long)
        w_tgt = W_U_eff.index_select(dim=-1, index=tgt_ids).mean(dim=-1)  # (D,)

        embed_acc += float(cached.embed[pos].detach().to(torch.float32).dot(w_tgt.detach()).cpu())

        # Per-head
        head_ex = torch.zeros(n_layers, n_heads, dtype=torch.float64)
        mlp_ex = torch.zeros(n_layers, dtype=torch.float64)
        w_tgt_d = w_tgt.detach()
        for ell in range(n_layers):
            z = cached.attn_z[ell, pos].detach()  # (H, d_head)
            head_contrib = torch.einsum("hd,hdm->hm", z, W_O[ell].detach())  # (H, D)
            head_dla = (head_contrib.to(torch.float32) @ w_tgt_d).cpu().double()  # (H,)
            head_ex[ell] = head_dla
            mlp_ex[ell] = float(
                cached.mlp_out[ell, pos].detach().to(torch.float32).dot(w_tgt_d).cpu()
            )

        head_acc += head_ex
        mlp_acc += mlp_ex
        n_examples += 1

    if n_examples == 0:
        raise ValueError("dla() got no examples.")
    head_scores = (head_acc / n_examples).float()
    layer_attn = head_scores.sum(dim=-1)
    layer_mlp = (mlp_acc / n_examples).float()
    return DLAScores(
        layer_attn=layer_attn,
        layer_mlp=layer_mlp,
        head_scores=head_scores,
        embed_score=embed_acc / n_examples,
        n_examples=n_examples,
    )
