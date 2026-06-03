"""Super-weight / super-activation detection (the "find", extreme case).

Yu, Bai, Jaiswal et al. (2024), "The Super Weight in Large Language Models",
https://arxiv.org/abs/2411.07191 (Apple). A handful — sometimes *one* — of
scalar weights, always in an early-layer `mlp.down_proj`, are so important
that zeroing a single one raises perplexity by orders of magnitude. They are
detectable data-free in a single forward pass because they create rare,
enormous activation spikes ("super activations"), which are the same
phenomenon Sun et al. (2024), "Massive Activations in Large Language Models"
(https://arxiv.org/abs/2402.17762) describe.

Why this matters for Arctos: phase-one's Q5 found component *importance*
(IFR/DLA, per-head/per-layer) does not predict quantization *sensitivity*.
Super weights are the per-weight counter-story — sensitivity is real and
extremely concentrated, just at a granularity two levels below a "component".
This module locates that concentration so the quantizer can *keep* it.

Detection (single forward pass):
  1. Hook every block's MLP output projection (down_proj / dense_4h_to_h /
     c_proj). Its input is the intermediate activation h; its output is the
     contribution written back to the residual stream.
  2. The super activation is the largest |down_proj output| over (token,
     out-dim), across layers. The candidate super weight is W_down[o*, j*]
     where o* is that output dim and j* = argmax_j |h[token*, j]| — the
     input channel feeding the spike.
  3. Optionally *verify* by zeroing that single scalar and re-measuring the
     spike and the next-token distribution (KL / top-1 prob drop).

We also report residual-stream massive activations directly (Sun et al.):
the (layer, dim) of the largest |resid_post| values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn


def _mlp_out_linear(model: Any, block: nn.Module) -> nn.Linear:
    """The output (down) projection of a block's MLP, per architecture."""
    mlp = model.arch.get_block_mlp(block)
    for attr in ("down_proj", "dense_4h_to_h", "c_proj"):
        if hasattr(mlp, attr):
            return getattr(mlp, attr)
    cands = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
    if not cands:
        raise ValueError("could not find MLP output linear")
    return cands[-1]


@dataclass
class SuperWeightCandidate:
    layer: int
    out_dim: int          # residual/output feature index (row of W_down)
    in_dim: int           # intermediate channel index (col of W_down)
    weight_value: float   # W_down[out_dim, in_dim]
    activation: float     # |down_proj output| spike that flagged it
    input_channel_act: float  # |h[token, in_dim]| feeding the spike


@dataclass
class SuperWeightResult:
    candidates: list[SuperWeightCandidate]
    # residual-stream massive activations: (layer, dim, value) sorted by |value|
    massive_resid: list[tuple[int, int, float]] = field(default_factory=list)
    n_prompts: int = 0


def detect_super_weights(
    model: Any,
    prompts: Sequence[str],
    *,
    top_k: int = 5,
    massive_top_k: int = 10,
) -> SuperWeightResult:
    """Find candidate super weights + residual-stream massive activations.

    Args:
        model: a HookedModel.
        prompts: a few prompts; super weights are input-agnostic so 1-8 suffice.
        top_k: how many super-weight candidates to return (ranked by spike).
        massive_top_k: how many residual massive activations to report.
    """
    blocks = model.arch.get_blocks(model.hf_model)
    out_lins = [_mlp_out_linear(model, b) for b in blocks]

    # Per-layer running best spike: (abs_out_value, token, out_dim, in_dim, in_act)
    best: dict[int, tuple[float, int, int, int, float]] = {}
    captured: dict[int, dict[str, torch.Tensor]] = {}

    handles = []

    def make_hook(li: int):
        def hook(_m, inp, out):
            x = inp[0].detach()                      # (B, T, d_intermediate)
            y = out.detach() if not isinstance(out, tuple) else out[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()   # (N, d_int)
            y = y.reshape(-1, y.shape[-1]).float()   # (N, d_model)
            # spike location in the output
            ay = y.abs()
            flat = ay.argmax()
            tok = int(flat // ay.shape[1])
            out_dim = int(flat % ay.shape[1])
            spike = float(ay[tok, out_dim])
            # input channel most responsible at that token
            in_dim = int(x[tok].abs().argmax())
            in_act = float(x[tok, in_dim].abs())
            prev = best.get(li)
            if prev is None or spike > prev[0]:
                best[li] = (spike, tok, out_dim, in_dim, in_act)
        return hook

    for li, lin in enumerate(out_lins):
        handles.append(lin.register_forward_hook(make_hook(li)))

    n = 0
    massive_resid_acc: list[tuple[int, int, float]] = []
    try:
        for prompt in prompts:
            tokens = model.to_tokens(prompt)
            _, cached = model.run_with_cache(tokens, capture=("resid_post",))
            resid = cached.resid_post.float()        # (L, T, D)
            a = resid.abs()
            # top massive activations across (layer, dim), max over tokens
            per_ld = a.amax(dim=1)                    # (L, D)
            vals, idx = per_ld.flatten().topk(min(massive_top_k, per_ld.numel()))
            for v, i in zip(vals.tolist(), idx.tolist()):
                massive_resid_acc.append((int(i // per_ld.shape[1]), int(i % per_ld.shape[1]), float(v)))
            n += 1
    finally:
        for h in handles:
            h.remove()

    # Build candidates ranked by spike magnitude.
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    candidates = []
    for li, (spike, _tok, out_dim, in_dim, in_act) in ranked:
        W = out_lins[li].weight.detach()             # (d_model, d_int)
        wv = float(W[out_dim, in_dim])
        candidates.append(SuperWeightCandidate(
            layer=li, out_dim=out_dim, in_dim=in_dim,
            weight_value=wv, activation=spike, input_channel_act=in_act,
        ))

    massive_resid = sorted(set(massive_resid_acc), key=lambda t: t[2], reverse=True)[:massive_top_k]
    return SuperWeightResult(candidates=candidates, massive_resid=massive_resid, n_prompts=n)


@torch.no_grad()
def verify_super_weight(
    model: Any,
    prompts: Sequence[str],
    candidate: SuperWeightCandidate,
) -> dict[str, float]:
    """Zero a single scalar weight and measure the damage on next-token preds.

    Returns mean top-1 probability drop and mean KL(clean || ablated) over the
    prompts — the causal confirmation that this one weight is load-bearing.
    """
    blocks = model.arch.get_blocks(model.hf_model)
    lin = _mlp_out_linear(model, blocks[candidate.layer])
    W = lin.weight
    orig = float(W[candidate.out_dim, candidate.in_dim])

    def _last_probs(prompt: str) -> torch.Tensor:
        toks = model.to_tokens(prompt)
        logits = model(toks)[0, -1].float()
        return logits.softmax(-1)

    clean = [_last_probs(p) for p in prompts]
    W[candidate.out_dim, candidate.in_dim] = 0.0
    try:
        ablated = [_last_probs(p) for p in prompts]
    finally:
        W[candidate.out_dim, candidate.in_dim] = orig

    top1_drops, kls = [], []
    for c, a in zip(clean, ablated):
        top1_drops.append(float(c.max() - a[c.argmax()]))
        kls.append(float((c * (c.clamp_min(1e-12).log() - a.clamp_min(1e-12).log())).sum()))
    return {
        "mean_top1_prob_drop": float(sum(top1_drops) / len(top1_drops)),
        "mean_kl_clean_vs_ablated": float(sum(kls) / len(kls)),
        "weight_value": orig,
    }
