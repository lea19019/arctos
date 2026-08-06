"""Hessian / Fisher-diagonal weight sensitivity (the loss-curvature "find").

The sensitivity-native signal that the GPTQ / OBS / SqueezeLLM / LeanQuant
family is built on: a weight's importance for the *loss* is governed by the
curvature of the loss in that weight's direction, i.e. the Hessian. Computing
the full Hessian is intractable, so two standard diagonal approximations are
used:

  - Empirical Fisher diagonal:  F_ii = E[ (dL/dw_i)^2 ]
    (the expected squared gradient of the loss w.r.t. each weight). Under an
    NLL loss the Fisher is a positive-semidefinite stand-in for the Hessian,
    needs only first-order autograd, and is what we compute here. One backward
    pass per calibration example; accumulate g^2.

  - GPTQ-style layer-input second moment:  H ≈ 2 X Xᵀ for a linear layer with
    input X (the diagonal is E[x_j^2] per input channel). This is activation-
    only (no labels), per-channel, and is what `salient_channels.py` /
    `activation_stats.py` already approximate. We expose a per-channel reducer
    so the two views line up.

Cite: Frantar et al. (2023) GPTQ https://arxiv.org/abs/2210.17323; Kim et al.
(2024) SqueezeLLM https://arxiv.org/abs/2306.07629; LeCun et al. OBD; Hassibi
& Stork OBS.

The loss is the MT target-token NLL: we score next-token prediction on the
gold target prefix at the generation position, so the Fisher is *MT-conditional*
— directly the task-aware sensitivity signal phase-two wants, and the thing
Arctos's Q5 (Gaussian noise vs component importance) could not see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FisherResult:
    """Per-weight-matrix Fisher diagonal, summarized per module and per layer.

    We keep memory bounded: instead of storing the full per-weight g^2 tensor
    for every module (which is the model's size again), we store per-module
    *summaries* (mean, max, and per-output-channel mean) plus a per-layer
    aggregate. Full per-weight tensors are available transiently if needed.
    """

    per_module_mean: dict[str, float] = field(default_factory=dict)
    per_module_max: dict[str, float] = field(default_factory=dict)
    # per-layer aggregate Fisher mass (sum over the layer's linears), shape (L,)
    layer_fisher: list[float] = field(default_factory=list)
    n_examples: int = 0


def _target_nll(model: Any, tokens: torch.Tensor, target_ids: Sequence[int]) -> torch.Tensor:
    """NLL of the gold target prefix at the final prompt position.

    Differentiable: forward in train-grad mode (no torch.no_grad), take the
    last-position logits, and sum -log p over the target prefix tokens. (We use
    the first target id as the immediate next-token label; the full prefix is
    averaged to reduce variance.)
    """
    out = model.hf_model(input_ids=tokens)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logp = F.log_softmax(logits[0, -1].float(), dim=-1)
    idx = torch.as_tensor(list(target_ids), device=logp.device, dtype=torch.long)
    return -(logp.index_select(0, idx).mean())


def fisher_diagonal(
    model: Any,
    examples: Sequence[tuple[torch.Tensor, Sequence[int]]],
    *,
    per_channel: bool = True,
) -> FisherResult:
    """Empirical Fisher diagonal of MT target-NLL over the linear weights.

    Args:
        model: HookedModel (must allow grad — caller should not wrap in no_grad).
        examples: list of (tokens (1,T), target_prefix_ids) pairs.
        per_channel: if True also keep per-output-channel mean g^2 (unused in
            the summary but available for finer allocation later).

    Returns:
        FisherResult with per-module mean/max Fisher and a per-layer aggregate.
    """
    blocks = model.arch.get_blocks(model.hf_model)
    # name -> (module, layer index)
    targets: dict[str, tuple[nn.Linear, int]] = {}
    for li, block in enumerate(blocks):
        for sub_name, sub in block.named_modules():
            if isinstance(sub, nn.Linear) and sub_name:
                targets[f"blocks.{li}.{sub_name}"] = (sub, li)

    # Set grad ONLY on the target weights; everything else stays frozen so the
    # backward graph and grad tensors are bounded. We never materialise a full
    # per-weight accumulator (that would be ~2x model size) — instead we reduce
    # each grad to a per-output-channel vector (out_features floats) and to
    # scalars immediately, then free it.
    for p in model.parameters():
        p.requires_grad_(False)
    params = {name: mod.weight for name, (mod, _) in targets.items()}
    for p in params.values():
        p.requires_grad_(True)

    sum_g2: dict[str, float] = {name: 0.0 for name in params}
    numel: dict[str, int] = {name: w.numel() for name, w in params.items()}
    chan_g2: dict[str, torch.Tensor] = {
        name: torch.zeros(w.shape[0], dtype=torch.float32) for name, w in params.items()
    }

    model.hf_model.zero_grad(set_to_none=True)
    n = 0
    for tokens, target_ids in examples:
        if len(target_ids) == 0:
            continue
        loss = _target_nll(model, tokens, target_ids)
        grads = torch.autograd.grad(loss, list(params.values()), retain_graph=False, allow_unused=True)
        for (name, _w), g in zip(params.items(), grads):
            if g is None:
                continue
            g2 = g.detach().float() ** 2
            sum_g2[name] += float(g2.sum())
            chan_g2[name] += g2.sum(dim=1).cpu() if g2.dim() == 2 else g2.cpu()
            del g, g2
        n += 1

    n_layers = len(blocks)
    layer_fisher = [0.0] * n_layers
    res = FisherResult(n_examples=n)
    nn_ = max(n, 1)
    for name, (mod, li) in targets.items():
        res.per_module_mean[name] = sum_g2[name] / (nn_ * numel[name])
        # per-output-channel mean Fisher; "max" is the most fragile channel
        chan_mean = chan_g2[name] / (nn_ * (numel[name] / chan_g2[name].numel()))
        res.per_module_max[name] = float(chan_mean.max())
        layer_fisher[li] += sum_g2[name] / nn_
    res.layer_fisher = layer_fisher
    return res


@torch.no_grad()
def channel_second_moment(
    model: Any,
    prompts: Sequence[str],
) -> dict[str, torch.Tensor]:
    """GPTQ-style per-input-channel second moment E[x_j^2] for each linear.

    This is the activation-only diagonal of the layerwise Hessian X Xᵀ used by
    GPTQ. Complementary to the (label-aware) Fisher above; returned per module
    as a (in_features,) tensor so it can be compared channel-for-channel with
    AWQ salience in `salient_channels.py`.
    """
    blocks = model.arch.get_blocks(model.hf_model)
    targets: dict[str, nn.Linear] = {}
    for li, block in enumerate(blocks):
        for sub_name, sub in block.named_modules():
            if isinstance(sub, nn.Linear) and sub_name:
                targets[f"blocks.{li}.{sub_name}"] = sub

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_m, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()  # (N, in)
            s = (x ** 2).sum(0).cpu()
            if name not in sums:
                sums[name] = s
                counts[name] = x.shape[0]
            else:
                sums[name] += s
                counts[name] += x.shape[0]
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        for prompt in prompts:
            model.hf_model(input_ids=model.to_tokens(prompt))
    finally:
        for h in handles:
            h.remove()
    return {name: (sums[name] / max(counts[name], 1)) for name in sums}
