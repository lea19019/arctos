"""Compression sandbox: faithful weight quantization + pruning primitives.

This is the "shrink / keep / prune" engine. Unlike Arctos's Q5 sensitivity
proxy (relative i.i.d. Gaussian noise), the perturbations here are the *real*
operations a compression method performs, so chrF++ measured under them is the
quantity the phase-two method actually cares about:

  SHRINK  — `absmax_quantize`: round-to-nearest INT-k weight quantization,
            per-output-channel scales, optional group size. The honest base
            case (what GPTQ/AWQ improve on). Error clips outliers — exactly
            the structure Gaussian noise misses.
  KEEP    — `keep_cols=` (mixed precision): leave a chosen set of input
            channels in full precision and quantize the rest. Also
            `awq_scale`: AWQ-style per-input-channel scaling that shrinks
            error on salient channels without a sparse side-path.
  PRUNE   — `wanda_mask` (|W|·‖X‖, Sun et al. 2024,
            https://arxiv.org/abs/2306.11695) and `magnitude_mask` (|W|).
            The super-weight stress test (prune-N-largest vs prune-the-1)
            falsifies magnitude as a saliency.

All mutators are context managers that edit weights in place and restore them
exactly on exit, so a runner can do `with quantize_linears(...): chrf(...)`.
Originals are cloned (transient 2x weight memory — fine on A100; fine for the
small models we validate on).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Core tensor ops (pure; unit-testable on CPU)
# --------------------------------------------------------------------------- #

def absmax_quantize(
    W: torch.Tensor,
    bits: int,
    *,
    group_size: int | None = None,
    symmetric: bool = True,
    keep_cols: Sequence[int] | None = None,
) -> torch.Tensor:
    """Round-to-nearest INT-`bits` quantize-dequantize of a weight matrix.

    W is (out, in). Scales are per-output-row (per group along the input dim if
    `group_size` is set). Returns a dequantized tensor of the same dtype/shape
    — the quantization *error* is W - return.

    keep_cols: input-channel indices to leave untouched (mixed precision). Use
    for "keep salient channels in FP16, shrink the rest".
    """
    Wf = W.detach().float()
    out_f, in_f = Wf.shape
    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1

    def _q_block(block: torch.Tensor) -> torch.Tensor:
        if symmetric:
            scale = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
            q = torch.clamp(torch.round(block / scale), -qmax - 1, qmax)
            return q * scale
        lo = block.amin(dim=1, keepdim=True)
        hi = block.amax(dim=1, keepdim=True)
        scale = (hi - lo).clamp_min(1e-8) / qmax
        z = torch.round(-lo / scale)
        q = torch.clamp(torch.round(block / scale) + z, 0, qmax)
        return (q - z) * scale

    if group_size is None or group_size >= in_f:
        Wq = _q_block(Wf)
    else:
        cols = []
        for start in range(0, in_f, group_size):
            cols.append(_q_block(Wf[:, start:start + group_size]))
        Wq = torch.cat(cols, dim=1)

    if keep_cols:
        idx = torch.as_tensor(list(keep_cols), device=Wq.device, dtype=torch.long)
        Wq[:, idx] = Wf[:, idx]
    return Wq.to(W.dtype)


def ternary_quantize(W: torch.Tensor, *, group_size: int | None = 128,
                     keep_cols: Sequence[int] | None = None) -> torch.Tensor:
    """1.58-bit ternary {-s, 0, +s} (BitNet b1.58 absmean), per-(row, group).

    s = mean(|W|) over the group; q = clamp(round(W/s), -1, 1). Nominal log2(3)
    ≈ 1.58 bits/weight (plus the group scale). The frontier below 2-bit.
    """
    Wf = W.detach().float()
    out_f, in_f = Wf.shape
    gs = group_size if (group_size and group_size < in_f) else in_f

    def _q(block):
        s = block.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.clamp(torch.round(block / s), -1, 1) * s

    cols = [_q(Wf[:, i:i + gs]) for i in range(0, in_f, gs)]
    Wq = torch.cat(cols, dim=1)
    if keep_cols:
        idx = torch.as_tensor(list(keep_cols), device=Wq.device, dtype=torch.long)
        Wq[:, idx] = Wf[:, idx]
    return Wq.to(W.dtype)


def binary_quantize(W: torch.Tensor, *, group_size: int | None = 128,
                    keep_cols: Sequence[int] | None = None) -> torch.Tensor:
    """1-bit binary {-s, +s} (XNOR-Net style), per-(row, group). s = mean(|W|)."""
    Wf = W.detach().float()
    out_f, in_f = Wf.shape
    gs = group_size if (group_size and group_size < in_f) else in_f

    def _q(block):
        s = block.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.where(block >= 0, s, -s)

    cols = [_q(Wf[:, i:i + gs]) for i in range(0, in_f, gs)]
    Wq = torch.cat(cols, dim=1)
    if keep_cols:
        idx = torch.as_tensor(list(keep_cols), device=Wq.device, dtype=torch.long)
        Wq[:, idx] = Wf[:, idx]
    return Wq.to(W.dtype)


# Bit-spec parsing: an int (>=2) routes to absmax; "ternary"/"1.58" -> ternary;
# "binary"/"1" -> binary. Lets the runner sweep 4,3,2,1.58,1 uniformly.
EFFECTIVE_BITS = {"ternary": 1.58, "1.58": 1.58, "binary": 1.0, "1": 1.0}


def parse_spec(spec):
    """Return (kind, nominal_bits): kind in {'int','ternary','binary'}."""
    s = str(spec).lower()
    if s in ("ternary", "1.58"):
        return "ternary", 1.58
    if s in ("binary", "1"):
        return "binary", 1.0
    return "int", int(spec)


def quant_weight(W: torch.Tensor, spec, *, group_size: int | None = 128,
                 keep_cols: Sequence[int] | None = None) -> torch.Tensor:
    """Quantize-dequantize W to a bit-spec (int >=2, 'ternary'/1.58, or 'binary'/1)."""
    kind, b = parse_spec(spec)
    if kind == "ternary":
        return ternary_quantize(W, group_size=group_size, keep_cols=keep_cols)
    if kind == "binary":
        return binary_quantize(W, group_size=group_size, keep_cols=keep_cols)
    return absmax_quantize(W, b, group_size=group_size, keep_cols=keep_cols)


def awq_scale(act_scale: torch.Tensor, alpha: float) -> torch.Tensor:
    """AWQ per-input-channel scale s_j = act_scale_j^alpha (normalized to mean 1).

    Applying s before quantization (W*s) and folding 1/s into the activations
    shrinks rounding error on high-activation channels. Here we return s so the
    caller can simulate weight-only error: quantize (W*s), then divide back.
    """
    s = act_scale.float().clamp_min(1e-8) ** alpha
    return s / s.mean().clamp_min(1e-8)


def quantize_awq_weight(
    W: torch.Tensor, act_scale: torch.Tensor, bits: int, *, alpha: float = 0.5,
    group_size: int | None = 128,
) -> torch.Tensor:
    """Simulate AWQ weight-only error: dequant( q(W * s) ) / s, per input channel."""
    s = awq_scale(act_scale, alpha).to(W.device)            # (in,)
    Wq = absmax_quantize(W.detach().float() * s.unsqueeze(0), bits, group_size=group_size)
    return (Wq / s.unsqueeze(0)).to(W.dtype)


def wanda_mask(W: torch.Tensor, act_norm: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Wanda keep-mask: prune lowest |W_ij|·‖X_j‖ per output row (per-row %)."""
    Wf = W.detach().float()
    score = Wf.abs() * act_norm.float().to(Wf.device).unsqueeze(0)   # (out,in)
    k = int(round(sparsity * Wf.shape[1]))
    if k <= 0:
        return torch.ones_like(Wf, dtype=torch.bool)
    thresh = score.kthvalue(k, dim=1, keepdim=True).values
    return score > thresh


def magnitude_mask(W: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep-mask: prune the globally smallest-|W| fraction."""
    Wf = W.detach().float().abs()
    k = int(round(sparsity * Wf.numel()))
    if k <= 0:
        return torch.ones_like(Wf, dtype=torch.bool)
    thresh = Wf.flatten().kthvalue(k).values
    return Wf > thresh


# --------------------------------------------------------------------------- #
# In-place model mutators (context managers; restore on exit)
# --------------------------------------------------------------------------- #

def _iter_named_linears(model: Any, layers: Iterable[int] | None):
    blocks = model.arch.get_blocks(model.hf_model)
    sel = range(len(blocks)) if layers is None else layers
    for li in sel:
        for sub_name, sub in blocks[li].named_modules():
            if isinstance(sub, nn.Linear) and sub_name:
                yield f"blocks.{li}.{sub_name}", sub


@contextmanager
def quantize_linears(
    model: Any,
    bits: int,
    *,
    layers: Iterable[int] | None = None,
    group_size: int | None = 128,
    act_scales: dict[str, torch.Tensor] | None = None,
    awq_alpha: float | None = None,
    keep_cols_by_module: dict[str, Sequence[int]] | None = None,
):
    """Quantize linears in place; restore exactly on exit.

    If `awq_alpha` and `act_scales` are given, uses AWQ per-channel scaling.
    Otherwise plain absmax RTN. `keep_cols_by_module` keeps named channels FP16.
    """
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for name, mod in _iter_named_linears(model, layers):
            W = mod.weight
            orig = W.detach().clone()
            saved.append((W, orig))
            kind, _ = parse_spec(bits)
            if awq_alpha is not None and act_scales and name in act_scales and kind == "int":
                Wq = quantize_awq_weight(orig, act_scales[name], int(bits),
                                         alpha=awq_alpha, group_size=group_size)
            else:
                keep = (keep_cols_by_module or {}).get(name)
                Wq = quant_weight(orig, bits, group_size=group_size, keep_cols=keep)
            with torch.no_grad():
                W.copy_(Wq)
        yield
    finally:
        with torch.no_grad():
            for W, orig in saved:
                W.copy_(orig)


@contextmanager
def prune_linears(
    model: Any,
    sparsity: float,
    *,
    method: str = "wanda",
    layers: Iterable[int] | None = None,
    act_norms: dict[str, torch.Tensor] | None = None,
):
    """Zero a `sparsity` fraction of weights per linear; restore on exit.

    method: "wanda" (needs act_norms[name]) or "magnitude".
    """
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for name, mod in _iter_named_linears(model, layers):
            W = mod.weight
            orig = W.detach().clone()
            saved.append((W, orig))
            if method == "wanda" and act_norms and name in act_norms:
                mask = wanda_mask(orig, act_norms[name], sparsity)
            else:
                mask = magnitude_mask(orig, sparsity)
            with torch.no_grad():
                W.mul_(mask.to(W.dtype))
        yield
    finally:
        with torch.no_grad():
            for W, orig in saved:
                W.copy_(orig)


# --------------------------------------------------------------------------- #
# GPTQ — the quantizer that actually USES calibration data
# --------------------------------------------------------------------------- #
# RTN ignores the calibration set and AWQ uses only a per-channel scale, so
# neither can show whether MT-vs-generic calibration matters. GPTQ minimizes the
# reconstruction error ‖WX − W_qX‖ using the input second-moment H = X Xᵀ, with
# error feedback across columns (Frantar et al. 2023, arXiv:2210.17323). This is
# where calibration domain has the most leverage — the proper test of
# "does MT calibration help quantization".

@torch.no_grad()
def collect_hessians(model: Any, prompts: Sequence[str], *,
                     layers: Iterable[int] | None = None) -> dict[str, torch.Tensor]:
    """One calibration pass; accumulate H = sum_t x_t x_tᵀ per linear (on CPU).

    Returns {module_name: H (in, in) float32 on CPU}. Memory: dominated by the
    MLP down-projection (in = intermediate size); ~tens of GB for an 8B model,
    so the caller's job should request generous RAM.
    """
    H: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_m, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)  # (N, in)
            h = (x.t() @ x).cpu()
            if name in H:
                H[name] += h
                counts[name] += x.shape[0]
            else:
                H[name] = h
                counts[name] = x.shape[0]
        return hook

    for name, mod in _iter_named_linears(model, layers):
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        for p in prompts:
            model.hf_model(input_ids=model.to_tokens(p))
    finally:
        for h in handles:
            h.remove()
    for name in H:
        H[name] /= max(counts[name], 1)
    return H


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, *,
                  group_size: int | None = 128, percdamp: float = 0.01) -> torch.Tensor:
    """GPTQ quantize-dequantize of W (out, in) given input second-moment H (in, in).

    Per-output-row symmetric grid (group-wise along input if group_size set),
    with OBS-style error feedback through the Cholesky of H⁻¹.
    """
    dev = W.device
    Wf = W.detach().float().clone()
    out_f, in_f = Wf.shape
    H = H.to(dev).float().clone()
    qmax = (1 << (bits - 1)) - 1

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    Wf[:, dead] = 0.0
    damp = percdamp * torch.diag(H).mean().clamp_min(1e-8)
    H[range(in_f), range(in_f)] += damp

    # Hinv (upper-Cholesky of the inverse), the standard GPTQ factorization.
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    gs = group_size if (group_size and group_size < in_f) else in_f
    # per-(row, group) symmetric scale from the original weights
    scales = torch.empty(out_f, (in_f + gs - 1) // gs, device=dev)
    for gi, start in enumerate(range(0, in_f, gs)):
        blk = Wf[:, start:start + gs]
        scales[:, gi] = blk.abs().amax(dim=1).clamp_min(1e-8) / qmax

    Q = torch.zeros_like(Wf)
    for i1 in range(0, in_f, 128):
        i2 = min(i1 + 128, in_f)
        W1 = Wf[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]
            s = scales[:, col // gs]
            q = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        Wf[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q.to(W.dtype)


@contextmanager
def gptq_quantize_linears(model: Any, bits: int, hessians: dict[str, torch.Tensor], *,
                          layers: Iterable[int] | None = None, group_size: int | None = 128):
    """GPTQ-quantize linears in place using precomputed Hessians; restore on exit.

    Modules without a Hessian fall back to plain absmax RTN.
    """
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for name, mod in _iter_named_linears(model, layers):
            W = mod.weight
            orig = W.detach().clone()
            saved.append((W, orig))
            if name in hessians:
                Wq = gptq_quantize(orig, hessians[name], bits, group_size=group_size)
            else:
                Wq = absmax_quantize(orig, bits, group_size=group_size)
            with torch.no_grad():
                W.copy_(Wq)
        yield
    finally:
        with torch.no_grad():
            for W, orig in saved:
                W.copy_(orig)


def bits_by_fisher(per_module_fisher: dict[str, float], avg_bits: float,
                   bit_choices: Sequence[int] = (2, 3, 4)) -> dict[str, int]:
    """Assign per-module bit-widths from an MT-conditional sensitivity signal.

    Modules are ranked by Fisher (loss-curvature) and split into bit tiers so
    the *average* bit-width ≈ avg_bits — high-Fisher modules get more bits, low
    get fewer. This is the sensitivity-native, MT-conditional allocator Q5 said
    we need (importance ⟂ sensitivity, so don't allocate by importance).
    """
    names = list(per_module_fisher)
    if not names:
        return {}
    order = sorted(names, key=lambda n: per_module_fisher[n])  # ascending sensitivity
    lo, hi = min(bit_choices), max(bit_choices)
    n = len(order)
    # binary search the fraction at hi-bits so mean hits avg_bits (2-tier lo/hi)
    # frac*hi + (1-frac)*lo = avg  ->  frac = (avg-lo)/(hi-lo)
    frac_hi = max(0.0, min(1.0, (avg_bits - lo) / (hi - lo)))
    n_hi = int(round(frac_hi * n))
    out: dict[str, int] = {}
    for i, name in enumerate(order):
        out[name] = hi if i >= n - n_hi else lo
    return out


def module_bits_from_layer_bits(model: Any, layer_bits: dict[int, int], *,
                                layers: Iterable[int] | None = None,
                                default: int = 4) -> dict[str, int]:
    """Expand a per-LAYER bit assignment to the per-MODULE dict quantize_mixed_
    precision wants. Module names are 'blocks.{li}.<sub>'."""
    out: dict[str, int] = {}
    for name, _mod in _iter_named_linears(model, layers):
        li = int(name.split(".")[1])
        out[name] = layer_bits.get(li, default)
    return out


def allocate_layer_bits(layer_drops: dict[int, float], avg_bits: float,
                        low: int = 3, high: int = 4) -> dict[int, int]:
    """Assign `high` bits to the most-sensitive layers, `low` to the rest, so the
    average ≈ avg_bits. Sensitivity = measured quality drop when that layer alone
    is quantized to `low` (a direct, sensitivity-native signal — not Fisher)."""
    layers = list(layer_drops)
    if not layers:
        return {}
    n = len(layers)
    frac_hi = max(0.0, min(1.0, (avg_bits - low) / (high - low)))
    n_hi = int(round(frac_hi * n))
    order = sorted(layers, key=lambda li: layer_drops[li], reverse=True)  # most sensitive first
    hi = set(order[:n_hi])
    return {li: (high if li in hi else low) for li in layers}


@contextmanager
def quantize_mixed_precision(model: Any, bits_by_module: dict[str, int], *,
                             layers: Iterable[int] | None = None,
                             group_size: int | None = 128):
    """Quantize each linear at its assigned bit-width (RTN); restore on exit."""
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for name, mod in _iter_named_linears(model, layers):
            W = mod.weight
            orig = W.detach().clone()
            saved.append((W, orig))
            b = bits_by_module.get(name, max(bits_by_module.values()) if bits_by_module else 4)
            with torch.no_grad():
                W.copy_(absmax_quantize(orig, b, group_size=group_size))
        yield
    finally:
        with torch.no_grad():
            for W, orig in saved:
                W.copy_(orig)


@contextmanager
def ablate_weights(model: Any, coords: Sequence[tuple[int, int, int]]):
    """Zero specific scalar weights of MLP output projections; restore on exit.

    coords: list of (layer, out_dim, in_dim) — e.g. detected super weights.
    Used for the super-weight stress test.
    """
    from src.interp.super_weights import _mlp_out_linear

    blocks = model.arch.get_blocks(model.hf_model)
    saved: list[tuple[torch.Tensor, int, int, float]] = []
    try:
        for (li, o, j) in coords:
            W = _mlp_out_linear(model, blocks[li]).weight
            saved.append((W, o, j, float(W[o, j])))
            with torch.no_grad():
                W[o, j] = 0.0
        yield
    finally:
        with torch.no_grad():
            for W, o, j, v in saved:
                W[o, j] = v
