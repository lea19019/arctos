"""Thin per-architecture activation-capture wrapper for HF causal LMs.

Replaces TransformerLens's `HookedTransformer.run_with_cache` for the three
phase-one models (Aya / omt-llama / Tower — all Llama-class) and for the
gpt2 CPU test fixture. Built directly on `torch.nn.Module.register_forward_hook`
so there's no third-party tracer in the way.

The activations we capture per forward pass are exactly what Q1 needs:

  resid_post[ℓ]   — block ℓ's output, shape (T, D)
  attn_z[ℓ]       — input to the attention output projection (before W_O),
                    reshaped to (T, H, d_head)
  mlp_out[ℓ]      — MLP block's output, shape (T, D)
  embed           — token-embedding output, shape (T, D)

A per-architecture `ArchPaths` tells the wrapper *which* HF submodules to
hook. Llama-class and GPT-2 are wired here. Adding a new architecture is
~30 lines.

The wrapper exposes a TL-like surface (cfg.n_layers, ln_final, W_U, W_O[ℓ])
so the interpretability methods don't have to branch per-arch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class HookedConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_vocab: int


@dataclass
class CachedActivations:
    """Activations from one forward pass, batch dim already squeezed."""

    resid_post: torch.Tensor  # (L, T, D)
    attn_z: torch.Tensor      # (L, T, H, d_head)
    mlp_out: torch.Tensor     # (L, T, D)
    embed: torch.Tensor       # (T, D)


@dataclass
class ArchPaths:
    """Per-architecture pointers into an HF model.

    Each callable takes the HF model and returns the relevant submodule(s).
    `extract_block_resid` and `extract_attn_pre_proj_in` know how to turn
    the raw forward output / input into the canonical shapes the methods
    expect.
    """

    get_blocks: Callable[[nn.Module], list[nn.Module]]
    get_embed: Callable[[nn.Module], nn.Module]
    get_ln_final: Callable[[nn.Module], nn.Module]
    get_lm_head: Callable[[nn.Module], nn.Linear]

    get_block_attn_proj: Callable[[nn.Module], nn.Linear]
    get_block_mlp: Callable[[nn.Module], nn.Module]

    extract_block_resid: Callable[[object], torch.Tensor]  # block fwd output -> (B, T, D)


def _block_resid_first_of_tuple(out):
    """For Llama / GPT-2: block forward output is a tuple whose first element is the hidden states."""
    return out[0] if isinstance(out, tuple) else out


LLAMA_PATHS = ArchPaths(
    get_blocks=lambda m: list(m.model.layers),
    get_embed=lambda m: m.model.embed_tokens,
    get_ln_final=lambda m: m.model.norm,
    get_lm_head=lambda m: m.lm_head,
    get_block_attn_proj=lambda b: b.self_attn.o_proj,
    get_block_mlp=lambda b: b.mlp,
    extract_block_resid=_block_resid_first_of_tuple,
)

GPT2_PATHS = ArchPaths(
    get_blocks=lambda m: list(m.transformer.h),
    get_embed=lambda m: m.transformer.wte,
    get_ln_final=lambda m: m.transformer.ln_f,
    get_lm_head=lambda m: m.lm_head,
    get_block_attn_proj=lambda b: b.attn.c_proj,
    get_block_mlp=lambda b: b.mlp,
    extract_block_resid=_block_resid_first_of_tuple,
)

# BLOOM (BigScience) — similar to GPT-2 layout but with different submodule
# names and ALiBi positional encoding. Uses LayerNorm, not RMSNorm; DLA's
# `_rms_norm_scale` falls back to a LayerNorm-compatible scale when gamma
# is present and the module is a LayerNorm.
BLOOM_PATHS = ArchPaths(
    get_blocks=lambda m: list(m.transformer.h),
    get_embed=lambda m: m.transformer.word_embeddings,
    get_ln_final=lambda m: m.transformer.ln_f,
    get_lm_head=lambda m: m.lm_head,
    # BLOOM's per-head output projection is named `dense`. Its input is the
    # concatenated per-head attention output (B, T, n_heads*d_head), so per-
    # head slicing works identically to Llama-class o_proj.
    get_block_attn_proj=lambda b: b.self_attention.dense,
    # BLOOM's MLP output is dense_4h_to_h; the mlp module's forward returns
    # the (B, T, D) hidden state directly (no tuple).
    get_block_mlp=lambda b: b.mlp,
    extract_block_resid=_block_resid_first_of_tuple,
)


class HookedModel:
    """HF causal LM + activation capture, with a TL-shaped API.

    Use `run_with_cache(tokens)` to get a `CachedActivations` for one forward.
    Use `to_tokens(text)` to tokenize a string into a (1, T) tensor on the
    model's device. Pass `capture=...` to skip un-needed components.
    """

    def __init__(self, hf_model: nn.Module, tokenizer, arch: ArchPaths, *, n_heads: int | None = None):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.arch = arch
        cfg = hf_model.config
        n_heads = n_heads or getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head")
        d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
        d_vocab = getattr(cfg, "vocab_size")
        n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.cfg = HookedConfig(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_head=d_model // n_heads,
            d_vocab=d_vocab,
        )
        # Precompute per-layer W_O reshaped to (n_heads, d_head, d_model)
        # so methods can use per-head decomposition uniformly. For HF
        # Linear, weight is (out, in) == (d_model, n_heads * d_head); the
        # per-head W_O[h] is weight[:, h * d_head : (h+1) * d_head].T.
        # For GPT-2's Conv1D, weight is (in, out) == (n_heads * d_head, d_model);
        # per-head W_O[h] is weight[h * d_head : (h+1) * d_head, :].
        W_O_per_layer = []
        for block in self.arch.get_blocks(hf_model):
            proj = self.arch.get_block_attn_proj(block)
            w = proj.weight.detach()
            if w.shape == (d_model, n_heads * self.cfg.d_head):
                # nn.Linear: (out, in)
                w_oh = w.t().reshape(n_heads, self.cfg.d_head, d_model)
            elif w.shape == (n_heads * self.cfg.d_head, d_model):
                # GPT-2 Conv1D: (in, out)
                w_oh = w.reshape(n_heads, self.cfg.d_head, d_model)
            else:
                raise ValueError(
                    f"Unexpected attn output proj weight shape {w.shape} for "
                    f"d_model={d_model}, n_heads={n_heads}, d_head={self.cfg.d_head}"
                )
            W_O_per_layer.append(w_oh)
        self.W_O = torch.stack(W_O_per_layer, dim=0)  # (L, H, d_head, D)

    # ----- TL-shaped surface that methods use -------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.hf_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.hf_model.parameters()).dtype

    @property
    def W_U(self) -> torch.Tensor:
        # HF lm_head: Linear(d_model, vocab) with weight shape (vocab, d_model).
        # TL convention: W_U is (d_model, vocab). Return the transpose.
        return self.arch.get_lm_head(self.hf_model).weight.t()

    @property
    def b_U(self) -> torch.Tensor | None:
        return self.arch.get_lm_head(self.hf_model).bias

    def ln_final(self, x: torch.Tensor) -> torch.Tensor:
        return self.arch.get_ln_final(self.hf_model)(x)

    def parameters(self):
        return self.hf_model.parameters()

    # ----- tokenization -----------------------------------------------------

    def to_tokens(self, text: str, *, add_special: bool = True) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special).input_ids
        return ids.to(self.device)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (B, T, V). For when methods just need logits."""
        with torch.no_grad():
            out = self.hf_model(input_ids=tokens)
        return out.logits if hasattr(out, "logits") else out[0]

    # ----- forward + capture ------------------------------------------------

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        *,
        capture: tuple[str, ...] = ("resid_post", "attn_z", "mlp_out", "embed"),
    ) -> tuple[torch.Tensor, CachedActivations]:
        """One forward + activation capture.

        Returns (logits, cached). Logits shape (B, T, V); cached has each
        tensor's batch dim squeezed (assumes batch size 1, which is what Q1
        uses — extension is straightforward).
        """
        if tokens.shape[0] != 1:
            raise NotImplementedError("HookedModel currently assumes batch size 1.")
        capture_set = set(capture)

        blocks = self.arch.get_blocks(self.hf_model)
        embed_mod = self.arch.get_embed(self.hf_model)

        resid_per_layer: list[torch.Tensor] = [None] * len(blocks)
        attn_z_per_layer: list[torch.Tensor] = [None] * len(blocks)
        mlp_per_layer: list[torch.Tensor] = [None] * len(blocks)
        embed_holder: list[torch.Tensor | None] = [None]

        handles = []

        def make_resid_hook(i):
            def hook(_module, _inp, out):
                resid_per_layer[i] = self.arch.extract_block_resid(out).detach()
            return hook

        def make_attn_z_hook(i):
            def hook(_module, inp, _out):
                # inp is a tuple; inp[0] is the actual input to the projection
                x = inp[0].detach()
                # reshape last dim to (H, d_head)
                resid_per_layer  # noop, just to silence linters
                t_shape = x.shape[:-1]
                attn_z_per_layer[i] = x.reshape(*t_shape, self.cfg.n_heads, self.cfg.d_head)
            return hook

        def make_mlp_hook(i):
            def hook(_module, _inp, out):
                mlp_per_layer[i] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook

        def embed_hook(_module, _inp, out):
            embed_holder[0] = (out[0] if isinstance(out, tuple) else out).detach()

        if "resid_post" in capture_set:
            for i, block in enumerate(blocks):
                handles.append(block.register_forward_hook(make_resid_hook(i)))
        if "attn_z" in capture_set:
            for i, block in enumerate(blocks):
                proj = self.arch.get_block_attn_proj(block)
                handles.append(proj.register_forward_hook(make_attn_z_hook(i)))
        if "mlp_out" in capture_set:
            for i, block in enumerate(blocks):
                mlp = self.arch.get_block_mlp(block)
                handles.append(mlp.register_forward_hook(make_mlp_hook(i)))
        if "embed" in capture_set:
            handles.append(embed_mod.register_forward_hook(embed_hook))

        try:
            with torch.no_grad():
                out = self.hf_model(input_ids=tokens)
            logits = out.logits if hasattr(out, "logits") else out[0]
        finally:
            for h in handles:
                h.remove()

        def _stack_or_empty(layer_list, expected_first_shape):
            if not all(t is not None for t in layer_list):
                # Capture wasn't requested → fill with zeros of an obvious shape
                return torch.zeros(0)
            return torch.stack([t.squeeze(0) for t in layer_list], dim=0)

        cached = CachedActivations(
            resid_post=_stack_or_empty(resid_per_layer, None) if "resid_post" in capture_set else torch.zeros(0),
            attn_z=_stack_or_empty(attn_z_per_layer, None) if "attn_z" in capture_set else torch.zeros(0),
            mlp_out=_stack_or_empty(mlp_per_layer, None) if "mlp_out" in capture_set else torch.zeros(0),
            embed=embed_holder[0].squeeze(0) if (embed_holder[0] is not None) else torch.zeros(0),
        )
        return logits, cached
