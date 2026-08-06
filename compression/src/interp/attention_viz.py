"""Attention pattern extraction + visualization.

The other four methods tell you *which* heads matter (IFR magnitude, DLA
sign) and *whether* they matter causally (patching). This one answers the
qualitative follow-up: *what does a given head actually do?* — by looking
at its attention pattern (the softmax(QKᵀ) weights) on a real MT prompt.

Typical head taxonomies you can read off a pattern:
  - source-attender : target positions attend back to specific source tokens
  - positional / previous-token : attends to position i-1 (a diagonal band)
  - BOS / attention-sink : mass dumped on the first token
  - induction-like : attends to the token after a previous occurrence

We get attention weights straight from HF with output_attentions=True, which
requires the eager attention implementation (SDPA/flash don't expose them).
The loader here reloads the HF model eager — cheap and isolated from the
main HookedModel path.
"""

from __future__ import annotations

from typing import Any

import torch


def attention_patterns(hf_model: Any, tokenizer, prompt: str) -> tuple[torch.Tensor, list[str]]:
    """Return (attn, str_tokens).

    attn: (n_layers, n_heads, T, T) attention weights for the prompt.
    str_tokens: the T decoded tokens (for axis labels).
    """
    device = next(hf_model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model(**enc, output_attentions=True)
    # out.attentions: tuple of (B, n_heads, T, T), one per layer
    attn = torch.stack([a[0] for a in out.attentions], dim=0).float().cpu()
    str_tokens = [tokenizer.decode([t]) for t in enc["input_ids"][0].tolist()]
    return attn, str_tokens


def classify_pattern(attn_head: torch.Tensor) -> str:
    """Cheap heuristic label for one head's (T, T) pattern at the last query.

    Looks at where the *last* query position puts its mass.
    """
    T = attn_head.shape[-1]
    last = attn_head[-1]  # (T,) weights from the last token
    amax = int(last.argmax())
    if amax == 0 and last[0] > 0.5:
        return "BOS-sink"
    if amax >= T - 2:
        return "local/recent"
    # diagonal band check
    diag = sum(attn_head[i, max(0, i - 1)] for i in range(1, T)) / max(1, T - 1)
    if diag > 0.4:
        return "previous-token"
    return f"attends->tok{amax}"
