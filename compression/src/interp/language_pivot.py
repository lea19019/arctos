"""Language-pivot trajectory — does the model 'think in English' mid-stack?

The single most direct probe of *how* translation works inside a
decoder LLM. Wendler et al. (2024, "Do Llamas Work in English?") showed
that for translation, models often move through a pivot/concept space
aligned with their dominant pretraining language (usually English) in the
middle layers, and only convert to the target language's tokens in the
final layers.

We measure this with the logit lens + **script detection**. For the
cross-script pairs (en→zh: Han; en→arz: Arabic) the target language uses a
different writing system from English/source (Latin), so the script of the
lens's top tokens at each layer cleanly reveals the trajectory:

  early/middle layers dominated by Latin (English/source pivot)
  → late layers switch to target script (Han / Arabic)

is the signature of "think in English, emit in target."

We report, per layer, the share of the top-k lens probability mass whose
decoded tokens are Latin vs target-script vs other.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

import torch

TARGET_SCRIPT = {"en-zh": "han", "en-arz": "arabic", "cs-de": "latin"}


def script_of(s: str) -> str:
    """Dominant alphabetic script of a decoded token string."""
    cats = set()
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "CJK" in name or "HAN " in name:
            cats.add("han")
        elif "ARABIC" in name:
            cats.add("arabic")
        elif "LATIN" in name:
            cats.add("latin")
        elif "CYRILLIC" in name:
            cats.add("cyrillic")
    for s_ in ("han", "arabic", "cyrillic", "latin"):  # target scripts first
        if s_ in cats:
            return s_
    return "other"


@dataclass
class PivotTrajectory:
    """Per-layer script-mass shares at the generation position.

    target_share[ℓ], latin_share[ℓ], other_share[ℓ] — each (L,), averaged
    over the calibration set. For cross-script pairs, target vs latin is the
    pivot signal.
    """

    target_share: torch.Tensor
    latin_share: torch.Tensor
    other_share: torch.Tensor
    n_examples: int


def _precompute_token_scripts(model: Any) -> list[str]:
    """Script label for every vocab token (decoded once, cached)."""
    tok = model.tokenizer
    vocab_size = model.cfg.d_vocab
    # decode in batches for speed
    scripts = []
    ids = list(range(vocab_size))
    for i in ids:
        scripts.append(script_of(tok.decode([i])))
    return scripts


def pivot_trajectory(
    model: Any,
    prompts: list[str],
    pair: str,
    *,
    top_k: int = 50,
    token_scripts: list[str] | None = None,
) -> PivotTrajectory:
    """Per-layer top-k script-mass shares under the logit lens.

    Args:
        model: HookedModel.
        prompts: MT prompts (ending where the model emits the target).
        pair: language pair, sets the target script.
        top_k: how many top lens tokens to attribute per layer.
        token_scripts: optional precomputed vocab->script (expensive to build,
            pass in to reuse across pairs).
    """
    from src.interp.logit_lens import _decode_resid

    target_script = TARGET_SCRIPT[pair]
    if token_scripts is None:
        token_scripts = _precompute_token_scripts(model)
    scripts_arr = token_scripts

    n_layers = model.cfg.n_layers
    tgt = torch.zeros(n_layers, dtype=torch.float64)
    lat = torch.zeros(n_layers, dtype=torch.float64)
    oth = torch.zeros(n_layers, dtype=torch.float64)
    n = 0

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cached = model.run_with_cache(tokens, capture=("resid_post",))
        pos = tokens.shape[-1] - 1
        resid = cached.resid_post[:, pos, :]            # (L, d)
        logits = _decode_resid(model, resid)             # (L, V)
        probs = logits.float().softmax(dim=-1)
        topv, topi = probs.topk(top_k, dim=-1)           # (L, k)
        for ell in range(n_layers):
            for v, idx in zip(topv[ell].tolist(), topi[ell].tolist()):
                s = scripts_arr[idx]
                if s == target_script:
                    tgt[ell] += v
                elif s == "latin":
                    lat[ell] += v
                else:
                    oth[ell] += v
        n += 1

    return PivotTrajectory(
        target_share=(tgt / n).float(),
        latin_share=(lat / n).float(),
        other_share=(oth / n).float(),
        n_examples=n,
    )
