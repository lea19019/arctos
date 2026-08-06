"""AWQ salient-channel detection, and whether MT calibration moves the set.

AWQ (Lin et al. 2024, https://arxiv.org/abs/2306.00978) protects ~1% of input
channels — the ones paired with large-magnitude activations — by per-channel
scaling, not by keeping them at higher precision. The salient set is chosen
from a *calibration* corpus, and the whole phase-two question (docs/research.md
§1D) is: **does calibrating on MT data pick different channels than generic
text?** If yes, MT-conditional AWQ is a genuinely new, kernel-free lever; if
no, calibration choice is a red herring and the Q5 dissociation stands.

This module reuses `activation_stats.collect_activation_stats` to get the
per-channel salience (q99 |activation|, AWQ's robust magnitude) under two or
more calibration regimes, then quantifies how much the salient set shifts:

  - top-1% Jaccard overlap of salient channels per module,
  - Spearman correlation of per-channel salience per module,
  - aggregated (mean over modules) so one number summarizes the model.

The default regimes are built from the same FLORES records with three framings
that are all available offline:
  - "mt"     : the full MT instruction prompt (what we actually quantize for),
  - "source" : the raw source sentence, no instruction (generic-ish text),
  - "target" : the raw target sentence (the output-language distribution).
This isolates "task framing" from "language content" without needing a C4
download on an offline compute node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from src.interp.activation_stats import collect_activation_stats


def _top_frac_indices(x: torch.Tensor, frac: float) -> set[int]:
    k = max(1, int(round(frac * x.numel())))
    return set(torch.topk(x, k).indices.tolist())


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.pow(2).sum() * rb.pow(2).sum()).sqrt()
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


@dataclass
class SalienceComparison:
    """How much the AWQ-salient channel set moves between two calibration regimes."""

    regime_a: str
    regime_b: str
    top_frac: float
    mean_jaccard: float                       # over modules present in both
    mean_spearman: float
    per_module_jaccard: dict[str, float] = field(default_factory=dict)
    per_module_spearman: dict[str, float] = field(default_factory=dict)
    n_modules: int = 0


def salience_by_regime(
    model: Any,
    regimes: dict[str, Sequence[str]],
    *,
    quantile: float = 0.99,
    max_positions_per_example: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Per-channel AWQ salience (q-quantile |activation|) per module, per regime.

    Returns {regime: {module_name: salience_vector}}.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for regime, prompts in regimes.items():
        res = collect_activation_stats(
            model, prompts, quantile=quantile,
            max_positions_per_example=max_positions_per_example,
        )
        out[regime] = {name: st.q99_abs for name, st in res.stats_by_module.items()}
    return out


def compare_salience(
    salience: dict[str, dict[str, torch.Tensor]],
    regime_a: str,
    regime_b: str,
    *,
    top_frac: float = 0.01,
) -> SalienceComparison:
    """Top-frac Jaccard + Spearman between two regimes' per-channel salience."""
    a = salience[regime_a]
    b = salience[regime_b]
    common = sorted(set(a) & set(b))
    jac, spr = {}, {}
    for name in common:
        va, vb = a[name], b[name]
        if va.numel() != vb.numel() or va.numel() == 0:
            continue
        sa = _top_frac_indices(va, top_frac)
        sb = _top_frac_indices(vb, top_frac)
        union = sa | sb
        jac[name] = (len(sa & sb) / len(union)) if union else float("nan")
        spr[name] = _spearman(va, vb)
    jv = [v for v in jac.values() if v == v]
    sv = [v for v in spr.values() if v == v]
    return SalienceComparison(
        regime_a=regime_a, regime_b=regime_b, top_frac=top_frac,
        mean_jaccard=float(sum(jv) / len(jv)) if jv else float("nan"),
        mean_spearman=float(sum(sv) / len(sv)) if sv else float("nan"),
        per_module_jaccard=jac, per_module_spearman=spr, n_modules=len(jac),
    )
