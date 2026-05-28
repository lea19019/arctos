"""Activation patching.

Replace activations from a clean prompt with those from a corrupt prompt at
specific layers / heads / MLPs; measure the causal effect on the model's
output for the clean prompt. Gold-standard causal method for circuit
discovery and component-importance scoring.

Cite:
- Vig et al. (2020), "Investigating Gender Bias in Language Models Using
  Causal Mediation Analysis", https://arxiv.org/abs/2004.12265.
- Meng et al. (2022), "Locating and Editing Factual Associations in GPT"
  (ROME), https://arxiv.org/abs/2202.05262.
- Heimersheim & Nanda (2024), "How to use and interpret activation patching",
  https://arxiv.org/abs/2404.15255 — best operational guide; see §4 on
  clean-from-corrupt vs corrupt-from-clean directionality and the metric
  (logit difference vs probability vs KL) trade-off.

Architecture notes:
- All three target models are decoder-only Llama-class; the patch sites
  (resid_pre/post per layer; per-head q/k/v/z; mlp_out) are uniform.
- For MT, "clean" = source the model translates correctly; "corrupt" = a
  paired source designed so that the gold target *would* differ lexically.
  A paraphrase-corrupted source is NOT a good corrupt unless its gold
  translation differs from the clean prompt's. See `src/data/clean_corrupt.py`
  for paired-prompt construction strategies.

Per-model adaptations:
- Aya: Cohere attention scaling means head-level patches must respect the
  scaling factor; verify on a known induction head before trusting MT-head
  patching numbers.
- omt-llama: very large vocabulary — any logit-difference metric on the
  full vocab may need chunking to avoid OOM during the metric computation
  (the residual cache itself is unaffected).
- Tower: SFT prompt template must be respected — clean and corrupt prompts
  share the template's instruction tokens; only the source-text region
  varies.

TODO: implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PatchSite(str, Enum):
    """Which component class to patch."""

    RESID_PRE = "resid_pre"
    RESID_POST = "resid_post"
    ATTN_OUT = "attn_out"
    MLP_OUT = "mlp_out"
    HEAD_Z = "head_z"
    HEAD_Q = "head_q"
    HEAD_K = "head_k"
    HEAD_V = "head_v"


class PatchMetric(str, Enum):
    """How to score the effect of a patch on the clean output."""

    LOGIT_DIFF = "logit_diff"          # clean_target_logit - corrupt_target_logit
    PROB = "prob"                       # P(clean target | patched run)
    KL_FROM_CLEAN = "kl_from_clean"     # KL(patched || clean)


@dataclass
class PatchResult:
    """Effect of patching one site (or set of sites) on the clean output.

    Attributes:
        site: which component(s) were patched.
        layer: layer index (or list of indices, if patching multiple sites).
        head: optional head index for head-level patches.
        position: optional sequence position.
        metric: which metric was used.
        effect: scalar effect size (sign per metric convention).
    """

    site: PatchSite
    layer: int | list[int]
    head: int | None
    position: int | None
    metric: PatchMetric
    effect: float


def activation_patch(
    model: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    *,
    site: PatchSite,
    layer: int,
    head: int | None = None,
    position: int | None = None,
    metric: PatchMetric = PatchMetric.LOGIT_DIFF,
) -> PatchResult:
    """Patch a single site and measure the effect on the clean run.

    Direction: run on `clean_prompt`, replacing the named site's activations
    with the values cached from `corrupt_prompt`. Measure the effect of that
    replacement on the model's output for the clean prompt. (See Heimersheim
    & Nanda 2024 §4 for why this direction is usually preferred.)

    TODO: implement.
    """
    raise NotImplementedError


def patch_sweep(
    model: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    *,
    site: PatchSite,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
    metric: PatchMetric = PatchMetric.LOGIT_DIFF,
) -> list[PatchResult]:
    """Patch every (layer, head) in the requested grid and return all effects.

    Used for ranking — e.g., the per-head MT-importance ranking in Q2.

    TODO: implement.
    """
    raise NotImplementedError
