"""Translation-quality metrics with caveats documented in-line.

TODO: implement wrappers; do not invent the metric math, use sacrebleu and
unbabel-comet directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MetricScores:
    bleu: float
    chrfpp: float
    comet: float | None  # None if COMET was disabled


def evaluate(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    pair: str,
    use_comet: bool = True,
) -> MetricScores:
    """Compute BLEU + chrF++ (+ COMET) for one batch.

    Tokenizer choices (passed through to sacrebleu):
    - cs-de: default (intl).
    - en-zh: `zh` tokenizer; required for meaningful BLEU on zh-Hans.
    - en-arz: default; document the choice once first results are in. arz
      uses Arabic script; sacrebleu's `intl` should handle it but the
      register shift may dominate the score.

    COMET caveat (en-arz): wmt22-comet-da is trained mostly on standard
    varieties; report chrF++ alongside as a sanity check.

    TODO: implement.
    """
    raise NotImplementedError
