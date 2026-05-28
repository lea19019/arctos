"""MT evaluation wrappers.

Reused conceptually from the prior paper, but reimplemented here so the
phase-one repo has no implicit dependencies on prior code. Three metrics:

- BLEU and chrF++ via sacrebleu (with explicit tokenizer choices: zh-Hans
  uses sacrebleu's `zh` tokenizer; en→ar-arz needs a documented choice).
- COMET via Unbabel's COMET (wmt22-comet-da for continuity with the prior
  paper; the WMT25 prescribed metric reported alongside where applicable).

Caveats documented in `metrics.py`: COMET wmt22-comet-da is trained mostly
on standard varieties and may misjudge en→ar-arz translations. Report
chrF++ alongside, plus a small human-judgment spot check.

Tests: `tests/eval/test_metrics.py`.
"""
