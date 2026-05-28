"""Tests for src.interp.probing.

CPU invariants:
- `ProbeResult.selectivity == accuracy - control_accuracy` exactly.
- Probe accuracy on a trivial separable input ≈ 1.0.
- Control-task accuracy on a probe trained against random labels ≤
  raw probe accuracy (selectivity ≥ 0 in expectation).

GPU behavior:
- On a real model, source-language ID should be linearly decodable above
  chance from at least one layer's residual (else something is very wrong
  with the loader or the data path).

TODO: implement once `src.interp.probing` is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.cpu
def test_selectivity_arithmetic():
    """ProbeResult.selectivity must equal accuracy - control_accuracy."""
    pytest.skip("TODO: implement after probing is implemented.")


@pytest.mark.cpu
def test_separable_input_high_accuracy(tiny_model):
    """A linearly-separable feature should probe at near-100% accuracy."""
    pytest.skip("TODO: implement after probing is implemented.")


@pytest.mark.cpu
def test_control_task_runs(tiny_model):
    """Control-task probes train without error and report a sensible accuracy."""
    pytest.skip("TODO: implement after probing is implemented.")


@pytest.mark.gpu
def test_source_language_id_decodable(target_model):
    """Source-language ID is linearly decodable from at least one layer."""
    pytest.skip("TODO: implement after probing + Q1 calibration data exist.")
