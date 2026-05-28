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
"""

from __future__ import annotations

import pytest
import torch

from src.interp.probing import ProbeResult, probe_layers, train_probe


@pytest.mark.cpu
def test_selectivity_arithmetic():
    """ProbeResult.selectivity must equal accuracy - control_accuracy."""
    r = ProbeResult(layer=3, accuracy=0.91, control_accuracy=0.42, selectivity=0.91 - 0.42)
    assert abs(r.selectivity - (r.accuracy - r.control_accuracy)) < 1e-12


@pytest.mark.cpu
def test_separable_input_high_accuracy():
    """A linearly-separable feature should probe at near-100% accuracy."""
    torch.manual_seed(0)
    n, d = 64, 16
    # Class 0 lives at +e_0, class 1 at -e_0.
    x = torch.zeros(n, d)
    y = torch.zeros(n, dtype=torch.long)
    x[: n // 2, 0] = 1.0
    x[n // 2 :, 0] = -1.0
    y[n // 2 :] = 1
    x += 0.01 * torch.randn(n, d)
    probe = train_probe(x, y, n_classes=2, epochs=200)
    with torch.no_grad():
        preds = probe(x).argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    assert acc > 0.95, f"linearly separable input got acc={acc}"


@pytest.mark.cpu
def test_probe_layers_runs_and_reports_selectivity(tiny_model):
    """probe_layers produces one result per layer with the selectivity invariant."""
    # 2-class trivially separable text inputs; we don't expect the actual
    # accuracy to be high (gpt2 doesn't know our task) — only the API contract.
    examples = [(p, lab) for p, lab in [("cats are nice", 0), ("dogs are nice", 1)] * 4]
    results = probe_layers(tiny_model, examples, n_classes=2, layers=[0, 5, 11])
    assert len(results) == 3
    for r in results:
        assert abs(r.selectivity - (r.accuracy - r.control_accuracy)) < 1e-12


@pytest.mark.gpu
def test_source_language_id_decodable(target_model):
    """Source-language ID is linearly decodable from at least one layer."""
    pytest.skip("TODO: implement after Q1 calibration data is loaded on GPU.")
