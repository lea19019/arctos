"""Tests for src.interp.activation_patching.

CPU invariants:
- Patching with the SAME source as both clean and corrupt is a no-op:
  effect ≈ 0 for any metric.
- Patching every layer simultaneously with the corrupt source yields the
  corrupt-prompt output exactly (full replacement check).
- `PatchResult` field types match.

GPU behavior:
- On a known cs→de clean/corrupt pair, patching at least one layer in the
  middle of the network should produce a non-zero effect; the sign of the
  effect under LOGIT_DIFF should be negative (patching corrupt info into
  clean degrades the clean target's logit relative to the corrupt's).

TODO: implement once `src.interp.activation_patching` is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.cpu
def test_self_patch_is_noop(tiny_model):
    """Patching with clean=corrupt should produce zero effect."""
    pytest.skip("TODO: implement after activation_patching is implemented.")


@pytest.mark.cpu
def test_full_layer_patch_replaces_output(tiny_model):
    """Patching every layer with corrupt activations yields the corrupt output."""
    pytest.skip("TODO: implement after activation_patching is implemented.")


@pytest.mark.gpu
def test_mid_layer_patch_has_nonzero_effect(target_model):
    """Patching some mid-network site on a real clean/corrupt pair moves the metric."""
    pytest.skip("TODO: implement after activation_patching + clean/corrupt generator exist.")


@pytest.mark.gpu
def test_logit_diff_sign_convention(target_model):
    """LOGIT_DIFF should be negative when corrupt info is patched into clean."""
    pytest.skip("TODO: implement after activation_patching + clean/corrupt generator exist.")
