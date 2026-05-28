"""Probing classifiers.

Train small classifiers on hidden states to detect language identity, source
vs target representation, register / dialect, and so on. Cheap; complements
the heavier methods (logit lens, activation patching, IFR).

Cite:
- Hewitt & Liang (2019), "Designing and Interpreting Probes with Control
  Tasks", https://arxiv.org/abs/1909.03368 — control-task setup is required
  to distinguish "the layer represents X" from probe expressivity. Report
  selectivity (accuracy − control-accuracy), not raw accuracy.
- Belinkov (2022), "Probing Classifiers: Promises, Shortcomings, and
  Advances", https://arxiv.org/abs/2102.12452 — survey.

Architecture notes:
- All three target models share Llama-class hidden-state shapes; probe input
  is `resid_post[ℓ]` at a chosen position.
- For MT, the natural probe targets are: source-language ID, target-language
  ID, source-vs-target token (if probing inside the target span), register
  / dialect (relevant for en→ar-arz, where MSA vs arz is the open question
  the project highlights).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass
class ProbeResult:
    """Probe accuracy at one layer.

    Attributes:
        layer: layer index.
        accuracy: held-out accuracy.
        control_accuracy: accuracy on the Hewitt & Liang 2019 control task
            (random labels with the same input distribution).
        selectivity: accuracy - control_accuracy. The selectivity metric is
            what should be reported, not raw accuracy.
    """

    layer: int
    accuracy: float
    control_accuracy: float
    selectivity: float


def train_probe(
    hidden_states: torch.Tensor,  # (N, D)
    labels: torch.Tensor,          # (N,)
    *,
    n_classes: int,
    hidden_dim: int = 0,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> torch.nn.Module:
    """Train a probe (linear by default, optional one-hidden-layer MLP).

    Trained on CPU in fp32 for determinism; hidden_states are converted as
    needed. Returns a probe module callable on (N, D) -> (N, n_classes).
    """
    g = torch.Generator().manual_seed(seed)
    x = hidden_states.detach().to(dtype=torch.float32, device="cpu")
    y = labels.detach().to(dtype=torch.long, device="cpu")
    d = x.shape[-1]

    if hidden_dim <= 0:
        probe: torch.nn.Module = torch.nn.Linear(d, n_classes)
    else:
        probe = torch.nn.Sequential(
            torch.nn.Linear(d, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_classes),
        )
    for p in probe.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02, generator=g) if p.dim() > 1 else torch.nn.init.zeros_(p)

    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    probe.train()
    for _ in range(epochs):
        optim.zero_grad()
        loss = F.cross_entropy(probe(x), y)
        loss.backward()
        optim.step()
    probe.eval()
    return probe


def _accuracy(probe: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        preds = probe(x.to(dtype=torch.float32)).argmax(dim=-1)
    return float((preds == y).float().mean().item())


def _shuffled_labels(labels: torch.Tensor, n_classes: int, seed: int) -> torch.Tensor:
    """Hewitt & Liang control labels: a fixed random label per input.

    For our setup each input is a unique prompt, so we just permute the
    label vector with a seeded RNG — equivalently, a fixed random assignment
    of labels to inputs that destroys the relationship between the feature
    and the input while keeping the label distribution intact.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(labels.shape[0], generator=g)
    # Replace with a fully random label vector (uniform over classes) so
    # the control task is "memorize an arbitrary mapping", not "predict a
    # permuted real label" (which would still be partially learnable when
    # any input feature correlates with the original label).
    return torch.randint(0, n_classes, labels.shape, generator=g)


def probe_layers(
    model: Any,
    examples: Iterable[tuple[str, int]],
    *,
    n_classes: int,
    layers: list[int] | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
) -> list[ProbeResult]:
    """Train one probe per layer and return per-layer accuracy + selectivity.

    Caches resid_post at the last token position of each prompt, splits
    train/test, trains a probe per layer plus a control probe with random
    labels, and reports (accuracy, control_accuracy, selectivity).
    """
    examples = list(examples)
    if not examples:
        raise ValueError("probe_layers got no examples.")
    n_layers = model.cfg.n_layers
    layer_idx = list(range(n_layers)) if layers is None else list(layers)

    # Collect hidden states at the last token of each prompt, per layer.
    names_filter = lambda name: name.endswith("hook_resid_post")  # noqa: E731
    all_hidden: list[list[torch.Tensor]] = [[] for _ in layer_idx]
    all_labels: list[int] = []
    for prompt, label in examples:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)
        last = tokens.shape[-1] - 1
        for i, ell in enumerate(layer_idx):
            h = cache[f"blocks.{ell}.hook_resid_post"][0, last]
            all_hidden[i].append(h.detach().to(dtype=torch.float32, device="cpu"))
        all_labels.append(int(label))

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    n = len(labels_tensor)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = max(1, math.floor(train_frac * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    if test_idx.numel() == 0:
        raise ValueError(f"train_frac={train_frac} on n={n} left no test examples.")

    results: list[ProbeResult] = []
    control_labels = _shuffled_labels(labels_tensor, n_classes, seed=seed + 7)
    for i, ell in enumerate(layer_idx):
        hidden = torch.stack(all_hidden[i], dim=0)  # (N, D)
        x_train, x_test = hidden[train_idx], hidden[test_idx]
        y_train, y_test = labels_tensor[train_idx], labels_tensor[test_idx]
        yc_train, yc_test = control_labels[train_idx], control_labels[test_idx]

        real_probe = train_probe(x_train, y_train, n_classes=n_classes, seed=seed + ell)
        ctrl_probe = train_probe(x_train, yc_train, n_classes=n_classes, seed=seed + 1000 + ell)
        acc = _accuracy(real_probe, x_test, y_test)
        ctrl_acc = _accuracy(ctrl_probe, x_test, yc_test)
        results.append(
            ProbeResult(layer=ell, accuracy=acc, control_accuracy=ctrl_acc, selectivity=acc - ctrl_acc)
        )
    return results
