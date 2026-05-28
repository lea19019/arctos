"""Q1 analysis — turn `results/{model}/q1/*` into proposal-ready charts.

Run for each model whose results directory is populated:

    python experiments/q1-language-emergence/make_charts.py \\
        --results-dir results/aya-expanse-8b/q1 \\
        --out-dir results/aya-expanse-8b/q1/charts

Produces four figures, one PNG each:

  1. lens_target_mass.png        — mean per-layer target-token mass, one line per pair.
  2. ifr_layer_importance.png    — IFR layer_scores (attn + mlp), one line per pair.
  3. ifr_head_heatmap_{pair}.png — per-(layer, head) IFR shading, one panel per pair.
  4. probing_selectivity.png     — per-layer target-language probe selectivity.

The script reads `summary.json` to discover which pairs are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAIR_LABELS = {
    "cs-de": "cs→de  (Czech → German)",
    "en-zh": "en→zh  (English → Chinese, Hans)",
    "en-arz": "en→arz (English → Egyptian Arabic)",
}
PAIR_COLOR = {"cs-de": "#1f77b4", "en-zh": "#d62728", "en-arz": "#2ca02c"}


def load_summary(results_dir: Path) -> dict:
    return json.loads((results_dir / "summary.json").read_text())


def plot_lens(results_dir: Path, out: Path, model_name: str, pairs: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pair in pairs:
        path = results_dir / f"lens_{pair}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        # target_mass: (N, L, K). Mean over examples and target-prefix tokens.
        mean_per_layer = d["target_mass"].mean(axis=(0, 2))
        ax.plot(range(len(mean_per_layer)), mean_per_layer,
                marker="o", markersize=3, label=PAIR_LABELS.get(pair, pair),
                color=PAIR_COLOR.get(pair))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(gold target prefix) under logit lens")
    ax.set_title(f"{model_name} — target-language mass across depth")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ifr_layers(results_dir: Path, out: Path, model_name: str, pairs: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pair in pairs:
        path = results_dir / f"ifr_{pair}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        scores = d["layer_scores"]
        ax.plot(range(len(scores)), scores,
                marker="o", markersize=3, label=PAIR_LABELS.get(pair, pair),
                color=PAIR_COLOR.get(pair))
    ax.set_xlabel("Layer")
    ax.set_ylabel("IFR layer score (attn + mlp, L1-normalized)")
    ax.set_title(f"{model_name} — per-layer importance at last token")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ifr_heads(results_dir: Path, out_dir: Path, model_name: str, pairs: list[str]) -> None:
    for pair in pairs:
        path = results_dir / f"ifr_{pair}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        head = d["head_scores"]  # (L, H)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(head, aspect="auto", origin="lower", cmap="magma")
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer")
        ax.set_title(f"{model_name} — per-(layer, head) IFR  [{PAIR_LABELS.get(pair, pair)}]")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="IFR score")
        fig.tight_layout()
        fig.savefig(out_dir / f"ifr_head_heatmap_{pair}.png", dpi=150)
        plt.close(fig)


def plot_probing(results_dir: Path, out: Path, model_name: str) -> None:
    path = results_dir / "probing_target_id.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    layers = [r["layer"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    ctrl = [r["control_accuracy"] for r in rows]
    sel = [r["selectivity"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, acc, marker="o", markersize=3, label="probe accuracy", color="#1f77b4")
    ax.plot(layers, ctrl, marker="x", markersize=3, label="control-task accuracy", color="#888888")
    ax.plot(layers, sel, marker="s", markersize=3, label="selectivity (acc − control)", color="#d62728")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{model_name} — target-language probe across depth")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.results_dir / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.results_dir)
    model_name = summary["model"]
    pairs = summary["pairs"]

    plot_lens(args.results_dir, out_dir / "lens_target_mass.png", model_name, pairs)
    plot_ifr_layers(args.results_dir, out_dir / "ifr_layer_importance.png", model_name, pairs)
    plot_ifr_heads(args.results_dir, out_dir, model_name, pairs)
    plot_probing(args.results_dir, out_dir / "probing_target_id.png", model_name)
    # Source-ID probe (no instruction): only present if the new runner produced it.
    src_path = args.results_dir / "probing_source_id.json"
    if src_path.exists():
        old_name = "probing_target_id.json"
        # Reuse plot_probing by temporarily swapping the file it reads.
        rows = json.loads(src_path.read_text())
        layers = [r["layer"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        ctrl = [r["control_accuracy"] for r in rows]
        sel = [r["selectivity"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(layers, acc, marker="o", markersize=3, label="probe accuracy", color="#1f77b4")
        ax.plot(layers, ctrl, marker="x", markersize=3, label="control-task accuracy", color="#888888")
        ax.plot(layers, sel, marker="s", markersize=3, label="selectivity", color="#d62728")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{model_name} — source-language probe (raw source, no instruction)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "probing_source_id.png", dpi=150)
        plt.close(fig)

    print(f"wrote charts under {out_dir}/")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
