"""Cross-model Q1 comparison charts.

Given two or more `results/{model}/q1` directories, plot Aya vs Tower (and
later vs others) on the same axes — depth profiles for logit lens, IFR
layer importance, and probing selectivity.

Run from repo root:

    python experiments/q1-language-emergence/make_combined_charts.py \\
        --results results/aya-expanse-8b/q1 results/tower-instruct-7b-v0.2/q1 \\
        --out-dir results/_combined/q1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAIR_LABELS = {"cs-de": "cs→de", "en-zh": "en→zh", "en-arz": "en→arz"}
PAIR_STYLE = {"cs-de": "-", "en-zh": "--", "en-arz": ":"}
MODEL_COLOR = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def load_summary(d: Path) -> dict:
    return json.loads((d / "summary.json").read_text())


def plot_lens_combined(results: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mi, rdir in enumerate(results):
        summary = load_summary(rdir)
        model = summary["model"]
        n_layers = None
        for pi, pair in enumerate(["cs-de", "en-zh", "en-arz"]):
            p = rdir / f"lens_{pair}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            m = d["target_mass"].mean(axis=(0, 2))
            n_layers = len(m)
            xs = np.linspace(0, 1, n_layers)  # depth fraction so models with different L overlay
            ax.plot(xs, m, PAIR_STYLE[pair], color=MODEL_COLOR[mi % len(MODEL_COLOR)],
                    label=f"{model}  {PAIR_LABELS[pair]}", marker="o", markersize=2.5, alpha=0.85)
    ax.set_xlabel("Depth fraction (0 = embed, 1 = final layer)")
    ax.set_ylabel("Mean P(gold target prefix) under logit lens")
    ax.set_title("Q1 — target-language emergence across depth (cross-model)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ifr_combined(results: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mi, rdir in enumerate(results):
        summary = load_summary(rdir)
        model = summary["model"]
        for pair in ["cs-de", "en-zh", "en-arz"]:
            p = rdir / f"ifr_{pair}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            scores = d["layer_scores"]
            xs = np.linspace(0, 1, len(scores))
            ax.plot(xs, scores, PAIR_STYLE[pair], color=MODEL_COLOR[mi % len(MODEL_COLOR)],
                    label=f"{model}  {PAIR_LABELS[pair]}", marker="o", markersize=2.5, alpha=0.85)
    ax.set_xlabel("Depth fraction")
    ax.set_ylabel("IFR layer importance (L1-normalized, attn+mlp)")
    ax.set_title("Q1 — per-layer IFR importance (cross-model)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_depth_partition(results: list[Path], out: Path) -> None:
    """Stacked bar of (first_quarter, mid_half, last_quarter) of IFR mass per (model, pair).

    The Q4 V2 hypothesis is that this partition is similar across models —
    this chart tests it visually.
    """
    rows = []  # (model_pair_label, q1, mid, q4)
    for rdir in results:
        summary = load_summary(rdir)
        model = summary["model"]
        for pair in ["cs-de", "en-zh", "en-arz"]:
            p = rdir / f"ifr_{pair}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            ls = d["layer_scores"]
            n = len(ls)
            q = n // 4
            first = ls[:q].sum() / ls.sum()
            last = ls[-q:].sum() / ls.sum()
            mid = 1.0 - first - last
            rows.append((f"{model}\n{PAIR_LABELS[pair]}", first, mid, last))
    if not rows:
        return
    labels = [r[0] for r in rows]
    first = np.array([r[1] for r in rows])
    mid = np.array([r[2] for r in rows])
    last = np.array([r[3] for r in rows])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(rows)), 4.5))
    ax.bar(x, first, label="first quarter (L0-7)", color="#aec7e8")
    ax.bar(x, mid, bottom=first, label="middle half (L8-23)", color="#ffbb78")
    ax.bar(x, last, bottom=first + mid, label="last quarter (L24-31)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylabel("Share of IFR layer mass")
    ax.set_ylim(0, 1.0)
    ax.set_title("Q1/Q4 — IFR depth partition (Q4 V2 hypothesis check)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_probing_combined(results: list[Path], out: Path, kind: str) -> None:
    """kind: 'target' (uses probing_target_id.json) or 'source' (probing_source_id.json)."""
    fname = {"target": "probing_target_id.json", "source": "probing_source_id.json"}[kind]
    title = {
        "target": "Q1 — target-language probe selectivity across depth (leaky: prompt names target)",
        "source": "Q1 — source-language probe selectivity across depth (raw source, no instruction)",
    }[kind]
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for mi, rdir in enumerate(results):
        summary = load_summary(rdir)
        model = summary["model"]
        p = rdir / fname
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        layers = [r["layer"] for r in rows]
        sel = [r["selectivity"] for r in rows]
        xs = np.linspace(0, 1, len(layers))
        ax.plot(xs, sel, marker="o", markersize=3,
                color=MODEL_COLOR[mi % len(MODEL_COLOR)], label=f"{model}")
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("Depth fraction")
    ax.set_ylabel("Probe selectivity")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, nargs="+", required=True,
                    help="Per-model results dirs, e.g. results/aya-expanse-8b/q1")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_lens_combined(args.results, args.out_dir / "lens_combined.png")
    plot_ifr_combined(args.results, args.out_dir / "ifr_layer_combined.png")
    plot_depth_partition(args.results, args.out_dir / "ifr_depth_partition.png")
    plot_probing_combined(args.results, args.out_dir / "probing_target_id_combined.png", "target")
    plot_probing_combined(args.results, args.out_dir / "probing_source_id_combined.png", "source")

    print(f"wrote combined charts to {args.out_dir}/")
    for p in sorted(args.out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
