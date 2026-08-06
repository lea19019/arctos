#!/usr/bin/env python3
"""
plot_loss.py — Parse a finetune job output log and plot train/eval loss curves.

Usage:
    python3 scripts/plot_loss.py mts-finetune_12500221.out
    python3 scripts/plot_loss.py mts-finetune_12500221.out --out outputs/loss.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_log(path: str):
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    epoch_steps, epoch_losses = [], []

    step_re   = re.compile(r"\[Step\s+(\d+)\].*loss=([\d.]+)")
    eval_re   = re.compile(r"\[Eval\s+step=(\d+)\]\s+eval_mel_loss=([\d.]+)")
    epoch_re  = re.compile(r"\[Epoch\s+(\d+)/(\d+)\]\s+avg_train_loss=([\d.]+)")

    # We need step number for epoch lines — track last step seen
    last_step = 0

    with open(path) as f:
        for line in f:
            m = step_re.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(2))
                train_steps.append(step)
                train_losses.append(loss)
                last_step = step
                continue

            m = eval_re.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(2))
                eval_steps.append(step)
                eval_losses.append(loss)
                continue

            m = epoch_re.search(line)
            if m:
                epoch, total, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
                epoch_steps.append(last_step)
                epoch_losses.append(loss)

    return (train_steps, train_losses,
            eval_steps,  eval_losses,
            epoch_steps, epoch_losses)


def plot(log_path: str, out_path: str):
    (train_steps, train_losses,
     eval_steps,  eval_losses,
     epoch_steps, epoch_losses) = parse_log(log_path)

    if not train_steps:
        print("No training data found in log.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"MMS-TTS Fine-tuning — {Path(log_path).name}", fontsize=13)

    # --- Left: step-level train loss + eval points ---
    ax = axes[0]
    ax.plot(train_steps, train_losses, color="#4C72B0", alpha=0.6,
            linewidth=0.8, label="Train loss (per 50 steps)")
    if eval_steps:
        ax.scatter(eval_steps, eval_losses, color="#DD4444", s=40, zorder=5,
                   label="Eval loss")
        # Highlight best eval
        best_idx = eval_losses.index(min(eval_losses))
        ax.scatter(eval_steps[best_idx], eval_losses[best_idx],
                   color="#FF0000", s=100, marker="*", zorder=6,
                   label=f"Best eval {eval_losses[best_idx]:.4f} @ step {eval_steps[best_idx]}")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Mel L1 loss")
    ax.set_title("Loss by step")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.3)

    # --- Right: epoch average train loss + eval overlay ---
    ax2 = axes[1]
    if epoch_steps:
        epochs = list(range(1, len(epoch_losses) + 1))
        ax2.plot(epochs, epoch_losses, "o-", color="#4C72B0",
                 linewidth=1.5, markersize=5, label="Avg train loss / epoch")
    if eval_steps:
        # Map eval steps to approximate epoch numbers
        max_step = max(train_steps)
        total_epochs = len(epoch_losses) if epoch_losses else 30
        eval_epochs = [s / max_step * total_epochs for s in eval_steps]
        ax2.plot(eval_epochs, eval_losses, "s--", color="#DD4444",
                 linewidth=1.2, markersize=4, alpha=0.8, label="Eval loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mel L1 loss")
    ax2.set_title("Loss by epoch")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Steps logged     : {len(train_steps)}")
    print(f"  Epochs complete  : {len(epoch_losses)}")
    if eval_losses:
        best_idx = eval_losses.index(min(eval_losses))
        print(f"  Best eval loss   : {eval_losses[best_idx]:.4f} @ step {eval_steps[best_idx]}")
    if epoch_losses:
        print(f"  Final train loss : {epoch_losses[-1]:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log", help="Path to SLURM .out file")
    p.add_argument("--out", default=None, help="Output PNG path (default: next to log file)")
    args = p.parse_args()

    out = args.out or str(Path(args.log).with_suffix(".png"))
    plot(args.log, out)
