"""Run the language-pivot trajectory analysis for one model.

Produces per-layer script-mass shares for the cross-script pairs (en-zh,
en-arz) — the 'does the model think in English then convert to target'
trajectory — and saves arrays + a chart.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.wmt import load_wmt_pairs
from src.interp.language_pivot import TARGET_SCRIPT, _precompute_token_scripts, pivot_trajectory
from src.models._prompt import build_mt_prompt


def _import_loader(dotted: str):
    mod, _, func = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), func)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-examples", type=int, default=100)
    ap.add_argument("--pairs", nargs="+", default=["en-zh", "en-arz"])
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    loader = _import_loader(cfg["model"]["loader"])
    name = cfg["model"]["name"]
    print(f"[pivot] loading {name} ...", flush=True)
    t0 = time.time()
    model = loader(dtype=cfg["model"]["dtype"], device=cfg["model"]["device"])
    print(f"[pivot] loaded in {time.time()-t0:.1f}s; building vocab script table ...", flush=True)
    token_scripts = _precompute_token_scripts(model)  # reuse across pairs
    print(f"[pivot] vocab script table built ({len(token_scripts)} tokens)", flush=True)

    summary = {"model": name, "per_pair": {}}
    for pair in args.pairs:
        records = list(load_wmt_pairs(pair, n=args.n_examples))
        prompts = [build_mt_prompt(r.source, pair) for r in records]
        t0 = time.time()
        traj = pivot_trajectory(model, prompts, pair, top_k=50, token_scripts=token_scripts)
        L = len(traj.target_share)
        print(f"[pivot] {pair}: {time.time()-t0:.0f}s  "
              f"target-script mass L0={traj.target_share[0]:.2f} "
              f"mid={traj.target_share[L//2]:.2f} last={traj.target_share[-1]:.2f}", flush=True)
        np.savez(args.output / f"pivot_{pair}.npz",
                 target=traj.target_share.numpy(), latin=traj.latin_share.numpy(),
                 other=traj.other_share.numpy())

        # chart: stacked trajectory
        xs = np.linspace(0, 1, L)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, traj.latin_share.numpy(), "-o", ms=3, label="Latin (English/source pivot)", color="#1f77b4")
        ax.plot(xs, traj.target_share.numpy(), "-o", ms=3,
                label=f"{TARGET_SCRIPT[pair]} (target script)", color="#d62728")
        ax.plot(xs, traj.other_share.numpy(), "-o", ms=3, label="other", color="#999999", alpha=0.6)
        ax.set_xlabel("depth fraction"); ax.set_ylabel("share of top-50 lens prob mass")
        ax.set_title(f"{name} — language-pivot trajectory ({pair})")
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.output / f"pivot_{pair}.png", dpi=140); plt.close(fig)

        # crossover layer: where target script overtakes latin
        tshare, lshare = traj.target_share.numpy(), traj.latin_share.numpy()
        cross = next((i for i in range(L) if tshare[i] > lshare[i]), None)
        summary["per_pair"][pair] = {
            "n_examples": traj.n_examples,
            "target_share_first": float(tshare[0]), "target_share_mid": float(tshare[L//2]),
            "target_share_last": float(tshare[-1]),
            "latin_share_mid": float(lshare[L//2]),
            "crossover_layer": cross, "crossover_depth_frac": (cross / L) if cross else None,
            "n_layers": L,
        }
    (args.output / "pivot_summary.json").write_text(json.dumps(summary, indent=2))
    print("[pivot] wrote pivot_summary.json", flush=True)


if __name__ == "__main__":
    main()
