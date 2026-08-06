"""Render attention patterns for a model's top-DLA heads (Q2 qualitative).

Picks the top-|DLA| heads from the model's Q1 results, runs one MT prompt
per language pair with output_attentions, and saves a heatmap per head plus
a heuristic role label. Answers "what do the MT-critical heads actually do?"

Run:
    python experiments/q2-attention-heads/attention_viz.py \
        --config experiments/q1-language-emergence/configs/aya.yaml \
        --q1-results results/aya-expanse-8b/q1 \
        --output results/aya-expanse-8b/q2/attention
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.wmt import load_wmt_pairs
from src.interp.attention_viz import attention_patterns, classify_pattern
from src.models._prompt import build_mt_prompt


def _import_loader(dotted: str):
    mod, _, func = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), func)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--q1-results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    loader = _import_loader(cfg["model"]["loader"])
    model = loader(dtype=cfg["model"]["dtype"], device=cfg["model"]["device"])
    # force eager attention so output_attentions works
    model.hf_model.config._attn_implementation = "eager"

    roles = {}
    for pair in cfg["language_pairs"]:
        dla = np.load(args.q1_results / f"dla_{pair}.npz")["head_scores"]  # (L,H)
        H = dla.shape[1]
        top = np.argsort(np.abs(dla).flatten())[::-1][: args.top_k]
        rec = next(iter(load_wmt_pairs(pair, n=1)))
        prompt = build_mt_prompt(rec.source, pair)
        attn, toks = attention_patterns(model.hf_model, model.tokenizer, prompt)
        for idx in top:
            L, h = int(idx // H), int(idx % H)
            if L >= attn.shape[0] or h >= attn.shape[1]:
                continue
            pat = attn[L, h]  # (T, T)
            role = classify_pattern(pat)
            roles[f"{pair}:L{L}.H{h}"] = {"role": role, "dla": float(dla[L, h])}
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(pat.numpy(), cmap="viridis", aspect="auto")
            ax.set_title(f"{cfg['model']['name']} {pair} L{L}.H{h}\nDLA={dla[L,h]:+.2f} role={role}",
                         fontsize=9)
            ax.set_xlabel("key position"); ax.set_ylabel("query position")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(args.output / f"attn_{pair}_L{L}_H{h}.png", dpi=130)
            plt.close(fig)
    (args.output / "head_roles.json").write_text(json.dumps(roles, indent=2))
    print(f"[q2] wrote {len(roles)} attention heatmaps + head_roles.json to {args.output}")
    for k, v in roles.items():
        print(f"  {k}: {v['role']}  (DLA={v['dla']:+.2f})")


if __name__ == "__main__":
    main()
