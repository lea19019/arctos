"""Q5-strong — layer-level perturbation measured by chrF++ quality drop.

The deciding experiment for the Q5 null. The per-head logit-drop version
found no correlation between DLA importance and quantization sensitivity,
but with three weaknesses: noise-floor magnitudes, single-head weight
fraction, and logit (not quality) as the metric. This version fixes all
three:

  1. Perturb WHOLE LAYERS (every q/k/v/o/gate/up/down projection) — the
     weight fraction quantization actually touches.
  2. Measure chrF++ drop on ACTUALLY GENERATED translations vs the clean
     baseline generation — the quantity the method cares about.
  3. Correlate per-layer chrF++ drop against per-layer IFR score AND
     per-layer |DLA| (summed over heads + MLP), from this model's Q1 results.

Headline metrics, per pair:
  - Spearman(per-layer IFR, per-layer chrF++ drop)
  - Spearman(per-layer |DLA|, per-layer chrF++ drop)
  - depth-block test: chrF++ drop when perturbing the first-quarter block
    of layers vs middle-half vs last-quarter (does the depth signature
    predict quality fragility?)

If IFR/DLA importance correlates with chrF++ drop here, the per-head
logit metric was simply too blunt and interpretability-guided allocation
is viable. If still null, importance != sensitivity is a real finding.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.wmt import load_wmt_pairs
from src.eval.metrics import sentence_chrfpp
from src.interp.sensitivity import perturb_layer
from src.models._prompt import build_mt_prompt


def _import_loader(dotted: str):
    mod, _, func = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), func)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _mean_chrf(model, records, pair, refs, max_new_tokens):
    """Generate translations for the records and return mean sentence chrF++ vs refs."""
    scores = []
    for rec, ref in zip(records, refs):
        hyp = model.generate(build_mt_prompt(rec.source, pair), max_new_tokens=max_new_tokens)
        # take the first line as the translation (models often continue chatting)
        hyp = hyp.strip().splitlines()[0] if hyp.strip() else ""
        scores.append(sentence_chrfpp(hyp, ref))
    return float(np.mean(scores)), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--q1-results", type=Path, required=True)
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="perturb every Nth layer to save time (1 = all layers)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = cfg["language_pairs"]

    loader = _import_loader(cfg["model"]["loader"])
    print(f"[q5q] loading {cfg['model']['name']} ...", flush=True)
    t0 = time.time()
    model = loader(dtype=cfg["model"]["dtype"], device=cfg["model"]["device"])
    n_layers = model.cfg.n_layers
    print(f"[q5q] loaded in {time.time()-t0:.1f}s; n_layers={n_layers}", flush=True)

    summary = {"model": cfg["model"]["name"], "sigma": args.sigma,
               "n_examples": args.n_examples, "max_new_tokens": args.max_new_tokens,
               "layer_stride": args.layer_stride, "per_pair": {}}

    for pair in pairs:
        print(f"[q5q] === {pair} ===", flush=True)
        records = list(load_wmt_pairs(pair, n=args.n_examples))
        refs = [r.target for r in records]

        # baseline generation quality
        t0 = time.time()
        base_chrf, _ = _mean_chrf(model, records, pair, refs, args.max_new_tokens)
        print(f"[q5q] {pair}: baseline chrF++ = {base_chrf:.2f} ({time.time()-t0:.0f}s)", flush=True)

        # per-layer perturbation -> chrF++ drop
        layers = list(range(0, n_layers, args.layer_stride))
        drops = {}
        t0 = time.time()
        for li in layers:
            g = torch.Generator(device=model.device).manual_seed(0)
            with perturb_layer(model, li, args.sigma, g):
                chrf, _ = _mean_chrf(model, records, pair, refs, args.max_new_tokens)
            drops[li] = base_chrf - chrf  # positive = quality dropped
        print(f"[q5q] {pair}: {len(layers)} layer perturbations in {time.time()-t0:.0f}s", flush=True)

        # importance signals from Q1 (align to the perturbed layers)
        ifr = np.load(args.q1_results / f"ifr_{pair}.npz")
        dla = np.load(args.q1_results / f"dla_{pair}.npz")
        ifr_layer = ifr["layer_scores"]                          # (L,)
        dla_layer = np.abs(dla["head_scores"]).sum(1) + np.abs(dla["layer_mlp"])  # (L,)

        drop_vec = np.array([drops[li] for li in layers])
        ifr_vec = np.array([ifr_layer[li] for li in layers])
        dla_vec = np.array([dla_layer[li] for li in layers])

        rho_ifr = _spearman(ifr_vec, drop_vec)
        rho_dla = _spearman(dla_vec, drop_vec)

        # depth-block test: mean chrF++ drop by quarter of depth
        q = n_layers // 4
        def block_mean(lo, hi):
            vals = [drops[li] for li in layers if lo <= li < hi]
            return float(np.mean(vals)) if vals else float("nan")
        blocks = {"first_quarter": block_mean(0, q),
                  "middle_half": block_mean(q, n_layers - q),
                  "last_quarter": block_mean(n_layers - q, n_layers)}

        np.savez(args.output / f"layer_chrf_drop_{pair}.npz",
                 layers=np.array(layers), drop=drop_vec, ifr=ifr_vec, dla=dla_vec,
                 baseline_chrf=np.array(base_chrf))
        summary["per_pair"][pair] = {
            "baseline_chrf": base_chrf,
            "spearman_ifr_vs_chrfdrop": rho_ifr,
            "spearman_absdla_vs_chrfdrop": rho_dla,
            "depth_block_mean_drop": blocks,
            "max_drop_layer": int(layers[int(np.argmax(drop_vec))]),
            "max_drop": float(drop_vec.max()),
        }
        print(f"[q5q] {pair}: rho(IFR,drop)={rho_ifr:+.3f} rho(|DLA|,drop)={rho_dla:+.3f} "
              f"blocks={blocks}", flush=True)

    (args.output / "q5_quality_summary.json").write_text(json.dumps(summary, indent=2))
    print("[q5q] wrote q5_quality_summary.json", flush=True)


if __name__ == "__main__":
    main()
