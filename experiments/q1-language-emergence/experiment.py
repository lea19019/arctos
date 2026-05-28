"""Q1 experiment runner — language emergence on a single model.

For each language pair in the config:
  - load MT calibration records from data/{pair}.jsonl
  - build MT prompts (Aya chat template via src.models.aya.build_mt_prompt)
  - run logit lens with gold-target-prefix tracking
  - run IFR over the prompt set
And once across the union of all pairs:
  - run target-language probing across layers

Outputs land in `output/`:
  output/lens_{pair}.npz         per-example per-layer target mass + max
  output/ifr_{pair}.npz          layer / head / mlp / embed importance
  output/probing_target_id.json  per-layer probe accuracy + selectivity
  output/summary.json            top-level numbers + meta
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

from src.data.wmt import LanguagePair, load_wmt_pairs
from src.interp.ifr import ifr
from src.interp.logit_lens import logit_lens
from src.interp.probing import probe_layers
from src.models.aya import build_mt_prompt, tokenize_target_prefix

PAIR_TO_TARGET_CLASS: dict[str, int] = {"cs-de": 0, "en-zh": 1, "en-arz": 2}


def _import_loader(dotted: str):
    """Import a `pkg.mod.func` string and return func."""
    mod, _, func = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), func)


def _run_lens_for_pair(
    model, pair: str, records: list, target_prefix_len: int
) -> dict:
    """Per-example logit lens; returns dict of stacked arrays."""
    per_layer_mass = []  # (N, L, K)
    per_example_max = []  # (N, L) — max mass across the K target tokens at each layer
    for rec in records:
        prompt = build_mt_prompt(rec.source, pair)
        target_ids = tokenize_target_prefix(model, rec.target, max_tokens=target_prefix_len)
        if not target_ids:
            continue
        result = logit_lens(model, prompt, target_tokens=target_ids)
        # Pad K to the requested length so arrays stack cleanly.
        mass = result.target_token_mass.float()  # (L, k)
        if mass.shape[-1] < target_prefix_len:
            pad = torch.zeros(mass.shape[0], target_prefix_len - mass.shape[-1])
            mass = torch.cat([mass, pad], dim=-1)
        per_layer_mass.append(mass.numpy())
        per_example_max.append(mass.max(dim=-1).values.numpy())
    return {
        "target_mass": np.stack(per_layer_mass, axis=0) if per_layer_mass else np.empty((0,)),
        "target_mass_max": np.stack(per_example_max, axis=0) if per_example_max else np.empty((0,)),
    }


def _run_ifr_for_pair(model, pair: str, records: list) -> dict:
    prompts = [build_mt_prompt(r.source, pair) for r in records]
    scores = ifr(model, prompts, target_position="last")
    return {
        "layer_scores": scores.layer_scores.numpy(),
        "head_scores": scores.head_scores.numpy(),
        "mlp_scores": scores.mlp_scores.numpy(),
        "embed_score": np.array(scores.embed_score),
        "n_examples": np.array(scores.n_examples),
    }


def _run_target_probing(model, all_examples: list[tuple[str, int]], n_classes: int) -> list[dict]:
    results = probe_layers(model, all_examples, n_classes=n_classes)
    return [
        {
            "layer": r.layer,
            "accuracy": r.accuracy,
            "control_accuracy": r.control_accuracy,
            "selectivity": r.selectivity,
        }
        for r in results
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-examples", type=int, default=None,
                    help="override config calibration.n_examples")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[q1] config: {args.config}", flush=True)
    print(f"[q1] output: {args.output}", flush=True)

    loader = _import_loader(cfg["model"]["loader"])
    print(f"[q1] loading model via {cfg['model']['loader']} ...", flush=True)
    t0 = time.time()
    model = loader(dtype=cfg["model"]["dtype"], device=cfg["model"]["device"])
    print(f"[q1] model loaded in {time.time() - t0:.1f}s; "
          f"n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads}", flush=True)

    n_examples = args.n_examples or cfg["calibration"]["n_examples"]
    target_prefix_len = cfg["logit_lens"].get("target_prefix_tokens", 8)
    pairs: list[LanguagePair] = cfg["language_pairs"]

    summary: dict = {
        "model": cfg["model"]["name"],
        "pairs": pairs,
        "n_examples_per_pair": n_examples,
        "target_prefix_tokens": target_prefix_len,
        "per_pair": {},
    }

    all_probe_examples: list[tuple[str, int]] = []
    for pair in pairs:
        print(f"[q1] === pair {pair} ===", flush=True)
        records = list(load_wmt_pairs(pair, n=n_examples))
        print(f"[q1] {pair}: loaded {len(records)} records", flush=True)

        t0 = time.time()
        lens = _run_lens_for_pair(model, pair, records, target_prefix_len)
        print(f"[q1] {pair}: lens in {time.time() - t0:.1f}s, shape={lens['target_mass'].shape}",
              flush=True)
        np.savez(args.output / f"lens_{pair}.npz", **lens)

        t0 = time.time()
        ifr_out = _run_ifr_for_pair(model, pair, records)
        print(f"[q1] {pair}: IFR in {time.time() - t0:.1f}s, n_examples={int(ifr_out['n_examples'])}",
              flush=True)
        np.savez(args.output / f"ifr_{pair}.npz", **ifr_out)

        cls = PAIR_TO_TARGET_CLASS[pair]
        for rec in records:
            all_probe_examples.append((build_mt_prompt(rec.source, pair), cls))

        summary["per_pair"][pair] = {
            "n_records": len(records),
            "lens_target_mass_shape": list(lens["target_mass"].shape),
            "ifr_layer_top": int(np.argsort(ifr_out["layer_scores"])[::-1].tolist()[0])
                if ifr_out["layer_scores"].size else None,
        }

    print(f"[q1] === target-language probing across {len(all_probe_examples)} examples ===",
          flush=True)
    t0 = time.time()
    probe_results = _run_target_probing(model, all_probe_examples, n_classes=len(pairs))
    print(f"[q1] probing in {time.time() - t0:.1f}s; "
          f"max selectivity = {max(r['selectivity'] for r in probe_results):.3f}", flush=True)
    (args.output / "probing_target_id.json").write_text(json.dumps(probe_results, indent=2))

    summary["probing_target_id"] = {
        "max_selectivity": max(r["selectivity"] for r in probe_results),
        "max_selectivity_layer": max(probe_results, key=lambda r: r["selectivity"])["layer"],
        "n_classes": len(pairs),
        "n_examples": len(all_probe_examples),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[q1] wrote summary.json", flush=True)


if __name__ == "__main__":
    main()
