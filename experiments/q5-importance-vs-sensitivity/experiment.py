"""Q5 runner — importance vs quantization sensitivity (the bridge to phase two).

The load-bearing experiment for the whole quantization story. It tests
the hypothesis:

    Components that interpretability flags as MT-critical (high |DLA|,
    high IFR) are MORE sensitive to quantization noise than components
    flagged as unimportant.

If true, an interpretability-guided bit-budget (protect critical
components, aggressively quantize the rest) is principled. If false,
interpretability importance and quantization sensitivity are different
things — which is itself a publishable finding (and the prior paper's
explicit warning).

Pipeline, per model:
  1. Load the model's own Q1 DLA results to rank heads by |DLA|.
  2. Form three head groups: top-K (predicted critical), bottom-K
     (predicted unimportant), and K random heads (control).
  3. Noise-sensitivity sweep on each head (src.interp.sensitivity):
     add Gaussian weight noise at increasing sigma, measure the drop
     in gold-target-token logit.
  4. Attribution patching (src.interp.attribution_patching) for a
     causal cross-check on a clean/corrupt pair set.
  5. AWQ-style activation magnitude stats (src.interp.activation_stats).
  6. Correlation: does |DLA| rank predict noise sensitivity? Report
     Spearman ρ between per-head |DLA| and per-head logit-drop-at-σ.

Outputs to --output:
  sensitivity_{pair}.json   per-group per-head sensitivity curves
  attribution_{pair}.npz    per-(layer,head) + per-mlp causal effects
  awq_stats_{pair}.npz      per-channel activation magnitude percentiles
  q5_summary.json           the headline correlation + group means
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

from src.data.clean_corrupt import CleanCorruptGenerator, CorruptStrategy
from src.data.wmt import load_wmt_pairs
from src.interp.activation_stats import collect_activation_stats
from src.interp.attribution_patching import attribution_patch
from src.interp.sensitivity import sensitivity_sweep
from src.models._prompt import build_mt_prompt, tokenize_target_prefix


def _import_loader(dotted: str):
    mod, _, func = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), func)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy (rank then Pearson)."""
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _select_head_groups(dla_head_scores: np.ndarray, k: int, seed: int):
    """Return (top, bottom, random) lists of (layer, head) by |DLA|."""
    L, H = dla_head_scores.shape
    absflat = np.abs(dla_head_scores).flatten()
    order = np.argsort(absflat)[::-1]  # high |DLA| first
    top = [(int(i // H), int(i % H)) for i in order[:k]]
    bottom = [(int(i // H), int(i % H)) for i in order[-k:]]
    rng = np.random.default_rng(seed)
    rand_idx = rng.choice(L * H, size=k, replace=False)
    random_heads = [(int(i // H), int(i % H)) for i in rand_idx]
    return top, bottom, random_heads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--q1-results", type=Path, required=True,
                    help="dir with this model's Q1 dla_{pair}.npz (to rank heads)")
    ap.add_argument("--k-heads", type=int, default=12, help="heads per group")
    ap.add_argument("--n-examples", type=int, default=50,
                    help="calibration examples for sensitivity (kept small — it's a sweep)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = cfg["language_pairs"]
    sigmas = (0.01, 0.05, 0.1, 0.2, 0.5)

    loader = _import_loader(cfg["model"]["loader"])
    print(f"[q5] loading {cfg['model']['name']} ...", flush=True)
    t0 = time.time()
    model = loader(dtype=cfg["model"]["dtype"], device=cfg["model"]["device"])
    print(f"[q5] loaded in {time.time()-t0:.1f}s; n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads}",
          flush=True)

    summary = {"model": cfg["model"]["name"], "k_heads": args.k_heads,
               "n_examples": args.n_examples, "sigmas": list(sigmas), "per_pair": {}}

    for pair in pairs:
        print(f"[q5] === {pair} ===", flush=True)
        records = list(load_wmt_pairs(pair, n=args.n_examples))
        examples = []
        for r in records:
            ids = tokenize_target_prefix(model, r.target, max_tokens=8)
            if ids:
                examples.append((build_mt_prompt(r.source, pair), ids))

        # 1. rank heads by |DLA| from this model's Q1 results
        dla = np.load(args.q1_results / f"dla_{pair}.npz")
        head_scores = dla["head_scores"]  # (L, H), signed
        top, bottom, rand = _select_head_groups(head_scores, args.k_heads, seed=0)

        # 2. noise sensitivity per group
        t0 = time.time()
        groups = {"top_dla": top, "bottom_dla": bottom, "random": rand}
        group_results = {}
        for gname, heads in groups.items():
            res = sensitivity_sweep(model, examples, heads=heads, sigmas=sigmas)
            group_results[gname] = [
                {"component": r.component, "sigmas": r.sigmas, "logit_drops": r.logit_drops}
                for r in res
            ]
        print(f"[q5] {pair}: sensitivity sweep ({3*args.k_heads} heads) in {time.time()-t0:.1f}s",
              flush=True)
        (args.output / f"sensitivity_{pair}.json").write_text(json.dumps(group_results, indent=2))

        # 3. correlation: per-head |DLA| vs noise-drop at the largest sigma
        #    (over the union of all three groups, so the range is wide)
        all_heads = top + bottom + rand
        dla_abs = np.array([abs(float(head_scores[l, h])) for (l, h) in all_heads])
        drop_at_max = []
        for gname, heads in groups.items():
            for r in group_results[gname]:
                drop_at_max.append(r["logit_drops"][-1])  # sigma = 0.5
        drop_at_max = np.array(drop_at_max[: len(dla_abs)])
        rho = _spearman(dla_abs, drop_at_max)

        # group-mean sensitivity at max sigma
        def _gmean(g):
            return float(np.mean([r["logit_drops"][-1] for r in group_results[g]]))
        group_means = {g: _gmean(g) for g in groups}

        # 4. attribution patching (causal cross-check) on a clean/corrupt set
        t0 = time.time()
        gen = CleanCorruptGenerator(records, seed=0)
        cc_pairs = []
        for r in records[:args.n_examples]:
            try:
                cc = gen.make_pair(r, strategy=CorruptStrategy.LEXICAL_SUB)
            except ValueError:
                continue
            ids = tokenize_target_prefix(model, r.target, max_tokens=8)
            if ids:
                cc_pairs.append((build_mt_prompt(cc.clean.source, pair),
                                 build_mt_prompt(cc.corrupt.source, pair), ids))
        try:
            attr = attribution_patch(model, cc_pairs)
            np.savez(args.output / f"attribution_{pair}.npz",
                     head_effects=attr.head_effects.numpy(),
                     mlp_effects=attr.mlp_effects.numpy(),
                     n_examples=np.array(attr.n_examples))
            attr_ok = int(attr.n_examples)
        except ValueError as e:
            print(f"[q5] {pair}: attribution patching skipped ({e})", flush=True)
            attr_ok = 0
        print(f"[q5] {pair}: attribution patching in {time.time()-t0:.1f}s (n={attr_ok})", flush=True)

        # 5. AWQ activation stats
        t0 = time.time()
        prompts = [e[0] for e in examples]
        stats = collect_activation_stats(model, prompts)
        # save compactly: per-module q99 vector
        np.savez(
            args.output / f"awq_stats_{pair}.npz",
            **{name.replace(".", "_"): s.q99_abs.numpy()
               for name, s in stats.stats_by_module.items()},
        )
        print(f"[q5] {pair}: AWQ stats ({len(stats.stats_by_module)} modules) in {time.time()-t0:.1f}s",
              flush=True)

        summary["per_pair"][pair] = {
            "spearman_absdla_vs_noisedrop": rho,
            "group_mean_logit_drop_at_max_sigma": group_means,
            "top_vs_random_ratio": (group_means["top_dla"] / group_means["random"]
                                    if group_means["random"] else None),
            "attribution_n_examples": attr_ok,
        }
        print(f"[q5] {pair}: Spearman(|DLA|, noise-drop) = {rho:.3f}; "
              f"group means {group_means}", flush=True)

    (args.output / "q5_summary.json").write_text(json.dumps(summary, indent=2))
    print("[q5] wrote q5_summary.json", flush=True)


if __name__ == "__main__":
    main()
