"""Q6 — find / keep / shrink / prune: the phase-two compression sandbox.

Ties together the phase-two ideas on one model and one calibration set:

  FIND   — locate the precision-fragile structure with sensitivity-native
           signals (NOT interpretability importance, which Q5 showed is
           uncorrelated): super weights (activation-spike, data-free), AWQ
           per-channel salience (and whether MT vs raw-text calibration moves
           it), and the MT-conditional Fisher diagonal.
  SHRINK  — RTN INT-k quantize the whole model; chrF++ vs bits (the honest
           base case; what AWQ/GPTQ improve on).
  KEEP    — at the hardest bit-width, compare RTN vs AWQ-scaling vs
           keep-salient-channels-FP16 vs super-weight-preservation; chrF++.
  PRUNE   — magnitude vs Wanda at a couple of sparsities; chrF++. Plus the
           super-weight stress test: ablate the 1 super weight vs ablate the
           N largest-magnitude weights (falsifies magnitude as a saliency).

Everything is measured by chrF++ on actually-generated translations — the
quantity the method cares about — not target-token logit (Q5's weak proxy).

Run:
  python experiments/q6-compression/experiment.py \
      --config experiments/q6-compression/configs/aya.yaml \
      --output results/aya-expanse-8b/q6
Flags subset the work for a fast smoke test (--stages, --n-examples, --bits).
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
from src.models._prompt import build_mt_prompt, tokenize_target_prefix

from src.interp.activation_stats import collect_activation_stats
from src.interp.compress import (
    quantize_linears, prune_linears, ablate_weights, magnitude_mask,
    collect_hessians, gptq_quantize_linears, bits_by_fisher, quantize_mixed_precision,
    parse_spec,
)
from src.interp.super_weights import detect_super_weights, verify_super_weight, _mlp_out_linear
from src.interp.salient_channels import salience_by_regime, compare_salience
from src.interp.hessian_diag import fisher_diagonal


def _import_loader(dotted: str):
    mod, _, func = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), func)


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); rx -= rx.mean()
    ry = np.argsort(np.argsort(y)).astype(float); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _gen_hyps(model, records, pair, max_new_tokens) -> list[str]:
    hyps = []
    for rec in records:
        h = model.generate(build_mt_prompt(rec.source, pair), max_new_tokens=max_new_tokens)
        hyps.append(h.strip().splitlines()[0] if h.strip() else "")
    return hyps


def _mean_chrf(model, records, pair, refs, max_new_tokens) -> float:
    hyps = _gen_hyps(model, records, pair, max_new_tokens)
    return float(np.mean([sentence_chrfpp(h, r) for h, r in zip(hyps, refs)]))


# Global toggles for COMET eval (set from args in main, so the gptq/alloc/calib
# stages can score with XCOMET-XL — the WMT25 compression-task metric).
_USE_COMET = False
_COMET_GPUS = 1


def _eval_q(model, eval_sets, pairs, max_new_tokens) -> dict:
    """Generate once per pair; return {pair: {'chrf':, 'comet':}} (comet if enabled)."""
    out = {}
    for pair in pairs:
        recs, _p, refs = eval_sets[pair]
        hyps = _gen_hyps(model, recs, pair, max_new_tokens)
        chrf = float(np.mean([sentence_chrfpp(h, r) for h, r in zip(hyps, refs)]))
        comet = None
        if _USE_COMET:
            from src.eval.metrics import comet_score
            srcs = [r.source for r in recs]
            comet, _segs = comet_score(srcs, hyps, refs, gpus=_COMET_GPUS)
        out[pair] = {"chrf": chrf, "comet": comet}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-examples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--calib-n", type=int, default=64, help="prompts for FIND signals")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3, 2])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--sparsities", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--stages", type=str, nargs="+",
                    default=["find", "shrink", "keep", "prune"])
    ap.add_argument("--calib-bits", type=int, nargs="+", default=[3],
                    help="bit-widths for the MT-vs-generic calibration head-to-head")
    ap.add_argument("--calib-sparsity", type=float, default=0.5,
                    help="sparsity for the MT-vs-generic Wanda head-to-head")
    ap.add_argument("--keep-frac", type=float, default=0.01,
                    help="fraction of channels kept FP16 in the KEEP stage")
    ap.add_argument("--levels", type=str, nargs="+", default=None,
                    help="SHRINK bit-specs to sweep, e.g. 4 3 2 ternary binary (default: --bits)")
    ap.add_argument("--keep-bits", type=str, nargs="+", default=None,
                    help="KEEP protection bit-specs, e.g. 3 2 ternary binary (default: all bits except largest)")
    ap.add_argument("--awq-alphas", type=float, nargs="+", default=[0.5],
                    help="AWQ scaling exponents to sweep in the KEEP stage")
    ap.add_argument("--gptq-bits", type=int, nargs="+", default=[4, 3],
                    help="bit-widths for the GPTQ MT-vs-generic head-to-head")
    ap.add_argument("--alloc-avg-bits", type=float, default=3.0,
                    help="average bit budget for the Fisher mixed-precision allocator")
    ap.add_argument("--use-comet", action="store_true",
                    help="score gptq/alloc/calib stages with XCOMET-XL (WMT25 metric)")
    ap.add_argument("--comet-gpus", type=int, default=1)
    args = ap.parse_args()
    global _USE_COMET, _COMET_GPUS
    _USE_COMET = args.use_comet
    _COMET_GPUS = args.comet_gpus

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = cfg["language_pairs"]
    mname = cfg["model"]["name"]

    loader = _import_loader(cfg["model"]["loader"])
    print(f"[q6] loading {mname} ...", flush=True)
    t0 = time.time()
    model = loader(dtype=cfg["model"]["dtype"], device=cfg["model"]["device"])
    print(f"[q6] loaded in {time.time()-t0:.1f}s; n_layers={model.cfg.n_layers}", flush=True)

    # Calibration prompts (mixed across pairs) for the FIND signals.
    calib_records = []
    for pair in pairs:
        calib_records += [(r, pair) for r in load_wmt_pairs(pair, n=args.calib_n // len(pairs))]
    mt_prompts = [build_mt_prompt(r.source, p) for r, p in calib_records]
    src_prompts = [r.source for r, _ in calib_records]
    tgt_prompts = [r.target for r, _ in calib_records]

    summary: dict = {"model": mname, "args": vars(args).copy()}
    summary["args"]["config"] = str(args.config)
    summary["args"]["output"] = str(args.output)

    # ---------------- FIND ------------------------------------------------- #
    act_scales: dict = {}
    act_norms: dict = {}
    fisher_per_module: dict = {}
    super_coords: list[tuple[int, int, int]] = []
    if "find" in args.stages:
        print("[q6][find] super weights ...", flush=True)
        # Detect one spike candidate per layer, then RANK BY CAUSAL ABLATION
        # (KL when the single scalar is zeroed) — raw spike magnitude alone is
        # fooled by the last layer, whose down_proj writes straight to the
        # final residual. The Apple super weight is the one whose ablation
        # actually moves the next-token distribution.
        sw = detect_super_weights(model, mt_prompts[:8], top_k=model.cfg.n_layers)
        ranked = []
        for c in sw.candidates:
            v = verify_super_weight(model, mt_prompts[:8], c)
            ranked.append((c, v))
        ranked.sort(key=lambda cv: cv[1].get("mean_kl_clean_vs_ablated", 0.0), reverse=True)
        sw.candidates = [c for c, _ in ranked]
        sw_verif = ranked[0][1] if ranked else {}
        super_coords = [(c.layer, c.out_dim, c.in_dim) for c in sw.candidates[:5]]
        summary["find_super_weights"] = {
            "candidates": [{**vars(c), "ablation_kl": v["mean_kl_clean_vs_ablated"],
                            "ablation_top1_drop": v["mean_top1_prob_drop"]}
                           for c, v in ranked[:10]],
            "massive_resid_top": sw.massive_resid[:5],
            "top_candidate_ablation": sw_verif,
        }
        print(f"[q6][find] top super weight (by causal KL): {sw.candidates[0] if sw.candidates else None}; "
              f"ablation={sw_verif}", flush=True)

        print("[q6][find] AWQ salience by calibration regime ...", flush=True)
        sal = salience_by_regime(model, {"mt": mt_prompts, "source": src_prompts, "target": tgt_prompts})
        cmp_ms = compare_salience(sal, "mt", "source", top_frac=0.01)
        cmp_mt = compare_salience(sal, "mt", "target", top_frac=0.01)
        summary["find_awq_calibration_shift"] = {
            "mt_vs_source": {"mean_top1pct_jaccard": cmp_ms.mean_jaccard,
                             "mean_spearman": cmp_ms.mean_spearman, "n_modules": cmp_ms.n_modules},
            "mt_vs_target": {"mean_top1pct_jaccard": cmp_mt.mean_jaccard,
                             "mean_spearman": cmp_mt.mean_spearman, "n_modules": cmp_mt.n_modules},
        }
        print(f"[q6][find] AWQ salient-set MT vs source: Jaccard(top1%)="
              f"{cmp_ms.mean_jaccard:.3f} Spearman={cmp_ms.mean_spearman:.3f}", flush=True)
        # reuse MT salience as the act_scales for KEEP/SHRINK-AWQ; act_norms for Wanda.
        act_scales = sal["mt"]
        stats = collect_activation_stats(model, mt_prompts)
        act_norms = {n: st.q99_abs for n, st in stats.stats_by_module.items()}  # robust proxy for ||X||

        print("[q6][find] MT-conditional Fisher diagonal ...", flush=True)
        fex = []
        for r, p in calib_records[:min(32, len(calib_records))]:
            toks = model.to_tokens(build_mt_prompt(r.source, p))
            tgt = tokenize_target_prefix(model, r.target, max_tokens=4)
            if tgt:
                fex.append((toks, tgt))
        fisher = fisher_diagonal(model, fex)
        fisher_per_module = dict(fisher.per_module_mean)
        summary["find_fisher"] = {
            "layer_fisher": fisher.layer_fisher,
            "top_modules_by_mean": sorted(fisher.per_module_mean.items(),
                                          key=lambda kv: kv[1], reverse=True)[:10],
        }
        np.savez(args.output / "fisher.npz",
                 layer_fisher=np.array(fisher.layer_fisher))
        # zero grads to free memory before the generation-heavy stages
        model.hf_model.zero_grad(set_to_none=True)
        for p_ in model.parameters():
            p_.requires_grad_(False)

    # Baselines per pair (clean generation quality).
    base: dict[str, float] = {}
    eval_sets: dict[str, tuple] = {}
    if any(s in args.stages for s in ("shrink", "keep", "prune", "calib", "gptq", "alloc")):
        for pair in pairs:
            recs = list(load_wmt_pairs(pair, n=args.n_examples))
            refs = [r.target for r in recs]
            eval_sets[pair] = (recs, pair, refs)
            base[pair] = _mean_chrf(model, recs, pair, refs, args.max_new_tokens)
            print(f"[q6] baseline chrF++ {pair} = {base[pair]:.2f}", flush=True)
        summary["baseline_chrf"] = base

    # ---------------- SHRINK ---------------------------------------------- #
    if "shrink" in args.stages:
        summary["shrink"] = {}
        levels = args.levels or [str(b) for b in args.bits]
        for lvl in levels:
            with quantize_linears(model, lvl, group_size=args.group_size):
                row = _eval_q(model, eval_sets, pairs, args.max_new_tokens)
            summary["shrink"][f"rtn_w{lvl}"] = row
            print(f"[q6][shrink] RTN W{lvl}: { {p: row[p] for p in pairs} }", flush=True)

    # ---------------- KEEP ------------------------------------------------- #
    if "keep" in args.stages:
        # Evaluate protection at the *cliff* bit-widths, not just dead 2-bit:
        # by default every requested bit except the largest (lossless) one, so
        # e.g. bits=[4,3,2] -> keep tested at W3 (where there is signal) and W2.
        keep_bits = args.keep_bits or sorted(b for b in set(args.bits) if b != max(args.bits))
        summary["keep"] = {"bits": keep_bits}
        # keep top-keep_frac salient input channels FP16 per module
        keep_cols = {}
        for name, sc in act_scales.items():
            k = max(1, int(round(args.keep_frac * sc.numel())))
            keep_cols[name] = torch.topk(sc, k).indices.tolist()
        for b in keep_bits:
            summary["keep"][f"w{b}"] = {}
            variants = {
                "rtn": dict(),
                "keep_salient_fp16": dict(keep_cols_by_module=keep_cols),
            }
            # AWQ scaling only applies to integer bit-widths (>=2); skip for
            # ternary/binary where there's no integer grid to scale into.
            if parse_spec(b)[0] == "int":
                for a in args.awq_alphas:
                    variants[f"awq_a{a}"] = dict(act_scales=act_scales, awq_alpha=a)
            for vname, kw in variants.items():
                with quantize_linears(model, b, group_size=args.group_size, **kw):
                    row = _eval_q(model, eval_sets, pairs, args.max_new_tokens)
                summary["keep"][f"w{b}"][vname] = row
                print(f"[q6][keep] W{b} {vname}: { {p: row[p] for p in pairs} }", flush=True)
            # super-weight preservation: RTN everything but restore the super weights
            if super_coords:
                with quantize_linears(model, b, group_size=args.group_size):
                    blocks = model.arch.get_blocks(model.hf_model)
                    for (li, o, j), cand in zip(super_coords, summary.get("find_super_weights", {}).get("candidates", [])):
                        _mlp_out_linear(model, blocks[li]).weight[o, j] = cand["weight_value"]
                    row = _eval_q(model, eval_sets, pairs, args.max_new_tokens)
                summary["keep"][f"w{b}"]["rtn_plus_superweight_fp16"] = row
                print(f"[q6][keep] W{b} RTN+superweight: { {p: row[p] for p in pairs} }", flush=True)

    # ---------------- PRUNE ------------------------------------------------ #
    if "prune" in args.stages:
        summary["prune"] = {}
        for sp in args.sparsities:
            with prune_linears(model, sp, method="magnitude"):
                rmag = {pair: _mean_chrf(model, *eval_sets[pair], args.max_new_tokens) for pair in pairs}
            with prune_linears(model, sp, method="wanda", act_norms=act_norms):
                rwan = {pair: _mean_chrf(model, *eval_sets[pair], args.max_new_tokens) for pair in pairs}
            summary["prune"][f"sparsity_{sp}"] = {"magnitude": rmag, "wanda": rwan}
            print(f"[q6][prune] sp={sp} magnitude={rmag} wanda={rwan}", flush=True)

        # super-weight stress test: ablate 1 super weight vs N largest-|W| weights
        if super_coords:
            sw_one = super_coords[:1]
            with ablate_weights(model, sw_one):
                r_one = {pair: _mean_chrf(model, *eval_sets[pair], args.max_new_tokens) for pair in pairs}
            blocks = model.arch.get_blocks(model.hf_model)
            li = sw_one[0][0]
            W = _mlp_out_linear(model, blocks[li]).weight.detach().float()
            flat = W.abs().flatten()
            N = 1000
            big = torch.topk(flat, N).indices
            coords_big = [(li, int(i // W.shape[1]), int(i % W.shape[1])) for i in big.tolist()
                          if (li, int(i // W.shape[1]), int(i % W.shape[1])) != sw_one[0]]
            with ablate_weights(model, coords_big):
                r_big = {pair: _mean_chrf(model, *eval_sets[pair], args.max_new_tokens) for pair in pairs}
            summary["prune"]["superweight_stress"] = {
                "ablate_1_superweight": r_one,
                f"ablate_{N}_largest_magnitude": r_big,
            }
            print(f"[q6][prune] stress: 1 super={r_one}  {N} biggest={r_big}", flush=True)

    # ---------------- CALIB (the linchpin: MT vs generic calibration) ----- #
    if "calib" in args.stages:
        from src.data.generic import generic_calib_for_pairs
        summary["calib"] = {}
        # MT-calibration salience/act-norms (reuse find's if present, else compute)
        if not act_scales:
            st = collect_activation_stats(model, mt_prompts)
            act_scales = {n: s.q99_abs for n, s in st.stats_by_module.items()}
            act_norms = act_scales
        # generic calibration of the SAME size, same languages, generic domain
        gen_prompts = generic_calib_for_pairs(pairs, args.calib_n)
        print(f"[q6][calib] generic calib: {len(gen_prompts)} XNLI sentences", flush=True)
        gst = collect_activation_stats(model, gen_prompts)
        gen_scales = {n: s.q99_abs for n, s in gst.stats_by_module.items()}

        # how different are the salient sets? (set-shift, the find-stage metric)
        cmp = compare_salience({"mt": act_scales, "generic": gen_scales}, "mt", "generic", top_frac=0.01)
        summary["calib"]["salient_set_shift_mt_vs_generic"] = {
            "mean_top1pct_jaccard": cmp.mean_jaccard, "mean_spearman": cmp.mean_spearman}

        # AWQ head-to-head: same bits, MT-calibrated scales vs generic-calibrated.
        for b in args.calib_bits:
            with quantize_linears(model, b, group_size=args.group_size):
                rtn = {p: _mean_chrf(model, *eval_sets[p], args.max_new_tokens) for p in pairs}
            with quantize_linears(model, b, group_size=args.group_size, act_scales=act_scales, awq_alpha=0.5):
                mt = {p: _mean_chrf(model, *eval_sets[p], args.max_new_tokens) for p in pairs}
            with quantize_linears(model, b, group_size=args.group_size, act_scales=gen_scales, awq_alpha=0.5):
                gen = {p: _mean_chrf(model, *eval_sets[p], args.max_new_tokens) for p in pairs}
            summary["calib"][f"awq_w{b}"] = {
                "rtn": rtn, "mt_calib": mt, "generic_calib": gen,
                "mt_minus_generic": {p: mt[p] - gen[p] for p in pairs}}
            print(f"[q6][calib] AWQ W{b}: rtn={rtn} mt={mt} generic={gen} "
                  f"Δ(mt-gen)={ {p: round(mt[p]-gen[p],2) for p in pairs} }", flush=True)

        # Wanda head-to-head: same sparsity, MT vs generic activation norms.
        sp = args.calib_sparsity
        with prune_linears(model, sp, method="wanda", act_norms=act_norms):
            wmt = {p: _mean_chrf(model, *eval_sets[p], args.max_new_tokens) for p in pairs}
        with prune_linears(model, sp, method="wanda", act_norms=gen_scales):
            wgen = {p: _mean_chrf(model, *eval_sets[p], args.max_new_tokens) for p in pairs}
        summary["calib"][f"wanda_sp{sp}"] = {
            "mt_calib": wmt, "generic_calib": wgen,
            "mt_minus_generic": {p: wmt[p] - wgen[p] for p in pairs}}
        print(f"[q6][calib] Wanda sp={sp}: mt={wmt} generic={wgen} "
              f"Δ(mt-gen)={ {p: round(wmt[p]-wgen[p],2) for p in pairs} }", flush=True)

    # ---------------- GPTQ (the real calibration test: MT vs generic) ----- #
    if "gptq" in args.stages:
        from src.data.generic import generic_calib_for_pairs
        summary["gptq"] = {}
        gen_prompts = generic_calib_for_pairs(pairs, args.calib_n)
        print(f"[q6][gptq] collecting Hessians (MT, then generic) ...", flush=True)
        # RTN reference (calibration-free), scored once per bit.
        for b in args.gptq_bits:
            with quantize_linears(model, b, group_size=args.group_size):
                rtn = _eval_q(model, eval_sets, pairs, args.max_new_tokens)
            summary["gptq"].setdefault(f"w{b}", {})["rtn"] = rtn
            print(f"[q6][gptq] W{b} rtn={ {p: rtn[p] for p in pairs} }", flush=True)
        # GPTQ with MT vs generic Hessians (collected sequentially to bound RAM).
        for label, cprompts in (("gptq_mt", mt_prompts), ("gptq_generic", gen_prompts)):
            H = collect_hessians(model, cprompts)
            for b in args.gptq_bits:
                with gptq_quantize_linears(model, b, H, group_size=args.group_size):
                    q = _eval_q(model, eval_sets, pairs, args.max_new_tokens)
                summary["gptq"][f"w{b}"][label] = q
                print(f"[q6][gptq] W{b} {label}={ {p: q[p] for p in pairs} }", flush=True)
            del H
        # MT-minus-generic deltas (chrf and comet)
        for b in args.gptq_bits:
            blk = summary["gptq"][f"w{b}"]
            if "gptq_mt" in blk and "gptq_generic" in blk:
                blk["mt_minus_generic"] = {
                    p: {m: (blk["gptq_mt"][p][m] - blk["gptq_generic"][p][m])
                        for m in ("chrf", "comet") if blk["gptq_mt"][p][m] is not None}
                    for p in pairs}
                print(f"[q6][gptq] W{b} Δ(mt-gen)={blk['mt_minus_generic']}", flush=True)

    # ---------------- ALLOC (Fisher-driven mixed precision) --------------- #
    if "alloc" in args.stages:
        summary["alloc"] = {}
        if not fisher_per_module:
            fex = []
            for r, p in calib_records[:min(32, len(calib_records))]:
                toks = model.to_tokens(build_mt_prompt(r.source, p))
                tgt = tokenize_target_prefix(model, r.target, max_tokens=4)
                if tgt:
                    fex.append((toks, tgt))
            for pp in model.parameters():
                pp.requires_grad_(False)
            fisher_per_module = dict(fisher_diagonal(model, fex).per_module_mean)
            for pp in model.parameters():
                pp.requires_grad_(False)
        avg = args.alloc_avg_bits
        bbm = bits_by_fisher(fisher_per_module, avg_bits=avg, bit_choices=(2, 4))
        with quantize_mixed_precision(model, bbm, group_size=args.group_size):
            mixed = _eval_q(model, eval_sets, pairs, args.max_new_tokens)
        with quantize_linears(model, int(round(avg)), group_size=args.group_size):
            uniform = _eval_q(model, eval_sets, pairs, args.max_new_tokens)
        n_hi = sum(1 for v in bbm.values() if v == max(bbm.values()))
        summary["alloc"] = {
            "avg_bits": avg, "n_modules": len(bbm), "n_high_bit": n_hi,
            "fisher_mixed": mixed, "uniform": uniform,
            "mixed_minus_uniform": {
                p: {m: (mixed[p][m] - uniform[p][m])
                    for m in ("chrf", "comet") if mixed[p][m] is not None} for p in pairs}}
        print(f"[q6][alloc] avg={avg}b mixed_minus_uniform={summary['alloc']['mixed_minus_uniform']}", flush=True)

    (args.output / "q6_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[q6] wrote {args.output/'q6_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
