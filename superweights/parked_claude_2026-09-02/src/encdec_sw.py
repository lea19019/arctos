"""Super-weight detection + causal ablation for encoder-decoder models (NLLB).

RQ2 asks whether super weights exist beyond decoder-only LLMs. NLLB-200 is
M2M100-style: two stacks (encoder, decoder), FFN output projection `fc2`
(weight shape (d_model, d_ff), same [j, k] convention as `down_proj`). The
detector and the ablation unit are the same as for decoder-only models —
v5's residual-stream persistence rule (`detect_sw.find_super_activations`),
the both-outliers decomposition (`detect_sw.decompose`), individual + joint
zeroing, a magnitude-matched null, and direct super-activation removal —
applied per stack. Only the damage metric differs: teacher-forced
translation loss on FLORES+ devtest (eng -> several targets), reported as
exp(mean loss) ablated / intact with a paired bootstrap over sentences.

Prior art at the activation level only: NLLB cross-attention sinks
(arXiv:2605.01229), T5 encoder outliers (2025.naacl-long.430). Nobody has
traced a massive activation in an encoder-decoder model to a weight.

    uv run src/encdec_sw.py --model facebook/nllb-200-distilled-600M \
        --tgt-langs fra_Latn,deu_Latn,spa_Latn,ces_Latn,cmn_Hans,arz_Arab
"""

import argparse
import datetime
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from bootstrap_ci import annotate, ratio_ci
from detect_sw import (EXPLAIN_FRAC, MASSIVE_FACTOR, MASSIVE_FRAC, MAX_K,
                       MIN_PERSIST, MIN_SHARE, OVERSHOOT, PLATEAU_FRAC, TOP_W,
                       TOP_X, decompose, find_super_activations)
from provenance import git_sha
from sw_arch import down_proj, get_stacks
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DET_ARGS = SimpleNamespace(massive_factor=MASSIVE_FACTOR, massive_frac=MASSIVE_FRAC,
                           plateau_frac=PLATEAU_FRAC, min_persist=MIN_PERSIST,
                           top_w=TOP_W, top_x=TOP_X, explain_frac=EXPLAIN_FRAC,
                           overshoot=OVERSHOOT, min_share=MIN_SHARE, max_k=MAX_K)
DEFAULT_PROMPT = "Language modeling is "
FLORES = ("openlanguagedata/flores_plus", "devtest")


# ------------------------------------------------------------ weights
def W(stacks, stack, layer):
    return down_proj(stacks[stack][layer]).weight


class Zeroed:
    def __init__(self, stacks, coords):
        self.stacks, self.coords = stacks, list(coords)

    def __enter__(self):
        self.saved = [W(self.stacks, s, L)[j, k].item() for s, L, j, k in self.coords]
        with torch.no_grad():
            for s, L, j, k in self.coords:
                W(self.stacks, s, L)[j, k] = 0.0
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for (s, L, j, k), v in zip(self.coords, self.saved):
                W(self.stacks, s, L)[j, k] = v


class ChannelIntervention:
    """Zero channel(s) at the output of stacks[stack][onset-1] wherever the
    trigger channel exceeds thr. targets: (stack, onset, trig, thr, chans)."""

    def __init__(self, stacks, targets):
        self.stacks, self.targets, self.hits = stacks, targets, []

    def _hook(self, specs):
        def hook(module, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).clone()
            n = 0
            for trig, thr, chans in specs:
                mask = h[..., trig].abs() > thr
                n += int(mask.sum().item())
                if mask.any():
                    b, t = mask.nonzero(as_tuple=True)
                    for c in chans:
                        h[b, t, c] = 0.0
            self.hits.append(n)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    def __enter__(self):
        by = {}
        for s, onset, trig, thr, chans in self.targets:
            by.setdefault((s, onset - 1), []).append((trig, thr, chans))
        self.handles = [self.stacks[s][L].register_forward_hook(self._hook(specs))
                        for (s, L), specs in by.items()]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------- detection
def forward_pass(model, stacks, enc, dec_ids):
    store = {}

    def make_hook(key):
        def hook(module, inp, out):
            store[key] = (inp[0][0].detach().float().cpu(),
                          out[0].detach().float().cpu())
        return hook

    handles = [down_proj(l).register_forward_hook(make_hook((s, i)))
               for s, layers in stacks.items() for i, l in enumerate(layers)]
    with torch.no_grad():
        out = model(**enc, decoder_input_ids=dec_ids, output_hidden_states=True)
    for h in handles:
        h.remove()
    H = {"encoder": torch.stack([h[0].float().cpu() for h in out.encoder_hidden_states]),
         "decoder": torch.stack([h[0].float().cpu() for h in out.decoder_hidden_states])}
    return H, store


def detect(model, stacks, tokenizer, enc, dec_ids, max_rounds=10, max_sw=6):
    found, seen, log, bars = [], set(), [], {}
    stop = f"hit max_rounds ({max_rounds})"
    for rnd in range(max_rounds):
        H, store = forward_pass(model, stacks, enc, dec_ids)
        fresh, rec = [], []
        for s in ("encoder", "decoder"):
            sas = find_super_activations(H[s], DET_ARGS, fixed_bar=bars.get(s))
            if s not in bars and sas:
                bars[s] = sas[0]["bar"]
            for sa in sas:
                X, Y = store[(s, sa["layer"])]
                cands, why = decompose(sa, X, Y, W(stacks, s, sa["layer"]), DET_ARGS)
                rec.append({**sa, "stack": s, "candidates": cands, "rejected": why})
                print(f"  round {rnd} {s} h[t{sa['token']},{sa['channel']}] peak "
                      f"{sa['peak']:.1f} onset {sa['onset']} (L{sa['layer']}) "
                      f"persists {sa['persist']}/{sa['window']}"
                      + (f" -- rejected: {why}" if why else ""), flush=True)
                for c in cands:
                    coord = (s, sa["layer"], c["j"], c["k"])
                    if coord not in seen:
                        fresh.append((sa, s, c))
        log.append({"round": rnd, "super_activations": rec})
        if not fresh:
            stop = ("no super activation survives" if not rec
                    else "no new candidate passed")
            break
        if len(found) + len(fresh) > max_sw:
            stop = f"exceeded max_sw ({max_sw}) at round {rnd}"
            break
        for sa, s, c in fresh:
            coord = (s, sa["layer"], c["j"], c["k"])
            seen.add(coord)
            found.append({"stack": s, "layer": sa["layer"], "j": c["j"], "k": c["k"],
                          "value": c["weight"], "round": rnd, "share": c["share"],
                          "peak": sa["peak"], "onset": sa["onset"],
                          "token": sa["token"], "channel": sa["channel"]})
            with torch.no_grad():
                W(stacks, s, sa["layer"])[c["j"], c["k"]] = 0.0
    with torch.no_grad():
        for f in found:
            W(stacks, f["stack"], f["layer"])[f["j"], f["k"]] = f["value"]
    profile = {s: [float(H[s][i].abs().max()) for i in range(H[s].shape[0])]
               for s in H}
    return found, stop, log, profile


# ------------------------------------------------------------- eval
def flores_pairs(src_lang, tgt_lang, n):
    from datasets import load_dataset
    path, split = FLORES
    src = load_dataset(path, src_lang, split=split)
    tgt = load_dataset(path, tgt_lang, split=split)
    by_id = {r["id"]: r["text"] for r in tgt}
    pairs = [(r["text"], by_id[r["id"]]) for r in src if r["id"] in by_id]
    return pairs[:n]


def translation_losses(model, tokenizer, batches):
    """Per-sentence teacher-forced cross-entropy (nats/token), all targets."""
    out = {}
    for lang, encs in batches.items():
        losses = []
        for enc in encs:
            with torch.no_grad():
                logits = model(**enc).logits
            labels = enc["labels"]
            lp = torch.log_softmax(logits.float(), -1)
            mask = labels != -100
            tok = lp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            per = -(tok * mask).sum(1) / mask.sum(1)
            losses += per.tolist()
        out[lang] = losses
    return out


def summarize(losses):
    allv = [v for l in losses.values() for v in l]
    return {"loss": sum(allv) / len(allv), "ppl": math.exp(sum(allv) / len(allv)),
            "per_lang": {l: math.exp(sum(v) / len(v)) for l, v in losses.items()},
            "loss_each": allv,
            "loss_each_by_lang": losses}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--src-lang", default="eng_Latn")
    ap.add_argument("--tgt-langs", default="fra_Latn,deu_Latn,spa_Latn,ces_Latn,cmn_Hans,arz_Arab")
    ap.add_argument("--n-sent", type=int, default=200, help="sentences per target")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--null-n", type=int, default=50)
    ap.add_argument("--null-pool", type=int, default=100)
    ap.add_argument("--sa-null-n", type=int, default=10)
    ap.add_argument("--sa-frac", type=float, default=0.5)
    ap.add_argument("--conc-ks", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--conc-top-sa", type=int, default=2, help="per stack")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out) if args.out else Path("results/encdec_v6") / (
        args.model.replace("/", "_") + "_encdec.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".manifest.json").write_text(json.dumps({
        "model": args.model, "args": vars(args), "git_sha": git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": datetime.datetime.now().isoformat(timespec="seconds")}, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, revision=args.revision,
        dtype=torch.float32 if device == "cpu" else "auto").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              src_lang=args.src_lang)
    stacks = get_stacks(model)
    tgts = args.tgt_langs.split(",")

    # ---- detection prompt: source = prompt; decoder = its own greedy
    # translation into the first target (teacher-forced), so both stacks see
    # a few real tokens.
    enc = tokenizer([args.prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**enc, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgts[0]),
                             max_new_tokens=16, do_sample=False)
    dec_ids = gen[:, :-1]
    print(f"detection: {args.prompt!r} -> {tokenizer.decode(gen[0], skip_special_tokens=True)!r}")
    found, stop, det_log, profile = detect(model, stacks, tokenizer, enc, dec_ids)
    print(f"\nfound {len(found)} candidate(s); stop: {stop}")
    for s in profile:
        print(f"  {s} max|h| by depth: {[round(v, 1) for v in profile[s]]}")
    sas, seen_sa = [], set()
    for rec in det_log[0]["super_activations"] if det_log else []:
        key = (rec["stack"], rec["token"], rec["channel"])
        if key not in seen_sa:
            seen_sa.add(key)
            sas.append({k: rec[k] for k in ("stack", "token", "channel", "onset",
                                            "layer", "peak")})

    # ---- eval batches: eng -> each target, teacher forced
    batches = {}
    for t in tgts:
        pairs = flores_pairs(args.src_lang, t, args.n_sent)
        tokenizer.tgt_lang = t
        bs = []
        for i in range(0, len(pairs), args.batch):
            chunk = pairs[i:i + args.batch]
            b = tokenizer([p[0] for p in chunk], text_target=[p[1] for p in chunk],
                          return_tensors="pt", padding=True, truncation=True,
                          max_length=256)
            b["labels"][b["labels"] == tokenizer.pad_token_id] = -100
            bs.append({k: v.to(device) for k, v in b.items()})
        batches[t] = bs
    coords = [(f["stack"], f["layer"], f["j"], f["k"]) for f in found]

    def sa_mag():
        if not sas:
            return []
        H, _ = forward_pass(model, stacks, enc, dec_ids)
        return [abs(H[s["stack"]][s["onset"], s["token"], s["channel"]].item()) for s in sas]

    base = summarize(translation_losses(model, tokenizer, batches))
    base["sa"] = sa_mag()
    tokenizer.tgt_lang = tgts[0]
    with torch.no_grad():
        sample = model.generate(**enc, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgts[0]),
                                max_new_tokens=16, do_sample=False)
    base["continuation"] = tokenizer.decode(sample[0], skip_special_tokens=True)
    print(f"\nbaseline translation ppl {base['ppl']:.3f}  per-lang "
          f"{ {k: round(v, 2) for k, v in base['per_lang'].items()} }  SA {base['sa']}")

    def run(name, cs, extra):
        with Zeroed(stacks, cs):
            m = summarize(translation_losses(model, tokenizer, batches))
            m["sa_after"] = sa_mag()
            with torch.no_grad():
                g = model.generate(**enc, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgts[0]),
                                   max_new_tokens=16, do_sample=False)
            m["continuation"] = tokenizer.decode(g[0], skip_special_tokens=True)
        m.update({"name": name, "coords": [list(c) for c in cs]})
        m["per_lang_ratio"] = {l: m["per_lang"][l] / base["per_lang"][l] for l in m["per_lang"]}
        m.update(extra)
        print(f"  {name:<28} ppl x{m['ppl'] / base['ppl']:<8.3f} per-lang "
              f"{ {k: round(v, 2) for k, v in m['per_lang_ratio'].items()} } "
              f"SA-after {[round(x) for x in m['sa_after']]}  | {m['continuation'][:40]!r}", flush=True)
        return m

    results = []
    print("\n[individual]")
    for f in found:
        c = (f["stack"], f["layer"], f["j"], f["k"])
        results.append(run(f"found {f['stack'][:3]} L{f['layer']}[{f['j']},{f['k']}]", [c],
                           {"kind": "individual", "weight": f["value"], **{k: f[k] for k in ("stack", "layer", "j", "k")}}))
    if len(coords) > 1:
        print("\n[joint + leave-one-out]")
        results.append(run("joint-all", coords, {"kind": "joint", "n": len(coords)}))
        for f, c in zip(found, coords):
            rest = [x for x in coords if x != c]
            results.append(run(f"loo-keep {f['stack'][:3]} L{f['layer']}[{f['j']},{f['k']}]", rest,
                               {"kind": "loo", "kept": list(c)}))

    null = []
    if coords and args.null_n:
        print(f"\n[null: {args.null_n} random top-{args.null_pool} weights]")
        per = {}
        for s, L, j, k in coords:
            per[(s, L)] = per.get((s, L), 0) + 1
        pools = {}
        for (s, L) in per:
            Wl = W(stacks, s, L).detach().float()
            D = Wl.shape[1]
            top = Wl.abs().flatten().topk(args.null_pool + len(coords)).indices.tolist()
            pool = [(s, L, i // D, i % D) for i in top]
            pools[(s, L)] = [p for p in pool if p not in set(coords)][:args.null_pool]
        pool_all = [p for v in pools.values() for p in v]
        for n in range(args.null_n):
            c = rng.choice(pool_all)
            null.append(run(f"null-ind-{n}", [c], {"kind": "null-individual"}))
        if len(coords) > 1:
            for n in range(args.null_n):
                cs = [x for key, cnt in per.items() for x in rng.sample(pools[key], cnt)]
                null.append(run(f"null-joint-{n}", cs, {"kind": "null-joint"}))

    # ---- concentration curve per stack: top-k contributors to the SA
    if sas:
        print("\n[concentration]")
        H0, store0 = forward_pass(model, stacks, enc, dec_ids)
        ks = [int(k) for k in args.conc_ks.split(",")]
        for stack in ("encoder", "decoder"):
            for sa in [s_ for s_ in sas if s_["stack"] == stack][:args.conc_top_sa]:
                L, t, j = sa["layer"], sa["token"], sa["channel"]
                x = store0[(stack, L)][0][t]
                w = W(stacks, stack, L).detach().float().cpu()[j]
                contrib = x * w
                order = contrib.abs().argsort(descending=True).tolist()
                total = contrib.sum().item()
                for k in ks:
                    if k > len(order):
                        break
                    top = order[:k]
                    results.append(run(f"conc {stack[:3]} ch{j} top{k}",
                                       [(stack, L, j, kk) for kk in top],
                                       {"kind": "concentration", "k": k, "sa": sa,
                                        "explained_share": contrib[top].sum().item() / total if total else float("nan")}))

    sa_results, sa_null = [], []
    if sas:
        print("\n[super-activation removal]")

        def run_sa(name, targets, extra):
            with ChannelIntervention(stacks, targets) as iv:
                m = summarize(translation_losses(model, tokenizer, batches))
                hits = iv.hits
                m["sa_after"] = sa_mag()
                with torch.no_grad():
                    g = model.generate(**enc, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgts[0]),
                                       max_new_tokens=16, do_sample=False)
                m["continuation"] = tokenizer.decode(g[0], skip_special_tokens=True)
            m.update({"name": name, "positions_hit_per_forward": sum(hits) / max(len(hits), 1)})
            m["per_lang_ratio"] = {l: m["per_lang"][l] / base["per_lang"][l] for l in m["per_lang"]}
            m.update(extra)
            print(f"  {name:<28} ppl x{m['ppl'] / base['ppl']:<8.3f} per-lang "
                  f"{ {k: round(v, 2) for k, v in m['per_lang_ratio'].items()} } "
                  f"hits/fwd {m['positions_hit_per_forward']:.1f} | {m['continuation'][:40]!r}", flush=True)
            return m

        targets = [(s["stack"], s["onset"], s["channel"], args.sa_frac * s["peak"], [s["channel"]])
                   for s in sas]
        for s, t in zip(sas, targets):
            sa_results.append(run_sa(f"sa-zero {s['stack'][:3]} ch{s['channel']} @h{s['onset']}", [t],
                                     {"kind": "sa-individual", "sa": s}))
        if len(sas) > 1:
            sa_results.append(run_sa("sa-zero all", targets, {"kind": "sa-joint"}))
        H0, _ = forward_pass(model, stacks, enc, dec_ids)
        for n in range(args.sa_null_n):
            ctrl = []
            for s in sas:
                h = H0[s["stack"]][s["onset"], s["token"]].abs()
                order = h.argsort().tolist()
                band = order[int(0.4 * len(order)):int(0.6 * len(order))]
                ctrl.append((s["stack"], s["onset"], s["channel"], args.sa_frac * s["peak"],
                             rng.sample(band, 1)))
            sa_null.append(run_sa(f"sa-null-{n}", ctrl, {"kind": "sa-null",
                                                         "channels": [c[4] for c in ctrl]}))

    doc = {
        "model": args.model,
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "git_sha": git_sha(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dtype": str(model.dtype), "device": device,
        "detector_version": "5-encdec", "detection_prompt": args.prompt,
        "detection_decoder_ids": dec_ids[0].tolist(),
        "params": vars(args), "seed": args.seed,
        "eval": {"dataset": FLORES, "src": args.src_lang, "tgts": tgts,
                 "n_sent_per_tgt": args.n_sent, "metric": "teacher-forced translation "
                 "cross-entropy, exp(mean) over all sentences; per_lang = per target"},
        "activation_profile": profile,
        "stop_reason": stop, "found": found, "rounds": det_log,
        "super_activations": sas,
        "baseline": {k: base[k] for k in ("ppl", "per_lang", "loss_each", "loss_each_by_lang", "sa", "continuation")},
        "results": results, "null": null, "sa_results": sa_results, "sa_null": sa_null,
    }
    # bootstrap over sentences (paired), on log-loss
    base_each = [math.exp(v) for v in base["loss_each"]]
    for key in ("results", "null", "sa_results", "sa_null"):
        for r in doc[key]:
            each = [math.exp(v) for v in r["loss_each"]]
            p, lo, hi = ratio_ci(base_each, each, seed=args.seed)
            r["ratio"], r["ratio_ci95"] = p, [lo, hi]
            r["ppl_each"] = each
    out.write_text(json.dumps(doc, indent=2))
    print(f"\nWritten to {out}")
    for r in results + sa_results:
        lo, hi = r["ratio_ci95"]
        print(f"  {r['name']:<28} x{r['ratio']:>8.3f} [{lo:.3f}, {hi:.3f}]")
    if null:
        mx = max(null, key=lambda r: r["ratio"])
        print(f"  null max x{mx['ratio']:.3f} ({mx['name']}) over {len(null)}")
    if not found:
        print("NO SUPER WEIGHT FOUND (no candidate passed detection); "
              "activation profile recorded.")


if __name__ == "__main__":
    main()
