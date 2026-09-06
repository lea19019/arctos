"""Lesion a known super-weight coordinate with the control triple and a dose
sweep, and measure perplexity, the massive activation, and repetition.

Conditions (docs/repetition_experiment_design.md §4):
  dose a in {1, .75, .5, .25, 0}     W[L][j,k] *= a            (a=1 is the intact baseline)
  contribution-mean                  W[L][j,k] = 0, and the scalar's contribution
                                     W[j,k]*x[.,k] is replaced by its calibration mean,
                                     bucketed by position (first token vs the rest) —
                                     the weight analogue of Sun et al. 2024's set-to-mean
  matched-random                     N random weights from the top-`pool` |W| of the same
                                     matrix (excluding the coordinate) set to zero, one at
                                     a time; report the max ratio (max-statistic null)

Per condition: wikitext-2 test perplexity over fixed windows with a paired
bootstrap CI on the ratio to baseline; the massive activation |h[0, j]| after
layer L on a probe prompt (dose check); and greedy continuations of FLORES+
documents scored with the pre-registered repetition metrics.

    uv run src/lesion_controls.py --config configs/lesion/olmo1b.yaml
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_metrics as rm  # noqa: E402
from gen_loops import build_prompts, wilson  # noqa: E402
from provenance import git_sha  # noqa: E402

DEFAULTS = {
    "revision": None, "dtype": "auto", "seed": 0,
    "doses": [1.0, 0.75, 0.5, 0.25, 0.0],
    "contribution_mean": True,
    "matched_random_draws": 50, "matched_random_pool": 100,
    "eval": {"segments": 32, "seq_len": 2048},
    "calibration": {"split": "validation", "segments": 8, "seq_len": 2048},
    "generation": {"language": "eng_Latn", "split": "devtest", "n_prompts": 50,
                   "prefix_tokens": 50, "max_new_tokens": 128},
    "probe_prompt": "Language modeling is ",
    "bootstrap": 2000,
    "out_dir": "results/lesion",
}
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")


# ---------------------------------------------------------------- model access
def layers_of(model):
    return model.model.layers


def down_proj(model, L):
    return layers_of(model)[L].mlp.down_proj


def wikitext_windows(tokenizer, split, n_segments, seq_len, device):
    from datasets import load_dataset
    ds = load_dataset(WIKITEXT[0], WIKITEXT[1], split=split)
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = min(n_segments, ids.numel() // seq_len)
    return [ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device) for i in range(n)]


@torch.no_grad()
def window_losses(model, windows):
    out = []
    for w in windows:
        out.append(model(input_ids=w, labels=w).loss.item())
    return out


def ppl(losses):
    return math.exp(sum(losses) / len(losses))


def bootstrap_ratio(base_losses, abl_losses, B, seed):
    """Paired bootstrap over windows of exp(mean(abl) - mean(base))."""
    rng = random.Random(seed)
    n = len(base_losses)
    diffs = [a - b for a, b in zip(abl_losses, base_losses)]
    point = math.exp(sum(diffs) / n)
    samples = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        m = sum(diffs[i] for i in idx) / n
        samples.append(math.exp(min(m, 700)))
    samples.sort()
    return point, [samples[int(0.025 * B)], samples[int(0.975 * B) - 1]]


@torch.no_grad()
def massive_activation(model, tokenizer, prompt, L, j, device):
    enc = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    h = hs[L + 1][0].float()          # residual stream after decoder layer L
    return {"h0_j": h[0, j].item(), "max_abs_j": h[:, j].abs().max().item(),
            "argmax_token_j": int(h[:, j].abs().argmax().item()),
            "max_abs_all": h.abs().max().item()}


@torch.no_grad()
def generate_one(model, tokenizer, prompt_ids, max_new, device, use_cache=True):
    bos = ([tokenizer.bos_token_id] if tokenizer.bos_token_id is not None
           and getattr(tokenizer, "add_bos_token", True) else [])
    ids = torch.tensor([bos + prompt_ids], device=device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new, do_sample=False, num_beams=1,
                         repetition_penalty=1.0, no_repeat_ngram_size=0, use_cache=use_cache,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                         eos_token_id=tokenizer.eos_token_id)
    g = out[0, ids.shape[1]:].tolist()
    ended = tokenizer.eos_token_id in g
    if ended:
        g = g[:g.index(tokenizer.eos_token_id)]
    return g, ended


def gen_summary(model, tokenizer, prompts, max_new, device, use_cache=True):
    recs = []
    for p in prompts:
        g, ended = generate_one(model, tokenizer, p["prompt_ids"], max_new, device, use_cache)
        recs.append({**rm.summarize(g, ended_with_eos=ended),
                     "gen_text": tokenizer.decode(g, skip_special_tokens=True)})
    n = len(recs)
    loops = sum(r["loop"] for r in recs)
    return {"n": n, "loop_rate": loops / n, "loop_rate_ci95": wilson(loops, n),
            "loop_with_window_rate": sum(r["loop_with_window"] for r in recs) / n,
            "eos_rate": sum(r["ended_with_eos"] for r in recs) / n,
            "seq_rep_4_mean": sum(r["seq_rep_4"] for r in recs) / n,
            "rep_r_mean": sum(r["rep_r"] for r in recs) / n,
            "examples": [r["gen_text"][:160] for r in recs[:3]]}, recs


# ---------------------------------------------------------------- contribution-mean hook
class ContributionMean:
    """Zero W[j,k] and add the calibration-mean contribution to output channel j,
    bucketed: position 0 gets mean_pos0, positions >= 1 get mean_rest.
    Requires full-sequence forwards (no KV cache) so positions are known."""

    def __init__(self, model, L, j, k):
        self.mod = down_proj(model, L)
        self.j, self.k = j, k
        self.w = self.mod.weight[j, k].item()
        self.mean_pos0 = self.mean_rest = None
        self.handle = None

    def calibrate(self, model, windows):
        xs0, xsr = [], []

        def cap(module, inp, out):
            x = inp[0][0, :, self.k].float()
            xs0.append(x[0].item())
            xsr.extend(x[1:].tolist())
        h = self.mod.register_forward_hook(cap)
        with torch.no_grad():
            for w in windows:
                model(input_ids=w)
        h.remove()
        self.mean_pos0 = self.w * sum(xs0) / len(xs0)
        self.mean_rest = self.w * sum(xsr) / len(xsr)
        return {"mean_contrib_pos0": self.mean_pos0, "mean_contrib_rest": self.mean_rest,
                "n_pos0": len(xs0), "n_rest": len(xsr)}

    def __enter__(self):
        with torch.no_grad():
            self.mod.weight[self.j, self.k] = 0.0

        def add(module, inp, out):
            out[:, 0, self.j] += self.mean_pos0
            if out.shape[1] > 1:
                out[:, 1:, self.j] += self.mean_rest
            return out
        self.handle = self.mod.register_forward_hook(add)
        return self

    def __exit__(self, *a):
        self.handle.remove()
        with torch.no_grad():
            self.mod.weight[self.j, self.k] = self.w


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for key, val in DEFAULTS.items():
        if isinstance(val, dict):
            cfg[key] = {**val, **cfg.get(key, {})}
        else:
            cfg.setdefault(key, val)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    L, j, k = cfg["coordinate"]["layer"], cfg["coordinate"]["j"], cfg["coordinate"]["k"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg["dtype"]]
    if device == "cpu":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=dtype).to(device).eval()
    W = down_proj(model, L).weight
    w0 = W[j, k].item()
    print(f"model {cfg['model']}  L{L}[{j},{k}] = {w0:+.5f}  rank by |W| = "
          f"{int((W.abs() > abs(w0)).sum().item()) + 1} of {W.numel()}", flush=True)

    seq_len = min(cfg["eval"]["seq_len"], getattr(model.config, "max_position_embeddings", 10**9))
    eval_w = wikitext_windows(tokenizer, "test", cfg["eval"]["segments"], seq_len, device)
    calib_w = wikitext_windows(tokenizer, cfg["calibration"]["split"],
                               cfg["calibration"]["segments"],
                               min(cfg["calibration"]["seq_len"], seq_len), device)
    g = cfg["generation"]
    prompts = build_prompts(tokenizer, g["language"], {"split": g["split"],
                            "prefix_tokens": g["prefix_tokens"], "n_prompts": g["n_prompts"]})

    def measure(name, use_cache=True):
        losses = window_losses(model, eval_w)
        ma = massive_activation(model, tokenizer, cfg["probe_prompt"], L, j, device)
        gsum, recs = gen_summary(model, tokenizer, prompts, g["max_new_tokens"], device, use_cache)
        print(f"  {name:<22} ppl {ppl(losses):10.2f}   |h0[{j}]| {abs(ma['h0_j']):9.1f}   "
              f"loop {gsum['loop_rate']:.2f}  seq-rep-4 {gsum['seq_rep_4_mean']:.3f}   "
              f"e.g. {gsum['examples'][0][:60]!r}", flush=True)
        return {"name": name, "losses": losses, "ppl": ppl(losses), "massive_activation": ma,
                "generation": gsum, "generation_records": recs}

    results = {}
    base = measure("baseline (a=1)")
    results["baseline"] = base
    B = cfg["bootstrap"]

    def attach_ratio(r):
        point, ci = bootstrap_ratio(base["losses"], r["losses"], B, cfg["seed"])
        r["ppl_ratio"], r["ppl_ratio_ci95"] = point, ci
        return r

    # dose sweep
    results["doses"] = []
    for a in cfg["doses"]:
        if a == 1.0:
            continue
        with torch.no_grad():
            W[j, k] = w0 * a
        r = attach_ratio(measure(f"dose a={a}"))
        r["dose"] = a
        results["doses"].append(r)
        with torch.no_grad():
            W[j, k] = w0

    # contribution-mean control
    if cfg["contribution_mean"]:
        cm = ContributionMean(model, L, j, k)
        calib = cm.calibrate(model, calib_w)
        with cm:
            r = attach_ratio(measure("contribution-mean", use_cache=False))
        r["calibration"] = calib
        results["contribution_mean"] = r
        assert W[j, k].item() == w0

    # magnitude-matched random null (ppl only, plus loop rate on a subset)
    n_draws, pool = cfg["matched_random_draws"], cfg["matched_random_pool"]
    if n_draws:
        flat = W.detach().abs().flatten()
        top = flat.topk(pool + 1).indices.tolist()
        cand = [(i // W.shape[1], i % W.shape[1]) for i in top if (i // W.shape[1], i % W.shape[1]) != (j, k)]
        rng = random.Random(cfg["seed"])
        draws = rng.sample(cand, min(n_draws, len(cand)))
        null = []
        for (jj, kk) in draws:
            orig = W[jj, kk].item()
            with torch.no_grad():
                W[jj, kk] = 0.0
            losses = window_losses(model, eval_w)
            with torch.no_grad():
                W[jj, kk] = orig
            point, ci = bootstrap_ratio(base["losses"], losses, 200, cfg["seed"])
            null.append({"j": jj, "k": kk, "weight": orig, "ppl_ratio": point, "ppl_ratio_ci95": ci})
        ratios = sorted(x["ppl_ratio"] for x in null)
        results["matched_random"] = {"pool": pool, "n_draws": len(null), "draws": null,
                                     "max_ratio": ratios[-1], "p95_ratio": ratios[int(0.95 * len(ratios)) - 1],
                                     "median_ratio": ratios[len(ratios) // 2]}
        print(f"  matched-random null   max x{ratios[-1]:.3f}  p95 x{results['matched_random']['p95_ratio']:.3f}  "
              f"over {len(null)} draws from top-{pool} |W|", flush=True)

    out = Path(cfg["out_dir"]) / (cfg["model"].replace("/", "_") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # per-generation texts are bulky: move them to a gitignored sidecar
    with open(out.with_suffix(".records.jsonl"), "w", encoding="utf-8") as fh:
        for cond in [results["baseline"], *results["doses"],
                     *([results["contribution_mean"]] if "contribution_mean" in results else [])]:
            for r in cond.pop("generation_records"):
                fh.write(json.dumps({"condition": cond["name"], **r}, ensure_ascii=False) + "\n")
    out.write_text(json.dumps({
        "provenance": {
            "date": datetime.datetime.now().isoformat(timespec="seconds"), "git_sha": git_sha(),
            "config_file": args.config, "config": cfg, "torch": torch.__version__,
            "transformers": transformers.__version__, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "dtype": str(model.dtype), "revision_resolved": getattr(model.config, "_commit_hash", None),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seq_len": seq_len,
            "eval_windows": len(eval_w), "calibration_windows": len(calib_w),
            "coordinate": {"layer": L, "j": j, "k": k, "weight": w0}},
        "results": results}, ensure_ascii=False, indent=1))
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
