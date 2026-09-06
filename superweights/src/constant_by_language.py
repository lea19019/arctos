"""Is the built-in constant the same for every input language?

For each language, run FLORES+ sentences through the model and record, per
sentence: where the massive activation appears (onset layer, residual channel,
token position), its magnitude at onset and at the last persisting layer, and
which down_proj weight(s) the v5 rule traces it to. Then summarise per
language with bootstrap CIs, and report whether channel / onset / coordinate
agree across languages.

    uv run src/constant_by_language.py --config configs/const_lang/towerbase7b.yaml

Architectures: llama-like (Tower, EuroLLM, Mistral, OLMo, Qwen3), Cohere (Aya),
BLOOM (transformer.h[i].mlp.dense_4h_to_h). Detection thresholds are v5's.
"""

import argparse
import datetime
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors.v5 import MASSIVE_FRAC, contributors, super_activations  # noqa: E402
from gen_loops import documents  # noqa: E402
from provenance import git_sha  # noqa: E402

DEFAULTS = {"revision": None, "dtype": "auto", "seed": 0, "split": "devtest",
            "n_sentences": 20, "add_bos": None, "out_dir": "results/const_lang"}


# ---------------------------------------------------------------- architecture
def ffn_layers(model):
    """(layers, get_out_proj) for the supported families."""
    mt = model.config.model_type
    if mt == "bloom":
        return list(model.transformer.h), (lambda l: l.mlp.dense_4h_to_h)
    return list(model.model.layers), (lambda l: l.mlp.down_proj)


def forward_pass(model, layers, out_proj, inputs):
    store = {}

    def make_hook(i):
        def hook(module, inp, out):
            store[i] = (inp[0][0].float().cpu(), out[0].float().cpu())
        return hook
    handles = [out_proj(l).register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    for h in handles:
        h.remove()
    H = torch.stack([h[0].float().cpu() for h in out.hidden_states])
    return H, store


def boot_ci(xs, B=1000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(open(args.config))}
    random.seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg["dtype"]]
    if device == "cpu":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=dtype).to(device).eval()
    layers, out_proj = ffn_layers(model)
    add_bos = cfg["add_bos"]
    if add_bos is None:   # tokenizer default
        add_bos = bool(tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True))

    # a fixed bar for "massive": v5 uses 10% of the largest residual peak seen;
    # we set it once on the English reference sentence so all languages share it
    ref = documents("eng_Latn", cfg["split"])[0]["text"]
    enc = tokenizer(ref, return_tensors="pt", return_token_type_ids=False,
                    add_special_tokens=add_bos).to(device)
    H, _ = forward_pass(model, layers, out_proj, enc)
    bar = MASSIVE_FRAC * H[:-1].abs().max().item()
    n_layers = len(layers)

    per_lang = {}
    for lang in cfg["languages"]:
        docs = documents(lang, cfg["split"])[:cfg["n_sentences"]]
        recs = []
        for d in docs:
            text = d["text"].split(". ")[0][:400]      # first sentence-ish
            enc = tokenizer(text, return_tensors="pt", return_token_type_ids=False,
                            add_special_tokens=add_bos).to(device)
            H, store = forward_pass(model, layers, out_proj, enc)
            sas = super_activations(H, bar)
            top = None
            if sas:
                t, j, L, pk = max(sas, key=lambda s: s[3])
                X, Y = store[L]
                W = out_proj(layers[L]).weight.float().cpu()
                contrib = [(int(k), float(w), float(share), bool(ok))
                           for k, w, share, ok in contributors(X[t], Y[t], W, j)]
                # magnitude on the plateau: median |h| over layers after onset (excl. last)
                plateau = H[L + 1:-1, t, j].abs()
                top = {"token": int(t), "token_str": tokenizer.decode(enc["input_ids"][0, t:t + 1]),
                       "channel": int(j), "onset_layer": int(L), "peak": float(pk),
                       "plateau_median": plateau.median().item() if plateau.numel() else None,
                       "last_layer_abs": H[-2, t, j].abs().item(),
                       "final_abs": H[-1, t, j].abs().item(),
                       "contributors": [{"k": k, "w": w, "share": share, "ok": ok} for k, w, share, ok in contrib]}
            recs.append({"doc_ids": d["ids"][:1], "n_tokens": int(enc["input_ids"].shape[1]),
                         "n_super_activations": len(sas),
                         "all": [{"token": int(t), "channel": int(j), "onset_layer": int(L), "peak": float(pk)}
                                 for t, j, L, pk in sas],
                         "top": top})
        tops = [r["top"] for r in recs if r["top"]]
        chan = Counter(t["channel"] for t in tops)
        onset = Counter(t["onset_layer"] for t in tops)
        tok = Counter(t["token"] for t in tops)
        coord = Counter((t["onset_layer"], t["channel"], t["contributors"][0]["k"])
                        for t in tops if t["contributors"] and t["contributors"][0]["ok"])
        peaks = [t["peak"] for t in tops]
        per_lang[lang] = {
            "n": len(recs), "n_with_massive": len(tops),
            "channel_mode": chan.most_common(1)[0] if chan else None,
            "onset_mode": onset.most_common(1)[0] if onset else None,
            "token_pos_mode": tok.most_common(1)[0] if tok else None,
            "coordinate_mode": [list(c) for c, n in coord.most_common(1)] + ([coord.most_common(1)[0][1]] if coord else []),
            "peak_mean": sum(peaks) / len(peaks) if peaks else None,
            "peak_ci95": boot_ci(peaks, seed=cfg["seed"]),
            "plateau_median_mean": (lambda xs: sum(xs) / len(xs) if xs else None)(
                [t["plateau_median"] for t in tops if t["plateau_median"] is not None]),
            "channels": dict(chan), "onsets": dict(onset), "token_positions": dict(tok),
            "records": recs,
        }
        s = per_lang[lang]
        print(f"[{lang}] massive in {s['n_with_massive']}/{s['n']}  channel {s['channel_mode']}  "
              f"onset {s['onset_mode']}  token {s['token_pos_mode']}  peak {s['peak_mean'] and round(s['peak_mean'],1)} "
              f"{s['peak_ci95'] and [round(x,1) for x in s['peak_ci95']]}  coord {s['coordinate_mode']}", flush=True)

    langs = [l for l in per_lang if per_lang[l]["channel_mode"]]
    agreement = {
        "channel_same_for_all": len({per_lang[l]["channel_mode"][0] for l in langs}) == 1 if langs else None,
        "onset_same_for_all": len({per_lang[l]["onset_mode"][0] for l in langs}) == 1 if langs else None,
        "coordinate_same_for_all": len({tuple(per_lang[l]["coordinate_mode"][0]) for l in langs
                                        if per_lang[l]["coordinate_mode"]}) == 1 if langs else None,
        "languages_with_massive": langs,
        "languages_without": [l for l in per_lang if l not in langs],
    }
    print("agreement:", json.dumps(agreement), flush=True)

    out = Path(cfg["out_dir"]) / (cfg["model"].replace("/", "_") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "provenance": {"date": datetime.datetime.now().isoformat(timespec="seconds"),
                       "git_sha": git_sha(), "config_file": args.config, "config": cfg,
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                       "dtype": str(model.dtype), "revision_resolved": getattr(model.config, "_commit_hash", None),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "model_type": model.config.model_type,
                       "n_layers": n_layers, "add_bos": add_bos, "massive_bar": bar,
                       "bos_token": tokenizer.bos_token},
        "agreement": agreement, "per_language": per_lang}, ensure_ascii=False, indent=1))
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
