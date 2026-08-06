"""PTQ-MT replication driver (arXiv:2508.20893) — one model, all methods.

Reproduces the paper's pipeline for a single model across the four official
quantizers (AWQ, BnB, GGUF, AutoRound) at 4-bit and 2-bit, on WMT24++ in both
directions for the six representative languages, scored with COMET
(wmt22-comet-da, the paper's metric) plus chrF and BLEU.

Design:
  * The atomic, resumable unit is one ``{method}-{bits}bit/{direction}.json``.
    A run skips any unit whose final JSON already exists, so a job killed at
    the SLURM time wall just continues on resubmit.
  * To avoid the LLM and the COMET model fighting for GPU memory (and to make
    big models tractable), each variant first *generates* all directions
    (caching raw hypotheses to ``.hyps.json``), then frees the LLM, then
    *scores*.
  * Disk policy: quantized artifacts (AWQ/AutoRound copies, GGUF files) are
    deleted right after a variant is scored unless ``--keep-artifacts``.

Run:
  python experiments/replication-uneven-ptq/experiment.py \
      --config experiments/replication-uneven-ptq/configs/qwen3-8b.yaml \
      --output results/replication-uneven-ptq/qwen3-8b
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import torch
import yaml
from sacrebleu.metrics import BLEU, CHRF

from src.data.wmt24pp import all_directions, load_wmt24pp
from src.eval.metrics import comet_score
from src.quant import calib, registry
from src.quant.hf_generate import translate as hf_translate

PAPER_COMET = "Unbabel/wmt22-comet-da"


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def _bleu_tok(tgt_lang: str) -> str:
    # sacrebleu tokenizers; ja gets the dedicated tokenizer, zh-like handling.
    return {"ja": "ja-mecab"}.get(tgt_lang, "13a")


def score(examples, hyps, *, comet_model: str, gpus: int) -> dict:
    sources = [e.source for e in examples]
    refs = [e.reference for e in examples]
    tgt = examples[0].tgt_lang
    comet_sys, comet_seg = comet_score(
        sources, hyps, refs, model_name=comet_model, gpus=gpus
    )
    chrf = float(CHRF().corpus_score(hyps, [refs]).score)          # plain chrF (paper)
    chrfpp = float(CHRF(word_order=2).corpus_score(hyps, [refs]).score)  # chrF++
    try:
        bleu = float(BLEU(tokenize=_bleu_tok(tgt)).corpus_score(hyps, [refs]).score)
    except Exception:
        bleu = float(BLEU(tokenize="13a").corpus_score(hyps, [refs]).score)
    return {
        "comet": comet_sys,
        "comet_model": comet_model,
        "comet_seg": comet_seg,
        "chrf": chrf,
        "chrfpp": chrfpp,
        "bleu": bleu,
    }


# --------------------------------------------------------------------------- #
# path + IO helpers
# --------------------------------------------------------------------------- #
def variant_dir(out: Path, method: str | None, bits: int | None) -> Path:
    return out / ("baseline-fp16" if method is None else f"{method}-{bits}bit")


def final_path(out: Path, method, bits, direction) -> Path:
    return variant_dir(out, method, bits) / f"{direction}.json"


def hyps_path(out: Path, method, bits, direction) -> Path:
    return variant_dir(out, method, bits) / f"{direction}.hyps.json"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


# --------------------------------------------------------------------------- #
# generation per backend (writes .hyps.json checkpoints)
# --------------------------------------------------------------------------- #
def _gen_hf(model, tok, data, out, method, bits, directions, cfg) -> None:
    for d in directions:
        if final_path(out, method, bits, d).exists() or hyps_path(out, method, bits, d).exists():
            continue
        t0 = time.time()
        hyps = hf_translate(
            model, tok, data[d],
            chat_kwargs=cfg.get("chat_kwargs"),
            max_new_tokens=cfg.get("max_new_tokens", 512),
            batch_size=cfg.get("batch_size", 16),
        )
        _write_json(hyps_path(out, method, bits, d),
                    {"hyps": hyps, "gen_seconds": round(time.time() - t0, 1)})
        print(f"   generated {method}-{bits}bit {d}  n={len(hyps)}  "
              f"{time.time()-t0:.0f}s", flush=True)


def _score_pending(data, out, method, bits, directions, comet_model, gpus, base_cfg) -> None:
    for d in directions:
        fp = final_path(out, method, bits, d)
        if fp.exists():
            continue
        hp = hyps_path(out, method, bits, d)
        if not hp.exists():
            continue
        hyps = json.loads(hp.read_text())["hyps"]
        s = score(data[d], hyps, comet_model=comet_model, gpus=gpus)
        _write_json(fp, {
            **base_cfg, "method": method or "baseline", "bits": bits,
            "direction": d, "n": len(hyps), "hyps": hyps, **s,
        })
        hp.unlink(missing_ok=True)
        print(f"   scored {method}-{bits}bit {d}  COMET={s['comet']:.2f} "
              f"chrF={s['chrf']:.1f}", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--methods", nargs="+", default=list(registry.ALL_METHODS) + ["baseline"])
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 2])
    ap.add_argument("--directions", nargs="+", default=None,
                    help="default: all six langs, both directions")
    ap.add_argument("--n", type=int, default=1000, help="examples per direction")
    ap.add_argument("--comet-model", default=PAPER_COMET)
    ap.add_argument("--comet-gpus", type=int, default=1)
    ap.add_argument("--calib-n", type=int, default=512, help="WikiText lines for AWQ/AutoRound")
    ap.add_argument("--imatrix-n", type=int, default=20000, help="WikiText lines for GGUF imatrix")
    ap.add_argument("--keep-artifacts", action="store_true")
    ap.add_argument("--gguf-port", type=int, default=8080)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["model"]
    base_path = cfg["hf_path"]
    gen_cfg = {
        "chat_kwargs": cfg.get("chat_kwargs"),
        "max_new_tokens": cfg.get("max_new_tokens", 512),
        "batch_size": cfg.get("batch_size", 16),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    directions = args.directions or all_directions()
    base_meta = {"model": cfg["name"], "hf_path": base_path}

    print(f">> loading WMT24++ ({len(directions)} directions, n<= {args.n})", flush=True)
    data = {d: load_wmt24pp(d, n=args.n) for d in directions}

    want_methods = set(args.methods)

    # ---- baseline (fp16) --------------------------------------------------- #
    if "baseline" in want_methods and any(
        not final_path(out, None, None, d).exists() for d in directions
    ):
        print(">> BASELINE fp16", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        _gen_hf(model, tok, data, out, None, None, directions, gen_cfg)
        _free(model)
        _score_pending(data, out, None, None, directions,
                       args.comet_model, args.comet_gpus, base_meta)

    # ---- HF quantizers (awq, bnb, autoround) ------------------------------- #
    calib_texts = None
    for bits in args.bits:
        for name in registry.methods_for_bits(bits, tuple(m for m in args.methods if m in registry.ALL_METHODS)):
            m = registry.get(name)
            if m.backend != "hf":
                continue
            pending = [d for d in directions if not final_path(out, name, bits, d).exists()]
            if not pending:
                continue
            print(f">> {name} {bits}-bit  ({len(pending)} directions pending)", flush=True)
            tok = None
            if m.needs_artifact:
                if calib_texts is None:
                    calib_texts = list(calib.wikitext_lines(args.calib_n))
                art = str(variant_dir(out, name, bits) / "artifact")
                if not Path(art).exists():
                    free = _disk_free_gb(out)
                    print(f"   disk free {free:.0f} GB; quantizing -> {art}", flush=True)
                    m.module.quantize_to_disk(base_path, bits, art, calib_texts)
                model, tok = m.module.load_model(art, bits)
            else:
                model, tok = m.module.load_model(base_path, bits)
            _gen_hf(model, tok, data, out, name, bits, pending, gen_cfg)
            _free(model)
            _score_pending(data, out, name, bits, pending,
                           args.comet_model, args.comet_gpus, base_meta)
            if m.needs_artifact and not args.keep_artifacts:
                shutil.rmtree(variant_dir(out, name, bits) / "artifact", ignore_errors=True)

    # ---- GGUF (shared f16 base + generic imatrix) -------------------------- #
    if "gguf" in want_methods:
        gguf_bits = [b for b in args.bits if b in (4, 2)]
        gguf_pending = any(
            not final_path(out, "gguf", b, d).exists()
            for b in gguf_bits for d in directions
        )
        if gguf_pending:
            from transformers import AutoTokenizer
            from src.quant import gguf
            tok = AutoTokenizer.from_pretrained(base_path)
            work = out / "gguf-work"
            work.mkdir(parents=True, exist_ok=True)
            f16 = str(work / "base.f16.gguf")
            if not gguf.is_valid_gguf(f16):
                gguf.convert_to_f16(base_path, f16)
            imat = str(work / "imatrix.dat")
            if not Path(imat).exists():
                cfile = calib.write_imatrix_file(
                    calib.wikitext_lines(args.imatrix_n), str(work / "wikitext.txt"))
                gguf.build_imatrix(f16, cfile, imat)
            for bits in gguf_bits:
                pending = [d for d in directions if not final_path(out, "gguf", bits, d).exists()]
                if not pending:
                    continue
                qpath = str(work / f"model.{gguf.QTYPE[bits]}.gguf")
                if not gguf.is_valid_gguf(qpath):
                    gguf.quantize(f16, bits, qpath, imatrix=imat)
                for d in pending:
                    if hyps_path(out, "gguf", bits, d).exists():
                        continue
                    t0 = time.time()
                    hyps = gguf.translate(
                        qpath, tok, data[d], chat_kwargs=gen_cfg["chat_kwargs"],
                        max_new_tokens=gen_cfg["max_new_tokens"], port=args.gguf_port)
                    _write_json(hyps_path(out, "gguf", bits, d),
                                {"hyps": hyps, "gen_seconds": round(time.time()-t0, 1)})
                _score_pending(data, out, "gguf", bits, pending,
                               args.comet_model, args.comet_gpus, base_meta)
                if not args.keep_artifacts:
                    Path(qpath).unlink(missing_ok=True)
            if not args.keep_artifacts:
                shutil.rmtree(work, ignore_errors=True)

    print(">> DONE", flush=True)


if __name__ == "__main__":
    main()
