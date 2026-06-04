"""C3 deep-dive: does language-matched calibration help at 2-bit? (the linchpin)

This is the load-bearing claim for the Arctos phase-two direction, so it gets
its own self-contained module. We reproduce the paper exactly: Llama-3.1-8B-
Instruct (already cached), GGUF Q4_K_M and Q2_K, with the imatrix estimated on
either English or Bengali FineWeb-2 text (~10k tokens each, materialized by
precache.py). Four directions: en->fr, en->bn, fr->en, bn->en.

Paper claim (Tables 7-8): at 2-bit, Bengali-matched calibration lifts en->bn
(+3.1 COMET) but barely moves at 4-bit, and French (uncalibrated) is
unaffected. NOTE: the paper's into-English Table 8 prose appears to contradict
its own numbers (it calls 71.5 vs 72.3 a "gain"); we report what we actually
measure in both directions and flag any contradiction in the findings doc.

Run (after the f16 base + calib files exist):
  python experiments/replication-uneven-ptq/c3_calibration.py \
      --output results/replication-uneven-ptq/c3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from src.data.wmt24pp import load_wmt24pp
from src.eval.metrics import comet_score
from src.quant import gguf

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PAPER_COMET = "Unbabel/wmt22-comet-da"
DIRECTIONS = ["en-fr", "en-bn", "fr-en", "bn-en"]
CALIB_LANGS = ["en", "bn"]
BITS = [4, 2]
CALIB_DIR = Path(__file__).resolve().parent / "calib"


def _score(examples, hyps, gpus):
    sources = [e.source for e in examples]
    refs = [e.reference for e in examples]
    comet_sys, comet_seg = comet_score(sources, hyps, refs, model_name=PAPER_COMET, gpus=gpus)
    return {
        "comet": comet_sys, "comet_seg": comet_seg, "comet_model": PAPER_COMET,
        "chrf": float(CHRF().corpus_score(hyps, [refs]).score),
        "bleu": float(BLEU(tokenize="13a").corpus_score(hyps, [refs]).score),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--comet-gpus", type=int, default=1)
    ap.add_argument("--gguf-port", type=int, default=8090)
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = {d: load_wmt24pp(d, n=args.n) for d in DIRECTIONS}

    # 1. f16 base GGUF (once).
    f16 = str(work / "base.f16.gguf")
    if not gguf.is_valid_gguf(f16):
        gguf.convert_to_f16(MODEL, f16)

    # 2. one imatrix per calibration language.
    imats = {}
    for lang in CALIB_LANGS:
        calib_file = CALIB_DIR / f"{lang}.txt"
        if not calib_file.exists():
            raise FileNotFoundError(
                f"Missing C3 calib text {calib_file}; run precache.py first.")
        imat = str(work / f"imatrix.{lang}.dat")
        if not Path(imat).exists():
            gguf.build_imatrix(f16, str(calib_file), imat)
        imats[lang] = imat

    # 3+4. per (calib_lang, bits): quantize then translate+score each direction.
    for lang in CALIB_LANGS:
        for bits in BITS:
            vdir = out / f"calib-{lang}-{bits}bit"
            pending = [d for d in DIRECTIONS if not (vdir / f"{d}.json").exists()]
            if not pending:
                continue
            qpath = str(work / f"model.{lang}.{gguf.QTYPE[bits]}.gguf")
            if not gguf.is_valid_gguf(qpath):
                gguf.quantize(f16, bits, qpath, imatrix=imats[lang])
            for d in pending:
                t0 = time.time()
                hyps = gguf.translate(qpath, tok, data[d], chat_kwargs=None,
                                      max_new_tokens=512, port=args.gguf_port)
                s = _score(data[d], hyps, args.comet_gpus)
                vdir.mkdir(parents=True, exist_ok=True)
                (vdir / f"{d}.json").write_text(json.dumps({
                    "model": "llama-3.1-8b-instruct", "calib_lang": lang,
                    "bits": bits, "direction": d, "n": len(hyps),
                    "hyps": hyps, "gen_seconds": round(time.time()-t0, 1), **s,
                }, ensure_ascii=False), encoding="utf-8")
                print(f"   calib={lang} {bits}bit {d}  COMET={s['comet']:.2f}", flush=True)
            if not args.keep_artifacts:
                Path(qpath).unlink(missing_ok=True)

    if not args.keep_artifacts:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    print(">> C3 DONE", flush=True)


if __name__ == "__main__":
    main()
