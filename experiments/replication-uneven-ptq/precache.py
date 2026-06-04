"""Login-node pre-caching for the PTQ-MT replication.

Compute nodes are offline, so everything the SLURM jobs touch must be cached
here first: the four not-yet-local models, the WMT24++ test configs, the
WikiText calibration corpus, and — because FineWeb-2 is far too large to cache
and can't be streamed offline — the *materialized* English/Bengali C3
calibration text files.

Run on the login node (has internet):
  python experiments/replication-uneven-ptq/precache.py
Add --skip-models to refresh only datasets/calib.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# This script runs ONLINE on the login node; make sure offline flags are off.
for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.pop(v, None)
os.environ.setdefault("OPENSSL_CONF", "/dev/null")

MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    # meta-llama/Llama-3.1-8B-Instruct is already cached.
]
WMT24PP_CONFIGS = ["en-ja_JP", "en-fr_FR", "en-pl_PL", "en-bn_IN", "en-ml_IN", "en-zu_ZA"]

CALIB_DIR = Path(__file__).resolve().parent / "calib"
C3_TOKENS = 10_000          # paper: ~10k tokens per language for the C3 imatrix
C3_LANGS = ["en", "bn"]     # paper's English-vs-Bengali contrast
TOKENIZER_FOR_TOKENCOUNT = "meta-llama/Llama-3.1-8B-Instruct"  # the C3 model


def cache_models() -> None:
    from huggingface_hub import snapshot_download

    for repo in MODELS:
        print(f">> caching model {repo}", flush=True)
        snapshot_download(repo, ignore_patterns=["*.pth", "*.bin.index.json.bak"])


def cache_wmt24pp() -> None:
    from datasets import load_dataset

    for cfg in WMT24PP_CONFIGS:
        print(f">> caching WMT24++ {cfg}", flush=True)
        load_dataset("google/wmt24pp", cfg, split="train")


def cache_wikitext() -> None:
    from datasets import load_dataset

    print(">> caching WikiText-103 (imatrix/AWQ/AutoRound calibration)", flush=True)
    load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")


def materialize_c3_calib() -> None:
    from transformers import AutoTokenizer

    from src.quant.calib import fineweb2_text

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_FOR_TOKENCOUNT)
    for lang in C3_LANGS:
        path = CALIB_DIR / f"{lang}.txt"
        if path.exists():
            print(f">> C3 calib {lang} already materialized ({path})", flush=True)
            continue
        print(f">> materializing C3 calib {lang} (~{C3_TOKENS} tokens)", flush=True)
        text = fineweb2_text(lang, C3_TOKENS, tok)
        path.write_text(text, encoding="utf-8")
        print(f"   wrote {path} ({len(text)} chars)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()

    if not args.skip_models:
        cache_models()
    cache_wmt24pp()
    cache_wikitext()
    materialize_c3_calib()
    print(">> precache DONE", flush=True)


if __name__ == "__main__":
    main()
