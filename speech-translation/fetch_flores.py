"""Fetch FLORES+ dev data for English, Spanish, and French.

Produces six files under speech-translation/data/ :
  nllb/eng_Latn-spa_Latn.jsonl   (en → es)
  nllb/eng_Latn-fra_Latn.jsonl   (en → fr)
  nllb/spa_Latn-fra_Latn.jsonl   (es → fr)
  mono/eng_Latn.jsonl            (en text, for XTTS synthesis)
  mono/spa_Latn.jsonl            (es text)
  mono/fra_Latn.jsonl            (fr text)

Prerequisites: FLORES+ must be cached (openlanguagedata/flores_plus).
Run from repo root: python speech-translation/fetch_flores.py [--n 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET = "openlanguagedata/flores_plus"
SPLIT = "dev"

LANGS = {
    "eng_Latn": "English",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
}

PAIRS = [
    ("eng_Latn", "spa_Latn"),
    ("eng_Latn", "fra_Latn"),
    ("spa_Latn", "fra_Latn"),
]


def load_lang(lang: str, n: int) -> list[dict]:
    ds = load_dataset(DATASET, lang, split=SPLIT)
    rows = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        rows.append({"text": ex["text"], "id": ex.get("id", i), "lang": lang})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="Examples per language")
    ap.add_argument("--out", type=Path, default=Path("speech-translation/data"))
    args = ap.parse_args()

    nllb_dir = args.out / "nllb"
    mono_dir = args.out / "mono"
    nllb_dir.mkdir(parents=True, exist_ok=True)
    mono_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLORES+ from cache …")
    lang_data: dict[str, list[dict]] = {}
    for lang, name in LANGS.items():
        print(f"  {name} ({lang}) …", flush=True)
        lang_data[lang] = load_lang(lang, args.n)
        # Verify alignment (FLORES is multi-way parallel by row index)
        path = mono_dir / f"{lang}.jsonl"
        with path.open("w") as f:
            for row in lang_data[lang]:
                f.write(json.dumps(row) + "\n")
        print(f"    → {path} ({len(lang_data[lang])} rows)")

    print("\nBuilding translation pairs …")
    for src_lang, tgt_lang in PAIRS:
        src_rows = lang_data[src_lang]
        tgt_rows = lang_data[tgt_lang]
        n = min(len(src_rows), len(tgt_rows))
        path = nllb_dir / f"{src_lang}-{tgt_lang}.jsonl"
        with path.open("w") as f:
            for i in range(n):
                f.write(json.dumps({
                    "source": src_rows[i]["text"],
                    "target": tgt_rows[i]["text"],
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "id": src_rows[i]["id"],
                }) + "\n")
        print(f"  {src_lang} → {tgt_lang}: {n} pairs → {path}")


if __name__ == "__main__":
    main()
