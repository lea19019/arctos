"""Fetch FLORES+ dev splits and write data/{pair}.jsonl for the three Q1 pairs.

Prerequisites:
1. `huggingface-cli login` (token from huggingface.co/settings/tokens).
2. Visit https://huggingface.co/datasets/openlanguagedata/flores_plus and
   click "Request access" (auto-approved).

Run from repo root:
    python scripts/fetch_flores.py [--n 200] [--out data]

FLORES+ is multi-way parallel: every language config carries the same set of
source sentences, so we can build any pair by aligning rows.

Pairs produced:
    cs-de   ces_Latn -> deu_Latn
    en-zh   eng_Latn -> zho_Hans
    en-arz  eng_Latn -> arz_Arab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# (pair_name, source_config, target_config)
PAIRS = [
    ("cs-de", "ces_Latn", "deu_Latn"),
    ("en-zh", "eng_Latn", "zho_Hans"),
    ("en-arz", "eng_Latn", "arz_Arab"),
]
DATASET = "openlanguagedata/flores_plus"
SPLIT = "dev"


def fetch_pair(src_cfg: str, tgt_cfg: str, n: int | None) -> list[tuple[str, str, str]]:
    """Return list of (source_text, target_text, source_id) rows for one pair."""
    src = load_dataset(DATASET, src_cfg, split=SPLIT)
    tgt = load_dataset(DATASET, tgt_cfg, split=SPLIT)
    if len(src) != len(tgt):
        raise RuntimeError(
            f"FLORES+ {src_cfg} and {tgt_cfg} have mismatched row counts "
            f"({len(src)} vs {len(tgt)}); cannot align by index."
        )
    upper = len(src) if n is None else min(n, len(src))
    rows = []
    for i in range(upper):
        s_row, t_row = src[i], tgt[i]
        # FLORES+ rows expose `text` and `id` (the parallel sentence id).
        sid = f"flores_plus.dev.{s_row.get('id', i)}"
        rows.append((s_row["text"], t_row["text"], sid))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="per-pair example cap")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for pair_name, src_cfg, tgt_cfg in PAIRS:
        print(f"[{pair_name}] {src_cfg} -> {tgt_cfg} (n={args.n}) ...", flush=True)
        rows = fetch_pair(src_cfg, tgt_cfg, args.n)
        path = args.out / f"{pair_name}.jsonl"
        with path.open("w") as f:
            for src, tgt, sid in rows:
                f.write(json.dumps({"source": src, "target": tgt, "source_id": sid}) + "\n")
        print(f"[{pair_name}] wrote {len(rows)} records to {path}", flush=True)


if __name__ == "__main__":
    main()
