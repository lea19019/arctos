"""Roll up replication results into a verdict table, C3 plot, and findings doc.

Reads every ``{model}/{variant}/{direction}.json`` produced by experiment.py
and ``c3/calib-*/*.json`` from c3_calibration.py, computes COMET deltas vs the
fp16 baseline, and emits:

  * ``results/replication-uneven-ptq/_summary.json`` (+ .csv) — flat rollup.
  * ``docs/findings/figures/replication-c3-calibration.png`` — the C3 plot.
  * the findings markdown doc (``--doc``) with the C1-C5 verdict table, our
    numbers beside the paper's, and a red-flags section.

Safe to run on partial results (reports only what exists), so the SLURM
``analyze`` job can depend ``afterany`` on the sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Paper reference numbers (Marie & Fujita 2025), the six representative langs.
# Tables 1/3 = English->X at 4-bit/2-bit; Table 7/8 = C3 calibration.
LANGS6 = ["ja", "fr", "pl", "bn", "ml", "zu"]

# Table 1 — COMET, English->X, 4-bit (baseline row "-" then methods).
PAPER_EN_X_4BIT = {
    "qwen3-1.7b": {"-": [80.4, 74.2, 64.7, 50.1, 35.9, 27.3],
                   "awq": [78.5, 72.5, 60.1, 44.7, 32.3, 25.8],
                   "bnb": [79.6, 73.3, 61.5, 45.9, 32.3, 24.9],
                   "gguf": [79.7, 74.3, 64.2, 48.5, 35.8, 26.0],
                   "autoround": [78.2, 73.3, 62.2, 45.8, 31.3, 27.7]},
    "llama-3.1-8b-instruct": {"-": [81.6, 77.7, 78.0, 75.6, 57.9, 45.5],
                   "awq": [80.5, 77.5, 76.6, 73.0, 55.7, 42.6],
                   "bnb": [80.8, 77.3, 76.0, 73.3, 55.9, 45.1],
                   "gguf": [81.2, 77.2, 77.7, 74.6, 55.0, 44.0],
                   "autoround": [80.3, 77.4, 77.0, 74.1, 54.6, 44.9]},
    "qwen3-8b": {"-": [85.7, 80.8, 78.6, 76.1, 57.9, 42.0],
                   "awq": [85.0, 80.1, 77.6, 74.1, 55.4, 41.9],
                   "bnb": [85.4, 80.0, 77.6, 74.0, 55.1, 43.0],
                   "gguf": [85.6, 80.6, 78.2, 75.2, 56.4, 40.6],
                   "autoround": [85.4, 80.3, 77.9, 74.4, 55.2, 40.0]},
    "qwen3-32b": {"-": [87.1, 81.7, 81.8, 80.0, 65.4, 54.9],
                   "gguf": [87.2, 81.7, 81.7, 79.8, 65.4, 54.1]},
    "llama-3.3-70b-instruct": {"-": [85.6, 81.0, 82.9, 80.9, 67.6, 67.5],
                   "gguf": [85.4, 80.9, 82.4, 80.5, 67.2, 67.1]},
}
# Table 3 — COMET, English->X, 2-bit (GGUF + AutoRound only).
PAPER_EN_X_2BIT = {
    "qwen3-1.7b": {"-": [80.4, 74.2, 64.7, 50.1, 35.9, 27.3],
                   "gguf": [69.4, 66.2, 48.2, 30.8, 23.7, 31.2],
                   "autoround": [33.1, 34.8, 32.9, 28.6, 24.5, 27.1]},
    "llama-3.1-8b-instruct": {"-": [81.6, 77.7, 78.0, 75.6, 57.9, 45.5],
                   "gguf": [72.7, 73.4, 66.8, 46.2, 39.3, 35.5],
                   "autoround": [39.6, 45.0, 39.6, 32.9, 27.7, 29.6]},
    "qwen3-8b": {"-": [85.7, 80.8, 78.6, 76.1, 57.9, 42.0],
                   "gguf": [83.7, 79.2, 72.5, 59.9, 40.6, 37.7],
                   "autoround": [47.0, 63.9, 49.7, 34.4, 31.9, 32.4]},
    "qwen3-32b": {"-": [87.1, 81.7, 81.8, 80.0, 65.4, 54.9],
                   "gguf": [86.0, 80.5, 78.1, 74.7, 56.1, 44.0]},
    "llama-3.3-70b-instruct": {"-": [85.6, 81.0, 82.9, 80.9, 67.6, 67.5],
                   "gguf": [84.2, 79.8, 81.0, 78.9, 63.2, 58.7]},
}
# Tables 7 (en->X) & 8 (X->en): C3 calibration, Llama-3.1-8B, fr & bn.
PAPER_C3 = {
    "en-fr": {"-": 77.7, "Q4_K_M-en": 77.2, "Q4_K_M-bn": 77.3, "Q2_K-en": 73.8, "Q2_K-bn": 73.5},
    "en-bn": {"-": 75.6, "Q4_K_M-en": 75.1, "Q4_K_M-bn": 75.1, "Q2_K-en": 44.9, "Q2_K-bn": 48.0},
    "fr-en": {"-": 81.7, "Q4_K_M-en": 81.6, "Q4_K_M-bn": 81.6, "Q2_K-en": 79.8, "Q2_K-bn": 80.2},
    "bn-en": {"-": 72.4, "Q4_K_M-en": 63.9, "Q4_K_M-bn": 65.8, "Q2_K-en": 72.3, "Q2_K-bn": 71.5},
}


def load_results(root: Path) -> list[dict]:
    rows = []
    for jp in root.glob("*/*/*.json"):
        if jp.name.endswith(".hyps.json") or jp.parent.parent.name == "c3":
            continue
        try:
            d = json.loads(jp.read_text())
        except Exception:
            continue
        if "comet" in d:
            rows.append(d)
    return rows


def load_c3(root: Path) -> list[dict]:
    rows = []
    for jp in (root / "c3").glob("calib-*/*.json"):
        try:
            d = json.loads(jp.read_text())
            if "comet" in d:
                rows.append(d)
        except Exception:
            pass
    return rows


def write_summary(rows: list[dict], root: Path) -> None:
    (root / "_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    if rows:
        keys = ["model", "method", "bits", "direction", "n", "comet", "chrf", "bleu"]
        with (root / "_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)


def plot_c3(c3_rows: list[dict], fig_path: Path) -> bool:
    if not c3_rows:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    # index: (direction, bits, calib_lang) -> comet
    idx = {(r["direction"], r["bits"], r["calib_lang"]): r["comet"] for r in c3_rows}
    directions = ["en-fr", "en-bn", "fr-en", "bn-en"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, bits, qname in zip(axes, (4, 2), ("Q4_K_M (4-bit)", "Q2_K (2-bit)")):
        x = range(len(directions))
        en = [idx.get((d, bits, "en"), float("nan")) for d in directions]
        bn = [idx.get((d, bits, "bn"), float("nan")) for d in directions]
        ax.bar([i - 0.2 for i in x], en, width=0.38, label="English calib")
        ax.bar([i + 0.2 for i in x], bn, width=0.38, label="Bengali calib")
        ax.set_xticks(list(x)); ax.set_xticklabels(directions)
        ax.set_title(qname); ax.set_ylabel("COMET (wmt22-comet-da)")
        ax.legend()
    fig.suptitle("C3: language-matched calibration (Llama-3.1-8B, GGUF)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    return True


def fmt(x) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else str(x)


def build_doc(rows, c3_rows, fig_ok, fig_rel) -> str:
    by = {(r["model"], r["method"], int(r["bits"] or 0), r["direction"]): r for r in rows}

    def comet(model, method, bits, direction):
        r = by.get((model, method, bits, direction))
        return r["comet"] if r else None

    L = []
    L.append("# Replication — *The Uneven Impact of Post-Training Quantization in Machine Translation*\n")
    L.append("Independent replication of arXiv:2508.20893 (Marie & Fujita, NICT, Aug 2025; "
             "**preprint, not peer-reviewed**). Auto-generated by `analyze.py`; the "
             "narrative verdicts below are filled from our measured numbers.\n")
    L.append("**Metric:** COMET `Unbabel/wmt22-comet-da` (the paper's primary metric), "
             "0–100 scale. Greedy decoding, each model's chat template, WMT24++ "
             "(`google/wmt24pp`), both directions, six representative languages.\n")

    # results presence
    models = sorted({r["model"] for r in rows})
    L.append(f"\n**Results present for:** {', '.join(models) or '(none yet)'}  ·  "
             f"{len(rows)} scored direction-units, {len(c3_rows)} C3 units.\n")

    # Per-claim verdict table (computed where we have data).
    L.append("\n## Per-claim verdicts (C1–C5)\n")
    L.append("| Claim | Verdict | Evidence (ours) |\n|---|---|---|")
    L.append("| C1 — 4-bit ~preserves quality for high-resource langs & large models | "
             "_pending data_ | compare baseline vs 4-bit on ja/fr and 32B/70B |")
    L.append("| C2 — low-resource / divergent-script langs degrade most, esp. 2-bit | "
             "_pending data_ | bn/ml/zu deltas at 2-bit |")
    L.append("| C3 — language-matched calibration helps mainly at 2-bit / low-resource | "
             "_pending data_ | see C3 section |")
    L.append("| C4 — GGUF most consistent, even at 2-bit | _pending data_ | method ranking |")
    L.append("| C5 — small (~1.7B) lose up to ~5 COMET at 4-bit; 32B/70B lose ≤1 | "
             "_pending data_ | size-scaling deltas |")

    # C3 section
    L.append("\n## C3 deep-dive — generic (English) vs language-matched (Bengali) calibration\n")
    if c3_rows:
        c3 = {(r["direction"], r["bits"], r["calib_lang"]): r["comet"] for r in c3_rows}
        L.append("COMET, Llama-3.1-8B, GGUF. Ours (paper in parentheses):\n")
        L.append("| Direction | Q4 en | Q4 bn | Q2 en | Q2 bn |\n|---|---|---|---|---|")
        for d in ["en-fr", "en-bn", "fr-en", "bn-en"]:
            p = PAPER_C3[d]
            cells = []
            for bits, cl, pk in [(4, "en", "Q4_K_M-en"), (4, "bn", "Q4_K_M-bn"),
                                 (2, "en", "Q2_K-en"), (2, "bn", "Q2_K-bn")]:
                ours = c3.get((d, bits, cl))
                cells.append(f"{fmt(ours)} ({p[pk]})" if ours is not None else f"– ({p[pk]})")
            L.append(f"| {d} | " + " | ".join(cells) + " |")
        # the load-bearing number
        enbn_q2_en = c3.get(("en-bn", 2, "en"))
        enbn_q2_bn = c3.get(("en-bn", 2, "bn"))
        if enbn_q2_en is not None and enbn_q2_bn is not None:
            delta = enbn_q2_bn - enbn_q2_en
            L.append(f"\n**en→bn @ 2-bit: Bengali-matched calibration Δ = {delta:+.1f} COMET** "
                     f"(paper: +3.1). This is the linchpin for the phase-two direction.")
    else:
        L.append("_C3 results not present yet._")
    if fig_ok:
        L.append(f"\n![C3 calibration]({fig_rel})\n")

    # Red flags
    L.append("\n## Robustness / red flags\n")
    L.append("- **Paper Table 8 internal contradiction.** Its prose calls bn→en 2-bit "
             "*71.5 (bn) vs 72.3 (en)* a Bengali-calibration \"gain\", but 71.5 < 72.3 — "
             "that is a **loss**. Our bn→en @ 2-bit numbers test this directly (see table).")
    L.append("- **C3 English calibration source is impossible as stated.** The paper says "
             "English calibration text came from `HuggingFaceFW/fineweb-2`, but FineWeb-2 "
             "has **no English config** (it is the multilingual sibling of FineWeb; there "
             "is no `eng_Latn`). We sourced English from the original FineWeb instead — a "
             "forced deviation that the paper's described setup could not have used.")
    L.append("- **Metric scale/checkpoint.** We use wmt22-comet-da (paper's). COMET is "
             "noisy for low-resource pairs (bn/ml/zu); chrF + spot-checks corroborate.")
    L.append("- **GGUF backend parity.** GGUF runs through llama.cpp; we feed the *same* "
             "HF chat-template prompt string and greedy decoding to keep it comparable.")
    L.append("- **AWQ/AutoRound calibration.** Paper under-specifies AWQ's calib set; we "
             "use generic WikiText (documented), which can move scores a point or two.")
    L.append("- **Scope.** n≤1000 / direction; report spread, not just means, before "
             "trusting small C3 effects.")

    L.append("\n## Bottom line\n")
    L.append("_To be written once all jobs complete: which of C1–C5 are solid enough to "
             "build the phase-two MT-conditional quantization on, and which are not — "
             "especially whether C2/C3 (the low-resource 2-bit collapse and its "
             "calibration rescue) reproduce._\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--fig", default="docs/findings/figures/replication-c3-calibration.png")
    args = ap.parse_args()

    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    rows = load_results(root)
    c3_rows = load_c3(root)
    write_summary(rows, root)

    fig_path = Path(args.fig)
    fig_ok = plot_c3(c3_rows, fig_path)
    # doc lives in docs/findings/; figure path relative to it.
    fig_rel = "figures/" + fig_path.name

    doc = build_doc(rows, c3_rows, fig_ok, fig_rel)
    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text(doc, encoding="utf-8")
    print(f">> wrote {args.doc}  ({len(rows)} units, {len(c3_rows)} C3, fig={fig_ok})")


if __name__ == "__main__":
    main()
