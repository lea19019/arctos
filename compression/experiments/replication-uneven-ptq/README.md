# Replication: *The Uneven Impact of PTQ in Machine Translation*

Independent, critical replication of **arXiv:2508.20893** (Marie & Fujita,
NICT, Aug 2025 — a **preprint, not peer-reviewed**). Brief:
`docs/replication-uneven-ptq-mt-brief.md`. Kept deliberately separate from the
Arctos q6 work so it stays an independent check.

## What it reproduces

Paper's exact setup, scaled to our budget:

| Axis | Paper | Here |
|---|---|---|
| Models | Qwen3-1.7B/8B/32B, Llama-3.1-8B, Llama-3.3-70B | all five |
| Test set | WMT24++ (`google/wmt24pp`), 55 langs | same; 6 representative langs (ja/fr/pl/bn/ml/zu), both directions |
| Quantizers | AWQ, BnB-NF4, GGUF (Q4_K_M/Q2_K), AutoRound | same official libs |
| Bit-widths | 4-bit (all), 2-bit (GGUF+AutoRound) | same |
| Metric | COMET `wmt22-comet-da` | same (primary); +chrF, BLEU |
| Decoding | greedy, each model's chat template | same |
| C3 | Llama-3.1-8B GGUF, English-vs-Bengali imatrix | same |

## Layout

- `experiment.py` — resumable per-(method×bits×direction) driver for one model.
- `c3_calibration.py` — the calibration deep-dive (its own module).
- `precache.py` — **login-node** downloads + materializes C3 calib text.
- `build_llama_cpp.sh` — builds the GGUF binaries (login node, CUDA).
- `analyze.py` — rollup → C1–C5 verdict table + C3 plot + findings doc.
- `configs/` — one YAML per model (`hf_path`, chat flags).
- `slurm/` — per-model jobs + `submit_all.sh` (dependency-ordered).

Code lives in `src/data/wmt24pp.py`, `src/models/_chat_prompt.py`,
`src/quant/` (the official-library wrappers). Results → `results/` (gitignored);
findings → `docs/findings/replication-uneven-ptq-mt.md`.

## Run order

```bash
# 1. login node (internet): build GGUF binaries + cache everything
bash experiments/replication-uneven-ptq/build_llama_cpp.sh
.venv/bin/python experiments/replication-uneven-ptq/precache.py

# 2. quick end-to-end smoke (tiny n) — validates the whole path
sbatch experiments/replication-uneven-ptq/slurm/run_repl_qwen3-1.7b.sh   # or smoke config

# 3. full sweep (dependency-ordered: sweeps -> analyze)
bash experiments/replication-uneven-ptq/slurm/submit_all.sh
```

Every result unit is skip-if-exists, so any job killed at the SLURM time wall
just continues on resubmit.
