#!/bin/bash --login
#SBATCH --job-name=sw-probe
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.out
#
# Smoke test for the GPU path, on the one model whose answer we already know.
# OLMo-1B was run on CPU on 2026-09-02 (notes.md): the detector finds a super
# weight at L1[1764,1710] and ablating it must be CATASTROPHIC, while the
# paper's second coordinate L1[1764,8041] holds ~0.0018 and must do nothing.
# If the GPU run reproduces that, the sweep is safe to launch.

set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"

export HF_HUB_OFFLINE=1          # compute nodes have no internet: fail fast
export HF_DATASETS_OFFLINE=1     # ... same for the wikitext-2 eval corpus
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

MODEL=allenai/OLMo-1B-0724-hf
SLUG=${MODEL//\//_}
OUT=${OUT_DIR:-results/probe}

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python "${DETECTOR:-src/detect_sw.py}" --model "$MODEL" --out "$OUT/${SLUG}_found.json"
.venv/bin/python src/ablate_sw.py  --model "$MODEL" \
    --candidates "$OUT/${SLUG}_found.json" --out "$OUT/${SLUG}_ablation.json" \
    --joint --null-n 5 --sa-remove --sa-null-n 2 --seed 0

# ---- assert the known-good result, so a silent wrong answer fails the probe
# (src/probe_check.py: every quantity bounded on both sides, CIs + null)
.venv/bin/python src/probe_check.py "$OUT/${SLUG}_ablation.json" "$OUT/${SLUG}_found.json"
