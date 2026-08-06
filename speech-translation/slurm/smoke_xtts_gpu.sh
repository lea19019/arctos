#!/usr/bin/env bash
#SBATCH --job-name=arctos-smoke-xtts
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_xtts_%j.out
#SBATCH --error=logs/smoke_xtts_%j.err
#
# GPU smoke test for xtts_experiment.py
# Runs: fp16 + bnb_int8_gpt variants, 3 sentences per language, 3 languages.
# Expected runtime: ~20-30 minutes on A100.
#
# Prerequisites (run precache.sh first to download XTTS):
#   bash speech-translation/slurm/precache.sh
#
# Submit from repo root: sbatch speech-translation/slurm/smoke_xtts_gpu.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OPENSSL_CONF=/dev/null
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export COQUI_TOS_AGREED=1

echo "===== GPU smoke test: XTTS compression ====="
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date
mkdir -p logs speech-translation/results/xtts_smoke_gpu

speech-translation/.venv/bin/python -u speech-translation/xtts_experiment.py \
    --config     speech-translation/configs/xtts.yaml \
    --output     speech-translation/results/xtts_smoke_gpu \
    --data-dir   speech-translation/data \
    --n-examples 3 \
    --n-warmup   1 \
    --variants   fp16 bnb_int8_gpt

echo "===== done $(date) ====="
echo ""
echo "Results:"
cat speech-translation/results/xtts_smoke_gpu/summary.tsv
