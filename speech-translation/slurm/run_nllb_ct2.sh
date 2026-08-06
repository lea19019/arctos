#!/usr/bin/env bash
#SBATCH --job-name=arctos-nllb-ct2
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/nllb_ct2_%j.out
#SBATCH --error=logs/nllb_ct2_%j.err
#
# NLLB ct2_int8 variant only — CTranslate2 INT8 with fused CUDA kernels.
# Prerequisites: ct2_int8 model must exist at speech-translation/results/nllb/ct2_int8/
# (already converted on login node before submitting).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OPENSSL_CONF=/dev/null
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "===== NLLB ct2_int8 experiment ====="
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date
mkdir -p logs speech-translation/results/nllb_ct2

speech-translation/.venv/bin/python -u speech-translation/nllb_experiment.py \
    --config    speech-translation/configs/nllb.yaml \
    --output    speech-translation/results/nllb_ct2 \
    --data-dir  speech-translation/data \
    --n-examples 100 \
    --variants  ct2_int8

echo "===== done $(date) ====="
