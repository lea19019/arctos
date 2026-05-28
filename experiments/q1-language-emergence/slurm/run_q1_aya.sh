#!/usr/bin/env bash
#SBATCH --job-name=arctos-q1-aya
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/q1_aya_%j.out
#SBATCH --error=logs/q1_aya_%j.err
#
# Q1 (language emergence) — Aya Expanse 8B across the 3 calibration pairs.
#
# Submit from repo root:
#     sbatch experiments/q1-language-emergence/slurm/run_q1_aya.sh
#
# Prerequisites:
#   - data/{cs-de,en-zh,en-arz}.jsonl already populated
#     (run scripts/fetch_flores.py from the login node beforehand)
#   - Aya cached at ~/.cache/huggingface/hub/models--CohereForAI--aya-expanse-8b

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Cluster OpenSSL workaround — see env commit message.
export OPENSSL_CONF=/dev/null
# Pin HF cache to the user home so the compute node hits the shared FS.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Compute nodes have no internet; force offline once weights are cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "===== job $SLURM_JOB_ID on $(hostname) ====="
nvidia-smi || true
date

mkdir -p logs results

.venv/bin/python -u experiments/q1-language-emergence/experiment.py \
    --config experiments/q1-language-emergence/configs/aya.yaml \
    --output results/aya-expanse-8b/q1

echo "===== done $(date) ====="
