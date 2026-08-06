#!/usr/bin/env bash
#SBATCH --job-name=arctos-q1-tower-base
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/q1_tower_base_%j.out
#SBATCH --error=logs/q1_tower_base_%j.err

set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"

export OPENSSL_CONF=/dev/null
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "===== job $SLURM_JOB_ID on $(hostname) ====="
nvidia-smi || true
date

mkdir -p logs results

.venv/bin/python -u experiments/q1-language-emergence/experiment.py \
    --config experiments/q1-language-emergence/configs/tower-base.yaml \
    --output results/tower-base-7b-v0.1/q1

echo "===== done $(date) ====="
