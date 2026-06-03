#!/usr/bin/env bash
#SBATCH --job-name=q6canary
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/q6canary_%x_%j.out
#SBATCH --error=logs/q6canary_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${1:?model}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== CANARY $MODEL : job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; date; mkdir -p logs results
.venv/bin/python -u experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/${MODEL}.yaml \
    --output results/_canary/${MODEL}/q6 \
    --stages find shrink keep prune calib \
    --n-examples 4 --max-new-tokens 16 --calib-n 24 \
    --bits 4 3 2 --group-size 128 --sparsities 0.5 \
    --keep-bits 3 --awq-alphas 0.5 --calib-bits 3 --calib-sparsity 0.5
echo "===== CANARY $MODEL done $(date) ====="
