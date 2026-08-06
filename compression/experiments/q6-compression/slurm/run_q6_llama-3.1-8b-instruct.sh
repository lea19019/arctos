#!/usr/bin/env bash
#SBATCH --job-name=arctos-q6-llama
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/q6_llama-3.1-8b-instruct_%j.out
#SBATCH --error=logs/q6_llama-3.1-8b-instruct_%j.err
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs results
.venv/bin/python -u experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/llama-3.1-8b-instruct.yaml \
    --output results/llama-3.1-8b-instruct/q6 \
    --stages find shrink keep prune calib \
    --n-examples 24 --max-new-tokens 48 --calib-n 48 \
    --bits 4 3 2 --group-size 128 --sparsities 0.25 0.5 \
    --keep-bits 3 --awq-alphas 0.25 0.5 1.0 --calib-bits 4 3 --calib-sparsity 0.5
echo "===== done $(date) ====="
