#!/usr/bin/env bash
#SBATCH --job-name=q6canary-gptq
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/q6canary_gptq_%j.out
#SBATCH --error=logs/q6canary_gptq_%j.err
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== GPTQ/ALLOC CANARY $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; date
.venv/bin/python -u experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/bloom-7b1.yaml \
    --output results/_canary/gptq/q6 \
    --stages find gptq alloc \
    --n-examples 4 --max-new-tokens 16 --calib-n 16 \
    --gptq-bits 4 3 --alloc-avg-bits 3 --group-size 128 \
    --use-comet --comet-gpus 1
echo "===== CANARY done $(date) ====="
