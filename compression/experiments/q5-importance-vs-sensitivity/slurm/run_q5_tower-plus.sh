#!/usr/bin/env bash
#SBATCH --job-name=arctos-q5-tower-plus
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/q5_tower-plus_%j.out
#SBATCH --error=logs/q5_tower-plus_%j.err
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs results
.venv/bin/python -u experiments/q5-importance-vs-sensitivity/experiment.py \
    --config experiments/q1-language-emergence/configs/tower-plus.yaml \
    --q1-results results/tower-plus-9b/q1 \
    --output results/tower-plus-9b/q5
echo "===== done $(date) ====="
