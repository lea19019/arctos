#!/usr/bin/env bash
#SBATCH --job-name=arctos-q5q-bloom
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/q5q_bloom_%j.out
#SBATCH --error=logs/q5q_bloom_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs results
.venv/bin/python -u experiments/q5-importance-vs-sensitivity/quality_sensitivity.py \
    --config experiments/q1-language-emergence/configs/bloom.yaml \
    --q1-results results/bloom-7b1/q1 --output results/bloom-7b1/q5 \
    --n-examples 20 --max-new-tokens 48 --sigma 0.1
echo "===== done $(date) ====="
