#!/usr/bin/env bash
#SBATCH --job-name=arctos-q5-gemma3
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/q5_gemma3_%j.out
#SBATCH --error=logs/q5_gemma3_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs results
.venv/bin/python -u experiments/q5-importance-vs-sensitivity/experiment.py \
    --config experiments/q1-language-emergence/configs/gemma3.yaml \
    --q1-results results/gemma-3-12b-it/q1 \
    --output results/gemma-3-12b-it/q5
echo "===== done $(date) ====="
