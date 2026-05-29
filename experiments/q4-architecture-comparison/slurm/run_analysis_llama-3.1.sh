#!/usr/bin/env bash
#SBATCH --job-name=arctos-an-llama-3.1
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/analysis_llama-3.1_%j.out
#SBATCH --error=logs/analysis_llama-3.1_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs results
echo "--- pivot trajectory ---"
.venv/bin/python -u experiments/q1-language-emergence/pivot_runner.py \
    --config experiments/q1-language-emergence/configs/llama-3.1.yaml \
    --output results/llama-3.1-8b-instruct/pivot --n-examples 100 --pairs en-zh en-arz cs-de
echo "--- attention viz ---"
.venv/bin/python -u experiments/q2-attention-heads/attention_viz.py \
    --config experiments/q1-language-emergence/configs/llama-3.1.yaml \
    --q1-results results/llama-3.1-8b-instruct/q1 --output results/llama-3.1-8b-instruct/q2/attention --top-k 6 || echo "attn-viz failed (non-fatal)"
echo "===== done $(date) ====="
