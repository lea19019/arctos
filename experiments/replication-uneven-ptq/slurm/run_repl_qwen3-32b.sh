#!/usr/bin/env bash
#SBATCH --job-name=repl-qwen3-32b
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/repl_qwen3-32b_%j.out
#SBATCH --error=logs/repl_qwen3-32b_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LLAMA_DIR="$HOME/llama.cpp"
export LD_LIBRARY_PATH="/apps/cudatoolkit/12.8.1/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/include/python3.11:${CPATH:-}" C_INCLUDE_PATH="$HOME/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/include/python3.11:${C_INCLUDE_PATH:-}"  # Triton JIT needs Python.h
echo "===== job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs
.venv/bin/python -u experiments/replication-uneven-ptq/experiment.py \
    --config experiments/replication-uneven-ptq/configs/qwen3-32b.yaml \
    --output results/replication-uneven-ptq/qwen3-32b
echo "===== done $(date) ====="
