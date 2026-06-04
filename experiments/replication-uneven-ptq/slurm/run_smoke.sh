#!/usr/bin/env bash
#SBATCH --job-name=repl-smoke
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/repl_smoke_%j.out
#SBATCH --error=logs/repl_smoke_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LLAMA_DIR="$HOME/llama.cpp"
export LD_LIBRARY_PATH="/apps/cudatoolkit/12.8.1/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/include/python3.11:${CPATH:-}" C_INCLUDE_PATH="$HOME/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/include/python3.11:${C_INCLUDE_PATH:-}"  # Triton JIT needs Python.h
echo "===== SMOKE job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs
# Full path on the smallest model, tiny n, two low-resource directions: exercises
# every backend (BnB load, AWQ+shim, AutoRound, GGUF convert/imatrix/quantize/serve),
# COMET scoring, JSON output, and artifact deletion.
.venv/bin/python -u experiments/replication-uneven-ptq/experiment.py \
    --config experiments/replication-uneven-ptq/configs/_smoke.yaml \
    --output results/replication-uneven-ptq/_smoke \
    --directions en-bn bn-en \
    --n 8 --calib-n 128 --imatrix-n 256
echo "===== smoke done $(date) ====="
ls -R results/replication-uneven-ptq/_smoke
