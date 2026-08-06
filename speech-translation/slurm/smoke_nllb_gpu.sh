#!/usr/bin/env bash
#SBATCH --job-name=arctos-smoke-nllb
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/smoke_nllb_%j.out
#SBATCH --error=logs/smoke_nllb_%j.err
#
# GPU smoke test for nllb_experiment.py
# Runs: fp16 + bnb_int8 variants, 5 examples per pair, all 3 pairs, XCOMET-XL on.
# Expected runtime: ~10-15 minutes on A100.
#
# Submit from repo root: sbatch speech-translation/slurm/smoke_nllb_gpu.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OPENSSL_CONF=/dev/null
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "===== GPU smoke test: NLLB compression ====="
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date
mkdir -p logs speech-translation/results/nllb_smoke_gpu

speech-translation/.venv/bin/python -u speech-translation/nllb_experiment.py \
    --config     speech-translation/configs/nllb.yaml \
    --output     speech-translation/results/nllb_smoke_gpu \
    --data-dir   speech-translation/data \
    --n-examples 5 \
    --n-warmup   2 \
    --variants   fp16 bnb_int8

echo "===== done $(date) ====="
echo ""
echo "Results:"
cat speech-translation/results/nllb_smoke_gpu/summary.tsv
