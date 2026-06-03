#!/usr/bin/env bash
#SBATCH --job-name=q6gem
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/q6gem_%x_%j.out
#SBATCH --error=logs/q6gem_%x_%j.err
#
# Phase-two GEM run (docs/findings/phase2-method-primer.md): the low-bit (W2+W3),
# MT-quality (XCOMET-XL) test of the verified novelty — multilingual super-weight
# + salient-channel FP16 preservation, MT-conditional GPTQ, per-pair (en-arz
# foregrounded). Writes results/<model>/q6gem/ (does NOT touch the q6 sweep).
#
# Usage: sbatch --job-name=<model> experiments/q6-compression/slurm/run_q6gem.sh <model>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${1:?model}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== q6gem $MODEL : job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; date; mkdir -p logs results
.venv/bin/python -u experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/${MODEL}.yaml \
    --output results/${MODEL}/q6gem \
    --stages find keep gptq calib alloc \
    --n-examples 32 --max-new-tokens 48 --calib-n 48 \
    --bits 4 3 2 --group-size 128 \
    --keep-bits 3 2 --awq-alphas 0.5 \
    --gptq-bits 3 2 --calib-bits 3 2 --calib-sparsity 0.5 \
    --alloc-avg-bits 3 \
    --use-comet --comet-gpus 1
echo "===== q6gem $MODEL done $(date) ====="
