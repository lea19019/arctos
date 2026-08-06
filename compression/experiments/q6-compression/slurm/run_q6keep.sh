#!/usr/bin/env bash
#SBATCH --job-name=arctos-q6keep
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/q6keep_%x_%j.out
#SBATCH --error=logs/q6keep_%x_%j.err
#
# Targeted KEEP-at-the-cliff refinement: re-tests the protection schemes at the
# 3-bit cliff (where there is signal) instead of dead 2-bit, with an AWQ-alpha
# sweep. Writes to results/<model>/q6keepw3/ so it does NOT overwrite the full
# q6_summary.json. Runs only find (needed for salience/super-weights) + keep.
#
# Usage:  sbatch --job-name=<model> experiments/q6-compression/slurm/run_q6keep.sh <model>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
MODEL="${1:?pass a model name, e.g. aya-expanse-8b}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== q6refine $MODEL : job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi || true; date; mkdir -p logs results
# Phase-two refinement: KEEP at the 3-bit cliff (alpha sweep) + the CALIB
# linchpin (MT vs generic-XNLI calibration, head-to-head). Separate output dir
# so it never overwrites the full q6_summary.json from the main sweep.
.venv/bin/python -u experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/${MODEL}.yaml \
    --output results/${MODEL}/q6refine \
    --stages find keep calib \
    --n-examples 24 --max-new-tokens 48 --calib-n 96 \
    --bits 4 3 2 --keep-bits 3 2 --group-size 128 \
    --awq-alphas 0.25 0.5 1.0 --calib-bits 4 3 --calib-sparsity 0.5
echo "===== done $(date) ====="
