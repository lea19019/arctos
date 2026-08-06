#!/usr/bin/env bash
#SBATCH --job-name=q6extreme
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/q6extreme_%x_%j.out
#SBATCH --error=logs/q6extreme_%x_%j.err
#
# EXTREME low-bit sweep: the full degradation cliff 4 -> 3 -> 2 -> 1.58 (ternary)
# -> 1 (binary), and whether super-weight + salient-channel FP16 preservation
# RESCUES the sub-2-bit collapse for MT (XCOMET-XL, per pair, en-arz foregrounded).
# Writes results/<model>/q6extreme/.
#
# Usage: sbatch --job-name=<model> experiments/q6-compression/slurm/run_q6extreme.sh <model> [canary]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
MODEL="${1:?model}"; MODE="${2:-full}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== q6extreme $MODEL ($MODE) : job $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; date; mkdir -p logs results
if [ "$MODE" = "canary" ]; then
  N=4; MNT=16; CN=16; OUT="results/_canary/extreme/${MODEL}/q6"
else
  N=32; MNT=48; CN=48; OUT="results/${MODEL}/q6extreme"
fi
.venv/bin/python -u experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/${MODEL}.yaml \
    --output "$OUT" \
    --stages find shrink keep \
    --n-examples $N --max-new-tokens $MNT --calib-n $CN --group-size 128 \
    --levels 4 3 2 ternary binary \
    --keep-bits 2 ternary binary --awq-alphas 0.5 \
    --use-comet --comet-gpus 1
echo "===== q6extreme $MODEL done $(date) ====="
