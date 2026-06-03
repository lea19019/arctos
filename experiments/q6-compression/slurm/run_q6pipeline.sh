#!/usr/bin/env bash
#SBATCH --job-name=q6pipe
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/q6pipe_%x_%j.out
#SBATCH --error=logs/q6pipe_%x_%j.err
#
# Decisive test of the pipeline-aware hypothesis: crush the language-neutral
# MIDDLE vs crush the language-specific ENDS at matched budget (+ super-weight),
# scored with XCOMET-XL. Writes results/<model>/q6pipe/.
# Usage: sbatch --job-name=<model> run_q6pipeline.sh <model> [canary]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${1:?model}"; MODE="${2:-full}"
export OPENSSL_CONF=/dev/null HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "===== q6pipe $MODEL ($MODE) $SLURM_JOB_ID on $(hostname) ====="; nvidia-smi --query-gpu=name --format=csv,noheader || true; date
if [ "$MODE" = "canary" ]; then N=4; MNT=16; CN=12; OUT="results/_canary/pipe/${MODEL}/q6"; LV="2 ternary"
else N=32; MNT=48; CN=48; OUT="results/${MODEL}/q6pipe"; LV="4 3 2 ternary binary"; fi
.venv/bin/python -u experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/${MODEL}.yaml \
    --output "$OUT" --stages find pipeline \
    --n-examples $N --max-new-tokens $MNT --calib-n $CN --group-size 128 \
    --pipe-levels $LV --use-comet --comet-gpus 1
echo "===== q6pipe $MODEL done $(date) ====="
