#!/usr/bin/env bash
#SBATCH --job-name=arctos-nllb-compress
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/nllb_compress_%j.out
#SBATCH --error=logs/nllb_compress_%j.err
#
# NLLB-200 compression experiment: FP16 / BnB INT8 / BnB NF4 / CT2 INT8.
# Languages: English, Spanish, French (three FLORES+ pairs).
# Prerequisites: run speech-translation/slurm/precache.sh on login node first.
#
# Submit from repo root: sbatch speech-translation/slurm/run_nllb.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OPENSSL_CONF=/dev/null
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "===== NLLB compression experiment ====="
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi | head -20 || true
date
mkdir -p logs speech-translation/results/nllb

# ---------- Experiment ----------
speech-translation/.venv/bin/python -u speech-translation/nllb_experiment.py \
    --config   speech-translation/configs/nllb.yaml \
    --output   speech-translation/results/nllb \
    --data-dir speech-translation/data \
    --n-examples 100

echo "===== done $(date) ====="

# If you want to run with the 3.3B model instead of distilled-600M,
# edit nllb.yaml and resubmit (add ~2h runtime).
