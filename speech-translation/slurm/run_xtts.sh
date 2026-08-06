#!/usr/bin/env bash
#SBATCH --job-name=arctos-xtts-compress
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/xtts_compress_%j.out
#SBATCH --error=logs/xtts_compress_%j.err
#
# XTTS v2 compression experiment: FP16 baseline vs BnB INT8 on GPT component.
# Languages: English, Spanish, French (50 FLORES+ sentences each).
# Metrics: CER, WER (via Whisper-medium back-transcription), RTF.
# Prerequisites: run speech-translation/slurm/precache.sh on login node first.
#
# Submit from repo root: sbatch speech-translation/slurm/run_xtts.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OPENSSL_CONF=/dev/null
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# TTS library auto-download should not run on compute nodes (no internet).
# The checkpoint must already be at ~/.local/share/tts/ from precache.sh.
export COQUI_TOS_AGREED=1

echo "===== XTTS v2 compression experiment ====="
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi | head -20 || true
date
mkdir -p logs speech-translation/results/xtts

# ---------- Experiment ----------
speech-translation/.venv/bin/python -u speech-translation/xtts_experiment.py \
    --config   speech-translation/configs/xtts.yaml \
    --output   speech-translation/results/xtts \
    --data-dir speech-translation/data \
    --n-examples 50

echo "===== done $(date) ====="
