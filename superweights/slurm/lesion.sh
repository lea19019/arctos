#!/bin/bash --login
#SBATCH --job-name=lesion
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lesion_%A_%a.out
#SBATCH --error=logs/lesion_%A_%a.out
#
# Control triple + dose sweep on a known coordinate: one array task per config
# in configs/lesion/. OLMo-1B is the smoke test (x3667 known).
#   sbatch --array=0-1 slurm/lesion.sh
set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
mkdir -p logs
CONFIGS=($(ls configs/lesion/*.yaml | sort))
CFG=${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0} config $CFG on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
.venv/bin/python src/lesion_controls.py --config "$CFG"
