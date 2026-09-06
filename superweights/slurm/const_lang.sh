#!/bin/bash --login
#SBATCH --job-name=const-lang
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/constlang_%A_%a.out
#SBATCH --error=logs/constlang_%A_%a.out
# Is the built-in constant the same for every input language? One array task
# per config in configs/const_lang/.   sbatch --array=0-6 slurm/const_lang.sh
set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
mkdir -p logs
CONFIGS=($(ls configs/const_lang/*.yaml | sort))
CFG=${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0} config $CFG on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
.venv/bin/python src/constant_by_language.py --config "$CFG"
