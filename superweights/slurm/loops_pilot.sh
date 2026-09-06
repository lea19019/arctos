#!/bin/bash --login
#SBATCH --job-name=loops-pilot
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/loops_%A_%a.out
#SBATCH --error=logs/loops_%A_%a.out
#
# Natural-repetition pilot: one array task per config in configs/loops_pilot/.
# No partition is requested (BYU ORC agent rules); --qos=cs places the job on
# the A100/H100/B200 partitions that accept that QOS. See ../BYU_ORC_AGENTS.md.
#   sbatch --array=0-5 slurm/loops_pilot.sh
set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
mkdir -p logs
CONFIGS=($(ls configs/loops_pilot/*.yaml | sort))
CFG=${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0} config $CFG on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
.venv/bin/python src/gen_loops.py --config "$CFG"
