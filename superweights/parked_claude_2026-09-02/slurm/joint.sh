#!/bin/bash --login
#SBATCH --job-name=sw-joint
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/joint_%A_%a.out
#SBATCH --error=logs/joint_%A_%a.out
#
# experiments/joint_ablation: re-score an EXISTING candidate set with
#   * individual ablation (as before, now with bootstrap CIs),
#   * joint ablation of the whole set + leave-one-out,
#   * a magnitude-matched null (random top-100 weights, individual and joint),
#   * direct super-activation removal at the onset layer (Sun et al. 2024).
#
#   CAND_DIR=results/v5     OUT_DIR=results/v6        sbatch --array=0-8 slurm/joint.sh
#   CAND_DIR=results/modern OUT_DIR=results/modern_v6 MODELS_ATTR=MODERN sbatch --array=0-2 slurm/joint.sh

set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

CAND_DIR=${CAND_DIR:-results/v5}
OUT_DIR=${OUT_DIR:-results/v6}
CORPUS=${CORPUS:-wikitext2}
NULL_N=${NULL_N:-50}
SA_NULL_N=${SA_NULL_N:-10}
EXTRA=${EXTRA:-}
mkdir -p "$OUT_DIR"

MODEL=$(.venv/bin/python -c \
  "import sw_models; print(getattr(sw_models, '${MODELS_ATTR:-MODELS}')[${SLURM_ARRAY_TASK_ID}])")
SLUG=${MODEL//\//_}

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${MODEL} on $(hostname); candidates ${CAND_DIR}; out ${OUT_DIR} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python src/ablate_sw.py --model "$MODEL" \
    --candidates "$CAND_DIR/${SLUG}_found.json" \
    --eval-corpus "$CORPUS" \
    --joint --null-n "$NULL_N" --sa-remove --sa-null-n "$SA_NULL_N" --seed 0 \
    $EXTRA \
    --out "$OUT_DIR/${SLUG}_ablation.json"

echo "=== task ${SLURM_ARRAY_TASK_ID} done: ${MODEL} ==="
