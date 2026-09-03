#!/bin/bash --login
#SBATCH --job-name=sw-reablate
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reablate_%A_%a.out
#SBATCH --error=logs/reablate_%A_%a.out
#
# Re-ablate an EXISTING candidate set under a different eval corpus.
#
# Detection is corpus-independent -- it is one forward pass on one prompt --
# so changing the damage metric does not require re-detecting. This lets a
# detector version and an eval corpus be varied one at a time instead of
# together, which is how v3 got confounded.
#
#   CAND_DIR=results/v1 OUT_DIR=results/v4 sbatch --array=0-8 slurm/reablate.sh

set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

CAND_DIR=${CAND_DIR:-results/v1}
OUT_DIR=${OUT_DIR:-results/v4}
CORPUS=${CORPUS:-wikitext2}
mkdir -p "$OUT_DIR"

MODEL=$(.venv/bin/python -c \
  "from sw_models import MODELS; print(MODELS[${SLURM_ARRAY_TASK_ID}])")
SLUG=${MODEL//\//_}

echo "=== ${MODEL}: candidates from ${CAND_DIR}, corpus ${CORPUS} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python src/ablate_sw.py --model "$MODEL" \
    --candidates "$CAND_DIR/${SLUG}_found.json" \
    --eval-corpus "$CORPUS" \
    --out "$OUT_DIR/${SLUG}_ablation.json"
