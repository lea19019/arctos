#!/bin/bash --login
#SBATCH --job-name=sw-sweep
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.out
#
# One array task per model in sw_models.MODELS: detect super weights, then
# ablate both the detector's finds and Yu et al. Table 2's coordinates.
#
# Sizing: the array is uniform because the largest model sets the floor --
# Llama-30B is 32.5B params in fp16 = 65 GB, which fits one A100-80GB (also
# H100-80GB on cs2, B200-180GB on cs3) with no model parallelism. --mem=160G
# covers the CPU-side load, since from_pretrained(...).to(cuda) materialises
# the whole checkpoint in host RAM first.
#
# OUT_DIR keeps each detector generation's results side by side, so a new
# version can never overwrite the one it is being compared against.
#
#   DETECTOR=src/detect_sw_v3.py OUT_DIR=results/v3 \
#     sbatch --array=0-8 slurm/sweep.sh

set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"

export HF_HUB_OFFLINE=1          # compute nodes have no internet: fail fast
export HF_DATASETS_OFFLINE=1     # ... same for the wikitext-2 eval corpus
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

# Single source of truth for the model list: read it out of sw_models.py
# rather than duplicating it in bash, so the two can never drift.
OUT_DIR=${OUT_DIR:-results}
DETECTOR=${DETECTOR:-src/detect_sw.py}
CORPUS=${CORPUS:-wikitext2}
EXTRA=${EXTRA:-}            # extra ablate_sw.py flags, e.g. "--joint --null-n 50 --sa-remove --sa-null-n 10"
mkdir -p "$OUT_DIR"

MODEL=$(.venv/bin/python -c \
  "import sw_models; print(getattr(sw_models, '${MODELS_ATTR:-MODELS}')[${SLURM_ARRAY_TASK_ID}])")
SLUG=${MODEL//\//_}

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${MODEL} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python "$DETECTOR" --model "$MODEL" \
    --out "$OUT_DIR/${SLUG}_found.json"
.venv/bin/python src/ablate_sw.py --model "$MODEL" \
    --candidates "$OUT_DIR/${SLUG}_found.json" \
    --eval-corpus "$CORPUS" $EXTRA \
    --out "$OUT_DIR/${SLUG}_ablation.json"

echo "=== task ${SLURM_ARRAY_TASK_ID} done: ${MODEL} ==="
