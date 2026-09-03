#!/bin/bash --login
#SBATCH --job-name=sw-prompts
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prompts_%A_%a.out
#SBATCH --error=logs/prompts_%A_%a.out
#
# Prompt confound: every detection so far used "Language modeling is ".
# Re-run detection only (no ablation) under three unrelated prompts and
# compare candidate sets. Array over MODELS (0-8) then MODERN (9-11).
set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src TOKENIZERS_PARALLELISM=false
OUT_DIR=${OUT_DIR:-results/prompts_v6}; mkdir -p "$OUT_DIR"
MODEL=$(.venv/bin/python -c "import sw_models; m=sw_models.MODELS+sw_models.MODERN; print(m[${SLURM_ARRAY_TASK_ID}])")
SLUG=${MODEL//\//_}
PROMPTS=("The quick brown fox jumps over the lazy dog." "In 1969, astronauts landed on the Moon for the first time" "def fibonacci(n):
    if n < 2:")
for i in 0 1 2; do
  echo "=== ${MODEL} prompt $i: ${PROMPTS[$i]}"
  .venv/bin/python src/detect_sw.py --model "$MODEL" --prompt "${PROMPTS[$i]}" --out "$OUT_DIR/${SLUG}_p${i}_found.json"
done
