#!/bin/bash --login
#SBATCH --job-name=sw-conc
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/conc_%A_%a.out
#SBATCH --error=logs/conc_%A_%a.out
#
# Concentration curve: how many of the onset layer's weights carry the super
# activation? Zero the top-k contributors (k=1..256) and also remove the
# activation directly. Runs on an EXISTING detection JSON, with or without
# candidates -- this is the outcome for models where the detector finds a
# persistent massive activation but no <=4-weight carrier.
#   CAND_DIR=results/small_v6 OUT_DIR=results/conc_v6 MODELS_ATTR=SMALL sbatch --array=0-3 slurm/conc.sh
set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src TOKENIZERS_PARALLELISM=false
CAND_DIR=${CAND_DIR:-results/v5}; OUT_DIR=${OUT_DIR:-results/conc_v6}; mkdir -p "$OUT_DIR"
MODEL=$(.venv/bin/python -c "import sw_models; print(getattr(sw_models, '${MODELS_ATTR:-MODELS}')[${SLURM_ARRAY_TASK_ID}])")
SLUG=${MODEL//\//_}
echo "=== ${MODEL} on $(hostname); candidates ${CAND_DIR} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python src/ablate_sw.py --model "$MODEL" --candidates "$CAND_DIR/${SLUG}_found.json" \
    --no-table2 --concentration --sa-remove --sa-null-n 5 --seed 0 \
    --out "$OUT_DIR/${SLUG}_ablation.json"
