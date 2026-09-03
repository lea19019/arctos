#!/bin/bash --login
#SBATCH --job-name=sw-encdec
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/encdec_%A_%a.out
#SBATCH --error=logs/encdec_%A_%a.out
#
# RQ2 encoder-decoder arm: NLLB-200 detection + ablation (src/encdec_sw.py).
#   sbatch --array=0-1 slurm/encdec.sh
set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src TOKENIZERS_PARALLELISM=false
MODELS=(facebook/nllb-200-distilled-600M facebook/nllb-200-3.3B)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
OUT_DIR=${OUT_DIR:-results/encdec_v6}
mkdir -p "$OUT_DIR"
echo "=== ${MODEL} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python src/encdec_sw.py --model "$MODEL" --null-n "${NULL_N:-50}" --sa-null-n "${SA_NULL_N:-10}" \
    --seed 0 --out "$OUT_DIR/${MODEL//\//_}_encdec.json"
