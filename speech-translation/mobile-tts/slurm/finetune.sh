#!/bin/bash
#SBATCH --job-name=mts-finetune
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=%x_%j.out

echo "=== MMS-TTS Fine-tuning Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

PYTHON=/usr/bin/python3

echo "--- GPU Diagnostics ---"
nvidia-smi
$PYTHON -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
echo "-----------------------"

cd "$(git rev-parse --show-toplevel)/speech-translation/mobile-tts"

echo "--- Data Preparation ---"
$PYTHON data/prep_swahili.py
echo "--- Data prep complete: $(date) ---"

echo "--- Starting Fine-tuning ---"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON scripts/finetune.py \
    --train-csv data/spk7_train.csv \
    --eval-csv data/spk7_eval.csv \
    --output-dir /home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/mobile-tts-checkpoints \
    --epochs 30 \
    --batch-size 8 \
    --grad-accum-steps 2

echo "=== Done: $(date) ==="
