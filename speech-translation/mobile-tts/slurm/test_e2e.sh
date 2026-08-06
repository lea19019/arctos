#!/bin/bash
#SBATCH --job-name=mts-e2e-test
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out

echo "=== MMS-TTS End-to-End Test (posterior encoder path) ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Date: $(date)"

PYTHON=/usr/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi | grep -E "GPU Name|MiB"
$PYTHON -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

cd "$(git rev-parse --show-toplevel)/speech-translation/mobile-tts"

# Run 3 epochs, eval every 50 steps, save every 50 steps — covers train + eval + checkpoint
$PYTHON scripts/finetune.py \
    --train-csv data/spk7_train.csv \
    --eval-csv  data/spk7_eval.csv \
    --output-dir /tmp/mts-e2e-ckpt \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --save-steps 50 \
    --warmup-steps 20 \
    --max-duration 6.0

echo "=== Done: $(date) ==="
