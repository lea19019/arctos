#!/bin/bash
#SBATCH --job-name=mts-gpu-test
#SBATCH --qos=gpu
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=cs
#SBATCH --output=%x_%j.out

echo "=== MMS-TTS GPU Inference Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

PYTHON=/usr/bin/python3

echo "--- GPU Info ---"
nvidia-smi
echo "----------------"

cd "$(git rev-parse --show-toplevel)/speech-translation/mobile-tts"

$PYTHON scripts/test_inference.py \
    --device cuda \
    --model-cache /home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/models/mms-tts-swh \
    --output-dir outputs/

echo "=== Done: $(date) ==="
