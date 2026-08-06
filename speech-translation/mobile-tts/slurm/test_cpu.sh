#!/bin/bash
#SBATCH --job-name=mts-cpu-test
#SBATCH --qos=normal
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out

echo "=== MMS-TTS CPU Inference Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

PYTHON=/usr/bin/python3

cd "$(git rev-parse --show-toplevel)/speech-translation/mobile-tts"

$PYTHON scripts/test_inference.py \
    --device cpu \
    --model-cache /home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/models/mms-tts-swh \
    --output-dir outputs/

echo "=== Done: $(date) ==="
