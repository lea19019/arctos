#!/usr/bin/env bash
#SBATCH --job-name=repl-analyze
#SBATCH --account=sdrich
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/repl_analyze_%j.out
#SBATCH --error=logs/repl_analyze_%j.err
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
export OPENSSL_CONF=/dev/null TOKENIZERS_PARALLELISM=false
echo "===== job $SLURM_JOB_ID on $(hostname) ====="; date; mkdir -p logs
.venv/bin/python -u experiments/replication-uneven-ptq/analyze.py \
    --results results/replication-uneven-ptq \
    --doc docs/findings/replication-uneven-ptq-mt.md
echo "===== done $(date) ====="
