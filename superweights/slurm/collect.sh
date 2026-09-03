#!/bin/bash --login
#SBATCH --job-name=sw-collect
#SBATCH --partition=m9
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.out
#
# Reads the JSON the array wrote and prints the one combined table: every
# coordinate tested, whether it came from our detector or Table 2 or both,
# and what ablation says. No GPU needed. Submitted with
# --dependency=afterany so it still reports if some models failed.

set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"
export PYTHONPATH=src
.venv/bin/python src/run_all.py --summary-only | tee results/summary.txt
