#!/usr/bin/env bash
set -euo pipefail; cd "$(git rev-parse --show-toplevel)"
ORDER=(eurollm-9b-instruct aya-expanse-8b llama-3.1-8b-instruct tower-instruct-7b-v0.2 \
       tower-base-7b-v0.1 tower-plus-9b gemma-3-12b-it bloom-7b1)
models=("${ORDER[@]}"); [ "$#" -gt 0 ] && models=("$@")
for m in "${models[@]}"; do
  echo "$(sbatch --parsable --job-name="$m" experiments/q6-compression/slurm/run_q6pipeline.sh "$m") $m"
done
