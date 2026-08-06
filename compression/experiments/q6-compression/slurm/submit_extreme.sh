#!/usr/bin/env bash
# Full extreme-low-bit sweep (4/3/2/ternary/binary) across decoder models.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
ORDER=(eurollm-9b-instruct tower-instruct-7b-v0.2 aya-expanse-8b llama-3.1-8b-instruct \
       tower-base-7b-v0.1 tower-plus-9b gemma-3-12b-it bloom-7b1)
models=("${ORDER[@]}"); [ "$#" -gt 0 ] && models=("$@")
for m in "${models[@]}"; do
  jid=$(sbatch --parsable --job-name="$m" experiments/q6-compression/slurm/run_q6extreme.sh "$m")
  echo "submitted q6extreme $m -> $jid"
done
