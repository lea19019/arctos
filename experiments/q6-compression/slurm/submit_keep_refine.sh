#!/usr/bin/env bash
# Fire the KEEP-at-3-bit refinement for all (or selected) models.
# Run AFTER the main q6 sweep finishes (separate output dir, no overwrite).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ORDER=(aya-expanse-8b llama-3.1-8b-instruct eurollm-9b-instruct tower-instruct-7b-v0.2 \
       tower-base-7b-v0.1 tower-plus-9b gemma-3-12b-it bloom-7b1)
models=("${ORDER[@]}"); [ "$#" -gt 0 ] && models=("$@")
for m in "${models[@]}"; do
  jid=$(sbatch --parsable --job-name="$m" experiments/q6-compression/slurm/run_q6keep.sh "$m")
  echo "submitted q6keep $m -> $jid"
done
