#!/usr/bin/env bash
# Submit the full q6 (find/keep/shrink/prune) sweep across the 8 decoder models.
# Usage: bash experiments/q6-compression/slurm/submit_all.sh [model ...]
# With no args, submits all. Aya first (the WMT25 anchor model).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
ORDER=(aya-expanse-8b llama-3.1-8b-instruct bloom-7b1 eurollm-9b-instruct \
       tower-base-7b-v0.1 tower-instruct-7b-v0.2 tower-plus-9b gemma-3-12b-it)
models=("${@:-${ORDER[@]}}")
[ "$#" -gt 0 ] && models=("$@") || models=("${ORDER[@]}")
for m in "${models[@]}"; do
  s="experiments/q6-compression/slurm/run_q6_${m}.sh"
  [ -f "$s" ] || { echo "no slurm script for $m"; continue; }
  jid=$(sbatch --parsable "$s")
  echo "submitted $m -> job $jid"
done
