#!/usr/bin/env bash
# Submit the phase-two GEM run (low-bit W2+W3, XCOMET-XL, super-weight/salient
# preservation + MT-conditional GPTQ) across the decoder models.
# Order leads with the strong-super-weight / low-resource-sensitive models.
# Usage: bash experiments/q6-compression/slurm/submit_gem.sh [model ...]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/compression"
ORDER=(eurollm-9b-instruct tower-instruct-7b-v0.2 aya-expanse-8b llama-3.1-8b-instruct \
       tower-base-7b-v0.1 tower-plus-9b gemma-3-12b-it bloom-7b1)
models=("${ORDER[@]}"); [ "$#" -gt 0 ] && models=("$@")
for m in "${models[@]}"; do
  jid=$(sbatch --parsable --job-name="$m" experiments/q6-compression/slurm/run_q6gem.sh "$m")
  echo "submitted q6gem $m -> $jid"
done
