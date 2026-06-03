#!/usr/bin/env bash
# Q6 sweep status — run on return to see what's done, running, or failed.
#   bash scripts/q6_status.sh
# Resubmit a single failed model with:
#   bash experiments/q6-compression/slurm/submit_all.sh <model-name>
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

MODELS=(aya-expanse-8b llama-3.1-8b-instruct bloom-7b1 eurollm-9b-instruct \
        tower-base-7b-v0.1 tower-instruct-7b-v0.2 tower-plus-9b gemma-3-12b-it)

echo "==================== SLURM QUEUE ===================="
squeue -u "$USER" -o "%.10i %.22j %.9T %.8M %.10l %R" 2>/dev/null || echo "(squeue unavailable)"

echo; echo "==================== PER-MODEL ======================"
for m in "${MODELS[@]}"; do
  done_json="results/$m/q6/q6_summary.json"
  out=$(ls -t logs/q6_${m}_*.out 2>/dev/null | head -1)
  err=$(ls -t logs/q6_${m}_*.err 2>/dev/null | head -1)
  if [ -f "$done_json" ]; then
    status="DONE  (q6_summary.json written)"
  elif [ -n "$out" ]; then
    last=$(grep -E "^\[q6\]" "$out" 2>/dev/null | tail -1 | cut -c1-70)
    status="...   $last"
  else
    status="(no log yet)"
  fi
  fail=""
  [ -n "$err" ] && grep -qiE "traceback|cuda error|out of memory" "$err" 2>/dev/null && fail="  <-- ERROR in .err"
  printf "%-26s %s%s\n" "$m" "$status" "$fail"
done

echo; echo "==================== RESULTS (if any) ==============="
.venv/bin/python scripts/q6_collect.py 2>/dev/null || echo "(no summaries to collect yet)"
