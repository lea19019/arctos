#!/usr/bin/env bash
# Dependency-ordered submission for the PTQ-MT replication.
#
# Prereq: run precache.py on the LOGIN node first (downloads models/datasets,
# materializes C3 calib) and build llama.cpp (build_llama_cpp.sh). This script
# only submits GPU jobs.
#
# All five model sweeps + C3 are mutually independent and run concurrently.
# 32B and 70B are serialized (70B afterany 32B) so their large GGUF f16 bases
# don't pile up on disk at once. analyze runs after everything (afterany, so
# partial results still produce a report). Re-running this script is safe:
# every result unit is skip-if-exists, so jobs resume rather than redo.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
S=experiments/replication-uneven-ptq/slurm

j_small1=$(sbatch --parsable $S/run_repl_qwen3-1.7b.sh)
j_small2=$(sbatch --parsable $S/run_repl_llama-3.1-8b.sh)
j_small3=$(sbatch --parsable $S/run_repl_qwen3-8b.sh)
j_c3=$(sbatch --parsable $S/run_c3_calibration.sh)
j_32b=$(sbatch --parsable $S/run_repl_qwen3-32b.sh)
j_70b=$(sbatch --parsable --dependency=afterany:$j_32b $S/run_repl_llama-3.3-70b.sh)

j_analyze=$(sbatch --parsable \
  --dependency=afterany:$j_small1:$j_small2:$j_small3:$j_c3:$j_32b:$j_70b \
  $S/run_analyze.sh)

echo "submitted:"
echo "  qwen3-1.7b     $j_small1"
echo "  llama-3.1-8b   $j_small2"
echo "  qwen3-8b       $j_small3"
echo "  c3-calibration $j_c3"
echo "  qwen3-32b      $j_32b"
echo "  llama-3.3-70b  $j_70b   (after 32b)"
echo "  analyze        $j_analyze (after all)"
