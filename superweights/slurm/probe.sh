#!/bin/bash --login
#SBATCH --job-name=sw-probe
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.out
#
# Smoke test for the GPU path, on the one model whose answer we already know.
# OLMo-1B was run on CPU on 2026-09-02 (notes.md): the detector finds a super
# weight at L1[1764,1710] and ablating it must be CATASTROPHIC, while the
# paper's second coordinate L1[1764,8041] holds ~0.0018 and must do nothing.
# If the GPU run reproduces that, the sweep is safe to launch.

set -euo pipefail
cd "${SW_DIR:-/home/vacl2/arctos/superweights}"

export HF_HUB_OFFLINE=1          # compute nodes have no internet: fail fast
export HF_DATASETS_OFFLINE=1     # ... same for the wikitext-2 eval corpus
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

MODEL=allenai/OLMo-1B-0724-hf
SLUG=${MODEL//\//_}
OUT=${OUT_DIR:-results/probe}

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python src/detect_sw.py --detector "${DETECTOR:-v5}" --model "$MODEL" --out "$OUT/${SLUG}_found.json"
.venv/bin/python src/ablate_sw.py  --model "$MODEL" \
    --candidates "$OUT/${SLUG}_found.json" --out "$OUT/${SLUG}_ablation.json"

# ---- assert the known-good result, so a silent wrong answer fails the probe
.venv/bin/python - "$OUT/${SLUG}_ablation.json" "$OUT/${SLUG}_found.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
det = json.load(open(sys.argv[2]))

# v2 passed this probe while returning 138 candidates for a model with one
# real super weight: the old assertion only looked at the known-good
# coordinate and never at what came with it. Both now.
n = len(det["found"])
if n < 1:
    print("PROBE FAIL: detector returned NOTHING -- OLMo-1B has a known "
          "super weight at L1[1764,1710]")
    sys.exit(1)
if n > det["params"]["max_sw"]:
    print(f"PROBE FAIL: detector returned {n} candidates "
          f"(--max-sw {det['params']['max_sw']}); the paper reports <= 6")
    sys.exit(1)
print(f"probe ok: {n} candidate(s), stop reason: {det['stop_reason']}")
by_coord = {(r["layer"], r["j"], r["k"]): r for r in d["results"]}
real, dud = (1, 1764, 1710), (1, 1764, 8041)
ok = True
r = by_coord.get(real)
if r is None or r["verdict"] != "CATASTROPHIC":
    print(f"PROBE FAIL: L1[1764,1710] verdict={r and r['verdict']}"); ok = False
else:
    print(f"probe ok: L1[1764,1710] ppl {d['baseline']['ppl']:.2f} -> {r['ppl']:.2f}, KL {r['kl']:.2f}")
r = by_coord.get(dud)
if r is not None:
    print(f"probe info: paper's 2nd coord weight={r['weight']:.5f} verdict={r['verdict']}")
print(f"probe info: corpus={d['eval_corpus']} {d['ppl_segments']}x{d['seq_len']} dtype={d['dtype']} git={d['git_sha']}")
sys.exit(0 if ok else 1)
PY
echo "PROBE PASSED"
