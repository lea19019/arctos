# Session handoff (2026-06-03)

Quick-resume state so a fresh session can continue without losing anything.
Everything below is on disk / git / SLURM — survives a session reset.

## Deep-research agents (run in background; transcripts persist on disk)
Persist each to `docs/findings/deep-research-raw/<taskid>.txt` when done, then
fold key claims into the primers.

| status | run id | task id | topic |
|---|---|---|---|
| ✅ done+saved | wf_36650cc2-1b3 | wf9a1207t | gap-mapping (the gem: MT-GPTQ, multilingual super-weights) |
| ✅ done+saved | wf_6b9537e1-2a6 | wco17ovot | healing-free low-bit (no PTQ reaches FP16 at W3; GPTVQ/QTIP-no-RFT best) |
| 🔄 running | wf_78db0b38-0a3 | wdxgmqpkt | pipeline-aware / endpoint-protection NOVELTY for MT |
| 🔄 running | wf_55df267e-2aa | wbj1okyez | BASE quantizer pick (GPTVQ/QTIP/LeanQuant/SqueezeLLM) + integration |
| 🔄 running | wf_a86cc7d6-bf4 | w430t0tix | eval protocol + WMT25 task exact metrics/numbers |

Recover results from: task output `/tmp/.../tasks/<taskid>.output` (if session
alive) OR the workflow transcript dir
`.claude/projects/-home-vacl2-arctos/*/subagents/workflows/<runid>/` (durable).

## Where the science stands
- **Gem (confirmed, 6 models, XCOMET-XL):** MT-calibrated GPTQ recovers the
  3-bit cliff; generic-calibrated GPTQ is worse than no quant. But: no
  healing-free PTQ reaches FP16 at W3 (deep-research). cs-de strongest.
- **Levers:** salient-channel FP16 recovers W3 (not sub-2-bit); super weights
  early-layer & model-varying; Fisher mixed-precision FAILED (use direct
  per-layer probe / Hessian instead); ternary>binary; COMET unreliable <2-bit.
- **Novel direction (the thesis):** `docs/findings/phase2-novel-direction.md` —
  protect language-specific endpoints (early super-weights + late conversion
  circuit), crush the language-neutral middle. Decisive test = the `pipeline`
  stage (crush_middle vs crush_ends at matched budget).

## Pending experiments (code committed, needs GPU run)
- **`pipeline` + `mixedlayer` stages** added to `compression/experiments/q6-compression/
  experiment.py`. CPU smoke timed out (slow); **needs a GPU canary** then a run.
  Launch: `sbatch --job-name=bloom-7b1 experiments/q6-compression/slurm/
  run_q6gem.sh ...` adapted, OR add a run_q6pipeline.sh with
  `--stages find pipeline --pipe-levels 3 2 ternary --use-comet`.
- gem/extreme stragglers (gemma/bloom gem, llama extreme) were resubmitted;
  check `bash scripts/q6_status.sh` and `python scripts/q6gem_collect.py
  [--subdir q6extreme]`.

## Cluster note
Intermittent transient `CUDA: unspecified launch failure` (~10-15% of jobs,
flaky nodes) — just resubmit. Login-node GPU is Prohibited → SLURM A100 only.
XCOMET-XL works offline (needs facebook/xlm-roberta-xl encoder, cached).

## Next actions on resume
1. Read + persist the 3 running deep-research outputs; update
   `phase2-method-primer.md` / `phase2-novel-direction.md` with verdicts.
2. GPU-canary the `pipeline` stage; if `crush_middle >> crush_ends`, the novel
   method holds → run across models, write `docs/findings/q6-pipeline.md`.
3. Pick base quantizer per the wbj1okyez agent; consider implementing it.
