# Parked: Claude session of 2026-09-02 (joint ablation)

Everything in here was produced by a Claude session on 2026-09-02 and moved out
of the working tree at Adrian's request. Nothing in the repo proper depends on
it. The tree outside this folder is back at commit 7e7471c (plus Adrian's own
edit to docs/proposal_draft.md).

- `src/` — my rewritten `ablate_sw.py` (joint ablation, magnitude-matched null,
  super-activation removal, bootstrap CIs), `detect_sw.py` (v5 + `sw_arch`
  refactor only), `sw_models.py` (BASES/SMALL/MULTI sets), and the new files
  `sw_arch.py`, `bootstrap_ci.py`, `encdec_sw.py` (NLLB), `probe_check.py`,
  `joint_summary.py`. To use any of them, copy back and set `PYTHONPATH=src`.
- `slurm/` — `joint.sh`, `conc.sh`, `encdec.sh`, `prompts.sh`, and my edits to
  `sweep.sh` / `probe.sh`.
- `experiments/joint_ablation/` — the spec (README, configs, notes).
- `results/` — finished JSON: `v6` (Table 2 models, joint), `modern_v6`,
  `probe_v6`, plus partial `bases_v6`, `small_v6`, `multi_v6`, `encdec_v6`,
  `conc_v6`, `prompts_v6`, `modern_v3_v6` (jobs were cancelled mid-run; treat
  anything without an `_ablation.json` as incomplete).
- `logs/` — the SLURM logs.

Headline number, if it is ever wanted: zeroing a model's candidate weights
*jointly* (Subramanian et al. COLM 2026 protocol) collapses models whose
weights are individually inert — Llama-3.1-8B-Instruct x1.02 each, x47,488
together; Phi-3's six Table 2 coordinates x671 together (their x374).
`results/*/…_ablation.json` carry git sha, revision, seed and bootstrap CIs.
