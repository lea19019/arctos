# src/ — what each file is

**The live detector is `detect_sw.py`.** It is currently **v5**; its first
docstring line and the `detector_version` field in every results JSON say so.
Superseded detectors are frozen with a version suffix and kept runnable, so
`ls` shows old versions by name and the current one by the plain name.

| file | what |
|---|---|
| `detect_sw.py` | **live detector (v5)** — residual-stream persistence, Yu et al. Fig 4 |
| `detect_sw_v3.py` | frozen — both-outliers + suppression stop; 18/21, 124 false positives |
| `detect_sw_v2.py` | frozen — top-j + contribution prefix; 20/21, 1,945 false positives |
| `detect_sw_v1.py` | frozen — three hand-made thresholds; 11/21, 10 false positives |
| `olmo_sw.py`, `olmo_ablate.py`, `olmo_explore.py` | v0 scratch, OLMo-1B only, written from the paper alone |
| `ablate_sw.py` | causal check: zero one scalar, measure wikitext-2 perplexity + KL, restore |
| `coord_check.py` | rank a Table 2 coordinate by \|W\| within its own matrix (CPU only) |
| `activation_profile.py` | residual-stream magnitude by depth — onset and persistence |
| `table2_agreement.py` | score any results dir against Table 2 |
| `sw_models.py` | `MODELS` (Yu et al. Table 2) and `MODERN` (no answer key) |
| `prefetch_models.py` | cache weights on the login node (compute nodes are offline) |
| `provenance.py` | `git_sha()` embedded in every results file |
| `run_all.py` | serial detect+ablate+summary; the SLURM array is the usual path |

There is no `detect_sw_v4.py`: "v4" is not a detector, it is v1's candidate
set re-ablated on wikitext-2 (`slurm/reablate.sh`) — the control that
isolated the eval-corpus change from the detector change.

Any frozen version stays runnable and writes to its own results directory:

    DETECTOR=src/detect_sw_v1.py OUT_DIR=results/v1 sbatch --array=0-8 slurm/sweep.sh
