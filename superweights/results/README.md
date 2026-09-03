# Results, by detector generation

One directory per generation, so a new detector can never overwrite the run
it is being compared against. Each `*_found.json` and `*_ablation.json`
carries its own provenance: `git_sha`, resolved model revision, dtype,
library versions, and the detector `params` that produced it.

| dir | detector | eval corpus | Table 2 recovered | extra candidates | causally real |
|---|---|---|---|---|---|
| `v1/` | `detect_sw_v1.py` | 4 paragraphs | 11/21 | 10 | 4 |
| `v2/` | `detect_sw_v2.py` | 4 paragraphs | 20/21 | 1,945 | 4 |
| `v3/` | `detect_sw_v3.py` | wikitext-2 | 18/21 | 124 | 5 |
| `v4/` | v1's candidates, **re-scored** | wikitext-2 | 11/21 | 10 | 5 |
| `v5/` | `detect_sw.py` | wikitext-2 | 14/21 | **5** | 5 |

`v4` is not a detector. It is `v1`'s candidate set re-ablated on wikitext-2
(`slurm/reablate.sh`), i.e. the control that isolates the eval-corpus change
from the detector change. Comparing `v1` to `v4` measures the metric;
comparing `v4` to `v3`/`v5` measures the detector.

The two `probe_*` directories are single-model smoke tests (OLMo-1B, whose
answer is known), not results.

`v2/*_found.json` had its full per-layer round logs stripped — 13 MB of the
17 MB — leaving a per-round summary. Regenerate with the recorded `params`.

See `../notes.md` for what each generation got wrong and why.
