# src/ — layout

**Brain and plumbing are separate.**

```
detect_sw.py        the motor: load a model, call a brain, write JSON. Never changes.
detectors/v1.py     brain: down_proj spikes + three checks (the original method)
detectors/v2.py     brain: + several channels per layer, contribution decomposition
detectors/v3.py     brain: + the paper's both-outliers, suppression stop, plausibility bound
detectors/v5.py     brain: residual-stream persistence (Yu et al. Fig 4)
sw_v5.py            the v5 brain as one standalone readable script, olmo_sw.py-style
```

A brain is one file exposing `find(model, layers, inputs) -> [{"layer","j","k","value"}]`.
A new idea is a new file in `detectors/`; the motor does not change.

```
uv run src/detect_sw.py --detector v5 --model allenai/OLMo-1B-0724-hf
CUDA_VISIBLE_DEVICES="" uv run src/detect_sw.py ...     # force CPU on the login node
```

Everything else:

| file | what |
|---|---|
| `ablate_sw.py` | causal check: zero one scalar, measure wikitext-2 perplexity + KL, restore |
| `olmo_sw.py`, `olmo_ablate.py`, `olmo_explore.py` | v0 scratch, OLMo-1B only, written from the paper alone |
| `detect_sw_v1.py` | the original standalone detector, untouched |
| `coord_check.py` | rank a Table 2 coordinate by \|W\| within its matrix (CPU only) |
| `activation_profile.py` | residual-stream magnitude by depth |
| `table2_agreement.py` | score a results dir against Table 2 |
| `sw_models.py` | `MODELS` (Yu et al. Table 2) and `MODERN` |
| `prefetch_models.py`, `provenance.py`, `run_all.py` | plumbing |
