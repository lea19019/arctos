# Q3 — Cross-lingual mapping in MLPs and layers

> Which MLPs and layers carry the cross-lingual mapping?

## Methods

- **Layer-level activation patching** at `mlp_out` and `resid_post` per layer.
- **IFR's MLP scoring** (per-MLP contribution magnitude averaged over the calibration set).
- **Optional probing on MLP outputs** — if Q1 probing reveals layers where source-language ID is decodable from `resid_post` but not from `attn_out`, the MLP at that layer is doing the work; a probe on `mlp_out[ℓ]` should confirm.

Validation: activation patching × IFR on layer-level MLP importance.

## Models covered

Aya, omt-llama, Tower.

## Embedded learning

- MLP-as-key-value-memory framing (Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"); why FFN width matters; how MLPs participate in next-token prediction distinct from attention.
- The Geva et al. 2022 follow-up on MLPs promoting concepts in vocabulary space — the "neuron as a fact" framing and whether cross-lingual mapping looks like this.

## Expected artifacts

- `results/{model}/q3/layer_mlp_importance.csv` — per-layer MLP importance from each method, with one column per method to make disagreement visible.
- `compression/docs/q3.md` — depth profile of where cross-lingual mapping concentrates per model.

## Satisfied when

For each model: a depth profile of MLP MT-importance, an explanation of how the dedicated-MT models (omt-llama, Tower) differ from the general LM (Aya), and explicit reporting of any case where IFR and activation patching disagree on the layer ranking.

## Tests

`tests/interp/test_activation_patching.py` and `tests/interp/test_ifr.py::test_ifr_agrees_with_patching_on_top_layer` — the GPU-tier cross-method agreement test for this question.

## Working notes

See `notes.md`.
