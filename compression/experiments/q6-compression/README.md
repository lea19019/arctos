# Q6 — find / keep / shrink / prune (phase-two compression sandbox)

Operationalizes the phase-two ideas on every decoder model, measured by
**chrF++ on generated translations** (not target-token logit — the fix for
Q5's weak metric) using **faithful** quant/prune operations (not Gaussian
noise). Concept + reading list: [`compression/docs/compression_primer.md`](../../docs/compression_primer.md).

## Stages

- **find** — sensitivity-native saliency (NOT interpretability importance,
  which Q5 showed is uncorrelated): super weights (`super_weights.py`), AWQ
  salient channels + MT-vs-generic calibration shift (`salient_channels.py`),
  MT-conditional Fisher diagonal (`hessian_diag.py`).
- **shrink** — RTN INT-k quantization, chrF++ vs bits {4,3,2}.
- **keep** — at the hardest bit-width: RTN vs AWQ-scaling vs
  keep-salient-FP16 vs super-weight-preservation.
- **prune** — magnitude vs Wanda at sparsity {0.25,0.5}; plus the super-weight
  stress test (ablate the 1 super weight vs the 1000 largest-magnitude).

## Run

```bash
# full run (one model)
python experiments/q6-compression/experiment.py \
    --config experiments/q6-compression/configs/aya-expanse-8b.yaml \
    --output results/aya-expanse-8b/q6 \
    --n-examples 24 --max-new-tokens 48 --calib-n 96 --bits 4 3 2 --sparsities 0.25 0.5

# subset stages / sizes for a smoke test
python experiments/q6-compression/experiment.py --config .../_smoke-bloom560m.yaml \
    --output results/_smoke/q6 --stages find shrink --bits 8 2 --n-examples 3
```

## Cluster

```bash
bash experiments/q6-compression/slurm/submit_all.sh            # all 8 decoder models
bash experiments/q6-compression/slurm/submit_all.sh aya-expanse-8b   # just one
```

A100 80GB, ~6h wall each (generation-heavy: every quant/prune config generates
n_examples × 3 pairs translations). NLLB (encoder-decoder) is excluded — the
quantizer targets decoder blocks.

## Outputs

`results/{model}/q6/q6_summary.json` — all stage results; `fisher.npz` —
per-layer Fisher. Validated end-to-end on CPU/bloom-560m before the sweep.
