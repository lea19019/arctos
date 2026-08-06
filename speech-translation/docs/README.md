# speech-translation/docs — the written record of this track

Everything written about *this* track. Cross-track material — the registry, the
research standards, the subject-matter foundations — lives in
[`../../docs/`](../../docs/). The measured results themselves are tabulated in
[`../README.md`](../README.md), backed by the committed
`../results/*/summary.tsv` and `results.json`.

- [`nllb_xtts_compression_survey.md`](nllb_xtts_compression_survey.md) — survey
  of PTQ, pruning, and distillation for NLLB-200 encoder-decoder MT and XTTS v2
  TTS. The landscape review that shaped which variants were worth measuring.
- [`nllb_xtts_interp_map.md`](nllb_xtts_interp_map.md) — the interpretability
  map for low-resource-language-preserving quantization: which components to
  protect in each model family, and the evidence for it.

Both are surveys written 2026-06-19, before the baseline sweep ran. Where they
disagree with the measured results in [`../README.md`](../README.md), the
measurements win — in particular, they do not anticipate that `bitsandbytes`
INT8 is *slower* than FP16 at this model size.

Background on what quantization is and how these two architectures work:
[`../../docs/learning/project_foundations.md`](../../docs/learning/project_foundations.md)
and
[`../../docs/learning/llm_quant_foundations.md`](../../docs/learning/llm_quant_foundations.md).
