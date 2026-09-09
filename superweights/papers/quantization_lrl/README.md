# Reading set for the quantization / low-resource-language proposal

Copies of PDFs from `superweights/papers/`, gathered 2026-09-08 for drafting the
CS 698R proposal. The findings they support are in
`superweights/docs/quantization_lrl_findings.md`; the per-paper verdicts are in
`superweights/docs/literature/notes_lrl_quantization_why.md`.

## Read first, in this order (the ones the proposal rests on)

1. Chang-2025-Why-Some-Inputs-Break-Low-Bit-Quantization.pdf — residual norm predicts breakage (within English; our direction is reversed)
2. Tran-2022-Pruning-Disparate-Impact-Accuracy.pdf — why under-represented groups sit near the decision boundary
3. Proskurina-2024-When-Quantization-Affects-Confidence-LLMs.pdf — GPTQ hurts low-confidence inputs (within English)
4. Marchisio-2024-Quantization-Multilingual-LLMs.pdf — the multilingual gap itself, with human evaluation
5. Kumar-2025-Scaling-Laws-For-Precision.pdf — more training tokens, more PTQ damage
6. Ouyang-2025-Low-Bit-Quantization-Favors-Undertrained-LLMs.pdf — same, over 1500 checkpoints
7. Catalan-Tatjer-2026-Training-Dynamics-PTQ-Robustness.pdf — Catalan-Tatjer 2026: the learning-rate confound
8. Timkey-vanSchijndel-2021-All-Bark-No-Bite-Rogue-Dimensions.pdf — rogue dimensions dominate norms; standardize first
9. Frantar-2022-GPTQ.pdf — GPTQ, including the act-order variant we needed

## Methods and baselines

- Lin-2023-AWQ.pdf — the other main 4-bit method (untested by us)
- Shao-2024-OmniQuant.pdf — source of the Llama-2-7B perplexity targets we validated against
- Williams-2024-Calibration-Data-Pruning-Quantization.pdf — calibration-set choice matters

## The gap between languages (evidence and the calibration debate)

- Marie-Fujita-2025-Uneven-Impact-PTQ-Machine-Translation.pdf — Marie & Fujita 2025, 55 languages, COMET
- Chimoto-2026-Calibrating-Beyond-English-Quantized-Multilingual-LLM.pdf — multilingual calibration helps
- Borgersen-2025-English-K-Quantization-Multilingual-Performance.pdf — calibration language does not matter for GGUF at 70B
- Soualhi-2026-Multilingual-Quantization-Tax-Edge-SLMs.pdf — Hindi and Arabic collapse under NF4 (preprint, n=1)
- Zhang-2024-Multilingual-Brain-Surgeon.pdf — Hessian-geometry explanation of calibration effects
- Ogueji-2022-Intriguing-Properties-Compression-Multilingual.pdf — moderate pruning helps low-data languages
- Mohammadshahi-2022-Compressed-Multilingual-MT-Forget.pdf — compressed M2M-100 forgets under-represented pairs
- Gurgurov-2025-Multilingual-Encoder-Compression-Low-Resource.pdf — encoder compression tracks data size

## The long-tail story (guess 1's origin)

- Hooker-2019-What-Do-Compressed-DNNs-Forget.pdf
- Hooker-2020-Characterising-Bias-Compressed-Models.pdf
- Ahia-2021-Low-Resource-Double-Bind-Pruning-MT.pdf
- Lotfi-2026-Quantized-Reasoning-Models-Think-Longer.pdf — damage lands where entropy is high
- Tropeano-2025-As-Easy-As-PIE-Pruning-Disagree.pdf — first PIE study on text

## Tokenizer fragmentation (guess 3)

- Rust-2021-How-Good-Is-Your-Tokenizer.pdf — defines fertility
- Ahia-2023-Do-All-Languages-Cost-The-Same-Tokenization.pdf
- Petrov-2023-Tokenizers-Introduce-Unfairness-Between-Languages.pdf
- Arnett-2025-Explaining-Mitigating-Crosslingual-Tokenizer-Inequities.pdf — a tool for a controlled fertility test

## Background: models, data, and where we started

- Martins-2025-EuroLLM-9B-Technical-Report.pdf — the model that is 10x more robust at 4 bits
- NLLB-2022-No-Language-Left-Behind.pdf — FLORES-200, which FLORES+ continues
- Sun-2024-COLM-Massive-Activations-in-LLMs.pdf, An-2025-Systematic-Outliers-in-LLMs.pdf — the parked massive-activation angle
- Zhong-2025-Language-Lives-in-Sparse-Dimensions.pdf, Dumas-2025-Separating-Tongue-From-Thought.pdf — the other two starting papers

Not in the folder: the Llama 3 report (arXiv:2407.21783) and the WikiText paper (Merity 2017), both online.
