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

## Everything else the survey cites (added 2026-09-08)

Every remaining PDF linked from `notes_lrl_quantization_why.md`, plus the repo's other
multilingual-compression papers:

- Diddee-2022-Too-Brittle-To-Touch-Quantization-Distillation-LowResource-MT.pdf — small Indic NMT; PTQ more stable than distillation
- Hossain-2026-Quantization-Effects-Bangla-NLU.pdf — one language, GGUF vs GPTQ (preprint)
- Ramesh-2023-Model-Compression-Fairness-Language-Models.pdf — compression and fairness
- Goncalves-2023-Model-Compression-Social-Bias-LLMs.pdf — quantization as regulariser
- Jaiswal-2024-Compressing-LLMs-Truth-Rarely-Pure-LLM-KICK.pdf — knowledge goes first under pruning
- Jin-2024-Cost-Of-Down-Scaling-Fact-Recall-Deteriorates.pdf — fact recall is the fragile part
- Wang-2026-Through-Compressed-Lens-Quantization-Factual-Recall.pdf — factual recall under quantization
- Wu-2026-Asymmetric-Harms-LLM-Compression.pdf — head/middle/tail retention (preprint)
- Liebenwein-2021-Lost-In-Pruning-Beyond-Test-Accuracy.pdf — robustness degrades before accuracy
- Limisiewicz-2023-Tokenization-Impacts-Multilingual-LM-Vocabulary-Allocation.pdf — fertility is not the only tokenizer variable
- Gumma-2023-Knowledge-Distillation-Compressing-Multilingual-NMT.pdf — distilling multilingual NMT
- Koishekenov-2023-NLLB200-Language-Specific-Expert-Pruning.pdf — pruning NLLB experts per language
- Hammerl-2023-Anisotropy-Outliers-Multilingual-LMs.pdf — outlier dimensions across languages
- Aji-2020-Compressing-NMT-4-Bit-Precision.pdf, Prato-2020-Fully-Quantized-Transformer-MT.pdf — early NMT quantization

## methods_and_outliers/ — quantization methods and the outlier literature

Not about languages; useful for the methods section and for the parked massive-activation
angle. SmoothQuant, Outlier Suppression+, Quantizable Transformers, PrefixQuant, DuQuant,
SpinQuant, Task-Circuit Quantization, the two-failure-modes paper, Free Lunch, NVFP4 outlier
dynamics, kurtosis, GLU activation spikes, frequency-driven outlier dimensions, outlier
dimensions across checkpoints, T5 outliers, Mamba PTQ.

## pruning/ — pruning papers

SparseGPT, Wanda, lottery-ticket heads for NMT, Shortened LLaMA, Pruner-Zero, AlphaPruning,
OWL, RIA, depth pruning, LaCo, FOCUS/RePAIR. Background for the long-tail line, which began
with pruning.

Left out on purpose: speech, audio and vision quantization papers (Whisper, EdgeASR, the ASR
and vision-encoder PTQ papers), and the attention-sink papers.
