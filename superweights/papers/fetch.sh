#!/usr/bin/env bash
# Re-download the indexed PDFs (see README.md). arXiv IDs resolve via /pdf/.
set -u
declare -A P=(
  [Yu-2024-The-Super-Weight-in-LLMs.pdf]=2411.07191
  [Subramanian-2026-Super-Weights-Failure-of-Selective-Training.pdf]=2607.08733
  [Ding-2026-Weibull-AdamW-Weight-Scale-Evolution.pdf]=2606.19367
  [Gu-2025-When-Attention-Sink-Emerges.pdf]=2410.10781
  [Sun-2024-Massive-Activations-in-LLMs.pdf]=2402.17762
  [Ding-2026-Weibull-Transformer-Weight-Distributions.pdf]=2605.18898
  [GallegoFeliciano-2025-Hidden-Dynamics-Massive-Activations.pdf]=2508.03616
  [Puccetti-2022-Outlier-Dimensions-Driven-by-Frequency.pdf]=2205.11380
  [He-2024-Outlier-Features-Kurtosis.pdf]=2405.19279
  [Dettmers-2022-LLM-int8.pdf]=2208.07339
  [Macocco-2025-Outlier-Dims-Across-Checkpoints.pdf]=2503.21718
  [QueipoDeLlano-2025-Sinks-Compression-Valleys-Same-Coin.pdf]=2510.06477
  [Sun-2026-Spike-Sparse-Sink.pdf]=2603.05498
  [Chen-2026-Measuring-Maximum-Activations.pdf]=2605.15572
  [Xu-2026-When-Do-Attention-Circuits-Form.pdf]=2606.02378
  [Biderman-2023-Pythia.pdf]=2304.01373
  [vanderWal-2025-PolyPythias.pdf]=2503.09543
  [Ettin-2025-Paired-Encoders-Decoders.pdf]=2507.11412
  [DataDecide-2025-Predict-Pretraining-Data.pdf]=2504.11393
  [Sellam-2022-MultiBERTs.pdf]=2106.16163
  [Zhao-2025-Random-Scaling-Emergent-Capabilities.pdf]=2502.17356
  [TrainingDynamics-2025-PTQ-Robustness.pdf]=2510.06213
  [NVFP4-2026-Outlier-Dynamics-Pretraining.pdf]=2602.02047
  [UnevenPTQ-2025-Multilingual-MT-Quantization.pdf]=2508.20893
  [Marchisio-2024-Quantization-Multilingual-LLMs.pdf]=2407.03211
  [Tang-2024-Language-Specific-Neurons-LAPE.pdf]=2402.16438
  [Zhao-2024-How-LLMs-Handle-Multilingualism.pdf]=2402.18815
  [AttentionSinks-2026-Multilingual-NMT-NLLB.pdf]=2605.01229
)
for f in "${!P[@]}"; do
  [ -s "$f" ] || curl -sSL --fail -o "$f" "https://arxiv.org/pdf/${P[$f]}" || echo "FAILED: $f"
done
[ -s Kovaleva-2021-BERT-Busters.pdf ] || curl -sSL --fail -o Kovaleva-2021-BERT-Busters.pdf "https://aclanthology.org/2021.findings-acl.300.pdf" || echo "FAILED: Kovaleva"
[ -s Amazon-2025-T5-Emergent-Outlier-Properties.pdf ] || curl -sSL --fail -o Amazon-2025-T5-Emergent-Outlier-Properties.pdf "https://aclanthology.org/2025.naacl-long.430.pdf" || echo "FAILED: Amazon-T5"
