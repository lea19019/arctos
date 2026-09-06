#!/usr/bin/env bash
# Re-download the arXiv-hosted PDFs indexed in README.md (generated 2026-09-05).
set -u
mkdir -p phenomenon
declare -A P=(
  [phenomenon/Achilles-2025-Altering-Neurons-Cripples-Language.pdf]=2510.10238
  [Alves-2024-Tower-Open-Multilingual-LLM.pdf]=2402.17733
  [phenomenon/An-2025-Systematic-Outliers-in-LLMs.pdf]=2502.06415
  [Apertus-2025-Open-Compliant-Multilingual-LLMs.pdf]=2509.14233
  [Ashkboos-2024-QuaRot.pdf]=2404.00456
  [Ashkboos-2024-SliceGPT.pdf]=2401.15024
  [AttentionSinks-2026-Hallucination-Detection-Signals.pdf]=2604.10697
  [AttentionSinks-2026-Multilingual-NMT-NLLB.pdf]=2605.01229
  [Bafna-2025-Translation-Barrier-Hypothesis.pdf]=2506.22724
  [phenomenon/Barbero-2025-Why-LLMs-Attend-First-Token.pdf]=2504.02732
  [Belrose-2023-Tuned-Lens.pdf]=2303.08112
  [Biderman-2023-Pythia.pdf]=2304.01373
  [Blevins-2022-Mono-CrossLingual-Pretraining-Dynamics.pdf]=2205.11758
  [phenomenon/Bondarenko-2023-Quantizable-Transformers.pdf]=2306.12929
  [phenomenon/Cancedda-2024-Spectral-Filters-Dark-Signals-Attention-Sinks.pdf]=2402.09221
  [Chen-2024-PrefixQuant.pdf]=2410.05265
  [phenomenon/Chen-2026-Attention-Sinks-Induce-Gradient-Sinks.pdf]=2603.17771
  [phenomenon/Chen-2026-Measuring-Maximum-Activations.pdf]=2605.15572
  [CopyFirstTranslateLater-2026-Translation-Dynamics-Pretraining.pdf]=2604.17633
  [Dale-2023-Detecting-Mitigating-Hallucinations-MT.pdf]=2212.08597
  [Dale-2023-HalOmi-Benchmark.pdf]=2305.11746
  [phenomenon/Darcet-2023-Vision-Transformers-Need-Registers.pdf]=2309.16588
  [DataDecide-2025-Predict-Pretraining-Data.pdf]=2504.11393
  [DecoderLens-2023-Layerwise-Interpretation-EncoderDecoder.pdf]=2310.03686
  [phenomenon/Dettmers-2022-LLM-int8.pdf]=2208.07339
  [phenomenon/Ding-2026-Weibull-AdamW-Weight-Scale-Evolution.pdf]=2606.19367
  [phenomenon/Ding-2026-Weibull-Transformer-Weight-Distributions.pdf]=2605.18898
  [Dong-2024-Pruner-Zero.pdf]=2406.02924
  [Ettin-2025-Paired-Encoders-Decoders.pdf]=2507.11412
  [EuroLLM-2025-9B-Technical-Report.pdf]=2506.04079
  [Ferrando-2022-ALTI-Measuring-Mixing-Contextual-Information.pdf]=2203.04212
  [Ferrando-2022-Opening-Black-Box-NMT-ALTI.pdf]=2205.11631
  [phenomenon/Fesser-2026-Unifying-View-Sinks-Two-Algorithms.pdf]=2606.08105
  [Frantar-2022-GPTQ.pdf]=2210.17323
  [Frantar-2023-SparseGPT.pdf]=2301.00774
  [Fu-2021-Theoretical-Analysis-Repetition-Problem.pdf]=2012.14660
  [phenomenon/GallegoFeliciano-2025-Hidden-Dynamics-Massive-Activations.pdf]=2508.03616
  [GarbageAttention-2026-BOS-Sink-Heads-Sink-Aware-Pruning.pdf]=2601.06787
  [Goldman-2025-ECLeKTic-Crosslingual-Knowledge-Transfer.pdf]=2502.21228
  [Groeneveld-2024-OLMo-Accelerating-Science.pdf]=2402.00838
  [Gromov-2024-Unreasonable-Ineffectiveness-Deeper-Layers.pdf]=2403.17887
  [phenomenon/Gu-2025-When-Attention-Sink-Emerges.pdf]=2410.10781
  [Guerreiro-2023-Hallucinations-Large-Multilingual-Translation.pdf]=2303.16104
  [Guerreiro-2023-Looking-for-a-Needle-in-a-Haystack.pdf]=2208.05309
  [phenomenon/Guo-2024-Active-Dormant-Attention-Heads.pdf]=2410.13835
  [phenomenon/Hammerl-2023-Anisotropy-Outliers-Multilingual-LMs.pdf]=2306.00458
  [He-2024-Not-All-Attention-Is-Needed.pdf]=2406.15786
  [phenomenon/He-2024-Outlier-Features-Kurtosis.pdf]=2405.19279
  [Hiraoka-2024-Repetition-Neurons.pdf]=2410.13497
  [Hodgkinson-2025-Heavy-Tailed-Mechanistic-Universality.pdf]=2506.03470
  [Holtzman-2019-Curious-Case-Neural-Text-Degeneration.pdf]=1904.09751
  [InductionHeadToxicity-2025-Repetition-Curse.pdf]=2505.13514
  [phenomenon/Interpreting-2025-Repeated-Token-Phenomenon.pdf]=2503.08908
  [phenomenon/Jin-2025-Massive-Values-in-Self-Attention.pdf]=2502.01563
  [phenomenon/Kaul-2024-From-Attention-to-Activation.pdf]=2410.17174
  [Kim-2024-Shortened-LLaMA-Depth-Pruning.pdf]=2402.02834
  [Koishekenov-2023-NLLB200-Language-Specific-Expert-Pruning.pdf]=2212.09811
  [Kojima-2024-Finding-Controlling-Language-Specific-Neurons.pdf]=2404.02431
  [Lad-2024-Remarkable-Robustness-Stages-of-Inference.pdf]=2406.19384
  [Language-Lives-in-Sparse-Dimensions-2025.pdf]=2510.07213
  [Li-2023-Repetition-In-Repetition-Out.pdf]=2310.10226
  [phenomenon/Li-2026-Structural-Origin-Attention-Sink-Super-Neurons.pdf]=2605.06611
  [Liao-2024-Free-Lunch-Removing-Outliers-Pretraining.pdf]=2402.12102
  [Lin-2023-AWQ.pdf]=2306.00978
  [Lin-2024-DuQuant.pdf]=2406.01721
  [Liu-2023-LLM360-Fully-Transparent-LLMs.pdf]=2312.06550
  [Liu-2024-SpinQuant.pdf]=2405.16406
  [Lu-2024-AlphaPruning-HeavyTailed-Layerwise.pdf]=2410.10912
  [Lucie-2025-Lucie-7B-Multilingual-Open-Resources.pdf]=2503.12294
  [MAP-Neo-2024-Bilingual-Transparent-LLM.pdf]=2405.19327
  [phenomenon/Macocco-2025-Outlier-Dims-Across-Checkpoints.pdf]=2503.21718
  [Marchisio-2024-Language-Confusion-in-LLMs.pdf]=2406.20052
  [Marchisio-2024-Quantization-Multilingual-LLMs.pdf]=2407.03211
  [Martins-2024-EuroLLM-Multilingual-Europe.pdf]=2409.16235
  [Mechanistic-Language-Confusion-English-Centric-2025.pdf]=2505.16538
  [Meister-2023-Natural-Bias-for-Language-Generation.pdf]=2212.09686
  [Men-2024-ShortGPT-Layer-Redundancy.pdf]=2403.03853
  [Mondal-2025-LangSpecific-Neurons-Do-Not-Facilitate-Transfer.pdf]=2503.17456
  [NLLB-2022-No-Language-Left-Behind.pdf]=2207.04672
  [NVFP4-2026-Outlier-Dynamics-Pretraining.pdf]=2602.02047
  [Nanda-2023-Progress-Measures-for-Grokking.pdf]=2301.05217
  [OLMo2-2025-2-OLMo-2-Furious.pdf]=2501.00656
  [OffTarget-2023-ZeroShot-Multilingual-NMT.pdf]=2305.10930
  [Ogueji-2022-Intriguing-Properties-Compression-Multilingual.pdf]=2211.02738
  [phenomenon/Oh-2024-House-of-Cards-Massive-Weights.pdf]=2410.01866
  [phenomenon/Owen-2025-Refined-Analysis-Massive-Activations.pdf]=2503.22329
  [phenomenon/Parodi-2026-Zero-Ablation-Overstates-Register-Dependence.pdf]=2604.14433
  [PhaseTransitions-2025-Small-Transformer-LMs.pdf]=2511.12768
  [phenomenon/Pierro-2024-Mamba-PTQ-Outlier-Channels.pdf]=2407.12397
  [Power-2022-Grokking.pdf]=2201.02177
  [Probing-Emergence-CrossLingual-Alignment-2024.pdf]=2406.13229
  [phenomenon/Puccetti-2022-Outlier-Dimensions-Driven-by-Frequency.pdf]=2205.11380
  [phenomenon/Qiu-2025-Gated-Attention-Sink-Free.pdf]=2505.06708
  [phenomenon/QueipoDeLlano-2025-Sinks-Compression-Valleys-Same-Coin.pdf]=2510.06477
  [phenomenon/Rajaee-Pilehvar-2022-Isotropy-Multilingual-BERT.pdf]=2110.04504
  [phenomenon/RanMilo-2026-Mechanistic-Account-Sinks-GPT2.pdf]=2604.14722
  [phenomenon/RanMilo-2026-Sinks-Provably-Necessary.pdf]=2603.11487
  [Raunak-2021-Curious-Case-Hallucinations-NMT.pdf]=2104.06683
  [Registering-Source-Tokens-Target-Language-Spaces-2025.pdf]=2501.02979
  [Schaeffer-2023-Emergent-Abilities-Mirage.pdf]=2304.15004
  [Schut-2025-Do-Multilingual-LLMs-Think-In-English.pdf]=2502.15603
  [Sellam-2022-MultiBERTs.pdf]=2106.16163
  [Shao-2024-OmniQuant.pdf]=2308.13137
  [Siddiqui-2024-Deeper-Look-Depth-Pruning.pdf]=2407.16286
  [SignalDegradation-2026-Two-Failure-Modes-Quantization.pdf]=2604.19884
  [phenomenon/SingleLayer-2026-Understanding-Massive-Activations.pdf]=2605.08504
  [SmolLM2-2025-When-Smol-Goes-Big.pdf]=2502.02737
  [phenomenon/Stolfo-2024-Confidence-Regulation-Neurons.pdf]=2406.16254
  [phenomenon/Su-2026-Attention-Sink-Survey.pdf]=2604.10098
  [phenomenon/Subramanian-2026-Super-Weights-Failure-of-Selective-Training.pdf]=2607.08733
  [Sun-2023-Wanda-Simple-Effective-Pruning.pdf]=2306.11695
  [phenomenon/Sun-2024-Massive-Activations-in-LLMs.pdf]=2402.17762
  [phenomenon/Sun-2026-Spike-Sparse-Sink.pdf]=2603.05498
  [phenomenon/SuperExperts-2025-Unveiling-Super-Experts-in-MoE.pdf]=2507.23279
  [phenomenon/Supernodes-2026-Loss-Critical-Hubs-FFN.pdf]=2604.23475
  [Tang-2024-Language-Specific-Neurons-LAPE.pdf]=2402.16438
  [Tezuka-2025-Transfer-Neurons-Hypothesis.pdf]=2509.17030
  [phenomenon/Timkey-vanSchijndel-2021-All-Bark-No-Bite-Rogue-Dimensions.pdf]=2109.04404
  [Tracing-Multilingual-Representations-CrossLayer-Transcoders-2025.pdf]=2511.10840
  [TrainingDynamics-2025-PTQ-Robustness.pdf]=2510.06213
  [Trinley-2025-What-Languages-Does-Aya-23-Think-In.pdf]=2507.20279
  [UnevenPTQ-2025-Multilingual-MT-Quantization.pdf]=2508.20893
  [Universal-Conceptual-Structure-NLLB200-Geometry-2026.pdf]=2603.02258
  [Voita-2021-Source-Target-Contributions-NMT.pdf]=2010.10907
  [Voita-2023-Neurons-Dead-Ngram-Positional.pdf]=2309.04827
  [Wang-2024-Sharing-Matters-Neurons-Across-Languages-Tasks.pdf]=2406.09265
  [Wang-2025-Task-Circuit-Quantization.pdf]=2504.07389
  [Wei-2023-Outlier-Suppression-Plus.pdf]=2304.09145
  [Welleck-2019-Unlikelihood-Training.pdf]=1908.04319
  [Wendler-2024-Do-Llamas-Work-in-English.pdf]=2402.10588
  [WhenMeaningsMeet-2026-Shared-Concept-Spaces-Multilingual-Training.pdf]=2601.22851
  [Williams-2024-Calibration-Data-Pruning-Quantization.pdf]=2311.09755
  [Wu-2021-Language-Tags-Matter-ZeroShot-NMT.pdf]=2106.07930
  [Wu-2025-Semantic-Hub-Hypothesis.pdf]=2411.04986
  [phenomenon/Xiao-2023-Efficient-Streaming-LMs-Attention-Sinks.pdf]=2309.17453
  [Xiao-2023-SmoothQuant.pdf]=2211.10438
  [Xu-2022-Learning-to-Break-the-Loop.pdf]=2206.02369
  [Xu-2023-Understanding-Detecting-Hallucinations-NMT-Introspection.pdf]=2301.07779
  [Xu-2026-When-Do-Attention-Circuits-Form.pdf]=2606.02378
  [phenomenon/Yang-2024-Activation-Spikes-GLU-Variants.pdf]=2405.14428
  [Yang-2024-LaCo-Layer-Collapse-Pruning.pdf]=2402.11187
  [Yao-2025-Understanding-the-Repeat-Curse.pdf]=2504.14218
  [Yin-2024-OWL-Outlier-Weighed-Layerwise-Sparsity.pdf]=2310.05175
  [phenomenon/Yu-2024-The-Super-Weight-in-LLMs.pdf]=2411.07191
  [Zhang-2024-Multilingual-Brain-Surgeon.pdf]=2404.04748
  [Zhang-2024-TinyLlama.pdf]=2401.02385
  [Zhao-2024-How-LLMs-Handle-Multilingualism.pdf]=2402.18815
  [Zhao-2025-Random-Scaling-Emergent-Capabilities.pdf]=2502.17356
  [phenomenon/Zuhri-2025-Softpick-Rectified-Softmax.pdf]=2504.20966
  [arXiv-2602.02385-Factored-Representations.pdf]=2602.02385
  [vanderWal-2025-PolyPythias.pdf]=2503.09543
)
for f in "${!P[@]}"; do
  [ -s "$f" ] || curl -sSL --fail -o "$f" "https://arxiv.org/pdf/${P[$f]}" || echo "FAILED: $f"
done
[ -s phenomenon/Kovaleva-2021-BERT-Busters.pdf ] || curl -sSL --fail -o phenomenon/Kovaleva-2021-BERT-Busters.pdf "https://aclanthology.org/2021.findings-acl.300.pdf"
[ -s phenomenon/Amazon-2025-T5-Emergent-Outlier-Properties.pdf ] || curl -sSL --fail -o phenomenon/Amazon-2025-T5-Emergent-Outlier-Properties.pdf "https://aclanthology.org/2025.naacl-long.430.pdf"
# Not on arXiv (ACL Anthology / transformer-circuits.pub / slides / course docs) — see README.md rows:
#   phenomenon/Amazon-2025-T5-Emergent-Outlier-Properties.pdf
#   Bills-2023-LMs-Can-Explain-Neurons-in-LMs.pdf
#   Bricken-2023-Towards-Monosemanticity.pdf
#   Bridge-Concepts-Attention-to-Causal-Editing.pdf
#   Bridge-Concepts-Vision-Circuits-to-Transformer-Circuits.pdf
#   CS601R-Syllabus.pdf
#   Chi-2020-Universal-Grammatical-Relations-in-mBERT.pdf
#   Choenni-Shutova-2020-What-Does-It-Mean-LanguageAgnostic.pdf
#   Clark-2019-What-Does-BERT-Look-At.pdf
#   Conmy-2023-Automated-Circuit-Discovery-ACDC.pdf
#   Conneau-2018-Cram-Single-Vector-SLIDES.pdf
#   Conneau-2018-What-You-Can-Cram-Into-a-Single-Vector.pdf
#   Cross-lingual-Context-SLIDES.pdf
#   DeYoung-2020-ERASER-Benchmark.pdf
#   DoshiVelez-Kim-2017-Rigorous-Science-of-Interpretable-ML.pdf
#   EACL-2026-long-32.pdf
#   Elhage-2021-Mathematical-Framework-for-Transformer-Circuits.pdf
#   Elhage-2022-Toy-Models-of-Superposition.pdf
#   FACTUM-2026-Citation-Hallucination-in-LongForm-RAG.pdf
#   Ferrando-2025-Primer-Inner-Workings-of-Transformer-LMs.pdf
#   Final-Project-Guide.pdf
#   Geva-2023-Dissecting-Recall-of-Factual-Associations.pdf
#   Hu-2020-XTREME-Benchmark.pdf
#   Karthikeyan-2020-Crosslingual-Ability-of-Multilingual-BERT.pdf
#   phenomenon/Kovaleva-2021-BERT-Busters.pdf
#   Lipton-2018-Mythos-of-Model-Interpretability.pdf
#   MI-Tutorial-Pranav-SLIDES.pdf
#   Meng-2022-Locating-and-Editing-Factual-Associations-ROME.pdf
#   Meng-2022-ROME-SLIDES.pdf
#   Miaschi-2020-Contextual-and-NonContextual-Embeddings.pdf
#   Olah-2017-Feature-Visualization.pdf
#   Olah-2020-Zoom-In-Introduction-to-Circuits.pdf
#   Olsson-2022-InContext-Learning-and-Induction-Heads.pdf
#   Oncevay-2020-Bridging-Linguistic-Typology-and-MT.pdf
#   Paper-Presentation-Rubric.pdf
#   Pimentel-2020-InformationTheoretic-Probing.pdf
#   Pires-2019-How-Multilingual-is-Multilingual-BERT.pdf
#   Rauker-2023-Toward-Transparent-AI-Survey.pdf
#   ReDeEP-2024-Detecting-Hallucination-in-RAG.pdf
#   Representation-Engineering-SLIDES.pdf
#   Rogers-2020-Primer-in-BERTology.pdf
#   Templeton-2024-Scaling-Monosemanticity.pdf
#   Tenney-2019-BERT-Rediscovers-Classical-NLP-Pipeline.pdf
#   Tenney-2019-What-Do-You-Learn-From-Context.pdf
#   Turner-2023-Activation-Addition.pdf
#   Vig-2019-BertViz-Tool.pdf
#   Vig-2019-Multiscale-Visualization-of-Attention.pdf
#   Wang-2020-Crosslingual-Transferability-of-Monolingual-Reps.pdf
#   Wang-2023-Interpretability-in-the-Wild-IOI-Circuit.pdf
#   Wu-Dredze-2019-Beto-Bentz-Becas.pdf
#   Zhang-2024-RIA-Plug-and-Play-Pruning.pdf
#   Zhao-2020-Inducing-LanguageAgnostic-Multilingual-Reps.pdf
#   Zou-2023-Representation-Engineering.pdf
