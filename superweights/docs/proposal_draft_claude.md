CS Master’s Project Draft Proposal
Adrian Castillo

Super Weights: Formation and Cross-Lingual Behavior

The emergence of LLMs has led to extraordinary applications, as they’ve been widely adopted and due to their significant memory consumption it has been necessary to develop compression techniques. Dettmers et al. (2022) discovered that a small fraction of activation outliers are disproportionately important to the quality of the model. Following this discovery Yu et al. (2024) traced this phenomenon down to the weights, discovering that a single parameter can destroy the ability of an LLM to generate text naming these parameters Super Weights. 

Yu et al. (2024) proposed a method to identify such parameters using a single forward pass, and also found they induce correspondingly rare and large activation outliers, naming them Super Activations. Related work showed that super weights, despite their importance, cannot be fine-tuned in isolation, restricting updates to their coordinates collapses the model entirely (Subramanian et al. 2026). Their work is a step forward to understand this phenomenon, yet we still don’t know how they form or why, and this project aims to contribute to their understanding. 

Our objective will be to try to answer the following questions:
RQ1: Does quality degradation form gradually or abruptly?
RQ2: Do SWs exist beyond large decoder-only models?
RQ3: Is the SW shared machinery or language-specific?
RQ4: What causes the formation of SW?

In order to answer these questions a series of experiments will be conducted. All of them rest on one shared piece of methodology, built first: since Yu's detection method always returns a candidate coordinate regardless of whether a genuine SW exists, we will develop a calibrated detector — every detection verified by causal ablation, validated by recovering Yu et al.'s published coordinates (starting with OLMo-1B, where the published answer and public training checkpoints coincide) and by planted-weight and random-initialization tests — together with a criterion for concluding "no SW found," an outcome no published method can currently express.

RQ1: Detect and verify the SW at the final checkpoint of models with public training checkpoints (OLMo-1B first, then Pythia, PolyPythias), then trace that coordinate across checkpoints and seeds by measuring text-generation quality when pruning it at each checkpoint. At each checkpoint we will also record cheap outlier statistics with built-in nulls (activation kurtosis (He et al., 2024), attention-sink fraction, weight spectra), to test whether SW criticality emerges synchronized with the known outlier phenomena or independently of them — a question the current literature disputes at the activation level and has never asked at the weight level. (The seeds arm uses PolyPythias, whose models span 14M–410M parameters, and is contingent on RQ2's sub-1B result.)

RQ2: Using the calibrated detector, search for SWs in LLMs with different architectures (e.g., NLLB, an encoder-decoder model) and in smaller LLMs of 1B parameters or less. Either outcome is a finding: a verified detection unlocks the multi-seed arm of RQ1, while a defensible absence would be the first of its kind in this literature and would pose a new question — why do SWs form only at larger scale? — which Pythia's 70M–12B size range lets us bracket directly, and which informs the design of RQ4.

RQ3: Measure cross-lingual quality in multilingual models after pruning the SW. Equal damage across languages would indicate shared machinery; unequal damage would provide a mechanistic lead on uneven multilingual quantization degradation.

RQ4: Train models with diverse training targets and architectures to find out whether these affect the formation of SWs. This experiment will be launched only once RQ2 confirms the phenomenon exists at a trainable scale.

Through these experiments we expect to gain experience in experimental design
(nulls, pre-registered criteria, multi-seed statistics), interpretability
techniques (activation hooks, causal ablation, logit-level analysis), model
training, and research engineering (provenance, reproducible pipelines,
evaluation harnesses). Upon successful experimentation we aim to contribute to
a deeper understanding of this fascinating phenomenon.

Deliverables for Grade:
Written and verbal reports
The calibrated SW detector: a tested, reusable tool validated against published coordinates, with the "no SW found" criterion — the first detector in this literature able to report absence
Repository with codebase, configs, and experimental results, reproducible from documented commands
Trained checkpoints from RQ4 archived separately if those runs launch

Bibliography
Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202. arXiv:2304.01373.
Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. Advances in Neural Information Processing Systems 35 (NeurIPS 2022). arXiv:2208.07339.
Groeneveld, D., et al. (2024). OLMo: Accelerating the science of language models. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024). arXiv:2402.00838.
He, B., Noci, L., Paliotta, D., Schlag, I., & Hofmann, T. (2024). Understanding and minimising outlier features in transformer training. Advances in Neural Information Processing Systems 37 (NeurIPS 2024). arXiv:2405.19279.
NLLB Team et al. (2022). No language left behind: Scaling human-centered machine translation. arXiv:2207.04672.
Subramanian, S., Akinfaderin, A., & Sehwag, A. (2026). Super weights in LLMs and the failure of selective training. Conference on Language Modeling (COLM 2026). arXiv:2607.08733.
van der Wal, O., Lesci, P., Müller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). PolyPythias: Stability and outliers across fifty language model pre-training runs. International Conference on Learning Representations (ICLR 2025). arXiv:2503.09543.
Yu, M., Wang, D., Shan, Q., Reed, C. J., & Wan, A. (2024). The super weight in large language models. arXiv:2411.07191.
