# Papers — superweights track (index)

PDFs live in `../../papers/`, one flat folder, **gitignored** (same policy as
`interlingua/papers/`); this file is the committed index, organised by topic. Re-download the
arXiv ones with `bash papers/fetch.sh` (run from `superweights/`); the rest list their source.
Reading notes, one file per topic, sit beside this file as `notes_<topic>.md`, written
2026-09-05/06 by Opus reader agents — each states what it read vs. inferred and marks anything
`[unopened]`; verify before citing. Papers copied from `../../interlingua/papers/` are duplicates
kept here so the final project has every paper in one place. Nine byte-identical PDFs that two
agents had fetched under different names were merged on 2026-09-07 (one name kept, rows repointed).

**The 47 papers about the phenomenon itself are listed first below; the map linking them
(terminology crosswalk, per-paper links in the causal chain, live disputes, reading order) is
[`phenomenon_crosswalk.md`](phenomenon_crosswalk.md).**

Topics (jump links): [Super weights and massive activations](#super-weights-and-massive-activations) · [Outlier features and attention sinks](#outlier-features-and-attention-sinks) · [Repetition, degeneration, and MT hallucination](#repetition-degeneration-and-mt-hallucination) · [Compression: pruning and quantization](#compression-pruning-and-quantization) · [Multilingual and MT interpretability](#multilingual-and-mt-interpretability) · [Training dynamics and checkpoint suites](#training-dynamics-and-checkpoint-suites) · [Interpretability methods (copied from interlingua/papers)](#interpretability-methods-copied-from-interlingua-papers) · [Misc: hallucination and RAG (copied from interlingua/papers)](#misc-hallucination-and-rag-copied-from-interlingua-papers) · [Course documents (CS 601R, copied from interlingua/papers/course-docs)](#course-documents-cs-601r-copied-from-interlingua-papers-course-docs)

## Super weights and massive activations

Reading notes: [`notes_super_weights_massive_activations.md`](notes_super_weights_massive_activations.md)

Papers on the **origin** of the phenomenon: single scalar weights (`super
weights`) in an early-layer `mlp.down_proj`, the massive activations they
induce, and the architectural/optimisation choices that create or remove them.
Sink-side and general-outlier papers live in
the *Outlier features and attention sinks* section below.

Synthesis and open problems: [`notes_super_weights_massive_activations.md`](notes_super_weights_massive_activations.md).

| File | Source | One line: why it is here |
|---|---|---|
| Yu-2024-The-Super-Weight-in-LLMs.pdf | arXiv:2411.07191 (preprint; v2 Jul 2025) | The origin paper: pruning one scalar in `down_proj` collapses the model; Table 2 is the coordinate directory this project replicates; Fig. 5 is the stopword result nobody has explained. |
| Sun-2024-COLM-Massive-Activations-in-LLMs.pdf | arXiv:2402.17762 (COLM 2024, peer-reviewed) | Defines massive activations; the zero-vs-mean intervention (Table 3) that makes the "fixed bias" claim; shows explicit learnable KV biases remove the need for them. |
| Sun-2026-Spike-Sparse-Sink.pdf | arXiv:2603.05498 (preprint, Mar 2026) | The from-scratch 7B ablation suite: SwiGLU as a directional quadratic amplifier, and the claim that spikes and sinks **decouple** under sandwich-norm / QK-norm / DyT. |
| QueipoDeLlano-2025-Sinks-Compression-Valleys-Same-Coin.pdf | arXiv:2510.06477 (ICLR 2026, peer-reviewed) | The opposing causal claim: ablating the layer-0 MLP contribution to BOS in a *pretrained* model kills sinks and the compression valley together (Thm 1, Fig. 4). |
| Subramanian-2026-Super-Weights-Failure-of-Selective-Training.pdf | arXiv:2607.08733 (COLM 2026, peer-reviewed) | Direct follow-up: training super weights in isolation fails at chance; Table 8 reports **no effect** for top-magnitude coordinates in four models — the negative this project's joint-set finding explains. |
| Oh-2024-House-of-Cards-Massive-Weights.pdf | arXiv:2410.01866 (preprint) | Independent line defining criticality at the **intermediate neuron** (rows of `W_gate`/`W_up`) rather than the `down_proj` scalar; reports repetition after ablation, and that Gemma-2 / Phi-3-medium are insensitive. |
| SuperExperts-2025-Unveiling-Super-Experts-in-MoE.pdf | arXiv:2507.23279 (ICLR 2026, peer-reviewed) | The closest published analogue to this project's ablation: pruning 3 of 6,144 experts yields **repetitive, uninformative output** plus ~90% attention-sink decay. Table 8 lists dense super weights sharing one `down_proj` input column. |
| An-2025-Systematic-Outliers-in-LLMs.pdf | arXiv:2502.06415 (ICLR 2025, peer-reviewed) | Unifies weight / activation / attention outliers into one causal chain; argues they are implicit **context-aware scaling factors** arising from softmax's inability to express a zero update. |
| Guo-2024-Active-Dormant-Attention-Heads.pdf | arXiv:2410.13835 (preprint) | Theory + toy model (Bigram-Backcopy) for *why* extreme tokens form: a mutual-reinforcement dynamic; Adam→SGD and softmax→ReLU each remove residual-state peaks. Analyses OLMo-7B pretraining checkpoints. |
| Kaul-2024-From-Attention-to-Activation.pdf | arXiv:2410.17174 (preprint) | The cleanest dissociation of the two causes: softmax-1 removes first-token dominance but **not** outlier activations; the adaptive optimiser does — OrthoAdam removes them. |
| Li-2026-Structural-Origin-Attention-Sink-Super-Neurons.pdf | arXiv:2605.06611 (ICML 2026, stated) | Terminologically nearest neighbour to "super weight" on the sink side: variance discrepancy → FFN **super neurons** with channel-sparse down-projections → dimension disparity → sink. |
| Chen-2026-Attention-Sinks-Induce-Gradient-Sinks.pdf | arXiv:2603.17771 (preprint) | Backward-pass account; claims a V-scale intervention keeps sinks while suppressing massive activations — another decoupling any causal claim must survive. |
| Zuhri-2025-Softpick-Rectified-Softmax.pdf | arXiv:2504.20966 (preprint) | Rectified softmax trained from scratch: removes sinks **and** massive activations together — the counterweight to the decoupling papers. |
| Qiu-2025-Gated-Attention-Sink-Free.pdf | arXiv:2505.06708 (preprint, Qwen team) | Sigmoid gating on attention output removes sinks at scale; the intervention Sun-2026 reuses to argue sinks are a learned input-conditioned gate. |
| Owen-2025-Refined-Analysis-Massive-Activations.pdf | arXiv:2503.22329 (preprint, BluOrion) | **Read this before writing any ablation number.** 17 models with BOS controlled: set-to-mean is harmless in all 16, set-to-zero is catastrophic in only 9, OLMo-2-7B has no massive activations at all, and DyT removes them while attention concentration persists. |
| Jin-2025-Massive-Values-in-Self-Attention.pdf | arXiv:2502.01563 (ICML 2025, peer-reviewed) | Massive values sit in Q and K but never V, and disrupting them collapses *contextual* tasks (passkey 100 → 0) far more than parametric recall. Its Appendix F is the warning that perplexity misses repetition — a fully looping generation scores PPL 2.99. |
| Chen-2026-Measuring-Maximum-Activations.pdf | arXiv:2605.15572 (preprint) | Measurement survey over 27 checkpoints; important because **4 of 24 fail Sun's binary criterion**, and binary-vs-continuous views of "massive" disagree in both directions. |
| GallegoFeliciano-2025-Hidden-Dynamics-Massive-Activations.pdf | arXiv:2508.03616 (preprint) | The only trajectory study across the full Pythia suite (14M–12B, steps 0–143k); step 0 is a free null. Single-seed, purely observational — see NOTES §1 for a citation-quality warning. |
| Ding-2026-Weibull-Transformer-Weight-Distributions.pdf | arXiv:2605.18898 (preprint) | The nearest thing to a **calibrated null** in this literature: an analytically derived random-init shape anchor k₀ = 1.205 for `\|W\|`. Its fit deliberately discards the top 10%, so it cannot itself flag a super weight. |
| Ding-2026-Weibull-AdamW-Weight-Scale-Evolution.pdf | arXiv:2606.19367 (preprint) | Sequel; decomposes the AdamW weight-scale force budget with ground-truth optimiser moments. The only paper in this folder with real seed replication (n=4). |

### Related, filed elsewhere in this repo

Not duplicated here — read them where they live.

| Paper | Location |
|---|---|
| Bondarenko 2023, *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing* (NeurIPS 2023) | Bondarenko-2023-Quantizable-Transformers.pdf |
| Barbero 2025, *Why do LLMs attend to the first token?* (COLM 2025) | Barbero-2025-Why-LLMs-Attend-First-Token.pdf |
| Cancedda 2024, *Spectral Filters, Dark Signals, and Attention Sinks* | Cancedda-2024-Spectral-Filters-Dark-Signals-Attention-Sinks.pdf |
| Xiao 2023, *Efficient Streaming LMs with Attention Sinks* (StreamingLLM) | Xiao-2023-Efficient-Streaming-LMs-Attention-Sinks.pdf |
| Yang 2024, *Activation Spikes in GLU Variants* | Yang-2024-Activation-Spikes-GLU-Variants.pdf |
| Stolfo 2024, *Confidence Regulation Neurons* (NeurIPS 2024) | Stolfo-2024-Confidence-Regulation-Neurons.pdf |
| Voita 2023, *Neurons in LLMs: Dead, N-gram, Positional* | `Voita-2023-Neurons-Dead-Ngram-Positional.pdf` |
| Hiraoka 2024, *Repetition Neurons* | `Hiraoka-2024-Repetition-Neurons.pdf` |

## Outlier features and attention sinks

The **surrounding** literature: outlier feature dimensions (the pre-2024
lineage), attention sinks, what they are for, how they are detected, and the
methodological cautions on ablating them. The super-weight/massive-activation
origin story lives in
the *Outlier features and attention sinks* section,
whose [`notes_outlier_features_attention_sinks.md`](notes_outlier_features_attention_sinks.md) is the
synthesis for both folders.

PDFs are **gitignored** (repo policy — see `../README.md`); this index is the
committed record.

|---|---|---|
| Dettmers-2022-LLM-int8.pdf | arXiv:2208.07339 (NeurIPS 2022, peer-reviewed) | Origin of "outlier feature"; §4.1's criterion (magnitude ≥ 6.0, ≥25% of layers, ≥6% of tokens) is the field's most-copied threshold — and was reverse-engineered to yield one detection in a 125M model. |
| Kovaleva-2021-BERT-Busters.pdf | ACL-IJCNLP 2021 Findings, peer-reviewed | The encoder precedent: zeroing 48 LayerNorm parameters costs up to 44 GLUE points. Has a real effect-side null (1,000 random pairs) and the cleanest magnitude-vs-position dissociation. |
| Puccetti-2022-Outlier-Dimensions-Driven-by-Frequency.pdf | arXiv:2205.11380 (Findings of EMNLP 2022) | The token-frequency link, plus the one from-scratch manipulation of it (§4.4). Reports that ablating outliers makes BERT predict **more** frequent tokens. |
| Macocco-2025-Outlier-Dims-Across-Checkpoints.pdf | arXiv:2503.21718 (preprint) | Decoder-side counterpart, and it **contradicts Puccetti's direction**: ablation makes 8 decoders predict *rarer* tokens. Best effect-side nulls in the set; the mechanism holds for only 5 of 8 models. |
| He-2024-Outlier-Features-Kurtosis.pdf | arXiv:2405.19279 (NeurIPS 2024, peer-reviewed) | The measurement design to copy: bounded, scale-invariant kurtosis and max-median ratio with **no threshold at all**, anchored to initialisation and validated against quantisation error. Also the strongest causal work on recipe knobs (LR, Adam ε, non-diagonal preconditioners). |
| Amazon-2025-T5-Emergent-Outlier-Properties.pdf | ACL Anthology 2025.naacl-long.430 (NAACL 2025, peer-reviewed) | The only **encoder-decoder** study here: T5/Flan-T5 60M–11B, and the finding that encoder and decoder outlier dimensions are *disjoint*, with decoders carrying more. Names pretraining dynamics as an open problem. |
| Xu-2026-When-Do-Attention-Circuits-Form.pdf | arXiv:2606.02378 (preprint, single author) | Checkpoint-resolved sink formation across Pythia/OLMo/OLMoE, and the only paper here with an **empirical null distribution** (500 random-target draws per revision) plus a check that the null does not drift across training. Single seed, and blunt about it. |
| Gu-2025-When-Attention-Sink-Emerges.pdf | arXiv:2410.10781 (ICLR 2025, peer-reviewed) | The pretraining-knob study: LR, weight decay, data volume, packing, loss, positional encoding, and the sigmoid-without-normalisation result. Defines the `Sink^ε_k` metric everyone else reuses. |
| Su-2026-Attention-Sink-Survey.pdf | arXiv:2604.10098 (preprint, 103pp) | Map of the field's competing interpretations. Orientation, not evidence. |
| RanMilo-2026-Sinks-Provably-Necessary.pdf | arXiv:2603.11487 (preprint, ACL format) | The theoretical floor under every "remove the sink" proposal: under sum-to-one normalisation, trigger-conditional behaviour *necessarily* induces a sink. |
| RanMilo-2026-Mechanistic-Account-Sinks-GPT2.pdf | arXiv:2604.14722 (preprint, ACL format) | A full parameter-level sink circuit in GPT-2 (query bias × first-layer-MLP positional output × key projection), with ten interventions and three matched controls (Table 1). Its own finding is that each component **is** necessary in GPT-2; the architecture-specificity caveat is cited, not measured. |
| Fesser-2026-Unifying-View-Sinks-Two-Algorithms.pdf | arXiv:2606.08105 (preprint, Kempner Institute) | Argues one sink signature hides two algorithms — "adaptive no-op" vs "broadcast" — so "the sink" may not be one thing to explain. |
| Parodi-2026-Zero-Ablation-Overstates-Register-Dependence.pdf | arXiv:2604.14433 (preprint) | **The methodological null this project most needs**: zero-ablation of ViT registers reads as catastrophic, while mean-, noise- and shuffle-substitution controls largely preserve performance. |
| Darcet-2024-ICLR-Vision-Transformers-Need-Registers.pdf | arXiv:2309.16588 (ICLR 2024, peer-reviewed) | The vision-side analogue: high-norm tokens in low-information patches, and the register fix. Sun-2024 reinterprets registers as the same fixed-bias mechanism. |
| Pierro-2024-Mamba-PTQ-Outlier-Channels.pdf | arXiv:2407.12397 (ICML 2024 workshop, non-archival) | Attention-free control: outlier **channels** do survive without softmax attention, and zeroing <1% of them drives Mamba-130m's LAMBADA to 0.00%. Says nothing about sinks or massive activations — keep the two claims separate. |
| Hammerl-2023-Anisotropy-Outliers-Multilingual-LMs.pdf | arXiv:2306.00458 (preprint) | The only **multilingual** outlier study in either folder: XLM-R, mBERT and multilingual S-BERT across 36 languages. Contradicts the prior claim that mBERT has no outliers, and §6.4 is the field's one explicit admission that the 3σ criterion has no null. |
|  Hodgkinson-2025-Heavy-Tailed-Mechanistic-Universality.pdf | arXiv:2506.03470 (ICML 2025, peer-reviewed) | The random-matrix side of the null question: an ensemble that generates heavy-tailed weight spectra. Calibrates *eigenvalue* outliers, not individual entries — see NOTES §4 for why that gap matters. |

### Related, filed elsewhere in this repo

Not duplicated here — read them where they live.

|---|---|
| Bondarenko 2023, *Quantizable Transformers* (NeurIPS 2023) — the "no-op attention" origin, clipped softmax + gated attention | Bondarenko-2023-Quantizable-Transformers.pdf |
| Barbero 2025, *Why do LLMs attend to the first token?* (COLM 2025) — the over-mixing account | Barbero-2025-Why-LLMs-Attend-First-Token.pdf |
| Xiao 2023, *StreamingLLM* — where "attention sink" was named | Xiao-2023-Efficient-Streaming-LMs-Attention-Sinks.pdf |
| Yona 2025, *Interpreting the Repeated Token Phenomenon* (ICML 2025) | Interpreting-2025-Repeated-Token-Phenomenon.pdf |
| Timkey & van Schijndel 2021, *All Bark and No Bite: Rogue Dimensions* | Timkey-vanSchijndel-2021-All-Bark-No-Bite-Rogue-Dimensions.pdf |
| Mutisya 2026, *Attention Sinks in Massively Multilingual NMT* (NLLB-200) | `AttentionSinks-2026-Multilingual-NMT-NLLB.pdf` |
| Cherilyn 2026, *Supernodes and Halos: Loss-Critical Hubs in FFN Layers* | Supernodes-2026-Loss-Critical-Hubs-FFN.pdf |
| Yang 2024, *Mitigating Quantization Errors due to Activation Spikes in GLU-based LLMs* | Yang-2024-Activation-Spikes-GLU-Variants.pdf |

## Repetition, degeneration, and MT hallucination

Reading notes: [`notes_repetition_degeneration.md`](notes_repetition_degeneration.md)

Reading list assembled 2026-09-05 for the super-weight project's question: **after
zeroing one scalar in `mlp.down_proj`, why does greedy decoding collapse into
repeated function words** (OLMo-1B → "We. We. We."; Mistral-7B → "the main.
without without. . . ."), and why do stopword probabilities rise (Yu et al. 2024,
Fig. 5)?

PDFs are gitignored repo-wide; this table is the committed record. Every file
below was downloaded from the listed source and its first page checked against the
title. Re-download with `fetch.sh` in the parent directory or per-file via the ID.

**Peer-review status is stated for each entry.** Five of the 26 are unrefereed
preprints; four of those are 2025–2026 and are used below only as leads, not as
evidence.

| File | Source | Venue / peer review | One line: why it is here |
|---|---|---|---|
| Holtzman-2019-Curious-Case-Neural-Text-Degeneration.pdf | arXiv:1904.09751 | ICLR 2020 — refereed | The founding statement that maximization decoding falls into repetitive loops; the "unreliable tail" and the positive-feedback observation. |
| Welleck-2019-Unlikelihood-Training.pdf | arXiv:1908.04319 | ICLR 2020 — refereed | Argues the likelihood objective itself over-assigns mass to repeats **and to frequent words** — the healthy-model version of the stopword shift. |
| Fu-2021-Theoretical-Analysis-Repetition-Problem.pdf | arXiv:2012.14660 | AAAI 2021 — refereed | The "high inflow" theory: many contexts predict the same word, so trajectories funnel into it and loop. High-inflow words are function words. |
| Xu-2022-Learning-to-Break-the-Loop.pdf | arXiv:2206.02369 | NeurIPS 2022 — refereed | Quantifies self-reinforcement: P(repeat) rises monotonically with prior repetitions and saturates — even for random token strings. |
| Li-2023-Repetition-In-Repetition-Out.pdf | arXiv:2310.10226 | NeurIPS 2023 — refereed | Data-side account: degeneration tracks repetition in training data; unifies high-inflow, likelihood and self-reinforcement explanations. |
| Hiraoka-2024-Repetition-Neurons.pdf | arXiv:2410.13497 | NAACL 2025 (short) — refereed | "Repetition neurons": FFN units that ramp up after repetition onset; deactivating them suppresses repetition, activating them induces it. |
| Yao-2025-Understanding-the-Repeat-Curse.pdf | arXiv:2504.14218 | Findings of ACL 2025 — refereed | SAE "repetition features" in intermediate + final layers of GPT2-small/Gemma-2-2B/Llama-3.1-8B; deactivation mitigates the loop. |
| InductionHeadToxicity-2025-Repetition-Curse.pdf | arXiv:2505.13514 | **preprint, unrefereed** | Claims induction heads dominate the logits during repetition and crowd out other heads; also reports frequency-dependent onset. |
| Interpreting-2025-Repeated-Token-Phenomenon.pdf | arXiv:2503.08908 | **preprint (Google DeepMind), unrefereed** | Closest analogue to the super weight: a sparse set of MLP "sink neurons" creates the attention sink; repeated tokens hijack it and the model diverges. |
| Stolfo-2024-Confidence-Regulation-Neurons.pdf | arXiv:2406.16254 | NeurIPS 2024 — refereed | Entropy neurons act through the **unembedding null space + final LayerNorm scale**; token-frequency neurons move the output toward/away from the unigram distribution. |
| Voita-2023-Neurons-Dead-Ngram-Positional.pdf | arXiv:2309.04827 | Findings of ACL 2024 — refereed | What FFN neurons do in OPT 125m–66b: dead, token/n-gram detectors that also *remove* information from the residual stream, positional. |
| Cancedda-2024-Spectral-Filters-Dark-Signals-Attention-Sinks.pdf | arXiv:2402.09221 | ACL 2024 (long) — refereed | "Dark" bottom-singular-value subspace of the unembedding carries the signals that implement attention sinking; a quantitative logit-lens extension. |
| Meister-2023-Natural-Bias-for-Language-Generation.pdf | arXiv:2212.09686 | ACL 2023 (short) — refereed | LMs converge to the corpus unigram distribution within a few hundred updates; sets the *default distribution* an uninformative residual should decode to. |
| Belrose-2023-Tuned-Lens.pdf | arXiv:2303.08112 | **preprint, unrefereed** (widely used) | The tool for H1: per-layer affine probes that are more reliable and less biased than the raw logit lens. |
| Xiao-2023-Efficient-Streaming-LMs-Attention-Sinks.pdf | arXiv:2309.17453 | ICLR 2024 — refereed | Attention sinks: softmax must sum to one, so surplus attention is dumped on globally visible initial tokens; keeping 4 of them restores perplexity. |
| Gromov-2024-Unreasonable-Ineffectiveness-Deeper-Layers.pdf | arXiv:2403.17887 | ICLR 2025 — refereed | Layer pruning tolerates ~half the deep layers, then collapses at a sharp model-dependent threshold — collapse as a threshold phenomenon. |
| SignalDegradation-2026-Two-Failure-Modes-Quantization.pdf | arXiv:2604.19884 | **preprint, unrefereed** | Distinguishes "Signal Degradation" (noise, repairable) from "Computation Collapse" (components fail, signal destroyed in early layers) — the framing super-weight ablation fits. |
| Raunak-2021-Curious-Case-Hallucinations-NMT.pdf | arXiv:2104.06683 | NAACL 2021 — refereed | Defines **oscillatory hallucinations** (inadequate output containing repeating n-grams) vs detached ones. This is the MT name for the post-ablation output. |
| Guerreiro-2023-Looking-for-a-Needle-in-a-Haystack.pdf | arXiv:2208.05309 | EACL 2023 — refereed | 3.4k-annotation natural-setting study; shows most prior hallucination detectors fail and sequence log-probability is the strong baseline. |
| Guerreiro-2023-Hallucinations-Large-Multilingual-Translation.pdf | arXiv:2303.16104 | TACL 2023 — refereed | >100 directions on M2M-100/NLLB/ChatGPT; oscillatory hallucinations dominate mid/high-resource pairs; TNG heuristic detects them. |
| Dale-2023-Detecting-Mitigating-Hallucinations-MT.pdf | arXiv:2212.08597 | ACL 2023 (long) — refereed | ALTI+ **source contribution** as an internal detector; doubles detection accuracy for the most severe hallucinations. The MT-side measurement to reuse. |
| Xu-2023-Understanding-Detecting-Hallucinations-NMT-Introspection.pdf | arXiv:2301.07779 | TACL 2023 — refereed | Saliency-based introspection; finds source-contribution *patterns* (not just magnitude) are the reliable hallucination symptom. |
| Ferrando-2022-Opening-Black-Box-NMT-ALTI.pdf | arXiv:2205.11631 | EMNLP 2022 — refereed | The ALTI+ method itself: per-token attributions for source **and** target prefix in encoder-decoder Transformers. |
| Dale-2023-HalOmi-Benchmark.pdf | arXiv:2305.11746 | EMNLP 2023 — refereed | 18-direction annotated hallucination/omission benchmark on NLLB output; the evaluation set for any NLLB ablation study. |
| AttentionSinks-2026-Internal-Signals-Hallucination-Detection.pdf | arXiv:2604.10697 | **preprint, unrefereed** | The only paper found that ties hallucination *detection* to sink structure (and to sinks with large-norm value vectors) — but in decoder LLMs, not MT. |

### Added 2026-09-05, repetition–sink link

Downloaded while answering "does repetition vary by language?" — reading notes:
[`notes_multilingual_repetition.md`](notes_multilingual_repetition.md).

| File | Source (venue) | One line: why it is here |
|---|---|---|
| Barua-2026-Long-CoT-Reasoning-Across-Languages.pdf | arXiv:2508.14828 (ICLR 2026 — refereed) | The closest thing to a measured language-dependent repetition result: "output generation" errors rise 0.7% → 11.3% when reasoning in the target language rather than English — but the category conflates degeneration with format violations. |
| Liu-2026-Crosslingual-SelfDistillation-Repeat-Rate.pdf | arXiv:2605.09548 (**preprint, unrefereed**) | Source of the reusable metric RepeatRate_n (one minus the ratio of distinct to total n-grams, n∈{2..6}); also the clearest instance of the "low-resource languages repeat more" folklore citation chain. |
| Ustun-2024-Aya-Model-Instruction-Finetuned-Multilingual.pdf | arXiv:2402.07827 (ACL 2024 — refereed) | The only documented per-language report of generation "loops" in a multilingual instruction model (§5.4, App. E.6) — qualitative annotator feedback, and the languages flagged are Russian and Serbian, not the lowest-resource ones. |

### Added 2026-09-05, repetition onset and the natural-vs-lesion comparison

Downloaded while answering "is natural repetition the same failure as lesion-induced
repetition, and has anyone linked either to attention sinks?" — reading notes:
[`notes_repetition_sink_link.md`](notes_repetition_sink_link.md). *(This is a second
2026-09-05 batch, distinct from the multilingual-repetition batch above.)*

| File | Source (venue) | One line: why it is here |
|---|---|---|
| Mahaut-2025-Repetitions-Are-Not-All-Alike.pdf | arXiv:2504.01100 (**preprint, unrefereed**) | The closest existing test of the project's question, asked of a different pair: natural vs ICL-induced repetition in Pythia-1.4B across 10 checkpoints are **mechanistically different** — ICL rides a specialising head circuit, natural repetition has none and attends 9.9× to newlines. |
| Duan-2026-Circular-Reasoning-Self-Reinforcing-Loops.pdf | arXiv:2601.05693 (**preprint, unrefereed**) | The only paper found that measures attention *before* the first repeated token (50 loop vs 50 non-loop, windows 64–512) — and its Fig. 6 caption says attention sinks are excluded from that measurement. |
| Xu-2026-LoopGuard-Self-Reinforcing-Attention-Loops.pdf | arXiv:2604.10044 (**preprint, unrefereed**) | Attributes repetition to attention "head locking" onto a narrow suffix, and shows attention-based KV-cache eviction *amplifies* loops by preferentially keeping repetitive tokens; supplies LoopBench and an onset criterion. Never mentions sinks. |
| Marwah-2026-Cognitive-Fatigue-Autoregressive-Transformers.pdf | arXiv:2605.30981 (ICML 2026 — refereed; PDF footer reads "PMLR 306, 2026") | The only per-decoding-step internal time series validated against repetition (ρ=0.94 over full generations, ρ≈0.4 over the first 20 tokens); includes an FP16-vs-4-bit arm where prompt attention is unchanged and only entropy collapses further. |
| Lee-2026-FOCUS-RePAIR-Pruned-LLM-Degeneration.pdf | arXiv:2608.26676 (**preprint, unrefereed**) | The same-model lesion-vs-intact repetition comparison: unpruned → pruned Llama, 5.9 → 12.4/15.4% (top-p) and 26.6 → 63.1/63.7% (greedy) at near-unchanged perplexity; decomposes degeneration into loop *entry risk* and loop *persistence*. |
| Roll-2026-Artificial-Aphasias-Lesioned-LMs.pdf | arXiv:2605.16222 (**preprint, unrefereed**) | The only large-n characterisation of lesioned-model *text* (112,426 scored generations, 5 models): FFN-Gate lesions give the highest repetition-loop rate, and a repetition penalty of 1.2 at T=0.7 takes it to zero. |
| Mohammadshahi-2022-Compressed-Multilingual-MT-Forget.pdf | arXiv:2205.10828 (Findings of EMNLP 2022 — refereed) | The one compressed-NMT study that measures both directions on the same M2M-100: off-target 5.2 → 17.5 under 8-bit, cross-attention alignment λ = 1.96–3.01 on losing pairs but **0.15–0.52 on winning pairs** (the baseline hallucinates, the compressed model does not). Measures hallucination and off-target rate, **not** repetition, so the oscillatory-hallucination gap stands. |

### Anchors read outside this folder (not re-downloaded)

- `Yu-2024-The-Super-Weight-in-LLMs.pdf` — §3.2 and Fig. 5.
- `Sun-2024-COLM-Massive-Activations-in-LLMs.pdf` — §3–4, the bias-term interpretation.
- `Puccetti-2022-Outlier-Dimensions-Driven-by-Frequency.pdf` — §4.1 / Fig. 3, the closest existing precedent for "ablate an outlier → predictions shift toward high-frequency tokens".

### Considered and not downloaded

- Olsson et al. 2022, *In-context Learning and Induction Heads* — on `transformer-circuits.pub`, not a PDF venue; unrefereed. **[unopened]** Cited only through papers that operationalize it.
- arXiv:2507.07810, *Understanding and Controlling Repetition Neurons and Induction Heads in In-Context Learning* — plausible follow-up to Hiraoka & Inui. **[unopened]**
- arXiv:2601.20520, *Context Tokens are Anchors: … Repetition Curse in dMLLMs* — diffusion multimodal LMs, off-architecture. **[unopened]**
- arXiv:2607.09999, *Silent Failures in Quantized LLM Reasoning* — accuracy-preserving reasoning shifts, not degenerate output. **[unopened]**

## Compression: pruning and quantization

Reading notes: [`notes_compression_pruning_quantization.md`](notes_compression_pruning_quantization.md)

Papers on **how compression (quantization, pruning) interacts with the outlier
structure that super weights create**. The organising question for this folder is
*not* "how do we protect the super weight" (that framing is off the table, see
`notes_compression_pruning_quantization.md` §5 preamble) but the chain

> super weight (scalar in an early `mlp.down_proj`) → super/massive activation
> (token-wise outlier, constant magnitude to the last layer) → the outlier
> structure that every quantizer and pruner has to deal with.

`notes_compression_pruning_quantization.md` holds the paper-by-paper reading notes (§1), the map of where
interpretability findings are *already* used inside compression methods (§2),
what is known about over-compressed models' failure mode (§3), the
multilingual/MT gap (§4), and candidate open problems (§5).

**Venue column convention.** "Venue" is what the PDF itself prints on its pages.
Where the PDF prints nothing, the entry says so — I did not fill it in from
memory. Treat those as *preprint, publication status unverified*.

|---|---|---|---|
| `TrainingDynamics-2025-PTQ-Robustness.pdf` | arXiv:2510.06213 | ICLR 2026 (printed on page) — peer-reviewed | PTQ error is driven by the LR schedule, not token count; spikes exactly at LR decay — and it never once looks at outliers, which is the opening. |
| `NVFP4-2026-Outlier-Dynamics-Pretraining.pdf` | arXiv:2602.02047 | "Preprint. February 3, 2026." — not peer-reviewed | The only longitudinal study of *where outliers live and how they move* during low-precision pretraining; outliers settle into fixed "hot channels". |
| `UnevenPTQ-2025-Multilingual-MT-Quantization.pdf` | arXiv:2508.20893 | no venue line — preprint | The MT-side measurement Adrian replicated: per-language COMET damage under 4- and 2-bit PTQ across 55 languages, with no mechanistic account. |
| `Marchisio-2024-Quantization-Multilingual-LLMs.pdf` | arXiv:2407.03211 | no venue line — preprint (widely cited as EMNLP 2024 Findings; **unverified from this PDF**) | First broad multilingual quantization study; automatic metrics understate human-judged damage ~10x, non-Latin scripts hurt most. |
| `Gromov-…` *(not here — see `Gromov-2024-Unreasonable-Ineffectiveness-Deeper-Layers.pdf`)* | arXiv:2403.17887 | ICLR 2025 (printed) — peer-reviewed | Deep layers deletable up to ~half the model; shallow layers are the ones that matter. Not duplicated into this folder. |
| `Men-2024-ShortGPT-Layer-Redundancy.pdf` | arXiv:2403.03853 | no venue line — preprint | Block Influence layer-importance metric; explicitly reports that *initial layers and the last layer* are the critical ones. |
| `Yang-2024-LaCo-Layer-Collapse-Pruning.pdf` | arXiv:2402.11187 | no venue line — preprint | Layer-collapse structured pruner; its Table 23 is one of the few printed examples of over-pruned models emitting repetitive degenerate text. |
| `Kim-2024-Shortened-LLaMA-Depth-Pruning.pdf` | arXiv:2402.02834 | no venue line — preprint | Depth vs width pruning head-to-head; depth pruning wins under memory-bound inference, and retraining is what recovers quality. |
| `He-2024-Not-All-Attention-Is-Needed.pdf` | arXiv:2406.15786 | no venue line — preprint | Attention layers are far more redundant than MLP layers (half of Llama-2-70B's attention removable); redundancy is stable across training checkpoints. |
| `Ashkboos-2024-SliceGPT.pdf` | arXiv:2401.15024 | ICLR 2024 (printed) — peer-reviewed | Structured slicing via computational invariance — the same rotation trick QuaRot later uses for outliers, here used for size. |
| `Siddiqui-2024-Deeper-Look-Depth-Pruning.pdf` | arXiv:2407.16286 | ICML 2024 TF2M workshop (printed) — workshop-reviewed | Splits block pruning into attention vs FFN: 33% of self-attention layers removable with no MMLU loss; FFN is not. |
| `Lad-2024-Remarkable-Robustness-Stages-of-Inference.pdf` | arXiv:2406.19384 | "Preprint. Under review." — not peer-reviewed | The clearest statement that **first-layer intervention is catastrophic in every model family** — and it explains that by detokenization, *not* by massive activations. |
| `Frantar-2023-SparseGPT.pdf` | arXiv:2301.00774 | no venue line (ICML 2023 per the reference list's self-citation style; **unverified**) | The OBS-style one-shot pruner; the baseline every later metric is measured against. |
| `Sun-2023-Wanda-Simple-Effective-Pruning.pdf` | arXiv:2306.11695 | ICLR 2024 (printed) — peer-reviewed | Importance = `|W_ij| · ‖X_j‖₂`, motivated *explicitly* by Dettmers' outlier features — the metric most likely to already "see" a super weight. |
| `Yin-2024-OWL-Outlier-Weighed-Layerwise-Sparsity.pdf` | arXiv:2310.05175 | ICML 2024 (printed) — peer-reviewed | Allocates per-layer sparsity proportional to a Layerwise Outlier Distribution built from Wanda's score — outlier structure used as a *budget allocator*. |
| `Zhang-2024-RIA-Plug-and-Play-Pruning.pdf` | ICLR 2024 proceedings PDF | ICLR 2024 (printed) — peer-reviewed | Relative Importance × Activations: fixes Wanda's tendency to wipe whole channels, which is exactly the failure mode a super-weight column would suffer. |
| `Dong-2024-Pruner-Zero.pdf` | arXiv:2406.02924 | ICML 2024 (printed) — peer-reviewed | Genetic-programming search over symbolic pruning metrics; the closest thing to an exhaustive answer to "is `|W|·‖X‖` the right score". |
| `Lu-2024-AlphaPruning-HeavyTailed-Layerwise.pdf` | arXiv:2410.10912 | no venue line — preprint | The non-outlier competitor to OWL: layerwise sparsity from ESD heavy-tail exponents. The control condition for "is it really outliers?" |
| Supernodes-2026-Loss-Critical-Hubs-FFN.pdf | arXiv:2604.23475 | no venue line — preprint | Cites Yu et al. directly; finds top-1% FFN channels carry ~59% of loss sensitivity **but overlap only weakly with activation-defined outliers**. |
| `GarbageAttention-2026-BOS-Sink-Heads-Sink-Aware-Pruning.pdf` | arXiv:2601.06787 | no venue line — preprint | Uses the `<BOS>` attention-sink score as a *structural pruning metric*: sink-heavy deep heads are redundant. Interpretability signal → pruning decision. |
| `Frantar-2022-GPTQ.pdf` | arXiv:2210.17323 | ICLR 2023 (printed) — peer-reviewed | The Hessian-based weight quantizer used as the workhorse in the TrainingDynamics paper and most PTQ baselines. |
| `Lin-2023-AWQ.pdf` | arXiv:2306.00978 | MLSys 2024 (printed) — peer-reviewed | Salient weight *channels* chosen by activation magnitude, protected by per-channel scaling — the closest mainstream method to "protect the important weights". |
| `Xiao-2023-SmoothQuant.pdf` | arXiv:2211.10438 | ICML 2023 (printed) — peer-reviewed | Migrates per-channel activation outliers into the weights by a smoothing factor; the baseline Yu et al. beat with RTN + super-activation handling. |
| `Shao-2024-OmniQuant.pdf` | arXiv:2308.13137 | ICLR 2024 (printed) — peer-reviewed | Learns the clip range for weight outliers and an equivalent transform for activation outliers instead of hand-setting them. |
| `Wei-2023-Outlier-Suppression-Plus.pdf` | arXiv:2304.09145 | no venue line — preprint (EMNLP 2023 per common citation; **unverified**) | Channel-wise shifting + scaling for *asymmetric* activation outliers; the "outliers are per-channel" worldview in its purest form. |
| `Ashkboos-2024-QuaRot.pdf` | arXiv:2404.00456 | NeurIPS 2024 (printed) — peer-reviewed | Randomized Hadamard rotations make the residual stream outlier-free, enabling W4A4KV4. The key question for this project: what happens to the super activation under rotation? |
| `Liu-2024-SpinQuant.pdf` | arXiv:2405.16406 | ICLR 2025 (printed) — peer-reviewed | Learned rotations beat random ones by up to 13 points — i.e. *which* rotation you pick matters, which implies the outlier being rotated is structured. |
| `Lin-2024-DuQuant.pdf` | arXiv:2406.01721 | NeurIPS 2024 (printed) — peer-reviewed | The one method that names the distinction this project cares about: "Normal Outliers" (per-channel) vs "Massive Outliers" (per-token, huge). |
| `Chen-2024-PrefixQuant.pdf` | arXiv:2410.05265 | no venue line — preprint | Cites Sun et al. 2024 and treats massive activations as *token-wise outliers*, removing them by prefixing the outlier tokens into the KV cache. |
| Bondarenko-2023-Quantizable-Transformers.pdf | arXiv:2306.12929 | NeurIPS 2023 (printed) — peer-reviewed | Outliers exist because attention heads need a no-op; clipped softmax / gated attention make models that never grow them. The causal-architecture answer. |
| Yang-2024-Activation-Spikes-GLU-Variants.pdf | arXiv:2405.14428 | "Preprint. Under review." — not peer-reviewed | Activation spikes in GLU FFNs concentrate in *specific early and late layers* and on *a couple of tokens* — the closest quantization paper to the super-activation picture. |
| `Wang-2025-Task-Circuit-Quantization.pdf` | arXiv:2504.07389 | no venue line — preprint | Mixed-precision PTQ that keeps a gradient-identified "task circuit" in 16 bits; the clearest existing template for "interpretability → bit allocation". |
| Achilles-2025-Altering-Neurons-Cripples-Language.pdf | arXiv:2510.10238 | ICLR 2026 (printed) — peer-reviewed | Ultra-sparse critical *neuron sets* concentrated in `mlp.down_proj`, with sharp phase transitions — the published paper closest to Adrian's joint-ablation result. |
| `Koishekenov-2023-NLLB200-Language-Specific-Expert-Pruning.pdf` | arXiv:2212.09811 | no venue line — preprint (ACL 2023 per common citation; **unverified**) | The only NLLB-200 compression paper here; 80% of MoE experts removable, and the pruning metric identifies *language-specific* experts. |
| `Zhang-2024-Multilingual-Brain-Surgeon.pdf` | arXiv:2404.04748 | no venue line — preprint | Makes the calibration set multilingual in proportion to the training language distribution; the direct answer to "English calibration hurts low-resource languages". |
| `Ogueji-2022-Intriguing-Properties-Compression-Multilingual.pdf` | arXiv:2211.02738 | no venue line — preprint | Sparsity on mBERT NER over 40 languages; the *counter*-result that compression sometimes helps low-resource languages and improves robustness. |
| `Williams-2024-Calibration-Data-Pruning-Quantization.pdf` | arXiv:2311.09755 | no venue line — preprint | Calibration data choice moves downstream numbers substantially — a confound that any per-language PTQ experiment must control. |

Read-only companions (same folder):
`Dettmers-2022-LLM-int8.pdf`,
`Yu-2024-The-Super-Weight-in-LLMs.pdf` (§4),
`Sun-2024-COLM-Massive-Activations-in-LLMs.pdf`,
`Sun-2026-Spike-Sparse-Sink.pdf`,
`QueipoDeLlano-2025-Sinks-Compression-Valleys-Same-Coin.pdf`,
`SignalDegradation-2026-Two-Failure-Modes-Quantization.pdf`,
`Gromov-2024-Unreasonable-Ineffectiveness-Deeper-Layers.pdf`,
`Tang-2024-Language-Specific-Neurons-LAPE.pdf`.

## Multilingual and MT interpretability

Reading notes: [`notes_multilingual_mt_interpretability.md`](notes_multilingual_mt_interpretability.md)

Scope: where language identity lives inside decoder LLMs and encoder-decoder NMT
models, and what that implies for a **single-scalar super-weight** project. Sibling
folders hold the adjacent literatures — ``,
``, ``,
``. Do not duplicate those here.

PDFs are gitignored repo-wide; this table is the committed record. Analysis lives in
[`notes_multilingual_mt_interpretability.md`](notes_multilingual_mt_interpretability.md).

**Peer-review status is stated as verified**, from ACL Anthology listings or the
arXiv `comment`/`journal_ref` field. "arXiv preprint" below means *no venue was
found* — not that one does not exist.

### Core to this project

| File | Source | Venue / review | One line: why it is here |
|---|---|---|---|
| Tang-2024-Language-Specific-Neurons-LAPE.pdf | arXiv:2402.16438 | ACL 2024 (peer-reviewed) | The occupied territory: LAPE finds language-specific FFN neurons in bottom+top layers; deactivating 1% collapses that language's PPL. Defines what the project may **not** claim. |
| Zhao-2024-How-LLMs-Handle-Multilingualism.pdf | arXiv:2402.18815 | NeurIPS 2024 (peer-reviewed) | MWork: understand → reason-in-English → generate. Deactivating **0.13%** of neurons drops XLSum by 99%. NB: the "as few as four neurons" figure is **not in this paper** (see NOTES §1). |
| AttentionSinks-2026-Multilingual-NMT-NLLB.pdf | arXiv:2605.01229 | arXiv preprint, **not peer-reviewed** | The NLLB cross-attention sink paper. Its sink is `</s>` (78–87%), **not** the language tag (1.5–2%). Contains internal contradictions — see NOTES §1. Read before repeating its premise. |
| Language-Lives-in-Sparse-Dimensions-2025.pdf | arXiv:2510.07213 | **EACL 2026 Main** (peer-reviewed) | Closest prior art to a "single coordinate controls language" claim: a small sparse set of **residual dimensions** at consistent indices governs the cross-lingual transition; training-free, beats neuron methods. |
| Rajaee-Pilehvar-2022-Isotropy-Multilingual-BERT.pdf | arXiv:2110.04504 | Findings of ACL 2022 | mBERT shows **no outlier dimension** while monolingual BERT does. The strongest existing reason to expect the multilingual outlier story to differ. |
| Timkey-vanSchijndel-2021-All-Bark-No-Bite-Rogue-Dimensions.pdf | arXiv:2109.04404 | EMNLP 2021 | 1–3 "rogue" dimensions dominate cosine similarity yet mismatch what drives model behaviour. The methodological null for any "this channel matters" claim. |

### Where LLMs "think" — latent language and pivoting

|---|---|---|---|
| Wendler-2024-Do-Llamas-Work-in-English.pdf | arXiv:2402.10588 | ACL 2024 (peer-reviewed) | The anchor English-pivot result; three phases (input → concept → output space) via logit lens. |
| Wu-2025-Semantic-Hub-Hypothesis.pdf | arXiv:2411.04986 | ICLR 2025 (peer-reviewed) | Shared representation space across languages *and* modalities, with cross-lingual interventions that propagate — causal, not just geometric. |
| Schut-2025-Do-Multilingual-LLMs-Think-In-English.pdf | arXiv:2502.15603 | arXiv preprint | Extends the pivot claim to open-ended multi-token generation; lexical words route through English, non-lexical words do not. |
| Trinley-2025-What-Languages-Does-Aya-23-Think-In.pdf | arXiv:2507.20279 | arXiv preprint ("pre-print") | Balanced-data Aya-23 activates typologically *related* languages instead of one English pivot; its language-specific neurons sit in **final** layers. Directly relevant since Aya is a local model. |
| Bafna-2025-Translation-Barrier-Hypothesis.pdf | arXiv:2506.22724 | IJCNLP-AACL 2025 (peer-reviewed) | Task-solving succeeds, "decode-out" fails; the translation barrier explains most error across 108 pairs, worst for low-resource targets. |
| Tezuka-2025-Transfer-Neurons-Hypothesis.pdf | arXiv:2509.17030 | EMNLP 2025 Main (peer-reviewed) | MLP neurons that *move* representations between language-specific and shared latent spaces; a more causal account than LAPE. |
| Kojima-2024-Finding-Controlling-Language-Specific-Neurons.pdf | arXiv:2404.02431 | NAACL 2024 (peer-reviewed) | Language-specific neurons overlap <5% across languages, concentrate in first/last layers; <1% tampering flips output language. |
| Mondal-2025-LangSpecific-Neurons-Do-Not-Facilitate-Transfer.pdf | arXiv:2503.17456 | Insights from Negative Results @ NAACL 2025 (peer-reviewed) | The negative result: LAPE-neuron interventions and neuron-LoRA give <1 point on XNLI/XQuAD. The anti-over-claiming citation. |
| Wang-2024-Sharing-Matters-Neurons-Across-Languages-Tasks.pdf | arXiv:2406.09265 | arXiv preprint | Counterweight to LAPE: **all-shared** neurons matter most; deactivating them hurts more than language-specific ones. |
| Tracing-Multilingual-Representations-CrossLayer-Transcoders-2025.pdf | arXiv:2511.10840 | arXiv preprint (under review) | CLT + attribution graphs on models trained with/without English; language identity is linearly encoded early, decoded by a **small set of high-frequency late-layer features**. |
| Mechanistic-Language-Confusion-English-Centric-2025.pdf | arXiv:2505.16538 | EMNLP 2025 Findings (peer-reviewed) | Confusion points localized to final-layer transition failures; editing a small neuron set mitigates them. |
| Marchisio-2024-Language-Confusion-in-LLMs.pdf | arXiv:2406.20052 | EMNLP 2024 (peer-reviewed) | The Language Confusion Benchmark; the standard behavioural measure of "wrong language out". |

### MT internals, hallucination, off-target

|---|---|---|---|
| Voita-2021-Source-Target-Contributions-NMT.pdf | arXiv:2010.10907 | ACL 2021 (peer-reviewed) | LRP-based source-vs-target contribution; the measure that makes "detached from the source" quantitative. |
| Ferrando-2022-ALTI-Measuring-Mixing-Contextual-Information.pdf | arXiv:2203.04212 | EMNLP 2022 (peer-reviewed) | ALTI: token-to-token mixing over the whole attention block, not attention weights alone. |
| Dale-2023-Detecting-Mitigating-Hallucinations-MT.pdf | arXiv:2212.08597 | ACL 2023 (peer-reviewed) | Low source contribution detects the most severe hallucinations 2x better; internals alone go a long way. |
| OffTarget-2023-ZeroShot-Multilingual-NMT.pdf | arXiv:2305.10930 | Findings of ACL 2023 | Off-target traced to failure to encode a discriminative target-language signal; vocabulary KL predicts the rate. |
| Wu-2021-Language-Tags-Matter-ZeroShot-NMT.pdf | arXiv:2106.07930 | Findings of ACL 2021 | Language-tag *placement* (encoder vs decoder side) changes zero-shot behaviour materially. |
| Registering-Source-Tokens-Target-Language-Spaces-2025.pdf | arXiv:2501.02979 | ACL 2025 Main (peer-reviewed) | "Registers" — artificial target-language tokens the decoder attends to exclusively; an engineered version of a language-tag sink. |
| DecoderLens-2023-Layerwise-Interpretation-EncoderDecoder.pdf | arXiv:2310.03686 | Findings of NAACL 2024 | Logit-lens equivalent for **encoder-decoder** models; the method that makes NLLB layer-wise analysis possible. |
| Universal-Conceptual-Structure-NLLB200-Geometry-2026.pdf | arXiv:2603.02258 | arXiv preprint | Closest existing NLLB-internals study: language-neutral conceptual store + per-language offsets. Swadesh word lists only, geometric, no interventions — weak effects (ρ=0.13). |

### Model cards for the local models

|---|---|---|---|
| NLLB-2022-No-Language-Left-Behind.pdf | arXiv:2207.04672 | arXiv tech report (190 pp) | Architecture reference. §8 confirms: **source** language token prefixes the source; **target** language token is the *first decoder input token*. |
| Alves-2024-Tower-Open-Multilingual-LLM.pdf | arXiv:2402.17733 | COLM 2024 (peer-reviewed) | TowerBase/TowerInstruct recipe; the models whose shared super-weight coordinate was already observed in-repo. |
| EuroLLM-2025-9B-Technical-Report.pdf | arXiv:2506.04079 | arXiv tech report | EuroLLM-9B architecture, tokenizer, data — the model whose super-weight ablation dropped en→X chrF++ 57.9 → 4.8. |

### Multilingual formation / training dynamics

|---|---|---|---|
| Blevins-2022-Mono-CrossLingual-Pretraining-Dynamics.pdf | arXiv:2205.11758 | EMNLP 2022 (peer-reviewed) | XLM-R checkpoints: in-language ability early, cross-lingual transfer onset varies by pair; final layer *degrades* over training. |
| Probing-Emergence-CrossLingual-Alignment-2024.pdf | arXiv:2406.13229 | Findings of ACL 2024 | BLOOM checkpoints; cross-lingual neuron overlap correlates with transfer, with mid-training degradation phases. |
| CopyFirstTranslateLater-2026-Translation-Dynamics-Pretraining.pdf | arXiv:2604.17633 | arXiv preprint | Dense-checkpoint 1.7B multilingual pretrain; translation develops in two phases (copying first, generalizing translation later). |

### Copies from the interlingua track

Copies of `../../interlingua/papers/<same filename>`. Detailed summaries already exist
in `../../interlingua/papers/paper_summaries.md` (grep, do not read whole).

|---|---|---|---|
| Pires-2019-How-Multilingual-is-Multilingual-BERT.pdf | ACL 2019 | peer-reviewed | Foundational mBERT zero-shot transfer result. |
| Wu-Dredze-2019-Beto-Bentz-Becas.pdf | EMNLP 2019 | peer-reviewed | mBERT transfer across 39 languages; layer-wise findings. |
| Chi-2020-Universal-Grammatical-Relations-in-mBERT.pdf | ACL 2020 | peer-reviewed | Syntactic subspace shared across languages. |
| Choenni-Shutova-2020-What-Does-It-Mean-LanguageAgnostic.pdf | arXiv | preprint | Probing what "language-agnostic" actually means in multilingual encoders. |
| Wang-2020-Crosslingual-Transferability-of-Monolingual-Reps.pdf | ACL 2020 | peer-reviewed | Monolingual representations transfer cross-lingually. |
| Zhao-2020-Inducing-LanguageAgnostic-Multilingual-Reps.pdf | arXiv | preprint | Methods for inducing language-agnostic representations. |
| Karthikeyan-2020-Crosslingual-Ability-of-Multilingual-BERT.pdf | ICLR 2020 | peer-reviewed | Ablates what drives mBERT transfer (not lexical overlap). |
| Goldman-2025-ECLeKTic-Crosslingual-Knowledge-Transfer.pdf | arXiv:2502.21228 | preprint (Google Research) | Closed-book QA isolating facts learnt in one language; SOTA models fail to share knowledge across languages. |
| Hu-2020-XTREME-Benchmark.pdf | ICML 2020 | peer-reviewed | The standard cross-lingual transfer benchmark suite. |
| Oncevay-2020-Bridging-Linguistic-Typology-and-MT.pdf | ACL 2020 | peer-reviewed | Typology features for MT; source of language-distance covariates. |
| Conneau-2018-What-You-Can-Cram-Into-a-Single-Vector.pdf | ACL 2018 | peer-reviewed | Probing-task methodology origin. |
| Conneau-2018-Cram-Single-Vector-SLIDES.pdf | ACL 2018 slides | — | Slide deck for the above. |
| EACL-2026-long-32.pdf | EACL 2026 | peer-reviewed | Jian & Manning, abstraction drives language learning: class-level behaviour precedes item-level; behaviours emerge abruptly in sequence. |
| arXiv-2602.02385-Factored-Representations.pdf | arXiv:2602.02385 | preprint | Transformers learn factored representations in orthogonal residual subspaces — the theoretical case that a language factor could occupy its own low-dimensional subspace. |
| Miaschi-2020-Contextual-and-NonContextual-Embeddings.pdf | COLING 2020 | peer-reviewed | Linguistic profiling of contextual vs non-contextual embeddings. |

## Training dynamics and checkpoint suites

Reading notes: [`notes_training_dynamics_checkpoint_suites.md`](notes_training_dynamics_checkpoint_suites.md)

Papers for **RQ1** (does super-weight criticality form gradually or abruptly during
pretraining?) and **RQ4** (what causes formation?). Two kinds of reading:

1. **Instruments** — public checkpoint suites you could actually trace a super weight
   through, and what each one publishes.
2. **Method** — how the literature measures "abrupt vs gradual", what a seed population
   does to that question, and which recipe knobs are known to create or abolish the
   neighbouring phenomena (massive activations, attention sinks, outlier channels).

Details, numbers and the read/inferred split are in [`notes_training_dynamics_checkpoint_suites.md`](notes_training_dynamics_checkpoint_suites.md).
Read-only companions live in ``,
``, ``.

| File | Source | Venue / peer-review status | One line: why it is here |
|---|---|---|---|
| `Biderman-2023-Pythia.pdf` | arXiv:2304.01373v2 | ICML 2023 — **peer-reviewed** (venue not stamped on this copy) | The reference decoder suite: 8 sizes × 2 datasets, **154 checkpoints/run** with a log-spaced early grid (steps 1…512) and an exact-dataloader replay script. |
| `vanderWal-2025-PolyPythias.pdf` | arXiv:2503.09543v2 | ICLR 2025 — **peer-reviewed** (stamped) | 45 extra Pythia runs = **10 seeds × 5 sizes (14M–410M)**, same 154-checkpoint grid; the only public suite that can ask "same super weight across seeds?". |
| `Sellam-2022-MultiBERTs.pdf` | arXiv:2106.16163v2 | ICLR 2022 — **peer-reviewed** (stamped) | 25 BERT-Base seeds; **only 5 have intermediate checkpoints (28 each)**. The multi-seed precedent, and the encoder counter-example to Pythia. |
| `Ettin-2025-Paired-Encoders-Decoders.pdf` | arXiv:2507.11412v2 | ICLR 2026 — **peer-reviewed** (stamped) | Encoder and decoder trained on identical data/order/recipe, 17M–1B, **236 checkpoints every 8.5B tokens**. The encoder-vs-decoder arm — but no early-training density. |
| `DataDecide-2025-Predict-Pretraining-Data.pdf` | arXiv:2504.11393v2 | ICML 2025 — **peer-reviewed** (venue not stamped on this copy) | 25 data recipes × 14 sizes × 3 seeds, 30k+ checkpoints — the only public *data-recipe* sweep with seeds, but seeds 2–3 stop at 25% of budget below 1B. |
| `Groeneveld-2024-OLMo-Accelerating-Science.pdf` | arXiv:2402.00838v4 | no venue stamp on this copy; ACL 2024 per publisher listing, **not verified from the PDF** | OLMo-1. The model family in which Yu et al. located a super weight; documents the checkpoint/data release that RQ1's backward trace depends on. |
| `OLMo2-2025-2-OLMo-2-Furious.pdf` | arXiv:2501.00656v3 | preprint / tech report | OLMo-2 checkpoint release (stage1/stage2 revisions, incl. an `-early-training` repo) and the QK-norm + reordered-norm recipe changes that plausibly alter outlier formation. |
| `Liu-2023-LLM360-Fully-Transparent-LLMs.pdf` | arXiv:2312.06550v1 | preprint copy, no venue stamp; COLM 2024 per publisher listing, **not verified from the PDF** | Amber (7B) and CrystalCoder: **360 checkpoints** plus the full data sequence — a second fully-replayable decoder trace outside Pythia. |
| `SmolLM2-2025-When-Smol-Goes-Big.pdf` | arXiv:2502.02737v1 | **preprint, "Under review"** (stamped) | Modern small-model recipe (WSD-ish, 11T tokens) with intermediate checkpoints on the Hub; the 135M/1.7B pair is the closest public thing to "is there a super weight under 1B?". |
| `Zhang-2024-TinyLlama.pdf` | arXiv:2401.02385v2 | preprint | 1.1B / 3T tokens, checkpoint releases; a cheap sub-1B decoder to test the "does the phenomenon exist at trainable scale?" gate. |
| `Martins-2024-EuroLLM-Multilingual-Europe.pdf` | arXiv:2409.16235v1 | preprint / tech report | **MT bridge.** 35-language 1.7B/9B trained with explicit parallel data; the suite whose 26 intermediate checkpoints the paper below analyses. |
| `WhenMeaningsMeet-2026-Shared-Concept-Spaces-Multilingual-Training.pdf` | arXiv:2601.22851v1 | preprint | The only paper here that traces a *multilingual* representational property across pretraining checkpoints (EuroLLM ×26, plus Apertus and OLMo-2); tells you what multilingual checkpoint granularity actually exists. |
| `Apertus-2025-Open-Compliant-Multilingual-LLMs.pdf` | arXiv:2509.14233v2 | preprint / tech report | 1811-language 8B/70B with **45 public pretraining revisions** — the largest multilingual checkpoint trace available, but its first checkpoint is already 210B tokens in. |
| `Lucie-2025-Lucie-7B-Multilingual-Open-Resources.pdf` | arXiv:2503.12294v1 | preprint | French-centric multilingual 7B with **35 public step-revisions** from step 5,000; a second multilingual instrument, denser early than Apertus. |
| `MAP-Neo-2024-Bilingual-Transparent-LLM.pdf` | arXiv:2405.19327v4 | preprint | Bilingual (Chinese–English) 7B released "360-style" with intermediate checkpoints; a non-European multilingual option. |
| Bondarenko-2023-Quantizable-Transformers.pdf | arXiv:2306.12929v2 | NeurIPS 2023 — **peer-reviewed** (stamped) | The causal existence proof that outliers are a *recipe* property: clipped softmax / gated attention trained from scratch cut max-∞-norm 735→21 (BERT). **OPT-125m pretraining on one A100, ~54 h.** |
| Qiu-2025-Gated-Attention-Sink-Free.pdf | arXiv:2505.06708v1 | NeurIPS 2025 Oral/Best Paper per publisher listing; **this copy carries no venue stamp** | Trains sink-free models at 1.7B/15B-MoE scale: head-specific sigmoid gate on SDPA drops max hidden activation 1053→94 and first-token attention 0.467→0.048. |
| Barbero-2025-Why-LLMs-Attend-First-Token.pdf | arXiv:2504.02732 | COLM 2025 — **peer-reviewed** (stamped) | Why sinks exist (over-mixing / representational collapse), and a **context-length** knob: 120M models, 5B tokens, ≈24 h on one H100 — the cheapest published sink-formation sweep. |
| `Liao-2024-Free-Lunch-Removing-Outliers-Pretraining.pdf` | arXiv:2402.12102v1 | preprint | Audits the Bondarenko-style "train outliers away" claim — what it costs elsewhere. The counterweight to assuming outlier suppression is free. |
| SingleLayer-2026-Understanding-Massive-Activations.pdf | arXiv:2605.08504v2 | preprint | Recent mechanistic account localising massive activations to a single layer; closest published neighbour to the super-weight claim. |
| `Olsson-2022-InContext-Learning-and-Induction-Heads.pdf` | Transformer Circuits Thread | **not peer-reviewed** (Anthropic in-house) | The canonical phase-change measurement: 200 snapshots **every 50 steps**, loss-derivative-in-context statistic, and an explicit admission that 15 log-spaced snapshots make co-occurrence weak evidence. |
| `Nanda-2023-Progress-Measures-for-Grokking.pdf` | arXiv:2301.05217 | ICLR 2023 — **peer-reviewed** (stamped) | The counter-hypothesis template: an abrupt observable sitting on a mechanism that formed gradually and earlier. Restricted/excluded loss is the design pattern to copy. |
| `Power-2022-Grokking.pdf` | arXiv:2201.02177v1 | preprint copy, no venue stamp; ICLR 2022 MATH-AI workshop per publisher listing, **not verified from the PDF** | The original abrupt-transition measurement, and its cautions: budget-dependence, 3–7 seeds, narrow LR window, bimodality across seeds. |
| `Zhao-2025-Random-Scaling-Emergent-Capabilities.pdf` | arXiv:2502.17356v5 | **preprint** (stamped) | Why small-seed studies mislead: with 80–250 seeds, per-seed curves are abrupt while P(success) moves smoothly. Supplies the statistical toolkit (dip test, Wasserstein, bootstrap CIs). |
| `Schaeffer-2023-Emergent-Abilities-Mirage.pdf` | arXiv:2304.15004v2 | NeurIPS 2023 Outstanding Paper per publisher listing; **this copy is stamped "Preprint"** | The other way abruptness can be an artifact: the *metric*. Forces you to state the super-weight criticality metric's continuity before calling anything a phase change. |
| `PhaseTransitions-2025-Small-Transformer-LMs.pdf` | arXiv:2511.12768v1 | arXiv copy typeset in the IEEE Access template; **acceptance not confirmed in the PDF** | A worked example of detecting a training-time phase transition in a small transformer from several independent metrics synchronising in one window. Method reference, not a result to lean on. |

### Read-only companions used in notes_training_dynamics_checkpoint_suites.md

| File (elsewhere in `papers/`) | Why it matters here |
|---|---|
| Gu-2025-When-Attention-Sink-Emerges.pdf | ICLR 2025. **The formation-cause study**: 60M LLaMA models, 5B tokens, sweeps over LR, weight decay, data amount, PE, norm placement, attention biases, softmax variants. |
| Macocco-2025-Outlier-Dims-Across-Checkpoints.pdf | Last-layer outlier *dimensions* across 15 Pythia-12B checkpoints; documents post-formation identity turnover. |
| `Xu-2026-When-Do-Attention-Circuits-Form.pdf` | Sink/circuit formation timing across Pythia-1B, OLMo-1B-0724, OLMoE; the existence proof that shape (abrupt vs gradual) is configuration-dependent. |
| GallegoFeliciano-2025-Hidden-Dynamics-Massive-Activations.pdf | Massive-activation trajectories across all 9 Pythia sizes; 5-parameter fit, R²=0.984. Activation-level, one seed, no ablation. |
| Ding-2026-Weibull-Transformer-Weight-Distributions.pdf | The only **weight-level** super-weight-adjacent trace over checkpoints (Pythia-70m, 4 steps). |
| QueipoDeLlano-2025-Sinks-Compression-Valleys-Same-Coin.pdf | ICLR 2026. The "everything emerges together around step 1k" claim, n=2 Pythia models; includes a causal MLP ablation. |
| `NVFP4-2026-Outlier-Dynamics-Pretraining.pdf` | Outlier channel statistics tracked across the authors' own from-scratch runs; drift→fixed transition ~10k steps. |
| `TrainingDynamics-2025-PTQ-Robustness.pdf` | ICLR 2026. Quantization robustness across six suites' checkpoints — the finding is about **LR schedule**, not outliers. |
| Kaul-2024-From-Attention-to-Activation.pdf | arXiv:2410.17174v1, preprint. **Downloaded, then removed as an exact byte-identical duplicate of this existing copy** (md5 `45fc57dd…`). Attributes outliers partly to **Adam**; proposes OrthoAdam + softmax-1. Read in full for §4. |
| Oh-2024-House-of-Cards-Massive-Weights.pdf | The "large FFN weights drive massive activations" citation that SingleLayer-2026 leans on — **already in the repo, and still unread**. Highest-priority next read. |
| Dettmers-2022-LLM-int8.pdf | The original emergence-with-scale observation for outlier features. |
| Yu-2024-The-Super-Weight-in-LLMs.pdf | The paper being extended. Table 2 gives the super-weight coordinates this project traces backward. |

## Interpretability methods (copied from interlingua/papers)

|---|---|---|
| Bills-2023-LMs-Can-Explain-Neurons-in-LMs.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Bills 2023: LMs Can Explain Neurons in LMs |
| Bricken-2023-Towards-Monosemanticity.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Bricken 2023: Towards Monosemanticity |
| Clark-2019-What-Does-BERT-Look-At.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Clark 2019: What Does BERT Look At |
| Conmy-2023-Automated-Circuit-Discovery-ACDC.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Conmy 2023: Automated Circuit Discovery ACDC |
| DeYoung-2020-ERASER-Benchmark.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | DeYoung 2020: ERASER Benchmark |
| DoshiVelez-Kim-2017-Rigorous-Science-of-Interpretable-ML.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | DoshiVelez & Kim 2017: Rigorous Science of Interpretable ML |
| Elhage-2021-Mathematical-Framework-for-Transformer-Circuits.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Elhage 2021: Mathematical Framework for Transformer Circuits |
| Elhage-2022-Toy-Models-of-Superposition.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Elhage 2022: Toy Models of Superposition |
| Ferrando-2022-Opening-Black-Box-NMT-ALTI.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Ferrando-2025-Primer-Inner-Workings-of-Transformer-LMs.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Ferrando 2025: Primer Inner Workings of Transformer LMs |
| Geva-2023-Dissecting-Recall-of-Factual-Associations.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Geva 2023: Dissecting Recall of Factual Associations |
| Guerreiro-2023-Hallucinations-Large-Multilingual-Translation.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Dale-2023-HalOmi-Benchmark.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Lipton-2018-Mythos-of-Model-Interpretability.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Lipton 2018: Mythos of Model Interpretability |
| MI-Tutorial-Pranav-SLIDES.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Pranav —: Mechanistic Interpretability: On the Internals of LLMs *(tutorial slides)* |
| Meng-2022-Locating-and-Editing-Factual-Associations-ROME.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Meng 2022: Locating and Editing Factual Associations ROME |
| Meng-2022-ROME-SLIDES.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Meng 2022: ROME *(slides)* |
| Olah-2017-Feature-Visualization.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Olah 2017: Feature Visualization |
| Olah-2020-Zoom-In-Introduction-to-Circuits.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Olah 2020: Zoom In Introduction to Circuits |
| Pimentel-2020-InformationTheoretic-Probing.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Pimentel 2020: InformationTheoretic Probing |
| Rauker-2023-Toward-Transparent-AI-Survey.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Rauker 2023: Toward Transparent AI Survey |
| Li-2023-Repetition-In-Repetition-Out.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Rogers-2020-Primer-in-BERTology.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Rogers 2020: Primer in BERTology |
| Templeton-2024-Scaling-Monosemanticity.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Templeton 2024: Scaling Monosemanticity |
| Tenney-2019-BERT-Rediscovers-Classical-NLP-Pipeline.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Tenney 2019: BERT Rediscovers Classical NLP Pipeline |
| Tenney-2019-What-Do-You-Learn-From-Context.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Tenney 2019: What Do You Learn From Context |
| Turner-2023-Activation-Addition.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Turner 2023: Activation Addition |
| Vig-2019-BertViz-Tool.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Vig 2019: BertViz Tool |
| Vig-2019-Multiscale-Visualization-of-Attention.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Vig 2019: Multiscale Visualization of Attention |
| Wang-2023-Interpretability-in-the-Wild-IOI-Circuit.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Wang 2023: Interpretability in the Wild IOI Circuit |
| Zou-2023-Representation-Engineering.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | Zou 2023: Representation Engineering |

## Misc: hallucination and RAG (copied from interlingua/papers)

|---|---|---|
| FACTUM-2026-Citation-Hallucination-in-LongForm-RAG.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | FACTUM 2026: Citation Hallucination in LongForm RAG |
| ReDeEP-2024-Detecting-Hallucination-in-RAG.pdf | copy of `../../interlingua/papers/`; summary in `paper_summaries.md` there | ReDeEP 2024: Detecting Hallucination in RAG |

## Course documents (CS 601R, copied from interlingua/papers/course-docs)

|---|---|---|
| Bridge-Concepts-Attention-to-Causal-Editing.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Bridge-Concepts-Vision-Circuits-to-Transformer-Circuits.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| CS601R-Syllabus.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Cross-lingual-Context-SLIDES.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Final-Project-Guide.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Paper-Presentation-Rubric.pdf | copy of `../../interlingua/papers/course-docs/` | course material |
| Representation-Engineering-SLIDES.pdf | copy of `../../interlingua/papers/course-docs/` | course material |

## Speech translation and speech models (added 2026-09-06)

### Other open problems in speech translation

Open problems the ST field names for itself in 2025–2026, excluding the massive-activation /
attention-sink line. Notes: `notes_speech_other_problems.md`.

| File | Source (venue) | one line why |
|---|---|---|
| IWSLT-2026-Findings-Speech-Translation-and-Metrics.pdf | Adelani et al., IWSLT 2026 (ACL) | ten tracks; the organisers' own list of what is still broken — African S2S, QE, long-form, voice cloning |
| IWSLT-2025-Findings-Evaluation-Campaign.pdf | Agostinelli et al., IWSLT 2025 (ACL) | seven tracks; "performance ceiling" verdict on low-resource pairs and the synthetic-TTS negative |
| Papi-2025-How-Real-Is-Your-RealTime-SimulST.pdf | Papi et al., TACL 2025 | survey of 110 SimulST papers: 81.8% use pre-segmented audio, 97.7% gold segmentation |
| Xue-2026-Practical-Evaluation-LongForm-Simultaneous-S2ST.pdf | Xue et al., IWSLT 2026 (ACL) | first practical long-form SimulS2ST eval protocol; shows latency accumulates over long speech |
| Zhang-2026-Redefining-Machine-Simultaneous-Interpretation.pdf | Zhang et al., IWSLT 2026 (ACL) | argues SimulMT should learn interpreter strategies (salami, omission), not incremental MT |
| Zhang-2024-StreamSpeech.pdf | Zhang et al., ACL 2024 | the reference direct Simul-S2ST model with a jointly learned policy (CVSS) |
| ZarzuZouhar-2026-Hurdles-Automatic-Metric-ST-Evaluation.pdf | Zarzu & Zouhar, IWSLT 2026 (ACL) | adding source audio to a QE metric does **not** beat text-only; ablation across both paradigms |
| Zufle-2026-Why-We-Need-Speech-To-Evaluate-ST.pdf | Züfle et al., 2026, arXiv:2605.28227 (**preprint**) | speech-aware QE still at chance on MuST-SHE gender and ContraProST prosody |
| Han-2024-SpeechQE-Estimating-Quality-Direct-ST.pdf | Han & Duh, EMNLP 2024 | defines SpeechQE; end-to-end beats cascaded text-QE for direct ST |
| Krahn-2026-HydraQE-Speech-QE.pdf | Krahn & Fosler-Lussier, IWSLT 2026 (ACL) | best segment-level metric in the 2026 QE track (34.5 τb vs human 45.8) |
| Dale-2024-BLASER-2.0.pdf | Dale & Costa-jussà, Findings of EMNLP 2024 | the reference-free speech metric everyone uses as a baseline (and it loses to text COMETKiwi) |
| PostHoang-2025-Effects-Automatic-Alignment-ST-Metrics.pdf | Post & Hoang, IWSLT 2025 (ACL) | mwerSegmenter realignment barely moves COMET system rankings; releases `mweralign` |
| Koneru-2025-Quality-Aware-Decoding.pdf | Koneru et al., IWSLT 2025 (ACL) | QE folded into decoding rather than reranking — the "use the metric" side of the QE problem |
| Li-2025-SSA-COMET-African-MT-Evaluation.pdf | Li et al., EMNLP 2025 | SSA-MTE (73k annotations, 14 African pairs) and the metric the IWSLT African track scores with |
| Papi-2026-MCIF-Multimodal-Crosslingual-Instruction-Following.pdf | Papi et al., ICLR 2026 | long- vs short-form multimodal crosslingual benchmark; the long-form degradation evidence |
| Ugan-2026-Multilingual-LongForm-Speech-Instruction-Following.pdf | Ugan et al., IWSLT 2026 (ACL) | long-form speech instruction following; Whisper's 30 s window and its successors' long-form gap |
| HB-2026-AURA-ST-African-LowResource-E2E.pdf | HB et al., IWSLT 2026 (ACL) | the *only* submission to the African S2S track; frozen encoders + LoRA'd Gemma |
| Zevallos-2026-CATENG-Cascaded-vs-EndToEnd.pdf | Zevallos et al., IWSLT 2026 (ACL) | cascade beats E2E; quality bounded by ASR, not MT (Catalan–English, 44.7 BLEU) |
| Ortega-2026-QUESPA-Quechua-LLM-Prompting.pdf | Ortega et al., IWSLT 2026 (ACL) | GPT-5/Gemini 3/Claude prompting reaches 10.8 BLEU vs a fine-tuned NLLB-200 at 19.5 |
| Reassessing-2025-LowResource-ST-Do-LLMs-Redefine-SOTA.pdf | Dauvet et al., MRL 2025 (EMNLP wksp) | 10 FLEURS low-resource pairs: audio LLMs lose to cascades; 2-stage ASR-corrector gives 5.8× BLEU |
| Mohammadamini-2026-LIUM-LowResource-Pseudolabel-vs-Synthetic.pdf | Mohammadamini & Tahon, IWSLT 2026 (ACL) | pseudo-labelling works (21.1 BLEU test), TTS-synthesised source speech does not |
| Mohammadamini-2025-Scaling-PseudoLabeling-LowResource-ST.pdf | Mohammadamini et al., Interspeech 2025 | 3,200 h pseudo-labelled Kurdish → 20.68 BLEU on FLEURS; the scaling recipe |
| Li-2025-KIT-LowResource-Synthetic-Data.pdf | Li et al., IWSLT 2025 (ACL) | the counter-case: TTS-synthesised speech *helps* Bemba ASR/ST; synthetic-only beats cascade for apc |
| Liu-2024-Rare-Word-Accuracy-Retrieval-Demonstration-ST.pdf | Li, Liu & Niehues, EMNLP 2024 | +17.6% rare-word accuracy with gold demonstrations, +8.5% retrieved |
| Gaido-2021-Named-Entities-Terminology-in-ST.pdf | Gaido et al., EMNLP 2021 | ST gets 75–80% of terms, 65–70% of NEs, and only 37–40% of person names right |
| Yan-2025-CS-FLEURS-CodeSwitched-Speech-Dataset.pdf | Yan et al., Interspeech 2025 | 113 code-switched pairs over 52 languages; Whisper CER 2× (3× cross-script) vs monolingual FLEURS |
| Bafna-2025-LID-Models-Are-Accent-Classifiers.pdf | Bafna & Wiesner, Interspeech 2025 | LID models key on accent, not language — the upstream failure for accented ST |
| Bar-2025-Swiss-German-ST-Curse-Multidialectality.pdf | Bär et al., IWSLT 2025 (ACL) | joint dialect training costs 2.29 BLEU: a "curse of multidialectality" |
| Blaschke-2025-German-Dialect-ASR-and-Dialect-to-Standard-ST.pdf | Blaschke et al., Interspeech 2025 | Betthupferl: dialect→standard ST where the target is a *normalisation* decision |
| Attanasio-2024-Gender-Performance-Gaps-Multilingual-ASR.pdf | Attanasio et al., EMNLP 2024 | gender WER gaps in 19 languages, unexplained by acoustics/lexicon but predicted by probes |
| Savoldi-2025-ST-Models-Encode-Speaker-Gender-Differently.pdf | Fucci et al., ACL 2025 (short) | enc-dec models encode speaker gender (F1 96.2), adapter-based ones do not (≤61.8) → masculine default |
| Gaido-2026-Voice-Bias-Coreference-Gender-in-ST.pdf | Conti et al., LREC 2026 | occluding 1–20% of salient features flips gender in 37–47% of cases; masculine prevalence, not stereotypes |
| Savoldi-2025-mGeNTE-Gender-Neutral-Translation.pdf | Savoldi et al., EMNLP 2025 | models recognise when neutrality is needed but cannot produce it (en→es/de/it/el) |
| Savoldi-2026-Does-ST-Meet-Users-Needs.pdf | Attanasio et al., EAMT 2026 | Ouvia: user-centred En→Pt ST study across demographics — the "is the metric the point?" question |
| Ahtasam-2026-Balancing-Intelligibility-Speaker-Identity-Voice-Cloning.pdf | Ahtasam et al., IWSLT 2026 (ACL) | cross-lingual voice cloning trades content accuracy against speaker identity; Arabic worst |
| Subramanian-2025-Length-Aware-ST-Video-Dubbing.pdf | Subramanian et al., Interspeech 2025 | length-sensitive ST + length-aware beam search for dubbing isochrony |
| Adelani-2025-AFRIDOC-MT-Document-Level-African.pdf | Alabi et al., EMNLP 2025 | document-level African MT; sentence-trained models fail to generalise to documents |

### Problems

Active, tractable problems in speech models where this project's phenomenon could give a foothold:
hallucination and repetition loops, outlier/sink/massive-activation structure in speech models, and
low-resource speech translation failures. Notes: `notes_speech_problems.md` (§5 lists verification
debt; venue tags below follow that file — `VERIFIED` = proceedings record seen, `author-declared` =
arXiv comment field only, `preprint` = arXiv-only, no venue stated).

| File | Source (venue) | one line why |
|---|---|---|
| Koenecke-2024-Careless-Whisper.pdf | Koenecke et al., ACM FAccT 2024 (`VERIFIED`) | ~1% of transcripts fully hallucinated, 38% of those carry explicit harm — the paper that makes this a safety problem |
| Lost-in-Transcription-2025-Hallucination-Speech-Foundation-Models.pdf | Atwany et al., ACL 2025 (author-declared) | introduces HER over 20+ ASR models; WER and hallucination decorrelate; distribution shift ↔ HER α=0.91 |
| HALAS-2026-Human-Annotated-ASR-Hallucinations.pdf | Barański et al., Interspeech 2026 (author-declared) | first human-annotated *natural* hallucinations; SOTA detection only 53.1% F1; prior mitigations all evaluated on synthetic audio |
| Frieske-2024-Hallucinations-Neural-ASR.pdf | preprint (venue unverified) | defines ASR hallucination; perturbation test separates models with equal WER but different propensity |
| Radford-2022-Whisper-Robust-Speech-Recognition.pdf | preprint on arXiv (ICML 2023 unverified) | origin of the deployed symptom heuristics: compression-ratio 2.4, temperature fallback, logprob threshold |
| Wu-2025-Beyond-Transcription-Mech-Interp-ASR.pdf | preprint (Glazer et al., aiOla) | **key paper**: repetition localised to cross-attention L23 (76%) + L18H13 (78.1%); self-attention and FFN "no measurable effect" |
| Wang-2025-Calm-Whisper-Crazy-Heads.pdf | Wang et al., Interspeech 2025 (`VERIFIED`, ISCA archive) | 3 of 20 decoder self-attention heads cause >75% of non-speech hallucination; targeted FT gives >80% reduction at <0.1% WER |
| Whisper-2026-Hallucination-SAE-Steering.pdf | preprint | SAE steering of Whisper *encoder* activations: 72.6%→14.1% (small), 86.9%→27.3% (large-v3); sparse feature subset, deeper = more separable |
| TextMetrics-2026-Whisper-Hallucination-Model-Internals.pdf | Jasiński et al., Interspeech 2026 (author-declared) | reference-free probing of Whisper decoder states beats text metrics and LLM judges on real speech |
| NullToken-2026-Message-Free-Hallucination-ASR-NMT.pdf | preprint (submitted AAAI-27) | ASR *and* NMT in one frame; decoder states separate non-speech after block 1; suppression/deletion trade-off; failure not monotone in scale |
| Dispersion-Attraction-2026-Spectral-Dynamics-Whisper-Hallucination.pdf | preprint | rival hypothesis: hallucination as a rank-collapse attractor; 13.4% cross-attn rank collapse (mid), −2.34% self-attn (large) |
| SpeechLLM-2026-Detecting-Hallucinations-Attention-Maps.pdf | Waldendorf et al., Findings of ACL 2026 (author-declared) | four audio-attention metrics detect SpeechLLM hallucination, +0.23 PR-AUC; ~100 heads suffice |
| Listen-Like-a-Teacher-2025-Whisper-Hallucination-Mitigation.pdf | preprint (Sony Research India) | encoder-side Adaptive Layer Attention + KD; frames hallucination as misaligned internal representations in encoder *and* decoder |
| Whisper-CD-2026-Contrastive-Decoding-LongForm.pdf | preprint | training-free contrastive decoding vs noise/silence/shift negatives; −24.3 pp WER on CORAAL; names the three long-form failure patterns |
| Xu-2026-LoopGuard-Self-Reinforcing-Attention-Loops.pdf | preprint | text-LLM repetition baseline: heads lock onto a narrow suffix, KV-cache importance amplifies the loop; >90 pp reduction |
| AttentionSinks-2026-Internal-Signals-Hallucination-Detection.pdf | preprint | text-side precedent that sink statistics carry hallucination information |
| Outlier-Reduction-2024-Gated-Attention-PTQ-Speech-Foundation.pdf | Wagner et al., Interspeech 2024 (author-declared) | **key paper**: sink on `<\|tr\|>` in Whisper decoder L31 with near-zero V; dims #819/#1054 = ~15% of all outliers; gating enables INT8 W8A8 |
| Cappellazzo-2025-Attention-Sinks-Massive-Activations-AVSR.pdf | Anand et al., IEEE ICASSP 2026 (author-declared) | first sinks+massive-activations study in speech recognition — but in the *Llama decoder*, MLP-origin layer 2, fixed feature indices; gains concentrated under high compression |
| OmniLLM-2026-Nature-of-Attention-Sink-Decoding.pdf | preprint | independent restatement of "the sink value vector acts as a shared bias added to every token's output" |
| CrossModal-2026-Information-Hubs-AudioVisual-LLMs.pdf | preprint | sink tokens are cross-modal information *hubs*, not junk; training-free hallucination mitigation from that |
| WnW-2026-Waxing-Waning-KV-Cache-Speech-LLMs.pdf | preprint | positional audio-start attention sink in prefill; prefill and decode attention rankings overlap weakly |
| Anisotropy-2025-Neural-Representations-of-Speech.pdf | preprint | only speech-side anisotropy measurement found (wav2vec2/XLSR-53); keyword-spotting case study, weaker than hoped |
| LayerWise-2026-Probing-wav2vec2-Whisper-AAE.pdf | preprint | layer-wise probing template across a supervised/self-supervised speech pair |
| Quantizing-Whisper-2025-Design-Choices.pdf | preprint | dynamic INT8 = 57% smaller at WER *below* baseline; static much worse — never connected to Wagner's outliers, though that is the obvious cause |
| LLMDecoders-2026-Listen-Fairly-Bias-ASR.pdf | preprint | **most surprising**: Whisper = 86% of 51,797 insertions under masking vs 38× fewer for LLM decoders, but high-compression Q-former reintroduces it; "audio encoder design, not LLM scaling" |
| Pashto-2026-Zero-Shot-ASR-Script-Failure.pdf | preprint (TASLP format, acceptance unstated) | script failure: no Whisper size emits Pashto script in >0.8% of utterances vs >93% for MMS/Seamless/OmniASR; WER 461% = a loop with a number |
| Whisper-2026-Decoder-Inconsistencies-Dravidian-LowResource.pdf | preprint | "decoder imbalance between self-attention (linguistic) and cross-attention (acoustic)" reached independently from a low-resource angle |
| AfriSwitch-2026-African-CodeSwitched-ASR-Benchmark.pdf | preprint | 61.36 h / 16 African languages code-switched; best zero-shot 35.93% WER, none below 24%; Africa-targeted training beats scale |
| WAXAL-2026-African-Language-Speech-Corpus.pdf | preprint (Google Research Africa) | 24 languages, 1,250 h ASR + 235 h TTS, CC-BY-4.0 — free evaluation data in the lab's exact languages |
| WAXAL-NET-2026-Edge-ASR-19-African-Languages.pdf | preprint | fine-tuned edge models 38.0% vs 64.9% macro-WER at 3–40× smaller: specialisation dominates scale |
| Makerere-2025-Benchmarking-ASR-African-Languages.pdf | Nahabwe et al., PMLR 302 / Deep Learning Indaba 2025 (`VERIFIED` from PDF header) | accredited-venue anchor for African ASR benchmarking |
| NaijaS2ST-2026-MultiAccent-Nigerian-S2ST-Benchmark.pdf | preprint (Mila/Masakhane/HausaNLP) | Igbo/Hausa/Yorùbá/Pidgin ↔ En across cascaded, E2E and AudioLLM; 0.85% of FLORES-101 is African |
| Bemba-2025-Low-Resource-African-Speech-Translation.pdf | Al Farouq et al., IWSLT 2025 (`VERIFIED` journal-ref) | Whisper→NLLB cascade at an accredited ST venue — the lab's own architecture |
| Omnilingual-ASR-2025-1600-Languages.pdf | preprint (Meta/FAIR) | 1,600+ languages; "LLM-inspired decoder"; 300M variants small enough to study on one A100 |
| Survey-2025-ASR-African-LowResource-Challenges-Future.pdf | preprint (HausaNLP/EthioNLP et al.) | the field's own list of African ASR barriers; framing, not numbers |
| Phonology-Guided-2024-S2ST-African-Languages.pdf | preprint | >80% of African languages have <5 h transcribed speech; transcript-free prosody-guided S2ST |
| BENYO-2025-English-Yoruba-Direct-S2ST-Corpus.pdf | preprint | 41.2 h English↔Yorùbá direct S2ST corpus + a Yorùbá TTS proof of concept |
| MURMUR-2026-Efficient-LongForm-ASR.pdf | preprint | long-form ASR inference: chunking loses cross-segment context, long-context models cost too much |
| Seamless-2023-Expressive-Streaming-Speech-Translation.pdf | preprint (Meta) | `[unopened]` — expressive/streaming S2ST reference point |
| CodeSwitch-2026-Benchmarking-Commercial-ASR.pdf | preprint | `[unopened]` — commercial ASR on Arabic/Persian/German code-switching |
| Swahili-2026-Continued-Pretraining-LowResource-ASR.pdf | preprint | `[unopened]` — continued pretraining for Swahili ASR, matches the lab's TTS language |

### ASR and TTS for low-resource languages

The two ends of the lab's cascade, 2022–2026, excluding the massive-activation / attention-sink line.
Notes: `notes_speech_asr_tts_lowresource.md`. Venue column says **where** the venue is stated:
`printed on PDF` = on the paper itself; `arXiv comment` = submitter's comment field, not verified
against proceedings; `preprint` = no venue claim found. Rows marked `[listing]` were never opened.
Twelve papers I downloaded were byte-identical to copies the parallel speech agents had already
placed here; I deleted mine and cite theirs (indexed in their subsections above), so this table lists
only files it introduces.

| File | Source (venue) | one line why |
|---|---|---|
| Omnilingual-ASR-2025-1600-Languages.pdf | FAIR at Meta, arXiv 2511.09690 (preprint) | *pre-existing here* — the §2 subject: 1,600+ languages, in-context language extension, and the Atlantic-Congo / Afro-Asiatic gap in its own Table 10 |
| SLR-2025-ASR-African-LowResource-Systematic-Review.pdf | preprint (arXiv 2510.01145) | PRISMA review: 74 datasets, 111 languages, ~11,206 h, 63.5% high risk of bias; says outright that CER/DER are underused for want of diacritic-level annotation |
| HowMuchData-2025-ASR-African-Languages-Hours.pdf | Sunbird AI (preprint) | the data-budget number: WER<13% at 50 h, <10% at 200 h; and 38.6% of high-error cases are bad ground-truth transcripts |
| AfriSpeech-MultiBench-2025-African-Accented-English-ASR.pdf | Intron Health; IJCNLP-AACL 2025 per arXiv comment | 100+ African English accents × 7 domains, including a hallucination-robustness vertical |
| AfriVox-v2-2026-Verticalized-African-ASR-Benchmark.pdf | preprint | in-the-wild African ASR across ten sectors; benchmarks the Omnilingual CTC models |
| Relatedness-2026-CrossLingual-Transfer-Multilingual-ASR.pdf | preprint (arXiv 2607.04814) | the negative: related-language pre-adaptation gives no meaningful gain once ≥1 h of target data exists |
| DonorRank-2026-Donor-Language-Selection-CrossLingual-ASR.pdf | preprint | learning-to-rank donor selection on WAXAL and VAANI-D; sits in tension with the row above |
| Pseudo2Real-2026-ACL-Findings-Task-Arithmetic-PseudoLabel.pdf | ACL 2026 Findings per arXiv comment | weight-difference correction vector for pseudo-label bias; 35% relative WER cut on AfriSpeech-200 |
| SyntheticVoice-2025-RANLP-African-ASR-Data.pdf | RANLP 2025 workshop per arXiv comment | synthetic voice corpora at <1% of the US$100–150/h cost of real African speech collection |
| SBPN-2026-Nigerian-Languages-Knowledge-Distillation-ASR.pdf | preprint | Yorùbá/Hausa/Igbo/Pidgin foundation ASR by distillation + self-training, −29% relative WER |
| Gaelic-2025-Interspeech-Practitioners-Guide-LowResource-ASR.pdf | Interspeech 2025 per arXiv comment | methods template for building a low-resource ASR model end to end |
| SoftPrompt-2025-Interspeech-Whisper-CodeSwitching.pdf | Interspeech 2025 per arXiv comment | parameter-efficient code-switching adaptation for Whisper |
| ToneCurriculum-2026-Bantu-ASR.pdf | Pretoria / TU Dublin / Lelapa (preprint) | tone-gated adapters for six Southern Bantu languages; foundation models start above 100% zero-shot WER |
| Tone-2025-Interspeech-Recognition-NorthEast-India.pdf | Interspeech 2025 per arXiv comment | the layer-wise tone probe (wav2vec2 layers ~4–6) that project A1 proposes to replicate on Niger-Congo |
| SITA-2026-Tone-Aware-LowResource-Speech-Representations.pdf | preprint `[listing]` | speaker-invariant tone-aware adaptation for wav2vec-style encoders |
| ScriptCollapse-2026-Reference-Free-Metric-Multilingual-ASR.pdf | preprint | Script Fidelity Rate: 21% of 100 model–language pairs output fluent text in the wrong script while WER stays finite |
| Igbo-2026-Corpus-Based-Diacritic-Restoration.pdf | Ezeani, PhD thesis, U. Sheffield (arXiv 2026) | the reference work on Igbo diacritic restoration |
| Diacritics-2025-LowResource-Indigenous-Bribri-Maori.pdf | preprint | diacritic (incl. tonal) restoration needs ~10,000 words; character-level LLMs beat massively multilingual ones |
| VoicesUnheard-2024-Yoruba-Regional-Dialects.pdf | preprint `[listing]` | YORÙLECT: four regional Yorùbá dialects, text and speech |
| IroyinSpeech-2024-LREC-COLING-Yoruba-Corpus.pdf | LREC-COLING 2024 per arXiv comment `[listing]` | multi-purpose Yorùbá speech corpus; the tone-marked transcripts project A1 needs |
| Xhosa-2025-Child-Reading-Assessment-ASR.pdf | preprint | Xhosa child-speech EGRA dataset; wav2vec2 / HuBERT / Whisper under data-scarce class imbalance |
| Bambara-2026-Children-Reading-ASR.pdf | preprint | 55 h of Bambara child reading; WER 0.42→0.22, and children under 10 are the residual error source |
| Yoruba-2026-Situational-Speech-Synthesizer-Tone.pdf | "under review, Speech Communication" per arXiv comment | rule-based Yorùbá diphone TTS: 651 units over five tonal variants per CV, plus a contour-tone orthography proposal |
| XTTS-2024-Interspeech-Massively-Multilingual-ZeroShot-TTS.pdf | Interspeech 2024 per arXiv comment | the lab's likely TTS baseline; 16 languages, no African language among them |
| CosyVoice-2024-Supervised-Semantic-Tokens-TTS.pdf | preprint ("work in progress") | supervised semantic tokens for multilingual zero-shot TTS |
| TransferTTS-2023-SSW-LowResource-Speech-Synthesis.pdf | "Submitted to SSW" printed on PDF | phone mapping vs phonological-features input for cross-lingual TTS transfer, with Swahili as a target |
| Edge-2025-Kinyarwanda-Swahili-STT-TTS.pdf | CMU-Africa (preprint) | edge/cloud split for Whisper + SpeechT5 on Kinyarwanda and Swahili; deployment shape, not modelling |
| MOS-Bench-2024-TASLP-Generalization-Speech-Quality-Assessment.pdf | IEEE TASLP per arXiv comment | why UTMOS-class predictors should not be trusted on an unseen African language: 8 train / 17 test sets, poor OOD generalisation |
| Efik-2026-AfricaNLP-English-Efik-Corpus-MT.pdf | AfricaNLP 2026 @ EACL per arXiv comment `[listing]` | text-side Efik resource for the language the ToAll app ships and Omnilingual does not cover |
| IbomNLP-2025-IJCNLP-AACL-Nigeria-Minority-Languages.pdf | IJCNLP-AACL per arXiv comment `[listing]` | Efik/Ibibio/Annang-area NLP resources |
| Kencorpus-2022-Kenyan-Language-Corpus.pdf | *J. Language Technology & Computational Linguistics* 36(2) 2023 per arXiv comment | Swahili, Dholuo, Luhya |
| AfriVoices-KE-2026-LREC-Kenyan-Speech-Dataset.pdf | RTUEL @ LREC 2026 per arXiv comment | ~3,000 h over five Kenyan languages, and a candid account of how field collection fails |
| Africa-Centric-2024-SSL-Pretraining-SubSaharan.pdf | **AfricaNLP 2024 @ ICLR, printed on PDF** | Sub-Saharan self-supervised pretraining (the SSA-HuBERT line) |
| Luganda-2024-AfricaNLP-Crowdsourced-TTS.pdf | **AfricaNLP 2024 @ ICLR, printed on PDF** | the complete recipe for turning noisy crowdsourced audio into a TTS voice: MOS 2.5 → 3.55 |
| AfriNames-2023-Interspeech-ASR-Butcher-African-Names.pdf | "Submitted to INTERSPEECH" printed on PDF | ASR systematically mis-transcribes African named entities |

### Landscape

Downloaded 2026-09-06 for `notes_speech_landscape.md` (model/task landscape, Omnilingual ASR, IWSLT open
problems, African speech data). Venues were verified via the Semantic Scholar `publicationVenue` field or
printed on the PDF itself; **`preprint` means arXiv-only, and every model technical report is one**.

| File | Source (venue) | one line why |
|---|---|---|
| Omnilingual-ASR-2025-1600-Languages.pdf | preprint (arXiv:2511.09690, FAIR at Meta, 53 pp.) | the 2025 ASR release the note is built around: 1,600+ languages, Apache-2.0, in-context zero-shot extension to unseen languages |
| Radford-2022-Whisper-Robust-Speech-Recognition.pdf | ICML | the ASR baseline everything else is measured against; its card names hallucination and repetition as known failures |
| Pratap-2024-MMS-Scaling-Speech-1000-Languages.pdf | JMLR | the prior massively-multilingual system Omnilingual ASR displaces; CC-BY-NC, per-language adapters |
| Makerere-2025-Benchmarking-ASR-African-Languages.pdf | Deep Learning Indaba 2025 (PMLR 302) | settles "MMS vs w2v-BERT vs XLS-R vs Whisper" for African languages by data regime — the most actionable ASR result here |
| DONDO-2026-w2v-BERT-African-ASR-Models.pdf | preprint (arXiv:2607.21540) | 26 Apache-2.0 w2v-BERT 2.0 fine-tunes over 27 African varieties, commercial use permitted |
| Alabi-2025-AfriHuBERT.pdf | Interspeech | small African-language SSL encoder; the natural baseline against `omniASR-W2V-300M` |
| SeamlessM4T-2023-Massively-Multilingual-Multimodal-MT.pdf | preprint (arXiv:2308.11596, 111 pp.) | the reference end-to-end S2ST system; CC-BY-NC, and the only open model with real African speech output |
| Zhang-2024-StreamSpeech-Simultaneous-S2ST.pdf | ACL | multi-task simultaneous S2ST baseline |
| Labiausse-2025-Hibiki-Simultaneous-S2ST.pdf | ICML 2025 | decoder-only multistream LM for simultaneous S2ST — but fr→en only |
| Labiausse-2026-Hibiki-Zero-S2ST-Without-Aligned-Data.pdf | ICML 2026 (printed on the PDF) | drops the word-alignment requirement and reports adding a new source language with **<1,000 h** — the most relevant open S2ST result for low-resource work |
| Qwen3-Omni-2025-Technical-Report.pdf | preprint (arXiv:2509.17765) | still the strongest open omni model a year on; 19 speech-input languages, none African |
| Voxtral-2025-Mistral-Audio-Models.pdf | preprint (arXiv:2507.13264) | Apache-2.0 speech LLM, 8 languages, none African |
| Phi-4-Multimodal-2025-Technical-Report.pdf | preprint (arXiv:2503.01743) | Mixture-of-LoRAs modality routing; 23 text languages but only 8 speech, none African |
| Granary-2025-Speech-Dataset-25-Languages.pdf | Interspeech 2025 | the ~1M-hour corpus behind Canary-1B-v2 and Parakeet v3 — and the reason both cover 25 European languages and zero African ones |
| IWSLT-2025-Findings-Evaluation-Campaign.pdf | IWSLT 2025 (ACL Anthology `2025.iwslt-1.44`; **no arXiv id**) | the organisers' own statement of a low-resource performance ceiling and the >50 h rule of thumb |
| IWSLT-2026-Findings-Speech-Translation-and-Metrics.pdf | IWSLT 2026 (ACL Anthology `2026.iwslt-1.39`; **no arXiv id**) | the new African/Celtic track (Hausa/Igbo/Yorùbá→En) with OmniASR→NLLB-200 as a published baseline; plus the metrics, compression and cascaded-vs-end-to-end findings |
| Papi-2025-How-Real-Is-Your-Realtime-SimulST.pdf | TACL | why reported simultaneous-ST latency numbers are not what they appear |
| Conneau-2023-FLEURS-Benchmark.pdf | IEEE SLT 2022 | the n-way parallel benchmark that gives free S2TT test data for Xhosa, Igbo, Swahili and Yoruba |
| Chen-2023-BLASER-Text-Free-S2ST-Metric.pdf | ACL | text-free S2ST metric; IWSLT's speech-to-speech ranking metric |
| Han-2024-SpeechQE-Quality-Estimation-Speech-Translation.pdf | EMNLP 2024 | reference-free QE for speech translation; an IWSLT 2026 metrics baseline |
| Wang-2024-AfriMTE-AfriCOMET.pdf | NAACL 2024 | the first COMET adapted to African languages |
| SSA-COMET-2025-MT-Evaluation-African-Languages.pdf | EMNLP 2025 | its successor, and IWSLT 2026's official metric for the African track |
| Linguistically-Informed-2026-Multilingual-ASR-African-Tone.pdf | AfricaNLP 2026 workshop | WER mismeasures tonal languages — Yoruba WER 0.788 vs Feature Error Rate 0.151; four of the lab's six are tonal |
| Koenecke-2024-Careless-Whisper.pdf | FAccT 2024 (printed on the PDF) | quantifies Whisper hallucination: ~1% of transcripts fabricated, 38% of them harmful |
| Wang-2025-Calm-Whisper-Crazy-Heads.pdf | preprint (arXiv:2505.12969) | the only mechanistic account found: 3 of 20 decoder heads cause ~75% of non-speech hallucinations |
| Hearing-to-Translate-2025-Speech-Modality-Integration-LLMs.pdf | preprint (arXiv:2512.16378) | 6 SpeechLLMs vs 16 alternatives: "cascaded systems remain the most reliable solution overall" |
| NaijaVoices-2025-Igbo-Hausa-Yoruba-Speech-Dataset.pdf | Interspeech 2025 | ~600 h each of Igbo, Hausa and Yoruba — the largest Nigerian-language speech corpus, but CC-BY-NC-SA |
| Olatunji-2023-AfriSpeech-200.pdf | TACL | 200 h pan-African speech — but *accented English*, not indigenous-language speech; a trap worth documenting |
| AfriSpeech-Dialog-2025-Benchmark.pdf | NAACL 2025 | spontaneous conversational follow-up to AfriSpeech-200 |
| Swivuriso-2025-African-Next-Voices-South-Africa.pdf | preprint (arXiv:2512.02201) | 500 h of isiXhosa — CC BY 4.0 but with an explicit clause forbidding TTS and voice cloning |
| WAXAL-2026-African-Language-Speech-Corpus.pdf | preprint (arXiv:2602.02734) | ~1,250 h ASR + 180 h TTS across 19 African languages, CC BY / BY-SA — the best purpose-recorded African TTS data |
| Meyer-2022-BibleTTS.pdf | Interspeech 2022 | studio-quality 48 kHz African TTS corpus under CC BY-SA 4.0 — the cleanest-licensed option for Twi and Yoruba |
| OpenBibleTTS-2026-African-Language-TTS.pdf | preprint (arXiv:2606.09553) | 3,469 h / 37 languages / 19 African, CC-BY-SA-4.0, with the result that 18M monolingual models beat a 600-language model on intelligibility |
| Efik-TTS-2026-Digital-Preservation.pdf | preprint (arXiv:2607.04515; PDF claims Interspeech 2026, venue unverified) | the first end-to-end Efik TTS, from 3 hours of speech — and the only Efik speech system that exists |
| Ogun-2024-Afro-TTS-1000-African-Voices.pdf | Interspeech 2024 | routinely miscited as African-language TTS; it is accented *English*, and the note records why that matters |
| Voice-of-a-Continent-2025-African-Speech-Survey.pdf | preprint (arXiv:2505.18436) | the pre-2026 baseline for the coverage gap: African TTS at 11 languages / 334 h vs ASR at 42 / 6,109 |
| Chen-2024-F5-TTS.pdf | ACL | the flow-matching TTS architecture used as the large arm of OpenBibleTTS |
| CrossLingual-F5-TTS-2025-Unseen-Language-Cloning.pdf | preprint (arXiv:2509.14579) | attacks the real blocker in cross-lingual cloning — the requirement for a transcript of the reference audio |
| Minixhofer-2026-TTSDS2-Benchmark.pdf | preprint (arXiv:2506.19441) | shows UTMOS/DNSMOS/WER correlate poorly with human MOS, and covers no African language — so TTS scores on Swahili or Yoruba are uncalibrated |

### Interpretability

Internals of speech and multimodal speech–text models: whether the text-side phenomena
(massive activations, attention sinks, a shared cross-lingual/cross-modal "semantic hub")
exist on the speech side, and what tooling exists to look. Notes:
`notes_speech_interpretability.md`. Venues are as stamped on the PDF or in the arXiv
`comment`/`journal_ref` field; anything else is labelled preprint.

| File | Source (venue) | one line why |
|---|---|---|
| Lee-2025-Multimodal-Encode-Text-and-Speech.pdf | Lee et al., **NAACL 2025** (Short) | SVCCA over 30 FLEURS languages × 2 modalities on Seamless/SONAR/SALMONN: modality gap > language gap; speech spreads more cross-lingually than text; **explicitly does not analyse the audio encoders** |
| GenStepAware-2026-CrossModal-Representation-Control-SeamlessM4T.pdf | preprint (2601.17387) | per-decoding-step language-selective neurons in **SeamlessM4T v2's shared decoder** — the closest prior work to the interlingua question in a speech model; read this first |
| FactualRecall-2026-Text-to-Speech-Multimodal-LMs.pdf | **\*SEM 2026** | causal mediation analysis on SpiritLM: factual-recall mechanisms only **partially** carry over from text to speech |
| AnatomyModalityGap-2026-Internal-States-Speech-LLMs.pdf | preprint (2603.01502) | cross-layer CKA on four speech LLMs; input-layer statistical calibration is insufficient and can hurt — the modality gap is not a distribution shift |
| ModalityGap-2025-SpeechText-Alignment-Mechanism-LSLM.pdf | Xiang et al., **EMNLP 2025** (Main) | deep layers converge in **direction** while diverging in **magnitude** — the one published hint that a norm-level effect separates the modalities |
| Interleaved-2026-Speech-LMs-Latently-Work-In-Text.pdf | preprint (2606.22473) | logit lens: interleaved speech LMs pass through an **implicit latent transcription** in text space; needs both text-LM init and interleaved data |
| AudioLLM-2026-Verbalizable-Multilingual-MiddleLayer-Workspace.pdf | preprint (2608.24958, Amazon AGI) | logit lens on base Qwen3-Omni: a **language-agnostic, verbalisable middle-layer workspace** (35–80% depth), causally confirmed by activation patching |
| CascadeEquivalence-2026-Speech-LLMs-vs-Pipelines.pdf | preprint (submitted Interspeech 2026) | `[unopened]` — when do speech LLMs behave like ASR→LLM cascades |
| Omnilingual-SONAR-2026-CrossLingual-CrossModal-Embeddings.pdf | preprint (2603.16606) | `[unopened]` — massively multilingual cross-lingual + cross-modal sentence embeddings; candidate substrate |
| TranslationEnhanced-2026-Speech-Encoder-Pretraining-SpeechLLMs.pdf | **Interspeech 2026** | `[unopened]` — does a translation objective in encoder pretraining change the downstream speech LLM |
| AudioSAE-2026-EACL-Sparse-Autoencoders-Audio-Models.pdf | Aparin et al., **EACL 2026** (main, pp. 3221–3254) | SAEs on **all encoder layers of Whisper and HuBERT**, code + checkpoints released; steering cuts false speech detection 70% |
| Whisper-SAE-2026-Interpretability-Whisper-Encodings.pdf | preprint (2605.12225) | Whisper's encoder holds a phonetic→semantic **hierarchy** beyond transcription needs, with a causal steering campaign across it |
| AudioSAE-2025-Interpretable-Features-Audio-Latent-Spaces.pdf | **NeurIPS 2025 MI Workshop** | `[unopened]` — SAEs on neural audio **codec** latents, nearest thing to codec interpretability |
| BehindScenes-2025-MechInterp-LoRA-Whisper-SER.pdf | Ma et al., **ICASSP 2026** | first mech-interp study of LoRA inside the Whisper encoder: delayed specialisation; forward-alignment / backward-differentiation |
| Whisper-2025-Internal-Word-Aligner.pdf | **ASRU 2025** | `[unopened]` — Whisper's internal cross-attention word-alignment mechanism |
| insideSSL-2026-Model-Centric-SSL-Speech-Representations.pdf | **Interspeech 2026** (long) | `[unopened]` — cross-layer Generative Compatibility Matrix: stable phonetic cores, deep-layer semantic pruning |
| Orthogonality-2024-Speaker-Phonetic-SSL-Speech.pdf | **Interspeech** (year unconfirmed) | `[unopened]` — speaker and phonetic information occupy near-orthogonal subspaces in SSL speech models |
| ListeningWithAttention-2026-Entropy-Guided-Explainability-Audio.pdf | **Interspeech 2026** | `[unopened]` — entropy-guided attention explainability for audio transformers |
| HeardNotHeeded-2026-Paralinguistic-Encoding-Loss-ALMs.pdf | preprint (2609.00727) | `[unopened]` — where paralinguistic information is encoded and where it is lost in audio LMs |
| ProsodicUnderuse-2026-Causal-Account-Audio-LMs.pdf | preprint (2608.19211) | `[unopened]` — causal account of prosody being represented but not used; relevant for tone languages |
| Outlier-Reduction-2024-Gated-Attention-PTQ-Speech-Foundation.pdf | Wagner et al., **Interspeech 2024** | **the key §3 paper**: Whisper-large decoder layer 31 puts ~all attention mass on `<\|transcribe\|>` with a small value vector (a **no-op sink**, never named as one); outliers concentrate in fixed dims #819/#1054; last encoder layer must stay FP |
| Cappellazzo-2025-Attention-Sinks-Massive-Activations-AVSR.pdf | Anand, Cappellazzo et al., **IEEE ICASSP 2026** | sinks + massive activations (τ=10³, MLP of layer 2, identical feature indices) in the **LLM decoder** of an audio-visual speech model; causal via hidden-state rotation |
| Whisper-Encoders-2025-Languages-Align-Phonetically-Semantically.pdf | preprint (submitted Interspeech 2026) | pronunciation-**controlled** spoken-translation retrieval: earlier ~80% Whisper alignment numbers were partly cognate shortcuts; alignment survives only in final layers of translation-trained encoders |
| LanguageControl-2026-Latent-Mechanisms-Multilingual-LMs.pdf | **EMNLP 2026** (main) | `[unopened]` — text-side mechanism vocabulary for language control, to be ported to speech |
| EmotionNeurons-2026-Multilingual-Large-Audio-LMs.pdf | preprint (2608.08772) | `[unopened]` — the only neuron-level *multilingual* study of an audio LM found |
| Darcet-2024-ICLR-Vision-Transformers-Need-Registers.pdf | Darcet et al., **ICLR 2024** | `[unopened]` — the vision precedent: high-norm artifact tokens and the register fix. **No audio analogue exists.** |
| VisionEncoders-2026-ECCV-Activation-Quantization-Prefixing-Registers.pdf | **ECCV 2026** | `[unopened]` — prefixed registers absorb the outliers that break activation quantization of vision encoders |
| Sun-2024-COLM-Massive-Activations-in-LLMs.pdf | Sun et al., **COLM 2024** | `[unopened]` — the text-side original the whole §3 question is ported from |
| PTQ-2026-LayerWise-Compensation-EncoderDecoder-ASR.pdf | preprint (2601.02455) | `[unopened]` — layer-wise diagnosis of which encoder-decoder ASR layers resist quantization |
| EdgeASR-2025-LowBit-Quantization-ASR-Models.pdf | preprint (2507.07877) | `[unopened]` — low-bit ASR quantization; knows the encoder is hard, does not localise an outlier token |
| nnterp-2025-Standardized-Interface-MechInterp.pdf | **NeurIPS 2025 MI Workshop** | `[unopened]` — nnsight wrapper standardising transformer-LM naming; no speech support |
| NNsight-2024-NDIF-Model-Internals.pdf | Fiotto-Kaufman et al. (2407.14561) | `[unopened]` — traces any HF module by name; **the right tool for Whisper and Seamless**, since TransformerLens supports neither |
| Inseq-2023-ACL-Interpretability-Toolkit-Sequence-Generation.pdf | **ACL 2023 System Demos**, pp. 421–435 | `[unopened]` — encoder-decoder + decoder-only **text** attribution; no speech support |
