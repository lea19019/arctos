# Reading list — super-weight formation program

**Drafted 2026-08-06; verified the same day.** Status flags: **[verified]** =
confirmed during the 2026-08 audits (`registry.md`, `method_landscape.md`,
`prior_work_map.md`); **[agent-verified]** = confirmed by the 2026-08-06
blind-spot agents (abstract/full-text fetched); **[preprint]** = unrefereed —
under this repo's claim-hygiene rules it cannot carry a claim on its own.
Local PDFs live in `../papers/` (gitignored; indexed in `../papers/README.md`).

Read tiers in order; within a tier, order is priority.

## Tier 1 — the super-weight corpus (all of it)

| Paper | Venue | Why | Status |
|---|---|---|---|
| Yu et al., *The Super Weight in Large Language Models* (arXiv:2411.07191) | arXiv 2024 (Apple) | The discovery. Detection recipe, coordinates for public models, the SW→massive-activation chain | [verified] |
| Subramanian et al., *Super Weights in LLMs and the Failure of Selective Training* (arXiv:2607.08733) | **COLM 2026** | The skeptical follow-up: damage not universal; training only the SWs drops OLMo to chance | [agent-verified] |
| Ding, *Weibull Weight-Scale Parameter Evolution under AdamW* (arXiv:2606.19367) | none | The "third corpus paper" — marginal: mentions SWs only in a robustness check. Honest count: **two substantive papers + this** | [agent-verified] [preprint] |

## Tier 2 — adjacent concentrated structure (what formation work exists)

| Paper | Venue | Why | Status |
|---|---|---|---|
| Gu et al., *When Attention Sink Emerges* (arXiv:2410.10781) | ICLR 2025 Spotlight | The best formation-cause study: from-scratch ~60M, every knob ablated; the non-monotonic weight-decay sweep | [verified] |
| Sun et al., *Massive Activations in LLMs* (arXiv:2402.17762) | COLM 2024 | Defines massive activations; "emergence" there is depth-wise, not temporal; decoder-only + ViT — **no enc-dec** | [verified] |
| Ding, *A Two-Parameter Weibull Framework for Transformer Weight Distributions* (arXiv:2605.18898) | none | **The only weight-level formation trace**: Pythia-70m, 14 ckpts, outlier at \|w\|≈1.0 by step 5k (max/q99 9.8×; 14.3× at step 143k). Magnitude only, no causal test, single seed — exactly the gap Axis 1 fills | [agent-verified] [preprint] |
| Gallego-Feliciano et al., *Hidden Dynamics of Massive Activations* (arXiv:2508.03616) | none | Activation-level formation across Pythia 14M–12B, ~154 ckpts; single seed | [verified] [preprint] |
| Kovaleva et al., *BERT Busters* (2021.findings-acl.300) | Findings ACL 2021 | Encoder outlier weights + from-scratch checkpoint study at ~40M. Also: the paper everyone mis-cites as super-weight formation | [verified] |
| Puccetti et al., *Outlier Dimensions … Driven by Frequency* (arXiv:2205.11380) | Findings EMNLP 2022 | The only encoder paper with an empirical null (≥5× random-dimension damage); ties outliers to token frequency | [agent-verified] |
| He et al. (arXiv:2405.19279) | NeurIPS 2024 | The only outlier statistic with a built-in null (kurtosis, ≈1 at init) — Phase 0 ingredient | [verified] |
| Dettmers et al., *LLM.int8()* (arXiv:2208.07339) | NeurIPS 2022 | Emergent outlier features; note their own walk-back of "sudden at 6.7B" | [verified] |
| Macocco et al. (arXiv:2503.21718) | BlackboxNLP 2025 | Outlier dims across Pythia-12B checkpoints (~steps 3000–4000) | [verified] |
| Queipo-de-Llano et al., *Attention Sinks and Compression Valleys … Same Coin* (arXiv:2510.06477) | none (LeCun group) | "All three phenomena emerge together ~step 1k" — ⚠️ rests on **n=2** Pythia sizes (410M, 6.9B) | [agent-verified] [preprint] |
| Sun, Canziani, LeCun, Zhu, *The Spike, the Sparse and the Sink* (arXiv:2603.05498) | none (LeCun group) | The counterclaim: massive activations and sinks are functionally distinct; pre-norm causes co-occurrence. **Cite the disagreement with 2510.06477, not either side alone** | [agent-verified] [preprint] |
| Chen et al., *Measuring Maximum Activations in Open LLMs* (arXiv:2605.15572) | none | 27 ckpts / 8 families; activation maxima span ~4 orders of magnitude; Gemma3-27B-it ≈7×10⁵ — magnitude/criticality decoupling (matches our Gemma finding) | [agent-verified] [preprint] |
| Xu, *When Do Attention Circuits Form?* (arXiv:2606.02378) | none | Induction circuits precede sinks by 10–20× in tokens — ⚠️ scope: **DCLM models only, n=2**; the "15/15 model–task pairs" attribution floating in our docs is **not in the paper** | [agent-verified] [preprint] |

*Dropped from an earlier draft:* arXiv:2603.27885 (Hill-estimator tail index) —
verification showed it contains **no transformers or LMs at all** (MNIST/CIFAR
label noise). Do not cite for anything in this program.

## Tier 3 — checkpoint suites (the Axis-1 instruments)

| Paper | Why | Status |
|---|---|---|
| Biderman et al., *Pythia* (arXiv:2304.01373) | 154 checkpoints, log-spaced early; the backward-trace substrate | [verified] |
| van der Wal et al., *PolyPythias* (arXiv:2503.09543, ICLR 2025) | 9 seeds × 5 sizes (14M–410M) — the seed axis, and the ≤410M existence test | [verified] |
| *Ettin* (arXiv:2507.11412) | Paired encoder/decoder, identical data+recipe, 250+ ckpts, per-checkpoint batch data | [verified] |
| *DataDecide* (arXiv:2504.11393, ICML 2025) | 25 corpora × sizes ≤1B × 3 seeds. ⚠️ the ">30k checkpoints" figure is not in the abstract — treat as unverified | [agent-verified] |
| Sellam et al., *MultiBERTs* (arXiv:2106.16163, ICLR 2022) | 25 pretraining seeds + the Multi-Bootstrap — the statistical reference class | [verified] |

## Tier 4 — statistics for Phase 0

| Paper | Why | Status |
|---|---|---|
| Nichols & Holmes, *Nonparametric permutation tests for functional neuroimaging*, Human Brain Mapping 15(1), 2002 | The max-statistic permutation null — the canonical answer to "I searched 19M weights for the largest effect." Not on arXiv; obtain via library | [verified] |
| Zhao et al., *Random Scaling of Emergent Capabilities* (arXiv:2502.17356, ICML 2025) | 250-seed bimodality — why small-n bootstraps mislead on training-dynamics quantities | [verified] |

## Tier 5 — the compression bridge (now partly occupied — read as prior art)

| Paper | Why | Status |
|---|---|---|
| *Training Dynamics Impact PTQ Robustness* (arXiv:2510.06213) | **Occupies the broad "quantization sensitivity across pretraining" question** (≤32B/15T): error rises early, then **surges during LR decay**, decoupling from validation loss. Design input for Axis 2 (WSD schedule interacts) and the fence Axis 2's narrower claim must be stated against | [agent-verified] |
| *Dissecting Outlier Dynamics in NVFP4 Pretraining* (arXiv:2602.02047) | Longitudinal outlier dynamics during pretraining: transient spikes → persistent "hot channels"; the other half of the occupied space | [agent-verified] |
| arXiv:2508.20893 (uneven PTQ across languages) | The phenomenon Axis 3 would mechanize; replicated in-repo (en→X holds, X→en does not — build on en→X only) | [verified, replicated in-repo] |
| Marchisio et al., *How Does Quantization Affect Multilingual LLMs?* (arXiv:2407.03211) | Per-language quantization degradation — adjacent prior art for Axis 3 | [agent-verified] |

## Tier 6 — the multilingual bridge (Axis 3 prior art; neuron-level is occupied)

| Paper | Why | Status |
|---|---|---|
| Tang et al., *Language-Specific Neurons* / LAPE (arXiv:2402.16438) | Deactivating language-specific neurons destroys one language, spares others — the occupied neuron-granularity neighbor of Axis 3 | [agent-verified] |
| Zhao et al., *How do LLMs Handle Multilingualism?* (arXiv:2402.18815, NeurIPS 2024) | Masking ~0.13% of FFN neurons → 99% multilingual drop; "as few as four neurons" collapse a language — the closest existing result to Axis 3's question, at neuron not weight granularity | [agent-verified] |
| *Attention Sinks in Massively Multilingual NMT* (arXiv:2605.01229) | **NLLB-200 already has activation-level sink analysis** (sinks on language tags; content tokens get 17–20% of cross-attention). Weight-level search remains unclaimed — this is the paper the NLLB work must cite and extend | [agent-verified] |
| Amazon, *Emergent Outlier Properties in Pre-trained LMs* (2025.naacl-long.430) | T5-11B encoder activation outliers to ~150k magnitude — the enc-dec activation-level prior art | [agent-verified] |

## In-repo prerequisites (read before any of the above)

1. `docs/registry.md` — the q6 super-weight section, the ruled-out list, and the
   discrepancies section.
2. `interlingua/docs/method_landscape.md` §5 — detection-with-null landscape,
   planted-structure gap, formation-timing table.
3. `interlingua/docs/prior_work_map.md` §8 — what is occupied vs. open, and the
   corrections to secondhand claims (the Kovaleva mis-citation, Dettmers' 6.7B).
