CS Master's Project Draft Proposal — v2 (2026-09-05)
Adrian Castillo

The Built-In Constant: What a Handful of Weights Do for a Language Model, and Whether It Holds Across Languages

*(v1 is `proposal_draft.md`. This version moves away from the term "super weight", states the
question in one sentence, reports only the results I have run and written up in `../notes.md`,
and folds in the changes from three adversarial reviews on 2026-09-05. A run from a parked
session is mentioned once, labelled unverified, and is not relied on.)*

## 1. The phenomenon

Large language models build something their architecture does not give them: a constant. In
one early layer, one neuron of the feed-forward block fires enormously, usually on the first
token of the sequence. The down-projection weights multiply that value into a few channels of
the residual stream, and because the residual stream is additive, the value rides through the
whole network unchanged. The result is an activation a thousand times larger than everything
around it, present for every input, in almost every model examined (Sun et al., 2024; An et al.,
2025; Owen et al., 2025). Attention heads use the token that carries it as a place to put
attention they do not need (Xiao et al., 2023; Gu et al., 2025).

Two facts about this constant organise the project. First, its value carries no information:
replacing it with its average changes nothing, while removing it destroys the model (Sun et al.,
2024, Table 3; Owen et al., 2025, Table 1). The model needs the constant to be there, not to be
any particular number. Second, a handful of weights create it. Yu et al. (2024) showed that
zeroing one down-projection scalar can collapse a 7B model; Subramanian et al. (2026) replicated
their coordinates; Su et al. (2025) find the same object as a few "super experts" in
mixture-of-experts models; Li et al. (2026) describe it as a "super neuron". Whether one scalar
or a small set is critical is only a matter of how concentrated the fan-out from that neuron is.

The phenomenon is established at NeurIPS, ICLR, ICML, COLM, ACL and NAACL between 2021 and 2026,
under several names (outlier features, massive activations, super weights, super neurons,
attention sinks). What is not established is what the constant does, why some models survive
without it, and whether any of this holds outside English decoder-only models.

## 2. What I have measured so far

Working from Yu et al.'s paper alone, I re-implemented their detector and a causal ablation
harness and ran both on all nine models in their Table 2 (21 coordinates), one GPU each.

- The published coordinates are real magnitude outliers: 20 of 21 rank 1–6 by |W| inside their
  own 16–119-million-entry matrices. The 21st revealed a layer-number typo in Table 2.
- Zeroed one at a time and scored on wikitext-2 (32 windows × 2048 tokens), only 5 of 21 are
  catastrophic: OLMo-1B ×3667, Mistral-7B ×1430, Llama-7B ×181 (Yu et al. report ×214 on the same
  corpus), OLMo-7B ×79, Llama-30B ×16. My OLMo-1B number matches Subramanian et al.'s independent
  ×3663 on the same coordinate.
- The other 16 coordinates, equally extreme in magnitude, move perplexity by ×1.0–×1.6 each.
  Magnitude is not importance. In every such model the candidates share one down-projection
  input column, that is, they are spokes of one neuron's fan-out. A preliminary run from a parked
  session suggests zeroing each set jointly collapses the model; I have not verified it, and
  verifying it under my own specification is the first task below.
- With the weight zeroed, generation collapses into repeated function words: OLMo-1B produces
  "We. We. We.", Mistral-7B "the main. without without. . . ." (two models, one prompt, greedy).
- Three models outside Table 2 (Llama-3.1-8B-Instruct, Qwen3-8B, TowerBase-7B) have the constant
  but no single scalar whose removal costs more than ×1.6.
- Method lessons: the damage metric had to be validated against a published number before it
  could judge anything (four hand-written paragraphs under-estimated Llama-7B's collapse by 30×),
  and detector thresholds tuned against Table 2 are legitimate only while the answer key is public.

## 3. The question

**A built-in constant that a handful of weights create is known to be essential to language
models. We do not know what it does, why some models can live without it, or whether it is the
same across languages. Using its weights as a controlled lesion and one uniform protocol on a
dozen models, half of them multilingual and one of them a translation model, I will test the
candidate accounts of its function and measure its language dependence.**

- **RQ1 — What does the constant do?** Yu et al. found that restoring the constant after removing
  the weight recovers only 42% of the damage. So the weight matters through two routes: the
  constant it writes at the sink position, and its input-dependent contribution at every other
  position. Which route carries the collapse, and which of three accounts explains the repeated
  function words: loss of the attention sink so heads copy nearby tokens (Cancedda, 2024;
  Barbero et al., 2025); fallback to the corpus frequency prior, whose mode is function words
  (Puccetti et al., 2022; Stolfo et al., 2024; but Macocco et al., 2025 find the opposite
  direction in decoders); or a rescaling of all logits through the final norm (Stolfo et al.,
  2024)?
- **RQ2 — What is the unit, and when can "this model has none" be stated?** Is criticality a
  property of the scalar or of the neuron it belongs to, and can a detection criterion be given a
  false-positive rate? No published criterion has one (Dettmers et al. chose their thresholds to
  yield one detection in their smallest model; Hämmerl et al. show 3σ fires on whitened noise).
- **RQ3 — Is it shared across languages, and does a translation model have it?** (a) After the
  lesion, does the repeated token follow each target language's own frequent words, or is it the
  same token regardless? (b) How does damage grow with the size of the lesion, per language,
  across 20 FLORES languages? (c) Does NLLB-200 have the constant in its encoder, its decoder, or
  neither, and does the decoder's sit on the target-language tag? No published study of this
  phenomenon is multilingual or covers an encoder-decoder model.
- **RQ4 (stretch) — When does it form?** Only if a model under 1B parameters turns out to have
  a critical weight: trace its criticality across the PolyPythias checkpoints and seeds.

## 4. Method

One protocol for every model. Detect the source neuron and its weights (my v5 detector, plus an
encoder-decoder adapter for NLLB). Ablate at two granularities, the scalar and the whole neuron
column, under three conditions: set to zero; replace the weight's *contribution* with its mean over
a calibration corpus, which is the bias test Sun et al. and Owen et al. ran on activations, made
well-defined for a weight; and a magnitude-matched random weight from the same matrix, with the
maximum over 50 draws as the null. Score damage the same way every time: perplexity with a
paired-bootstrap interval, a repetition measure (a looping generation can have low perplexity),
and for multilingual models per-language chrF++, COMET and forced-decode KL. Every result file
carries the git hash, resolved config, library versions, seed and pinned model revision; one YAML
per run; a planted-weight test the detector must pass.

**RQ1**, on the five models with an established catastrophic ablation. Position-split ablation:
zero the weight's effect at the sink position only, then at all other positions only. Dose sweep
of the weight through {1, 0.75, 0.5, 0.25, 0.1, 0}. Sink mass and per-head attention entropy.
KL of the next-token distribution to the corpus unigram, against a norm-matched random residual
decoded through the final norm. The frozen-final-norm ablation of Stolfo et al. to size the
rescaling account. A regression of the logit change on the intact logits and the centred
log-frequency direction, with partial R² (the all-ones direction is annihilated by softmax and
is not used). Two dissociations: patch the constant back in at its onset layer with *generation*
as the readout, and remove the sink without touching the weight by masking attention to the sink
position (removing the BOS token does not work; it moves the sink to whatever token is first).
Satisfied when the position split is measured per model with intervals and each account is
supported, excluded, or shown to be partial.

**RQ2**, on the twelve cached models. Scalar versus column versus gate/up rows, crossed with the
three conditions; does fan-out count predict the individual-versus-joint gap? Criterion: the
ablation ratio against the maximum of the magnitude-matched null, stated per model with coverage.
Validation: recovers Yu et al.'s coordinates, does not flag the top-magnitude coordinates
Subramanian et al. show are inert, and recovers a weight planted in a small model. Models whose
ratio sits at the null ceiling are reported as undecidable, not as negatives.

**RQ3.** (a) Teacher-force references in each of ten target languages through the ablated model,
average the full next-token distribution, and rank-correlate it against that language's unigram
distribution and against the other languages'; the null is the model's own unconditional unigram.
(b) On TowerBase-7B and EuroLLM-9B, the dose sweep per language: the dose at which chrF++ falls to
half its intact value, with a damage-matched noise control on the same matrix. Correlations with
resource level or with the lab's per-language quantization results are reported as exploratory.
(c) NLLB-600M and 3.3B (600M is distilled): detection per stack, per-position residual norms,
decoder self-attention sink rate. In the HuggingFace implementation the decoder starts with the
end-of-sentence token and the language tag is position 1, which separates "first position" from
"language tag" for free. Readout for the decoder: output language identification and off-target
rate under the dose sweep.

## 5. Related work and positioning

I anchor on Sun et al. (COLM 2024), An et al. (ICLR 2025) and Subramanian et al. (COLM 2026);
Yu et al. (2024) is the preprint that supplied the coordinate directory and was rejected from
ICLR 2025 mainly on the quantization method it proposed, which this project does not pursue. The
project's object differs from language-specific neurons (Tang et al., 2024; Zhao et al., 2024),
whose removal disables one language, and from the sparse residual dimensions of Zhong et al.
(2025) that switch the output language: this constant's removal destroys every language,
including English. Per-language quantization degradation is measured (Marchisio et al., 2024;
Marie & Fujita, 2025) and unexplained; RQ3 asks whether this constant is part of the explanation,
without claiming a quantization method.

## 6. Plan (about 130 hours)

| Weeks | Work | Hours |
|---|---|---|
| 1–3 | Harness: contribution-mean control, magnitude-matched null, repetition metric, bootstrap CIs, provenance, config and planted-weight tests | 20 |
| 3–5 | RQ2 on the twelve cached models; re-verify the joint-set lead under my own spec | 20 |
| 5–8 | RQ1 on the five catastrophic models | 35 |
| 8–9 | RQ3(a); **week-9 draft report** | 10 |
| 9–12 | RQ3(b) dose sweep on TowerBase and EuroLLM; RQ3(c) NLLB gate | 25 |
| 13–15 | Writing and presentation; RQ4 only if time and the sub-1B gate are positive | 25 |

Everything is forward passes on cached models as single-GPU SLURM jobs. No training.

## 7. Deliverables

- Written and verbal reports.
- A mechanistic account of the collapse, per model, with controls and intervals, whichever way it
  falls.
- The first uniform table of this phenomenon across a dozen models including multilingual and
  translation models: scalar and neuron granularity, three conditions, coverage stated.
- A tested, installable analysis package: detector with a criterion that has a false-positive
  rate, the control triple, repetition metrics, provenance in every result, one YAML per run,
  reproducible from documented commands.

## 8. Risks

The constant may mediate only part of the effect, so RQ1 may end in a measured split rather than
one winner; that is the result. Full ablation floors translation metrics, which is why RQ3 uses
the dose sweep rather than zeroing. NLLB is non-GLU with LayerNorm, the class Owen et al. found
unaffected by removing the constant, so a bounded negative is likely and is budgeted as a gate.
Twelve models from six architecture families is exploratory; no cross-model claim will be made
without its coverage stated, and no per-language correlation will be called more than exploratory.

## Bibliography

*(Venues as verified against arXiv or ACL Anthology metadata; entries without a venue were not
venue-verified.)*

An, M., et al. (2025). Systematic outliers in large language models. ICLR 2025. arXiv:2502.06415.
Barbero, F., et al. (2025). Why do LLMs attend to the first token? COLM 2025. arXiv:2504.02732.
Cancedda, N. (2024). Spectral filters, dark signals, and attention sinks. arXiv:2402.09221.
Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. NeurIPS 2022. arXiv:2208.07339.
Gu, X., Pang, T., Du, C., Liu, Q., Zhang, F., Du, C., Wang, Y., & Lin, M. (2025). When attention sink emerges in language models: An empirical view. ICLR 2025. arXiv:2410.10781.
Hämmerl, K., et al. (2023). Exploring anisotropy and outliers in multilingual language models for cross-lingual semantic sentence similarity. arXiv:2306.00458.
Li, et al. (2026). The structural origin of attention sink. ICML 2026 (stated on the PDF). arXiv:2605.06611.
Macocco, I., et al. (2025). Outlier dimensions across Pythia checkpoints. arXiv:2503.21718.
Marchisio, K., Dash, S., Chen, H., Aumiller, D., Üstün, A., Hooker, S., & Ruder, S. (2024). How does quantization affect multilingual LLMs? arXiv:2407.03211.
Marie, B., & Fujita, A. (2025). Uneven post-training quantization degradation across languages in multilingual machine translation. arXiv:2508.20893.
NLLB Team et al. (2022). No language left behind: Scaling human-centered machine translation. arXiv:2207.04672.
Owen, L., Roy Chowdhury, N., Kumar, A., & Güra, F. (2025). A refined analysis of massive activations in LLMs. arXiv:2503.22329.
Puccetti, G., Rogers, A., Drozd, A., & Dell'Orletta, F. (2022). Outlier dimensions that disrupt transformers are driven by frequency. Findings of EMNLP 2022. arXiv:2205.11380.
Stolfo, A., Wu, B., Gurnee, W., Belinkov, Y., Song, X., Sachan, M., & Nanda, N. (2024). Confidence regulation neurons in language models. NeurIPS 2024. arXiv:2406.16254.
Su, Z., et al. (2025). Unveiling super experts in mixture-of-experts large language models. ICLR 2026. arXiv:2507.23279.
Subramanian, S., Akinfaderin, A., & Sehwag, A. (2026). Super weights in LLMs and the failure of selective training. COLM 2026. arXiv:2607.08733.
Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive activations in large language models. COLM 2024. arXiv:2402.17762.
Tang, T., Luo, W., Huang, H., et al. (2024). Language-specific neurons: The key to multilingual capabilities in large language models. ACL 2024. arXiv:2402.16438.
van der Wal, O., Lesci, P., Müller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). PolyPythias: Stability and outliers across fifty language model pre-training runs. ICLR 2025. arXiv:2503.09543.
Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient streaming language models with attention sinks. arXiv:2309.17453.
Yu, M., Wang, D., Shan, Q., Reed, C. J., & Wan, A. (2024). The super weight in large language models. arXiv:2411.07191.
Zhao, Y., Zhang, W., Chen, G., Kawaguchi, K., & Bing, L. (2024). How do large language models handle multilingualism? NeurIPS 2024. arXiv:2402.18815.
Zhong, et al. (2025). Language lives in sparse dimensions: Toward interpretable and efficient multilingual control for large language models. EACL 2026. arXiv:2510.07213.
