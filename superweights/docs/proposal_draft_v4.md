CS Master's Project Draft Proposal — v4 (2026-09-05)
Adrian Castillo

The Built-In Constant in Multilingual Language Models: What It Does, Which Models Depend on It, and Whether It Shapes How Low-Resource Languages Break

*(v1 is `proposal_draft.md`; v2 was mechanism-first, v3 problem-first. v4 merges them after six
adversarial reviews: v3's problems supply the motivation, v2's method is the mandatory core,
and one intervention arm is kept behind a pilot gate. The hypotheses are not expected to be true;
the design is such that each answer, positive or negative, is a result. Only numbers I have run
and written up in `../notes.md` are reported as established; one run from a parked session is
mentioned as unverified and is not relied on.)*

## 1. The phenomenon

Language models build a constant their architecture does not provide. In one early layer, one
feed-forward neuron fires enormously on one token, usually the first; its down-projection
weights write that value into a few residual-stream channels; the value rides the residual
stream unchanged to the last layer, a thousand times larger than everything around it, present
for every input (Sun et al., 2024; An et al., 2025; Owen et al., 2025). Attention heads use the
token that carries it as a place to put attention they do not need (Xiao et al., 2023; Gu et
al., 2025). Its value carries no information: replacing it with its average changes nothing,
removing it destroys the model (Sun et al., 2024, Table 3; Owen et al., 2025, Table 1). A handful
of weights create it. Yu et al. (2024) showed one down-projection scalar can be enough;
Subramanian et al. (2026) replicated their coordinates; Su et al. (2025) find the same object as
a few "super experts" in mixture-of-experts models; Li et al. (2026) describe it as a "super
neuron". Whether one scalar or a small set is critical is only how concentrated the fan-out from
that neuron is.

The phenomenon is established across NeurIPS, ICLR, ICML, COLM, ACL and NAACL from 2021 to 2026.
Three things are not: what the constant does for generation (why removing it produces loops of
function words rather than noise); which models depend on it and why (Owen et al. find removing
it harmless in about 7 of 16 models and offer no indicator); and whether any of this holds
outside English decoder-only models. No study of it is multilingual or covers a translation model.

## 2. Why a translation lab should care

**Quantization hurts low-resource languages more.** Marie & Fujita (2025) report Qwen3-8B losing
about 2 COMET points on French and Japanese but 17 on Bengali and Malayalam at 2-bit weight-only
quantization; Marchisio et al. (2024) find non-Latin scripts hit hardest. My lab replicated the
English-to-X effect under 4-bit weight-only quantization; the X-to-English effect did not
replicate. The explanations on offer are training-data volume and script. The constant and its
token are precisely what activation quantization has to route around (SmoothQuant, QuaRot,
PrefixQuant, DuQuant all exist for this, all evaluated in English), and the lab's NLLB deployment
runs on CTranslate2 int8, which quantizes activations per row at runtime, that is, dynamic W8A8.
Whether the constant contributes to a per-language gap *in that regime* is unasked. Two
constraints from this repo bound the claim: keeping the creating weights in FP16 under
weight-only quantization is a no-op on eight models, and parameter importance is orthogonal to
quantization sensitivity, so any effect must run through the activations, not the weights.

**Translation models hallucinate, often by repeating.** The output of a model whose constant is
removed is a loop of function words (§3), which by Raunak et al.'s (2021) definition is an
oscillatory hallucination; Guerreiro et al. (2023) find that category dominates in mid- and
high-resource directions of NLLB and M2M. I do not claim the two share a mechanism: one is a
model-wide lesion, the other an input-conditional failure of a working model, and the only
annotated benchmark (HalOmi; Dale et al., 2023b) was decoded with repeated 4-grams forbidden.
What can be asked cheaply is whether the sink's attention statistics in a healthy NLLB carry any
hallucination information beyond sequence log-probability (§5, stretch).

## 3. What I have measured so far (established)

Working from Yu et al.'s paper alone, I re-implemented their detector and a causal ablation
harness and ran both on all nine models in their Table 2 (21 coordinates).

- The published coordinates are real magnitude outliers: 20 of 21 rank 1–6 by |W| inside their
  own 16–119-million-entry matrices; the 21st revealed a layer-number typo in Table 2.
- Zeroed one at a time and scored on wikitext-2 (32 windows × 2048 tokens), only 5 of 21 are
  catastrophic: OLMo-1B ×3667 (Subramanian et al. report ×3663 independently on the same
  coordinate), Mistral-7B ×1430, Llama-7B ×181 (Yu et al. report ×214 on the same corpus),
  OLMo-7B ×79, Llama-30B ×16. The other 16, equally extreme in magnitude, move perplexity by
  ×1.0–×1.6 each and sit on one shared down-projection input column per model, that is, they are
  spokes of one neuron's fan-out. A preliminary run from a parked session suggests each set is
  jointly catastrophic; I have not verified it, and doing so under my own specification is the
  first task below.
- With the weight zeroed, greedy generation from one prompt collapses into repeated function
  words in the two models checked: OLMo-1B "We. We. We.", Mistral-7B "the main. without without".
- Three models outside Table 2 (Llama-3.1-8B-Instruct, Qwen3-8B, TowerBase-7B) have the constant
  but no single scalar whose removal costs more than ×1.6.
- Two method lessons already paid for: the damage metric had to be validated against a published
  number (a hand-written evaluation set under-estimated Llama-7B's collapse by 30×), and detector
  thresholds tuned against a published answer key are legitimate only while the answer key is
  public.

## 4. Questions

**RQ1 — Where is the constant in multilingual and translation models, which weights create it,
and is it load-bearing?** For TowerBase-7B, EuroLLM-9B, Aya-Expanse-8B, BLOOM-7B1, NLLB-600M and
NLLB-3.3B, plus the decoder-only models already done: one uniform protocol, three controls, two
granularities. This table does not exist for any multilingual or encoder-decoder model. It also
answers the unit question: is a scalar catastrophic only when it carries most of one neuron's
fan-out?

**RQ2 — What does the constant do, such that removing it produces function-word loops?** Yu et
al. found that restoring the constant after zeroing the weight recovers only 42% of the damage.
So the weight acts through two routes: the constant it writes at the sink position, and its
input-dependent contribution at every other position. Which route carries the collapse, and
which of three accounts explains the loops: loss of the attention sink so heads copy nearby
tokens (Cancedda, 2024; Barbero et al., 2025); fallback to the corpus frequency prior (Puccetti
et al., 2022; Stolfo et al., 2024; Macocco et al., 2025 find the opposite direction in decoders);
or rescaling of all logits through the final norm (Stolfo et al., 2024)?

**RQ3 — Is the constant shared across languages?** After the lesion, does the repeated token
follow each target language's own frequent words, or is it the same regardless? How does damage
grow with lesion size per language? In NLLB, is the constant in the encoder, the decoder, or
neither, and does the decoder's sit at position 0 (the end-of-sentence token) or position 1 (the
target-language tag)?

**RQ4 (gated) — Does the constant contribute to the per-language quantization gap under
dynamic W8A8, the regime the lab deploys?** Hypothesis: quantizing the massive channels at the
sink token distorts the bias every position reads, and languages with flatter next-token
distributions are more sensitive to it. Not expected to be true; designed to be decidable.

## 5. Method

**Common protocol (RQ1, RQ3).** Detect the source neuron and its weights (my v5 detector; an
encoder-decoder adapter for NLLB, detecting per stack). Ablate at two granularities, the scalar
and the whole neuron column, under three conditions: zero; replace the weight's *contribution*
with its mean over a calibration corpus (the bias test Sun et al. and Owen et al. ran on
activations, made well-defined for a weight); a magnitude-matched random weight from the same
matrix, maximum over 50 draws as the null. A model whose ratio sits at the null ceiling is
reported as undecidable, not as inert. Score perplexity with a paired-bootstrap interval, a
repetition measure (rep-n; a looping generation can have low perplexity), and for translation
models per-language chrF++ and COMET on a dose sweep of the weight through {1, 0.75, 0.5, 0.25,
0.1, 0} rather than full zeroing, which floors every translation metric. Every result file
carries the git hash, resolved config, library versions, seed and pinned model revision; one
YAML per run; a planted-weight test the detector must pass; a test that every committed config
loads.

**RQ2**, on the five models with an established catastrophic ablation. Position-split ablation:
zero the weight's effect at the sink position only, then at all other positions only. Dose sweep.
Sink attention mass and per-head attention entropy. KL of the next-token distribution to the
corpus unigram, against a norm-matched random residual decoded through the final norm.
Stolfo et al.'s frozen-final-norm ablation to size the rescaling account. A regression of the
logit change on the intact logits and the centred log-frequency direction, with partial R².
Two dissociations: patch the constant back at its onset layer with *generation* as the readout,
and remove the sink without touching the weight by masking attention to the sink position
(removing the BOS token does not work; it moves the sink to whatever token is first). Satisfied
when the position split is measured per model with intervals and each account is supported,
excluded, or shown to be partial.

**RQ3.** (a) Teacher-force references in ten target languages through the ablated Tower,
EuroLLM and NLLB, average the full next-token distribution, and rank-correlate it against that
language's unigram distribution and against the other languages'; the null is the model's own
unconditional unigram. (b) On TowerBase-7B and EuroLLM-9B, the dose sweep per language over ten
en→X FLORES directions: the dose at which chrF++ halves, with a damage-matched noise control on
the same matrix; any correlation with resource level is exploratory. (c) NLLB-600M and 3.3B:
per-position residual norms and decoder self-attention sink rate at positions 0 and 1; a control
that swaps the language tag for an arbitrary token; readout under the dose sweep is output
language identification and off-target rate.

**RQ4, gated.** *Pilot (week 6):* TowerBase-7B, four FLORES directions spanning resource
levels, FP16 versus simulated per-token dynamic W8A8 (fake-quantization on every linear layer,
labelled simulated), paired bootstrap over sentences, three calibration draws. Pre-registered
gate: the low-resource minus high-resource COMET delta must exceed its 95% interval. *If the gate
passes:* isolate the constant by Yu et al.'s channel-level procedure (median-replace the massive
channels at the sink token, quantize, restore in FP16) against two controls, a position-matched
non-sink token restored and the sink's non-massive channels restored; add bitsandbytes int8
(which already routes outlier channels to FP16) as a free arm; run TowerBase and NLLB-3.3B on ten
en→X directions with chrF++, COMET and output language identification. *If the gate fails:* report
the bounded null as the first activation-level per-language test at 8-bit and stop; the 2- and
4-bit weight-only gap remains explained, if at all, by other causes.

**Stretch — hallucination signal.** Teacher-force HalOmi's NLLB-600M outputs through NLLB-600M;
test whether decoder self-attention sink mass adds detection power to sequence log-probability
(nested ΔAUC with a paired-bootstrap interval, pooled across directions with direction-stratified
cross-validation, baselines recomputed in-house), and whether sink mass shifts before the first
hallucinated token at the word level. Cross-attention statistics are excluded (HalOmi's appendix
shows them near chance).

## 6. Plan (about 140 hours)

| Weeks | Work | Hours |
|---|---|---|
| 1–3 | Harness: three controls, repetition metric, bootstrap CIs, provenance, config and planted-weight tests; encoder-decoder adapter | 25 |
| 3–6 | RQ1 table on the six multilingual/translation models; re-verify the joint-set lead; RQ3(c) positions in NLLB | 30 |
| 6 | RQ4 pilot and gate decision | 8 |
| 6–9 | RQ2 on the five catastrophic models; **week-9 draft report** with the RQ1 table, RQ2 first results, and the gate outcome | 30 |
| 9–12 | RQ3(a,b); RQ4 full arm if the gate passed, otherwise the stretch | 25 |
| 13–15 | Writing and presentation | 22 |

Everything runs as single-GPU SLURM jobs on cached models. The only new tooling of substance is
the simulated W8A8 path (about one week), and nothing else depends on it.

## 7. Deliverables

- Written and verbal reports.
- The first uniform table of this phenomenon across multilingual and translation models, with
  controls, granularities and coverage stated.
- A mechanistic account of the collapse, per model, with controls and intervals, whichever way
  it falls.
- A decided RQ4 gate, and if it passes, an answer with intervals to whether handling the
  constant's channels changes the per-language gap under the deployed quantization regime.
- A tested, installable analysis package (detector with a criterion that has a false-positive
  rate, control triple, repetition and per-language metrics, provenance in every result),
  reproducible from documented commands.

## 8. Risks

The constant may mediate only part of the effect, so RQ2 may end in a measured split rather than
one winner; that is the result. NLLB is non-GLU with LayerNorm, the class Owen et al. found
unaffected by removing the constant; the RQ1 table may show it present but inert there, which is
a finding. The RQ4 gate may fail because an 8-bit gap is below the interval; the pilot exists so
that this costs eight hours, not fifty. Twelve models from six architecture families and ten
languages are exploratory samples; every cross-model or cross-language claim states its
coverage, and no correlation with resource level is called more than exploratory. Yu et al.
(2024) is a preprint rejected from ICLR 2025 mainly on the quantization method it proposed; this
project anchors on Sun et al., An et al. and Subramanian et al. and uses Yu et al. as the source
of the coordinate directory only.

## Bibliography

*(Venues as verified against arXiv or ACL Anthology metadata; entries without a venue were not
venue-verified.)*

An, M., et al. (2025). Systematic outliers in large language models. ICLR 2025. arXiv:2502.06415.
Ashkboos, S., et al. (2024). QuaRot: Outlier-free 4-bit inference in rotated LLMs. NeurIPS 2024. arXiv:2404.00456.
Barbero, F., et al. (2025). Why do LLMs attend to the first token? COLM 2025. arXiv:2504.02732.
Cancedda, N. (2024). Spectral filters, dark signals, and attention sinks. arXiv:2402.09221.
Chen, M., et al. (2024). PrefixQuant: Static quantization beats dynamic through prefixed outliers in LLMs. arXiv:2410.05265.
Dale, D., Voita, E., Barrault, L., & Costa-jussà, M. R. (2023a). Detecting and mitigating hallucinations in machine translation: Model internal workings alone do well, sentence similarity even better. ACL 2023. arXiv:2212.08597.
Dale, D., Voita, E., et al. (2023b). HalOmi: A manually annotated benchmark for multilingual hallucination and omission detection in machine translation. EMNLP 2023. arXiv:2305.11746.
Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. NeurIPS 2022. arXiv:2208.07339.
Gu, X., Pang, T., Du, C., Liu, Q., Zhang, F., Du, C., Wang, Y., & Lin, M. (2025). When attention sink emerges in language models: An empirical view. ICLR 2025. arXiv:2410.10781.
Guerreiro, N. M., Alves, D., Waldendorf, J., Haddow, B., Birch, A., Colombo, P., & Martins, A. F. T. (2023). Hallucinations in large multilingual translation models. TACL 2023. arXiv:2303.16104.
Li, et al. (2026). The structural origin of attention sink. ICML 2026 (stated on the PDF). arXiv:2605.06611.
Lin, H., et al. (2024). DuQuant: Distributing outliers via dual transformation makes stronger quantized LLMs. NeurIPS 2024. arXiv:2406.01721.
Macocco, I., et al. (2025). Outlier dimensions across Pythia checkpoints. arXiv:2503.21718.
Marchisio, K., Dash, S., Chen, H., Aumiller, D., Üstün, A., Hooker, S., & Ruder, S. (2024). How does quantization affect multilingual LLMs? arXiv:2407.03211.
Marie, B., & Fujita, A. (2025). Uneven post-training quantization degradation across languages in multilingual machine translation. arXiv:2508.20893.
NLLB Team et al. (2022). No language left behind: Scaling human-centered machine translation. arXiv:2207.04672.
Owen, L., Roy Chowdhury, N., Kumar, A., & Güra, F. (2025). A refined analysis of massive activations in LLMs. arXiv:2503.22329.
Puccetti, G., Rogers, A., Drozd, A., & Dell'Orletta, F. (2022). Outlier dimensions that disrupt transformers are driven by frequency. Findings of EMNLP 2022. arXiv:2205.11380.
Raunak, V., Menezes, A., & Junczys-Dowmunt, M. (2021). The curious case of hallucinations in neural machine translation. NAACL 2021. arXiv:2104.06683.
Stolfo, A., Wu, B., Gurnee, W., Belinkov, Y., Song, X., Sachan, M., & Nanda, N. (2024). Confidence regulation neurons in language models. NeurIPS 2024. arXiv:2406.16254.
Su, Z., et al. (2025). Unveiling super experts in mixture-of-experts large language models. ICLR 2026. arXiv:2507.23279.
Subramanian, S., Akinfaderin, A., & Sehwag, A. (2026). Super weights in LLMs and the failure of selective training. COLM 2026. arXiv:2607.08733.
Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive activations in large language models. COLM 2024. arXiv:2402.17762.
Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient streaming language models with attention sinks. arXiv:2309.17453.
Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023b). SmoothQuant: Accurate and efficient post-training quantization for large language models. ICML 2023. arXiv:2211.10438.
Yu, M., Wang, D., Shan, Q., Reed, C. J., & Wan, A. (2024). The super weight in large language models. arXiv:2411.07191.
