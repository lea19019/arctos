CS Master's Project Draft Proposal — v3 (2026-09-05, problem-first)
Adrian Castillo

Does a Model's Built-In Constant Explain Why Low-Resource Languages Break First?
Massive activations in multilingual and translation models, studied through two problems they may cause

*(v1 is `proposal_draft.md`; v2 is `proposal_draft_v2.md`, a mechanism-first framing of the same
phenomenon. v3 starts from two multilingual problems that certainly exist, states a hypothesis
linking each to the phenomenon, and proposes interventions whose effect on the problem is the
headline measurement. The hypotheses are not expected to be true; the project is designed so
that each answer, positive or negative, is a result. Only numbers I have run and written up in
`../notes.md` are reported as established.)*

## 1. Two problems that certainly exist

**Problem 1 — quantization hurts low-resource languages more.** When a multilingual model is
quantized, translation quality drops unevenly across languages: Marie & Fujita (2025) report
Qwen3-8B losing about 2 COMET points on French and Japanese but 17 on Bengali and Malayalam at
2-bit, and Marchisio et al. (2024) find non-Latin scripts hit hardest, with human evaluation
showing roughly ten times the damage automatic metrics report. My lab replicated the effect for
English-to-X translation. The explanations offered so far are training-data volume and script.
No paper has tested an activation-level cause, and the only known mitigation, language-matched
calibration data, helps only at 2 bits (Marie & Fujita) or in the multilingual-calibration
variant of Zhang et al. (2024). The lab deploys NLLB in int8 on 16 GB GPUs for a dubbing
pipeline, so this is a problem we have, not one we borrowed.

**Problem 2 — translation models hallucinate, and often by repeating.** Guerreiro et al. (2023)
annotated hallucinations in NLLB and M2M across more than 100 directions: "oscillatory"
hallucinations (translations that loop on repeated n-grams) dominate in mid- and high-resource
directions, "detached" ones in low-resource directions. Detection today relies on sequence
log-probability, source-contribution attribution (Dale et al., 2023) and n-gram heuristics; the
HalOmi benchmark (Dale et al., 2023b) supplies 18 human-annotated directions on NLLB output.
Nothing in this literature looks inside the model at outlier structure.

## 2. The phenomenon, and why it plausibly connects to both problems

Language models build a constant their architecture does not provide. In one early layer, one
feed-forward neuron fires enormously on one token, usually the first; its down-projection
weights write that value into a few residual channels; the value rides the residual stream
unchanged to the last layer, a thousand times larger than everything around it, present for every
input (Sun et al., 2024; An et al., 2025; Owen et al., 2025). Attention heads use that token as a
place to put attention they do not need (Xiao et al., 2023; Gu et al., 2025). Replacing the
constant with its average changes nothing; removing it destroys the model (Sun et al., 2024,
Table 3; Owen et al., 2025, Table 1). A handful of weights create it: Yu et al. (2024) showed one
down-projection scalar can be enough; Subramanian et al. (2026) replicated their coordinates; Su
et al. (2025) find the same object as a few "super experts" in mixture-of-experts models. The
phenomenon is established across NeurIPS, ICLR, ICML, COLM, ACL and NAACL; what it does, which
models depend on it, and whether it behaves the same across languages are open, and no study of
it is multilingual or covers a translation model.

**Why it may cause Problem 1.** The constant and its token are precisely what activation
quantization fights. SmoothQuant scales the outlier channels into the weights; QuaRot and
SpinQuant rotate them away; PrefixQuant holds the sink token out of the quantization grid;
DuQuant names "massive outliers" as what channel-wise methods miss (Xiao et al., 2023b; Ashkboos
et al., 2024; Chen et al., 2024; Lin et al., 2024). All of these are evaluated in English.
Two ways the constant could produce a language-dependent gap: (a) quantization error at the sink
token distorts the constant every position reads, and languages whose next-token distributions
are flatter (higher-fertility tokenization, less training data) are more sensitive to a distorted
bias; (b) per-channel quantization scales fitted on English calibration data are dominated by
the massive channels and misfit the activation ranges other languages produce. These two make
different predictions about which intervention closes the gap, which is what makes them testable.

**Why it may relate to Problem 2.** The output of a model whose constant is removed is a loop of
function words: OLMo-1B produces "We. We. We.", Mistral-7B "the main. without without" (my
measurements, §3). That is, by Raunak et al.'s (2021) definition, an oscillatory hallucination.
Cancedda (2024) produces the same loops by filtering the sink's residual component in Llama-2,
and one 2026 preprint finds sink statistics predictive of hallucination in decoder-only LLMs
(Binkowski et al., 2026). Whether hallucinated translations in a healthy NLLB show anomalous
constant or sink statistics is unmeasured.

## 3. What I have measured so far (established)

Working from Yu et al.'s paper alone, I re-implemented their detector and a causal ablation
harness and ran both on all nine models in their Table 2 (21 coordinates). The published
coordinates are real magnitude outliers (20 of 21 rank 1–6 by |W| in their matrices; one layer
typo found). Zeroed individually and scored on wikitext-2, only 5 of 21 are catastrophic
(OLMo-1B ×3667, matching Subramanian et al.'s independent ×3663; Mistral-7B ×1430; Llama-7B ×181
against Yu et al.'s ×214; OLMo-7B ×79; Llama-30B ×16). The other 16, equally extreme in
magnitude, move perplexity by ×1.0–×1.6 each and sit on one shared down-projection input column
per model, that is, they are spokes of one neuron's fan-out; a preliminary, unverified run
suggests each set is jointly catastrophic. Generation after ablation collapses into repeated
function words. Three models outside Table 2 (Llama-3.1-8B-Instruct, Qwen3-8B, TowerBase-7B)
have the constant but no single scalar whose removal costs more than ×1.6. Two method lessons
already paid for: the damage metric had to be validated against a published number (a
hand-written evaluation set under-estimated Llama-7B's collapse by 30×), and detector thresholds
tuned against a published answer key are legitimate only while the answer key is public.

## 4. Hypotheses and what would count as an answer

**H1 (Problem 1).** Part of the per-language quantization gap is caused by how the sink token
and its channels are quantized. *Prediction:* holding the sink token out of the activation
quantization grid, or fitting channel scales without it, shrinks the gap between high- and
low-resource languages more than it improves the average. *Discriminating sub-predictions:*
if (a) is right, exempting the sink token closes the gap even with English calibration; if (b) is
right, language-matched calibration closes it and sink exemption adds little. *Answer looks
like:* per-language COMET deltas versus FP16 under each scheme, with bootstrap intervals over
sentences and a gap statistic (spread across languages, and its slope against resource level)
with a bootstrap interval over languages. "The gap is unchanged" is reported as an interval, not
as no effect.

**H2 (Problem 2).** Hallucinated translations in NLLB show measurably different constant or sink
statistics during decoding than clean ones. *Prediction:* sink attention mass, the constant's
magnitude at the decoder's first positions, and their variation across decoding steps separate
HalOmi's hallucinated from non-hallucinated outputs at least as well as sequence log-probability
does, and add information to it. *Answer looks like:* detection AUC per direction with intervals
against the published baselines (Dale et al., 2023a), and a statement of whether the internal
signal is redundant with log-probability. A negative here is a clean negative.

**Instrument (needed by both).** Where is the constant in each model, which weights create it,
and is it critical? One uniform protocol on Tower-7B, EuroLLM-9B, Aya-Expanse-8B, BLOOM-7B1,
NLLB-600M and NLLB-3.3B, plus the decoder-only models already done: detect the source neuron and
its weights; ablate at scalar and whole-neuron granularity under three conditions (zero; replace
the weight's contribution with its calibration-corpus mean; a magnitude-matched random weight,
max over 50 draws as the null); score perplexity, a repetition measure and, for translation
models, per-language chrF++/COMET on a dose sweep of the weight rather than full zeroing. This
table does not exist for any multilingual or encoder-decoder model. It tells the intervention arm
what to hold out, and it is where the mechanism questions (does the constant's criticality come
from the sink position or from other positions; does the repeated token follow each language's
frequency table) get answered as supporting results.

## 5. Design

**Arm A — quantization intervention (Problem 1).** Models: TowerBase-7B, EuroLLM-9B, NLLB-3.3B.
Languages: 20 FLORES-200 directions spanning resource levels and scripts, en→X and X→en, using
the lab's existing n≈960 chat-template COMET protocol. Conditions: FP16 reference; W8A8 naive;
W8A8 with the sink token exempt from activation quantization (PrefixQuant-style); W8A8 with
per-channel smoothing (SmoothQuant); each with English calibration and with language-matched
calibration; weight-only 4-bit as the reference the lab already has. Metrics: per-language COMET
and chrF++ deltas versus FP16, the gap statistic above, and per-language output language-ID (to
catch off-target output). Controls: calibration-set choice is a first-order confound (Williams &
Aletras, 2024) and is crossed deliberately; tokenizer fertility per language is recorded and
reported as a covariate.

**Arm B — hallucination signal (Problem 2).** Data: HalOmi (NLLB-600M outputs, 18 directions,
sentence-level hallucination and omission labels). Procedure: teacher-force each annotated
translation through NLLB-600M and NLLB-3.3B; record per decoding step the attention mass on the
sink position(s), the constant's magnitude in the decoder residual stream, per-layer entropy, and
the encoder-side statistics on the source. Fit a simple detector (logistic regression on those
features) with cross-validation by direction; compare to sequence log-probability, ALTI+ source
contribution and TNG as reported by Dale et al. (2023a) and Guerreiro et al. (2023); test
redundancy by adding the internal features to log-probability. Report AUC per direction and
pooled, with intervals.

**Arm C — instrument and mechanism.** The uniform table (§4), then on the models where the
constant is critical: position-split ablation (the weight's effect at the sink position only
versus all other positions only), the dose sweep, and the repeated-token identity across ten
teacher-forced target languages against each language's unigram distribution. These explain
whatever Arms A and B find.

## 6. Plan (about 135 hours)

| Weeks | Work | Hours |
|---|---|---|
| 1–3 | Harness: controls, repetition metric, bootstrap CIs, provenance, config and planted-weight tests; encoder-decoder adapter; activation-quantization path (W8A8, sink exemption) | 30 |
| 3–6 | Arm C table on the six multilingual/translation models; re-verify the joint-set lead | 25 |
| 6–9 | Arm A on TowerBase and EuroLLM; **week-9 draft report** with the table and first Arm A results | 25 |
| 9–12 | Arm A on NLLB-3.3B; Arm B on HalOmi | 30 |
| 13–15 | Arm C mechanism items as time allows; writing and presentation | 25 |

Everything runs as single-GPU SLURM jobs on cached models, one YAML per run. The only new
tooling of substance is the activation-quantization path, budgeted in week 1–3; if it slips,
Arm A runs on weight-only quantization first and the sink-exemption condition is the stretch.

## 7. Deliverables

- Written and verbal reports.
- An answer to H1 with intervals: whether handling the sink token changes the per-language
  quantization gap, and which of (a)/(b) the pattern supports. A positive result is a
  deployment-relevant recipe; a negative rules out an activation-level cause the field has not
  tested.
- An answer to H2 with intervals: whether the model's constant and sink statistics carry
  hallucination information beyond log-probability.
- The first uniform table of this phenomenon across multilingual and translation models, with
  controls and coverage stated.
- A tested, installable analysis package (detector, control triple, repetition and per-language
  metrics, provenance in every result), reproducible from documented commands.

## 8. Risks

H1 may be false: the gap may be entirely a training-data and tokenizer effect. The design
detects that (sink exemption changes nothing; matched calibration does), and that negative is
still the first activation-level test of the question. Full ablation floors translation metrics,
which is why Arm C uses a dose sweep. NLLB is non-GLU with LayerNorm, the class Owen et al. found
unaffected by removing the constant, so the constant may be present but inert there; Arm A does
not depend on it being critical, only on it being quantized. HalOmi outputs come from the
distilled 600M model; the 3.3B arm checks transfer. The internal hallucination signal may reduce
to log-probability; the redundancy test is designed to show that. Activation-quantization tooling
is the largest engineering risk and has a fallback. Twenty languages and six models are
exploratory samples; every cross-language claim states its coverage, and no correlation with
resource level is called more than exploratory.

## Bibliography

*(Venues as verified against arXiv or ACL Anthology metadata; entries without a venue were not
venue-verified.)*

An, M., et al. (2025). Systematic outliers in large language models. ICLR 2025. arXiv:2502.06415.
Ashkboos, S., et al. (2024). QuaRot: Outlier-free 4-bit inference in rotated LLMs. NeurIPS 2024. arXiv:2404.00456.
Binkowski, J., Adamczewski, K., & Kajdanowicz, T. (2026). Attention sinks as internal signals for hallucination detection. arXiv:2604.10697.
Cancedda, N. (2024). Spectral filters, dark signals, and attention sinks. arXiv:2402.09221.
Chen, M., et al. (2024). PrefixQuant: Static quantization beats dynamic through prefixed outliers in LLMs. arXiv:2410.05265.
Dale, D., Voita, E., Barrault, L., & Costa-jussà, M. R. (2023a). Detecting and mitigating hallucinations in machine translation: Model internal workings alone do well, sentence similarity even better. ACL 2023. arXiv:2212.08597.
Dale, D., Voita, E., et al. (2023b). HalOmi: A manually annotated benchmark for multilingual hallucination and omission detection in machine translation. EMNLP 2023. arXiv:2305.11746.
Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. NeurIPS 2022. arXiv:2208.07339.
Ferrando, J., Gállego, G. I., Alastruey, B., Escolano, C., & Costa-jussà, M. R. (2022). Towards opening the black box of neural machine translation: Source and target interpretations of the transformer. EMNLP 2022. arXiv:2205.11631.
Gu, X., Pang, T., Du, C., Liu, Q., Zhang, F., Du, C., Wang, Y., & Lin, M. (2025). When attention sink emerges in language models: An empirical view. ICLR 2025. arXiv:2410.10781.
Guerreiro, N. M., Alves, D., Waldendorf, J., Haddow, B., Birch, A., Colombo, P., & Martins, A. F. T. (2023). Hallucinations in large multilingual translation models. TACL 2023. arXiv:2303.16104.
Lin, H., et al. (2024). DuQuant: Distributing outliers via dual transformation makes stronger quantized LLMs. NeurIPS 2024. arXiv:2406.01721.
Marchisio, K., Dash, S., Chen, H., Aumiller, D., Üstün, A., Hooker, S., & Ruder, S. (2024). How does quantization affect multilingual LLMs? arXiv:2407.03211.
Marie, B., & Fujita, A. (2025). Uneven post-training quantization degradation across languages in multilingual machine translation. arXiv:2508.20893.
NLLB Team et al. (2022). No language left behind: Scaling human-centered machine translation. arXiv:2207.04672.
Owen, L., Roy Chowdhury, N., Kumar, A., & Güra, F. (2025). A refined analysis of massive activations in LLMs. arXiv:2503.22329.
Raunak, V., Menezes, A., & Junczys-Dowmunt, M. (2021). The curious case of hallucinations in neural machine translation. NAACL 2021. arXiv:2104.06683.
Su, Z., et al. (2025). Unveiling super experts in mixture-of-experts large language models. ICLR 2026. arXiv:2507.23279.
Subramanian, S., Akinfaderin, A., & Sehwag, A. (2026). Super weights in LLMs and the failure of selective training. COLM 2026. arXiv:2607.08733.
Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive activations in large language models. COLM 2024. arXiv:2402.17762.
Williams, M., & Aletras, N. (2024). On the impact of calibration data in post-training quantization and pruning. arXiv:2311.09755.
Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient streaming language models with attention sinks. arXiv:2309.17453.
Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023b). SmoothQuant: Accurate and efficient post-training quantization for large language models. ICML 2023. arXiv:2211.10438.
Yu, M., Wang, D., Shan, Q., Reed, C. J., & Wan, A. (2024). The super weight in large language models. arXiv:2411.07191.
Zhang, et al. (2024). Multilingual Brain Surgeon: Large language models can be compressed leaving no language behind. arXiv:2404.04748.
