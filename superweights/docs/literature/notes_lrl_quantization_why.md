# Why does quantization hurt low-resource languages more? — literature notes

Date: 2026-09-07. Searched: ACL Anthology, arXiv, NeurIPS/ICLR/MLSys proceedings,
Semantic Scholar listings, general web. Question: why does post-training
quantization (PTQ: rounding a trained model's weights to fewer bits, no
retraining) hurt low-resource languages more than high-resource ones, in
multilingual LLMs and translation models?

Two candidate causes under consideration:

- **Cause 1, "thin knowledge / long tail":** the model's competence in a
  low-resource language is fragile. Its per-token confidence margin (the gap
  between the top logit and the runner-up) is small and there is little
  redundancy, so rounding noise flips its decisions first.
- **Cause 3, "tokenizer fragmentation":** low-resource languages are split into
  more subword tokens per word (high *fertility* = tokens per word). Each token
  carries a weaker signal, and this interacts with quantization.

Conventions (same as the other notes in this folder):

- **[read]** — I opened the PDF and read the abstract plus the passage that
  carries the claim I cite. I did **not** read any paper front to back.
- **[unopened]** — venue and finding taken from the arXiv/Anthology abstract
  page only; the PDF was not opened. Confirm before citing.
- Venues were checked by opening the Anthology / proceedings / arXiv page, not
  from memory. OpenReview was behind a bot wall, so ICLR papers were verified
  via `iclr.cc` and the printed first-page header.
- "preprint" = no peer-reviewed venue found on the arXiv page or in the PDF.

---

## 1. Summary — what is known, what is not

1. The *fact* is well replicated: PTQ hurts some languages much more than
   others. Non-Latin scripts and low-data languages lose most (Marchisio 2024,
   Marie & Fujita 2025, Soualhi 2026, Chimoto 2026, plus the repo's own runs).
2. Damage tracks pretraining data size: Marchisio 2024 finds degradation
   "negatively correlated with training data set size"; Gurgurov 2025 finds the
   same for encoder compression. That is consistent with cause 1 but does not
   test it, because data size, script and fertility are collinear.
3. Within one language, quantization damage lands on the inputs the model was
   already unsure about: Proskurina 2024 (low-confidence samples), Lotfi 2026
   (high-entropy positions), Chang 2025 (large residual-stream norms). This is
   the mechanism cause 1 needs — but **none of these papers vary language.**
4. The long-tail story has a theory: Tran 2022 (NeurIPS) derives that groups
   with larger gradient norms and smaller distance to the decision boundary are
   hurt more by pruning, and that under-represented groups have exactly those
   properties. Vision only; never tested with languages as the groups.
5. The naive version of cause 1 is contradicted at model level: Kumar 2025 and
   Ouyang 2025 show that *more* training tokens make a model *more* sensitive
   to PTQ. Catalan-Tatjer 2026 (ICLR) argues that this is mostly a learning-rate
   schedule confound. Nobody has asked the per-language version of the question.
6. Tokenizer fragmentation is well measured (Rust 2021, Ahia 2023, Petrov 2023,
   Arnett 2025) and is known to hurt accuracy and cost — **without**
   quantization. No paper measures fertility against quantization damage.
7. So cause 3 is untested, not refuted. The only text that comes close is
   Marie & Fujita 2025 speculating that calibration data "biases the learned
   scale factors toward … Latin-based scripts".
8. Compression does not always hurt the tail more: Ogueji 2022 and Ahia 2021
   find *moderate* pruning helps low-resource languages, and Wu 2026 finds
   moderate compression preserves head, middle and tail knowledge equally.
   Only aggressive compression is uneven.
9. The clean "which cause" experiment does not exist: no study holds data size
   fixed and varies fertility (or the reverse), and no study reports per-language
   confidence margins before and after quantization.
10. Every multilingual-quantization paper found is a single run per cell; none
    reports a CI, and only Marchisio 2024 has human evaluation.

---

## 2. Papers

Verdict codes: **S** supports, **C** contradicts, **N** neutral / does not test,
**M** supplies a mechanism that would be needed but does not test it on languages.
"Present" = PDF was already in `papers/` before this search.

### 2.1 Per-language effects of quantization and pruning

| File / URL | Venue | Finding (plain language) | Cause 1 | Cause 3 |
|---|---|---|---|---|
| [Marchisio-2024-Quantization-Multilingual-LLMs](../../papers/Marchisio-2024-Quantization-Multilingual-LLMs.pdf) (present) [read] | **Findings of EMNLP 2024**, pp. 15928–15947 (note: `papers.md` lists it as "preprint (Cohere)"; the Anthology page shows it was published) | Quantizing Aya/Command models hurts languages unevenly. Damage is "negatively correlated with training data set size" and non-Latin scripts lose more (−1.9% vs −0.7% at 103B). Humans see damage automatic metrics miss (Japanese −1.7% automatic vs −16% human). No tokenization analysis. | S (correlation only) | N (script confounded with fertility) |
| [UnevenPTQ-2025-Multilingual-MT-Quantization](../../papers/UnevenPTQ-2025-Multilingual-MT-Quantization.pdf) (present) [read] | preprint, arXiv:2508.20893 (Marie & Fujita) | 5 LLMs, 55 languages, AWQ/BnB/GGUF/AutoRound. 4-bit costs <2 COMET on ja/fr but 8–10 on Bengali/Malayalam. Calibration language helps only at 2-bit. Speculates calibration data is biased toward "Latin-based scripts". Single runs. | S | N (speculation, no fertility measured) |
| [Soualhi-2026-Multilingual-Quantization-Tax-Edge-SLMs](../../papers/Soualhi-2026-Multilingual-Quantization-Tax-Edge-SLMs.pdf) [read] | **preprint**, arXiv:2608.09941, "under review at EMNLP 2026"; single independent author | 4-bit NF4 on Gemma 4 / Qwen 3.5, 8 languages. Hindi and Arabic "collapse" (stop producing valid task logits) depending on the model's pretraining mix. Reports that apparent post-quantization gains are noise. | S (weak; n=1) | N |
| [Chimoto-2026-Calibrating-Beyond-English-Quantized-Multilingual-LLM](../../papers/Chimoto-2026-Calibrating-Beyond-English-Quantized-Multilingual-LLM.pdf) (present) [read] | EACL 2026 | Multilingual calibration sets cut perplexity by up to 3.5 for Llama-3.1-8B / Qwen2.5-7B. GPTQ is more sensitive to calibration language than AWQ. Argues rare tokens and activation outliers in the calibration set matter. | N | N (calibration-side, not fertility) |
| [Borgersen-2025-English-K-Quantization-Multilingual-Performance](../../papers/Borgersen-2025-English-K-Quantization-Multilingual-Performance.pdf) [read] | **preprint**, arXiv:2503.03592 | llama.cpp K-quant of Llama-3.3-70B with English vs Norwegian vs Malayalam importance matrices: no significant difference on MixEval. Suggests calibration language is not the driver for GGUF at 70B. | N | C (weak: if fertility drove damage via calibration, Malayalam imatrix should have helped) |
| [Hossain-2026-Quantization-Effects-Bangla-NLU](../../papers/Hossain-2026-Quantization-Effects-Bangla-NLU.pdf) [unopened; abstract only] | **preprint**, arXiv:2608.24615 | Bangla NLU under GGUF vs GPTQ: GPT-OSS GGUF loses up to 57%, Qwen/Llama GPTQ <1.5%. "Architecture and method matter more than bit width." One language, so cannot compare across languages. | N | N |
| [Zhang-2024-Multilingual-Brain-Surgeon](../../papers/Zhang-2024-Multilingual-Brain-Surgeon.pdf) (present) [read] | preprint | Calibrate pruning/quantization on a language mix. Proposes a Hessian explanation: languages whose activations share directions with the calibration language survive; low-resource languages with different activation geometry break. | M (representation-geometry version of cause 1) | N |
| [Gurgurov-2025-Multilingual-Encoder-Compression-Low-Resource](../../papers/Gurgurov-2025-Multilingual-Encoder-Compression-Low-Resource.pdf) [read] | AACL 2025 Student Research Workshop | Distill/prune/truncate XLM-R and mBERT for Slovak, Swahili, Maltese. At 92% compression the loss is 2.9% (Slovak, most data), 5.2% (Swahili), more for Maltese (least data). Degradation "correlates with the amount of language-specific data in the teacher". | S (correlation) | N |
| [Ogueji-2022-Intriguing-Properties-Compression-Multilingual](../../papers/Ogueji-2022-Intriguing-Properties-Compression-Multilingual.pdf) (present) [read] | EMNLP 2022 | Pruning mBERT across 40 languages: 50–70% sparsity *helps* most languages, especially low-data ones; 70–98% sparsity hurts low-data languages most. | S only at high sparsity; C at moderate | N |
| [Ahia-2021-Low-Resource-Double-Bind-Pruning-MT](../../papers/Ahia-2021-Low-Resource-Double-Bind-Pruning-MT.pdf) (present) [read] | Findings of EMNLP 2021 | Pruning en→yo/ha/ig/de NMT: sparsity "preserves performance on frequent sentences but has a disparate impact on infrequent ones", yet improves out-of-distribution robustness. Frequency-of-sentence, not language, is the unit. | S (within-language long tail) | N |
| [Diddee-2022-Too-Brittle-To-Touch-Quantization-Distillation-LowResource-MT](../../papers/Diddee-2022-Too-Brittle-To-Touch-Quantization-Distillation-LowResource-MT.pdf) (present) [read] | WMT 2022 | Small Indic NMT models: distillation is brittle (depends on teacher confidence, data, hyperparameters); PTQ is more stable. Notes teacher confidence as a prior. | N (mentions confidence for distillation, not PTQ) | N |
| [Mohammadshahi-2022-Compressed-Multilingual-MT-Forget](../../papers/Mohammadshahi-2022-Compressed-Multilingual-MT-Forget.pdf) (present) [read] | Findings of EMNLP 2022 | Compressing M2M-100 hurts under-represented pairs and raises off-target rate; measures both directions on one model. | S | N |
| [Ramesh-2023-Model-Compression-Fairness-Language-Models](../../papers/Ramesh-2023-Model-Compression-Fairness-Language-Models.pdf) [read] | ACL 2023 | Pruned/distilled/quantized classifiers on bias benchmarks, mono- and multilingual. Compression can worsen fairness; the effect depends on base model and method. Does not compare languages against each other by resource level. | N | N |

### 2.2 The long-tail / "compression forgets rare things" line

| File / URL | Venue | Finding (plain language) | Cause 1 | Cause 3 |
|---|---|---|---|---|
| [Hooker-2019-What-Do-Compressed-DNNs-Forget](../../papers/Hooker-2019-What-Do-Compressed-DNNs-Forget.pdf) [read] | **preprint**, arXiv:1911.05248 (no venue printed) | Pruned/quantized image models keep the same top-line accuracy but fail on a small set of "Pruning Identified Exemplars" (PIEs) — atypical, low-quality or long-tail images. Compression "impairs the model's ability to predict accurately on the long-tail". | S (origin of the idea) | N |
| [Hooker-2020-Characterising-Bias-Compressed-Models](../../papers/Hooker-2020-Characterising-Bias-Compressed-Models.pdf) [read] | **preprint**, arXiv:2010.03058 | Same method on CelebA: compression amplifies existing bias; errors concentrate on under-represented attributes (CIEs). | S | N |
| [Tran-2022-Pruning-Disparate-Impact-Accuracy](../../papers/Tran-2022-Pruning-Disparate-Impact-Accuracy.pdf) [read] | NeurIPS 2022 | Theory + experiments: under-represented groups have larger gradient norms and sit closer to the decision boundary, so pruning hurts them more (Theorem 1). Distance to decision boundary is the classifier version of a logit margin. Vision datasets only. | M (the margin theory cause 1 needs; untested on languages) | N |
| [Tropeano-2025-As-Easy-As-PIE-Pruning-Disagree](../../papers/Tropeano-2025-As-Easy-As-PIE-Pruning-Disagree.pdf) [read] | Findings of NAACL 2025 | First PIE study on text (BERT/BiLSTM, classification). PIEs are longer and "more semantically complex", and are the examples that matter most for generalisation. English only. | M (a length/complexity link that could be read as fertility) | weak S by analogy: longer inputs are hurt more |
| [Liebenwein-2021-Lost-In-Pruning-Beyond-Test-Accuracy](../../papers/Liebenwein-2021-Lost-In-Pruning-Beyond-Test-Accuracy.pdf) [read] | MLSys 2021 | Pruned nets match test accuracy but the safe prune ratio varies by task; out-of-distribution and noise robustness degrade first. | N (background) | N |
| [Jaiswal-2024-Compressing-LLMs-Truth-Rarely-Pure-LLM-KICK](../../papers/Jaiswal-2024-Compressing-LLMs-Truth-Rarely-Pure-LLM-KICK.pdf) [read] | ICLR 2024 | LLM-KICK benchmark: pruning fails on knowledge-intensive tasks even at 25–30% sparsity; quantization holds up better; in-context retrieval survives 50% sparsity. English. | S (knowledge is what goes first) | N |
| [Jin-2024-Cost-Of-Down-Scaling-Fact-Recall-Deteriorates](../../papers/Jin-2024-Cost-Of-Down-Scaling-Fact-Recall-Deteriorates.pdf) [read] | ICLR 2024 | Removing >30% of parameters (pruning or dense down-scaling) kills fact recall; removing 60–70% leaves in-context processing intact. Stored knowledge is the fragile part. | S | N |
| [Wang-2026-Through-Compressed-Lens-Quantization-Factual-Recall](../../papers/Wang-2026-Through-Compressed-Lens-Quantization-Factual-Recall.pdf) [unopened; abstract only] | TrustNLP workshop @ ACL 2026 | Quantization reduces factual recall, more in smaller models; the effect is inconsistent across implementations (BnB preserves most). | S (weak) | N |
| [Wu-2026-Asymmetric-Harms-LLM-Compression](../../papers/Wu-2026-Asymmetric-Harms-LLM-Compression.pdf) [read] | **preprint**, arXiv:2608.19670 | 3 LLMs, 11 compression methods, head/middle/tail entities. Moderate compression preserves all three at similar rates; aggressive compression "non-uniformly redistributes retention" rather than simply cutting the tail. Compressed models stay confident when wrong. | C for moderate compression; mixed for aggressive | N |
| [Goncalves-2023-Model-Compression-Social-Bias-LLMs](../../papers/Goncalves-2023-Model-Compression-Social-Bias-LLMs.pdf) [read] | EMNLP 2023 | Controlled study: longer pretraining and larger models raise social bias; quantization acts as a regulariser. Compression is not uniformly harmful. | N (compression can help) | N |

### 2.3 Quantization error vs confidence, margin, entropy

| File / URL | Venue | Finding (plain language) | Cause 1 | Cause 3 |
|---|---|---|---|---|
| [Proskurina-2024-When-Quantization-Affects-Confidence-LLMs](../../papers/Proskurina-2024-When-Quantization-Affects-Confidence-LLMs.pdf) [read] | Findings of NAACL 2024 | GPTQ 4-bit lowers confidence on true labels and raises calibration error; "quantization disproportionately affects samples where the full model exhibited low confidence levels in the first place." English benchmarks; BLOOM included but languages not compared. | M — the within-language mechanism, never crossed with language | N |
| [Chang-2025-Why-Some-Inputs-Break-Low-Bit-Quantization](../../papers/Chang-2025-Why-Some-Inputs-Break-Low-Bit-Quantization.pdf) [read] | EMNLP 2025 | 50 pairs of 3–4-bit methods agree on *which* inputs break (ρ=0.82). The full-precision model's residual-stream norm predicts the error; hard inputs depend on precise late-layer activations and MLP gate outputs. FineWeb (English). | M — a per-input predictor that could be measured per language | N |
| [Lotfi-2026-Quantized-Reasoning-Models-Think-Longer](../../papers/Lotfi-2026-Quantized-Reasoning-Models-Think-Longer.pdf) [read] | **preprint**, arXiv:2606.00206 | Token-level KL between quantized and full-precision models is largest at positions where the full model's next-token entropy is high; "quantization most affects positions where the model is already uncertain." Reasoning tasks, English. | M — direct evidence that noise flips decisions first where the margin is small | N |

### 2.4 Data size vs PTQ robustness

| File / URL | Venue | Finding (plain language) | Cause 1 | Cause 3 |
|---|---|---|---|---|
| [Kumar-2025-Scaling-Laws-For-Precision](../../papers/Kumar-2025-Scaling-Laws-For-Precision.pdf) [read] | ICLR 2025 | "Degradation due to post-train quantization increases with tokens seen during pretraining, so that eventually additional pretraining data can be harmful." Models up to D/N ≈ 1000. Model-level; no per-language or per-domain result. | C for the naive "more data = more robust" version of cause 1 at model level | N |
| [Ouyang-2025-Low-Bit-Quantization-Favors-Undertrained-LLMs](../../papers/Ouyang-2025-Low-Bit-Quantization-Favors-Undertrained-LLMs.pdf) [read] | ACL 2025 | 1500+ checkpoints: quantization-induced degradation (loss gap after quantization) grows with training tokens and shrinks with model size; proposes it as a measure of "training level". No per-language result. | C (same as Kumar) | N |
| [TrainingDynamics-2025-PTQ-Robustness](../../papers/TrainingDynamics-2025-PTQ-Robustness.pdf) (present) [read] | ICLR 2026 (Catalan-Tatjer et al.) | Along OLMo/SmolLM3/Apertus trajectories the quantization error is flat through the constant-LR phase and spikes during LR decay; says Kumar/Ouyang's data-size effect is "mostly confounded by the learning rate" (under WSD it disappears at 160M). | Reopens the question; N on per-language | N |
| [Williams-2024-Calibration-Data-Pruning-Quantization](../../papers/Williams-2024-Calibration-Data-Pruning-Quantization.pdf) (present) [read] | NAACL 2024 | Calibration-set choice changes downstream results substantially; the effect depends on method. English. | N | N |

### 2.5 Tokenizer fragmentation and low-resource languages

| File / URL | Venue | Finding (plain language) | Cause 1 | Cause 3 |
|---|---|---|---|---|
| [Rust-2021-How-Good-Is-Your-Tokenizer](../../papers/Rust-2021-How-Good-Is-Your-Tokenizer.pdf) [read] | ACL 2021 | mBERT over-segments AR, FI, KO, RU, TR (fertility much higher than monolingual tokenizers; lowest for EN). A language-specific tokenizer matters as much as pretraining data size for downstream accuracy. Defines fertility and "proportion of continued words". | N | S for "fragmentation hurts", N for the quantization link |
| [Ahia-2023-Do-All-Languages-Cost-The-Same-Tokenization](../../papers/Ahia-2023-Do-All-Languages-Cost-The-Same-Tokenization.pdf) [read] | EMNLP 2023 | 22 languages on OpenAI APIs: same content needs many more tokens in some scripts; the disparity is "rooted in the language properties or the ways they are represented in Unicode", not only data imbalance; more fragmented languages get less in-context-learning utility. | N | S for "fragmentation hurts"; N on quantization |
| [Petrov-2023-Tokenizers-Introduce-Unfairness-Between-Languages](../../papers/Petrov-2023-Tokenizers-Introduce-Unfairness-Between-Languages.pdf) [read] | NeurIPS 2023 | Same text is up to 15× longer in tokens (Shan vs English); byte-level tokenizers still 4× off. Defines "token premium". Cost, latency and context effects; no accuracy-under-noise result. | N | S for the premise; N on quantization |
| [Limisiewicz-2023-Tokenization-Impacts-Multilingual-LM-Vocabulary-Allocation](../../papers/Limisiewicz-2023-Tokenization-Impacts-Multilingual-LM-Vocabulary-Allocation.pdf) [unopened; abstract only] | Findings of ACL 2023 | New measures of vocabulary allocation and overlap; overlap helps sentence-level tasks and hurts token-level ones (POS, parsing). | N | N (shows fertility is not the only tokenizer variable) |
| [Arnett-2025-Explaining-Mitigating-Crosslingual-Tokenizer-Inequities](../../papers/Arnett-2025-Explaining-Mitigating-Crosslingual-Tokenizer-Inequities.pdf) [unopened; abstract only] | NeurIPS 2025 | ~7000 monolingual tokenizers over 97 languages: token premiums are driven by vocabulary size and pre-tokenization, not train/test similarity; superword tokenizers reduce them. | N | N (shows fertility can be manipulated — useful for a controlled test) |

### 2.6 Seen only as abstracts, not downloaded

- Xu & Hu 2022, "Can Model Compression Improve NLP Fairness", arXiv:2201.08542,
  preprint [unopened] — distillation/pruning of GPT-2 *reduces* toxicity and bias
  (compression as regulariser). Counter-weight to the long-tail story.
- Sandler, Üstün, Romanelli, Hooker, Fioretto 2025, "The Disparate Impacts of
  Speculative Decoding", arXiv:2510.02128, preprint [unopened] — speed-up from a
  draft model shrinks on "under-fit, and often underrepresented tasks". Same
  shape as cause 1 for a different efficiency method.
- Uppadhyay et al. 2026, "DEPART", arXiv:2605.28163, preprint [unopened] —
  language features (script, family, similarity to English) explain 79–92% of
  cross-language performance variance in unquantized models. A baseline any
  quantization-gap study would need to subtract.
- Cacioli 2026, arXiv:2604.08976, preprint [unopened] — Q5_K_M vs f16 changes
  domain-level confidence profiles but not AUROC. Single model.
- Rababah et al. 2026, "The Illusion of Equivalency", arXiv:2607.08734, preprint
  [unopened] — accuracy and perplexity match but per-example decisions shift;
  Q/K projections more fragile than V/O. English.
- Siniaev et al. 2026, "Compressed code", arXiv:2601.02563, preprint [unopened]
  — quantization/distillation change programming-token probabilities; a
  token-level lens, not natural language.

---

## 3. What nobody has measured

Each item is stated so that it could be run.

1. **Per-token margin by language, before and after quantization.** No paper
   reports the top-1/top-2 logit gap (or entropy) per language on the same
   model and relates it to that language's quantization loss. Proskurina 2024
   and Lotfi 2026 show the within-language effect; Marchisio 2024 and Marie &
   Fujita 2025 show the between-language effect. The join is missing.
2. **Fertility vs quantization damage on the same model.** Not one of the 35
   papers plots tokens-per-word (or token premium) against ΔCOMET / Δaccuracy
   under quantization. Marie & Fujita 2025 only speculate ("high-entropy Indic
   scripts", calibration "biased toward Latin-based scripts").
3. **Separating data size, script and fertility.** In every study the three
   move together (Bengali/Malayalam/Zulu are low-data, non-Latin or high-fertility
   all at once). Designs that would separate them and do not exist: same
   language under two tokenizers (Arnett 2025 gives the tool); transliterated
   input; language pairs matched on data size but differing in fertility.
4. **Per-language version of Kumar/Ouyang.** Their result is "more tokens seen
   → more PTQ damage" for a whole model. Within a multilingual model the
   high-resource language has seen the most tokens, so the naive reading
   predicts *it* should degrade most — the opposite of what is observed.
   Nobody has checked which reading holds per language, and Catalan-Tatjer 2026
   says the model-level effect is largely a schedule artefact anyway.
5. **Tran 2022's theory with languages as groups.** The gradient-norm /
   distance-to-boundary derivation was only checked on vision datasets and
   pruning. Its language-model, quantization analogue would be: do
   low-resource-language tokens have larger gradient norms and smaller margins
   at convergence?
6. **Chang 2025's residual-norm predictor across languages.** Residual-stream
   magnitude predicts which inputs break at 3–4 bits. Whether low-resource
   inputs have larger (or smaller) residual norms — and how this relates to the
   massive-activation / super-weight machinery this repo studies — is unmeasured.
7. **Long-tail *within* a low-resource language.** Ahia 2021 showed pruning hurts
   infrequent sentences; nobody has repeated this for PTQ or asked whether the
   per-language gap is really a per-frequency gap in disguise.
8. **Confidence intervals.** All multilingual-quantization results found are
   single runs (Marie & Fujita say so; Soualhi attributes gains to noise;
   Borgersen's null has no power analysis). The repo's own n≈960 protocol is
   the only one with a CI.
9. **NLLB-class translation models under GPTQ/AWQ per language.** Diddee 2022
   is small Indic models; Mohammadshahi 2022 is M2M-100 pruning/distillation;
   the peer-reviewed LLM papers are all decoder-only. The repo's replication is
   currently the only per-language PTQ result on a dedicated NMT model.
10. **Does calibration language interact with fertility?** Chimoto 2026 (helps),
    Marie & Fujita 2025 (helps only at 2-bit) and Borgersen 2025 (no effect)
    disagree; none asks whether the benefit is larger for high-fertility
    languages, which is the prediction cause 3 would make.

---

## 4. Downloads

Downloaded and verified as PDFs (25): all files linked above that are not marked
"present". Naming follows `FirstAuthor-Year-Short-Title-Words.pdf`.

Failed: none.

Already present, skipped (10): Marchisio 2024, UnevenPTQ / Marie & Fujita 2025,
Chimoto 2026, Zhang 2024 (MBS), Ahia 2021, Diddee 2022, Ogueji 2022,
Mohammadshahi 2022, TrainingDynamics / Catalan-Tatjer 2026, Williams 2024.

Correction for `papers.md`: Marchisio 2024 is listed there as "preprint
(Cohere)"; the ACL Anthology page `2024.findings-emnlp.935` shows Findings of
EMNLP 2024, pp. 15928–15947. Not edited here (the instruction was to touch only
the new section).

---

## 5. Back-annotation, 2026-09-08: what the E1 preliminary answered

Results and numbers are in `../quantization_lrl_findings.md` (cluster repo
`/home/vacl2/quantization_lrl`, commit `72f12a4`). Preliminary: two models
(Llama-3.1-8B, EuroLLM-9B-Instruct), 21 FLORES+ languages, RTN and GPTQ at 4
and 3 bits, one calibration seed except where stated.

- **Gap 1 (per-token margin by language) is now measured.** Cause 1 is NOT
  MET: matching tokens on confidence margin removes 4.5–15.5% of the
  between-language variance in damage (pre-set line 25%), low-resource
  languages lose more inside every margin decile, and their mean margins are
  not lower. Matching on gold log-prob or entropy never helps.
- **Gap 2 (fertility vs damage on the same model) is now measured.** Cause 3
  is NOT MET by the pre-set line: Spearman(tokens per word, damage) is
  0.29–0.74 across ten runs, only two above 0.6. Continuation pieces take
  2–3x the damage of word-initial pieces in Indic, Arabic, Amharic and Korean,
  but *less* in English and Romance languages.
- **Gap 6 (Chang 2025's residual-norm predictor across languages) is now
  measured**, and the direction is the reverse of Chang's within-English
  result: *smaller* residual norms break more, between languages (Spearman
  −0.6 to −0.8) and within every language (smallest-norm quartile takes 2–10x
  the damage of the largest). Best predictor found. Untested causally.
- **Gap 4 (per-language Kumar/Ouyang).** The naive reading (most tokens seen
  degrades most) is the opposite of what E1 shows: the languages with the
  least data degrade most. Not a test of their claim, which is model-level.
- **Gap 8 (confidence intervals).** E1 reports sentence-bootstrap CIs and a
  three-seed spread; GPTQ's run-to-run nondeterminism exceeds the sentence CI.
- Summary item 3 above ("large residual-stream norms" break, per Chang 2025)
  should be read with the reversal noted here.
- Untouched: gaps 3, 5, 7, 9, 10. Method coverage so far is RTN and GPTQ only;
  AWQ, GGUF and bitsandbytes, which Marie & Fujita 2025 used, are untested.
