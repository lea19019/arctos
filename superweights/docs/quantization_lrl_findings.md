# Why does quantization hurt low-resource languages more? What we found so far

*One document for everything found, understood and learned between 2026-09-05 and 2026-09-08.
Written in plain language on purpose. Numbers come from the cluster repo
`/home/vacl2/quantization_lrl` (commit `72f12a4`); each results folder has a `manifest.json`
with the git sha, config and library versions that produced it.*

**Status:** preliminary. One calibration seed per run except where stated. Two models.
Nothing here is a paper claim yet. It is enough to pick a project direction.

---

## 0. How we got here

The starting point was four papers on the internals of multilingual models: massive
activations [Sun 2024], systematic outliers [An 2025], language-specific sparse dimensions
[Zhong 2025], and the separation of concept from language [Dumas 2025]. The massive-activation
angle was set aside for three reasons:

- The "super weight" paper that named the phenomenon was rejected, and the phenomenon is
  already known under other names (attention sinks, outlier features).
- A prior Claude session's run on seven models (unreviewed, see section 8) suggests the
  massive-activation constant is the same in every language, so it cannot by itself explain a
  gap *between* languages.
- Massive activations are a problem for *activation* quantization (W8A8 and similar, the
  lab's CTranslate2 int8 regime), where a few huge values blow up the scale of a whole
  tensor. The low-resource gap in the literature shows up already under *weight-only* 4-bit
  quantization, which never rounds an activation.

Adrian then chose the question "why does quantization hurt low-resource languages more",
leaning toward guesses 1 and 3 below and setting guess 2 aside. What the four original
papers still offer this direction: Zhong 2025's language-specific dimensions and Dumas
2025's concept/language split are candidate places to look if the residual-norm lead
(section 7) turns out to be about *which* dimensions carry a low-resource token's signal.

## 1. The question

When you round a trained model's weights down to 4 bits (post-training quantization), the
model gets slightly worse. Several papers show it gets *much* worse in low-resource languages
than in English [Marchisio 2024; Marie & Fujita 2025; Chimoto 2026], but nobody has said why. We wanted to find out why.

## 2. The three guesses we started with

From the literature (`literature/notes_lrl_quantization_why.md` has the full survey):

1. **Thin knowledge.** The model is less sure of itself in these languages, and rounding noise
   flips decisions that were already close calls [Proskurina 2024; Lotfi 2026; Tran 2022;
   Hooker 2019; Hooker 2020; Ahia 2021].
2. **English calibration.** GPTQ and AWQ use a small English text sample to decide how to
   round. Non-English gets rounded badly [Chimoto 2026; Zhang 2024; Marie & Fujita 2025].
3. **Tokenizer fragmentation.** Low-resource words get chopped into many pieces, each piece
   carries less evidence, and more pieces means more chances to break [Rust 2021; Ahia 2023;
   Petrov 2023].

Adrian judged guess 2 to be a knob, not a cause: any real answer should be general. We tested
guesses 1 and 3. Guess 2 is untested here, but our plain round-to-nearest runs use no
calibration data at all and show the same gap, which already says guess 2 cannot be the whole
story.

## 3. What we built and ran

**Models.** Llama-3.1-8B [Grattafiori 2024] and EuroLLM-9B-Instruct [Martins 2025], both from
the cluster cache.

**Languages.** 21 languages from FLORES+ [NLLB Team 2022], 1012 sentences each: English, Spanish, French,
Portuguese, Italian, German, Dutch, Czech, Russian, Chinese, Korean, Hindi, Bengali,
Malayalam, Egyptian Arabic, Amharic, Hausa, Igbo, Swahili, Xhosa, Yoruba.

**Quantizers, written from the papers in about 300 lines.**

- RTN: round to nearest, 4 and 3 bits, groups of 128 weights. No calibration data.
- GPTQ [Frantar 2023]: same grid, but rounds one weight column at a time and fixes each rounding error using
  the columns not yet rounded. Uses 128 English calibration windows of 2048 tokens.
- GPTQ with act-order (added on day 2, see section 8).

Both quantizers were checked against published WikiText-2 perplexities [Merity 2017; Shao 2024;
Lin 2024] before any result was read (section 9).

**The measurement.** For every token in every sentence, compare the full model and the
rounded model. The main number is `d_gold`: how much the log-probability of the correct next
token dropped. Also recorded: whether the top prediction flipped, the KL divergence between
the two distributions, the confidence margin (top-1 minus top-2 log-prob, the margin of
Tran 2022 and Proskurina 2024), the gold log-prob, the entropy, the length of the
residual-stream vector at the second-to-last block (the predictor of Chang 2025),
and whether the token is word-initial or a continuation piece.

**Cost.** RTN takes seconds; GPTQ a few minutes. Scoring 21 languages takes 8 to 17 minutes
per run on one 80 GB GPU. The whole preliminary, 10 experiment runs plus 11 validation runs,
fit in one day on a single GPU at a time. This is why the question could be tested at all
before the project was chosen.

**The tests we wrote before running.** Two pre-registered statistics:

- For guess 1: how much of the between-language variance in damage disappears once you
  compare tokens of equal confidence (eta-squared, raw vs. after removing confidence-decile
  means, with a shuffled-margin null). Supported if the drop is over 50%. Dead if under 25%.
- For guess 3: Spearman correlation between tokens-per-word (fertility, as defined by Rust
  2021) and per-language damage.
  Supported if over 0.6, and if continuation pieces take more damage than word-initial ones.

## 4. Main result: the gap is real, large, and tight

Llama-3.1-8B, GPTQ at 4 bits, drop in gold log-prob per token, 95% CI over sentences:

| Language | d_gold | 95% CI | times English |
| --- | --- | --- | --- |
| Hausa | 0.458 | [0.447, 0.472] | 5.8 |
| Igbo | 0.445 | [0.434, 0.457] | 5.6 |
| Xhosa | 0.432 | [0.421, 0.442] | 5.5 |
| Swahili | 0.424 | [0.412, 0.435] | 5.4 |
| Yoruba | 0.375 | [0.366, 0.384] | 4.7 |
| Amharic | 0.38 | | 4.8 |
| Arabic (Egyptian) | 0.28 | | 3.5 |
| Czech | 0.24 | | 3.0 |
| Korean | 0.19 | | 2.4 |
| Dutch | 0.18 | | 2.2 |
| Malayalam, Bengali | 0.16 | | 2.0 |
| Hindi, Chinese, Italian | 0.13 | | 1.6 |
| Portuguese, German, Russian, French | 0.11–0.12 | | 1.4 |
| Spanish | 0.10 | | 1.3 |
| English | 0.079 | [0.071, 0.087] | 1.0 |

This is the gap reported at the benchmark level by Marchisio 2024 and Marie & Fujita 2025,
now seen per token. The same ordering holds under RTN-4, RTN-3, and GPTQ-3. At 3 bits the numbers are about
three times larger (RTN-3: Xhosa 1.371 vs English 0.393).

**Seed check.** Three GPTQ calibration seeds on Llama-3.1-8B give Hausa 5.8 / 5.2 / 5.3 times
English, Igbo 5.6 / 5.4 / 5.0, Xhosa 5.5 / 5.1 / 4.4, Amharic 4.6 / 4.5 / 4.4. The gap does not
depend on which calibration text was drawn.

## 5. Guess 1 (thin knowledge): not met

If the damage were about confidence, as Proskurina 2024 found within English and Tran 2022
derived for under-represented groups, then comparing only tokens where the model is equally
sure should shrink the gap between languages a lot. It does not.

| Run | Drop in language variance after matching on confidence | Verdict line |
| --- | --- | --- |
| Llama GPTQ-4 | 12.4% | under 25% |
| Llama RTN-4 | 15.5% | under 25% |
| Llama RTN-3 | 6.1% | under 25% |
| Llama GPTQ-4 act-order | 11.1% | under 25% |
| Llama GPTQ-3 act-order | 4.5% | under 25% |
| EuroLLM RTN-3 | 7.7% | under 25% |
| EuroLLM GPTQ-3 act-order | 6.9% | under 25% |
| EuroLLM 4-bit runs | too little damage to test | |

Three further facts point the same way:

- **Low-resource languages lose more inside every confidence decile**, and the ratio to
  English grows with confidence. Under Llama GPTQ-4, Hausa loses 4.5 times English in the
  least-confident decile and 17 times in the most-confident decile.
- **The model was not less confident in these languages.** Mean margins are not lower for
  Hausa or Swahili than for English, and Amharic, Malayalam and Bengali have the *highest*
  margins of all (byte-level fragmentation makes continuation pieces easy to predict).
- **The model was not weaker in these languages by the gold log-prob either.** Llama-3.1-8B
  predicts Swahili tokens about as well as English ones before quantization (gold log-prob
  −3.06 vs −3.09) yet loses 5.4 times more under GPTQ-4. Matching on gold log-prob never
  reduces the language gap in any run (the drop is negative every time).

The "what would change my mind" line in the spec was: a low-resource language losing more
than French inside every confidence decile kills the confidence story. That is what happened.

This fits the model-level results of Kumar 2025 and Ouyang 2025, who find that *more* training
makes a model *more* sensitive to quantization, not less; "thin knowledge" was never going to
be a simple story. Catalan-Tatjer 2026 argues their effect is mostly a learning-rate confound,
so the per-language version is still open (section 11).

## 6. Guess 3 (tokenizer fragmentation): not met by the pre-set line

| Run | Spearman(tokens per word, damage) across 21 languages |
| --- | --- |
| Llama GPTQ-4 | 0.52 |
| Llama RTN-4 | 0.29 |
| Llama RTN-3 | 0.57 |
| Llama GPTQ-4 act-order | 0.59 |
| Llama GPTQ-3 act-order | 0.56 |
| EuroLLM GPTQ-4 | 0.35 |
| EuroLLM RTN-4 | 0.34 |
| EuroLLM RTN-3 | 0.74 |
| EuroLLM GPTQ-3 act-order | 0.62 |

Only two of ten runs clear 0.6. Fragmentation is real and well measured [Rust 2021; Ahia 2023;
Petrov 2023] and is correlated with damage, but it does not order the languages: the African Latin-script languages are not the most fragmented (Amharic,
Malayalam and Bengali are), yet they break the most.

**Within a language** the picture is mixed. Continuation pieces take 2 to 3 times the damage
of word-initial pieces in Hindi, Bengali, Malayalam, Arabic, Amharic and Korean; about 1.1 to
1.3 times in Hausa, Igbo, Yoruba and Swahili; and *less* damage than word-initial pieces in
English and the Romance languages (0.4 to 0.7 times). So fragmentation matters for some
scripts, but it is not the general cause.

## 7. What does line up with the damage: the size of the model's internal vector

This was not one of the three guesses. It came out of the secondary measures, which were
included because Chang 2025 found the residual-stream norm predicts which English inputs break
under low-bit quantization.

As the model reads a sentence, it builds a vector for each token (the residual stream). We
recorded the length of that vector at the second-to-last block.

**Between languages** the length of that vector is the best single predictor of damage:

| Run | Spearman(mean residual norm per language, damage) |
| --- | --- |
| Llama GPTQ-4 | −0.71 |
| Llama RTN-3 | −0.76 |
| Llama GPTQ-4 act-order | −0.80 |
| Llama GPTQ-3 act-order | −0.72 |
| EuroLLM RTN-4 | −0.60 |
| EuroLLM GPTQ-4 | −0.66 |
| EuroLLM RTN-3 | −0.81 |

Languages with smaller vectors break more. On Llama the African Latin-script languages run at
norms around 51 to 54, the high-resource languages around 65 to 70.

**Within every language too.** Splitting each language's tokens into quartiles by vector
length, the smallest-norm quartile takes several times the damage of the largest, in all 21
languages, on both models. Llama RTN-4: English 0.117 vs 0.046; Amharic 0.441 vs 0.047; Hausa
0.360 vs 0.171. The per-token rank correlation is weak (−0.02 to −0.29) because single-token
damage is noisy, but the quartile contrast never reverses.

**Our hypothesis (not a result).** Rounding the weights adds noise of roughly the same size to
every token's vector. A small vector drowns in that noise; a large one does not. The final
RMSNorm rescales every vector to the same length before the logits, so the confidence margins
look normal while the signal-to-noise underneath is worse. That would explain why matching on
confidence does nothing.

Two cautions:

- The direction is the opposite of Chang 2025, who found within English that *large*-norm
  tokens break. Both could be true at different scales; we have not reconciled
  them.
- Timkey & van Schijndel 2021 show a few rogue dimensions can dominate the norm. The effect
  must survive standardizing those before it is trusted.

**How to test it causally.** Scale a language's residual stream up before quantization and
see whether its damage falls to English levels. Not yet run.

## 8. Other things we found along the way

**EuroLLM-9B is about ten times more robust at 4 bits than Llama-3.1-8B**, in every language
(English 0.008 vs 0.079 under GPTQ-4). Its gap only becomes measurable at 3 bits, where the
languages it was not trained on (Swahili, Xhosa, Igbo, Yoruba, Hausa) top the list. So the
size of the gap is a property of the model, not only of the language.

**Act-order is what GPTQ needs on Llama-3.** Act-order is the "GPTQ-R" variant of Frantar
2023, later reported by Shao 2024 and Lin 2024 as the standard GPTQ baseline. GPTQ rounds
weight columns one at a time and
compensates each error with the columns still to come. Act-order rounds the columns with the
largest activations first, while there are still many columns left to absorb their error.
On Llama-2-7B it barely matters. On Llama-3.1-8B, GPTQ-3 *without* it is worse than plain
rounding (WikiText-2 12.23 vs RTN 10.85); *with* it, 7.75. With act-order at 4 bits the
absolute damage halves (English 0.079 to 0.039) and the gap is unchanged (Hausa 6.1x, Xhosa
5.1x, Swahili 4.9x, Amharic 4.5x).

**GPTQ is not deterministic run to run on the GPU.** Same config, same seed, same calibration
windows: per-language means differ by up to 0.029 (Xhosa 0.403 vs 0.432), larger than the
sentence-bootstrap CI. RTN is exactly reproducible. So a GPTQ claim needs replicate runs, not
only sentence CIs.

**bf16 logits are too coarse for this measurement.** bf16 logits come in steps of 0.125, so
margins tie in large blocks, and the tie-break silently sorted by language and produced a fake
bin with negative damage. The final projection is now done in float32. The v1 results are
kept in `results/e1_bf16logits/` for comparison.

**The massive-activation angle is parked.** Massive activations [Sun 2024; An 2025] are a few
fixed channels with huge values at the first token and delimiters. A previous Claude session
ran a check on seven models (`superweights/results/const_lang/`, 2026-09-06) suggesting the massive-activation
constant is language-independent. Adrian has not reviewed that run; treat it as an unreviewed
lead. Nothing in this experiment depends on it.

## 9. What we learned about the methods

**Two families.** *Weight-only* quantization rounds the weights and leaves the activations in
16 bits: RTN, GPTQ, AWQ, GGUF k-quants, bitsandbytes NF4. *Weight-and-activation*
quantization rounds both: W8A8, W4A4, the lab's CTranslate2 int8 path. Everything in this
document is weight-only. Massive activations and outlier channels are the central problem
for the second family, not the first.

**Why quantization makes inference faster.** Generating one token means reading every weight
once. For an 8B model that is 16 GB in bf16 and 4 GB at 4 bits. The GPU is waiting on memory,
not arithmetic, so reading a quarter of the bytes is close to a 4x speedup per token. Fused
kernels dequantize the weights on the fly inside the matrix multiply. For W8A8, int8 tensor
cores also do the arithmetic faster. Our runs use fake quantization (below), so they measure
damage only, not speed.


- **RTN** is the baseline every paper reports against [Frantar 2023; Shao 2024]. It needs no
  data and takes seconds.
- **GPTQ** [Frantar 2023] is RTN plus error compensation using a small calibration set. It is measurably
  better at 4 bits and much better at 3 bits, but only with act-order on Llama-3.
- **AWQ** [Lin 2024] scales weight channels before rounding using activation statistics. Different family
  from GPTQ. Not implemented here.
- **Fake quantization** means the weights are rounded and immediately stored back as bf16.
  The model runs at full speed and full memory; only the values are on the 4-bit grid. Fine
  for measuring damage. Real speedups come from fused dequantization kernels and int8 tensor
  cores, which are not needed to answer our question.
- **Validation against published numbers is the sanity check.** Llama-2-7B WikiText-2
  perplexity, targets from Shao 2024 Table 2 and Lin 2024 Table 4: FP16 5.472 (paper 5.47), RTN-4 5.724 (5.72), GPTQ-4 5.717 (5.69 no act-order),
  GPTQ-4 act-order 5.578 (5.63), GPTQ-3 6.204 (6.43), GPTQ-3 act-order 6.175 (6.42). The 3-bit
  numbers are better than published because we calibrate on WikiText-2 train, in-domain.
- **Perplexity hides the language gap.** A model can lose 0.3 perplexity on WikiText-2 and
  five times more per token in Hausa. Marchisio 2024 make the same point against automatic
  metrics: human raters saw damage in Japanese that the metrics missed. Per-token, per-language measurement is what shows it.

## 10. What we learned about running experiments

- **Write the tests first.** Quantizer invariants (error bounded by half a step, GPTQ equals
  RTN under an identity Hessian, GPTQ beats RTN on correlated inputs), scorer sanity (model
  vs itself is zero), and "every config loads". They caught real bugs before GPU time.
- **Validate the method before reading the result.** The Llama-3 GPTQ-3 anomaly would have
  been read as a language finding if the WikiText check had not been run.
- **Write the manifest first, never overwrite it.** A resumed run overwrote a manifest once;
  now later attempts get a timestamped file.
- **Batch by token budget, not by sentence count.** Malayalam under EuroLLM's tokenizer gave
  sequences long enough to run an 80 GB GPU out of memory at batch 16.
- **Capture one layer with a hook** instead of `output_hidden_states` for all 43 blocks.
- **Set `LC_ALL=C` in SLURM scripts.** The compute node's locale sorted config names
  differently from the login node, so array index 6 ran the wrong config.
- **Move ad-hoc logic into script files.** Nested quotes in one-line ssh python commands cost
  more time than writing `seed_spread.py` and `validate_table.py`.
- **Per-language parts with resume.** A crash at language 15 of 21 should not cost the first
  14.
- **Cluster rules that mattered:** `--qos=cs --gpus=1 --cpus-per-task=8 --mem=64G`, no
  partition, `HF_HUB_OFFLINE=1` on compute nodes, poll no faster than every 60 s.

## 11. What is open

Ordered by how much it would change the picture.

1. **Causal test of the residual-norm hypothesis.** Scale a language's residual stream before
   quantization; does its damage drop? This is the experiment that turns a correlation into a
   mechanism.
2. **Rogue-dimension check.** Standardize the few dimensions that dominate the norm and see
   whether the quartile effect survives.
3. **EuroLLM act-order at 4 bits is anomalous** (English 0.060 vs 0.008 without act-order,
   flat language profile). Run the WikiText check on EuroLLM before reading
   `results/e1/eurollm9b_gptq4_actorder`.
4. **Reconcile with Chang 2025**, whose within-English direction is opposite.
5. **Method coverage.** Only RTN and GPTQ were run. AWQ, GGUF, bitsandbytes and any
   activation quantization are untested, and Marie & Fujita 2025's gap was measured under
   AWQ, bitsandbytes and GGUF. A claim about "quantization" needs at least AWQ.
6. **Generation quality (E3).** Everything here is next-token damage. The n=960 COMET
   protocol in `compression/experiments/replication-uneven-ptq/` exists to reuse.
7. **A third model family** and at least three seeds on everything before any claim. Marie &
   Fujita 2025 note that almost no paper in this area reports run-to-run spread.
8. **Training arm** (Adrian's earlier idea): fine-tune on low-resource data, watch the
   residual norm, requantize. Only worth building if item 1 comes out positive. Kumar 2025,
   Ouyang 2025 and Catalan-Tatjer 2026 are the framing for it.

## 12. What this built, skill-wise

Against `skills_to_build.md`, three days of this work exercised:

- Writing two quantizers from the papers and validating them against published numbers.
- A tested package (41 CPU tests, run before every GPU submission) with one YAML per run,
  a provenance manifest per results folder, and SLURM array scripts with resume.
- An evaluation pipeline that scores two models per token over 21 languages within GPU
  memory, after fixing an out-of-memory failure and a numerical-precision bug.
- Reading failures from logs instead of guessing: the locale bug, the bf16 tie-break
  artefact, GPTQ nondeterminism, the Llama-3 act-order requirement.

Not yet exercised: CI, Docker, a public release, multi-GPU training, real (not fake)
quantized inference kernels.

## 13. Papers to read, in this order

1. Chang 2025. Residual norm and quantization breakage within English.
2. Tran 2022. Why under-represented groups sit closer to the decision boundary.
3. Proskurina 2024. Confidence and quantization damage within English.
4. Marchisio 2024. The multilingual gap itself, with human evaluation.
5. Kumar 2025; Ouyang 2025; Catalan-Tatjer 2026. More training tokens make a model more
   sensitive to PTQ, and the learning-rate confound.
6. Timkey & van Schijndel 2021. Rogue dimensions in the residual stream.
7. Frantar 2023. GPTQ itself, including the act-order variant.

The full survey with links is `literature/notes_lrl_quantization_why.md`; the PDFs are in
`superweights/papers/`.

## 14. Where everything lives

| What | Where |
| --- | --- |
| Code, configs, tests, SLURM scripts | `/home/vacl2/quantization_lrl` on the cluster (git, commit `72f12a4`) |
| Spec, satisfied-when, verdict | `README.md` in that repo |
| Dated changelog with every failure | `notes.md` in that repo |
| Per-run tables with CIs | `results/e1/<run>/analysis.md` |
| Published-perplexity checks | `results/validate/<run>/summary.json` |
| Seed spread, within-language norm check | `scripts/seed_spread.py`, `scripts/resid_within.py` |
| Literature survey | `superweights/docs/literature/notes_lrl_quantization_why.md` |
| Runs not to read | `results/e1/llama31_8b_gptq3` (no act-order), `results/e1/eurollm9b_gptq4_actorder` (unvalidated) |

## 15. Bibliography

Links point into `superweights/papers/`. Venues marked *preprint* have not been peer reviewed;
the notes in `literature/notes_lrl_quantization_why.md` say which ones were read in full.

- **Ahia 2021.** Ahia, O., Kreutzer, J., Hooker, S. *The Low-Resource Double Bind: An
  Empirical Study of Pruning for Low-Resource Machine Translation.* Findings of EMNLP 2021.
  [PDF](../papers/Ahia-2021-Low-Resource-Double-Bind-Pruning-MT.pdf)
- **Ahia 2023.** Ahia, O., Kumar, S., Gonen, H., Kasai, J., Mortensen, D., Smith, N.,
  Tsvetkov, Y. *Do All Languages Cost the Same? Tokenization in the Era of Commercial
  Language Models.* EMNLP 2023.
  [PDF](../papers/Ahia-2023-Do-All-Languages-Cost-The-Same-Tokenization.pdf)
- **An 2025.** An, Y., et al. *Systematic Outliers in Large Language Models.* ICLR 2025.
  [PDF](../papers/An-2025-Systematic-Outliers-in-LLMs.pdf)
- **Catalan-Tatjer 2026.** Catalan-Tatjer, et al. Training dynamics of post-training
  quantization robustness along OLMo, SmolLM3 and Apertus trajectories. ICLR 2026.
  [PDF](../papers/TrainingDynamics-2025-PTQ-Robustness.pdf)
- **Chang 2025.** Chang, T.-Y., Zhang, M., Thomason, J., Jia, R. *Why Do Some Inputs Break
  Low-Bit LLM Quantization?* EMNLP 2025.
  [PDF](../papers/Chang-2025-Why-Some-Inputs-Break-Low-Bit-Quantization.pdf)
- **Chimoto 2026.** Chimoto, E., et al. Calibrating beyond English for quantized
  multilingual LLMs. EACL 2026 (long papers).
  [PDF](../papers/Chimoto-2026-Calibrating-Beyond-English-Quantized-Multilingual-LLM.pdf)
- **Frantar 2023.** Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. *GPTQ: Accurate
  Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023.
  [PDF](../papers/Frantar-2022-GPTQ.pdf)
- **Grattafiori 2024.** Grattafiori, A., et al. *The Llama 3 Herd of Models.* arXiv:2407.21783,
  *preprint*. Not in the papers folder.
- **Hooker 2019.** Hooker, S., Courville, A., Clark, G., Dauphin, Y., Frome, A. *What Do
  Compressed Deep Neural Networks Forget?* arXiv:1911.05248, *preprint*.
  [PDF](../papers/Hooker-2019-What-Do-Compressed-DNNs-Forget.pdf)
- **Hooker 2020.** Hooker, S., Moorosi, N., Clark, G., Bengio, S., Denton, E. *Characterising
  Bias in Compressed Models.* arXiv:2010.03058, *preprint*.
  [PDF](../papers/Hooker-2020-Characterising-Bias-Compressed-Models.pdf)
- **Kumar 2025.** Kumar, T., et al. *Scaling Laws for Precision.* ICLR 2025.
  [PDF](../papers/Kumar-2025-Scaling-Laws-For-Precision.pdf)
- **Lin 2024.** Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G.,
  Dang, X., Gan, C., Han, S. *AWQ: Activation-aware Weight Quantization for LLM Compression
  and Acceleration.* MLSys 2024. [PDF](../papers/Lin-2023-AWQ.pdf)
- **Lotfi 2026.** Lotfi, et al. Quantized reasoning models think longer; token-level KL is
  largest where the full model's entropy is high. arXiv:2606.00206, *preprint*.
  [PDF](../papers/Lotfi-2026-Quantized-Reasoning-Models-Think-Longer.pdf)
- **Marchisio 2024.** Marchisio, K., Dash, S., Chen, H., Aumiller, D., Üstün, A., Hooker, S.,
  Ruder, S. *How Does Quantization Affect Multilingual LLMs?* Findings of EMNLP 2024,
  pp. 15928–15947. [PDF](../papers/Marchisio-2024-Quantization-Multilingual-LLMs.pdf)
- **Marie & Fujita 2025.** Marie, B., Fujita, A. Uneven impact of post-training quantization
  across 55 languages in multilingual MT with LLMs. arXiv:2508.20893, *preprint*.
  [PDF](../papers/UnevenPTQ-2025-Multilingual-MT-Quantization.pdf)
- **Martins 2025.** Martins, P. H., et al. *EuroLLM-9B: Technical Report.* *preprint*.
  [PDF](../papers/EuroLLM-2025-9B-Technical-Report.pdf). See also Martins 2024,
  [PDF](../papers/Martins-2024-EuroLLM-Multilingual-Europe.pdf).
- **Merity 2017.** Merity, S., Xiong, C., Bradbury, J., Socher, R. *Pointer Sentinel Mixture
  Models.* ICLR 2017. The WikiText-2 dataset. Not in the papers folder.
- **NLLB Team 2022.** NLLB Team, Costa-jussà, M. R., et al. *No Language Left Behind: Scaling
  Human-Centered Machine Translation.* arXiv:2207.04672. The FLORES-200 benchmark, which
  FLORES+ continues. [PDF](../papers/NLLB-2022-No-Language-Left-Behind.pdf)
- **Ouyang 2025.** Ouyang, X., et al. *Low-Bit Quantization Favors Undertrained LLMs.*
  ACL 2025. [PDF](../papers/Ouyang-2025-Low-Bit-Quantization-Favors-Undertrained-LLMs.pdf)
- **Petrov 2023.** Petrov, A., La Malfa, E., Torr, P., Bibi, A. *Language Model Tokenizers
  Introduce Unfairness Between Languages.* NeurIPS 2023.
  [PDF](../papers/Petrov-2023-Tokenizers-Introduce-Unfairness-Between-Languages.pdf)
- **Proskurina 2024.** Proskurina, I., Brun, L., Metzler, G., Velcin, J. *When Quantization
  Affects Confidence of Large Language Models?* Findings of NAACL 2024.
  [PDF](../papers/Proskurina-2024-When-Quantization-Affects-Confidence-LLMs.pdf)
- **Rust 2021.** Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., Gurevych, I. *How Good is Your
  Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021.
  [PDF](../papers/Rust-2021-How-Good-Is-Your-Tokenizer.pdf)
- **Shao 2024.** Shao, W., et al. *OmniQuant: Omnidirectionally Calibrated Quantization for
  Large Language Models.* ICLR 2024. Source of the Llama-2-7B RTN and GPTQ targets.
  [PDF](../papers/Shao-2024-OmniQuant.pdf)
- **Sun 2024.** Sun, M., Chen, X., Kolter, J. Z., Liu, Z. *Massive Activations in Large
  Language Models.* COLM 2024. [PDF](../papers/Sun-2024-COLM-Massive-Activations-in-LLMs.pdf)
- **Timkey & van Schijndel 2021.** Timkey, W., van Schijndel, M. *All Bark and No Bite: Rogue
  Dimensions in Transformer Language Models Obscure Representational Quality.* EMNLP 2021.
  [PDF](../papers/Timkey-vanSchijndel-2021-All-Bark-No-Bite-Rogue-Dimensions.pdf)
- **Tran 2022.** Tran, C., Fioretto, F., Kim, J.-E., Naidu, R. *Pruning Has a Disparate Impact
  on Model Accuracy.* NeurIPS 2022.
  [PDF](../papers/Tran-2022-Pruning-Disparate-Impact-Accuracy.pdf)
- **Zhong 2025.** Zhong, et al. *Language Lives in Sparse Dimensions.* 2025.
  [PDF](../papers/Language-Lives-in-Sparse-Dimensions-2025.pdf). Venue not checked.
- **Dumas 2025.** Dumas, C., Wendler, C., Veselovsky, V., Monea, G., West, R. *Separating
  Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations
  in Transformers.* ACL 2025 (long papers).
  [PDF](../papers/Dumas-2025-Separating-Tongue-From-Thought.pdf)
- **Zhang 2024.** Zhang, X., et al. *Multilingual Brain Surgeon: Large Language Models Can Be
  Compressed Leaving a Small Set of Languages.* *preprint*.
  [PDF](../papers/Zhang-2024-Multilingual-Brain-Surgeon.pdf)

Author lists and titles for Catalan-Tatjer 2026, Chimoto 2026, Lotfi 2026 and Marie & Fujita
2025 are given from the survey notes rather than from the title pages. Check them against the
PDF before citing in the report.
