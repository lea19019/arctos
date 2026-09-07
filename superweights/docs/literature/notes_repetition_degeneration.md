# Notes — repetition and function-word collapse after super-weight ablation

Compiled 2026-09-05. Everything below marked "read" was extracted from the PDF in
this folder (or the named anchor elsewhere) with `pdftotext`; section and figure
references point at what was actually read. Anything I did not open is marked
**[unopened]**. Where I am reasoning past the papers rather than reporting them, the
sentence starts with *Inference:*.

**Scope note.** For all 26 papers I read the abstract, introduction and conclusion.
For the thirteen marked ★ below I additionally read the mechanism/analysis sections in
full. For the remainder I read the mechanism section only where one exists and is
short (Xiao §3, Ferrando's ALTI+ definition, Gromov's threshold discussion).

---

## §1 Paper by paper

### 1.1 Neural text degeneration in healthy models

**Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration* (ICLR 2020).** ★
Beam search on GPT-2 Large produces literal loops (Fig. 1: "Universidad Nacional
Autónoma de México/…" repeated); pure sampling produces incoherence. The diagnosis is
distributional: the model's tail is unreliable, and maximization is the wrong
objective for open-ended generation. Nucleus sampling truncates to the top-p mass.
**Causal only for the fix** (changing the decoding rule changes the output); the
account of *why* the model prefers loops is observational. Note the paper explicitly
records "cases of a positive feedback loop of repetitions" — later formalized by Xu
et al. 2022.

**Welleck et al. 2019, *Neural Text Generation with Unlikelihood Training* (ICLR 2020).**
Claims the likelihood objective itself is the primary culprit: it "assigns too much
probability to sequences containing repeats **and frequent words**", and generated
text is "dull, with high frequency tokens used too often and interesting content words
used too rarely" (§1). Causal via a training-objective intervention (unlikelihood
loss reduces repetition at equal perplexity). *This is the healthy-model shape of
exactly what Yu et al. Fig. 5 reports for the ablated model* — frequent up, content
down — which matters, because it means the shift is not unique to ablation damage.

**Fu et al. 2021, *A Theoretical Analysis of the Repetition Problem in Text Generation*
(AAAI 2021).** ★ Defines Average Repetition Probability on a Markov generation model
and derives upper bounds (§2). The headline mechanism is the **high inflow problem**:
too many distinct words predict the same next word with high probability, so
trajectories funnel back into that word and cycle. Mitigation is a "rebalanced
encoding" that merges high-inflow pairs into new vocabulary items; tested on NMT and
LM. Theory + empirical validation; the causal claim rests on the encoding
intervention. *Inference:* high-inflow words are by construction high-frequency
function words and punctuation, so this theory predicts the *identity* of the
repeated token, not just that repetition happens.

**Xu et al. 2022, *Learning to Break the Loop* (NeurIPS 2022).** ★ Three controlled
findings (§1, Fig. 2): (i) LMs prefer to repeat the previous *sentence*; (ii)
**self-reinforcement** — P(repeating) rises almost monotonically with the number of
prior repetitions in context and then saturates at a ceiling; (iii) sentences with
higher initial probability self-reinforce faster. Critically, Fig. 2 right shows this
also holds for a *random* token string ("fría backed rounds Manganiello …"), so
self-reinforcement is not semantic. Fix: DITTO, training on pseudo-repetitive data.
The self-reinforcement measurement is a clean controlled experiment; the fix is
causal.

**Li et al. 2023, *Repetition In Repetition Out* (NeurIPS 2023).** Argues degeneration
is downstream of **repetitions present in the training data**: dropping attention to
repetitive words during training substantially reduces degeneration, and the
high-inflow, likelihood-objective and self-reinforcement accounts all reduce to
"penalize training-data repetition." Causal via a training-data/attention
intervention. *Not applicable to an inference-time weight ablation* except as
background on why the loop is available at all.

### 1.2 Mechanistic accounts of repetition

**Hiraoka & Inui 2024, *Repetition Neurons* (NAACL 2025 short).** ★ Identifies FFN
neurons by Δn = (mean activation in the 30 tokens after repetition onset) − (mean in
the 30 before), Eq. 3, over 1,000 self-generated repetitive texts per model
(Gemma-2B, Pythia-2.8B, Llama-3.2-3B, LLM-jp-3-1.8B). Fig. 1: the top neurons ramp up
*progressively* as repetition continues. Fig. 3: the last layer holds the most
repetition neurons, with a secondary peak in intermediate layers — read by the authors
as detection (mid) vs. execution (top). Deactivating them reduces repetition and
activating them induces it (Figs. 4–5, Table 1) — **causal**. §5 compares repetition
neurons against induction heads and "self-finding heads" (Figs. 6–7) on the same
texts. *Note the direction of the evidence:* these neurons respond to repetition
already in progress, so they are at least as much an effect as a cause.

**Yao/Yang et al. 2025, *Understanding the Repeat Curse … from a Feature Perspective*
(Findings of ACL 2025).** ★ Two-stage pipeline ("Duplicatus Charm"): logit-lens layer
localization, then SAE feature localization, on GPT2-small (GPT-sm-res-jb),
Gemma-2-2B (GemmaScope-res-16k) and Llama-3.1-8B (LlamaScope-res-32k). Repetition
features sit in the **intermediate and final layers of all three models**; stimulating
them induces repetition and deactivating them mitigates it. Causal via SAE feature
steering; the layer-localization step is correlational.

**Wang et al. 2025, *Induction Head Toxicity Mechanistically Explains Repetition Curse*
(arXiv:2505.13514, unrefereed).** ★ Prompts models to repeat a token N times; models
overshoot and "lose control", with a burst effect as N grows (Fig. 1 left). §2.2
reports **vocabulary sensitivity**: common/frequent tokens make models fall into the
loop prematurely, rare tokens make them halt early (Fig. 1 right). §3 defines induction
causal importance by activation patching on a randomized copy task (Def. 3.1) and
argues induction heads dominate the output logits during repetition, crowding out
other heads. The patching is a causal method, but the paper is thin, unrefereed, and
the "toxicity" construct is defined rather than measured against an alternative.
**Treat as a lead.**

**Yona, Shumailov, Hayes, Barbero & Gandelsman 2025, *Interpreting the Repeated Token
Phenomenon* (arXiv:2503.08908, Google DeepMind, unrefereed).** ★ **The most directly
transferable paper in this folder.** Two-stage mechanism for attention sinks: (1) the
*first attention layer* marks the first token, mapping it to a linearly separable
subspace (Fig. 4); (2) a **sparse set of MLP "sink neurons"**, selected as
TopK_j(‖MLP_j(BoS)‖), writes a huge-magnitude contribution into that token's hidden
state, and the resulting high norm attracts the sink attention (§4.2). Zero-ablating
those neurons removes the high norm for both BoS and repeated tokens (Fig. 3) —
**causal**. §4.3: because RoPE is an isometry, the first attention layer cannot
distinguish the first token from a long run of identical tokens, so repeated tokens
get marked too, acquire BoS-like norms (Fig. 2) and BoS-like attention (Fig. 1), and
the model diverges. §5 adds a "cluster attack" (non-repeating sequences that trigger
the same failure) and a targeted patch.

### 1.3 Frequency, the unigram prior, and what a dead residual decodes to

**Stolfo, Wu et al. 2024, *Confidence Regulation Neurons in Language Models*
(NeurIPS 2024).** ★ Two component families in the final layers. **Entropy neurons**
have unusually high weight norm but low composition with the unembedding: they write
into an **effective null space of the unembedding** (singular values drop to ~0 around
index 755 in GPT-2), which changes the residual-stream norm and therefore the **final
LayerNorm scale**, scaling all logits and modulating entropy with minimal effect on
the argmax. **Token frequency neurons** (new in this paper) boost or suppress each
token's logit *proportionally to its log frequency*, moving the output distribution
toward or away from the unigram distribution, "which the model defaults to in settings
of high uncertainty" (§1). §6 is a case study showing induction heads use entropy
neurons to hedge on repeated subsequences. Causal via mean ablation. **This paper
supplies the machinery for both H1 and H4 below**, and warns that direct logit
attribution would have missed the null-space channel entirely.

**Meister et al. 2023, *A Natural Bias for Language Generation Models* (ACL 2023
short).** Fig. 1: within a few hundred updates, a generation model's per-token output
distribution matches the *training-corpus unigram distribution* for all contexts, long
before it learns syntax or semantics. Setting the final linear layer's bias to
log-unigram bypasses that stage, improves NMT learning efficiency, and "disentangles
strong frequency effects." Causal via initialization. **Establishes what "the default
distribution" is** for a model with no usable contextual signal.

**Voita, Ferrando & Nalmpantis 2023, *Neurons in LLMs: Dead, N-gram, Positional*
(Findings of ACL 2024).** OPT 125m–66b. Early layers are extremely sparse (>70% dead
neurons in some layers of the 66b). Many alive early neurons are token/n-gram
detectors whose residual-stream updates both promote next-token candidates **and
explicitly remove information about the triggering token** — the first documented
information-removal mechanism in the residual stream. Also positional neurons.
Descriptive/statistical, not causal. Relevant as the null model for "what an early-layer
`down_proj` neuron normally does."

**Cancedda 2024, *Spectral Filters, Dark Signals, and Attention Sinks* (ACL 2024).** ★
Partitions the singular vectors of the embedding/unembedding matrices into bands and
projects residual-stream contents onto them ("logit spectroscopy"). Finds that the
**"dark" band** — the bottom singular values, invisible to the logit lens — carries the
signals that implement attention sinking; NLL stays low while suppressing large parts
of the spectrum *as long as the dark signals are preserved*; and tokens that receive
much attention have large projections onto the dark band. Causal via spectral
filtering interventions. **Direct warning for H1:** a plain logit lens on the ablated
residual can be blind to the component that actually broke.

**Puccetti et al. 2022, *Outlier Dimensions … Driven by Frequency* (Findings of ACL
2022, in ``).** ★ Zeroing 48 LayerNorm outlier
parameters costs BERT-base ~30% on MNLI. §4.1 / Fig. 3 is the key result for this
project: with the outliers disabled, the MLM **"consistently predicts more tokens that
were highly frequent in the training data, and fewer tokens that were rare"**, and
shifts toward nouns, punctuation, symbols and adpositions. Outlier magnitudes correlate
with the pre-training frequency of the encoded token, and the outliers drive the
"vertical" attention pattern onto special tokens. Causal (ablation) + correlational
(frequency correlation). **This is the closest existing precedent for the super-weight
stopword result, and unlike Yu et al. Fig. 5 it uses an actual frequency baseline.**

**Belrose et al. 2023, *Tuned Lens* (arXiv:2303.08112, unrefereed but widely used).**
Trains an affine probe per block so every hidden state decodes to a vocabulary
distribution; shown to be more predictive, reliable and less biased than the raw logit
lens, with causal basis-alignment checks. The trajectory of latent predictions detects
anomalous inputs. **The tool for H1**, and the reason not to report bare logit-lens
curves.

### 1.4 Sinks, massive activations, and what they do for the output distribution

**Sun et al. 2024, *Massive Activations in LLMs* (anchor, ``).**
Read §3–4. Setting massive activations to **zero** at the layer where they first appear
(layer 2 in Llama-2-7B, layer 4 in 13B) explodes perplexity, while zeroing an equal
number of median-magnitude activations costs nothing (Table 3). Setting them to their
**empirical mean** is harmless — "their values are constants and input agnostic, i.e.
functioning similarly to bias terms." §4.1: after they appear, attention concentrates
on their tokens; §4.2 (Eq. 2, Fig. 8) decomposes the attention output and shows the
value updates from those tokens are **near-identical across all query positions**, i.e.
a genuine additive bias. §4.3: training GPT-2 with *explicit* attention bias terms makes
massive activations never form. Strong causal evidence.

**Yu et al. 2024, *The Super Weight in LLMs* (anchor).** Read §3.1–3.2 and the Fig. 5
discussion. Removing the super weight and *restoring only the super activation*
recovers accuracy from 35.14 to 49.94, i.e. ~42% of the quality loss — so the super
weight acts partly, but not only, through the super activation (Table 1). Over 500
LAMBADA prompts, removing the super weight amplifies stopword probability: for
Llama-7B "the" ×2, "." ×5, "," ×10 (Fig. 5, App. Fig. 11), while non-stopwords fall by
2–3× to as little as 0.1%. Case study: "Summer is hot. Winter is __" goes from "cold"
at 81.4% to "the" at 9.0%. **What is missing:** any frequency baseline, any
random-vector null, and any measurement at layers other than the output.

**Xiao et al. 2023, *StreamingLLM* (ICLR 2024).** The sink exists because softmax must
sum to one, so surplus attention is dumped onto globally visible initial tokens (§3,
Eq. 1). Keeping the KV of just four initial tokens restores window-attention
perplexity (Tables 1–2); a single learnable sink token at pretraining suffices. Causal
via KV-cache interventions.

**Barbero et al. 2025, *Why do LLMs attend to the first token?* (COLM 2025).** Argues
sinks exist to prevent **over-mixing** — without them, representations at different
positions homogenize with depth. ~80% of attention on ⟨bos⟩ in a typical Llama-405B
prompt (§4.2). Context length, depth and data packing modulate sink strength. Theory
plus measurement on frontier models and models trained from scratch. **Supplies the
falsifiable prediction for H2: destroy the sink → inter-position representation
similarity should rise.**

### 1.5 Collapse under compression

**Gromov et al. 2024, *The Unreasonable Ineffectiveness of the Deeper Layers* (ICLR
2025).** Up to ~half the deepest layers of Llama-2-70B can be removed with minimal
QA degradation after QLoRA "healing", then performance collapses at a sharp,
**model-dependent threshold** (Fig. 3; robust to 20–55% pruning depending on model).
Causal (pruning + finetuning). *It does not characterize the degenerate text itself* —
it reports benchmark scores. Use it for "collapse is a threshold phenomenon", not for
"pruned models repeat."

**Zhou et al. 2026, *From Signal Degradation to Computation Collapse* (arXiv:2604.19884,
unrefereed).** Proposes two qualitatively distinct PTQ failure modes: **Signal
Degradation** (computational patterns intact, cumulative numerical noise, repairable
by training-free targeted intervention) and **Computation Collapse** (key components
stop functioning and the signal is "completely destroyed in the early layers",
requiring structural reconstruction). Validated by layer-wise information-flow tracing
and causal-pathway analysis, then by targeted repair experiments. *Inference:*
super-weight ablation is the cleanest possible instance of their Failure Mode II, and
this vocabulary is worth borrowing even though the paper is unrefereed.

### 1.6 Machine-translation hallucinations

**Raunak, Menezes & Junczys-Dowmunt 2021, *The Curious Case of Hallucinations in NMT*
(NAACL 2021).** ★ §3 defines the taxonomy this project needs: **Detached
Hallucinations** (fluent but wholly inadequate) and **Oscillatory Hallucinations** (an
inadequate translation containing repeating n-grams). The Fig. 2 example is literally a
function-word loop: "the us , for example , has been in the past two decades , but has
been in the same position as the us , and has been in the united states ." §4 shows
specific corpus-level noise patterns generate each type, and connects
perturbation-induced hallucinations to Feldman's long-tail memorization. §5 measures
IRS-OH and unique-bigram fractions. Causal at the data level.

**Guerreiro, Voita & Martins 2023, *Looking for a Needle in a Haystack* (EACL 2023).**
Natural (unperturbed) setting; 3,415 structured human annotations; shows the ten
previously proposed detection heuristics are largely inadequate and plain sequence
log-probability is the strongest, on par with reference-based COMET. Introduces
DeHallucinator (detect-then-rewrite). Rigorous evaluation, not a mechanism paper.

**Guerreiro, Alves et al. 2023, *Hallucinations in Large Multilingual Translation
Models* (TACL 2023).** ★ M2M-100 (up to 12B), NLLB as a fallback system, and ChatGPT,
over >100 directions. Key findings for this project: **oscillatory hallucinations
overwhelmingly dominate in mid- and high-resource pairs while detached hallucinations
dominate in low-resource ones** (§4, Fig. 2); the **TNG** heuristic (top repeated
translation n-gram count, controlled against the source's top repeated n-gram) detects
oscillatory hallucinations well; ALTI+ detects detached ones; oscillatory hallucinations
have consistently higher reversal rates under a fallback system, i.e. they are "less
related to model uncertainty."

**Dale, Voita, Barrault & Costa-jussà 2023, *Model Internal Workings Alone Do Well*
(ACL 2023).** ★ Uses **ALTI+ percentage source contribution** as an internal detector:
hallucinations are "detached" from the source, so they show low source contribution.
Improves detection accuracy for the most severe hallucinations by a factor of 2 over
prior methods and matches the best external-model mitigation; cross-lingual sentence
similarity does better still. The detector is a measurement; the mitigation is causal.

**Xu, Agrawal et al. 2023, *… via Model Introspection* (TACL 2023).** Contrastive
saliency analysis of hallucinated vs. non-hallucinated outputs under adversarial source
perturbations. Finds, against the prior hypothesis, that **source-contribution
*patterns*** (notably concentrated source attention) are stronger indicators than the
source-vs-target contribution ratio. Builds a lightweight detector beating QE and
pretrained-classifier baselines on En-Zh and De-En.

**Ferrando et al. 2022, *Towards Opening the Black Box of NMT* (EMNLP 2022).** The
ALTI+ method itself: layer-wise token attributions for **both** source and target
prefix in encoder-decoder Transformers (Alg. 1, §4). Shows perturbation-induced
hallucinations appear as a drop in source contribution (Fig. 11), and analyzes the EOS
token, the residual's share of cross-attention, and language-tag contributions in
multilingual models (Figs. 12–13).

**Dale, Voita et al. 2023, *HalOmi* (EMNLP 2023).** 18 translation directions, mixed
resource levels and scripts, sentence- and word-level annotations of full/partial
hallucination **and omission**, on naturally generated NLLB output. Shows conclusions
drawn from a single language pair largely do not hold at scale, and re-baselines the
detectors. The evaluation set for any NLLB intervention study.

**Binkowski, Adamczewski & Kajdanowicz 2026, *Attention Sinks as Internal Signals for
Hallucination Detection* (arXiv:2604.10697, unrefereed).** SinkProbe detects
hallucinations from attention-sink scores; notably, the classifier "preferentially
relies on sinks whose associated **value vectors have large norms**", and the paper
shows prior attention-based detectors are mathematically reducible to sink scores.
Decoder-only LLMs, not MT. **This is the only paper I found linking hallucination to
sink/outlier structure at all.**

---

## §2 Taxonomy of proposed repetition mechanisms

I group the literature's mechanisms and then ask which could be *caused* by zeroing one
scalar, as opposed to being properties of a healthy model at decoding time.

| # | Mechanism | Chief sources | Requires | Applies to super-weight damage? |
|---|---|---|---|---|
| M1 | **Probability feedback / self-reinforcement.** P(repeat) rises monotonically with prior repetitions and saturates. | Holtzman 2019; Xu 2022 (Fig. 2); Li 2023 | The repeated string already in context | **Only as amplifier.** Cannot explain the *first* repeated token. |
| M2 | **High-inflow funnel.** Many contexts predict the same word, so trajectories cycle through it. | Fu 2021 §2 | Nothing — a corpus property | **Background.** Predicts *which* token (frequent function words) if the context signal is weak, but is present in the healthy model too. |
| M3 | **Copy circuits / induction heads.** [A][B]…[A]→[B] locks onto a prior occurrence. | Olsson 2022 **[unopened]**; Hiraoka 2024 §5; Wang 2025 (preprint) | A prior occurrence in context | **Only as lock-in.** Predicts prompt-dependent repeated content, which the observed outputs do not obviously show. |
| M4 | **Repetition neurons / SAE repetition features.** Dedicated components that ramp up during repetition. | Hiraoka 2024; Yao 2025 | Repetition already under way | **Effect at least as much as cause.** Testable: does the damaged model even recruit them? |
| M5 | **Fallback to the unigram prior.** With no usable contextual signal, the output decodes to the corpus unigram distribution, which is dominated by function words and punctuation. | Meister 2023 (Fig. 1); Stolfo 2024 (token-frequency neurons); Welleck 2019; **Puccetti 2022 §4.1** | Loss of contextual signal | **Yes — the leading candidate.** This is the only mechanism whose *stated prediction* is "frequent tokens up, rare tokens down," which is what Yu et al. Fig. 5 reports. |
| M6 | **Loss of the sink / implicit attention bias.** The massive activation is a bias term; remove it and surplus attention has nowhere to go, causing over-mixing. | Sun 2024 §3–4; Xiao 2023 §3; Barbero 2025; Cancedda 2024; Yona 2025 | Sink destroyed | **Yes.** Mechanistically the closest to the intervention actually performed. |
| M7 | **Final-norm scale / unembedding null space.** A huge residual component changes the LayerNorm denominator, rescaling every logit and the entropy. | Stolfo 2024 §3 | A large-norm residual component | **Yes, at least partially** — the super activation *is* a large-norm residual component that persists through the skip connections (Yu 2024 Fig. 2, panel II). |
| M8 | **Training-data repetition.** | Li 2023 | Training-time control | **No.** Inference-time ablation cannot invoke it. |
| M9 | **Likelihood-objective artefact.** | Welleck 2019 | Training-time control | **No** as an explanation of the ablation, but it explains why function-word mass is sitting there to be unmasked. |

**The split that matters.** M1, M3, M4, M8 and M9 were all characterized in *healthy*
models generating fluent text, where repetition is a subtle preference. M5, M6 and M7
are the only three that can be *initiated* by removing one weight. The project's first
job is therefore to establish which of M5/M6/M7 fires first, and only then to ask
whether M1/M3/M4 take over as lock-in.

---

## §3 Candidate mechanistic hypotheses for the "We. We. We." collapse

Each is stated so that it can be false, with the measurement that separates it from its
neighbours and the null it must beat.

### H1 — Unigram-prior fallback ("the residual stops saying anything, so the unembedding says what it always says")

**Statement.** After ablation the final-layer residual carries little or no
context-specific information, so the logits approach the model's context-free default,
which is the training-corpus unigram distribution. The repeated token is then simply a
high-frequency function word, and the loop is Fu's high-inflow funnel (M2) operating on
a distribution that has lost its contextual tilt.

**Discriminating measurements.**
1. Compute the full next-token distribution over ~500 prompts, healthy vs. ablated, and
   report `KL(p_ablated ‖ unigram)` against `KL(p_healthy ‖ unigram)`. H1 predicts the
   ablated model moves *toward* unigram.
2. Regress log p_ablated(token) on log corpus frequency; H1 predicts a slope near 1.
3. **Logit lens at every layer, plus the tuned lens** (Belrose 2023) as the calibrated
   version — the raw logit lens is brittle and, per Cancedda 2024, structurally blind to
   the dark band that implements sinking.
4. **The null:** decode norm-matched random Gaussian residuals through the final norm
   and unembedding. If the ablated distribution equals the random-vector distribution,
   the residual is pure noise; if it is *closer to unigram than the random null is*, the
   frequency bias lives in the unembedding/final bias rather than in the residual, which
   is the more interesting version of H1.

**What the literature already predicts.** Meister 2023 Fig. 1 says an uninformative
model outputs unigram. Stolfo 2024 says there are dedicated neurons whose job is to move
the output toward the unigram distribution "in settings of high uncertainty." Puccetti
2022 §4.1 Fig. 3 shows exactly this outcome after ablating LayerNorm outliers in BERT.
So H1 is the hypothesis the literature most strongly endorses — and, notably, the one
Yu et al. did not test, because a stopword list is not a frequency baseline.

### H2 — Sink/bias destruction and over-mixing ("every position ends up saying the same thing")

**Statement.** The massive activation is the implicit attention bias (Sun 2024 §4.2:
value updates from sink tokens are near-identical across all query positions). Removing
it means the softmax's surplus attention (Xiao 2023 §3) has no reservoir, attention
spreads, and depth-wise over-mixing (Barbero 2025) drives every position's residual
toward the same average vector. Every decoding step then emits the same token.

**Discriminating measurements.**
1. Attention entropy per layer and per head, and **sink mass** (attention on position 0
   and on the delimiter tokens), healthy vs. ablated.
2. **Inter-position representation similarity**: mean pairwise cosine and the effective
   rank of the (positions × d_model) final-layer residual matrix. H2 predicts similarity
   up and effective rank down; H1 predicts no such collapse.
3. **The causal arm:** patch the massive activation back at its onset layer and ask
   whether the *repetition* disappears — not just whether accuracy recovers. Yu et al.
   report the accuracy version (42% recovered, Table 1) but never the generation version.
   This is a cheap, unreported experiment.

**Separation from H1.** H2 predicts *positional* collapse; H1 predicts *distributional*
collapse without positional collapse. They are independently measurable and can both be
true, in which case the ordering (H2→H1) is the finding.

### H3 — Induction/copy lock-in ("an arbitrary first token gets copied forever")

**Statement.** The first emitted token is arbitrary (set by H1 or H2), and an
induction/copy circuit plus the self-reinforcement effect then locks it in.

**Discriminating measurements.**
1. Does the identity of the repeated token depend on the prompt? Run 200 prompts × the
   models already replicated and compute the mutual information between prompt and
   repeated token. Near-zero MI falsifies H3-as-initiator.
2. Does the repeated token depend on the *last* prompt token specifically?
3. Ablate induction heads in the already-damaged model (prefix-matching + copying scores
   on random-token sequences, as operationalized in Wang 2025 §3.1) and check whether the
   loop survives.
4. Run Xu 2022's self-reinforcement protocol on the damaged model: repeat a sentence n
   times and plot p(token) vs. n. Flat curves would mean the damaged model's repetition is
   *not* the familiar phenomenon.

**What the literature already predicts.** Xu 2022 Fig. 2 shows self-reinforcement fires
even on random token strings, and Hiraoka 2024 Fig. 1 shows repetition neurons ramp up
*after* onset. Both make H3 an amplifier rather than an initiator. Wang 2025's
frequency-sensitivity result (frequent tokens fall into the loop earlier) is the one
piece of evidence that H3 and H1 might be the same story seen from two ends.

### H4 — Final-norm scale effects ("the logits are rescaled, not redirected")

**Statement.** The super activation is a large, roughly constant component of the
residual stream that persists through the skip connections (Yu 2024 Fig. 2, panel II).
RMSNorm divides by the residual norm, so removing it changes the normalization
denominator and rescales every logit — flattening or sharpening the distribution
without necessarily rotating it.

**Discriminating measurement.** Decompose the logit change into a *direction* term and a
*scale* term: recompute the output using the ablated residual **direction** with the
healthy final-norm scale, and vice versa. If the stopword shift survives when the scale
is held fixed, H4 is not the driver.

**What the literature already predicts.** Stolfo 2024 §3 shows this channel is real and
is deliberately used by entropy neurons — but also that its signature is *entropy change
with the argmax preserved*. Yu et al. report the argmax **changing** ("cold" → "the"), so
H4 cannot be the whole story. It is a good candidate for the *low confidence* of the
post-ablation prediction (9.0%), and it must be controlled for before any H1 claim, since
a pure rescale would also inflate the relative mass of already-frequent tokens.

### H5 — The super weight *is* a sink neuron (identity hypothesis, not yet in either literature)

**Statement.** Yona 2025's "sink neurons" — a sparse set of MLP neurons selected as
TopK_j(‖MLP_j(BoS)‖) whose output writes a huge-norm contribution to the first token's
hidden state — and Yu 2024's super weight — a scalar in an early `down_proj` that creates
a huge activation — are the same object described by two literatures that do not cite
each other on this point.

**Discriminating measurement.** For each replicated model, take the super weight's
(output row, input column) in `down_proj`; the column indexes an intermediate neuron.
Check (a) whether that neuron is in Yona's top-K by ‖MLP_j(BoS)‖, and (b) whether zeroing
the super weight collapses the BoS hidden-state norm spike as Yona's Fig. 3 ablation does.
Costs no generation and no training. A positive result unifies two mechanisms; a negative
result is itself publishable as a distinction.

---

## §4 The MT angle

**The bridge is already named.** Raunak et al. 2021 §3 calls "an inadequate translation
that contains repeating n-grams" an **oscillatory hallucination**, and their Fig. 2
example is a function-word loop. Post-super-weight-ablation output *is* an oscillatory
hallucination in an LM. That gives the project a pre-existing label, a pre-existing
detector, and a pre-existing evaluation set — none of which the super-weight literature
uses.

**What is reusable, concretely.**

1. **TNG (top repeated n-gram)** — Guerreiro 2023 TACL §5: count the top repeated
   translation n-gram, controlled against the top repeated *source* n-gram. A black-box,
   language-agnostic oscillation metric that runs on any NLLB / Tower / EuroLLM output. It
   is the right dependent variable for a super-weight dose-response curve, in place of the
   ad-hoc "rep-n" measures used in LM work.
2. **ALTI+ source contribution** — Ferrando 2022 (method), Dale 2023 (as a detector).
   This is something a decoder-only LM *cannot* provide: an encoder-decoder model lets you
   separate "how much of this token came from the source" from "how much came from the
   target prefix." Both H1 and H2 predict source contribution should collapse as the super
   weight is scaled toward zero. **A monotone dose-response curve from one scalar weight to
   ALTI+ source contribution would be a genuinely new result** and is squarely within this
   project's budget.
3. **Source-contribution *patterns*, not just magnitude** — Xu 2023 (TACL) found the
   *pattern* (concentrated source attention) discriminates better than the ratio. Worth
   measuring both, since H2 predicts a specific pattern change (dispersal), not merely a
   magnitude change.
4. **HalOmi** — Dale 2023: 18 directions, sentence- and word-level annotations on NLLB
   output, with re-baselined detectors. Both the evaluation set and the warning that
   single-language-pair conclusions do not generalize (relevant to CLAUDE.md's rule 3 on
   stating coverage).
5. **The resource-level asymmetry** — Guerreiro 2023 TACL Fig. 2: oscillatory
   hallucinations dominate mid/high-resource directions, detached ones dominate
   low-resource. *Inference:* if super-weight ablation pushes NLLB toward oscillation, the
   damage should look different by resource level, which is a real cross-lingual question
   rather than a decorative one. It also connects to ``
   (Tang 2024 language-specific neurons; the 2026 NLLB attention-sink paper) and to the
   repo's existing multilingual-quantization work.

**Has anyone linked hallucinations to outlier structure?** In what I read: **almost no
one.** The single instance is Binkowski et al. 2026 (unrefereed), which detects
hallucinations from attention-sink scores in decoder-only LLMs and notes the classifier
prefers sinks whose *value vectors have large norms* — i.e. massive activations. Nothing I
found links **NMT** hallucinations to super weights, massive activations, or outlier
dimensions. Guerreiro 2023, Dale 2023, Xu 2023 and Ferrando 2022 all work at the level of
attention/attribution and never mention outliers; Raunak 2021 works at the data level.
*Inference:* that is the opening. State it as "not found in this search", not as "does not
exist" — I searched arXiv, ACL Anthology and the web, not an exhaustive citation graph.

---

## §5 Open problems

Each as: **problem — why it is open — experiment within budget — what would count as an
explanation — confounds.** All are sized for a 125–150h MS project on a single-GPU SLURM
allocation with 7–13B models, and all assume the repo's rigor floor (effect + 95% CI,
seeds as the unit of independence, multiplicity correction on any layer scan with the
family size stated).

### OP1 — Is the post-ablation distribution the unigram prior, or is it noise?
*Open because* Yu et al. Fig. 5 reports stopword probabilities against no frequency
baseline and no null. "Stopwords went up" and "the distribution reverted to unigram" are
different claims with different mechanisms, and only the second is explanatory.
*Experiment:* on ~500 wikitext-2 / LAMBADA prompts per model, dump the full next-token
distribution for healthy, ablated, and ablated-with-super-activation-restored; compare
each by KL against (a) a corpus unigram distribution estimated on a pretraining-like
corpus, (b) the healthy distribution, (c) a norm-matched random-residual null decoded
through the final norm and unembedding. ~1 GPU-hour per model.
*Explanation looks like:* the ablated distribution is significantly closer to unigram
than the random null is, with the log-frequency regression slope reported with a CI.
*Confounds:* the unigram estimate is corpus- and tokenizer-dependent and each model has a
different tokenizer; "stopword" is not the same as "high frequency"; instruction-tuned
models (Llama-3.1-8B-Instruct, Phi-3-mini) have a different effective prior than base
models and should not be pooled with them.

### OP2 — Positional collapse or distributional collapse?
*Open because* nobody has measured inter-position representation similarity after a
*targeted* outlier ablation; Barbero 2025 predicts over-mixing when sinks are absent, but
tested that by training models without sinks, not by breaking one.
*Experiment:* mean pairwise cosine and effective rank of the final-layer residual matrix
across positions, for healthy / ablated / super-activation-restored, on the models already
replicated. Cheap — a single forward pass per condition.
*Explanation looks like:* effective rank collapsing under ablation and recovering under
the patch, which would put M6 upstream of M5.
*Confounds:* RMSNorm makes cosine scale-invariant, so report unnormalized distances too;
the first few positions are always similar, so exclude a burn-in prefix; effective rank is
sensitive to sequence length, so hold it fixed.

### OP3 — What is the causal ordering: sink loss → context loss → repetition, or the reverse?
*Open because* every quantity involved is a monotone function of the same scalar, so
correlation across the ablation is uninformative on its own.
*Experiment:* a dose-response sweep, scaling the super weight through {1.0, 0.75, 0.5,
0.25, 0.1, 0} (Yu et al. already sweep 0–3 for accuracy but not for these quantities), and
plotting on a common axis: massive-activation magnitude, sink attention mass, attention
entropy, KL-to-unigram, and repetition rate. Then the patching arm: restore *only* the
massive activation; separately, restore *only* the attention pattern.
*Explanation looks like:* the knees of the five curves ordering consistently across
models, with the patching arm confirming that restoring the earlier quantity restores the
later ones and not vice versa.
*Confounds:* the ordering of knees on a shared scalar axis is suggestive, not causal —
the patching arm is what carries the claim; scaling a weight is not the same intervention
as zeroing it, and Yu et al. found *increasing* the weight also degrades quality, so the
curve is not monotone at the top end.

### OP4 — Is the super weight the same object as Yona's sink neuron?
*Open because* the two literatures do not connect: Yu et al. describe a scalar in
`down_proj`; Yona et al. describe a sparse set of MLP neurons selected by the norm of
their BoS contribution. Nobody has checked whether these index the same units.
*Experiment:* for each replicated model, map the super weight's `down_proj` column to its
intermediate neuron and test membership in TopK_j(‖MLP_j(BoS)‖); then check whether zeroing
the super weight collapses the BoS norm spike (Yona Fig. 3's ablation, run backwards).
Hours of work, no generation.
*Explanation looks like:* set identity (or a clean non-overlap) reported per model with the
K threshold stated.
*Confounds:* Yona works on Llama-2 at "sink layer 1" while Yu's super weights sit at layers
1–4 depending on model, so a layer match may be trivial and does not itself establish
identity; the top-K criterion is threshold-sensitive, so report the rank, not a binary.

### OP5 — Does the repeated token depend on the prompt at all?
*Open because* the evidence is anecdotal: OLMo-1B says "We", Mistral-7B says "the main.
without without", and these look like model-specific fixed points rather than
prompt-driven completions — but that has never been measured.
*Experiment:* 200 prompts × the 5 replicated models under greedy decoding; record the
repeated token, its onset position, and the cycle length; compute mutual information
between prompt and repeated token, with a permutation null.
*Explanation looks like:* near-zero MI (falsifying H3-as-initiator and supporting a fixed
point), or substantial MI (rehabilitating H3).
*Confounds:* greedy decoding only — a sampled decoder may not fix a point at all, so state
the decoder in the claim; chat templates inject tokens for the instruct models; short
prompts may not give induction heads anything to lock onto, so vary prompt length.

### OP6 — Does any of this transfer to an encoder-decoder MT model?
*Open because* everything above (and every mechanism paper in §1.2–1.4) is decoder-only.
NLLB is encoder-decoder, and where its massive activations live — encoder, decoder, or
cross-attention — is not established here.
*Experiment:* run Yu et al.'s detection procedure on NLLB-200-distilled-600M and 1.3B;
ablate; measure spBLEU/chrF++, the TNG oscillation rate, and ALTI+ source contribution on
FLORES-200 across ~8 directions spanning resource levels, with the dose-response sweep from
OP3.
*Explanation looks like:* a monotone dose-response from one scalar to source contribution
and to oscillation rate, holding across directions, with coverage stated per CLAUDE.md
rule 3.
*Confounds:* the detection procedure assumes a decoder-only `down_proj` layout and may need
adaptation per component; the language tag already has an anomalous ALTI+ contribution
(Ferrando 2022 Figs. 12–13) and must be excluded from the source-contribution denominator;
FLORES is out-of-domain for several directions; 8 directions is exploratory, not a paper
claim, and Guerreiro 2023 TACL's resource-level asymmetry means direction choice can
manufacture either result.

### OP7 — Is post-ablation repetition even the same phenomenon as healthy-model repetition?
*Open because* self-reinforcement, induction heads and repetition neurons were all
characterized in models generating fluent text. Nobody has run those protocols on a model
whose perplexity is 10^3× worse.
*Experiment:* run Xu 2022's self-reinforcement protocol (repeat a sentence n times, plot
p(token) vs. n) and Hiraoka 2024's Δn repetition-neuron detector on both the healthy and
the ablated model, and compare the identified neuron sets.
*Explanation looks like:* either the damaged model recruits the same repetition neurons and
shows the same self-reinforcement curve — in which case the ablation merely opens the door
to a known mechanism — or it does not, in which case this is a distinct failure mode that
needs its own account and its own name.
*Confounds:* the ablated model's probabilities are compressed toward the prior, so
probability-based statistics are not comparable across conditions — use rank-based ones;
Δn is defined relative to a repetition *onset*, which may be ill-defined if the ablated
model repeats from token one; the two neuron sets will overlap by chance, so report the
overlap against a random-neuron null.
