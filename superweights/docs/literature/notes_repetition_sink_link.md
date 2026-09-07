# Notes — the onset of natural repetition, and whether anyone has tied it to the sink

Compiled 2026-09-05, in answer to a single question: **is natural repetition in a healthy
LM the same failure as the repetition a super-weight lesion produces?**

Method and provenance conventions, same as the sibling notes files:
- Statements marked **read** were extracted from a PDF in `papers/`
  with `pdftotext` and are cited to the section, figure, equation or table actually opened.
- **[unopened]** marks a paper known only from a citation or an abstract.
- Sentences beginning *Inference:* are my reasoning past the papers, not a report of them.
- Venues are as printed on the PDF or in arXiv metadata; "preprint" means unconfirmed.

---

## §0 The answer, up front

**The hypothesis is partly answered, and the part that is answered comes back negative.**

1. Nobody has run the experiment. **No paper measures sink mass, massive-activation magnitude,
   or the constant's contribution per decoding step across the onset of spontaneous repetition
   in an unmodified model.** Both literatures exist and do not cite each other on this point.
2. But the *naive* version of the prediction is already refuted. **Gu-2025 §3.3 Table 1**: a
   repeated-token input drives the position-0 sink metric to **0.00%** in healthy Mistral-7B,
   LLaMA-2-7B and LLaMA-3-8B — the same observable the lesion produces — because every repeated
   token acquires the massive activation and the mass **disperses**. Observing "sink mass falls
   during natural repetition" would therefore confirm nothing. What survives is (a) whether it
   falls **before** the first repeat, and (b) whether **total** sink-like mass falls or is merely
   redistributed. No published metric can tell those apart.
3. Three of the four measurements that touch natural repetition point the **wrong way** for the
   hypothesis: attention piles *onto* sink-like delimiter tokens (Mahaut §4.2, newlines at 9.9×
   uniform); forcing it there or away changes nothing (Mahaut §4.2 causal test); and the one
   natural-failure-mode paper that names sinks reads the sign as *more* sink concentration
   (Binkowski). Stolfo §6.2 supplies an innocent explanation for any sink drop that does appear:
   attending to BOS **is** the dormant state, so an induction head firing is attention moving off
   the sink in a perfectly healthy model.
4. **The strongest standing objection is Mahaut & Franzon 2025**, which asked exactly this
   question of a different pair of conditions — natural vs ICL-induced repetition in Pythia-1.4B
   — and found *qualitatively different internal processes*. "Same mechanism" is therefore a
   claim needing positive evidence, not a reasonable prior.
5. The comparison has now been made behaviourally but never internally: **Lee-2026 Table 1**
   (same Llama, 5.9 → 12.4/15.4% sampling, 26.6 → 63.1/63.7% greedy, perplexity ≈ unchanged) and
   **Roll-2026 Fig. 4B** (lesion repetition goes to **zero** at T = 0.7 with repetition penalty
   1.2). Damage multiplies the rate of the same measured phenomenon and the same decoding knobs
   suppress both — which is the best available argument *for* a shared route.
6. The clean opening is narrow and real: re-run **Duan §3.2 Fig. 6/7's design** — 50 looping vs
   50 matched non-looping generations, a fixed window before onset, window sizes 64–512 —
   **without** its stated exclusion of attention sinks, reporting total sink-like mass and not
   only position-0 concentration, on a partial-dose lesion sweep as the comparison arm.

---

**This file does not repeat `notes_repetition_degeneration.md` or
`notes_super_weights_massive_activations.md` §3.** Those cover the mechanism taxonomy
(M1–M9), the ablation-side hypotheses (H1–H7), and the MT angle. This file covers only the
**onset** question and the **sink/constant** link, and it corrects two things the earlier
notes got slightly wrong (§2.4, §5.3).

---

## §1 What each paper actually shows about the *onset* of natural repetition

The organising question: **is any internal quantity measured before the first repeated
token, or only after?** The short answer, across everything read, is *almost nothing is
measured before onset, and the one thing that is (Hiraoka Fig. 1) has a contaminated
pre-window.*

### 1.1 The papers that define the phenomenon but measure only probabilities

**Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration* (ICLR 2020).** read.
Fig. 4 caption: *"The probability of a repeated phrase increases with each repetition,
creating a positive feedback loop. We found this effect to hold for the vast majority of
phrases we tested, regardless of phrase length or if the phrases were sampled randomly
rather than taken from human text."* §5.2: *"Nor does natural language tend to fall into
repetition loops, even though the model tends to assign high probability to this, as seen
in Figure 4."* The repetition metric is defined in the Fig. 9 caption: a phrase of minimum
length 2 counts as a repetition when it repeats **at least three times at the end of the
generation**, within the first 200 tokens.
**Onset status: nothing before onset.** Fig. 4 starts at repetition 1 — i.e. the phrase is
already in the context. No internal quantity is measured at all; the paper is entirely at
the output-distribution level. The "random phrases" control is the same one Xu 2022 later
reproduces.

**Xu et al. 2022, *Learning to Break the Loop* (NeurIPS 2022).** read §2.1–2.2, conclusion.
Three metrics over a 16-layer decoder trained on Wikitext-103: TP_n (average token
probability of the n-th repeat), IP_n (fraction of tokens whose probability rose relative
to the first occurrence), WR_n (fraction that are also the argmax). Corpora D_random,
D_book, D_wiki, 1,000 sentences each.
**This is the closest thing in the literature to a pre-onset measurement, and it is still
not one.** §2.2, "Why sentence repetitions occur?": *"IP₁ is higher than 90% across the
various corpora … Note that the token repetition has not occurred at the current prediction
step, and there is only the same sentence-level context."* So the boost is measured at the
step where the model is about to emit the first repeat — but the *sentence it will repeat
is already in the context*. This is a measurement at n = 1, not at n = 0. There is no
condition in which the model is measured before any copy-able material exists.
Their own conclusion says the mechanism question is untouched: *"there should be deeper
reasons for why the model raises the probability of repeating tokens from the perspective
of model embedding, neural network architecture or intrinsic characteristics of language.
Our current analysis has not touched these aspects."*
**Onset status: n = 1 probabilities only; no attention, norm or sink quantity anywhere in
the paper.**

**Fu et al. 2021, *A Theoretical Analysis of the Repetition Problem in Text Generation*
(AAAI 2021).** read §2.2–2.3 and Corollaries 1.1–1.2. The bound on Average Repetition
Probability blows up when the **inflow** of a word — "the probability sum of all words that
take it as the subsequent word" (§2.2 discussion of Cor. 1.2) — is large. The paper is
explicit that this is *not* a frequency claim: *"As is shown in many previous works, most
repetition sentences contain high-frequency words. By Corollary 1.2, we can conclude that
it is not the high-frequency words, but the high inflow words that really lead to repetition
generation."*
**Onset status: no onset concept exists in this theory.** Inflow is a property of the
transition matrix B, i.e. of the trained model + corpus, not of a particular forward pass.
*Inference:* Fu therefore predicts **no pre-onset internal signature at all** — repetition
is a property of the trajectory through a fixed stochastic matrix. That makes it the cleanest
null for the brief's hypothesis: if a pre-onset attention signature exists, Fu's account is
incomplete.

**Li et al. 2023, *Repetition In Repetition Out* (NeurIPS 2023).** read §"Relation to
Previous Hypotheses" and §"Why LMs Learn the Repetition Patterns?". Empirically ties Fu's
construct back to data: **26% of high-inflow word pairs in each Wikitext-103 sentence are
repetitive**, and merging *only* the repetitive high-inflow pairs (8.1% of training words)
matches the full HI-RE method (31.1% of words), while merging an equal number of *random*
high-inflow pairs does nothing (their Table 3, lines 4–6). Their taxonomy of 300 sampled
natural repetitive bigrams: Grammar / Theme / Limited inventory.
**Onset status: training-time only.** Nothing measured at inference.

### 1.2 The papers that open the model, and all of them open it after onset

**Hiraoka & Inui 2024, *Repetition Neurons* (NAACL 2025 short).** read §2.2, §3.1–3.2, §5,
App. A–E. This is the only paper with a **pre-onset window**, and it is worth stating its
construction precisely because the contamination matters.
- Dataset (§2.2): 1,000 self-generated repetitive texts per model. A text counts as
  repetitive if **the same 10-gram appears three times at equal intervals within 100
  tokens**; texts without ≥50 tokens on each side of onset are dropped. **"The onset of
  repetition is defined as the point where the repeated sequence appears for the second
  time."**
- Δ_n (Eq. 3) = mean activation over the r = 30 tokens **after** onset minus mean over the
  r = 30 tokens **before** onset.
- Fig. 1 plots the top-4 Gemma-2B repetition neurons across those 30 tokens before and after
  onset, averaged over 1,000 texts.
**The pre-onset window is contaminated by construction.** Because onset s is the *second*
occurrence, the 30 tokens before s largely *are* the first occurrence of the sequence that
will be repeated. So Hiraoka's "normal" range is exactly Xu's n = 1 condition. Anyone reusing
this protocol to look for a pre-onset sink signature would be measuring during the first copy,
not before it.
**§5** compares repetition neurons with two head families defined on the same texts:
"induction heads" (attending to repeating tokens that come *after* the current input token)
and "self-finding heads" (attending to the repeating token *identical* to the current input
token); a head qualifies if its total attention to the target tokens after the second
repeating position **exceeds 0.5**. Result (Fig. 6): Gemma-2B's repetition-neuron peaks
share two layers with induction heads, but the highest induction peak (L14/18) does **not**
coincide with the highest repetition-neuron peak (L18/18); for Pythia-2.8B and LLaMA-3.2-3B
there is **no strong alignment** with induction heads, only some early-layer alignment with
self-finding heads. Their reading: repetition neurons respond to self-finding early and do
something else late.
**App. B/C are the most useful part for this project and are not in the earlier notes.**
The natural loops are *phrase-level and semantically coherent* — Pythia: "I also added a
little more garlic powder. … a little more oregano. … a little more basil. I also added a
little more basil."; Gemma: "I'm not sure if it's the right one for me." ×3; App. C: an
enumeration ("The view? The location? … The garage? The driveway? The driveway? The
driveway?") that decays into a fixed point. *Inference:* this is a **different surface
phenomenon** from the lesion output. Natural loops are long, syntactically well-formed
templates that lose their variable slot; the lesion output is a single function word or
punctuation mark from the first generated token ("We. We. We."). Any claim that they are
"the same failure" has to explain that difference, not gloss it.
**Onset status: measured across ±30 tokens around onset, but the pre-window is the first
copy; and the measured quantity is FFN activation, never attention or norm.**

**Yao/Yang et al. 2025, *Understanding the Repeat Curse … from a Feature Perspective*
(Findings of ACL 2025).** read §4.3.1–4.3.2, §5.1. Layer localisation (§4.3.1) is a logit-
difference attribution computed on **8 hand-built templates that already contain the
repetition** — the worked example is *"He hit Jack Jack Jack Jack Jack"* with "Jack" as
the correct continuation and "Jackson" as the incorrect one. Feature localisation (§4.3.2)
then steers SAE features on the top-2 attributed layers with coefficient λ = 2 and calls a
feature a repetition feature if the steered generation's repeat score ≥ ρ = 0.4.
**Onset status: post-onset by construction, and the "repetition" is induced by the prompt,
not spontaneous.** The localisation cannot in principle say anything about what happens
before the first repeated token.

**Wang et al. 2025, *Induction Head Toxicity Mechanistically Explains Repetition Curse*
(arXiv:2505.13514, preprint).** read §2.2, §3, §"Empirical Validation" and Table 1.
Setup: prompt a model to repeat a token N times and watch it overshoot. §2.2 vocabulary
sensitivity: **common/frequent tokens make models fall into the loop prematurely; rare tokens
make them halt early** (Fig. 1 right). Toxicity ratio τ_t = share of logit norm contributed
by induction heads; measured over 512 generated tokens, τ_t "increases sharply in
repetition-curse scenarios, while it remains relatively low in normal tasks" (MMLU-Pro,
ARC-Challenge as the normal condition). Lemma 3.5 predicts exponential entropy decay under
toxicity; Table 1 (Qwen2.5-7B-Instruct) reports initial entropy 5.2 → decay rate 0.3 in the
normal condition (τ < 0.5) and 4.8 → **1.7** in the toxic condition (τ ≥ 0.65).
**This is the single most decisive prediction in the natural-repetition literature for
distinguishing it from the lesion**, and §3 below turns it into a test. Natural repetition
should be **low-entropy / sharpened**; the lesion produces a **flat** distribution whose
argmax has moved ("cold" 81.4% → "the" 9.0%, Yu 2024 §3.2).
**Onset status: τ_t is tracked per generated token across 512 steps, which is the only
per-step internal time series in this literature — but the generation is prompted to repeat
from the start, so there is no clean pre-onset segment.** Caveats: unrefereed, τ and γ = 0.65
are stipulated not fitted, and one model in Table 1.

**Yona, Shumailov, Hayes, Barbero & Gandelsman 2025, *Interpreting the Repeated Token
Phenomenon* (arXiv:2503.08908, Google DeepMind, preprint).** read §4.1–4.4, §5, §6–7 in full.
This is the paper that runs the arrow **backwards** — repetition → sink — and its scope
limits matter a great deal for this project.
- §4.1: in intermediate layers the average attention received by repeated tokens is
  comparable to the attention received by the first token (Fig. 1, LLaMA-2, averaged over
  heads in a layer).
- §4.2: sink neurons = TopK_j(‖MLP_j(BoS)‖). Zero-ablating them collapses the high norm for
  BoS **and** for repeated tokens (Fig. 3: residual-stream activation norms ~1000 → <150 at
  layers 1/16/29 on an input of 1200 repeats). Separately, a single **first-detector neuron**
  (MLP₀ gate neuron 912 in LLaMA-2) perfectly separates first from non-first tokens after
  the first attention layer (Fig. 4).
- §4.3: Theorem 4.1 — for a fixed k-token prefix followed by n repeats of x, as n → ∞ the
  last token's representation converges strongly to the representation of the singleton
  sequence (x). Mechanism: softmax leakage (Veličković et al. 2024) drives the constant-size
  prefix's influence to 0. §4.3 also explains *how* the first attention layer marks the first
  token: near-orthogonal Q and K in layer 1 make the heads "other-token detectors", so causal
  masking forces the first token to self-attend. A long run of identical tokens is
  indistinguishable from a single-token sequence to that layer, so it gets marked too.
- **§4.4 Table 1 — the number that constrains everything.** Repeats needed to induce a sink:
  LLaMA-1-7B **450**, LLaMA-2-7B **1000**, Llama-3-8B-Instruct **4000**, Mistral-7B-Instruct
  **1200**. Sink layer 1 or 2; sink-neuron IDs listed.
- §6 Limitations, Fig. 6: **for some tokens ("Another", "bit", "dust") repetition does not
  create an attention sink at all.**
- §5.3: the mitigation is a two-line patch on MLP₁ neuron 7890 that freezes the sink neuron's
  value at its "no-sink" level; Table 2 shows ≤1.8 pp change on MMLU/HellaSwag/TruthfulQA/
  WinoGrande/ARC across three models.
**Onset status: this mechanism is not about onset at all.** It requires 450–4000 repeats.
*Inference:* it cannot explain a spontaneous loop that starts at repeat 1–3, which is what
natural degeneration is (Holtzman's criterion is three repeats; Hiraoka's is a 10-gram
appearing three times). Yona is a theory of what a *very long* established loop does to a
model, not of what starts one. It is also the strongest existing evidence that the
sink↔repetition relation runs **repetition → sink**, i.e. the opposite of the brief's
hypothesis.

**Stolfo, Wu et al. 2024, *Confidence Regulation Neurons* (NeurIPS 2024).** read §6.1–6.2.
Induction setup: 100 tokens from C4 duplicated into a 200-token sequence, ×100 sequences,
GPT-2 Small. During the second occurrence, loss and entropy both drop sharply (Fig. 5a) and
four of six entropy neurons change activation substantially; clipping neuron **11.2378** to
its first-occurrence mean gives **up to a 70% reduction in entropy** (Fig. 5b) — so that
neuron's job during induction is to *raise* entropy against the model's own overconfidence.
**§6.2 is the bridge this project needs and it is not in the earlier notes.** To test whether
induction heads cause the entropy-neuron activation, they perform **BOS ablations**: *"motivated
by the observation that attention heads frequently attend to BOS when inactive (e.g., when
induction is not occurring), we set the attention pattern of an attention head to always
attend to the first token of the sequence (i.e., BOS)"*. Ablating GPT-2's top prefix-matching
head L5H1 drops 11.2378's activation significantly; adding L5H5 and L6H9 brings it back to
its first-100-token level. Their baseline heads are **6 heads in layers 5–6 with mean
attention to BOS > 0.6 but prefix-matching score < 0.1** — i.e. high BOS attention without
induction.
*Inference, and this is the load-bearing one for §3:* in this framing, **attending to BOS is
the dormant state and induction is the active state**, so an induction head firing *is*
attention moving **off** the sink. That gives a sink-mass drop at repetition onset a
completely innocent explanation — one that has nothing to do with the sink weakening — and
predicts the drop is **localised to induction heads** rather than model-wide.

### 1.3 The lesion side, re-read for what it says about onset

**Cancedda 2024, *Spectral Filters, Dark Signals, and Attention Sinks* (ACL 2024).** read
§5–§7, §9, App. D Tables 5–7. Already the nearest precedent in the earlier notes; three
things there were under-stated.
1. **The object.** Fig. 9: the BoS residual norm is decomposed by layer into MLP and MHA
   contributions and into U-Dark / U-Light projections. *"After an initial phase of input
   enrichment that apparently does not need an attention sink, the MLP at L3 blasts off a
   vector of large norm and almost completely U-dark. This vector acts as an attention
   collector for heads in need of a sink, and is kept around until the last few layers"*
   (L3 in 13B; L1 in 7B).
2. **The shavings-swap control (Fig. 10–11) is the best control design in either literature.**
   Instead of suppressing the filtered component, they *swap it with the same component from
   a different sample at the same position and layer*. Because the BoS residual is identical
   across samples (autoregressive causal masking), the swap perturbs **every residual stream
   except BoS**. Result (Fig. 11 left): the step-decrease in NLL on adding the last 5% of
   singular vectors **disappears**; filtering only Token 0 (Fig. 11 middle) makes it clearly
   visible. Conclusion: *"the primary function of the dark subspace is to enable the crucial
   attention sink mechanism."*
3. **The repetition claim, and its caveat.** §6.1: *"We notice that the application of Ψ
   filters often results in the model entering repetitive patterns. This is consistent with
   the possibility that attention heads largely copy representations from the RSs of previous
   tokens, and inhibiting attention sinking results in over-copying."* **But App. D shows the
   dose matters and the control is not clean.** At 5% removal (Table 6) the Ψ column loops
   ("The full, 200 year, 1,166,1,1661,1,1,1661…") while the matched random-orthonormal-basis
   column Rnd is fluent. At 20% removal (Table 7) **Rnd loops too** ("At the time of the early
   2000-600, 2000 and 2000: 2000, 2000, 2000 and 2000: 2000, 2000,"). So repetition is
   dose-dependently producible by a *generic* rank-matched perturbation, and the
   sink-specific claim rests on the **low-dose** regime where Ψ loops and Rnd does not.
   *Inference:* this is the single most important methodological lesson in this file for the
   project's H6 (off-manifold artefact): the discriminating experiment is a **partial-dose**
   one, not a full ablation, because at full dose everything loops.
**§7 "Dark Signals and Attention Bars" is directly about natural text and was missed
earlier.** In ordinary prompts some non-BoS tokens receive large, low-variance attention
("attention bars"). Operationalised as High-Mean Low-Variance token–layer pairs: mean received
attention over subsequent tokens > τ_µ = .018 and variance < τ_σ = .01, ignoring layers 0–3,
BoS, and the last 4 tokens — **12,236 of 404,748 token–layer pairs** on LLaMA-2-13B over
ccnet-405. Fig. 15: from ~L13 the HMLV tokens' U-Dark ratio rises well above the population
(which stays below .75), **but their cosine similarity with the BoS residual is close to zero
and indistinguishable from the overall distribution.** Cancedda's own verdict: *"These results
neither confirm nor invalidate the hypotheses that attention bars are auxiliary attention
sinks."*
**Onset status: this is a natural-text measurement of secondary sinks, at ~3% of token–layer
pairs, with no link to generation and no repetition measurement.** It is nevertheless the only
place anyone has asked "which ordinary tokens act sink-like on this input", which is the
measurement the brief's hypothesis needs.

**Su et al. 2025, *Unveiling Super Experts in MoE LLMs* (ICLR 2026).** read §5.1–5.2, App. C, E.
Two things beyond the earlier notes.
- **Eq. 7's Attention Sink Decay Rate is a lesion-only metric.** D_sink = 1 − (1/H)·Σ_h
  [Σ_{i∈S} p'_ti / Σ_{i∈S} p_ti], where p and p' are the sink-token attention scores *before
  and after* SE pruning. It is a **ratio between two model versions**, so it cannot be
  computed for natural repetition in an unmodified model. Any transfer needs a re-definition
  (e.g. Gu's absolute `sink rate` at ε = 0.3, which Barbero also uses). Reported result: ≥90%
  across layers on Qwen3-30B-A3B, C4, first token as sink (Fig. 9); Fig. 8 shows sinks
  disappearing entirely at L25H10 and L35H10.
- **App. C is a dissociation nobody has used.** Final-layer "outlier experts" also carry
  extreme `down_proj` activations, but (i) pruning them barely moves PPL, (ii) *"in both
  Qwen3-30B-A3B and DeepSeek-R1, pruning outlier experts does not result in repetitive outputs
  on reasoning benchmarks such as Math-500, whereas pruning SEs does"*, (iii) their
  distribution **varies with the input dataset** while the SE distribution is stable across
  C-Eval/GSM8K/HumanEval (Figs. 14–19).
  *Inference:* repetition after ablation is **not** a generic consequence of removing a large
  activation. It is specific to the early-layer, sink-supporting one. This is a free control
  the project should cite.

**Sun et al. 2024, *Massive Activations in LLMs* (COLM 2024).** re-read §4.1 for one detail
that matters for §5: the tokens carrying massive activations in LLaMA-2-7B and Phi-2 are
**not only the first token** — the attention-logit maps in Figs. 5–6 are labelled with "The",
"." and "\n". *"our results on LLaMA2-7B and Phi-2 indicate that LLMs also allocate substantial
attention to other tokens and they are associated with massive activations."* §4.1 also gives
the sign structure: after the massive activations appear, most attention logits go negative,
while logits whose key belongs to a massive-activation token are slightly positive.

**Barbero et al. 2025, *Why do LLMs attend to the first token?* (COLM 2025).** read §2, §3–3.2,
§"Impact of attention sinks on downstream performance". The sink metric is Gu's:
`sink rate = (1/LH)·Σ_{h,ℓ} 1[(1/T)·Σ_j α^{(ℓ,h)}_{1,j} > ε]`, ε = 0.3 — the **fraction of
heads** whose mean attention to position 1 exceeds ε. §3 defines two collapse notions and
proves rank collapse ⇒ representational collapse (Prop. 3.1). **Representational collapse
(Eq. 2) is stated for "some token sequence in which the tokens n−1 and n are repeated and the
underlying sequence or 'prefix' grows"** — i.e. the formal object is literally a repeated
final token. §3.2 measures mixing by perturbing one token ("greatest" → "best") in Gemma-7B
and comparing representation change with and without ⟨bos⟩ (Fig. 2): the perturbation
propagates much further without the sink.
**The gap: the ⟨bos⟩-removal experiment reports benchmark drops (Table 3) and attention maps
(Fig. 3), and never reports the generated text.** The closest sink-removal experiment in the
healthy-model literature does not say whether the model repeats.

**Guo et al. 2024, *Active-Dormant Attention Heads* (preprint).** read §"active-dormant
mechanism", Claim 1, §3.1, Fig. 8. **The one paper that shows sink strength is
input-dependent in a real LLM.** LLaMA-2-7B-Base **L16H25** is *"dormant (i.e., an attention
sink) on samples from Wikipedia, which resemble prose, and active (i.e., not an attention
sink) on samples from GitHub, which resemble code"*; zeroing the head significantly hurts
loss on GitHub and not on Wikipedia (Fig. 8b). Claim 1: dormant = dominant weight on ⟨s⟩,
minimal value added to the residual. In the BB toy task, Fig. 2c shows the ⟨s⟩ token has the
**smallest value-state norm** ("value-state drain").
*Inference:* this is the existence proof the brief's hypothesis needs — sink mass at the head
level **does** vary with input in a healthy model, by a large margin, and covaries with
whether the head is doing work. It is measured across *domains*, not across positions within
a generation, and never against repetition.

**Gu et al. 2025, *When Attention Sink Emerges in Language Models* (ICLR 2025).** read §3.2
(metric), §3.3, §3.4, App. C.1 pointer. **This paper contains the single most directly relevant
measurement in the whole search and neither literature cites it for this.**
- The metric: `Sink^ε_k = (1/L)Σ_l (1/H)Σ_h 1[α_k^{l,h} > ε]`, i.e. the **percentage of heads**
  whose mean attention to position k exceeds ε. §3.2: *"There is no principal way to find an
  optimal threshold and we only use this metric to quantify the emergence of attention sink
  empirically"*; they choose ε = 0.3 and fix T = 64 for comparability.
- §3.3, Table 1 (Left), `Sink^ε_1` (%), input = natural / random tokens / **one token repeated
  T times**:
  GPT2-XL 77.00 / 70.29 / **62.28**; Mistral-7B 97.49 / 75.21 / **0.00**;
  LLaMA-2-7B Base 92.47 / 90.13 / **0.00**; LLaMA-3-8B Base 99.02 / 91.23 / **0.00**.
  Caption: *"Even with random sequence as input, there still exists an obvious attention sink.
  But with repeated tokens, the attention sink disappears for Mistral/LLaMA models."*
- The explanation (§3.3, proved in App. C.1 **[the appendix proof itself unopened]**): for LMs
  with NoPE / relative PE / ALiBi / RoPE, if the first T tokens are identical then their hidden
  states are identical; *"They all have massive activations, thus dispersing the attention
  sink."*
- §3.3 also reports that across the Pile's 17 domains, *"input domains have negligible effects
  on our attention sink metric"* (App. C.2 **[unopened]**).
**Why this matters more than anything else in §1.** In a **healthy, unmodified** LLaMA-2-7B or
Mistral-7B, a repeated-token input drives the position-0 sink metric to **exactly zero** — the
same observable the lesion produces (Su-2025 Fig. 9: D_sink ≥ 90%; Fig. 8: sinks gone). So
"repetition co-occurs with loss of the position-0 sink" is **already established in healthy
models, and by a mechanism that has nothing to do with the sink being damaged**: every repeated
token acquires the massive activation, so the mass disperses rather than disappearing.
*Inference, and it is the central one in this file:* a position-0 sink-mass drop measured during
natural repetition is **not evidence for the brief's hypothesis**. Distinguishing "the sink was
lost" from "the sink was shared out" requires measuring the **total** sink-like mass — attention
to *all* massive-activation-bearing positions, plus their residual norms — not the concentration
at position 0. Gu's metric and Su's D_sink both measure concentration at position 0 and would
report the two cases identically.
**Onset status: this is a static prompted measurement (T = 64 identical tokens), not a
generation-time series, and there is no pre-onset condition.** It also formally *reconciles*
Gu with Yona: Yona says repeated tokens acquire BoS-like norms and attention; Gu says that is
precisely why the position-0 concentration metric collapses. Same fact, opposite-sounding
sentences, because the metrics differ.

**Queipo-de-Llano, Arroyo, Barbero et al. 2025, *Attention Sinks and Compression Valleys …
Two Sides of the Same Coin* (ICLR 2026).** read abstract, §1, §2 related work. Proves that a
single token whose norm dominates *necessarily* creates a dominant singular value in the
representation matrix, hence low matrix entropy; documents sinks and compression valleys
emerging simultaneously in middle layers across 410M–120B. Related work §2 states plainly
that Sun-2024, Yona-2025 and the rest *"none of these works link this phenomenon to
representational structure or a unified theory of information flow in LLMs."*
**Methodological consequence, and it corrects OP2 in `notes_repetition_degeneration.md`:**
because their compression is *caused by* the BOS norm, removing the massive activation
**raises** matrix entropy / effective rank while (per Barbero) **increasing** over-mixing
among the non-sink tokens. Those two measures move in opposite directions under the same
lesion. Any effective-rank measurement must therefore **exclude the sink position**, or it
will report the sign backwards.

**Fesser 2026, *A Unifying View of Sinks: Two Algorithms* (preprint).** read §"Hypothesis 2",
Lemmas 3–4. Under the *broadcast* sink hypothesis (v_s carries information), Lemma 3 gives a
rank-1 attention output and Lemma 4 bounds the token-representation variance as
(1/n)Σ‖x^{ℓ+1}_i − x̄^{ℓ+1}‖² ≤ (1/n)Σ‖x^ℓ_i − x̄^ℓ‖² + O(‖v_s‖²·V(A_s)) — i.e. a broadcast
sink **increases** token alignment. Under the *NOP* hypothesis v_s ≈ 0 and this term vanishes.
*Inference:* Barbero's "sinks prevent over-mixing" and Fesser's "broadcast sinks increase
similarity" are not contradictory but they apply to different sink types, and the type is an
empirical question per head. Guo's value-state drain says LLM sinks are mostly the NOP kind;
Binkowski (below) says the *hallucination-predictive* ones are the large-‖v‖ kind.

**Binkowski, Adamczewski & Kajdanowicz 2026, *Attention Sinks as Internal Signals for
Hallucination Detection* (arXiv:2604.10697v2, preprint).** read abstract, §1, §2.1–2.2,
App. "Important Head Selection". SinkProbe = ℓ1-regularised logistic regression on top-k sink
scores. The framing sentence is the one that matters here: hallucinations are entangled with
sinks, *"indicating a transition from distributed, input-grounded attention to compressed,
prior-dominated computation"*, and the classifier *"preferentially relies on sinks whose
associated value vectors have large norms"*.
**Note the sign.** Their story is **more** sink concentration under failure, not less — the
opposite of the brief's hypothesis. Being precise: the appendix says *"positive coefficients
indicate that higher sink scores increase hallucination probability, while negative
coefficients indicate the opposite"*, and both signs are retained, so the paper does not
establish a single direction; only the *framing* is "more sink = prior-dominated". Also:
hallucination ≠ repetition, and this is a decoder-only LLM QA setting.

### 1.4 The 2025–2026 onset literature — found by search, and it does not cite the sink literature

Four papers found on 2026-09-05 that were not in the folder. Together they are the first
work to measure *anything* internal across the onset boundary, and **none of them measures a
sink statistic**; two of them exclude sinks by construction.

**Mahaut & Franzon 2025, *Repetitions are not all alike: distinct mechanisms sustain repetition
in language models* (arXiv:2504.01100, v1 2025-04-01, v2 2025-11-04, preprint).** read §1.2,
§2, §3.1–3.3, §4.1–4.2, §5, §6. **This is the closest existing test of the brief's question,
asked of a different pair of conditions, and its answer is "different mechanisms."**
- Setup (§2): Pythia-1.4B, **10 training checkpoints**, all layers, greedy decoding, A30s
  (~10 h per layer per checkpoint). 1,000 human sentences from Minipile truncated to 32
  tokens; continuations to 1,000 tokens. **Natural** prompts = those that spontaneously fall
  into a cycle of length > 1 persisting to the end (excluding EOT loops); if the cycle did not
  start immediately, the model's own output up to the first cycle token is appended, so the
  prompt loops **from the first generated token**. **ICL** prompts = sequences that did *not*
  loop, then repeated at least once in the prompt. Cycle 0 = the prompt; cycle 1 = one
  iteration; "from cycle 2 the model has a pattern to detect".
- §3.1 Fig. 1: natural repetition is present **from the first training checkpoint** and about
  half the finally-repeating prompts have been repeating since then; ICL repetition is **absent
  at step 0** and grows, appearing around step 10k.
- §3.3 Fig. 2 (tuned-lens-style affine probe per head, Ortu et al. contrastive score): in the
  ICL condition a **few heads specialise** (heads 2–8 of layer 7 at the last checkpoint) with
  increasingly polarised contrast as cycles accumulate. **In the natural condition, at the last
  checkpoint, no head exceeds the ±0.5 × 10⁻⁶ noise band, and contrast does not change as the
  cycle count rises.** MLP contribution (App. A) is likewise much weaker for natural repetition.
  *"Repetition in the natural setting does not rely on the same circuits … natural repetition
  emerges early and lacks a defined circuitry."*
- **§4.2 Fig. 3 is the result that bears hardest on this project.** Attention to each token
  class, as a ratio to that class's share of the dataset: **newline tokens receive 9.9× uniform
  in the natural condition and 3.1× in ICL**; content words receive **less than half** uniform
  in the natural condition. Fig. 4: with newlines removed, natural-condition attention shifts to
  *other* semantically empty structural/tokenizer tokens and brackets, while ICL attention
  shifts to function words and numbers.
- **§4.2 also reports the causal test, and it is negative:** *"We make comprehensive causal
  tests to determine whether it is the newline tokens in themselves that cause repetition
  (adding newlines, removing newlines, forcing attention heads to give additional / less weight
  to newline). All these experiments lead to 0 changes on our experimental subset of 100 prompts
  of each dataset."*
- §5 Fig. 5: median next-token entropy at the first token of a cycle **falls** with cycle number
  in both conditions, slightly faster for ICL; but the natural distribution is much wider, with
  outliers above 6 at cycle 4, i.e. *"natural repetitions start with lower confidence"*. Their
  reading: ICL repetition is overconfidence, natural repetition is failed retrieval.
**Why this matters more than the sink-side papers.** Newline is one of **Sun-2024 §4.1's
massive-activation-bearing delimiter tokens** (their Figs. 5–6 are labelled "The", ".", "\n").
So the one careful measurement of where attention goes during *natural* repetition finds it
piling **onto** a sink-like delimiter, not away from position 0 — the same sign as Binkowski
(E9), the opposite of the brief's hypothesis. *Inference, and it is mine not theirs:* Mahaut
never connects newline attention to massive activations or sinks — the words "attention sink",
"BOS" and "first token" do not appear — so the identification of newline-as-sink-token is my
inference across two literatures, and the right next step is to check whether the natural
condition's newline tokens actually carry massive activations in Pythia-1.4B.
**Onset status: no.** §3.2.1 analyses *"the first four cycles generated after each prompt"*,
and the Natural prompts are constructed to loop **from token 1**, so there is by design no
pre-onset segment. The training-checkpoint axis is a different axis entirely.
**Also note the standing methodological warning:** two behaviourally identical repetitions were
already shown, once, carefully, to have qualitatively different internals. "They look the same
so they are the same" is exactly the inference this paper refutes.

**Duan, Pang, Wei, Duan, Tian, Xu, Deng, Yin & Cheng 2026, *Circular Reasoning: Understanding
Self-Reinforcing Loops in Large Reasoning Models* (arXiv:2601.05693, 2026-01-09, preprint).**
read §3.1–3.3, §4, Figs. 3–8. **The only paper found that measures an attention quantity
*before* the first repeated token — and it explicitly excludes the sinks from that measurement.**
- Models: DeepSeek-R1-Distill-Qwen2.5-14B and Qwen3-8B as primary case studies, on their own
  LoopBench (numerical loops and statement loops).
- §3.1: at onset, maximum logit rises sharply and **entropy drops toward zero** (Fig. 3); the
  cosine similarity between activation vectors of identical tokens across cycles saturates near
  **1.0** with vanishing norm differences in deep cycles (Fig. 4). A hidden-state classifier
  separates loop from non-loop states with AUC reported as consistently high (Table 2).
- §3.2 Fig. 5c: **"semantic circularity significantly precedes explicit textual repetition"** —
  K-Means (K = 200) over per-sentence mean final-layer hidden states, plotted as cluster labels
  against sentence ID, shows the cycle in cluster space starting before the surface repetition.
- **§3.2 Fig. 6 is the pre-onset attention measurement.** They plot the attention proportion
  assigned to preceding *identical* high-entropy sentence-initial tokens ("But", "Wait",
  "Alternatively", "Maybe", "Therefore") against token position, with the repetition-start line
  marked, and state: *"A distinct concentration of attention emerges prior to repetition onset."*
  Fig. 7 quantifies this over **50 loop and 50 non-loop samples** across window sizes 64–512:
  both the density and the attention share of these tokens **peak during the "Repetition Onset"
  window** (a fixed window *preceding* repetition), above both the pre-repetition history and
  the non-looping baseline.
  **The figure caption reads: *"Note: Attention sinks and recent tokens (last 128) are
  excluded."*** So the one pre-onset attention measurement in the literature has the sink
  subtracted out before plotting.
- §3.3 Fig. 8: the loop *persists* via a **"V-shaped" attention pattern** — mass concentrated
  on *"the initial attention sinks and the most recent tokens"*, citing Xiao 2023. Once repeats
  saturate the recent window, the **local arm** rises abnormally (Fig. 8b). The sink arm is
  invoked descriptively; **its mass is never measured across onset.**
- §4: CUSUM over prediction scores gives early prediction, reported at a lead of roughly
  40 sentences / 1,500 tokens before repetition starts.
- §3.1: higher temperature reduces but does not eliminate loops; numerical loops persist at
  T = 1.0 (their Table 7).
**Onset status: yes, uniquely — but of the wrong quantity, and with the sink deliberately
removed from it.** This is the paper whose Fig. 6 the project should re-run **without** the
exclusion.

**Xu, Wu, Shi, Cui, Liu, Li, Ma, Liu, Zhu & Xu 2026, *LoopGuard: Breaking Self-Reinforcing
Attention Loops via Dynamic KV Cache Intervention* (arXiv:2604.10044, 2026-04-11, preprint).**
read abstract, §1, §5.1–5.2, App. B. Models: Qwen3-1.7B, Llama-3.2-1B.
- Diagnosis: **head locking** — *"a subset of heads locks onto a narrow suffix of the history"*;
  App. Fig. 5 shows *"narrow, vertically aligned stripes"* ("barcode-like") across layers and
  heads under repetition collapse. Qualitative; no head-level statistic is reported.
- The second-order effect is the useful one for the compression side of this project:
  attention-based KV-cache importance scores become **spuriously high for repetitive tokens**,
  so eviction policies like H2O preferentially *keep* the loop and discard diverse context —
  cache management amplifies repetition (Fig. 1b).
- Detection (§5.2): an online changepoint on **windowed output diversity** — type–token ratio
  ≤ 0.2 and compression ratio ≤ 0.12 over W = 256, with a debounced trigger and near-max token
  budget. LoopGuard then prunes repetitive tail spans while *"preserving anchor tokens"*,
  where the anchor set I_anchor is **a short prefix** (§5.3), i.e. the StreamingLLM initial
  tokens; StreamingLLM is used as a baseline (§6).
- Result: loop incidence reduced by >90 percentage points under a fixed cache budget.
**Onset status: no.** The trigger is a *surface* diversity collapse measured after it happens,
not an internal precursor. **The paper never mentions attention sinks, BOS, or massive
activations except when citing StreamingLLM as a cache baseline** — despite its whole thesis
being about attention collapse during repetition, and despite its own fix preserving the sink
prefix. *Inference:* that silence is itself a finding about how disjoint these two literatures are.

**Marwah, Garimella, Pallagani, Jain, Stewart & Sheth 2026, *Cognitive Fatigue in Autoregressive
Transformers: Formalization and Measurement* (ICML 2026 — the PDF footer reads "Proceedings of
the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026";
arXiv:2605.30981).** read §3.1–3.2, §6.1, §7.2, §7.4, Figs. 4–5, Tables 3 and 5. Nine models,
1B–13B. **The only per-decoding-step internal time series that is validated against repetition.**
- Three online signals (§3.1), all per token, constant overhead:
  **A_t** = *"mean last-layer attention weight from the current token to the tokens in the
  initial prompt slice"*; **D_t** = ‖h_t − h_0‖₂ against the last prompt token's hidden state;
  **E_t** = Shannon entropy of the next-token distribution, scored against a calibrated healthy
  band. Fatigue Index FI_t = w_A φ_A(A_t) + w_D φ_D(D_t) + w_E φ_E(E_t) with **fixed, hand-set**
  weights w_A = 0.40, w_E = 0.35, w_D = 0.25 (§3.2 states the ordering is a domain prior and
  that weight sensitivity and learned weights *"remain open extensions"*).
- §7.2 Table 3: FI–repetition Spearman ρ > 0.8 over the full generation (up to 120 tokens),
  the headline ρ = 0.94. **But over the first 20 tokens ρ ≈ 0.4**, and the paper says so
  explicitly: *"FI is designed to track trajectories rather than provide strong single-step
  early warnings."*
- §7.4 Table 5, HotpotQA AUROC for discriminating severe degeneration: **FI 0.978, Drift alone
  0.951, and Attention alone 0.31** — below chance. The paper keeps attention at the highest
  weight anyway, on the domain-prior grounds above.
- **§6.1 Fig. 4 is a direct lesion-vs-natural comparison, in one model, matched prompts and
  seeds: FP16 vs 4-bit NF4.** *"Attention to prompt and embedding drift trajectories remain
  nearly identical across precisions … Entropy, however, exhibits deeper and more variable
  collapse under 4-bit NF4, suggesting that reduced precision primarily destabilizes predictive
  calibration."* FI onset also accelerates under longer contexts and mid-positioned evidence.
  **Read the figure note before citing it:** *"the 4-bit NF4 entropy trajectory is reconstructed
  from paper-reported statistics due to a plotting error in the original experimental run;
  attention and drift curves are extracted directly from experimental output."* So the
  *attention* half of the comparison is measured and the *entropy* half — which carries the
  claim — is reconstructed.
- Own limitation (§7.2): the repetition ratio is not independent of the entropy signal, because
  repetitive outputs are low-entropy by construction.
**Onset status: yes for the trajectory, weak at the step level (ρ ≈ 0.4 in the first 20
tokens).** A_t contains the sink (position 0 is in the prompt slice) but is a last-layer,
whole-prompt average, so it cannot separate sink mass from prompt-content mass, and it is the
weakest of the three signals.

### 1.5 The three papers that actually compare a damaged model with its undamaged self

Also found on 2026-09-05. None of them mentions attention sinks, massive activations, or super
weights; all three are nevertheless the only same-model lesion-vs-intact repetition comparisons
that exist.

**Lee, Park, Jang, Ryu, Kim, Lee, Suh & Kim 2026, *FOCUS & RePAIR: Mitigating Text Degeneration
via Token-Level Guidance for Pruned Large Language Models* (arXiv:2608.26676v1, 2026-08-27; the
keyword line says ICML but no acceptance is printed — treat as **preprint**).** read §1, §3.1–3.4.
**This is the same-model comparison the project is missing, done for pruning.**
- Setup: 200 tokens generated from a 50-token WikiText-103 prefix, Llama, pruned models
  LoRA-finetuned on Alpaca afterwards; CREP = the percentage of sequences in which one dominant
  N-gram covers more than a threshold fraction (30% in their App.) of the tokens (Eq. 1–2).
- **Table 1**, repetition rate, unpruned / width-pruned / depth-pruned:
  top-p 0.9 sampling **5.9% / 12.4% / 15.4%**; greedy **26.6% / 63.1% / 63.7%**.
  §1: *"the degeneration phenomenon becomes more severe after pruning"* while *"perplexity and
  task accuracy remain largely unchanged"*.
- **§3.1 Table 2, the entry-event result.** Locate the first loop onset (earliest position where
  the dominant N-gram begins to recur over a contiguous span), then **replace only the first two
  onset tokens** with an alternative from the top-2 candidates at the same prefix, and regenerate:
  CREP falls **5.9% → 0.7%** (sampling) and **26.6% → 10.8%** (greedy). *"degeneration often
  depends on a small subset of loop-sensitive contexts near the first onset."*
- §3.2–3.3 decomposition: entry risk R_T (hitting time τ_L of the recurrent context set L) and
  persistence ρ̄, where persistence is governed by the **escape mass** inside the nucleus set —
  *"reducing persistence requires increasing escape mass inside the nucleus, not merely
  increasing entropy in the full vocabulary."*
- §3.4, the two named pruning distortions: **tail leakage** (the student allocates mass to
  teacher-suppressed tokens, which occasionally enter the nucleus, raising entry risk) and
  **alternative collapse** (*"Pruning-induced representational loss can distort these near-ties,
  producing a single dominant continuation and suppressing competing alternatives"*, lowering
  escape mass and raising persistence).
- App. A replicates Xu-2022's saturating self-reinforcement curve on the **pruned** model.
**Two things this settles and one it sharpens.** It settles that damage raises the repetition
rate on the same model with perplexity roughly held — the ×2.4 in both decoders is the number to
quote. It settles that the *onset* is a discrete, locally reversible event, not a slow drift.
And it sharpens the entropy question: **alternative collapse is a *sharpening* account**, i.e.
mild pruning makes the model *more* confident on the loop token, whereas super-weight zeroing
*flattens* the distribution and moves the argmax (Yu §3.2, "the" at 9.0%). *Inference:* if both
are true, they are different points on a severity axis, not one mechanism — and that is a
testable dose–response prediction, since it says the sign of the entropy change should **flip**
somewhere between LoRA-healed structured pruning and zeroing one super weight.

**Roll, Kries, Gwilliams & Shain 2026, *Artificial Aphasias in Lesioned Language Models*
(arXiv:2605.16222, "Preprint" on p.1).** read §Results, §Limitations, Fig. 4, App. summaries.
Zero-ablates whole parameter matrices (attention Q/K/V/O and FFN up/gate/down) across depth and
severity in five 1B-scale LMs and scores **112,426 generations** with a 21-symptom Text Aphasia
Battery, against a deterministic unlesioned condition-level baseline (2,528 lesions).
- **FFN-Gate lesions give the highest repetition-loop symptom rate**; FFN lesions exceed
  attention lesions on Perseverations (+1.23) and on the "Other" category, which is explicitly
  *"repetition-loop, stereotypy/automatism, and off-topic symptoms"*. The FFN-vs-attention
  contrast survives in 77/97 model–prompt pairs, 18/20 prompts, 4/4 severities, 5/5 variants,
  3/3 families, and survives removing the repetition-loop symptoms altogether.
- Scale: at 7B, **single-layer ablation produces no detectable symptoms**; the contrast returns
  only with multiple simultaneous layer ablations, where FFN-Gate reaches a repetition-loop rate
  of 0.41 at 25% severity against attention-K at 0.02.
- **Fig. 4B is the decoding result the brief asks about.** A 5 × 5 temperature × repetition-penalty
  grid on Gemma-3-1B-IT: FFN-Gate repetition-loop rates are *"concentrated at
  low-temperature/no-penalty settings and suppressed by higher repetition penalties; they are
  **zero at T = 0.7, repetition penalty 1.2**."* §Limitations concedes that primary inference uses
  greedy decoding, *"which can amplify repetition loops"*, and that repetition-heavy symptoms are
  decoder-sensitive while the attention/FFN contrast is not.
- Instruction tuning cuts overall symptom burden by 81% while leaving the *profile* highly
  correlated (r̄ = 0.857).
**Why this matters.** It is the only large-n characterisation of lesion-induced *text*, and its
answer to "is lesion repetition suppressible by the same knobs that suppress natural repetition?"
is **yes, completely, at these severities**. It also localises the effect to FFN-Gate, which is
the same subsystem as Oh-2024's `W_gate`/`W_up` massive weights — Roll does not make that
connection, and it is my inference, not theirs. Caveats: whole-matrix ablation at 25–100%
severity is a far coarser lesion than one scalar; 1B models; one scored-by-LLM instrument.

**Mohammadshahi, Nikoulina, Berard, Brun, Henderson & Besacier 2022, *What Do Compressed
Multilingual Machine Translation Models Forget?* (arXiv:2205.10828v4; Findings of EMNLP 2022).**
read §4.1.2–4.1.3 and Tables 3–5 (checked with `pdftotext -layout`, because the two-column
extraction mis-associates the two λ tables). M2M-100 under 8-bit PTQ and 30%/45% magnitude
pruning on FLORES-101.
- Table 3, off-target rate base → compressed: **5.9 → 13.7 (+7.8)** for 30% pruned,
  **6.4 → 30.3 (+23.9)** for 45% pruned, **5.2 → 17.5 (+12.3)** quantized.
- Hallucination is measured by a relative cross-attention alignment ratio λ = var_comp/var_base
  (Eq. 2, attention averaged over all layers and heads); high variance = target attends a very
  small subset of source tokens. **Table 4, losing pairs: λ = 2.95 / 3.01 / 1.96** (n = 1,312 /
  7,192 / 221). **Table 5, winning pairs: λ = 0.42 / 0.15 / 0.52** (n = 863 / 1,455 / 308) —
  §4.1.3: *"a lot of them are matching cases where the baseline model generates hallucinations,
  while the compressed model generates acceptable translations"*, attributed to compression
  dropping memorised noisy samples.
- Their Table 3(c) compressed-model example is itself near-repetitive: *"It is believed to have
  been one of the earliest inhabitants of this place, and it is believed to be one of the oldest
  inhabitants of this place."*
**It measures hallucination and off-target rate, not repetition or n-gram statistics**, so it
does not close the MT gap; it does show that the same-model, both-directions comparison is doable
in NMT, and that the *undamaged* baseline is worse on a non-trivial subset — a control the
LM-side lesion literature never runs.

---

## §2 The evidence for and against sink/constant involvement in *natural* repetition

Read this table as: nobody has done the experiment; these are the fragments that bear on it.
"Nat." = evidence concerns natural (uncaused) repetition. "Les." = lesion-induced.

| # | Evidence | Nat./Les. | Direction | What it actually establishes | What it does not |
|---|---|---|---|---|---|
| E1 | Cancedda §6.1 + App. D Tables 5–7: Ψ (dark-subspace) filtering at L3–L5 of LLaMA-2-13B → `"the, the, the, …"` | Les. | **for** | Removing the sink-supporting vector in a *healthy, untrained-on* model causes loops, and at **5% removal** the matched random-basis control does not loop | At 20% removal the random control loops too; no dose–response curve; no repetition metric, only sample generations |
| E2 | Su-2025 §5.2 Fig. 9 + §4: SE pruning → D_sink ≥ 90% **and** repetitive Math-500 output | Les. | **for** | The two co-occur under one intervention, in an MoE, at 30B | Never tested for causation; D_sink is defined only across two model versions; no natural condition |
| E3 | Su-2025 App. C: pruning final-layer **outlier** experts → no repetition, small PPL change | Les. | **for (as control)** | Repetition is specific to the early-layer sink-supporting object, not to removing any big activation | Two models; qualitative |
| E4 | Guo-2024 §3.1 Fig. 8: LLaMA-2-7B L16H25 is a sink on Wikipedia and not on GitHub | Nat. | **for (enabling)** | Sink mass at the head level is strongly **input-dependent** in a healthy model — the precondition the hypothesis needs | Domain-level, not step-level; no repetition; one head, one model. **Gu §3.3 reports the opposite at model level** — "input domains have negligible effects on our attention sink metric" over the Pile's 17 domains. Both can hold: Guo's is one head, Gu's is a head-fraction aggregate. Measure per head |
| E5 | Cancedda §7 Fig. 15: 12,236/404,748 HMLV token–layer pairs have elevated U-Dark ratio from ~L13 | Nat. | **weakly for** | Ordinary tokens in ordinary text acquire sink-like spectral signatures | Their cosine similarity to the BoS residual is ~0 and matches the population; Cancedda declines to call them sinks; no generation |
| E6 | Yona §4.1–4.4, Table 1: repeated tokens acquire BoS-like norms and attention via the same sink neurons | Nat.-ish | **for, but reversed arrow** | Repetition **produces** sink-like structure at the repeated tokens | Needs **450–4000** repeats; Fig. 6 shows it fails for some tokens; prompted, not spontaneous |
| E6b | **Gu-2025 §3.3 Table 1**: repeated-token input drives `Sink^ε_1` to **0.00%** in Mistral-7B, LLaMA-2-7B, LLaMA-3-8B (77.00 → 62.28 in GPT2-XL), in **unmodified** models | Nat. | **against (confound), decisive** | Repetition alone destroys the position-0 sink metric in a healthy model, because every repeated token gets the massive activation and the mass **disperses** | Static prompted input (T = 64 identical tokens), not generation; no pre-onset condition; the App. C.1 proof is unopened |
| E7 | Stolfo §6.2: attention-to-BOS is the *dormant* state; induction heads firing = attention off BOS | Nat. | **against (confound)** | A sink-mass drop at repetition onset is expected *even if the sink mechanism is perfectly healthy* | Only GPT-2 Small, 3 heads; measured as an intervention, not as an observed time series |
| E8 | Wang 2025 Table 1 + Fig. 4: entropy decay rate 0.3 (normal) → 1.7 (toxic) | Nat.-ish | **against** | Natural/induced repetition is a **sharpening** of the distribution | Lesion repetition is a **flattening** with a moved argmax (Yu §3.2). One model, unrefereed |
| E9 | Binkowski 2026: hallucination ⇒ "compressed, prior-dominated" attention, more sink mass | Nat. | **against (sign)** | The one paper linking a natural failure mode to sinks reads the sign the other way | Both coefficient signs retained; hallucination ≠ repetition |
| E10 | Barbero §3.2, Table 3: removing ⟨bos⟩ from Gemma-7B raises mixing and drops benchmarks | Les. | **untested** | The cleanest sink removal in a healthy model | **The paper never reports the generated text.** Whether Gemma repeats without ⟨bos⟩ is, as far as I found, unpublished |
| E11 | Owen-2025 §3.1, §4.1 + Sun-2024 Table 3: set-to-mean harmless in 16/16, set-to-zero catastrophic in 9/16 | Les. | **neutral, often misread** | The massive activation's **value** carries no input-specific information | It says **nothing** about whether the *attention mass drawn to* that token varies per input. See §5.4 |
| E13 | **Mahaut §4.2 Fig. 3**: in natural repetition, newline tokens get **9.9× uniform** attention (3.1× in ICL); content words < 0.5× | Nat. | **against (sign), or for with the arrow flipped** | Attention during natural repetition piles **onto** a delimiter token — one of Sun-2024 §4.1's massive-activation-bearing token types | Mahaut never links newline to massive activations; the identification is my inference. Pythia-1.4B, one model |
| E14 | **Mahaut §4.2 causal test**: adding newlines, removing newlines, and forcing heads to weight newline up or down all give *"0 changes"* on 100 prompts per condition | Nat. | **against** | Manipulating attention to the sink-like token does **not** change whether the model repeats | n = 100 per condition; only newline, not position 0 or BOS; effect size not reported, only "0 changes" |
| E15 | **Mahaut §3.3 Fig. 2**: no attention head exceeds the ±0.5×10⁻⁶ contrast noise band in the natural condition, at any cycle count | Nat. | **against** | Natural repetition has **no head-level circuit at all**, so a head-level attention reallocation story has no support | Tuned-lens-style probe, contrast metric; absence of signal in one probe is not absence of mechanism |
| E16 | **Duan §3.2 Figs. 6–7**: attention to preceding identical high-entropy tokens peaks in a window *preceding* onset, over 50 loop vs 50 non-loop samples | Nat. | **neutral, and the key precedent** | A pre-onset attention signature **exists** and is detectable with a matched non-looping control | *"Attention sinks and recent tokens (last 128) are excluded"* from the measurement. Reasoning models, high-entropy pivot tokens |
| E17 | **Lee-2026 Table 1**: repetition rate unpruned → pruned, same Llama, ppl ≈ unchanged: sampling 5.9 → 12.4/15.4%, greedy 26.6 → 63.1/63.7% | Les. vs Nat. | **for (co-location)** | Damage **raises the rate of the natural phenomenon** ~2.4× rather than producing a different one | Structured pruning + LoRA healing, not a scalar lesion; no internal measurement; sinks never mentioned |
| E18 | **Roll-2026 Fig. 4B**: FFN-Gate lesion repetition-loop rate is **zero at T = 0.7 with repetition penalty 1.2** | Les. | **for (co-location)** | Lesion repetition is suppressed by exactly the knobs that suppress natural repetition | Whole-matrix ablation at 25–100% severity in 1B models; LLM-scored symptom instrument |
| E19 | **Marwah §6.1 Fig. 4**: FP16 vs 4-bit NF4, matched prompts and seeds — prompt-attention and drift trajectories *"nearly identical"*, only entropy collapses deeper | Les. vs Nat. | **against** | A numerical lesion changes **calibration, not attention allocation** | The entropy curve is *"reconstructed from paper-reported statistics"*; A_t is a last-layer whole-prompt mean, not a sink statistic |
| E12 | Fu 2021 Cor. 1.2; Li 2023 | Nat. | **against** | A complete account of natural repetition exists that involves no attention quantity at all | Markov-model theory; validated only through an encoding intervention |

**Summary of the table.** Every "for" row is on the lesion side or is an enabling condition;
every row that touches natural repetition directly is either neutral, reversed in arrow (E6),
reversed in sign (E9), or offers an innocent explanation for the predicted signature (E7, E6b).
**No paper measures sink mass, attention entropy, or the constant's contribution during
spontaneous degeneration in an unmodified model.**

**Three of the new rows point the same way.** E13/E14/E15 are one careful study (Mahaut) saying
that during natural repetition attention goes *to* semantically empty, sink-like tokens; that
forcing it there or away changes nothing; and that no head has a repetition-favouring
contribution at all. E19 says a numerical lesion leaves prompt attention untouched and only
breaks calibration. Against that, E17 and E18 say damage and natural degeneration are at least
*co-located* — damage multiplies the rate of the same measured phenomenon, and the same decoding
knobs suppress both.

**And the naive version of the brief's prediction is already dead.** E6b shows that in a healthy
Mistral-7B or LLaMA-2-7B, repetition alone takes `Sink^ε_1` from ~92–97% to **0.00%**. So
observing "sink mass at position 0 collapses during natural repetition" would confirm nothing:
it is the *expected* consequence of the repeated tokens themselves, via Gu's dispersal argument
and Yona's norm-acquisition result, in a model with a perfectly intact sink mechanism. Two
things survive this and are what the hypothesis has to be re-stated around:
(a) whether the collapse happens **before** the first repeated token — Gu's and Yona's mechanisms
both require the repeats to already exist, so a *pre-onset* drop is not explained by either; and
(b) whether **total** sink-like mass (attention to every massive-activation-bearing position,
weighted by their residual norms) falls, versus merely being redistributed — the lesion removes
it, dispersal conserves it.

---

## §3 The four mechanistic hypotheses and what each predicts *before* onset

Stated so they are separable by measurements on healthy models. "Pre-onset" means: at the
decoding step that emits the first repeated token, and at the ~10 steps before it, compared
against matched positions in generations from the same model on the same prompts that do
**not** go on to loop.

### M-A. Over-copying after sink loss (the brief's hypothesis; Cancedda's mechanism)
*Claim.* On some inputs the sink weakens; heads that should dump attention on it instead
distribute it over recent context; copying dominates; the output falls toward the frequency
prior and locks.
*Pre-onset prediction, healthy model.* Sink mass (Gu's `sink rate`, ε = 0.3) **falls**, and
does so **model-wide across heads**, in the ~5–10 steps before the first repeat; per-head
attention entropy falls (sharper, more local); attention to recent positions rises.
*Lesion signature it must match.* Su-2025 Fig. 9: D_sink ≥ 90% at **every** layer; Su Fig. 8:
sinks disappear completely in the visualised heads. Cancedda Fig. 11 middle: the effect is
carried by the Token-0 residual specifically.
*Discriminating feature:* **model-wide**, **not head-selective**, and — critically — present
**before the first repeat**, since Gu (E6b) and Yona (M-D) both explain a *post*-onset drop
without any sink damage.
*Strongest support:* E1 at low dose, E2, E3. *Strongest problems:* E7 predicts the same
observable from an intact sink in induction heads; **E6b predicts the same observable from an
intact sink model-wide** once repeats exist. M-A therefore only survives as a *pre-onset*,
*total-mass* claim.

### M-B. Frequency-prior fallback (Yu Fig. 5; Puccetti §4.1; Meister 2023; Stolfo's token-frequency neurons)
*Claim.* The contextual signal degrades, the readout defaults to the unigram prior, whose mode
is "the" / "." / ",", and a high-inflow function word then cycles.
*Pre-onset prediction, healthy model.* **Output entropy rises** and KL to the unigram prior
**falls** before onset; the repeated token is a high-frequency function word largely
independent of the prompt.
*Lesion signature.* Yu §3.2: "cold" 81.4% → **"the" at 9.0%** — flat and re-pointed;
stopword probability up 2×–10×.
*Discriminating feature:* **entropy up, argmax moved, mass toward frequent tokens.**
*Note the direct conflict with M-C, which predicts entropy down.* Also note the unresolved
dispute the earlier notes flag: Macocco-2025 finds decoder ablations move toward **rarer**
tokens, against Puccetti's encoder result.
*New support from the healthy side:* Mahaut §5 Fig. 5 finds natural repetitions *"start with
lower confidence"* (wider entropy distribution, outliers above 6 at cycle 4) and §4.2 finds
attention avoiding content words — both read as failed retrieval. Mahaut's own §6 wording is
*"a fallback behavior that emerges when the model fails to retrieve or attend to relevant
information"*, which is M-B stated from the attention side rather than the readout side.
*But the lesion side now splits:* Lee-2026 §3.4's **alternative collapse** is a sharpening story
for pruning, while Yu §3.2 is a flattening story for a super weight. M-B as written matches the
super weight, not the pruned model.

### M-C. Self-reinforcement / induction lock-in (Holtzman Fig. 4; Xu §2.2; Wang 2025; Hiraoka)
*Claim.* The first repeat is arbitrary; induction heads and repetition neurons then amplify it
monotonically to a ceiling.
*Pre-onset prediction, healthy model.* **Nothing distinguishes pre-onset steps** — the
signature is entirely post-onset and monotone: induction-head logit share ↑ (Wang's τ_t),
repetition-neuron activation ↑ (Hiraoka Fig. 1), **entropy ↓** (Wang Lemma 3.5, Table 1:
decay rate 0.3 → 1.7), sink mass ↓ *only in induction heads* (Stolfo §6.2's dormant→active
switch).
*Discriminating features:* entropy **down** not up; sink-mass change **head-selective**, and
selective for heads with high prefix-matching score; onset itself unpredictable from internals.
*This is the hypothesis best supported for **prompted/ICL** repetition and it makes the lesion
look different, not the same.*
**Mahaut splits M-C in two, and this is the most consequential correction in this file.** In
Pythia-1.4B, ICL-induced repetition rides a dedicated head circuit that specialises over training
(§3.3 Fig. 2, layer-7 heads 2–8), while **natural repetition shows no head contrast above the
±0.5 × 10⁻⁶ noise floor at any cycle count**. So M-C as an account of *natural* onset has a direct
negative result against it in the one model where it was tested. Duan's §3.2 pre-onset attention
concentration is the counter-evidence, but on high-entropy pivot tokens in a *reasoning* model,
and it is an attention-mass measurement, not a head-circuit one. Note also that Xu-2022's
protocol and Hiraoka's Δ_n both start from an already-present copy, so neither tested this.

### M-D. Repeated tokens become sinks (Yona; the reverse arrow)
*Claim.* A run of identical tokens is misread by attention layer 1 as a sequence start; sink
neurons fire on it; the model then behaves as if the context had been truncated, and diverges.
*Pre-onset prediction, healthy model.* **None** — the mechanism is strictly post-onset, and by
Yona Table 1 it needs 450–4000 repeats before a *new* sink actually forms at the repeats.
**But its post-onset prediction fires far earlier than that in the concentration metric:** Gu
Table 1 already reads 0.00% at T = 64 identical tokens, because dispersal only needs the repeated
tokens to carry massive activations, not to have become full sinks.
*Observable that would confirm it in a natural loop:* the residual norm and the sink-neuron
activation at the *repeated* positions climbing toward the BoS value as the loop lengthens
(Yona Fig. 2), plus the first-detector neuron (MLP₀ gate 912 in LLaMA-2) turning on at those
positions.
*Discriminating feature:* the signature lives at the **repeated token positions**, not at
position 0. This makes M-D and M-A trivially separable by *where* you measure.

### What the four predict jointly — the actual decision table

| Measurement, pre-onset unless stated | M-A sink loss | M-B prior fallback | M-C lock-in | M-D repeats-as-sinks | Lesion (observed) |
|---|---|---|---|---|---|
| Sink mass at position 0, **pre-onset** | ↓ model-wide | — | — | — | n/a (lesion has no onset) |
| Sink mass at position 0, **post-onset** | ↓ model-wide | — | ↓ in induction heads only | ↓ to ~0 by dispersal (Gu Table 1) | ↓ ≥90% all layers (Su Fig. 9) |
| **Total** sink-like mass (all MA-bearing positions) | ↓ | — | ~flat | **~flat (conserved)** | ↓ |
| Output entropy | ↑ | ↑ | **↓** | ↓ | ↑, argmax moved (Yu §3.2) |
| KL to unigram | ↓ | ↓ strongly | ↑ (sharpens onto one token) | ↑ | not measured by anyone |
| Attention entropy per head | ↓ (local copying) | — | ↓ in induction heads | — | not measured by anyone |
| Repeated-token residual norm | — | — | — | ↑ toward BoS (needs ≫100 repeats) | n/a |
| Repeated token identity | function word | function word | prompt-dependent content | the prompted token | function word (anecdotal) |
| Survives switch to sampling | yes | yes | **largely no** (Xu §2.2; Holtzman Fig. 9) | yes | **partly no** — Roll Fig. 4B: 0 at T=0.7 + penalty 1.2; Lee Table 1: 63.7% greedy → 15.4% sampling |
| Head-level circuit detectable | some heads | no | **yes** (Mahaut ICL: L7 h2–8) | no | untested |
| Attention to sink-like delimiter tokens | ↓ | — | — | ↑ | untested |

*Inference:* the **entropy row is the cheapest decisive measurement**, because M-B and M-C
disagree on its sign — but the lesion side no longer sits cleanly on M-B's side: Yu's super
weight flattens, Lee's pruning sharpens (alternative collapse), and Marwah's 4-bit quantization
deepens the entropy collapse. So the honest statement is that **the entropy sign is a function of
lesion severity and type**, and measuring where it flips is itself the experiment. Natural loops
do sharpen: Mahaut §5 Fig. 5 and Wang Table 1 both report entropy falling with cycle count. So
the discriminating comparison is *natural sharpening vs the super weight's flattening*, and the
prediction that would unify them is that a **partial-dose** super-weight lesion sharpens before
it flattens.

---

## §4 The gap, stated precisely

Everything here is "not found in a search of `papers/`, arXiv, ACL
Anthology and the web" — not "does not exist".

1. **No per-decoding-step time series of any *sink* statistic during spontaneous degeneration
   in an unmodified model.** Four per-step internal series now exist and none of them is a sink
   statistic: Wang-2025's induction-head logit share τ_t (512 tokens, prompted repetition);
   **Duan §3.2 Fig. 6's attention to preceding identical high-entropy tokens, which states in
   its own caption that attention sinks are excluded**; Marwah's A_t, which is a last-layer mean
   over the whole prompt slice and so contains the sink without isolating it; and Xu-2026's
   windowed output diversity, which is a surface statistic. **This is the experiment.**
2. **Almost no measurement of anything internal strictly before the first repeated token, and
   the one that exists is not of a sink.** Xu's n = 1 and Hiraoka's 30-token pre-window both
   already contain the material that will be repeated; Mahaut's Natural prompts loop from
   token 1 by construction. **Duan §3.2 is the exception and the template**: 50 loop vs 50
   non-loop samples, a fixed window preceding onset, window sizes 64–512, both density and
   attention share reported — exactly the matched-control design this project needs, minus the
   sink. Marwah shows the trajectory is predictive over a full generation (ρ > 0.8) but weak in
   the first 20 tokens (ρ ≈ 0.4), so the effect at the step level may be small.
3. **Nobody has reported the generated text from the healthy-model sink removals.** Barbero
   removes ⟨bos⟩ from Gemma-7B and reports benchmarks (Table 3) and attention maps (Fig. 3);
   Queipo ablates the layer-0 MLP→BOS contribution and reports entropy and sink rate. Neither
   says whether the model loops. Cancedda is the only one who looked, and he looked
   qualitatively.
4. **No dose–response curve linking a sink statistic to a repetition statistic.** Cancedda's
   App. D is a two-point dose comparison read off sample generations; Su reports one dose;
   Yu sweeps the weight for accuracy only. Given that Cancedda's random-basis control also
   loops at high dose, a dose–response with a matched-magnitude control is the *minimum* needed
   to claim the sink is what matters.
5. **Nobody has separated sink *destruction* from sink *dispersal*.** Every published sink
   metric — Gu's `Sink^ε_1` (fraction of heads over ε at position 0), Su's D_sink (Eq. 7,
   position-0 attention ratio between two models), Barbero's sink rate (Gu's, ε = 0.3) —
   measures **concentration at position 0**, and therefore cannot tell "the sink was removed"
   from "the sink was shared out over the repeated tokens". Gu Table 1 and Yona §4 together
   show dispersal is exactly what natural repetition does. A conserved-vs-lost measurement
   (total attention to all massive-activation-bearing positions, and the sum of their residual
   norms) does not exist in the literature and is cheap to define.
6. **The sink metrics are not commensurable across the two conditions.** Su's D_sink (Eq. 7) is
   a before/after **ratio between two models** and is undefined for natural repetition. Gu's
   `sink rate` (used by Barbero) is a **fraction of heads** above ε = 0.3 and is therefore
   coarse and threshold-dependent. Cancedda's U-Dark ratio (Eq. 5) is per-token and per-layer
   but requires the SVD of W_u. **No shared metric exists**, which is a large part of why the
   comparison has not been made.
7. **No frequency baseline on either side.** Yu Fig. 5 reports stopword probabilities with no
   unigram baseline (already in the earlier notes); on the natural side, Wang 2025 §2.2 reports
   vocabulary sensitivity qualitatively from a figure. Nobody has regressed repeated-token
   identity on log-unigram frequency *or* on Fu's inflow, in either condition — and Fu and
   Stolfo make opposite predictions about which variable it should be.
8. **The two conditions have now been compared behaviourally in one model, but never
   internally.** Lee-2026 Table 1 gives unpruned-vs-pruned repetition rates on the same Llama
   (5.9 → 12.4/15.4% sampling; 26.6 → 63.1/63.7% greedy) and Roll-2026 gives a 112,426-generation
   symptom characterisation of lesioned models against an unlesioned baseline — **but neither
   measures a single internal quantity in both conditions**, and neither mentions sinks, massive
   activations or super weights. Marwah §6.1 Fig. 4 is the only internal both-conditions
   comparison anywhere (FP16 vs 4-bit NF4, prompt attention and drift *"nearly identical"*,
   entropy deeper) and its entropy curve is reconstructed rather than measured. Su-2025 App. C
   compares two *lesions* (SE pruning loops, outlier-expert pruning does not), not lesion vs
   natural.
9. **Nobody has tested whether a repetition penalty interacts with the sink at all.** Roll-2026
   Fig. 4B shows a repetition penalty of 1.2 takes FFN-Gate lesion repetition to zero, and
   Xu-2026 shows attention-based KV-cache scoring *amplifies* loops by preferentially keeping
   repetitive tokens (Fig. 1b) — but no paper found measures sink mass or massive-activation
   magnitude under a repetition penalty, under no-repeat-n-gram, or under KV eviction. The
   second search returned nothing on this.
10. **Nothing multilingual or encoder-decoder.** Everything above is English decoder-only. The
   MT-side repetition literature (Raunak 2021's oscillatory hallucinations; Guerreiro 2023
   TACL) never mentions sinks or outliers, and the sink literature never mentions oscillatory
   hallucination. Guerreiro 2023 TACL and Dale 2023's detectors use no entropy or attention-
   concentration feature at all (checked: "attention entropy" appears zero times in either PDF).
   The one compressed-NMT both-conditions study, Mohammadshahi 2022 (Findings EMNLP 2022),
   measures off-target rate and a cross-attention alignment ratio, **not**
   repetition or repeated n-grams — so the oscillatory-hallucination-under-compression question
   is still untouched.

---

## §5 Pitfalls that will bite this specific experiment

### 5.1 Self-reinforcement contaminates every post-onset statistic
Holtzman Fig. 4 and Xu Fig. 3 both show TP/IP/WR rising **monotonically to a ceiling** with
repeat count, on random token strings as well as natural sentences. So any internal quantity
measured after onset is measured on a model whose input distribution has already changed.
Consequences:
- The comparison must be **pre-onset vs. matched clean positions**, never pre-onset vs.
  post-onset in the same generation.
- The matched control must be *the same model, the same prompts, continuations that do not
  loop* — not a different corpus, because Xu §2.2 shows the effect size depends on the
  sentence's initial probability TP⁰.
- Hiraoka's Δ_n protocol (Eq. 3) cannot be reused as-is: its "normal" window is the first copy.
- Post-onset probabilities are compressed toward a ceiling, so **use rank-based statistics**
  when comparing conditions (this point is already in `notes_repetition_degeneration.md` OP7).

### 5.2 Greedy vs. sampling changes whether the phenomenon exists
Xu §2.2 attributes the loop to maximisation decoding specifically and says stochastic sampling
avoids it for two distinct reasons (lower likelihood of the previously generated sentence; and
not taking the argmax). Holtzman Fig. 9 shows repetition rising sharply below temperature 0.9.
Fu §2.1 sets ζ_n = 1 under greedy, which is exactly the regime where his ARP bound diverges.
Three quantitative anchors now exist: Lee-2026 Table 1 (same Llama, 26.6% greedy vs 5.9%
top-p 0.9; pruned 63.7% vs 15.4%); **Roll-2026 Fig. 4B (FFN-Gate lesion repetition-loop rate is
zero at T = 0.7 with repetition penalty 1.2)**; Duan §3.1 (numerical loops in reasoning models
persist even at T = 1.0, their Table 7).
**Consequences:** state the decoder in every claim; the lesion results in `notes.md` are greedy,
so the natural-repetition arm must be greedy too for the comparison to mean anything; a finding
that the pre-onset signature disappears under sampling is evidence for M-C, not a null; and
**report the repetition penalty explicitly**, because Roll's grid shows it alone can take the
lesion effect to zero, which means an unstated penalty in a generation config can silently
delete the phenomenon under study. Also note that repetition rate and entropy are not independent
statistics — Marwah §7.2 concedes this for their own validation — so do not report an
entropy-based detector's correlation with a repetition metric as if it were external validation.

### 5.3 "The sink" is a different object in each model family — pick per model, state it
- **Position.** Barbero §2 focuses "exactly at the first token, as this is the most common
  pattern by far", and Gu §3.3's whole analysis is `Sink^ε_1`, but **Sun §4.1 shows LLaMA-2-7B and Phi-2 also carry massive activations on
  "." and "\n"** and states explicitly that attention goes there too. Su-2025 identifies "the
  first token as the attention sink token" for Qwen3. Cancedda §5 fn. 3 says that although
  Xiao's operational definition uses the first four tokens, in his observations BoS dominates
  by far. **Gemma is the special case:** Owen §4.2 finds Gemma-7B / 2-2B / 2-9B have **no
  massive activations at all without a BOS token**, and Barbero's Gemma experiment is precisely
  BOS removal — so a "sink" measured on Gemma without BOS is measuring a different model state.
- **Metric.** Three incompatible ones are in play (§4 item 6). Pick Gu's `Sink^ε_k` at a stated
  ε and a **fixed T** (Gu §3.2 fixes T = 64 because the metric is length-sensitive, and says
  outright there is no principled way to choose ε) for cross-model work — but additionally
  report (i) the raw per-head attention mass on the sink position, because a thresholded
  fraction-of-heads statistic will hide the graded change a "partial dose" would produce, and
  (ii) **total attention to all massive-activation-bearing positions**, because every published
  metric is a position-0 concentration statistic and so cannot distinguish sink destruction
  from sink dispersal (§4 item 5). Gu's ε and Su's D_sink are also not the same *kind* of
  quantity: one is absolute, one is a two-model ratio.
- **Type.** Guo's dormant sinks have **small** value-state norms; Fesser's broadcast sinks have
  large ones and (Lemma 4) increase token similarity; Binkowski finds the hallucination-
  predictive sinks are the **large-‖v‖** kind. Attention mass alone does not identify which you
  are looking at. Report ‖v_s‖ alongside the attention mass.
- **Layer.** Cancedda §7 excludes layers 0–3 because "the attention sink is not in place yet";
  Yona's sink layer is 1 or 2; Yu's super weights sit at layers 1–4; Queipo's compression valley
  is mid-depth. A single-layer measurement will be wrong for some model in any sweep.

### 5.4 "Input-agnostic constant" ≠ "input-agnostic contribution" — do not conflate these
The literature's input-agnosticism claim is about the **value** of the massive activation:
Sun-2024 Table 3 and Owen-2025 §3.1 establish it by showing **set-to-mean is harmless** (16/16
models in Owen; the mean is taken over 100 RedPajama samples, bucketed separately for starting
and non-starting tokens, §3.2). That result bounds the *information content* of the scalar. It
does **not** bound:
- how much attention mass the sink token attracts on a given input — Guo-2024 Fig. 8 shows this
  varies enormously with input domain, per head;
- how large the resulting additive term is in a given head's output — Sun §4.2 shows the value
  update from sink tokens is near-identical **across query positions within one forward pass**,
  which is a statement about positions, not about inputs;
- whether the constant's downstream effect is uniform across generation steps.
**As far as I found, nobody has measured the per-input variance of the massive activation's
value or of its downstream contribution.** Owen reports no standard deviations for either.
So the brief's question — "does the constant's contribution vary per input at all?" — is
**open, and answerable cheaply**: it is one forward pass per prompt.
*This also corrects a slide in the earlier notes*, where "constants and input agnostic … functioning
similarly to bias terms" is quoted in contexts that read as if it settled the attention-mass
question. It does not.

### 5.5 The zero-vs-mean and matched-magnitude controls are not optional
Owen §4.1 (mean harmless / zero catastrophic in the same coordinates), Parodi-2026 (ViT
registers: zero −36.6 pp, mean/noise/shuffle within ~1 pp), and **Cancedda App. D Table 7 (a
random rank-matched filter loops at 20% removal)** all say the same thing: a full ablation
measures an upper bound, and degenerate text is producible by generic damage. The design that
survives this is Cancedda's: **a low-dose intervention plus a magnitude-matched random control
plus the shavings swap** (Fig. 10 — replace the component with the same component computed on a
different input, which leaves BoS untouched because the BoS residual is input-independent, and
so isolates the BoS/sink pathway).

### 5.6 Effective rank and matrix entropy read backwards if BOS is included
Per Queipo-de-Llano: the massive activation *causes* the low matrix entropy. Removing it
**raises** matrix entropy while (per Barbero) increasing over-mixing among the remaining tokens.
Exclude the sink position — and, per Cancedda §7's HMLV recipe, ideally the last few positions
too — before computing effective rank or mean pairwise cosine. Also hold sequence length fixed;
Barbero §3 makes representational collapse an explicit function of prefix length.

### 5.7 Perplexity does not see this failure mode
Jin-2025 App. F: a fully looping generation ("divisible by 9 and 12, which is the product of X,
if it is divisible by 9 and 12…") scores **perplexity 2.99**. Use a repetition metric —
Holtzman's Fig. 9 criterion (a phrase of length ≥ 2 repeated ≥ 3× at the end of a 200-token
generation) is the simplest published one and matches Hiraoka's 10-gram-thrice rule closely
enough to report both.

### 5.8 The two phenomena do not look alike on the page
Hiraoka App. B/C: natural loops are **long, grammatical, semantically coherent templates** that
lose their variable slot. Lesion loops (`superweights/notes.md`; Cancedda App. D) are
**single function words or punctuation from the first generated token**. Cancedda's Ψ-filtered
output is the one lesion output that looks in between. Lee-2026 §1's example of *pruned*-model degeneration sits exactly in between and is
sentence-shaped like the natural case: *"... for the deceased. The cemetery is designed to be a
peaceful place. The cemetery is designed to be a peaceful place. ..."*.
*Inference:* if the project's answer is "same route", it needs an account of why the surface forms
differ; the natural candidate is that the lesion is a full dose and natural degeneration a partial
one, with LoRA-healed structured pruning in the middle — which makes the **dose–response curve
the central experiment**, not the endpoint comparison. Lee's entry/persistence split (§3.2) is
the right dependent-variable decomposition for that curve: report **loop entry rate** and
**loop persistence** separately, because a lesion could plausibly move only one of them, and a
single repetition rate would hide which.

### 5.9 A pre-onset window has to be defined against a matched non-looping control, not against itself
Every "before onset" window in the literature is either contaminated (Xu n = 1, Hiraoka's r = 30
tokens preceding the second occurrence, Mahaut's Natural prompts looping from token 1) or
compared only against its own later segment. **Duan §3.2 Fig. 7 is the only design that gets this
right** — 50 loop and 50 non-loop samples, a fixed window preceding onset, compared against both
the pre-repetition history *and* the non-looping baseline, across window sizes 64/128/256/512.
Copy that design. Two further requirements it implies: report the result across several window
sizes rather than one, because the effect is a function of the window; and generate the matched
control from the **same model on the same prompts**, because Xu §2.2 shows the strength of the
effect depends on the initial sentence probability TP⁰, so a different prompt set is not a control.

### 5.10 "Repetition" is four different measurements in these papers — pick one and say so
Holtzman Fig. 9: a phrase of length ≥ 2 repeated ≥ 3× at the end of a 200-token generation.
Hiraoka §2.2: the same 10-gram three times at equal intervals within 100 tokens.
Mahaut §2: a cycle of length > 1 that persists to the end of a 1,000-token generation.
Lee-2026 Eq. 1–2: CREP, the fraction of sequences where one dominant N-gram covers > 30% of the
tokens. Xu-2026 §5.2: type–token ratio ≤ 0.2 and compression ratio ≤ 0.12 over a 256-token window.
Duan: a loop-vs-non-loop label from a hidden-state classifier plus manual loop typology.
These give materially different rates on the same generations. Lee's CREP and Holtzman's rule are
the two with published cross-condition numbers, so report at least one of those for comparability,
and state the threshold.

---

## §6 What is read and what is inferred, in one place

**Read for this file (papers new to the folder on 2026-09-05):** Mahaut §1.2, §2, §3.1–3.3,
§4.1–4.2, §5, §6 and Figs. 1–5; Duan §3.1–3.3, §4 and Figs. 3–8; Xu-2026 abstract, §1, §5.1–5.3
and App. Fig. 5; Marwah §3.1–3.2, §6.1, §7.2, §7.4, Figs. 4–5 and Tables 3, 5; Lee-2026 §1,
§3.1–3.4 and Tables 1–2; Roll-2026 Results, Limitations, Fig. 4 and the 3B/7B appendix summaries;
Mohammadshahi-2022 §4.1.2–4.1.3 and Tables 3–5.

**Read for this file (already in the folder):** Yona §4.1–4.4 / §5 / §6–7 and Table 1 and Fig. 6; Cancedda §5–7, §9,
App. D Tables 6–7 and Figs. 9–11, 15; Xu §2.1–2.2 and conclusion; Hiraoka §2.2, §3.1–3.2, §5,
App. A–D and Tables 2–3; Yao §4.3.1–4.3.2, §5.1; Wang 2025 §2.2, §3, Table 1; Stolfo §6.1–6.2;
Sun-2024 §4.1; Barbero §2, §3–3.2 and the Table 3 discussion; Guo §"active-dormant", Claim 1,
§3.1; **Gu-2025 §3.2–3.4 and Table 1**; Su-2025 §5.1–5.2, App. C and E; Owen §3.1–3.2, §4.1–4.2; Queipo abstract, §1, §2;
Fesser §"Hypothesis 2", Lemmas 3–4; Fu §2.2–2.3 and Cor. 1.2; Li-2023 §"Relation to Previous
Hypotheses"; Holtzman Fig. 4 / Fig. 9 captions and §4–5.2; Binkowski abstract, §1, §2.1–2.2,
App. "Important Head Selection".

**Inferred (mine, not the papers'):** that Mahaut's 9.9×-attended newline tokens are
Sun-2024's massive-activation-bearing delimiters (Mahaut never says "sink", "BOS" or "massive
activation"); that Roll's FFN-Gate localisation is the same subsystem as Oh-2024's `W_gate`/`W_up`
massive weights (Roll never says so); that Lee's *alternative collapse* and Yu's flattening are
opposite ends of a severity axis rather than one mechanism; that Yona's 450–4000-repeat requirement rules his
mechanism out as an account of onset; that Stolfo §6.2's dormant→active framing supplies an
innocent explanation for a sink drop at onset; that Fu predicts *no* pre-onset signature and is
therefore the right null; that Queipo's theorem means effective rank must exclude BOS; that
Cancedda's App. D dose comparison makes the low-dose regime the only place the sink-specific
claim is testable; that Su App. C is usable as a free control; and the whole of §3's decision
table, which is assembled from the papers' separate predictions and has not been asserted by
anyone.

**Not checked:** Olsson et al. 2022 on induction heads **[unopened]**; Veličković et al. 2024
on softmax leakage, cited by Yona Theorem 4.1 **[unopened]**; Barbero et al. 2024 *Transformers
need glasses!* **[unopened]** — this one matters, because Barbero 2025's Eq. 2 representational
collapse is defined on a repeated final token and the 2024 paper is where that result lives;
Wang et al. 2024 on repetition neurons in MT, cited by Hiraoka App. A **[unopened]**;
Skean et al. 2025 on compression valleys **[unopened]**; Nasr et al. 2023 **[unopened]**;
**Gu-2025 App. C.1 (the proof that identical first-T tokens give identical hidden states) and
App. C.2 (the domain-invariance result) [unopened]** — C.1 in particular carries the dispersal
argument and should be read before the argument in §1/§2 is relied on;
Puccetti-2022 and Macocco-2025 were read for the sibling notes file, not re-read here;
Duan App. proofs and CUSUM
implementation **[unopened]**; Mahaut App. A (MLP contribution) and App. B **[unopened]**;
Lee-2026 App. A and G **[unopened]**; Roll-2026 Appendices **[unopened]**.

**Found in search but not opened, listed so they are not re-searched:** Alimaskina et al. 2026
(arXiv:2606.02011, 2-bit Qwen3 failure-mode taxonomy including repetitive loops); Xu et al. 2026
(arXiv:2605.08894, "smoothness degradation" / reduced effective token candidates under ultra-low
bit — quantization's analogue of Lee's alternative collapse); Chang, Zhang et al. 2025
(**EMNLP 2025**, arXiv:2506.12044, full-precision residual-stream magnitude predicts which inputs
break under low-bit quantization); Lazaridis et al. 2026 (arXiv:2606.13705, one sign-inverted
neuron in Gemma 4 E2B cuts detected loops 46/384 → 12/384 in a *healthy* model — the natural-model
counterpart of the single-scalar result, and a useful null); Weidmann et al. 2026
(arXiv:2608.22761, DRY sampler on AWQ-quantized Llama-3-70B); Dong et al. 2026 (arXiv:2602.10504,
a token-level repetition-based degeneration detector used as a damage diagnostic); Luo et al. 2026
(arXiv:2605.17887, sink and outlier statistics under W8A8/W4A4 — the closest thing to sink stats
under quantization, no repetition metrics); Myrzakhan et al. 2026 (arXiv:2602.17664, sink-aware
pruning for diffusion LMs); Ettifouri et al. 2026 (arXiv:2607.11317, e-CUSUM decoding monitor on
FP16 vs INT4, self-described pilot at n = 100 with McNemar p = 0.18); arXiv:2605.09992
(*Attention Drift* in speculative decoding — sink mass measured per step, wrong setting, right
protocol); arXiv:2604.10027 (*SinkTrack*, arXiv comments state ICLR 2026, injects context into BOS
to fix attention drift). **All of these are [unopened] and their descriptions come from a
subagent's search summaries, not from the PDFs.**
