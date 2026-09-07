# Super weights, massive activations, attention sinks — reading notes

Covers both [``](.) and
[``](),
plus papers already filed elsewhere in `papers/` that bear on the same
questions. Written 2026-09-05.

**Provenance convention.** Everything below is from a PDF opened in this
session unless marked. `[unopened]` marks a paper named but not read — do not
cite it without opening it. Inferences of mine are marked *(inference)*.
Section/table/figure references are to the PDF as filed.

---

## §1 Paper by paper

### 1.1 The super-weight line

**Yu et al. 2024, *The Super Weight in Large Language Models*** (arXiv:2411.07191
v2; **preprint** — no venue banner, despite being widely cited as 2025).
Pruning one scalar in an early `mlp.down_proj` raises Llama-7B Wikitext-2
perplexity 5.67 → 1211 and drops zero-shot average 70.1 → 35.1 (Table 1). The
detector is **activation-based, not magnitude-based**: plot max-magnitude
`down_proj` input and output per layer, read the intermediate-neuron index off
the input spike and the residual-channel index off the output spike, remove,
repeat (§3.1, Fig. 3). Table 2 is the coordinate directory. §3.2 claims two
mechanisms: super weights create super activations, and they *suppress stopword
likelihood* (Fig. 5: removing the SW multiplies P("the") ≈2×, P(".") ≈5×,
P(",") ≈10× over 500 Lambada prompts).
- **Causal:** the pruning ablations, and the *Prune SW, +SA* row (restoring the
  super activation recovers 42% of the accuracy loss, so the super activation
  only *partially* mediates the effect — Table 1).
- **Correlational:** everything about stopwords. Fig. 5 and Appendix A.3 are
  **descriptive only**. The paper offers no mechanism for the stopword shift and
  no limitations section. Its §3.2 case study ("Summer is hot. Winter is __":
  correct token "cold" at 81.4% → top-1 "the" at **9.0%**) is a single prompt.
- **Watch:** Fig. 4 says the super activation "persists throughout the entire
  model, at exactly the same magnitude". Sun-2024 §2.1 says the opposite about
  the last few layers — see §5.6.

**Subramanian et al. 2026, *Super Weights in LLMs and the Failure of Selective
Training*** (arXiv:2607.08733; **COLM 2026, peer-reviewed**). Training super
weights in isolation collapses to chance (Tables 2–3), while an equal-size
random-coordinate control in the *same* `down_proj` layers *improves* over
baseline (64.18% vs 60.65%, §4.8) — so the failure is specific to the
coordinates, not to sparsity. Confirms Yu's coordinates across 5 models
(Table 8, top block). **The result that matters most here is the bottom block of
Table 8: top-magnitude identification with 2 coordinates produces literally no
change** (8.92 → 8.92, 9.51 → 9.51, 11.96 → 11.96, 12.21 → 12.21) in
Llama-3.1-8B-Instruct, Llama-3.2-3B, Llama-3.2-1B and Qwen2.5-1.5B-Instruct.
They conclude "high magnitude alone is insufficient". Also: 9 positions produce
the largest `down_proj` activation spikes in 1000/1000 WikiText-2 samples
(Table 1), so the coordinates are input-independent.
- **Causal:** all of it. 10-seed ablation on the LoRA comparison.
- **Limitations they state:** ARC-Easy primary, one seed for the random-position
  control, OLMo-only for the training experiments.

**Oh et al. 2024, *House of Cards: Massive Weights in LLMs*** (arXiv:2410.01866;
preprint). Independent line, and the important one for this project's finding
(b): it defines **top-k massive weights as the *rows* of `W_gate` and `W_up`
that produce the top-k magnitudes of the FFN intermediate state**, i.e. the unit
is the *intermediate neuron*, not a `down_proj` scalar (§2.2, Fig. 1a). Top-5
zeroing: Llama-3-8B avg perplexity 8.22 → 122.7; Llama-3-70B 5.43 → 7707;
Llama-3.1-405B 3.61 → 1135. Top-5 *retaining* (zero everything else in those two
matrices) is far less damaging. **Reports repetition after ablation**: "when
massive weights are set to zero, the model repeats the same text as the user
prompt" (§2.2, Fig. 1b). Appendix F reports **Gemma-2 and Phi-3-medium are not
sensitive** to massive weights — a coverage negative most of this literature
omits.

**Su et al. 2025, *Unveiling Super Experts in MoE LLMs*** (arXiv:2507.23279;
**ICLR 2026, peer-reviewed**). The closest published analogue to this project's
ablation. Pruning **3 of 6,144** experts in Qwen3-30B-A3B: WikiText-2 perplexity
8.70 → 59.86 while a 1000-random-expert control gives 10.85 (Fig. 1); Math-500
and AIME Pass@1 to 0.00 (Tables 4–5); and, on review of the generations, "the
model consistently generated **repetitive responses** in nearly every test"
(§4, Tables 11–12). §5.2 measures an **Attention Sink Decay Rate** (Eq. 7) of
≈90% or more across layers after pruning, with sinks visually gone (Figs. 8–9).
Its detector (Eq. 6) is the least arbitrary in the literature: expert *e* in
layer *l* is a Super Expert if `a_{l,e} > P99.5(A)` **and** `a_{l,e} > max(A)/10`
**and** *l* is in the MA-forming layer set — motivated by a CCDF power-law fit
with α < 2 (Fig. 12). Appendix H maps SEs down to weights and gives **Table 8**,
which matters for finding (b): Llama-3.2-1B's four super weights
`(400,1417) (698,1417) (2029,1417) (1159,1417)` are four residual channels
driven by **one** intermediate neuron.
- **Caveat:** they show sink collapse *and* repetition after the same
  intervention but never test whether the first causes the second.

**Owen et al. 2025, *A Refined Analysis of Massive Activations in LLMs***
(arXiv:2503.22329; **preprint**, BluOrion). The most important corrective in
this folder, and the paper this project should read first. 17 models, GLU and
non-GLU, with the BOS token controlled (prior work defaulted to no BOS).
- **"Not all massive activations are detrimental."** Table 1, with BOS: zeroing
  changes essentially nothing in GPT-2 (30.42 → 30.57), Phi-2 (64.40 → 64.30),
  OPT-6.7B (10.99 → 10.99), Gemma-3-1B, Gemma-2-9B, Phi-4-14B, Gemma-3-12B —
  roughly **7 of 16 models** — while it is catastrophic in 9 (LLaMA-2-7B
  5.13 → 8,977; Mistral-7B-v0.3 4.99 → 2.6 × 10⁷; Gemma-7B 6.40 → 2.8 × 10²⁴).
  The split tracks GLU vs non-GLU **imperfectly** (Falcon-7B is non-GLU and
  moderately affected; several Gemmas are GLU and unaffected), and they state
  plainly: "we have yet to identify a clear indicator that distinguishes between
  detrimental and non-detrimental massive activations."
- **Set-to-mean is harmless in every single one of the 16 models**, to 2–3
  decimal places, including the catastrophic ones. Their mean is computed more
  carefully than Sun's: positions bucketed into starting-token vs
  non-starting-token before averaging. **This is the single most consequential
  methodological fact in this document** — see §5.2.
- **Detection is fragile.** OLMo-2-1124-7B has **no massive activations at all**
  under Sun's criterion. Gemma-7B, Gemma-2-2B and Gemma-2-9B have none *without*
  a BOS token and do *with* one — i.e. whether a model "has" massive activations
  depends on the prompt convention.
- **Breaks two of Sun's three stated characteristics:** Gemma-2-2B and the
  Gemma-3s show an *uptrend* in magnitude across layers, not constancy; and in
  Gemma-3-4B/-12B and Falcon-7B the activations attach to content words
  ("polished", "mass", "cold"), not only delimiters and weak-semantic tokens.
- **Undercuts the normalisation mechanism:** replacing all RMSNorm with Dynamic
  Tanh removes massive activations but **attention concentration persists**
  (Table 2), which "challenges the hypothesis that RMSNorm or LayerNorm play a
  key role". Sun-2024's Attention-KV-bias mitigation **reproduces on GPT-2 and
  fails entirely on their LLaMA-1B** (max magnitude 1416 → 1512).
- **Weaknesses:** single run per cell, no CIs, no seeds; the mitigation arm is
  LLaMA-1B only.

### 1.2 Massive activations and their function

**Sun et al. 2024, *Massive Activations in Large Language Models***
(arXiv:2402.17762; **COLM 2024, peer-reviewed**). The definitional paper.
Massive activations are ≤4 scalars per hidden state, 10³–10⁴× the median,
input-agnostic, emerging abruptly at one early layer and — note — "start to
**diminish in the last few layers**" (§2.1). Their loose criterion: magnitude
> 100 and ≥ ~1000× the median of its hidden state (§2, Table 1). §2.3 shows
they are **not** the same as Dettmers' outlier features (zero overlap in
LLaMA2-7B/13B).
- **The zero-vs-mean experiment (§3, Table 3) — exactly what it shows.** They
  intervene *once*, on the hidden state at the layer where MAs first appear
  (layer 2 for LLaMA2-7B, layer 4 for 13B). **Set to zero:** WikiText/C4/PG-19
  perplexity → `inf`/`inf`/`inf` (7B) and 5729/5526/4759 (13B); zero-shot 68.95%
  → 36.75% and 71.94% → 37.50%. **Set to their empirical mean over 100
  RedPajama sequences:** 5.47 → 5.47, 7.85 → 7.86, 8.57 → 8.59, zero-shot 68.95%
  → 68.94%. A matched control that zeroes an equal number of *median-magnitude*
  activations causes no drop. The inference they draw, and it is the right one:
  the *value* carries no input-dependent information — MAs act as **fixed
  biases**, and what the model cannot survive is their *absence*, not their
  variation. **This is the single most important methodological precedent for
  this project** (see §5.2).
- §4 is the attention story: MAs make the associated tokens' post-LayerNorm
  representations near-identical sparse vectors (Fig. 7b, App. B.2), attention
  concentrates on them (Figs. 5–6), and the resulting value updates are
  essentially constant across query tokens (Fig. 8) — i.e. an **implicit
  attention bias**. §4.3 is the causal test: GPT-2 trained with *explicit*
  learnable K/V biases (Eq. 3) reaches the same loss and **has no massive
  activations** (Figs. 9–10).
- **Limitations they state (§7):** the alternative-attention experiments are
  GPT-2-scale only; training stability untested.

**An et al. 2025, *Systematic Outliers in Large Language Models***
(arXiv:2502.06415; **ICLR 2025, peer-reviewed**). Unifies weight, activation and
attention outliers into one chain and quantifies the alignment: weight outliers
in `W_down` align with `x_down` and `h` activation outliers at **100%** in the
feature dimension, and activation outliers overlap attention outliers at **95%**
in the sequence dimension (Table 1). §4.2: weight outliers in up/gate produce
extreme neuron responses, amplified by SiLU/GLU, then broadcast by `W_down` into
specific residual channels (1415, 2533 in LLaMA2-7B) — Fig. 7. §5 argues the
functional role is an **implicit context-aware scaling factor**, tested against
five attention variants (Table 2). Appendix: the formation account is that MHA
needs a *near-zero update* for some tokens, softmax cannot express it, so the
dynamic range must blow up.

**Sun et al. 2026, *The Spike, the Sparse and the Sink*** (arXiv:2603.05498;
preprint). The best from-scratch ablation suite. §3.1.2 derives why SwiGLU
amplifies: each output coordinate *k* is a quadratic form `h^T U_k h`, spike
channels are exactly those with huge `‖U_k‖_F` and a **rank-one dominant
eigenvalue** whose eigenvector `s*` is nearly shared across spike channels
(Figs. 3–4) — so one trigger direction fires all spike channels at once. §3.1.1
gives step-up/step-down block indices per model (Table 1: Llama-2-7B step-up
block 4, step-down 62 of 64). §3.1.3: >98% of *all* vocabulary items become
spike tokens at position 0 (Table 2), so it is positional, not semantic.
- **§4 is the causal core, all trained from scratch at Llama-7B shape on 100B
  DCLM tokens.** Table 3 (optimisation): no weight decay → spikes exceed 12,275
  with no change in sink ratio or perplexity; sink ratio tracks "optimisation
  health" (extreme LR, no weight decay, wrong β₂ all reduce it). Table 4 (FFN
  design): spikes and sinks appear even with linear or attention-only blocks —
  SwiGLU/GeLU are amplifiers, not prerequisites. **Table 5 (normalisation):
  sandwich-norm gives spike 520 with sink ratio 44.7% vs baseline spike 3818 /
  46.0%; QK-norm gives spike 92; DynamicTanh gives spike 153 with the *highest*
  sink ratio, 61.0%.** Table 6: head dimension is the dominant driver of sinks
  (8 → 128 raises sink ratio 4.1% → 46.0%). Table 7: conditional gating
  eliminates sinks; unconditional or static gating does not. Table 8: training
  only on long contexts collapses the sink ratio to 1.2–13.0%.
- **Conclusion (§4.4):** sinks and spikes are "decoupled architectural
  artifacts"; MAs are global implicit parameters, sinks are local head-level
  modulators.

**Queipo-de-Llano et al. 2025, *Attention Sinks and Compression Valleys are Two
Sides of the Same Coin*** (arXiv:2510.06477; **ICLR 2026, peer-reviewed**). The
opposing causal claim. Theorem 1: `σ₁² ≥ M + αR`, so a dominant token norm
mathematically forces spectral dominance; Corollary 2 bounds anisotropy and
matrix entropy. Fig. 3 shows the bounds become nearly exact once MAs appear.
**§3.3 is the ablation: zeroing only the MLP contribution to the BOS token at
layer 0 in Llama-3-8B keeps entropy at 0.4–0.5 bits instead of dropping to 0.02,
drives sink rate from 0.85–1.0 to 0.0, and keeps the BOS norm within 2× of other
tokens (Fig. 4).** Correlations across 6 models, 410M–120B: Δ(BOS norm) vs
Δentropy r = −0.9 ± 0.18; vs sink rate r = 0.58 ± 0.251.
- **Limitations they state:** decoder-only; **"we observe model-dependent
  exceptions (e.g., sinks persisting despite decompression)"**; theory assumes a
  single massive row.

**Chen et al. 2026, *Attention Sinks Induce Gradient Sinks*** (arXiv:2603.17771;
preprint). Reverses the arrow: causal masking routes many tokens' training
signal back through token 0, creating a **gradient sink**; RMSNorm's Jacobian
attenuates gradient roughly inversely with input norm, so growing the residual
norm at token 0 is how the network throttles that pressure (Thm 2). Definition 1
gives a gradient-sink ratio and **deliberately declines to fix a threshold**.
Their V-scale intervention (one parameter per layer per head, 0.1B/0.3B/1B
trained from scratch on C4) **suppresses massive activations while attention
sinks are preserved and in places strengthened** (Figs. 6–7); gradient-sink
median 40.3 → 5.6. They concede it is "a partial intervention".

**Chen et al. 2026, *Measuring Maximum Activations in Open LLMs***
(arXiv:2605.15572; preprint, Baidu). Pure measurement, 27 checkpoints, 8
families. Deliberately replaces Sun's binary criterion with a continuous
`M = max|a|`. **4 of 24 checkpoints fail Sun's criterion** (§3.2, Table 1):
Qwen2.5-1.5B has peak 7,968 but a local ratio of only ≈574; Qwen3.5-0.8B/-9B/
-35B-A3B have peaks of 122/956/132. And the two views disagree in both
directions — "some checkpoints failing the binary criterion are the easiest to
quantize, while some passing it are the hardest". Everything is correlational
and the paper says so. Two internal inconsistencies to avoid citing: the Ling
training-stage effect is 1.34× in text and 1.54× in Fig. 11; Table 1's "Top 1"
is a representative-layer value, not the global max plotted in the figures.

**Li et al. 2026, *The Structural Origin of Attention Sink*** (arXiv:2605.06611;
**ICML 2026, PMLR 306, peer-reviewed**). Chain: causal masking leaves token 0 a
high-*variance* outlier → `W_O` preserves it (Kendall τ mean 0.32) → **super
neurons** amplify it → RMSNorm collapses the result onto one basis direction
(Llama-2-7B: |x_2533| = 1.2568 vs dimension mean 0.0048, **dominance ratio
262.88×**, Table 1) → QK locking. Their causal work is on *variance*: forcing
token 10 to attend only to itself makes it a new sink, and amplifying its
variance does too — but **scaling its norm alone does not** (Figs. 5–6), which
is a genuinely sharp control. Head-wise RMSNorm after value aggregation
suppresses both phenomena at 152M with **4 seeds** and *better* loss
(2.7812 ± 0.0109 → 2.7421 ± 0.0066; effective rank 343.7 → 446.0).
- **The point for this project.** §4.2 describes `W_down[7890, :]` in Llama-2-7B
  layer 1 as heavy-tailed with a few huge entries channelling one massive
  activation into dimension 2533 — **this is Yu et al.'s super weight in the row
  basis, and Yu et al. are not cited.** Their "super neuron" has *no numeric
  criterion*: it is read off Fig. 8. And they never zero it. *(inference: an
  ablation of `W_down[7890,:]` is a gap left open inside a peer-reviewed ICML
  paper.)*

### 1.3 Sinks: function, formation, removal

**Gu et al. 2025, *When Attention Sink Emerges in Language Models***
(arXiv:2410.10781; **ICLR 2025, peer-reviewed**). The pretraining-knob study,
60M LLaMA-shaped models on 5B Pile tokens, plus a 1B replication. Defines the
`Sink^ε_k` metric (ε = 0.3, T = 64) everyone else reuses. Results, in detail:
- **Optimisation (§4):** sinks emerge between **1k and 2k steps** under the
  default setup; smaller LR both delays and weakens them, "even if we compensate
  for more training steps"; batch size has no effect (Table 10).
- **Data (§5):** with less training data the sink **disappears** — and Fig. 28
  shows this is not overfitting. Randomising the first token *raises* the sink
  metric to 27.03%; randomising the first two shifts the sink to position 2
  (Sink^ε₂ = 14.08%, Sink^ε₁ = 1.98%); fixing a token at position 2 or 3 moves
  the sink there. Sink position follows the loss and data, not semantics.
- **Weight decay (§6, Table 2):** γ = 0 still gives sinks (15.20%); γ = 0.5
  maximises (41.08%); γ = 2.0 collapses it (6.13%) and γ = 5.0 kills it (0.01%)
  — but validation loss also degrades 3.72 → 5.24, so the high-γ end is
  confounded with broken optimisation. **The honest reading is inverted-U, not
  "weight decay encourages sinks".**
- **Architecture (§7):** positional encoding, FFN design, LN placement and
  multi-head design all make **no** difference. Sinks are **key biases**: adding
  explicit learnable `k*` (with `v* = 0`) moves the sink off token 1 entirely
  (Sink^ε₁ = 0.00%) at equal loss (Table 4); adding only `v*` does nothing.
  Table 5: as ‖v*‖ grows the sink migrates back to the first token.
  **§7.4: sigmoid or elu+1 attention *without normalisation* gives no sink and
  no massive activations**, replicated at 1B (45.11% → 2.46%, val loss 3.07 vs
  3.10).
- **Single seed throughout**; the paper reports no seed variation. Future work
  (§8): delimiter-token sinks, and whether sinks help downstream at all.

**Barbero et al. 2025, *Why do LLMs attend to the first token?*** (COLM 2025,
peer-reviewed; filed in ``). The over-mixing account:
sinks slow information mixing and so protect against rank/representational
collapse. §4.1 — pre-training context length causally controls sink strength at
120M (sinks nearly absent at 128 tokens, saturating by 2048, matched token
budget). §4.2 — sink metric rises with model size across LLaMA 3.1: 45.97 (8B),
73.49 (70B), 78.29 (405B). **Table 3 is the number to hold onto**: removing
`<bos>` from Gemma 7B at inference drops ARC-e 80.77 → 28.49, HellaSwag
80.61 → 27.35, and Ruler-4096 **82.57 → 0.00**.

**Guo et al. 2024, *Active-Dormant Attention Heads*** (arXiv:2410.13835;
preprint). Theory plus a toy model. Extreme-token phenomena are three things at
once — attention sink, **value-state drain**, residual-state peak — and they
arise by a **mutual reinforcement** loop: softmax shifts mass toward tokens with
drained value states, and the resulting sink further suppresses those value
states (Claim 2, Thm 2). Reproduced in a 1–3-layer Bigram-Backcopy model.
**Adam → SGD removes residual-state peaks while sinks remain; softmax → ReLU
removes both** (Fig. 7c). Analyses OLMo-7B pretraining checkpoints (§3.2) and
finds the same simultaneous formation.

**Kaul et al. 2024, *From Attention to Activation*** (arXiv:2410.17174;
preprint). The cleanest cause-separation in the literature: **softmax-1 removes
first-token dominance but leaves outlier activations intact; the adaptive
optimiser is what produces the outliers, and OrthoAdam removes them.** Llama
models attend maximally to the first token in 98% of heads. Their 4-bit
quantisation table is the payoff (GPT-2-Medium 28.8 → 2435 for the standard
model; 16.3 → 17.1 for theirs).

**Ran-Milo 2026, *Attention Sinks Are Provably Necessary in Softmax
Transformers*** (arXiv:2603.11487; preprint). Theorem 1: for a single-layer
softmax model solving a trigger-conditional task with loss ≤ η, every
non-trigger position puts ≥ 1−ε of its mass on the anchor, with high
probability. Theorem 3 constructs a ReLU-attention model with **exactly zero
loss and zero sink**. So the sink is the *only way* softmax can express a
no-op. Theorem 2 is **existential** — at least one layer and position must sink,
not every head (Fig. 6 shows a non-sinking head in a 4-layer model). Footnote 1
concedes the result does not cover gated attention or Mamba.

**Ran-Milo et al. 2026, *A Mechanistic Account of Attention Sinks in GPT-2***
(arXiv:2604.14722; preprint). A full parameter-level circuit: a source-agnostic
shift `b_Q^T W_k^T x_j`, an **effective positional encoding**
`EPE_i = MLP¹(p_i) + p_i` whose first entry carries **massive activations** at
three coordinates (138, 378, 447 — selected at 3σ, in fact >15σ), and a key
projection whose largest columns sit at exactly those coordinates.
**Ten interventions, Table 1**: nullifying `b_Q` leaves 44.7% of baseline BOS
attention, replacing PE₁ leaves 3.0%, swapping EPE leaves 6.2%, removing all
MLPs 50.3%, removing all PEs 17.5%, zeroing the top-3 `W_k` columns 65.2% — and
three matched controls (swapping raw PEs 99.0%, nullifying the BOS token
embedding 99.7%, zeroing 3 random `W_k` columns 100.0%) do nothing.
- **Correction worth stating plainly, because it is easy to get backwards:**
  their finding is that **each component *is* necessary in GPT-2** (Table 1
  caption: "Each component of the sink circuit is necessary"). The
  *architecture-specific* caveat — that other architectures lacking `b_Q`, MLPs
  or positional encodings still sink — is **cited, not measured** (footnote 11).
  The transferable caution is therefore: *a sink circuit is architecture-
  specific; killing it in one model does not identify a universal cause.*
- §5.3 concedes their interventions **reduce but never eliminate** the sink, so
  unidentified secondary contributors exist.

**Fesser et al. 2026, *A Unifying View of Attention Sinks: Two Algorithms, Two
Solutions*** (arXiv:2606.08105; **preprint**, Kempner). One sink signature hides
two algorithms. **Adaptive NOP** (Hyp. 1): attention concentrates on *s*, the
output is negligible and `x^{l+1}_i ≈ x^l_i` — a learned null token that lets a
head conditionally do nothing despite softmax forcing mass somewhere.
**Broadcast** (Hyp. 2): `v_s` carries real information and
`x^{l+1}_i − x^l_i ≈ A_is v_s`, so every attending token acquires a *shared
linear component*.
- **The discriminating measurements, which are the reusable contribution:**
  (i) **sink value-norm ratio** `‖v_s‖/‖v_i‖` — NOP gives < 0.2×, broadcast ≈ 1;
  (ii) **stable rank of the attention update** — broadcast ≈ 1, NOP higher.
  In DINOv2-G both regimes coexist, split by depth (Fig. 6).
- **Directly relevant hypothesis (§3.1, after Lemma 2):** adaptive gating
  requires either a spectral gap in `W_Q W_K^T` **or a massive activation at the
  sink token**, so massive activations "might be **the energetic cost of
  enforcing a reliable NOP gate without distorting the weights**". Stated as a
  maybe, but it is the most specific *functional* account of why a massive
  activation should exist at all.
- **Caveats:** all experiments are ViT; LLMs appear only in related work. There
  is **no activation patching and no ablation of individual sinks** in a
  pretrained model — the causal evidence is constructive toy tasks plus
  from-scratch architecture variants (3 seeds, Table 1). Their ε-sink threshold
  is not calibrated.

**Su et al. 2026, *Attention Sink in Transformers: A Survey***
(arXiv:2604.10098; preprint, 103pp). Nine interpretations across five levels,
presented as *complementary*, not competing: softmax/no-op, structural bias,
active-dormant, outlier circuits, outlier-driven rescaling,
mix-compress-refine, geometric anchoring, anti-over-mixing, spectral energy,
implicit attention bias. §4.2 files **Yu et al.'s super weight under "weight
outliers"**. Its §2.2.2 sink criterion is a cumulative-mass rule with
"τ > 1 … empirically set to a large value (e.g. 1000)", and it stresses that
high attention alone does not make a sink — the token must also be
low-information. Open problems (§7.1–7.2): training dynamics of outlier-circuit
emergence are "largely unexplored"; **no standardised benchmark for outlier
detection or mitigation exists**; all current fixes require training from
scratch, and lightweight adaptation is an open gap.

### 1.4 The outlier-feature lineage, and where the nulls are

**Dettmers et al. 2022, *LLM.int8()*** (NeurIPS 2022, peer-reviewed). Origin of
"outlier feature". §4.1 criterion, verbatim: magnitude **≥ 6.0**, affecting
**≥ 25% of layers** and **≥ 6% of sequence dimensions**. Crucially, the paper
says how the thresholds were chosen — 6.0 because "perplexity degradation stops"
there, and the layer/sequence fractions "in such a way as to limit detection to
a single outlier in our smallest model with 125M parameters". **Reverse-
engineered from a desired detection count.** Causal on the effect side (zeroing
outlier dims: top-1 softmax 40% → 20%, perplexity +600–1000%, vs 7 random dims
at −0.1%); correlational on emergence, and the paper itself notes the "sudden at
6.7B" story becomes smooth when plotted against perplexity instead of parameters
(Fig. 3b). Single run per model, stated in the checklist.

**Kovaleva et al. 2021, *BERT Busters*** (Findings of ACL-IJCNLP 2021,
peer-reviewed). The encoder precedent: zeroing LayerNorm γ and β at 2 indices
across 12 layers (48 weights, <0.0001% of BERT) costs up to 44 GLUE points and
raises MLM cross-entropy 2.30 → 4.53. Criterion: **both γ and β beyond 3σ of the
per-layer mean, consistently in ≥ half the layers** — relaxed per model (2σ for
RoBERTa, 1/3 of layers for BERT-large) to make detection work. Has a real
**effect-side null** (random pairs averaged over 1000 runs) and the field's
cleanest magnitude-vs-position dissociation (Table 5: largest-scaling-factor and
largest-bias baselines are far less damaging than the position-consistent
outliers). Emergence is **gradual, from ~50k steps**, single seed, and they flag
that causality is unknown. Footnote 1: HuggingFace and Google checkpoints of the
*same* BERT-base config have outliers at **different locations and signs** —
i.e. outlier identity is run-dependent.

**Puccetti et al. 2022** (arXiv:2205.11380; Findings of EMNLP 2022). Adds an
**effect-size gate** to Kovaleva's criterion — a dimension counts only if it is
"at least 5× more damaging on MNLI" — which is a real improvement in kind and
still arbitrary in degree. The frequency claim: Pearson correlation, per layer,
between the hidden-state coefficient at the outlier index and the token's
pretraining-corpus count, reported as figures not numbers. §4.4 is the one
manipulation: three BERT-medium pretrainings differing only in `[SEP]` density
and token-frequency flattening; the frequency-flattened run develops outliers
whose removal costs 1.0–2.4 points instead of 47.4. **They report that ablating
outliers makes BERT predict *more* frequent tokens** and shifts POS toward
nouns, punctuation and adpositions. Confound they state: the tokenisation
changes degrade the models independently. English only.

**Macocco et al. 2025** (arXiv:2503.21718; preprint). The decoder counterpart,
and it **contradicts Puccetti's direction**: ablating last-layer outlier
dimensions makes 8 decoders predict *rarer* and more diverse tokens (pythia-12b
log-log slope 1.35 → 0.74, Spearman 0.70 → 0.53). Criterion: a dimension whose
**median** activation across 50k samples is in the **top 1%** of all
|activations| — i.e. extreme for ≥ 50% of inputs. Best effect-side nulls in the
set: ablate-random and only-random matched for count over **10 seeds**, plus a
**genuine permutation p-value** (p ≈ 0) for the overlap between ODs and the
MLP-singular-vector/LayerNorm spikes that produce them. Coverage is stated
honestly: **the mechanism holds for 5 of 8 models** (opt-13b, gemma-9b and
stable-12b diverge). Appendix H is directly relevant to §3 below: OD ablation
stops pythia-12b predicting "a" after **5** dimensions and "," after 20, but
**repetitions only start at 100**; mistral-7b and llama-8b loop at 10; one needs
**2000–3000 random** dimensions to get comparable damage. Training dynamics:
pythia-12b only, ODs appear at steps 3000–4000, and **only 7 of 22 ODs at step
3000 survive to the final checkpoint**.

**He et al. 2024, *Understanding and Minimising Outlier Features in Transformer
Training*** (NeurIPS 2024, peer-reviewed). **The measurement design to copy.**
Two metrics with **no threshold at all**: kurtosis of the per-neuron activation
RMS, bounded in [1, d] with 1 = no outliers; and max-median ratio, ≥ 1. Both
scale- and architecture-invariant, measured on the residual stream, anchored to
the **initialisation value** and validated externally against quantisation
error. Causal work is the bulk: Norm type, an unnormalised Outlier-Protected
block matching Pre-LN loss to 7B while cutting kurtosis by up to 4 orders of
magnitude, LR, **Adam ε (monotone, 3 seeds)**, and **non-diagonal
preconditioners (SOAP/Shampoo) which greatly reduce outliers even with Pre-Norm**.
Footnote 10 is the synthesis: extreme outliers come from an **interaction** of
Pre-Norm with diagonal adaptive preconditioning. Also: outlier emergence is
**not monotonic in training time** under Pre-LN but is under their OP block —
the emergence *shape* is architecture-dependent. Directly tests and undercuts
Dettmers' 6.7B cutoff (7B model, peak kurtosis < 10 at matched loss).

**Zhao et al. 2025 (AWS), *Emergent Outlier Properties in T5*** (NAACL 2025,
peer-reviewed). **The only encoder-decoder study.** T5/Flan-T5, 60M–11B.
Criterion: γ beyond 3σ of the per-layer distribution (T5's RMSNorm has no bias),
then rank dimensions by how many layers flag them and disable the top 4/20/100.
Zeroing **4 of 1024 dimensions in T5-11B costs 14.7% absolute**. Findings that
matter here: **encoder and decoder outlier dimensions are disjoint, and decoders
have more**; outlier magnitude tracks **layer depth, not model size** — at fixed
depth, magnitude *decreases* from T5-Large → 3B → 11B, which directly qualifies
Dettmers; and Flan-T5-XXL is *less* outlier-sensitive than smaller Flan-T5s.
**Names pretraining dynamics of outlier formation as an explicit open problem.**
No MT task, despite T5's pretraining including translation.

**Xu 2026, *When Do Attention Circuits Form?*** (arXiv:2606.02378; preprint,
single author, leaning on unpublished companion papers — treat with care).
Checkpoint-resolved sink formation in Pythia-1B, OLMo-1B and OLMoE-1B-7B.
**The only paper in either folder with an empirical null distribution**: §8.1
computes induction selectivity against a **random non-special target position**,
500 null draws per revision, and reports that **null-p99 is stable across
training (1.45–2.08 over 143k steps) while null-max is not (7.3 → 767)** — so a
single per-model threshold suffices and max-based rules do not. Substantive
findings: layers 0–1 **never** host a sink head at any checkpoint; sinks emerge
mid-layer-outward; the emergence *shape* differs between OLMo (7.4% → 70.3%
between adjacent checkpoints) and OLMoE (smooth) despite shared data; induction
forms 10–20× earlier in tokens than the sink. **Sinks only accrete — zero
dropouts at any threshold — while induction circuits turn over heavily
(Jaccard 0.29–0.33).** Single seed, and blunt about it.

**Hodgkinson, Wang & Mahoney 2025, *Models of Heavy-Tailed Mechanistic
Universality*** (ICML 2025, peer-reviewed). The RMT side of the null question.
The HTMP ensemble models heavy-tailed **eigenvalue spectra** of weight and
feature matrices, with a fitted repulsion parameter κ and a KS goodness-of-fit
test on the inverse-Gamma part. **The decisive point for a super-weight
project:** the paper states repeatedly that "the elements of HTMP models need
not have heavy-tailed behavior", and that empirically, while eigenvalues of
modern weight matrices are heavy-tailed, **elements are not**. It provides no
distribution against which to test whether one scalar entry is an outlier — if
anything it argues against borrowing spectral heavy-tailedness to justify
entrywise heavy-tailedness. Largest model 11.1M, CNNs on CIFAR-10.

**Ding 2026 ×2** (arXiv:2605.18898, 2606.19367; preprints, single independent
author). Paper 1 fits a two-parameter Weibull to |W| **per matrix, per layer,
per checkpoint**, and its genuinely valuable contribution is an **analytically
derived random-init anchor**: i.i.d. Gaussian init ⟹ |w| ~ HalfNormal ⟹ their
fit protocol gives **k₀ ≈ 1.205, independent of σ_init**, verified empirically
at step 0 as **k = 1.205 ± 0.001**. That is a real calibrated null *for the bulk
shape of a weight matrix*. But the protocol **discards the top and bottom 10% of
|W| before fitting** — by design, "anti-interference" — so it structurally
cannot see a super weight. Its tail statistics (max/q99, kurtosis, and an
undefined `log₁₀ P_tail` reaching −44.7) are descriptive and uncalibrated, with
no multiplicity correction for ~10⁷ draws. Paper 2 decomposes the AdamW
weight-scale force budget with ground-truth optimiser moments and is the only
paper in this folder with **real seed replication (n = 4)** plus LR and
architecture sweeps; alignment contributes 88–94% of the force budget in the
rise phase. Neither is a per-weight detector.
*(These two were read by a delegated pass this session, not by me directly.)*

**Gallego-Feliciano et al. 2025, *Hidden Dynamics of Massive Activations***
(arXiv:2508.03616; preprint). The only trajectory study across the full Pythia
suite (9 sizes, 14M–12B, ≥37 of 154 checkpoints). Fits a 5-parameter
exponentially-decaying log-modulated curve to the top-1/median ratio, mean
R² = 0.984 over 188 layers, and predicts some parameters from architecture
(K: R² 0.847; λ 0.664; **γ 0.055**, i.e. timing is essentially unpredictable).
Two useful facts: **Sun's criterion does not generalise below ~1B** (Pythia-14M's
layer-3 spike has a top-1/median ratio of 83, not 1000), and **step 0 is a
genuine random-init null** — MAs are absent at initialisation.
**Warnings:** zero seeds (Pythia ships one run per size); n = 188 treats layers
as independent; nothing is intervened on, yet the paper uses causal "control"
language throughout; and it contains at least two numeric self-contradictions
plus a garbled reference ([7] merges GPTQ and LLM.int8() and calls them 1-bit).
Use the fits; distrust the rhetoric and the bibliography.

### 1.5 Beyond decoder-only LLMs

**Darcet et al. 2023, *Vision Transformers Need Registers*** (ICLR 2024,
peer-reviewed). High-norm patch tokens (criterion: **norm > 150**, hand-picked
off a bimodal histogram, ≈2.37% of tokens, ~10× normal) appear in redundant
patches, hold *less* local information and *more* global information (Table 1:
single-patch linear probe on Flowers, 59.5 normal vs **99.6** outlier). Present
in DINOv2 ≥ ViT-L, DeiT-III, OpenCLIP; **absent in DINO ViT-B/16, in DINOv2
below ViT-L, and in MAE**. The register fix is causal — retrained variants, no
performance regression, artifacts fully migrate into the registers, and **one
register suffices**. Sun-2024 §5 reinterprets registers as the same fixed-bias
mechanism rather than global-information aggregation.

**Pierro & Abreu 2024, *Mamba-PTQ*** (ICML 2024 workshop, non-archival,
self-described preliminary). Outlier **channels** do exist in an attention-free
architecture: criterion is absmax activation beyond **6σ** of the layer mean,
<1% of channels, mostly consistent across layers in `in_proj`. Zeroing only the
`in_proj` outlier channels drives Mamba-130m LAMBADA to **0.00%** and the 6-task
average 52.64 → 39.37, with `dt` as a near-null control (51.91). **Scope
caution:** this refutes "softmax is necessary for outlier *channels*". It says
nothing about position-wise sinks or massive activations, which it never
measures.

**Hämmerl et al. 2023, *Anisotropy and Outliers in Multilingual LMs***
(arXiv:2306.00458; preprint). The only multilingual outlier study in either
folder. XLM-R, **cased** mBERT, and multilingual S-BERT; 36 Tatoeba languages,
6 Wikipedia languages. **Contradicts the widely-repeated claim that mBERT has no
outliers** — they find 3 (dims 227, 195, 731 at layer 8), and attribute the
discrepancy to the *cased* checkpoint plus sentence-level rather than word-level
representations. XLM-R is dominated by a single dimension **588** (cosine
contribution 0.77 at layer 8, **0.89** at the output layer) and is anisotropy-
saturated at 0.995–0.997 in every language. mBERT is not: anisotropy ranges
0.49 (en) to 0.69 (sw). Zeroing 15–18 dimensions raises Tatoeba accuracy
50.35 → 60.09 and BUCC F1 59.1 → 64.4, still far short of the S-BERT reference
(85.17).
- **§6.4 is the field's one explicit admission that 3σ has no null:** after
  whitening to near-zero anisotropy, "applying the outlier definition of three
  times the standard deviation, we **still find outlier dimensions** … Therefore,
  we do not consider these dimensions true outliers. In an (artificially) highly
  isotropic space, the traditional outlier definition … may simply not apply."
- Their tokenizer/script attributions (Arabic vs Turkish, Swahili) are
  self-labelled speculation; **no tokenizer statistics were measured**.
- Data-hygiene flag: §5 says "18 dimensions" but footnote 3 lists **15**.

**Mutisya & Mugane 2026, *Attention Sinks in Massively Multilingual NMT***
(arXiv:2605.01229; preprint; filed in ``).
NLLB-200 600M **cross-attention**: `</s>`, language-identifier tags and
punctuation take **83–91%** of the mass across four African languages, validated
on four more. Attributes the cause to **vocabulary design rather than positional
bias**. Practical consequence: uncorrected cross-attention similarity
underestimates content similarity by ~2× (36.7% raw vs 70.7% filtered).
**Important scope limit: this is a cross-attention measurement-artifact paper.
It does not report massive activations, hidden-state outliers or super weights
in NLLB.** *(inference: whether NLLB has a super weight at all is unmeasured.)*

### 1.6 Repetition and confidence (all filed in ``)

**Cancedda 2024, *Spectral Filters, Dark Signals, and Attention Sinks***
(arXiv:2402.09221; Meta tech report, **no venue**). The most important paper in
this folder for §3. Partitions the right singular vectors of `W_U` into 20 bands;
"U-dark" is the bottom 5%, where singular values collapse. **An early-layer MLP
writes a huge, almost entirely U-dark vector into the BOS residual stream — L3 in
LLaMA2-13B, L1 in 7B, L2 and L8 in 70B — and that vector is the attention sink.**
Causal: nested spectral filters with a random-orthonormal-basis control, plus a
**shavings swap** that exchanges the filtered component with the same component
from a different sample, which (because the BOS stream is input-independent)
perturbs every stream *except* BOS and abolishes the effect. A sink-preserving
filter suppresses 25% of the singular values while raising NLL only 2.47 → 2.74.
**§6.1 and Appendix D Table 7 are the generations that match this project's
failure mode almost exactly** — see §3.0. Metrics worth reusing: the U-dark ratio
(Eq. 5), the read/write aptitude of a parameter matrix into a spectral band
(Eq. 2), and the BOS-norm-by-layer plot split into dark/light and MLP/MHA
(Fig. 9). Limitations: LLaMA2 only, no instruction tuning, other scripts untested.

**Stolfo et al. 2024, *Confidence Regulation Neurons in Language Models***
(arXiv:2406.16254; **NeurIPS 2024, peer-reviewed**). Supplies most of §3's
tooling. **Entropy neurons** write into an effective **null space of `W_U`** (in
GPT-2 Small the singular values collapse around index 755 of 768), adding norm to
the residual stream, inflating the final LayerNorm denominator, and uniformly
scaling logits *down* — raising entropy with minimal change to the argmax.
**Token-frequency neurons** move the output toward or away from the unigram
distribution along a centred log-frequency direction `v_freq`. Causal throughout:
the LN-frozen mediation ablation (Eqs. 4–6), the frequency-frozen ablation
(Eqs. 7–8), clip-ablation during induction (up to a **70% reduction in entropy**),
and BOS-ablation of induction heads with matched controls. Two facts that bite on
this project: **the LN-mediated fraction varies from ~40% (Pythia, Gemma) to ~80%
(GPT-2, LLaMA2, Phi-2)**, so H2's size is model-specific; and §5 finds
Pythia-410M is *baseline-biased toward common tokens*, needing a neuron that
actively suppresses the frequency direction to stay calibrated. Final-layer only,
by design (footnote 4); English only.

**Voita et al. 2023, *Neurons in LLMs: Dead, N-gram, Positional***
(arXiv:2309.04827; **preprint**, no venue). Observational study of OPT FFN
neurons. The finding that matters here: projecting `W_out` rows onto the
vocabulary and reading **both** ends shows that for **>80% of token-detecting
neurons the update points away from the tokens that trigger them** — active
suppression is a documented motif, which is what H3 needs. Also: >70% dead
neurons in some layers of OPT-66B, sparsity confined to the first half of the
network. **No ablation of any neuron, and repetition is not addressed.** The
reusable tool is the *signed* logit lens (§4.4, Fig. 6).

**Hiraoka & Inui 2024, *Repetition Neurons*** (arXiv:2410.13497; **preprint**).
Detects "repetition neurons" by contrasting mean FFN activation in a window after
vs before repetition onset (Δ_n, Eq. 3) over 1,000 generated repetitive texts per
model. Bimodal layer distribution: most in the **final** layer, secondary peak
mid-network. Causal in both directions — deactivating them cuts repetitive
samples by up to 25–35% (so ~30% of the problem), amplifying them induces
repetition. **The sharpest discriminator they provide:** after amplification the
model finishes its current sentence and *then* starts copying earlier output, so
these neurons encode *copying previous output*, not *emitting easily-repeated
tokens*. Appendix D hints at language specificity (Japanese LLM-jp-3-1.8B behaves
differently), but the evidence is confounded by evaluating it on English
WikiText-2. Their own open question: "What causes the rest 70%?"

**Fu et al. 2021, *A Theoretical Analysis of the Repetition Problem in Text
Generation*** (arXiv:2012.14660; **AAAI 2021, peer-reviewed**). Models generation
as a Markov chain, defines Average Repetition Probability, and derives bounds.
Two results matter here. **Under greedy decoding `ζn = 1`, so ARP can diverge** —
which is why this project's failure mode shows up under greedy generation
specifically. And Corollary 1.2 decomposes the bound into inflow and outflow
terms and concludes: *"it is **not the high-frequency words, but the high inflow
words** that really lead to repetition generation"* — a direct, falsifiable
disagreement with the frequency account, adjudicable in this project's setting.
Supplies rep-w / rep-n / rep-r (§4.1) as repetition metrics.

**Also relevant, read only as citations this session** `[unopened]`: Yona et al.
2025, *Interpreting the Repeated Token Phenomenon* (ICML 2025,
`phenomenon/Interpreting-2025-Repeated-Token-Phenomenon.pdf`) — Qiu-2025 §4.3 and Li-2026 §6
both cite it for the claim that an early layer marks the first token and a later
neuron amplifies it, and that repeated tokens acquire BOS-like extreme norms;
and Xu et al. 2022, *Learning to Break the Loop*
(`Xu-2022-Learning-to-Break-the-Loop.pdf`) for the self-reinforcement result.
Both bear directly on §3 and should be opened before that section is acted on.

---

## §2 Settled vs disputed

| Claim | Status | Best evidence for | Best evidence against |
|---|---|---|---|
| Massive activations exist, are input-agnostic, emerge abruptly at one early layer and propagate by the residual | **Settled** | Sun-2024 §2.1–2.2 Fig. 4; Yu-2024 Fig. 4; Sun-2026 §3.1.1 Table 1; project finding (c) | — |
| Within a trained model they behave as a **fixed bias**, not an input-dependent signal | **Settled** | Sun-2024 Table 3 (zero → `inf`, mean → no change); **Owen-2025 Table 1: set-to-mean is harmless in all 16 models** | — |
| An early `down_proj` / GLU intermediate neuron is the locus | **Settled** | Yu-2024 §3.1; Sun-2026 §3.1.2; An-2025 §4.2 Table 1 (100% feature alignment); Oh-2024 §2.2; Li-2026 §4.2 | — |
| Sinks and massive activations co-occur | **Settled** | universal | — |
| **Every** massive activation is critical | **False** | Sun-2024, Yu-2024 | **Owen-2025 Table 1: ~7 of 16 models unaffected by zeroing; OLMo-2-7B has none at all**; Chen-2026-Measuring §3.2 (4 of 24 fail Sun's criterion) |
| **A single top-magnitude weight is catastrophic** | **False as stated** | Yu-2024 Table 1 (but their detector is activation-based) | **Subramanian-2026 Table 8: top-magnitude identification gives literally no change in 4 models**; project finding (b): 16 of 21 coordinates individually inert |
| Massive activations **cause** attention sinks | **Disputed** | Queipo-2025 §3.3 (ablating the layer-0 MLP→BOS contribution kills the sink, post-hoc, in a trained model); Sun-2024 §4 | Sun-2026 Tables 4–5 (sandwich-norm / QK-norm / DyT suppress spikes, sinks survive); **Qiu-2025 Table 4 row 5 (value-projection gate: M-Act 1053 → 125, F-Attn 0.467 → 0.297)**; Chen-2026-Gradient Figs. 6–7 (V-scale); Owen-2025 Table 2 (DyT kills MAs, attention concentration persists) |
| Sinks can be removed without loss | **Disputed, scale-dependent** | Gu-2025 §7.4 (sigmoid w/o normalisation, replicated at 1B); Qiu-2025 (gated attention *improves* PPL at 15B-A2B on 3.5T tokens); Li-2026 (head-wise RMSNorm, 4 seeds, better loss at 152M) | Zuhri-2025: softpick is free at 340M but costs 5 accuracy points and +2.8 ppl at 1.8B — "does not seem to scale well"; RanMilo-2026-Necessity Thm 1–2 (under sum-to-one normalisation a trigger-conditional task *forces* a sink) |
| Softmax is the cause | **Partly** | An-2025 App.; Bondarenko-2023; RanMilo-2026-Necessity | **Kaul-2024: softmax-1 removes first-token dominance but leaves outlier activations**; Pierro-2024 (outlier channels exist in attention-free Mamba) |
| The **adaptive optimiser** is a cause of the activation outliers | **Well supported** | Kaul-2024 (OrthoAdam); Guo-2024 Fig. 7c (Adam → SGD removes residual peaks, sinks remain); **He-2024** (Adam ε monotone, 3 seeds; SOAP/Shampoo) | — |
| Emergence is sudden at ~6.7B parameters | **Superseded** | Dettmers-2022 Fig. 3a | Dettmers' own Fig. 3b (smooth in perplexity); He-2024 (7B with peak kurtosis < 10 at matched loss); Amazon-2025 (magnitude tracks *depth*; at fixed depth it *decreases* from T5-Large → 11B) |
| Ablating outliers shifts predictions toward **frequent** tokens | **Disputed, and it may be an encoder/decoder split** | Puccetti-2022 §4.1 (BERT); Yu-2024 Fig. 5 (stopwords up) | **Macocco-2025** (8 decoders: predictions get *rarer*, slope 1.35 → 0.74). Macocco states the conflict and blames encoder-vs-decoder plus different criteria. **Unresolved.** |
| Detection has a calibrated criterion or null | **No.** See §4.1 | — | Every threshold in the literature is hand-set: Dettmers ≥6.0/25%/6% (reverse-engineered to give one detection at 125M); Sun >100 and ×1000; Kovaleva/Puccetti/Amazon 3σ (relaxed per model); Macocco top-1%/median; Darcet norm > 150; Pierro 6σ; Jin λ=5; Li-2026 read off a figure. **Hämmerl §6.4 is the one explicit admission** that 3σ fires on isotropic noise. |
| Zero-ablation measures content dependence | **No — it is an upper bound** | — | **Owen-2025** (mean harmless, zero catastrophic, same coordinates); **Parodi-2026** (ViT registers: zero −36.6 pp, mean/noise/shuffle all within ~1 pp); Sun-2024's own zero-vs-mean |
| Language dependence of the outlier structure | **Unknown** | Hämmerl-2023 is the only study, and it *contradicts* the prior claim that mBERT has no outliers | Nothing else exists. Puccetti, Macocco, He, Dettmers, Amazon, Gu, Sun ×2, Yu, Queipo: all English. Su-2026's 103-page survey cites the one NLLB paper in a table and never discusses it. |
| Massive activations occur outside decoder-only LLMs | **Settled** | Sun-2024 §5 and Darcet-2023 (ViTs); Su-2025 (MoE); Pierro-2024 (Mamba, channels only) | Absent in MAE, in DINO ViT-B/16, and in DINOv2 below ViT-L (Darcet §2.1, App. E) |

**The asymmetry worth stating on its own.** Across every from-scratch
intervention read here, **no one has removed attention sinks while keeping
massive activations.** Interventions either kill the massive activations and
leave sinks intact (Qiu's value-projection gate G2, Chen's V-scale, Owen's DyT,
Sun-2026's sandwich/QK-norm) or kill both (softpick, Qiu's SDPA gate G1,
head-wise RMSNorm, sigmoid-without-normalisation). *(inference: this is
consistent with sinks being the more fundamental object and massive activations
being one implementation of the gate they need — which is exactly Fesser's
Lemma 2. It is not consistent with the plain reading of Sun-2024/Yu-2024 that
massive activations are the cause.)*

---

## §3 The repetition-collapse question

**What the literature actually offers.** Yu et al. **report** Fig. 5 and explain
nothing: §3.2 is two paragraphs of description plus a one-prompt case study, and
Appendix A.3 just widens the token axis to the top 50. No paper in either folder
explains Yu's Fig. 5. Three papers *reproduce* the failure mode causally without
analysing it — Oh-2024 ("the model repeats the same text as the user prompt"),
Su-2025 (repetitive output on Math-500 after pruning 3 of 6,144 experts), and
Jin-2025 (2-gram diversity 0.921 → 0.668, and for Gemma2-9B on IMDB
0.809 → **0.085**). **One paper produces almost exactly this project's
generations, causally, and offers a mechanism: Cancedda 2024.**

### 3.0 The one near-exact precedent

Cancedda §6.1, on filtering the dark subspace out of LLaMA2-13B at layer 3:
*"We notice that the application of Ψ filters often results in the model
entering repetitive patterns. This is consistent with the possibility that
attention heads largely copy representations from the RSs of previous tokens,
and inhibiting attention sinking results in **over-copying**."* His Table 2 and
Appendix D Table 7 generations read:

> `"The first "tone" is the one, the first, the one, the first, the S, the first, the, the, the, the, the, the, the, ..."`
> `", in the, The , in the, the , in the, The , the , the, and , the, the, the, the, ..."`

That is OLMo-1B's "We. We. We." and Mistral's "the main. without without. . . ."
in a different model. And the object he removes is a **large-norm, almost
entirely dark-subspace vector written into the BOS residual stream by an
early-layer MLP** — L3 in LLaMA2-13B, **L1 in 7B**, L2 and L8 in 70B. *(inference:
structurally this is the same object as a super weight, described in the
spectral basis instead of the coordinate basis.)*

### 3.1 Candidate hypotheses, with the discriminating measurement for each

**H1 — Sink destruction → over-copying.** The super weight builds the
sink-supporting vector; without it, heads that should be dormant have nowhere to
dump attention and instead copy from nearby tokens.
*Predicts:* BOS/sink attention mass collapses; per-head attention entropy falls
(sharper, more local); the degeneration is copy-shaped (repeating *recent
context*), not prior-shaped.
*Measure:* Gu's `Sink^ε_1` (ε = 0.3, T = 64) and Su-2025's Attention Sink Decay
Rate (Eq. 7) before/after; per-head attention entropy; Cancedda's BOS
residual-stream norm by layer split into dark/light and MLP/MHA contributions
(his Fig. 9).
*Status in the literature:* **best supported.** Cancedda §6.1 causally; Su-2025
observes ~90% sink decay and repetition from the same intervention;
Queipo-2025 §3.3 shows the ablation kills the sink; Barbero Table 3 shows what
losing the sink costs (Gemma 7B HellaSwag 80.61 → 27.35).
*Caution:* Su-2025 never tests whether the sink collapse **causes** the
repetition; both are downstream of the same cut.

**H2 — Confidence / final-norm-scale dysregulation.** The massive activation
dominates the final RMSNorm denominator; removing it rescales all logits.
*Predicts, and this is the useful part:* **the wrong sign.** Stolfo §3.3 shows
*adding* norm flattens the distribution, so *removing* norm should **sharpen**
it — and Stolfo §6.1 measures the lowering direction directly (clip-ablating
GPT-2's neuron 11.2378 gives up to a **70% reduction in entropy**). Stolfo also
stresses that entropy neurons change entropy "with minimal impact on the
prediction", i.e. the argmax is preserved. Yu observes the opposite: a **flat**
distribution whose argmax has **moved** to "the" at 9.0%.
*Measure — the single most decisive experiment available:* Stolfo's Eq. 6
applied to a super weight. Ablate the weight but **freeze the final
RMSNorm denominator at its unablated value**, and report `1 − DE_LN/TE`. If the
degeneration survives with the scale frozen, H2 is dead. If it vanishes, H2 is
the whole story.
*Status:* **probably not sufficient alone**, by the sign argument above. But
Stolfo's LN-mediated fraction varies from ~40% (Pythia, Gemma) to ~80% (GPT-2,
LLaMA2, Phi-2), so it is likely a real contributor whose size is model-specific.

**H3 — Loss of an active stopword suppressor.** Yu's own words are that super
weights "**suppress** stopword likelihood". If the super weight's `down_proj`
column writes a direction that is negative on function words, zeroing it simply
releases them.
*Measure — the cheapest decisive test, roughly ten lines of code:* Voita §4.4's
**signed logit lens**. Project the super weight's `down_proj` output column
through `W_U` and read **both ends** of the vocabulary ranking, not just the top.
Voita found that for **>80% of token-detecting neurons** the update points
*away* from the tokens that trigger them, so active suppression is a documented
motif. Then correlate that column's logit projection with centred log-unigram
frequency.
*Status:* untested by anyone. It is the most direct restatement of Yu's own
claim and nobody has run it.

**H4 — Fallback to the unigram prior.** The output collapses toward `P_freq`,
whose mode is exactly "the", ".", ",".
*Measure:* Stolfo §4. Build `v_freq` = centred log-unigram vector over the
pretraining corpus. Decompose Δlogits from the ablation onto (i) the all-ones
direction, (ii) `v_freq`, (iii) the residual — **this one regression separates
H2, H4 and "something else" at once**. Then report `|Δ D_KL(P_freq ‖ P_model)|`,
and Stolfo's frequency-frozen ablation (Eq. 7).
*Status:* strongly suggested but **contested in a way this project can
adjudicate.** Macocco's decoders move *away* from frequent tokens when outlier
dimensions are ablated, while Yu's models move *toward* them. Stolfo §5 adds the
twist that Pythia-410M is baseline-biased toward common tokens and needs a
neuron actively suppressing the frequency direction to stay calibrated.

**H5 — High-inflow loop dynamics under greedy decoding.** Fu 2021 §2.1: under
greedy decoding `ζn = 1`, so the average repetition probability can diverge; and
Corollary 1.2 attributes repetition to **high-inflow** words — words many
predecessors assign high probability — explicitly *not* to high-frequency words.
*Predicts:* the collapse largely disappears under temperature/stochastic
sampling; and *inflow*, not unigram frequency, predicts which tokens get
repeated.
*Measure:* re-run the ablated models at T > 0 and with nucleus sampling; and
rank the repeated tokens by unigram frequency vs by `inflow(w) = Σ_v P(w|v)`
estimated from the model. **Fu and Stolfo directly disagree on whether frequency
is the right variable, and this project's setting can settle it.**
*Status:* partly answerable already — Yu's own case study reports the top-1
becoming "the" at 9.0% on the **first** generated token, so the *distributional*
damage is immediate and is not itself a loop. Loops are plausibly a second-order
consequence. Worth confirming, cheaply.

**H6 — Off-manifold artifact of zeroing.** The intervention pushes the residual
stream out of distribution; any comparably large perturbation would look like
this.
*Measure:* the control triad. Replace the super weight with (a) 0, (b) the mean
or median of its row/column, (c) a random weight of matched magnitude from
elsewhere in the same matrix; and add Cancedda's **shavings swap** — replace the
weight's *contribution* with the same contribution computed on a different
input. Because the BOS residual stream is input-independent, a swap perturbs
everything except BOS, so if degeneration appears under zeroing but not under
swap, the BOS/sink pathway is implicated rather than the generic loss of a
vector.
*Status:* **this is not a fringe worry.** Owen-2025 shows set-to-mean is harmless
in all 16 models where set-to-zero is catastrophic in 9; Parodi-2026 shows the
same shape in ViTs (zero −36.6 pp, mean/noise/shuffle within ~1 pp). Jin-2025 is
the useful counterpoint: when the perturbation stays *within* a vector's
coordinates rather than nulling it, zero and mean agree (Table 2).

**H7 — Downstream repetition neurons.** Hiraoka's Δ_n-selected neurons fire once
a copy signal appears.
*Measure:* run Hiraoka Eq. 3 with the ablation point as "onset"; then test
whether deactivating the top-Δ_n neurons partially rescues generation. Their
ceiling on natural repetition is ~30%, so expect a partial effect at best.
*Status:* a secondary check, not a primary explanation.

### 3.2 What the literature favours, and why

**H1 + H3 in combination**, with H6 as a live confound that must be measured
first. The reasoning: Cancedda is the only causal reproduction of this exact
generation pattern and his mechanism is sink destruction; Su-2025 independently
pairs sink collapse with repetition under a super-weight-like ablation; and
Yu's own framing (stopword *suppression*) is H3, which nobody has tested. H2 is
disfavoured on a sign argument. H5 is likely a second-order amplifier under
greedy decoding rather than the cause.

**One metric caution, from Jin-2025 Appendix F.** Perplexity does not detect
this failure mode: their looping example
(`"divisible by 9 and 12, which is the product of X, if it is divisible by 9 and 12..."`)
has **perplexity 2.99**. Report n-gram diversity (Fu's rep-w / rep-n / rep-r,
formulas in his §4.1) or a readability judgement alongside perplexity, not
perplexity alone.

---

## §4 Open problems

Ordered by how much this project's existing findings (b)–(e) advance them.

### 4.1 A calibrated detector for "this weight is special"

**Problem.** There is no test, in any paper in either folder, that assigns a
p-value or a null distribution to the claim that a *specific scalar weight or
activation* is an outlier.

**Why it is open.** Every criterion is hand-set, and several authors say so.
Dettmers §4.1 chose ≥6.0 / 25% / 6% so that the smallest model in the study
yielded exactly one detection. Kovaleva relaxed 3σ to 2σ for RoBERTa and half-of-
layers to a third for BERT-large "to make detection work". Li-2026 reads super
neurons off Fig. 8 with no criterion at all. **Hämmerl §6.4 is the field's one
honest statement of the consequence:** after whitening to near-zero anisotropy,
"applying the outlier definition of three times the standard deviation, we still
find outlier dimensions … Therefore, we do not consider these dimensions true
outliers." The nearest things to nulls are all for the *wrong object*:
Hodgkinson's HTMP calibrates **eigenvalue** spectra and states repeatedly that
"the elements of HTMP models need not have heavy-tailed behavior"; Ding-2026a's
k₀ = 1.205 half-Normal anchor is a real, analytically derived random-init null
but its fit **discards the top 10% of |W| by design**; Gallego-Feliciano's step 0
is a free random-init null used only qualitatively; and Xu-2026 §8.1 is the only
empirical null distribution anywhere, but it is for attention selectivity.
Su-2026's survey names the missing **standardised benchmark** as an open problem
(§7.1–7.2).

**Experiment (feasible).** Build the null the field lacks, for one matrix at a
time. For each `mlp.down_proj`: (i) fit Ding's protocol to get the bulk shape,
but **without the top-decile trim**, and compare to the untrimmed and trimmed
fits; (ii) generate the null by the two cheapest routes available — the
random-init checkpoint (step 0, free in Pythia/OLMo) and a within-matrix
permutation; (iii) report the observed maximum's rank and an
**exceedance probability with a Bonferroni or FDR correction for the ~10⁷ entries
in the matrix**, which no paper does; (iv) validate the detector by whether it
recovers Yu's Table 2 coordinates *and* separates the 5 catastrophic from the 16
inert ones in the project's finding (b).

**Satisfied when.** A criterion with a stated false-positive rate that (a) flags
Yu's coordinates, (b) does **not** flag the top-magnitude coordinates that
Subramanian's Table 8 shows are inert, and (c) is stable across the checkpoints
of one model.

**Risks.** The honest possibility is that **magnitude is simply the wrong
statistic** — Subramanian already showed top-magnitude identification transfers
to nothing, and the project's finding (a) shows Yu's coordinates rank 1–6 by |W|
but finding (b) shows 16 of 21 are individually inert. A detector calibrated on
|W| may be well-calibrated and useless. The fallback, which is more likely to
work, is to calibrate the *activation-spike* detector Yu actually used, where the
null is easier (permute the intermediate-neuron index).

### 4.2 The unit of criticality: the scalar or the intermediate neuron?

**Problem.** The literature has four incompatible units for the same phenomenon:
Yu's `down_proj` **scalar**, Oh-2024's rows of `W_gate`/`W_up` feeding one
**intermediate neuron**, Li-2026's **super neuron**, and Su-2025's **expert**.
Nobody has tested which one predicts criticality.

**Why it is open, and why this project is ahead.** The project's finding (b) is
already the sharpest evidence on this question that exists: 16 of 21 of Yu's
coordinates are individually inert, but each model's candidate *set* collapses it
(Phi-3-mini ×671 jointly vs ×1.0 each; Llama-3.1-8B-Instruct ×47,488 jointly vs
×1.02 each), and the members share a `down_proj` input column. That converges
independently on Oh-2024's unit from the other side of the matrix. It also
**explains Subramanian's Table 8 negative**, which they left as "high magnitude
alone is insufficient": they zeroed **2** coordinates in Llama-3.1-8B-Instruct
and Llama-3.2-1B and got nothing — and Su-2025's Table 8 lists Llama-3.2-1B as
having **four** super weights, `(400,1417) (698,1417) (2029,1417) (1159,1417)`,
all on intermediate neuron **1417**. *(inference: Subramanian cut two spokes of a
four-spoke wheel.)*

Note that Yu's Table 2 contains **both** geometries and the project should not
over-generalise: Phi-3-mini shares an intermediate neuron within each layer
(808 at L2, 2723 at L4), while Llama-13B/30B and OLMo-1B/7B share the *output*
channel instead (2231, 5633, 1764, 269).

**Experiment.** Ablate at three granularities on the same models — the scalar,
the whole `down_proj` input column (the intermediate neuron), and the
corresponding `W_gate`/`W_up` rows — and cross the three with the control triad
from §5.2. Then test the prediction directly: **does the number of coordinates
sharing an intermediate neuron predict the individual-vs-joint gap?**

**Satisfied when.** A stated rule of the form "criticality is a property of the
intermediate neuron; a scalar is catastrophic iff it carries most of that
neuron's fan-out" that holds across the ~10 models the project already has, with
the shared-output-channel models correctly predicted to behave differently.

**Risks.** Ablating a whole column is a much larger perturbation than a scalar,
so H6 bites hardest here — the control triad is not optional. And Li-2026 has
already described the row-basis version of this object in a peer-reviewed ICML
paper without citing Yu; a literature check before writing is warranted.

### 4.3 Does the sink collapse *cause* the repetition?

**Problem.** Su-2025 shows ~90% sink decay and repetitive generation from one
intervention but never links them; Cancedda argues the link but in a different
formalism (spectral filtering) and only on LLaMA2; Yu reports the stopword shift
with no mechanism at all.

**Experiment.** On the 5 models where the project already has a catastrophic
individual ablation and the models where it has a catastrophic joint one:
measure `Sink^ε_1`, per-head attention entropy, and the Δlogit decomposition onto
`{1, v_freq, residual}` before and after; then run the two dissociations —
(i) **restore the sink without restoring the weight**, by adding an explicit
learnable-free key bias at the sink position or by patching the massive
activation back in at the onset layer (Yu's own *Prune SW, +SA* condition, which
recovered only 42%, is the seed of this and should be re-run with generation
rather than accuracy as the readout); and (ii) **destroy the sink without
touching the weight**, e.g. by removing BOS as Barbero does, and check whether
the *same* stopword signature appears.

**Satisfied when.** Either the stopword signature is reproduced by sink removal
alone and abolished by sink restoration alone — in which case H1 is established
— or it is not, in which case H3/H4 are promoted and the field's default
assumption is falsified.

**Risks/confounds.** Yu's own Table 1 says the super activation mediates only
42% of the accuracy loss, so a partial result should be expected and must not be
written up as a clean dissociation. And patching the activation back in at the
onset layer is itself an off-manifold intervention.

### 4.4 Formation: when does the super weight appear, and is it seed-stable?

**Problem.** Nobody has tracked a **super weight** — as opposed to a massive
activation or a sink — across pretraining checkpoints. Gallego-Feliciano tracks
the activation ratio in Pythia and is single-seed and purely observational;
Xu-2026 tracks sink *heads* and is single-seed; Guo-2024 tracks OLMo-7B's
extreme-token statistics; Amazon-2025 names outlier-formation dynamics as an
explicit open problem; Su-2026's survey says the training dynamics of
outlier-circuit emergence are "largely unexplored".

**The seed question is the sharpest sub-problem and nobody has answered it.**
Kovaleva's footnote 1 reports that HuggingFace and Google checkpoints of the
*same* BERT-base config have outliers at **different locations and opposite
signs**; Macocco reports that **only 7 of 22** outlier dimensions present at
Pythia step 3000 survive to the final checkpoint; Xu-2026 reports induction-head
Jaccard of 0.29–0.33 between formation and final but **zero dropouts** among sink
heads. So identity turnover is documented for related objects and unmeasured for
super weights.

**Experiment.** Pythia (9 sizes × 154 checkpoints) and OLMo are already public
and cheap to scan — running Yu's activation-spike detector over checkpoints is
one forward pass per checkpoint. Report: onset step, whether the coordinate is
stable once formed, whether the *intermediate neuron* is more stable than the
scalar, and — the part with a genuine null — whether step 0 ever produces a
detection. **PolyPythias** (already indexed in `../README.md`) supplies multiple
seeds at small scale, which turns this from a single-run anecdote into a
seed-varying claim, and the repo's rigor floor requires exactly that.

**Satisfied when.** An onset step with a CI across seeds, plus a statement of
whether the coordinate identity is seed-stable, seed-stable-up-to-permutation,
or not stable.

**Risks.** Sun's criterion **does not generalise below ~1B** — Gallego-Feliciano
found Pythia-14M's spike has a top-1/median ratio of 83, not 1000 — so small
Pythias need the continuous statistic, not the binary one. And Pythia is one
architecture and one corpus; Xu-2026 found OLMo and OLMoE produce opposite
emergence *shapes* on shared data, so architecture-generality should not be
assumed.

### 4.5 The multilingual / MT arm

**Problem.** The outlier-structure literature is **almost entirely English and
almost entirely decoder-only**, and the two gaps compound. Hämmerl-2023 is the
only multilingual outlier study in either folder; Amazon-2025 is the only
encoder-decoder one and runs no MT task despite T5's pretraining including
translation; Mutisya-2026 is the only NLLB paper and measures *cross-attention
mass on language tags and EOS*, not massive activations or weights. Su-2026's
103-page survey cites Mutisya once, in a table, and never discusses it.
**Whether NLLB — or any encoder-decoder MT model — has a super weight at all is
unmeasured.**

**Why this project is well placed.** Finding (c) already includes **TowerBase-7B**,
an MT-adapted continued pretrain of Llama-2-7B, with a massive activation and a
×3235 collapse under finding (e). Llama-2-7B's own super weight is `[2533, 7890]`
at layer 1 (Yu Table 2), and Sun-2024's LLaMA2-7B massive-activation channels are
**1415 and 2533** — so there is a directly checkable question with a known
answer on one side.

**Three experiments, in increasing cost.**
1. **Does continued multilingual/MT pretraining move the super weight?** Compare
   Llama-2-7B against TowerBase-7B coordinate by coordinate. If the coordinate is
   identical, the structure is inherited and frozen by adaptation — which is
   consistent with Yu's finding that instruction tuning does not move it, and
   with Su-2025's finding that post-training does not move Super Experts. If it
   moved, that is a novel finding with an obvious follow-up.
2. **Is the collapse language-dependent?** Ablate the super weight and measure
   degradation per language — perplexity on parallel text, and translation
   quality with COMET rather than chrF, on the ~55-language protocol the repo
   already has in `compression/experiments/replication-uneven-ptq/`. Puccetti's
   frequency account makes a directional prediction: **if outlier structure is
   driven by token frequency in the pretraining corpus, then in an
   English-dominated multilingual model the outlier should be tuned to English
   token statistics, and ablating it should hurt low-resource and non-Latin-script
   languages more.** Hämmerl's anisotropy ordering (en 0.49, es 0.56, tr 0.60,
   su 0.64, ar 0.65, sw 0.69 in mBERT) is a ready-made ranking to test against.
   **The competing prediction is null** — if the super weight is a language-
   agnostic bias, degradation should be uniform. Either result is publishable at
   this project's scale.
3. **Does an encoder-decoder MT model have one at all?** Run Yu's activation-spike
   detector on NLLB-600M/1.3B separately in the encoder and the decoder. Amazon-
   2025's finding that **encoder and decoder outlier dimensions are disjoint, with
   decoders carrying more**, is the specific prediction to test.

**Satisfied when.** A per-language effect-size table with CIs and a stated family
size for the multiplicity correction, plus a yes/no on whether NLLB has a
detectable super weight in either stack.

**Risks/confounds.** Tokenizer fertility differs wildly by language and will
confound any per-language perplexity comparison — normalise per character or use
a translation metric. Hämmerl's own tokenizer/script attributions are
self-labelled speculation with **no tokenizer statistics measured**, so this
project should measure fertility rather than inherit her conjecture. And the
project's finding (f) already rules out the quantization framing, so this arm
must be posed as a question about *mechanism and robustness*, not about
compression.

### 4.6 Which massive activations matter, and why

**Problem.** Owen-2025 shows ~7 of 16 models are unaffected by zeroing their
massive activations, and says outright: "we have yet to identify a clear
indicator that distinguishes between detrimental and non-detrimental massive
activations." The project's finding (b) is the weight-side version of the same
puzzle.

**Experiment.** The project already has the rarest ingredient — a set of models
split by outcome (5 catastrophic, 16 inert individually, several catastrophic
only jointly). Cross that against the candidate predictors the literature
offers: the fan-out count from §4.2; Owen's layer-profile shape (sharp
rise-plateau-fall vs gradual); GLU vs non-GLU; whether the model has a BOS
convention (Owen found three Gemmas have massive activations only *with* BOS);
and Fesser's value-norm ratio at the sink.

**Satisfied when.** A predictor that classifies catastrophic from inert across
the project's ~10 models better than chance, with the family size stated.

**Risks.** n ≈ 10 models is an exploratory sample. Per the repo's rigor floor
this is not a paper claim on its own; it is a hypothesis to state with an
interval.

---

## §5 What the project's current framing should change

**5.1 "The super weight" presumes the unit.** Say **candidate coordinate** until
§4.2 is settled. Four units are live in the literature and the project's own
finding (b) points away from the scalar. Concretely: "zeroing a single scalar
collapses the model" is supported for 5 of 21 coordinates and **contradicted for
the other 16** — write the coverage into the sentence, per the repo's claim-
hygiene rule 3.

**5.2 Zero-ablation is an upper bound, and the control is not optional.** This is
the most important change. **Owen-2025 Table 1 shows set-to-mean is harmless in
all 16 models tested, including every one where set-to-zero is catastrophic.**
Sun-2024's Table 3 shows the same for the activation. Parodi-2026 shows the same
shape in vision. Any perplexity ratio the project reports from zeroing is
therefore a measurement of *removing a near-constant bias plus a distributional
shift*, and the two are not separated. **Every headline number should be reported
as a triple — zero / mean / matched-random — or explicitly labelled an upper
bound.** The project's finding (e) (zeroing the super-activation channel at
onset: OLMo-1B ×2095, TowerBase-7B ×3235) is precisely Sun's "set to zero"
condition **without** his "set to mean" control, and should not be written up
until that control is run. It is one extra forward pass.

**5.3 "Yu's coordinates are real magnitude outliers" is true and easy to
over-read.** Finding (a) is a nice replication, but Yu's detector is
**activation-based, not magnitude-based**, and Subramanian's Table 8 shows
magnitude-based identification transfers to *nothing* in four models. Keep the
two claims separate and never let (a) imply that magnitude ranking is a detector.

**5.4 Say "our hypothesis", not "the mechanism".** Per `CLAUDE.md`. This matters
more than usual here because §2 shows the field's central causal claim (massive
activations cause sinks) is actively disputed, with the from-scratch evidence
running against it.

**5.5 The quantization framing is doubly closed.** Finding (f) already killed it
in-repo, and independently, protecting super outliers during quantization **is
Yu et al.'s own headline application** (§4–5). There is nothing left there.

**5.6 Finding (c) may overstate persistence, and the discrepancy is checkable.**
The project says the massive activation "persists at constant magnitude to the
final layer". Yu's Fig. 4 agrees. But **Sun-2024 §2.1 says they "start to
diminish in the last few layers"** (LLaMA2-7B: constant from layer 2 "until layer
30" of 32), and **Sun-2026 §3.1.1 names the specific step-down blocks** —
Llama-2-7B block 62 of 64, Llama-2-13B blocks 78–79 of 80, Qwen3-8B blocks 70 and
72. Check the last two or three layers explicitly. If the project's models really
do persist to the final layer, that is a discrepancy with two papers and worth
reporting; if not, finding (c) needs rewording.

**5.7 Perplexity alone will not characterise the collapse.** Jin-2025 Appendix F
gives a fully degenerate looping generation with **perplexity 2.99**. Report
n-gram diversity or Fu's rep-w/rep-n/rep-r alongside it.

**5.8 State coverage on every cross-model claim.** Finding (c) says "every model
examined" has a persistent massive activation. Owen-2025 found **OLMo-2-1124-7B
has none at all** under the standard criterion, and Chen-2026-Measuring found 4
of 24 checkpoints fail it. Name the models examined, and check at least one
model from the negative list before writing "every".
