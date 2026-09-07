# Design memo — comparing natural repetition with lesion-induced repetition

Written 2026-09-05. This is a **design memo, not a literature review**: how the comparison in
`repetition_brief.md` would actually be built, what would break it, and which published
protocols to reuse verbatim so we are not inventing measurements.

Conventions, per `CLAUDE.md` and the brief:

- *Read* = extracted from the PDF in `papers/` with `pdftotext`, with the
  section/figure/equation number given. **[unopened]** = never opened, do not cite as evidence.
- *Inference:* = my reasoning past the papers, marked at the start of the sentence.
- Nothing under `parked_claude_2026-09-02/` is used as evidence anywhere below.
- Verbatim quotes are in quotation marks; everything else is paraphrase.

Nothing here has been run. Every number attributed to this repo is from `notes.md`,
`docs/prior_experiments_and_ideas.md` §2, or the source in `src/`, all read.

---

## §0 The hypothesis has to be restated before it can be tested

The brief's H is: *"natural repetition and lesion-induced repetition share a route — a weakening of
the sink bias on that input."* As written, **the premise is refuted a priori by three papers**, and
the design has to start by fixing that.

- Yu et al. 2024 §1 (read): the super activation "persists throughout the model at exactly the same
  magnitude and position **regardless of the prompt**".
- Sun et al. 2024 §3 (read, arXiv:2402.17762v2; the PDF prints **no venue line** — do not write
  "COLM 2024" without checking elsewhere): setting the massive activations to their empirical mean
  is harmless (Table 3: LLaMA2-7B WikiText 5.47 → 5.47; 13B 4.88 → 4.88) while setting them to zero
  is catastrophic (7B → `inf`; 13B → 5729). Their Table 2 gives the across-input variance directly:
  LLaMA2-7B Top-1 activation `2556.8 ± 141.0` over 100 RedPajama sequences — a coefficient of
  variation of ~5.5%.
- Queipo-de-Llano et al. 2025 (ICLR 2026), via `notes_training_dynamics_checkpoint_suites.md`
  (read there, paper itself **[unopened]** by me): "the layer index where these phenomena emerge is
  fixed for each model … in Pythia 410M, the transition consistently occurs at layer 5 regardless of
  input."

So **the constant cannot weaken on a particular input**. It is a fixed bias; that is the settled
finding the whole super-weight line rests on. Any version of H that predicts the constant's
*magnitude* to drop before a natural loop is predicting something the literature says does not
happen, and a null result would be uninformative because it was guaranteed.

**The restatement that is testable.** The constant is fixed; what varies per input is **how much of
it each query position draws in**. That quantity is input-dependent and there is a named mechanism
for it:

> Guo et al. 2024, Claim 1 (read, `papers/Guo-2024-Active-Dormant-Attention-Heads.pdf`),
> the **active–dormant mechanism**: "Dormant phase: On non-trigger tokens, the attn head assigns
> dominant weights to the ⟨s⟩ token, adding minimal value to the residual stream and having little
> impact on the model's output." §3.1 demonstrates it in a real model — Llama-2-7B **L16H25** is
> dormant (a sink) on Wikipedia prose and active (not a sink) on GitHub code, confirmed by the
> loss change when the head is zeroed on each domain.

So H becomes:

> **H′.** Before the first repeated token, more heads leave the dormant phase than at matched clean
> positions: the *draw* on the constant (sink attention mass, and the value update it delivers)
> falls while the constant itself is unchanged. A partial-dose lesion instead shrinks the constant
> itself. **The two therefore move on different axes**, and the experiment is only informative if it
> measures both axes separately.

*Inference:* this is also why a one-dimensional "the sink weakened" statistic cannot answer the
question. A lesion lowers the constant, which lowers the sink token's residual norm, which (per Yona
et al. 2025 §4.2, read via `notes_repetition_degeneration.md` §1.2) lowers the attention the sink
attracts — so a lesion moves *both* axes, just in a fixed order. A natural loop, under H′, moves only
one. **The dissociation is the ordering and the 2-D shape, not the sign of any single number.**

---

## §1 Definitions to pre-register

### 1.1 Loop onset

Five published definitions, all read, none directly reusable:

| source | rule | has an onset? | window |
|---|---|---|---|
| Welleck et al. 2019 §6.1 Eq. 9–10 (ICLR 2020) | `rep/ℓ` = fraction of top-1 next-token predictions occurring in the previous ℓ tokens; `seq-rep-n` = `1 − |unique n-grams|/|n-grams|` on the continuation | no — whole-sequence | — |
| Fu et al. 2021 §4.1 (AAAI 2021) | `rep-w` (Welleck's `rep/ℓ` with variable length), `rep-n` = `1 − distinct n-grams / (|s|−n+1)`, `rep-r` = fraction of token positions inside some repeated bigram (forward or backward) | no | — |
| Xu et al. 2022 §2.1 (NeurIPS 2022) | none — repetition is *constructed*: a sentence `s` repeated `N = 100` times | `n` indexes the copy | — |
| Hiraoka & Inui 2024 §2.2 | "the same 10-gram token sequence appeared three times at equal intervals within 100 tokens"; ≥50 tokens on each side | **yes** — "the point where the repeated sequence appears for the second time" | `r = 30` before / after (Eqs. 1–3) |
| Guerreiro et al. 2023 §5.1 (TACL 2023) | **TNG**: `max 4-gram count in hypothesis − max 4-gram count in source ≥ 2`, applied only to translations below a `spBLEU > 9` quality gate. `n = 4, t = 2` | no | — |

**Pre-registered rule (call it `L-onset`).** For a generated token sequence `y₁…y_T` (prompt
excluded), with `g = 4`, `k = 3`, `W_scan = 100`, `t_min = 32`:

> `t*` = the smallest `t` such that the `g`-gram starting at `t` is equal to a `g`-gram starting at
> some `t′ < t`, **and** that `g`-gram occurs at least `k` times within `[t, t + W_scan]`. A
> generation *loops* iff such a `t*` exists with `t* ≥ t_min`. `t*` is the onset.

Provenance of each constant, and the one deliberate departure:

- `k = 3` and `W_scan = 100` are Hiraoka §2.2, taken unchanged.
- Onset = the **second** occurrence is Hiraoka §2.2, taken unchanged. That is the decision point:
  the model has just chosen to re-enter a string it already emitted.
- `g = 4` rather than Hiraoka's 10, because `n = 4` is the shared convention of Welleck's
  `seq-rep-4`, Fu's tables and Guerreiro's TNG (`n = 4, t = 2`, §5.1), so `L-onset` and the
  continuous metrics agree on a substrate.
- **Hiraoka's "at equal intervals" constraint is dropped**, and this matters: the lesioned
  generations in this repo are `"the main. without without.  . . . ."` (Mistral-7B) and
  `"We. We. We."` (OLMo-1B) — read, `notes.md`. Those are *drifting* loops with unequal periods.
  Keeping the equal-interval rule would systematically fail to detect onset in the lesion arm while
  detecting it in the natural arm, which would manufacture the "different failures" answer.
- `t_min = 32` so that a pre-onset window exists at all. Hiraoka requires ≥50 tokens each side;
  we require 32 before and (see §3) 30 after, which is looser and must be stated as such.

**Report alongside, never instead of:** `seq-rep-4` (Welleck Eq. 10) and `rep-r` (Fu §4.1) as
continuous severity measures, and `TNG` (Guerreiro §5.1) whenever the prompt is a source sentence —
TNG's source-side control is the only published way to separate *self*-repetition from *copying the
input*, and Cancedda's proposed mechanism for sink loss is literally "over-copying" (§6.1, read),
so the two must not be pooled.

⚠️ Fu's `w` in `rep-w` and `n` in `rep-n` are **never stated in the paper** (checked §4.1 and
App. A.6). Do not cite a value for them; use Welleck's Eq. 9–10, which are fully specified.

### 1.2 What counts as "the sink", per model

**This must be measured per model, not assumed.** The pre-flight is three lines of code and it is
the first thing that runs.

1. **Tokenizer convention.** Record, for each model, whether `add_special_tokens=True` prepends a
   BOS-like token, its id and string, and what a chat template (if any) inserts before the first
   real token.
2. **Per-position residual norm by layer.** `src/activation_profile.py` (read) already dumps the
   residual stream via `output_hidden_states=True`, but traces the top *channels*; extend it to
   `‖h_l[p]‖` per position `p`. Cancedda 2024 Fig. 9 (read) is the reference shape for Llama-2-13B:
   flat through L0–L2, an abrupt jump at **L3** driven by that layer's MLP, a plateau, then erasure
   in the last few layers (L1 for the 7B; L2 and L8 for the 70B — App. C).
3. **Find the sink, don't assume it.** Compute Gu et al. 2025's per-token importance score for
   *every* position, not only position 0. Read, §3.2:

   > `α_k^{l,h} = 1/(T−k+1) · Σ_{i=k}^{T} A_{i,k}^{l,h}` — the mean attention token `k` receives from
   > all query positions `i ≥ k` in head `h` of block `l`.
   > `Sink^ε_k = 1/L Σ_l 1/H Σ_h 𝟙(α_k^{l,h} > ε)`.
   > "we prefer to select a threshold that is both strict … and less sensitive to the token length
   > `T`. This gives us the selection of **ε = 0.3**. For fair comparisons, we need to fix `T` when
   > computing the metric, e.g., **T = 64**." (§3.2)

   Define the **sink set** `S = { p : Sink^{0.3}_p > τ }` at `T = 64` on natural text, with `τ`
   pre-registered at 0.25 (i.e. a quarter of all heads treat `p` as a sink). Expect `S = {0}` for
   Llama-family models, but Guo Fig. 9a (read) measures OLMo's sink mass on ⟨s⟩ **and** "Delim"
   (⟨period⟩) together, so delimiters must be allowed into `S`.

   Three things about `Sink^ε` that are easy to get wrong:
   - **It is a fraction of heads, not attention mass.** "92.47" means 92.47% of `L×H` heads exceed
     ε, not 92% of attention. Reference values on natural text (Gu Table 1): GPT2-XL 77.00,
     Mistral-7B 97.49, LLaMA2-7B Base 92.47, LLaMA3-8B Base 99.02.
   - **It is not comparable across sequence lengths** — Gu Fig. 3: the metric "tends to decrease
     with larger token lengths T", more so at larger ε. Fix `T = 64` and say so, every time.
   - **Barbero 2025 reuses the same statistic but at ε = 0.8** (§4.2, Table 1), giving 45.97 / 73.49
     / 78.29% for LLaMA 3.1 8B / 70B / 405B over 170 prompts. Any table mixing the two must state ε
     per row. Barbero's *intro* phrases 78.29 as "almost 80% of the attention is concentrated on the
     ⟨bos⟩ token" — that is loose; **it is 78% of heads, not 78% of attention mass**, and this memo
     does not reuse the intro's phrasing.
   Gu's own measurement condition for from-scratch models is "100 sequences with `T = 64` (no BOS
   token)"; **BOS handling for the pretrained-LLM numbers in Table 1 is not stated**, so their
   absolute values should not be compared against ours without fixing the convention.

Per family, what to expect and what has to be verified before anything else runs:

| model | positional / MLP | BOS convention | prior evidence in this repo | what breaks |
|---|---|---|---|---|
| TowerBase-7B-v0.1 | Llama-2, RoPE, GLU | `<s>` prepended | v5 found `L1[2533,7890] = 1.5391`; ablation ×1.58, KL 0.464 (read, `notes.md`) — **at the magnitude-matched null ceiling**, i.e. undecidable, not inert | nothing; this is the reference model |
| EuroLLM-9B | Llama-style, RoPE, GLU *[inferred from the tech report's architecture section being Llama-shaped; verify from `config.json`]* | *[verify]* | q6: super weight at **L9**, ablation KL 3.284 — the strongest in that sweep (read, `prior_experiments_and_ideas.md` §2). L9 is unusually late; q6 is n=24–32, greedy, chrF++, below this repo's rigor floor | a late-layer super weight may not be a *sink*-creating weight at all — check the residual profile before assuming |
| Aya-Expanse-8B | Cohere architecture: LayerNorm, parallel attention+MLP block, tied embeddings *[all inferred, verify]* | `<BOS_TOKEN>` *[inferred, verify]* | q6: 2 candidates, ablation KL **0.002** — no critical scalar | `src/ablate_sw.py:get_layers` needs `model.model.layers[i].mlp.down_proj`; Cohere probably satisfies it, but a parallel block changes what "the layer's residual output" means |
| BLOOM-7B1 | **ALiBi**, LayerNorm, **non-GLU** (`dense_h_to_4h` / `dense_4h_to_h`) *[inferred, verify]* | **no BOS prepended by default** *[inferred, verify]* | q6: **0 candidates**, KL 0.001 | `get_layers` raises `SystemExit` on BLOOM (read, `src/ablate_sw.py`) — an adapter is required before BLOOM can be in the study at all |

Two things that look like escape hatches and are not:

- **ALiBi does not exempt BLOOM.** Gu 2025 §3.3 (read): "In Appendix C.1, we prove that for LMs with
  NoPE/relative PE/ALiBI/Rotary, if the first `T` tokens are the same, their corresponding hidden
  states are the same. They all have massive activations, thus dispersing the attention sink."
  Yona's RoPE-isometry argument is RoPE-specific; Gu's is not.
- **Dropping BOS is not a way to remove the sink.** Sun 2024's account is that it relocates to
  whatever comes first, and `docs/reviews_2026_09_05/review_methods.md` obj. 6 (read) already flags
  this: "Use an attention mask on position 0 for all queries instead." Owen 2025 reports that three
  Gemmas have massive activations *only with* BOS (read via `notes_super_weights_massive_activations.md`
  §4.6). So: measure **with and without** the BOS-like token, and treat the difference as a model
  property to report, not a nuisance to average over.

### 1.3 The "constant's contribution" statistic — and which parts of it can vary at all

This is the crux, so the statistic is split by whether the literature predicts it can move on a
per-input basis.

| # | statistic | definition | source (all read) | input-dependent? |
|---|---|---|---|---|
| **A** | constant magnitude `m` | `\|h_l[p*, c*]\|` at the sink position and massive channel, at and after the onset layer | Yu Fig. 4; Sun §2 criterion: "magnitude surpasses 100 and is at least or around 1,000 times larger than the median magnitude of its hidden state" (their words: "a loose but broad definition") | **No.** Yu §1 "regardless of the prompt"; Sun Table 2, CV ≈ 5.5% |
| **B** | sink attention mass | `α_p^{l,h}` per head, and `Sink^{0.3}` at `T = 64` | Gu §3.2 | **Yes** (Guo Claim 1) — but see the warning below |
| **C** | sink value update | `u_t = Σ_{i∈S} p_{t,i} v_i`, the first term of Sun **Eq. 2**: `Attention(Q,K,V)_k = Σ_{i≤k} p_{ki} v_i = Σ_{i∈C} p_{ki} v_i + Σ_{i∉C} p_{ki} v_i` | Sun §4.2 | **Yes, but only through `p`** — Sun §4.2 claims `v_i` for `i ∈ C` is near-constant |
| **D** | sink residual norm | `‖h_l[p*]‖` by layer | Cancedda Fig. 9 | Mostly no |
| **E** | attention entropy | `H(p_{t,·})` per head | — | Yes |
| **F** | over-mixing | `µ(V^(L)) = ‖V^(L) − (1/n)11ᵀV^(L)‖_F` — deviation of the last layer's value matrix from its own centroid | Barbero 2025 Eq. 1 / App. B | Yes |
| **G** | distributional fallback | `KL(p_next ‖ P_freq)` with `v_freq,i = log p_freq,i − mean(log p_freq)` | Stolfo §4 | Yes |

**The primary statistic is a 7-vector `Z(t)`**, computed at a single query position `t` from one
forward pass: `(m, Sink^{0.3} at query t, ‖u_t‖, ‖h_last[t]‖, mean head entropy,
‖h_last[t] − mean_p h_last[p]‖, KL(p_next ‖ P_freq))`. The sixth coordinate is the per-position
contribution to Barbero's `µ`, since `µ` itself is sequence-level and `Z` is per-position.
⚠️ **Barbero 2025 contains no effective-rank and no pairwise-cosine measure** — their footnote 1
says outright that "the rank can remain full for any ∆ > 0" and they keep the "rank collapse"
terminology only for consistency with prior work. Do not write "effective rank" and cite Barbero.
Coordinates A and D are included
**precisely because they are predicted invariant** — they are the internal control. If A moves with
input, that contradicts Yu §1 and Sun Table 2 and is itself the finding, but only after the
set-to-mean control (Sun §3) has been run.

**Sun's "constant value update" claim has no statistic behind it, and we have to supply one.**
Verified by reading §4.2 and App. B.3: the entire claim is Fig. 8's caption ("Value updates from
tokens associated with massive activations are essentially the same") plus the sentence "the value
updates from `C` are nearly identical across tokens, i.e., they serve as additive bias terms". There
is **no variance, cosine, norm ratio, error bar, `n`, or null** anywhere. Fig. 8's layer index and
head aggregation are not stated either; App. B.3 pins the same measurement to LLaMA2-7B layers
**3, 15 and 30**. So pre-register:

> **bias-constancy** `Bl = mean_t ‖u_t − ū‖ / ‖ū‖`, with `ū = mean_t u_t` over query positions after
> a burn-in of 8, computed per layer, **against a null**: the same quantity for a random size-`|S|`
> set of source positions. Reporting `Bl` with its null is a small, genuine addition to Sun §4.2 and
> costs one extra forward pass.

⚠️ **The single most important warning in this memo.** Gu 2025 §3.3, Table 1 (Left), read:
`Sink^{0.3}_1` (%) under natural / random-token / **repeated-token** input —

| model | natural | random | repeat |
|---|---|---|---|
| GPT2-XL | 77.00 | 70.29 | **62.28** |
| Mistral-7B | 97.49 | 75.21 | **0.00** |
| LLaMA2-7B Base | 92.47 | 90.13 | **0.00** |
| LLaMA3-8B Base | 99.02 | 91.23 | **0.00** |

*(column assignment confirmed on a second independent read of the PDF.)*

**A repeated-token input by itself takes `Sink^ε_1` to zero in three of four models.** So *after*
onset, "the sink collapsed" is guaranteed by the repetition and carries no information about its
cause. Everything in this design that bears on H′ has to live in the **pre-onset** window.
Gu's App. C.2 adds the other half of the problem: across the Pile's 17 domains, "input domains have
negligible effects on our attention sink metric `Sink^ε_1`" — so the coarse binary-count metric may
be too blunt to carry a pre-onset signal at all. *Inference:* that is why `Z` also carries the
continuous per-head mass and `‖u_t‖`, and why `Sink^ε` is one coordinate of seven rather than the
headline.

### 1.4 A free result that fell out of the pre-registration — the super weight *is* a sink neuron

`notes_repetition_degeneration.md` §5 (OP4) and `notes_super_weights_massive_activations.md` §3.1
(H5) both list, as untested, whether Yu's super weight and Yona's "sink neuron" are the same object.
**Reading the two tables side by side answers it, at zero compute cost, and the answer is yes in
four models out of four.**

Yona Table 1 reports, per model, the "Sink-Layer" and the surviving "Sink-Neurons" IDs after causal
filtering — where a neuron index `j` indexes the MLP intermediate dimension. Yu Table 2 reports
super weights as `(layer, j, k)` in `mlp.down_proj.weight[j, k]`; Yu's own caption gives the access
pattern as `layers[2].mlp.down_proj.weight[3968, 7003]`, so **`k` indexes that same intermediate
dimension** (and `src/coord_check.py` already verified the row/column convention: `W[j,k]` ranks
1–6 of 16–119 M while the transposed reading lands in the millions — read, `notes.md`).
Lining them up:

| model | Yona: sink layer / sink neurons | Yu Table 2 (or this repo's detection) | match |
|---|---|---|---|
| LLaMa-1-7b-HF | **2** / **7003** | `huggyllama/llama-7b` **L2**[3968, **7003**] | layer **and** index |
| LLaMa-2-7b-HF | **1** / **7890**, 10411 | `meta-llama/Llama-2-7b-hf` **L1**[2533, **7890**] | layer **and** index |
| Mistral-7B-Instruct-v0.1 | **1** / **7310**, 8572 | `mistralai/Mistral-7B-v0.1` **L1**[2070, **7310**] | layer **and** index |
| Meta-Llama-3-8B-Instruct | **1** / 198, **2427** | this repo, v3/v5 on Llama-3.1-8B: "three rows on column **2427**" (read, `notes.md`) | layer 1, index |

*Inference:* four independent four-digit index matches, each also agreeing on the layer, is not
coincidence. The two literatures are describing the same unit from two sides: **Yona's sink neuron
is the intermediate neuron; Yu's super weight is one scalar in that neuron's output fan-out.** That
also explains this repo's own finding (read, `notes.md`) that the 16 individually-inert Table 2
coordinates "share one `down_proj` INPUT COLUMN per model" — the shared column *is* the sink neuron,
and the several rows are its fan-out.

**What this does for the design.** It is not decoration: it means the lesion arm and the natural-sink
arm are provably acting on the same object, which is the premise the whole comparison needs. It also
converts the H5/OP4 open problem into a measurement with a stated answer.

**Verification status.** The first three rows were checked directly against both PDFs with
`pdftotext -layout` while writing this memo — Yu Table 2 (p. 4) and Yona Table 1 (§4.4) — and the
layer and index agree in all three. The fourth row is weaker: it pairs Yona's Table 1 against this
repo's own detection notes, not against Yu.

⚠️ **Caveats, before this is written up anywhere.** (i) The model checkpoints do not match exactly:
Yona uses `Mistral-7B-Instruct-v0.1` and `Meta-Llama-3-8B-Instruct` where Yu and this repo use base
`Mistral-7B-v0.1` and `Llama-3.1-8B-Instruct`. Yu §3 reports that instruction tuning does not move
the super weight and Gu Table 1 (Right) reports it barely moves the sink metric, so this is probably
benign — but "probably benign" is an argument, not a check, and the check is one forward pass.
(ii) Yona's neurons survive a *causal* filter (zero-ablate the TopK candidates, keep those that
collapse the repeated-token norm), so this is a match between two causally-validated sets, not two
magnitude rankings — which makes it stronger, not weaker. (iii) `K` in `TopK_j(Norm(MLP_j(BoS)))`
is never stated in Yona, so the "how many candidates were screened" denominator is unknown, and a
by-chance match cannot be formally priced. (iv) Yona's second neuron per model (10411, 8572, 198)
has no counterpart in Yu Table 2 — one column of the correspondence is unexplained, and pretending
otherwise would be the exact claim-drift `CLAUDE.md` is about.

---

## §2 Generation protocol

### 2.1 The decision that determines whether the experiment is possible

Base rates, all read, differ by **two to four orders of magnitude** between settings:

| setting | metric | value | source |
|---|---|---|---|
| open-ended LM continuation, greedy | `rep-r` | **0.917** | Fu Table 2 (Wiki-103 Transformer decoder) |
| same | `rep-n` / `rep-w` | 0.733 / 0.590 | Fu Table 2 |
| same | `seq-rep-4` | **0.442** (human 0.006) | Welleck Table 2 |
| same | `rep/ℓ` | 0.627 (human 0.487) | Welleck §6.2 |
| same, nucleus p=0.9 | `rep-r` | 0.368 | Fu Table 2 |
| same, pure sampling | `rep-r` | 0.155 | Fu Table 2 |
| human Wikitext-103 | consecutive sentence-level repetition | **0.02%** | Xu, abstract |
| NMT, greedy, IWSLT En–De | `rep-r` | **0.0512** | Fu Table 1 |
| NMT, beam 4, FLORES, mid/high-resource | hallucination rate (all types) | **0.005–0.47%** | Guerreiro Tables 2, 4 |
| NMT, beam 4, FLORES, low-resource | hallucination rate (all types) | 2–15%, but "detached hallucinations are more prevalent in low-resource settings" (§5.3) | Guerreiro Tables 2, 4 |

**Therefore: the natural-repetition arm must be open-ended continuation, not translation.**
Guerreiro never prints a standalone oscillatory rate (the split is only in raster heatmaps,
Figs. 2/9), but it is bounded above by the combined rate, so a translation-shaped arm gives well
under 1% in mid/high-resource directions. At 1,000 prompts per cell that is ~5 looping generations.
The experiment would be unpowered before it started.

⚠️ Every LM base rate above is from a 2019-era ~250M–750M Wikitext-103 model. **The loop rate of a
modern 7–9B multilingual model under greedy decoding is, as far as this reading goes, unpublished.**
A pilot is mandatory and its output is a deliverable in its own right (§6).

### 2.2 The runs

**Arm N1 — open-ended continuation (headline).**
Per language, take documents from a pre-cached per-language corpus (compute nodes have no internet;
`slurm/probe.sh` already sets `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` — read). Prefix
`k = 50` tokens (Welleck §6 and Xu §4.1 both use `k = 50`), generate `N = 256` new tokens.
`N = 256` rather than Welleck's 100 because `t_min = 32` plus a 30-token post-onset window plus a
30-token pre-onset window must all fit, with room for onset to be late.

**Arm N2 — MT-shaped (pilot scale only, reported as a measurement).**
FLORES-200 devtest source sentence + the model's own translation prompt format, greedy,
`max_new_tokens = 3·len(src) + 50`. The length rule is HalOmi's (`beam 5, no-repeat-4-gram,
≤ 3·src + 5`, quoted in `docs/reviews_2026_09_05/v3_review_hallucination.md`, read) **minus the
`no-repeat-4-gram` constraint** — which is the entire reason HalOmi cannot be used as the natural
corpus here: it decodes the phenomenon away before annotation. TNG (Guerreiro §5.1) is the metric
for this arm.

**Decoding.** `do_sample=False`; **no** `repetition_penalty`, **no** `no_repeat_ngram_size`, **no**
`min_new_tokens`; EOS allowed and EOS-terminated generations counted separately (Holtzman §1, read:
at beam ≥64 GPT-2 Large/XL "prefer to stop generating immediately after the given context" — early
stopping is a real degenerate mode and must not be silently dropped). `dtype` = the checkpoint's own
(already the harness default, read `src/ablate_sw.py`). **Batch size 1**, or a documented
batch-invariance check: padding plus batched matmul changes the argmax at near-ties, so a "greedy"
result can silently depend on batch composition.

**Languages.** The four MT models do not share a language set. Read: Tower is ten languages —
"English (en), German (de), French (fr), Dutch (nl), Italian (it), Spanish (es), Portuguese (pt),
Korean (ko), Russian (ru), and Chinese (zh)" (Alves 2024 §2); EuroLLM-9B is "all 24 official
European Union languages and 11 additional languages" (EuroLLM-9B report, abstract).
Aya-Expanse-8B (23) and BLOOM-7B1 (46) are *[inferred from model cards, verify]*. The intersection
of all four is approximately **{en, es, fr, pt, zh}** — five languages, all high-resource.
*Inference:* **the low-resource contrast the brief gestures at cannot be run with a shared language
set**, and Guerreiro §5.3 says the resource gradient for oscillation runs the *opposite* way from
intuition anyway. Report per-model language sets; make no cross-model low-resource claim.

**n.** Pilot: 200 prompts × 5 languages × 4 models = 4,000 generations, N1 only.
Main: `n = ceil(K / r̂)` per cell with `K = 50` looping generations, where `r̂` is the pilot rate.

**Power.** The pre-onset contrast is a paired comparison of a 7-vector at matched positions. Two
groups of `K = 50` detect `d ≈ 0.57` at 80% power, two-sided α = 0.05. Correcting over
7 coordinates × 5 languages (family size **35**, stated in every caption per `CLAUDE.md`) pushes the
detectable effect to `d ≈ 0.75` at `K = 50`, or requires `K ≈ 128` for `d = 0.5`. Pre-register:
**if `r̂ < 0.02` in a cell, that cell is not runnable at `n ≤ 5,000` and the result is reported as a
bound, not as a negative.**

---

## §3 The matched-control design

### 3.1 Matching

For each looping generation with onset `t*` in cell (model, language, arm), draw matched clean
positions from **non-looping generations in the same cell**, matched on:

- **absolute position** `t*` exactly, or within ±2 tokens (attention mass on position 0 falls
  mechanically as `T` grows under softmax; Barbero 2025 reports context length modulating sink
  strength — read via `notes_repetition_degeneration.md` §1.4);
- **prompt length** in tokens;
- model, language, arm.

Under greedy a prompt either loops or it does not, so this is a **between-prompt** design: content
is not matched. That is the design's weakest joint, and §3.3 fixes it cheaply.

### 3.2 The pre-onset window, and why the post-onset window is worthless here

- **Pre-onset window** `[t* − W, t* − 1]`, `W = 30` (Hiraoka's `r`, Eqs. 1–3). Also report `W = 10`:
  Hiraoka App. E ablates `r ∈ {5,…,50}` and finds `r = 10` or `15` gives the strongest intervention
  effect.
- Aggregate the window two ways: the **mean** of `Z(t)` (Hiraoka's `ā_n − a_n` shape, Eq. 3) and the
  **slope** of `Z(t)` across the window. Hiraoka Fig. 1 reports repetition neurons ramping up
  "progressively"; a pre-onset slope is the natural analogue and is a different claim from a level
  shift.
- **Post-onset window** `[t*, t* + W]` is recorded and reported **but is never evidence for H′.**
  Gu Table 1 (§1.3 above) shows repeated-token input alone drives `Sink^{0.3}_1` to 0.00, and
  Gu App. C.1 proves identical prefixes produce identical hidden states under RoPE/ALiBi/rel/NoPE.
  Post-onset sink collapse is mechanically guaranteed by the repetition.

### 3.3 A sampled arm, purely to get within-prompt matching

Same prompts, temperature 0.7 / top-p 0.9, **8 seeds per prompt**. Fu Table 2 shows nucleus p=0.9
still gives `rep-r = 0.368`, so loops survive sampling. Prompts that yield *both* looping and
non-looping continuations give **within-prompt, within-position matched pairs** — prompt content
held exactly fixed, only the trajectory differing. Greedy stays the headline (it is the condition in
which the lesion was measured — read, `notes.md`), and the sampled arm is the confirmation that the
between-prompt effect is not a content artifact. Fu §2.1 notes that under greedy `ζn = 1`, so the
average repetition probability can diverge: greedy maximizes power and bounds the claim to greedy.

### 3.4 Positive control 1 — Xu's self-reinforcement protocol

Read, Xu §2.1: build `x = (s⁰, s¹, …, s^N)` by repeating a sentence, **N = 100**, over three corpora
of **1,000 sentences each** — `D_wiki` (Wikitext-103 dev), `D_book` (BookCorpus) and **`D_random`**
(tokens sampled uniformly from the model vocabulary). **Teacher-forced**, not generated:
`P_θ(x_{n,l} | x_{<n,l})`. Appendix B is explicit that this is deliberate. Metrics, verbatim §2.1:

- `TP(sⁿ) = (1/L_s) Σ_l P_θ(x_{n,l}|x_{<n,l})`
- `IP(sⁿ) = (1/L_s) Σ_l 𝟙(P_θ(x_{n,l}|x_{<n,l}) > P_θ(x_{0,l}|x_{<0,l}))`
- `WR(sⁿ) = (1/L_s) Σ_l 𝟙(x_{n,l} is a winner)`, where a winner also satisfies
  `x_{n,l} = argmax P(·|x_{<n,l})`
- each averaged over the corpus to `TP_n, IP_n, WR_n`

Reproduce this per model per language, and compute `Z` at the same positions. Two jobs:

1. **Pipeline validation.** Xu §2.2: "IP₁ is higher than 90% across the various corpus". If our
   pipeline does not reproduce `IP₁ > 90%` on `D_wiki`, nothing downstream is trustworthy.
   ⚠️ Xu's ceiling values and saturation `n` are **not stated numerically** — only "converge around
   certain ceiling values" — so do not pre-register a ceiling target.
2. **Contamination calibration.** `Z` measured against `n` gives the size of the sink-statistic
   movement attributable to repetition *already present*. That is the artifact the pre-onset window
   is designed to avoid, and quantifying it is what licenses the claim that the pre-onset effect is
   not the same thing arriving early. `D_random` is the discriminator: if `Z` moves identically on
   random-vocabulary strings, `Z` tracks surface repetition, not anything semantic.

### 3.5 Positive control 2 — Yona's repeated-token protocol

Yona et al. 2025, `papers/Interpreting-2025-Repeated-Token-Phenomenon.pdf`, read.
**The PDF prints no venue** — sidebar `arXiv:2503.08908v1`, footer "© 2025 Google DeepMind".
Cite as an arXiv preprint.

The protocol is not stated once; it varies per figure. What is stated:

- **Stimulus.** Fig. 2 sweeps one repeated word (`the`, `one`, `es`) from 0 to 1500 repetitions,
  where "0 repetitions is the BoS token". Fig. 3's ablation input is "1200 repeats of the tokens
  `['Another', 'one', 'bit', 'es', 'the', 'dust']`" — a repeated **6-token cycle**, not one token.
  Fig. 5 prepends BOS with the literal template `<BoS> some prefix:` + token × 500.
  Appendix C uses `<s> Repeat: poem poem poem …`. **The number of sequences is never stated** for
  any figure, and BOS handling is stated only for Fig. 5.
- **Where the norm is measured.** The **residual stream / hidden state at the model's own sink
  layer**, at every repeated position — Fig. 3's y-axis is "Norm of Residual Stream Activations".
  Sink layers, Table 1: LLaMa-1-7b **2**, LLaMa-2-7b **1**, Llama-3-8B-Instruct **1**,
  Mistral-7B-Instruct **1**. Fig. 3 plots LLaMa-2 layers **1, 16, 29**.
  Baseline is "the average norm of tokens from Tiny Shakespeare"; printed legend values give
  LLaMa-2 `BoS Norm 6.93 / Typical Norm 2.41`.
- **Sink neurons**, §4.2, unnumbered equation: `candidates = TopK_j(Norm(MLP_j(BoS)))` where
  `MLP_j(x) = (W_out^j)ᵀ (σ(W_in^j x) ⊗ W_gate^j x)` — a **single** neuron's contribution, i.e. the
  `down_proj` row scaled by its gated activation. **`K` is never stated.** TopK yields *candidates*,
  then a causal filter keeps only those whose zero-ablation collapses the repeated-token norm; 1–2
  survive per model.
- **Attention statistic (Fig. 1):** the full causal attention matrix over positions 0–19,
  **averaged over all heads within a layer**, one heatmap per layer. No head selection.
- **Ablation effect (Fig. 3):** LLaMa-2 control peaks ≈1000 → ablated axis maxes at 150; LLaMa-1
  ≈1400 → ≈300; LLaMa-3 ≈400 → ≈8; Mistral ≈250 → ≈17.5. **All read off axes; no table, no
  percentage, no effect size anywhere.**
- **Theorem 4.1 (informal)** is the general claim, and it is *not* RoPE-specific: for any positional
  encoding acting only on queries/keys and bounded, a sequence of `n` repetitions of `x` after a
  fixed `k`-token prefix converges as `n → ∞` to the representation of the singleton sequence `x`,
  "due to softmax leakage" since the prefix length stays constant. RoPE's isometry appears only in
  footnote 2, as the reason boundedness is immediate.

Two jobs:

1. **Measurement validation** — if our norm/attention code cannot reproduce Fig. 2's qualitative
   shape in Tower and EuroLLM, it is wrong.
2. **A warning that changes how sink mass must be reported.** Repetition *creates* high-norm,
   BOS-like positions. So sink mass must be reported **decomposed by position** — position 0, the
   repeated run, everything else — never as one scalar. A fall at position 0 post-onset is
   redistribution, which is what Yona predicts, not destruction, which is what H′ would need.

⚠️ **Yona's regime is not the natural-loop regime, and this bounds what the control licenses.**
Table 1's "Repeats" column is the number of repetitions required to induce a sink: LLaMa-1 **450**,
LLaMa-2 **1000**, Llama-3-8B-Instruct **4000**, Mistral **1200**. A natural greedy loop repeats a
4-gram a handful of times over a 256-token generation — three orders of magnitude short. So Yona
validates the instrument and supplies the redistribution warning; it does **not** establish that
anything Yona-like is happening in a natural loop. Do not write it up as if it does.

⚠️ **The paper has three internal inconsistencies.** Fig. 1's caption names layers 2, 3, 17, 31
while its own panels are labelled 1, 2, 17, 31; the caption puts the repeated-`the` case in the top
panel while the figure's row titles put it below; §4.2's prose says "layer 2" while Fig. 2's title
for the same model says `sink layer: 1`. **Cite the panel labels and Table 1, not the prose.** §6
also concedes that "the first attention layer's behavior was unique to LLaMa2", so the mechanism is
not established to generalise to Tower/EuroLLM/Aya/BLOOM — which is exactly why it is a control here
and not a premise.

### 3.6 The near-precedent, corrected

`notes_super_weights_massive_activations.md` §3.0 treats Cancedda 2024 §6.1 as the near-exact
precedent, and the brief describes it as filtering the dark component "that an early MLP writes into
the **BOS** residual stream". **That description is wrong and the memo should not inherit it.**
Read, §6.1 and App. D:

- The `"the, the, the, …"` generations (Table 7, Ψ 80%, layers L5/L7/L10/L14) come from applying the
  Ψ spectral filter to the **whole residual stream at one layer**, all token positions — not to BOS.
- BOS-only filtering appears only as an NLL heatmap (Fig. 11 middle) with **no generations reported**.
- The repetition is **not quantified anywhere** — no rep-n, no diversity, no counts; the claim is
  sample tables plus one sentence ("the application of Ψ filters often results in the model entering
  repetitive patterns … inhibiting attention sinking results in over-copying").
- **The random-projection control loops too** in several cells (Table 4/L3, Table 7/L5), and the
  paper does not address it.

The genuinely reusable piece is the **shavings-swap control** (Fig. 10, caption verbatim): "rather
than being suppressed, the subspaces blocked by spectral filters are swapped with the corresponding
ones from a token at the same position but in a different sample. This swap perturbs all residual
streams except the one for Token 0, since this is identical for all samples due to the
autoregressive nature of the model." *Inference:* that is an off-manifold control with the sink
carved out of it, and it is directly portable: **swap the super weight's contribution
`W[j,k]·x_k` with the same contribution computed on a different input.** If degeneration appears
under zeroing but not under the swap, the BOS/sink pathway is implicated rather than the generic
loss of a vector. Cancedda's own swap result is stated only as the presence/absence of a
"step-decrease" in a heatmap — **no numbers** — so we would be supplying the quantification.

---

## §4 The lesion signature

### 4.1 The sweep

- **Dose** `α ∈ {1, 0.75, 0.5, 0.25, 0}` applied to `W[L, j, k]`. `src/ablate_sw.py:set_weight`
  already takes an arbitrary value (read) — this is a one-line change, not a new runner.
  Yu §3.2 swept "a scaling factor ranging from 0.0 to 3.0" with **average zero-shot accuracy** as the
  readout; the sub-1 half of that sweep with **generation** as the readout is unreported.
- **Models.** The five with an established individually catastrophic scalar, wikitext-2, read from
  `notes.md`: OLMo-1B ×3667, Mistral-7B ×1430, Llama-7B ×181, OLMo-7B ×79, Llama-30B ×16. Plus the
  four MT models, where the unit must be **the intermediate neuron column** — BLOOM 0 candidates,
  Aya KL 0.002 (read, `prior_experiments_and_ideas.md` §2), and 16 of 21 Table-2 coordinates are
  individually inert while sharing one `down_proj` input column per model (read, `notes.md`). For
  the column, `α` scales `down_proj.weight[:, k]` entire.
- **Same prompts, same decoding, same `L-onset` rule** as §2. No exceptions; the whole design is a
  comparison of two conditions measured identically.

### 4.2 What to record

Per (model, dose, cell): the 7-vector `Z` at matched positions; onset `t*`; loop rate; `seq-rep-4`;
`rep-r`; wikitext-2 perplexity (existing harness); `KL(p_next ‖ P_freq)`; and the constant's
magnitude `m` — which must fall roughly linearly with `α`, as a check that the dose did what it says.

⚠️ **Perplexity will not detect this failure mode.** `notes_super_weights_massive_activations.md`
§3.2 (read there) records Jin 2025 App. F giving a fully degenerate looping generation with
**perplexity 2.99**. Every table carries a repetition metric next to perplexity.

### 4.3 Match on onset latency, not on α

At `α = 0` the model "loops on function words from the first token" (read, `notes.md`) — **there is
no pre-onset window at all**, so `α = 0` cannot enter the signature comparison. Pre-register:

> `α*` = the largest `α` whose **median onset `t*` falls inside the interquartile range of the
> natural onset distribution** for the same model and language.

That one choice removes the two worst confounds simultaneously: position (the windows are at
comparable depths) and damage magnitude (the model is not already destroyed). **If no `α` produces a
loop with `t* ≥ t_min`, the lesion arm has no comparable window for that model and the comparison is
declared not runnable there** — stated in advance, so it cannot be rationalised afterwards.

### 4.4 Comparing two signatures

Per condition, define

```
Δ_cond = mean over cells of [ Z̄(pre-onset, looping) − Z̄(matched clean) ]
```

with each coordinate z-scored by that cell's matched-clean pooled SD, so the seven coordinates are
commensurate.

- **Statistic 1** — `cos(Δ_natural, Δ_lesion(α*))`.
- **Statistic 2** — **transfer AUC**: fit an L2-regularised logistic classifier on the natural
  (pre-onset vs matched-clean) vectors, evaluate it on the lesion vectors; fit on lesion, evaluate on
  natural; report the mean of the two AUCs.
- **Ceiling (the null the brief asks for — two seeds of the same condition).** Split each condition
  into two halves by seed (for greedy, by a hash of the prompt; for the sampled arm, by generation
  seed) and compute `cos(Δ_natural,A, Δ_natural,B)` and the *within*-condition transfer AUC. These
  are the ceilings. **The test is whether the cross-condition value's 95% CI overlaps the
  within-condition ceiling's CI**, not whether it differs from zero. CIs by cluster bootstrap over
  prompts (prompts, not positions, are the unit of independence — positions within a generation are
  not independent).
- **Floor.** Permute the looping/clean labels within cell, 1,000 permutations, **max-statistic**
  across the 7 coordinates (Nichols & Holmes 2002 **[unopened]**, the method this repo has already
  chosen for its Phase 0 null).
- **Report per coordinate as well.** A whole-vector cosine can be carried by one coordinate; the
  interesting result in §7 is precisely a coordinate-level dissociation.

### 4.5 The control triple, non-optional

`notes_super_weights_massive_activations.md` §5.2 (read there): every headline number from zeroing
is "removing a near-constant bias plus a distributional shift", and the two are not separated. Sun
Table 3 and Owen 2025 Table 1 both show set-to-mean is harmless where set-to-zero is catastrophic.
So each dose is reported as a **triple**: `α`-scaled / **contribution-mean** (replace `W[j,k]·x_k` at
`down_proj` output with its calibration-corpus mean, via a forward hook — the well-defined weight
analogue of Sun's set-to-mean, per `review_methods.md` obj. 1) / **magnitude-matched random weight**
from the same matrix, max-statistic over ≥50 draws. A dose whose effect sits at the random ceiling is
**undecidable**, not inert.

---

## §5 Confounds, and how each is handled

1. **Self-reinforcement contamination.** Xu §2.2: `IP₁ > 90%` — one prior repetition already raises
   the probability in most cases; Hiraoka Fig. 1: repetition neurons ramp up *after* onset. Any
   post-onset measurement is contaminated. *Handling:* pre-onset window only for inference;
   post-onset reported separately and labelled as such; §3.4 calibrates the contamination size.

2. **Input-agnostic constant vs input-dependent draw on it.** §0 and §1.3. *Handling:* the 7-vector
   carries the predicted-invariant coordinates (A, D) as an internal control; the hypothesis is
   carried by B and C. Also — **Gu App. C.2 says input domain has negligible effect on `Sink^ε_1`**,
   so if the pre-onset effect appears only in the coarse `Sink^ε` coordinate and not in the
   continuous per-head mass or `‖u_t‖`, suspect an artifact.

3. **Repetition destroys the sink by itself.** Gu Table 1: `Sink^{0.3}_1` → 0.00 under repeated
   tokens in Mistral-7B / LLaMA2-7B / LLaMA3-8B, and Gu App. C.1 proves the mechanism holds for
   RoPE, ALiBi, relative and NoPE alike. Yona reports the apparently opposite thing — repetition
   *creates* BOS-like high norms at the repeated positions. **They are not in conflict, and the
   design depends on keeping them apart:** Gu measures *attention received by position 1*, Yona
   measures *residual-stream norm at the repeated positions*, and Gu's own App. C.1 states the
   reconciliation — identical prefixes give identical hidden states, so the repeated positions "all
   have massive activations, thus **dispersing** the attention sink". One object splits into many;
   the mass on position 1 falls because there are now competitors, not because the bias is gone.
   *Handling:* never report a single "sink mass" scalar. Report attention-on-position-0,
   attention-on-the-repeated-run, and residual norm per position, as three separate coordinates;
   pre-onset only for inference; §3.5 as the positive control that makes the redistribution visible.

4. **BOS convention — larger than it looks.** Owen 2025 (read; the PDF prints **"A Preprint"**, no
   venue) adopts Sun's criterion unchanged (`max|h| > 100` **and** `max|h| ≥ 1000·median|h|`,
   their Eq. 1) and evaluates **every model under both conditions**, which is the de-facto
   recommendation — an explicit "always report both" sentence is not in the paper. §4.2: "when the
   BOS token is excluded, massive activations are **not observed** in Gemma-7B, Gemma-2-2B, and
   Gemma-2-9B. However, when the BOS token is included, massive activations emerge in these models."
   The perplexity cost of dropping BOS in that family is not subtle — Gemma-7B WikiText/C4/PG-19
   `6.40 / 10.71 / 11.39` → **`3.11×10⁸ / 9.61×10⁷ / 6.00×10⁷`** (Table 1 vs Table 6), while
   LLaMA-2-7B moves 5.13 → 5.12. Barbero Table 2 makes the same point from pretraining: a 120M model
   trained with ⟨bos⟩ always first has sink metric **90.84%** with ⟨bos⟩ at inference and **0.05%**
   without, valid loss 2.69 → **7.56**; Gemma 7B's Table 3 shows HellaSwag 80.61 → 27.35 and Ruler
   82.57 → **0.00**. *Handling:* measure with and without the BOS-like token and treat the delta as a
   reported model property; but **never use BOS deletion as the sink-removal intervention** — it can
   destroy the model outright, and Sun's account is that the sink relocates anyway. Use an
   **attention mask on position 0 for all queries** (`review_methods.md` obj. 6). BLOOM needs its own
   determination because it has no BOS-prepending convention *[inferred, verify]*.
   Owen §4.5 also refutes the "delimiter tokens only" assumption — in Gemma-3-4B, Gemma-3-12B and
   Falcon-7B, massive activations attach to ordinary tokens ("polished", "mass", "cold") — so `S`
   must be found by measurement (§1.2), not by a token-type rule. **Where the massive activation
   goes when BOS is absent is not tracked in any paper read here**; the nearest datum is Owen §4.3's
   own 1B model, where *attention* (not activation magnitude) splits between the starting token and
   the full-stop token.

5. **Tokenizer fertility.** Fertility differs 2–3× across languages, so a fixed `max_new_tokens`
   yields different amounts of *text* per language, and token-level `rep-n` is not comparable across
   tokenizers. `notes_super_weights_massive_activations.md` §4.5 flags this and warns that Hämmerl's
   tokenizer attributions are self-labelled speculation with no tokenizer statistics measured.
   *Handling:* measure and report fertility per language; use token-level metrics only for
   within-model comparisons; for any cross-language claim use a character- or word-level duplicate
   n-gram measure and TNG, which are computed on detokenized text.

6. **Prompt language vs generation language (off-target), and source-copying.** An off-target
   generation and a loop both reduce diversity, and copying the prompt is a third thing again —
   Cancedda's proposed mechanism for sink loss is literally over-copying. *Handling:* run language ID
   (GlotLID or fastText `lid.176` **[unopened]**) on every generation; report the off-target rate per
   cell; stratify rather than pool; apply **TNG's source-side control** (Guerreiro §5.1: subtract the
   source's top 4-gram count) wherever a source sentence is in the prompt, so `L-onset` cannot fire
   on material copied from the input.

7. **Greedy-only claims.** Fu §2.1: under greedy `ζn = 1`, so the average repetition probability can
   diverge — greedy is the maximum-power setting *and* the most special one. *Handling:* write
   "under greedy decoding" into every sentence; run the sampled arm (§3.3) so at least one result
   generalises; report Fu Table 2's greedy-vs-nucleus gap (`rep-r` 0.917 vs 0.368) as the reference
   for how much decoding alone moves the outcome.

8. **Models with no critical single weight.** BLOOM-7B1 returned 0 candidates and Aya KL 0.002 (read,
   `prior_experiments_and_ideas.md` §2); 16 of 21 published coordinates are individually inert while
   sharing one `down_proj` input column per model (read, `notes.md`). *Handling:* the lesion unit is
   the **intermediate neuron column**, not the scalar, wherever the scalar is not causally real; the
   dose `α` for a column is not comparable to a scalar's `α`, so comparability is established through
   **matched onset latency** (§4.3), never through matched `α`. And BLOOM cannot run at all until
   `get_layers` has a non-`down_proj` adapter (read, `src/ablate_sw.py`).

9. **Instruct vs base, and chat templates.** A chat template inserts tokens before the first real
   token, which moves the sink; instruct models have a different effective prior. Gu Table 1 (Right)
   is the reassuring half — "instruction tuning has an insignificant impact on attention sink"
   (Mistral-7B 97.49 → 88.34; LLaMA2-7B 92.47 → 92.88; LLaMA2-13B 91.69 → 90.94; LLaMA3-8B
   99.02 → 98.85). *Handling:* do not pool base and instruct; use TowerBase for the headline and
   TowerInstruct as a separate stratum (the repo has both coordinates — read, `notes.md`); record the
   exact template string in the results JSON; note Aya-Expanse is instruct-only, so it can never be a
   base-model datapoint.

10. **Position and sequence length.** Attention mass on position 0 falls mechanically as `T` grows.
    *Handling:* exact position matching (§3.1); report every statistic as a function of position, not
    only as a window mean.

11. **Multiplicity.** The scan is over 7 statistics × 5 languages, and any layer/head breakdown
    multiplies that. *Handling:* max-statistic permutation, family size stated in every caption.

12. **Silent mechanical failures.** `transformers` 5.15.1 is installed (verified) and `modeling_llama`
    contains **no `output_attentions` handling** — attention weights come only from the
    `eager_attention_forward` path selected by `config._attn_implementation` (verified by reading the
    installed source). Loading with SDPA and asking for attentions yields nothing rather than an
    error. *Handling:* load with `attn_implementation="eager"` for every hooked run, and assert
    non-`None` attention in the harness's smoke test — `slurm/probe.sh` already has the right shape
    for this (it asserts a known-good OLMo-1B result before the sweep launches; read).

---

## §6 Feasibility

### 6.1 Hooks and code, against what exists

Existing, read from `src/`:

| capability | where |
|---|---|
| `mlp.down_proj` forward hook capturing input and output | `detectors/v1,v2,v3,v5.py` |
| residual stream at every layer | `output_hidden_states=True` in `detectors/v5.py`, `activation_profile.py` |
| greedy generation | `ablate_sw.greedy_continuation` (currently 15 tokens, one prompt) |
| arbitrary weight set/restore by `(layer, j, k)` | `ablate_sw.set_weight` — already takes a value, so the dose sweep is a one-line change |
| wikitext-2 perplexity, KL over fixed prompts | `ablate_sw.measure` |
| provenance (git sha, revision, dtype, versions) into every JSON | `provenance.py`, used by both runners |

Needed, none of it large:

| # | what | how | cost |
|---|---|---|---|
| 1 | per-layer per-head attention | `attn_implementation="eager"`; hook `self_attn` or use the returned attentions. Memory at `L=32, H=32, T=300`: ~368 MB fp32 for one sequence — reduce to `Z` on the fly, never store | small |
| 2 | per-head value vectors at sink positions | hook `self_attn.v_proj`, reshape to heads | small |
| 3 | `u_t = Σ_{i∈S} p_{t,i} v_i` and bias-constancy `Bl` | derived from (1)+(2), no new hook | small |
| 4 | per-position residual norms | extend `activation_profile.py` | trivial |
| 5 | dose scaling; column ablation | `set_weight` × `α`; `down_proj.weight[:, k] *= α` | trivial |
| 6 | contribution-mean hook (control triple) | forward hook replacing `W[j,k]·x_k` with its calibration mean at `down_proj` output | small |
| 7 | attention mask on position 0 for all queries | additive 4-D mask, or zero column 0 of the attention weights and renormalise | fiddly, ~half a day |
| 8 | `get_layers` adapter for BLOOM (and verification for Cohere/Aya) | dispatch on architecture | ~half a day |
| 9 | `L-onset`, `seq-rep-4`, `rep-r`, `rep/ℓ`, TNG | pure Python, CPU | small |
| 10 | language ID | fastText/GlotLID, CPU, must be pre-cached | small |
| 11 | Xu's `TP/IP/WR` (teacher-forced) | one forward pass per constructed sequence | small |

**The one architectural decision: run it as two passes.** Pass 1 generates text only — batched, no
hooks, cheap. Pass 2 re-runs *only the selected windows* under teacher forcing with hooks, batch 1,
reducing to `Z` on the fly. This keeps the expensive instrumentation off the 10⁴-generation path.
*Inference:* teacher-forcing a generated sequence reproduces the same computation as generation for a
causal LM up to KV-cache numerics; verify that on one sequence per model rather than assuming it.

### 6.2 GPU-hours

Anchored on the existing sweep: `slurm/sweep.sh` requests 1 GPU, 160 GB, **2 h** per model for
detection plus 21-coordinate ablation on 32 × 2048 wikitext-2 windows, on A100-80 / H100-80 / B200-180
(read). `notes.md` records the nine-model sweep completing in ~15 min total.

| item | estimate |
|---|---|
| pilot, N1: 4 models × 5 langs × 200 prompts × 256 tokens, generation only, batched | 2–4 GPU-h |
| main N1 at `n ≈ 1,000/cell` | 8–15 GPU-h |
| pass 2 (hooked, ~100 windows per cell, batch 1) | ~1 GPU-h |
| lesion arm: 5 doses × 9 models at pilot prompt volume | 10–20 GPU-h |
| control triple (contribution-mean, ≥50 random draws) on 9 models | 5–10 GPU-h |
| positive controls (Xu `N=100` × 1,000 sentences; Yona repeated-token) | 2–4 GPU-h |
| **total** | **30–55 GPU-h** |

Comfortably inside the project's compute; the binding constraint is **engineering hours**, which
`review_methods.md` §(b)7 already shows is where this program overruns.

### 6.3 What a week-9 result could be

The pilot alone is a self-contained deliverable, and it is one nobody has published as far as this
reading goes:

> **Natural greedy loop rates in TowerBase-7B, EuroLLM-9B, Aya-Expanse-8B and BLOOM-7B1, across
> five languages, `n = 200` per cell, with CIs** — plus the onset-position distribution,
> `seq-rep-4` / `rep-r` next to it, the off-target rate, and both positive controls reproduced
> (Xu's `IP₁ > 90%`; Yona's repeated-token norms).

That directly answers the brief's un-verified anecdote ("worse in low-resource languages — NOT
verified by our sweep") and it fixes the base rate `r̂` that determines whether the rest of the
design is affordable. It stands as a result whether or not the signature comparison ever runs.

**And §1.4 is available today, at zero GPU cost** — the super weight and Yona's sink neuron index the
same intermediate neuron in three models verified against both PDFs. That closes OP4 /
H5, which two of this repo's reading-notes files list as open, and it is the premise the rest of the
design leans on.

---

## §7 Decision tree

Read the pre-onset contrast first, then the signature comparison. `d` refers to the pre-onset effect
after max-statistic correction (family = 35).

**A. Supports "same route".** Pre-onset `Δ_natural` is significant in ≥3 of 4 (or of 5) models, with
coordinate B (sink attention mass) and C (`‖u_t‖`) **down**, A (constant magnitude) and D (sink
residual norm) **unchanged**; and `cos(Δ_natural, Δ_lesion(α*))`'s 95% CI **overlaps** the
within-condition two-seed ceiling; and transfer AUC in both directions is inside the within-condition
ceiling's CI. *Inference:* the strongest version of this is A and D unchanged in the natural arm but
falling in the lesion arm, with B and C falling in both — i.e. the two conditions reach the same
downstream state by different upstream routes, which is a more interesting and more defensible claim
than "the same failure".

**B. Supports "different failures".** Pre-onset `Δ_natural` is significant but the cross-condition
cosine sits well below the ceiling, or the classifier does not transfer in either direction. The
diagnostic sub-cases to look for: (i) natural loops show B unchanged and F (inter-position similarity)
*falling* — a copy dynamic that leaves the sink intact, which is the brief's stated alternative and
matches Cancedda's over-copying framing; (ii) natural loops show G (KL-to-unigram) moving without B
moving — a distributional fallback with no sink involvement; (iii) the lesion arm moves A, D, B, C
together while the natural arm moves only E and F.

**C. A legitimate negative.** Pre-onset `Δ_natural` is not distinguishable from the permutation null
in any model. Report as a **bound** with the interval, per `CLAUDE.md` ("Never report 'no effect'").
This kills H′ in its stated form without supporting the alternative, and it is a publishable outcome
for the project because the pre-onset window is the only place the question could have been decided.

**D. Uninterpretable — declare in advance, do not rescue.**
- `r̂ < 0.02` in a cell, so fewer than ~20 looping generations exist there (§2.2).
- Loops concentrated in one model or one language, so the "cross-model" claim rests on `n = 1`.
- The "loops" are mostly source-copying or off-target generation once TNG's source control and
  language ID are applied (§5.6).
- No `α` yields a lesion loop with `t* ≥ t_min` (§4.3), so there is no matched pre-onset window.
- The only `α` that produces a comparable onset latency already has wikitext-2 perplexity ×100 or
  worse — the "healthy-ish partial lesion" does not exist for that model.
- Attention captured under SDPA and silently `None` (§5.12) — a mechanical failure that produces a
  clean-looking null.
- The pre-onset effect appears in `Sink^ε` alone and not in the continuous coordinates, given
  Gu App. C.2's finding that `Sink^ε` is insensitive to input domain (§5.2).

---

## §8 Provenance — read, inferred, and outstanding

**Read directly from PDFs in this repo** (section/figure numbers cited inline): Yu et al. 2024 §1,
§3.2, Fig. 5, Table 1; Sun et al. 2024 §2, §3 + Table 2/3, §4.2 Eq. 2, App. B.3; Gu et al. 2025 §3.2,
§3.3 + Table 1, and the §3.3 statement of App. C.1/C.2; Guo et al. 2024 abstract, Claim 1, §3.1,
Fig. 9; Cancedda 2024 §3, §4 Eqs. 3–4, §5, §6.1, §7 Eq. 5, Figs. 8–11 and App. C/D Tables 2, 4–7;
Stolfo et al. 2024 §2 Eqs. 1–2, §3.1 Eq. 3, §3.2 Eqs. 4–6, §3.3, §3.4, §4 Eqs. 7–8, App. D/E/F;
Yona et al. 2025 §3.1, §4.1, §4.2, §4.3 (Thm. 4.1), §5.2, §5.3, §6, Figs. 1–5, 7, 9, Table 1,
App. B/C; Barbero et al. 2025 §3.2 Eqs. 1–3, §4.2 + Table 1, §5 + Table 2, Table 3, Figs. 2–3, 6,
App. B/C.1/C.3; Owen et al. 2025 §1, §2 Eq. 1, §3, §3.1, §4.2, §4.3, §4.5, Tables 1 and 6;
Xu et al. 2022 §2.1, §2.2, §4.1, App. A/B/D; Hiraoka & Inui 2024 §2.1, §2.2, §3.1 Eqs. 1–3, §3.2,
§4.1, §4.2, App. A/C/D/E/F; Fu et al. 2021 §4.1, Tables 1–3; Welleck et al. 2019 §4, §6.1 Eqs. 9–10,
§6.2, Table 2; Holtzman et al. 2019 abstract, §1, Fig. 4; Guerreiro et al. 2023 §3.1, §4.1, §5.1,
§5.2, §5.3, Tables 2–5, App. B; Alves et al. 2024 §2 (Tower's ten languages); EuroLLM-9B report,
abstract and §1.

**Read from this repo:** `notes.md` in full; `docs/literature/notes_repetition_degeneration.md`;
`docs/literature/notes_super_weights_massive_activations.md` §2–§5; `docs/prior_experiments_and_ideas.md` §2;
`docs/proposal_draft_v4.md` §1–§5; `docs/reviews_2026_09_05/review_methods.md` and
`v3_review_hallucination.md`, `v3_review_pi.md`, `review_mt_career.md`; `src/detect_sw.py`,
`src/detectors/v1,v2,v3,v5.py`, `src/ablate_sw.py`, `src/activation_profile.py`, `src/sw_models.py`,
`src/provenance.py`, `slurm/*.sh`, `pyproject.toml`. `transformers` 5.15.1 / torch 2.11.0+cu128 and
the absence of `output_attentions` in `modeling_llama` were verified by executing against the venv.

**Venues, exactly as printed on the PDFs** (checked, because `CLAUDE.md` requires it): Gu 2025 —
"Published as a conference paper at ICLR 2025". Barbero 2025 — "Published as a conference paper at
COLM 2025". Stolfo 2024 — "38th Conference on Neural Information Processing Systems (NeurIPS 2024)".
Sun 2024 — **no venue line**, `arXiv:2402.17762v2` only. Cancedda 2024 — **no venue line**,
`arXiv:2402.09221v1`, FAIR at Meta. Yona 2025 — **no venue line**, `arXiv:2503.08908v1`, Google
DeepMind. Owen 2025 — **"A Preprint"**, BluOrion, dated 2025-03-28, no arXiv stamp.

**Read only via this repo's own reading notes, not the PDF** (weaker; verify before citing in a
write-up): Queipo-de-Llano 2025 on the input-invariant transition layer; Su 2025's sink decay rate;
Jin 2025 App. F's perplexity-2.99 loop; Subramanian 2026 Table 8; Macocco 2025; Hämmerl 2023.

**[unopened]:** Olsson et al. 2022; Nichols & Holmes 2002; GlotLID / fastText `lid.176`;
Binkowski et al. 2026.

**Inferred, and flagged as such in the text:** the four models' architecture details for Aya and
BLOOM and their BOS conventions; the language-set intersection {en, es, fr, pt, zh}; Gu Table 1's
column assignment; the two-pass design's teacher-forcing equivalence; every `Inference:` sentence.

**Outstanding before this design is executable** (none of it blocking the pilot):

1. **Verify §1.4's four-row table by eye against both PDFs.** It is the memo's one novel claim and
   it currently rests on this repo's transcription of Yu Table 2 plus one extraction pass over Yona
   Table 1. Ten minutes of work; do it before the claim leaves this file.
2. `config.json` / `tokenizer_config.json` for EuroLLM-9B, Aya-Expanse-8B and BLOOM-7B1 — none of the
   three is in the local HF cache (checked; only OLMo-1B, mBERT and GPT-2 are), so this happens on
   the cluster login node. Needed for §1.2's four-row table, all of whose Aya and BLOOM entries are
   currently inferred.
3. Yona's `K` in `TopK_j(Norm(MLP_j(BoS)))` — not in the paper; may be recoverable from
   `github.com/yossigandelsman/attn_sinkhole`, which the PDF cites.
4. A per-language corpus decision for Arm N1, pre-cached on the login node (compute nodes are
   offline). FLORES-200 is already needed for Arm N2.
