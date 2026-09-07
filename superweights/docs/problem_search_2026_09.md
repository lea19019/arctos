# Problem search, 2026-09-05 — what the literature leaves open, and what to pick

**Purpose.** Adrian asked for the project to *explain or solve something*, to carry an
MT/cross-lingual component, and to build interpretability and research-engineering skills.
A professor suggested explaining the "We. We. We." repetition that follows super-weight
ablation. This document is the synthesis of a five-topic literature sweep run the same day
(≈130 new PDFs, all in `../papers/`, notes in `literature/notes_*.md`) against the project's
own measured state, and it ends with a recommendation.

**Provenance and trust.**
- Literature claims below are taken from the five reader notes. Each note marks what was
  read vs. inferred and flags `[unopened]` items; anything cited here was opened by a
  reader, but **venues and author lists were verified only where the notes say so** —
  re-check before a citation lands in the proposal or a paper.
- "Established" project results are only those Adrian ran and wrote up in `../notes.md`
  (Table 2 coordinate ranks; single-scalar ablations on wikitext-2; the repetition pattern;
  the v5 detector's modern-model negatives).
- ⚠️ Everything in `../parked_claude_2026-09-02/` (joint ablation, magnitude-matched null,
  super-activation zeroing, NLLB run, prompt-confound run) is an **unverified lead**, per
  Adrian's instruction. It appears here only as "a parked run suggests …, to be re-verified",
  never as a number to build on.

---

## 1. Nine facts that change the project's footing

1. **The unit of criticality is probably the intermediate neuron, not the scalar.**
   Independent lines converge: Oh et al. 2024 define "massive weights" as the `W_gate`/`W_up`
   rows feeding one intermediate neuron; Su et al. (ICLR 2026) Table 8 lists Llama-3.2-1B's
   four super weights all on intermediate neuron 1417; Li et al. (ICML 2026) describe
   `W_down[7890,:]` in Llama-2-7B — Yu's super weight in the row basis — without citing Yu;
   An et al. (ICLR 2025) show 100% feature alignment between `W_down` weight outliers and the
   activation outliers. Subramanian (COLM 2026) Table 8 found top-magnitude coordinates inert
   in four models; if the unit is the neuron, they cut two spokes of a four-spoke wheel. The
   project's own established result — 16 of 21 Table 2 coordinates individually inert, all
   clustered on shared `down_proj` columns — is already the sharpest evidence on this, and a
   parked run suggests the sets are jointly catastrophic (to be re-verified).
   *(notes_super_weights §1.1, §4.2)*

2. **Zeroing is an upper bound on dependence; the mean-replacement control is not optional.**
   Owen et al. 2025 (17 models): set-to-mean is harmless in all 16 tested, set-to-zero is
   catastrophic in 9. Sun et al. 2024 Table 3 showed the same for the activation. Every
   headline number must be a triple — zero / mean / magnitude-matched random — or be labelled
   an upper bound. *(notes_super_weights §5.2)*

3. **Not every massive activation is load-bearing, and not every model has one.** Owen:
   ~7 of 16 models unaffected by zeroing; OLMo-2-7B has none under Sun's criterion; Gemma
   models have them only with a BOS token. Oh: Gemma-2 and Phi-3-medium insensitive. Chen et
   al. 2026: 4 of 24 checkpoints fail the criterion. Owen states no indicator separates
   detrimental from harmless ones — the project has the rare ingredient (a model set split by
   outcome) to look for one. *(§1.1, §4.6)*

4. **"Massive activations cause attention sinks" is disputed, and the from-scratch evidence
   runs against it.** Qiu 2025 (value gate), Chen 2026 (V-scale), Sun 2026 (sandwich/QK
   norm, DyT) all remove massive activations while sinks survive; no intervention has removed
   sinks while keeping massive activations. Queipo-de-Llano (ICLR 2026) is the post-hoc
   counter-evidence. Write "our hypothesis", never "the mechanism". *(§2)*

5. **Nobody has explained Yu's Fig. 5, but the pieces exist.** Cancedda 2024 produces almost
   exactly our generations (`"the, the, the, …"`) by filtering the "dark" subspace an early
   MLP writes into the BOS stream, and attributes it to sink destruction → over-copying.
   Puccetti 2022 showed outlier ablation in BERT shifts predictions toward *frequent* tokens,
   with a real frequency baseline Yu lacks; Macocco 2025 finds the *opposite* direction for
   decoders' last-layer outlier dimensions — an open contradiction this project can
   adjudicate. Stolfo 2024 supplies the tooling (entropy neurons via the final-norm scale;
   token-frequency neurons; the frozen-LN ablation). Yona 2025's "sink neurons" may be the
   same object as a super weight. Su 2025 shows ~90% sink decay *and* repetition after
   super-expert pruning but never links them. *(notes_super_weights §3; notes_repetition §2–3)*

6. **The MT literature already has a name, a detector and a benchmark for the failure
   mode.** Raunak 2021: *oscillatory hallucination* = inadequate translation with repeated
   n-grams; Guerreiro 2023 (TACL): the TNG detector, and the finding that oscillatory
   hallucinations dominate mid/high-resource directions while detached ones dominate
   low-resource; Dale 2023: ALTI+ source contribution detects detached hallucinations;
   HalOmi: 18 annotated NLLB directions. No paper links NMT hallucinations to outlier
   structure. *(notes_repetition §4)*

7. **No calibrated criterion exists anywhere for "this weight/activation is special".**
   Dettmers' thresholds were chosen to yield one detection at 125M; Kovaleva relaxed 3σ per
   model; Hämmerl §6.4 admits 3σ still fires on whitened noise; Hodgkinson's random-matrix
   nulls are for eigenvalues and say elements are *not* heavy-tailed; Ding's k₀=1.205 anchor
   discards the top decile by design. Xu 2026 §8.1 has the only empirical null, for attention
   selectivity. *(notes_super_weights §4.1)*

8. **Formation: the hole is exactly RQ1, and the instrument is PolyPythias, gated on
   existence.** No paper traces the causal criticality of a named coordinate across
   checkpoints; across twelve outlier-formation papers the number reporting a pretraining
   seed count is zero. OLMo-1B-0724 has 1,446 Hub revisions but only two inside the ~1–5B
   token window where four independent measurements place formation. PolyPythias (10 seeds ×
   5 sizes × 154 checkpoints, 11 before 2B tokens) is the only instrument for formation and
   seeds — but Pythia is absent from Yu's Table 2, so "does any ≤1B model have one?" must
   run first. Abruptness is configuration-dependent (Xu 2026: same measure, gradual in
   Pythia-1B, sharp in OLMo-1B). From-scratch training at Gu's 60M recipe costs ≈10–15
   GPU-h per run; a 12-arm × 3-seed sweep is 400–900 single-GPU hours. No multilingual suite
   has early checkpoints; the MT arm must attach to final checkpoints.
   *(notes_training_dynamics §2–5)*

9. **Compression and multilingual: the gap is real but the framing must be a measurement,
   not a protection recipe.** No compression method operates on a scalar weight (all act on
   channels, tokens, layer budgets, or architecture), so `rtn+SW ≈ rtn` is expected; NVFP4
   ranks `mlp.down_proj` the *least* quantization-sensitive projection, corroborating
   "importance ⟂ sensitivity". Per-language PTQ damage is large and unexplained (Marchisio;
   Marie & Fujita) — no activation statistic has ever been tried as a covariate. On the
   multilingual side there is **no super-weight × multilingual or × translation paper**;
   the closest competitor is *Language Lives in Sparse Dimensions* (EACL 2026: sparse residual
   dimensions at consistent indices switch output language), which must be positioned
   against explicitly (a scalar weight whose removal destroys *all* languages vs. dimensions
   that *switch* language). Two corrections to in-repo premises: the NLLB sink is `</s>`
   (78–87%), not the language tag (1.5–2%) — but decoder position 0 *is* the target-language
   tag by construction (NLLB report §8), so the question survives; and Zhao 2024's claim is
   0.13% of neurons, not "four neurons". *(notes_compression §2–5; notes_multilingual §2–5)*

---

## 2. Candidate problems, consolidated and ranked

Each: question · why open · experiment · what counts as an answer · cost · risk. Costs assume
the existing detector/ablation harness on single-GPU SLURM jobs with 7–13B models.

### A. Why does removing a super weight make the model repeat function words? — **centre**
- *Why open:* fact 5. Yu reports, nobody explains; Cancedda explains a similar effect in a
  different formalism on one family; the frequency direction is contested (Puccetti vs
  Macocco).
- *Experiment.* On the five models with an established catastrophic single-scalar ablation
  (OLMo-1B, Mistral-7B, Llama-7B, OLMo-7B, Llama-30B), under the zero/mean/random triple:
  (i) sink mass (`Sink^ε_1`, ε=0.3) and per-head attention entropy before/after; (ii) KL of
  the next-token distribution to the corpus unigram, against a norm-matched random-residual
  null decoded through the final norm; (iii) Stolfo's frozen-final-norm ablation to size the
  rescaling component; (iv) Δlogit regression onto {all-ones, centred log-frequency,
  residual}; (v) the dissociations — patch the massive activation back at its onset layer
  with *generation* as the readout (Yu did accuracy only), and destroy the sink without
  touching the weight (drop BOS, as Barbero) to see whether the same signature appears;
  (vi) a dose–response sweep of the weight through {1, .75, .5, .25, .1, 0}; (vii) the
  repeated-token identity across 200 prompts (mutual information with a permutation null).
  Report rep-n / TNG alongside perplexity (a looping generation can have perplexity 2.99).
- *Answer:* per model, each hypothesis (sink loss → over-copying; unigram-prior fallback;
  final-norm rescaling) confirmed or excluded with CIs, and the sink-restoration /
  sink-destruction dissociation decided.
- *Cost:* ~1 GPU-hour per model per measurement; all forward passes. **Cheap.**
- *Risk:* Yu's own Table 1 says restoring the activation recovers only 42% of accuracy, so
  expect a partial dissociation; patching is itself off-manifold — the triple controls it.

### B. Does the collapse look the same in every language, and in translation models? — **the MT arm**
- **B1 — attractor identity across target languages.** Ablate; force each target language
  (Tower/EuroLLM/Aya, and NLLB); collect the degenerate output; rank-correlate its unigram
  distribution against that language's corpus unigram and against other languages'. Fixed
  token across languages ⇒ a pinned direction; language-varying ⇒ fallback to the
  target-language prior. Null = the corpus unigram argmax. Contrast arm: LAPE-neuron ablation
  (Tang 2024 Table 2 shows code-switching, not repetition) — the cheapest defence of "a
  different object from language neurons". *Cost: hours.*
- **B2 — per-language damage profile and what predicts it.** ≥20 FLORES languages × 3–4
  models × {intact, super weight (or minimal set) zeroed, magnitude-matched random weight
  zeroed}; chrF++ and COMET plus forced-decode KL (chrF++ floors under collapse); Spearman
  the per-language drop against per-language PTQ degradation (already measured in this repo
  and in Marie & Fujita), pretraining token share, script. Reuse the n≈960 / chat-template /
  COMET protocol in `compression/experiments/replication-uneven-ptq/`. Answer = a profile
  with CIs and family size, and either a covariate with ρ and CI or a bounded null. *Cost:
  the largest item; a few GPU-days.* Risk: resource level is collinear with everything;
  report partial effects.
- **B3 — NLLB per stack.** Yu's detection on encoder and decoder of NLLB-600M *and* 3.3B
  (600M is distilled); per-position residual norms and decoder self-attention sink rate; is
  the decoder's massive activation on position 0 = the language tag, and does replacing the
  tag with an arbitrary token move it? Then dose–response of ALTI+ source contribution and
  TNG oscillation rate. A parked run suggests the 600M encoder's activation grows
  monotonically with no single weight behind it — unverified; treat as motivation only.
  Either outcome is a result (a super weight per stack, or a bounded absence in an
  encoder-decoder MT model). *Cost: moderate; the harness needs an encoder-decoder adapter.*

### C. What is the unit, and when is "no super weight" a defensible claim? — **the methods floor**
- *Why open:* facts 1, 2, 7.
- *Experiment.* Ablate at three granularities (scalar; whole `down_proj` input column = the
  intermediate neuron; corresponding gate/up rows) × the control triple, on the ~12 models
  already cached; test whether fan-out count predicts the individual-vs-joint gap; re-verify
  the parked joint-ablation result under a spec Adrian owns. Build the null: step-0
  checkpoints (free in Pythia/OLMo) and within-matrix permutation, exceedance probability
  with FDR over the ~10⁷ entries; validate by recovering Yu's coordinates, *not* flagging
  Subramanian's inert top-magnitude ones, and recovering a planted weight in a small model.
- *Answer:* a rule of the form "criticality is a property of the intermediate neuron; a
  scalar is catastrophic iff it carries most of that neuron's fan-out", and a criterion
  with a stated false-positive rate.
- *Cost:* low (forward passes, already-cached models). Risk: magnitude may be the wrong
  statistic entirely; the fallback is to calibrate the activation-spike detector Yu actually
  used, permuting the intermediate-neuron index.

### D. When does the super weight form, and is it seed-stable? — **stretch**
- Gate **D0** first: detection + single/set ablation on Pythia {70m,160m,410m,1b} × 3 seeds,
  OLMo-2-1B, SmolLM2, and Ettin encoder vs decoder at 150M, at final checkpoints. Cheap, and
  it answers the proposal's RQ2 (sub-1B / encoder) question on its own.
- If positive: PolyPythias 160M/410M, 10 seeds, 154 checkpoints, criticality score
  `c(t) = log10(PPL_zeroed/PPL)` and a pre-registered transition-width criterion with the
  grid floor stated; row-vs-column seed agreement (Yu's two OLMo-1B runs share row 1764);
  the `-data-seed`/`-weight-seed` runs to attribute agreement. Name PolyPythias 410M seeds
  3 and 4 (the loss-spike outliers).
- From-scratch training (RQ4): budget-feasible (400–900 GPU-h) but not this semester unless
  D0 is positive and A–C are done.

### E. Compression-side diagnostics — **deprioritised**
- E1: what a QuaRot/SpinQuant rotation does to the super activation and whether the super
  weight stays localised (cheap; either outcome publishable). E2: is the 2-bit attractor the
  ablation attractor (compare token distributions, matched on perplexity). E3: does Wanda/OWL
  importance rank the super weight top, and does forcing it out matter. E4: does the
  massive-activation onset layer predict earliest prunable depth (weakest: constant
  relative-depth confound). None of these carry the MT component; E2 is the one that would
  slot into A as an extra arm if time allows.

### Ranking

| Rank | Problem | Novelty | Cost | Fit to goals (explain · MT · interp skills · RE skills) |
|---|---|---|---|---|
| 1 | **A** mechanism of collapse | high — unexplained since 2024, tooling exists | low | ✓ ✓(via B) ✓✓ ✓ |
| 2 | **B1** attractor across languages | high, unoccupied | very low | ✓ ✓✓ ✓ ✓ |
| 3 | **C** unit + calibrated criterion | high (no null exists) | low | ✓ – ✓ ✓✓ |
| 4 | **B2** per-language damage profile | high, unoccupied | high | ✓ ✓✓ ✓ ✓✓ |
| 5 | **B3** NLLB per stack | high, high variance | moderate | ✓ ✓✓ ✓✓ ✓ |
| 6 | **D0** gate ≤1B / encoder | medium | low | – – ✓ ✓ |
| 7 | **D** formation trace | high | high | ✓ – ✓ ✓✓ |
| 8 | **E** compression diagnostics | medium | low–mod | – – ✓ ✓ |

---

## 3. Recommendation

**Reorganise the project around one question — "what does a super weight do, such that
removing it makes a model emit repeated function words, and does the answer hold across
languages and in translation models?" — with A as the core, B1+B2 as the MT arm, C as the
methods floor, and D0 as the cheap existence gate. Formation (D) and from-scratch training
become stretch.**

Why this shape:
- It is the professor's suggestion, sharpened: A is exactly "explain the we-we-we", and the
  literature has left it open while supplying every instrument (fact 5).
- It carries the MT component by construction, not decoration: B1 discriminates two
  mechanisms *because* it varies the language, and B2 answers a question the MT-compression
  literature has posed without a mechanism (fact 9).
- The interp skills it exercises are the ones worth having: hooks and activation patching,
  logit/tuned lens, attention analysis, ALTI+, controlled ablation with nulls.
- The research-engineering skills are forced, not optional: the control triple, provenance
  per run, a planted-weight test, multiplicity-corrected families, and re-verifying a parked
  result under an owned spec (C).
- It survives null outcomes. If no hypothesis in A wins cleanly, the dissociation table is
  still a finding; if B1 gives a fixed token, that is a mechanism claim; if B2 gives no
  covariate, it is a bounded null over ≥20 languages — the first such table in the
  literature.

What it costs relative to the earlier RQ1+RQ2 scope (memory: formation trace + calibrated
detector as core): the formation trace is demoted. Two reasons that is defensible: fact 8
(the instrument problem is worse than the proposal assumed — OLMo cannot see formation,
PolyPythias needs an existence gate first, and no multilingual suite exists), and the
professor's request for clarity and a solvable target. D0 keeps the door open at almost no
cost; if it is positive and A–C finish early, D is the natural continuation.

**Things the proposal wording must change**, regardless of scope (all from the notes):
say *candidate coordinate* until C decides the unit; report every ablation as zero / mean /
matched-random; state coverage on every cross-model sentence and check a model from Owen's
negative list before writing "every"; report rep-n or TNG next to perplexity; check the last
2–3 layers before claiming "persists to the final layer" (Sun 2024 §2.1 and Sun 2026 Table 1
report decay there); position against *Language Lives in Sparse Dimensions*; drop the
"four neurons" and "sinks on language tags" premises; write "our hypothesis".

## 4. Where to look next

`literature/notes_super_weights_massive_activations.md` §3–5 (mechanism hypotheses H1–H7 and
the control triad), `literature/notes_repetition_degeneration.md` §3–5 (H1–H5 with
measurements, the MT bridge, OP1–OP7), `literature/notes_multilingual_mt_interpretability.md`
§3–5 (bridge questions (i)–(v), P1–P4, proposal-wording changes),
`literature/notes_training_dynamics_checkpoint_suites.md` §2 (instrument table, Hub-verified)
and §5 (P0–P5 with the abrupt-vs-gradual criterion), `literature/notes_compression_pruning_quantization.md`
§2 and §5. Highest-priority unread PDF per the readers: `Oh-2024-House-of-Cards-Massive-Weights.pdf`.
