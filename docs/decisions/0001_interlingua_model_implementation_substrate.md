# 0001 — Model implementation substrate for interlingua from-scratch training

- **Status:** proposed — **blocked on an open PI decision**, see Context
- **Date:** 2026-08-06 (rewritten same day; see "Correction" at the end)

## Context

The interlingua track trains small models from scratch and computes
interpretability analyses per checkpoint. Two documents disagree about **what
gets trained**, and that disagreement is not settled:

- **`interlingua/docs/does_the_interlingua_grok_ringger_2026.pdf`** (the
  proposal) specifies **three architecture arms** at the same scale — 6 layers,
  512-dim, 8 heads:
  1. encoder-only, mBERT-like, MLM objective
  2. encoder-only, XLM-R-like, RoBERTa-style training
  3. **encoder-decoder, NLLB-like**, translation objective on parallel data

  **H3a is a comparison between the mBERT-like and NLLB-like arms** — that is,
  the architecture contrast is a hypothesis under test, not an implementation
  choice. The proposal also flags H3a as confounded (the arms differ in
  regularization as well as architecture) and proposes extra variants to
  deconfound it. Tier 3 replicates the progress measures on real mBERT and XLM-R.

- **`interlingua/docs/tier1_plan.md`** §3.1 narrows Tier 1 to **decoder-only**,
  and §2 scopes the NLLB-like encoder-decoder arm **out**, on tooling grounds:
  Jian & Manning's JSD measures are defined over next-token distributions, and
  TransformerLens / SAELens / circuit-tracer are decoder-first with second-class
  encoder support. That section states plainly: *"This is a PI decision, not
  mine,"* and §8 lists decoder-vs-encoder as open question #1.

**So the architecture family is open.** Any substrate decision that assumes
decoder-only presupposes the answer to a question the PI has not answered, and
would foreclose the arm that H3a needs.

Separate constraint, independent of which arms are run: `transformer-lens`'s
successor package has two open bugs that silently corrupt from-scratch training
and checkpoint reloading (`docs/research_standards.md`, interlingua section),
so its native training path is not a safe base for any arm.

## Decision (proposed)

We will use **standard HuggingFace `transformers` model classes** as the
training substrate, chosen per arm — `LlamaForCausalLM` for a decoder arm,
`BertForMaskedLM` / `RobertaForMaskedLM` for the encoder arms, an
`M2M100`/`EncoderDecoderModel`-style class for the NLLB-like arm — and attach
interpretability tooling on top via a wrapper rather than training inside an
interpretability framework.

The point of this choice is that **it does not commit to an architecture
family.** Because the architecture contrast is itself a hypothesis, the
substrate must be the one that supports all three arms at equal quality.

Where `transformer-lens` is needed for analysis, pin `transformer-lens==3.6.0`
and use the deprecated `HookedTransformer` path.

## Alternatives considered

- **Train inside `transformer-lens` natively** — rejected: the successor's two
  open bugs hit exactly from-scratch training and checkpoint reloading, which is
  the entire workload, and silent corruption is the worst failure mode available
  because it yields plausible checkpoints. It is also decoder-first, which would
  bias the substrate toward one arm of a comparison the study is trying to make.
- **Commit to decoder-only and use `LlamaForCausalLM` alone** — rejected *for
  now*: this is tier1_plan's recommendation and it may well be what the PI
  chooses, but adopting it as a substrate decision would quietly settle open
  question #1 and drop the NLLB-like arm that H3a is defined over. If the PI
  narrows Tier 1 to decoder-only, this record gets superseded, not edited.
- **Bespoke minimal transformer** — rejected: forfeits HF checkpoint/tokenizer
  tooling and breaks comparability with Pico's layer naming, which brings free
  CKA/PWCCA/effective-rank tooling and two public baselines bracketing 36M.

## Consequences

- Analysis code must not assume a decoder. Anything written against next-token
  distributions needs an explicit story for the MLM arms — the JSD measures are
  the concrete case, and tier1_plan is right that reconstructing masked-token
  distributions is workable but non-standard and breaks the comparison to Jian &
  Manning's baseline. **That is an unresolved measurement problem, not a
  formatting detail.**
- Interpretability tooling support is genuinely uneven across arms
  (TransformerLens, SAELens and circuit-tracer are decoder-first). This is a real
  cost of keeping the encoder arms, and it needs its own decision record once
  the PI settles the arms.
- HF classes are not hook-native, so internal-activation analyses go through a
  wrapper. `HookedModel` exists at `compression/src/models/_hooked.py:133`;
  **how much of `compression/src/` (5,140 lines total) transfers unchanged has
  not been measured** — an earlier "~900 lines" figure in `CLAUDE.md` is
  unverified.
- ~~`run_with_cache` raises on batch size > 1. Fix before running the full
  checkpoint × seed matrix.~~ **Stale — 2026-08-06.** It does not; the
  `NotImplementedError` is scoped to `generate(return_cache=True)` and issue
  #1265 closed 2026-04-22. Struck rather than deleted: the record is immutable.
  See `interlingua/docs/method_landscape.md` §3.3.
- Checkpoints are HF-standard: reusable and publishable, larger than a minimal
  state dict.

## Open — must be answered before this can move to accepted

1. **Which architecture arms does Tier 1 run?** (`tier1_plan.md` §8 Q1.) Until
   the PI answers, this record stays `proposed`.
2. If encoder arms are in, how are the JSD measures defined on MLM outputs
   without breaking comparability to the exemplar-first baseline?

---

## Correction

The first version of this record (2026-08-06) asserted a single decoder-only
substrate and was marked `accepted`. Both were wrong. It was written from a
bullet in `CLAUDE.md` without reading the proposal PDF, which specifies three
architecture arms and makes the architecture contrast a hypothesis; and it was
marked accepted without the decision having been made by anyone. Recorded here
rather than silently rewritten, per `docs/research_standards.md` §20.6.
