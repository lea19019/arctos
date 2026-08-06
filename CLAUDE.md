# Arctos — working rules

Three research tracks, each self-contained with its own `pyproject.toml` and
`.venv`. Nothing runs at the top level.

```
interlingua/         training dynamics of cross-lingual alignment (next; docs only)
compression/         interp → quantization (concluded)
speech-translation/  NLLB + XTTS dubbing pipeline
notebooks/           method tutorials (import from compression/src)
docs/                only what spans tracks
```

**A track's write-ups live with the track**, in `<track>/docs/`. Top-level
`docs/` holds only `registry.md`, `research_standards.md`, `decisions/`,
`templates/`, `learning/` (explanations of the subject), `archive/`, and
`papers/`. Put a new finding in the track that produced it.

**There is no top-level `plans/`.** The compression program's roadmap and its
ranked backlog are in `docs/archive/` — those directions are still *open*, but
they are not being pursued. The live direction is `interlingua/`, and its plan
lives with the track in `interlingua/docs/tier1_plan.md`. Do not propose
compression work unless asked for it by name.

**`tier1_plan.md` is partly superseded.** Two 2026-08-06 audits sit above it —
`interlingua/docs/prior_work_map.md` (prior art, claim by claim: the surviving-
novelty claim in §7 does not survive) and `interlingua/docs/program_critique.md`
(the program's premises; verdict *salvageable, but much smaller*, with two
findings that are measurements rather than literature). **Read both before
proposing anything in this track.** Tier 1 as written is not to be built.

Markdown docs are named `lower_case_with_underscores.md`; `README.md` and
bibliographic PDFs are the exceptions.

**Read [`docs/registry.md`](docs/registry.md) before proposing any experiment.**
It records what was done, what was ruled out with evidence, and which documented
claims failed an audit against the raw data. Several plausible ideas are already
dead.

Full sourced standards: [`docs/research_standards.md`](docs/research_standards.md).

**Workflows live as skills, not as prose.** Invoke them rather than
improvising: `/new-experiment` (spec before runner), `/close-experiment`
(verdict, back-annotation, filing the negative), `/audit-claim` (check a claim
against the raw data before it lands), `/record-decision` (a decision record in
`docs/decisions/`).

**Writing the first code in a track? Read `docs/research_standards.md` §20.3
first.** It is a short trigger table saying which mechanism is due *now* — the
provenance manifest writer, a test that every committed config loads, invariant
tests for any similarity/probe measure — and which can wait. All of these are
cheap to add up front and expensive to retrofit; provenance especially, because
retrofitting it is how you end up with results you cannot cite.

---

## Claim hygiene — the rule that matters most here

A 2026-08 audit found the measurements in this repo are sound and several
*claims about them* are not. The failures were never bad math; they were claims
drifting from their evidence. Guard against exactly that:

1. **State the comparison, not the impression.** "MT-calibrated GPTQ beats
   generic-calibrated GPTQ" is supported. "MT-calibrated GPTQ recovers the 3-bit
   cliff" is not — against plain RTN, Aya gets much worse. Write the baseline
   into the sentence.
2. **Never write a claim whose baseline was not measured.** "Worse than not
   quantizing" appeared in three documents; FP16 was never scored.
3. **State coverage next to every cross-model claim.** Two headline results here
   silently rest on 6 of 8 and 7 of 8 models, in both cases missing the model
   most likely to complicate them.
4. **If a metric is substituted, say so where the number is used**, not only in
   the runner's docstring. NLLB's "IFR" is a different quantity in the same table
   column as real IFR values.
5. **If you drop a data point, name it and say why, in the doc.** A published
   n=17-of-18 statistic has no record of which cell was dropped.
6. **Config ≠ what ran.** Report the values actually used. Q5's config specifies
   COMET/n=200/σ∈{1e-3,1e-2,5e-2}; the run used chrF++/n=20/σ=0.1.
7. **A failed run is a finding until proven otherwise.** "Transient CUDA faults"
   was written for what the logs show to be a deterministic
   `linalg.cholesky: not positive-definite` bug. Read the log before explaining
   the failure.
8. **Say "our hypothesis", not "the mechanism."**

## Rigor floor for any new result

- **Report effect size with a CI, not a bare mean.** Format:
  `effect [95% CI], n_seeds=N, family=M`.
- **Seeds are the unit of independence** — not checkpoints, not layers, not
  heads. Randomizing *only* weight init caps effective sample size at ≈2
  regardless of seed count; vary data order and shuffling too.
- **Every measure must beat its random-init value.** Probes, saliency maps, and
  SAEs all look interpretable on randomly initialized networks. In a
  checkpoint study, step 0 is a free null.
- **Never report "no effect"** — bound it with the interval.
- **Correct for multiplicity** on any scan over layers/checkpoints/heads, and
  state the family size in the caption.
- **n=24–32 is exploratory.** Nothing at that scale is a paper claim. The
  rigorous protocol already exists in
  `compression/experiments/replication-uneven-ptq/` (n≈960, chat templates,
  COMET) — reuse it rather than reinventing it.

## Adding an experiment

Use `/new-experiment`. Each track follows `experiments/<name>/` with `README.md`
(question, satisfied-when — template: `docs/templates/experiment_readme.md`),
`configs/*.yaml` (one per run), `slurm/`, and `notes.md` (a dated changelog,
including what broke — template: `docs/templates/notes_entry.md`).

- **One YAML per run.** No hidden defaults in code that override config.
- **Write `git_sha`, the resolved config, library versions, and the seed into
  every results directory.** A result that cannot be traced to its config is not
  a result.
- **A stub runner must not appear in any table as if it ran.** Two questions here
  were listed as experiments for months while raising `NotImplementedError`.
- **Define satisfied-when before running**, and record the verdict against it —
  including "not met."

## Recording negatives and dead ends

Negatives are the most reused content in this repo — and the least well kept.
`docs/learning/learning_log.md` was created for dead ends and has zero entries.

- A ruled-out idea goes in `docs/registry.md` under **Ruled out**, with the
  evidence and the numbers that killed it.
- When a proposal becomes a negative, **update the proposal document itself**
  with a results section. `phase2_novel_direction.md` still reads as a live
  proposal for an idea that died.
- Keep a **chronicle** for any long training run: a timestamped prose log of
  instabilities, restarts, and fixes, keyed to step numbers.

## What gets committed

Code, configs, SLURM scripts, docs, and **small result files** (`summary.tsv`,
`results.json`) that back a claim. Never: model weights, checkpoints, corpora,
bulk audio, conversion intermediates, venvs. Paper PDFs are indexed, not
committed. See `.gitignore` — it is organized per track with the reasoning inline.

## Environments

```bash
cd compression && uv sync --extra quant
cd speech-translation && uv sync --extra ct2 --extra quant
uv run --project compression python notebooks/01_logit_lens.py
```

SLURM: `compression/` scripts `cd` to their own folder (submit from anywhere);
`speech-translation/` scripts expect submission **from the repo root**. Compute
nodes have no internet — pre-cache on the login node. Python 3.11, torch cu128.

## For the interlingua track specifically

Not yet built. Decisions worth making before code exists, from
`docs/research_standards.md`:

- **This track is not decoder-only.** The proposal
  (`interlingua/docs/does_the_interlingua_grok_ringger_2026.pdf`) specifies three
  arms at 6L/512d/8h — encoder-only mBERT-like, encoder-only XLM-R-like, and
  **encoder-decoder NLLB-like** — and **H3a is the comparison between two of
  them**, so the architecture contrast is a hypothesis under test, not an
  implementation detail. `tier1_plan.md` §3.1 narrows Tier 1 to decoder-only on
  tooling grounds but says outright that this is a PI decision, and §8 lists it
  as open question #1 — and the 2026-08-06 method survey found **neither of
  §3.1's two stated reasons holds**: an MLM at `[MASK]` gives the JSD measures
  what they need and the encoder-decoder arm needs no adaptation, and TL 3.6.0
  ships an NLLB adapter while `inseq` is enc-dec-first and `pico-analyze` is
  architecture-agnostic. What is genuinely decoder-only is **SAEs and circuit
  tracing**. **Until the PI answers, write nothing that assumes a decoder** —
  substrate, measures, or tooling. See
  [`docs/decisions/0001`](docs/decisions/0001_interlingua_model_implementation_substrate.md).
- **Use standard HF `transformers` classes** as the training substrate, chosen
  per arm, and attach interpretability tooling via a wrapper. `HookedModel` is at
  `compression/src/models/_hooked.py:133`; how much of `compression/src/` (5,140
  lines) transfers unchanged is **unmeasured**.
- **Pin `transformer-lens==3.6.0`.** Its two open bugs (#1568, #1587) corrupt
  from-scratch training and checkpoint reloading, but both live in the
  `boot_native` train-inside-TransformerLens path — training in HF
  `transformers` and wrapping with `boot_transformers(hf_model=…)` avoids both,
  which is what decision 0001 already proposes. **Separate venvs are mandatory:**
  `transformer-lens` 3.6.0 needs `transformers>=5.9`, `circuit-tracer` pins
  `<=4.57.3`, `jlens` needs `>=5.5`.
- **Use a WSD/trapezoidal LR schedule, not cosine**, so checkpoints are
  comparable to each other.
- **Log-spaced checkpoints must stay dense past the induction-formation window**
  — Pythia's documented mistake is that its dense grid stops right before it.
- **`run_with_cache` does *not* raise on batch size > 1** — an earlier rule here
  said it does. The `NotImplementedError` is scoped to
  `generate(return_cache=True)`; issue #1265 closed 2026-04-22.
- Match **Pico**'s layer naming: free tooling (CKA, PWCCA, effective rank) and two
  public baselines bracketing 36M.

---

*Rules above are grounded in sourced practice where `docs/research_standards.md`
cites a source, and are editorial judgment otherwise. The research-engineering
layer — configuration, provenance, tracking, artifacts, repo structure, testing
numerical code, and how failure gets recorded — is §§12–19, compiled from
primary sources read as code. §20 says which of these rules can be mechanised,
which cannot, and what to build at which point.*

*Enforcement is deliberately unbuilt: there is no CI, no pre-commit hook and no
fitness-function suite, because the track they would govern has no code yet.
§20.3 names the trigger for each. Build them when the trigger fires, not before
— a gate whose failures are all pre-existing debt gets switched off within a
week.*
