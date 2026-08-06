---
name: new-experiment
description: Start an experiment spec-first — question, satisfied-when, null, and analysis plan before any runner exists. Use when beginning any new experiment, run, sweep, ablation, or research question in any track.
---

# Start an experiment, spec first

The spec is not a software design document. It is **the falsifiable claim the
run is supposed to settle, written down before the run can bias it.**

This is preregistration scaled to one person. An analysis plan chosen after
seeing the numbers is not a plan, it is a search for a result.

## Order of operations — do not reorder

**1. Read `docs/registry.md` first.** Several ideas here are already dead with
evidence. If the proposal is adjacent to a ruled-out one, say explicitly what is
different about it.

**2. Write `<track>/experiments/<name>/README.md`** from
`docs/templates/experiment_readme.md`. Every section, before any code:

- **Question** — one sentence, ends in a question mark.
- **Hypothesis** — what we expect and why. "Our hypothesis", never "the mechanism".
- **Satisfied when** — the falsifiable criterion, *with the number in it*.
  "A per-component map exists" is a deliverable, not a criterion.
  "Spearman ρ between importance and sensitivity has a 95% CI excluding 0,
  n_seeds ≥ 5" is a criterion.
- **Null / baseline** — what this is measured against. **Every measure must beat
  its random-init value**; in a checkpoint study, step 0 is a free null. If the
  baseline cannot be named, the experiment cannot run — "worse than not
  quantizing" reached three documents because nobody wrote this section.
- **Analysis plan** — the statistic, the seed count, the family size for
  multiplicity correction, and what counts as "no effect" (a bounded interval,
  never a bare negative).
- **What would change my mind** — the result that kills the hypothesis.
- **Out of scope** — what this run deliberately does not answer.

**3. Confirm the satisfied-when with the user before writing configs.** This is
worth interrupting them for. A criterion agreed after the fact is worth nothing.

**4. Then** create `configs/` (one YAML per run, **seed declared in every one**),
`slurm/`, and `notes.md` with a dated first entry.

**5. Wire provenance into the runner before the first real run**, not after. The
first thing the runner does with an output directory is write a manifest
carrying: git sha, working-tree hash, dirty flag, the **resolved** config
contents (not its path), seed, library versions, SLURM job id. A results
directory without one is not a result. Retrofitting this is how you end up with
numbers you cannot cite.

## Rigor floor — apply while designing, not while writing up

- Report `effect [95% CI], n_seeds=N, family=M`. Never a bare mean.
- **Seeds are the unit of independence** — not checkpoints, not layers, not
  heads. Randomising only weight init caps effective n at ≈2 however many seeds
  you run; vary data order and shuffling too.
- Correct for multiplicity on any scan over layers/checkpoints/heads, and state
  the family size in the caption.
- n=24–32 is exploratory. Nothing at that scale is a paper claim.

## Refuse to proceed if

- The satisfied-when is a deliverable rather than a falsifiable criterion.
- The baseline is unnamed.
- The user asks for the runner before the README exists. Say why, write the
  README, then write the runner.

## When the runner is the first code in a track

Check the trigger table in `docs/research_standards.md` §20.3. Some mechanisms
are due the moment code exists — the config-loads test, the manifest writer —
and are far cheaper to add now than to retrofit.
