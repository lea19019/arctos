---
name: close-experiment
description: Close out an experiment — record the verdict against its satisfied-when, back-annotate whatever it invalidates, and file the negative. Use when a run finishes, a result gets written up, or an idea is abandoned.
---

# Close an experiment

Most of the value in a research repo is in the closing, and closing is what gets
skipped. `docs/learning/learning_log.md` was created for dead ends and has zero
entries. Negatives are the most reused content here and the least well kept.

## 1. Record the verdict — including "not met"

Append to the experiment README:

```markdown
## Verdict

**Not met.** <one sentence, with the number>
Evidence: <path to results dir> · <git sha> · n_seeds=<N>
```

Three verdicts are legitimate: **met**, **not met**, **undecided**. "Undecided"
must say what would decide it. A satisfied-when with no verdict is an experiment
that never ended.

## 2. Back-annotate what this invalidates

**The highest-value practice in the whole survey, and the one this repo is worst
at.** OPT's chronicle edits `[WARNING: see debrief]` banners into the *earlier*
entries a later finding invalidated. Anthropic puts retraction banners at the top
of superseded Circuits Updates. Append-only correction is not enough, because the
wrong number is the one that gets read.

Search every place the superseded number appears — findings docs, the registry,
the paper, experiment READMEs, notebook outputs — and edit each **in place**:

```markdown
> **Superseded 2026-08-06.** This section reported ρ = −0.18; the raw data gives
> −0.058. Corrected in <sha>. The conclusion below does not survive.
```

A findings doc and a paper that disagree is a state this repo has already been in.

## 3. File the negative, the day it happens

- Ruled-out idea → `docs/registry.md` under **Ruled out**, with the evidence and
  **the numbers that killed it**. "Didn't work" is not an entry.
- When a proposal becomes a negative, **update the proposal document itself**
  with a results section. A dead proposal still reading as live gets
  re-proposed — that has happened here.
- One dated line in `docs/learning/learning_log.md`. BigScience's equivalent is
  six bare lines and is among the most reused artifacts in that repo.

## 4. A failed run is a finding until proven otherwise

Read the log before explaining the failure. "Transient CUDA faults" was written
here for what the logs show to be a deterministic
`linalg.cholesky: not positive-definite` bug. If the cause is unknown, write
"cause unknown" and quote the traceback.

## 5. Before the writeup lands

Run `/audit-claim` over it. Check every number traces to a results directory
with a manifest.
