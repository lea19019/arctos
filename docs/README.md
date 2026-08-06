# docs — what spans the tracks

**The rule:** a track's own write-ups live with the track. This folder holds only
what is shared or outlives any one track.

```
docs/
├── registry.md            ← what has been done and what is ruled out. Read before proposing anything.
├── research_standards.md  ← how experiments are run here, with sources.
├── learning/              understand the subject matter — read these first if you're new
├── plans/                 forward-looking, across tracks
├── archive/               history from before the tracks split
└── papers/                shared PDFs
```

Per-track write-ups:

| Track | Its docs |
|---|---|
| `compression/` | [`../compression/docs/`](../compression/docs/) — q1/q4/q5/q6, phase two, the primers, the bibliography |
| `speech-translation/` | [`../speech-translation/docs/`](../speech-translation/docs/) — the NLLB/XTTS surveys |
| `interlingua/` | [`../interlingua/docs/`](../interlingua/docs/) + [`../interlingua/papers/`](../interlingua/papers/) — the Tier 1 plan, the MS proposals, the literature sweep |

## I want to…

| …do this | …go here |
|---|---|
| Check whether an idea was already tried or killed | [`registry.md`](registry.md) |
| Know how to run an experiment properly here | [`research_standards.md`](research_standards.md) |
| **Understand the subject matter from scratch** | [`learning/project_foundations.md`](learning/project_foundations.md) |
| Read a whole track in order | that track's `docs/README.md` |
| Look up a compression result | [`../compression/docs/`](../compression/docs/) |
| Pick up an open thread | [`plans/open_work.md`](plans/open_work.md) |
| Find a paper to read | [`learning/reading_list.md`](learning/reading_list.md) |

## The two documents at the top

[`registry.md`](registry.md) — the audited record (2026-08-05). What was
attempted, what was established, **what was ruled out with the numbers that
killed it**, and where documentation and raw data disagree. Several plausible
ideas are already dead; this is where you find out before spending a week. It
spans all three tracks, which is why it lives here.

[`research_standards.md`](research_standards.md) — the sourced backing for
[`../CLAUDE.md`](../CLAUDE.md). §§1–11 are research method (statistics, null
models, interventions, representational similarity, phase transitions,
checkpoint suites, reporting). §§12–19 are the engineering layer. A lookup
reference, not a read-through.

## [`learning/`](learning/) — explanations

The teaching material, kept here because it is about the *subject*, not about
any one track's results: what quantization is, how these architectures work,
every paper worth reading, and the math curriculum. Start with
[`project_foundations.md`](learning/project_foundations.md). See
[`learning/README.md`](learning/README.md).

Track-specific primers are with their track — e.g. the find/keep/shrink/prune
framework is
[`../compression/docs/compression_primer.md`](../compression/docs/compression_primer.md).

## [`plans/`](plans/) — what's next

[`open_work.md`](plans/open_work.md) is the useful one: a ranked brief of open
directions mined from the future-work sections of 95 cited papers, cross-checked
against what is already done or ruled out — and it spans tracks, which is why
it's here rather than in one of them. Also [`roadmap.md`](plans/roadmap.md) (the
6-month program) and [`skills_plan.md`](plans/skills_plan.md).

## [`archive/`](archive/) — history

Advisor briefs, the phase-one plan, the MS project plan, proposal paperwork,
process notes. These were written **before the repo split into tracks**, when
"phase one / phase two" *was* the whole project — which is why they're here and
not under `compression/`. Accurate for their moment, superseded now. See
[`archive/README.md`](archive/README.md).

## Naming

Markdown files are `lower_case_with_underscores.md`. The only exceptions are
`README.md` (a convention every tool expects) and PDFs, which keep bibliographic
filenames so they sort by author.

Experiment folders under `compression/experiments/` still use dashes
(`q6-compression/`) — those names are hard-coded in SLURM scripts and configs,
so they were deliberately left alone.
