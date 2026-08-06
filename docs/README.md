# docs — what spans the tracks

**The rule:** a track's own write-ups live with the track. This folder holds only
what is shared or outlives any one track.

```
docs/
├── registry.md            ← what has been done and what is ruled out. Read before proposing anything.
├── research_standards.md  ← how experiments are run here, with sources.
├── decisions/             decision records — one per choice that constrains future work
├── templates/             the shapes an experiment README, notes entry, and decision record take
├── learning/              understand the subject matter — read these first if you're new
├── archive/               written for a moment that has passed; kept for context
└── papers/                shared PDFs
```

**There is no `plans/` here.** The live direction is `interlingua/`, and its plan
lives with the track. The compression program's roadmap and backlog were moved to
[`archive/`](archive/) — see [below](#a-note-on-whats-not-being-pursued).

That plan is **under audit**: two 2026-08-06 reviews partly supersede it, and
`interlingua/README.md` carries both verdicts. Read them before proposing work
in that track.

Per-track write-ups:

| Track | State | Its docs |
|---|---|---|
| `interlingua/` | **the live direction** — docs only, no code | [`../interlingua/docs/`](../interlingua/docs/) + [`../interlingua/papers/`](../interlingua/papers/) |
| `compression/` | concluded, being left behind | [`../compression/docs/`](../compression/docs/) |
| `speech-translation/` | baseline sweep done | [`../speech-translation/docs/`](../speech-translation/docs/) |

## I want to…

| …do this | …go here |
|---|---|
| See what's actually being pursued | [`../interlingua/README.md`](../interlingua/README.md) — the plan **and** the two audits above it |
| Check whether an idea was already tried or killed | [`registry.md`](registry.md) |
| Know how to run an experiment properly here | [`research_standards.md`](research_standards.md) |
| **Understand the subject matter from scratch** | [`learning/project_foundations.md`](learning/project_foundations.md) |
| Find out why a past choice was made | [`decisions/`](decisions/) |
| Start an experiment / a notes file | [`templates/`](templates/) |
| Look up a compression result | [`../compression/docs/`](../compression/docs/) |
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
checkpoint suites, reporting). §§12–19 are the engineering layer, and §20 says
which of these can be mechanised and when to build each. A lookup reference, not
a read-through.

## [`decisions/`](decisions/) and [`templates/`](templates/)

One file per decision that constrains future work, Nygard format, immutable once
written — a superseded record gets a status line, not an edit. Use
`/record-decision`. The templates are the shapes that
`/new-experiment`, `/close-experiment`, and `/record-decision` fill in.

## [`learning/`](learning/) — explanations

The teaching material, kept here because it is about the *subject*, not about any
one track's results: what quantization is, how these architectures work, every
paper worth reading, and the math and skills plans. Start with
[`project_foundations.md`](learning/project_foundations.md). See
[`learning/README.md`](learning/README.md).

Track-specific primers are with their track — e.g. the find/keep/shrink/prune
framework is
[`../compression/docs/compression_primer.md`](../compression/docs/compression_primer.md).

## A note on what's *not* being pursued

[`archive/roadmap_compression_program.md`](archive/roadmap_compression_program.md)
and [`archive/open_work_compression.md`](archive/open_work_compression.md) are
the six-month compression-for-MT program and its ranked backlog of open
directions. **Nothing in them was refuted** — the backlog is a genuinely good
piece of work, mined from the future-work sections of 95 papers. They are
archived because the direction changed, not because the ideas died. If
compression ever comes back, start there.

Everything else in [`archive/`](archive/) predates the split into tracks:
advisor briefs, the phase-one plan, proposal paperwork, process notes. See
[`archive/README.md`](archive/README.md).

## Naming

Markdown files are `lower_case_with_underscores.md`. The only exceptions are
`README.md` (a convention every tool expects) and PDFs, which keep bibliographic
filenames so they sort by author.

Experiment folders under `compression/experiments/` still use dashes
(`q6-compression/`) — those names are hard-coded in SLURM scripts and configs,
so they were deliberately left alone.
