# docs — the written record

Findings, reading, and planning for the `compression/` and `speech-translation/`
tracks. Most of what's here describes work that is **done or parked**, not in
progress — this is the archive you mine when picking up a thread, not a to-do
list.

The `interlingua/` track keeps its own docs and bibliography under
[`../interlingua/`](../interlingua/), since they're specific to it.

## Start here

| Document | What it is |
|---|---|
| [`OPEN-WORK.md`](OPEN-WORK.md) | **The most useful doc in this folder.** A ranked brief of open directions, mined from the future-work and limitations sections of 95 cited papers and cross-referenced against what has already been done or ruled out. Honest novelty labels. |
| [`READING-GUIDE.md`](READING-GUIDE.md) | An ordered path through every experiment and result, phase one → phase two. |
| [`project-summary.md`](project-summary.md) | The thesis spine. |
| [`ROADMAP.md`](ROADMAP.md) | The multi-dimensional "sweet spot of compression for translation" program (quant × prune × distill across bit-scales). |

## Findings (`findings/`)

Per-question writeups backing the two tracks.

**Phase one — interpretability:** [`q1.md`](findings/q1.md) (when and where the
target language emerges), [`q5.md`](findings/q5.md) (**the pivotal negative** —
importance ⟂ quantization sensitivity), [`architecture-comparison.md`](findings/architecture-comparison.md)
(does the depth signature generalize).

**Phase two — compression:** [`phase2-synthesis.md`](findings/phase2-synthesis.md)
(the honest consolidated read — start here for conclusions),
[`phase2-results.md`](findings/phase2-results.md) (cross-model tables),
[`q6.md`](findings/q6.md) (the chrF++ sweep),
[`compression-primer.md`](findings/compression-primer.md) (the find/keep/shrink/prune
framework), [`phase2-method-primer.md`](findings/phase2-method-primer.md) (method +
literature-gap map), [`phase2-novel-direction.md`](findings/phase2-novel-direction.md)
(the pipeline-aware idea — read knowing it became a **negative result**).

**Speech + deployment:** [`compression-nllb-xtts-research.md`](findings/compression-nllb-xtts-research.md),
[`interp-lrl-nllb-xtts.md`](findings/interp-lrl-nllb-xtts.md).

**Replication:** [`replication-uneven-ptq-mt.md`](findings/replication-uneven-ptq-mt.md) —
independent replication of arXiv:2508.20893, with the paper's red flags noted.

`findings/deep-research-raw/` holds the unedited deep-research transcripts behind
the synthesis docs.

## Background and learning

- [`llm-quant-foundations.md`](llm-quant-foundations.md) — architecture + compression reference covering all three tracks.
- [`project-foundations.md`](project-foundations.md) — what quantization is, conceptually. Read before the above.
- [`READING-LIST.md`](READING-LIST.md) — every cited paper, grouped by theme, plus videos and sites for the foundations.
- [`MATH-PLAN.md`](MATH-PLAN.md) — 6-month math curriculum.
- [`research.md`](research.md) — annotated bibliography + deep-research addenda.
- [`learning-log.md`](learning-log.md) — running notes.

## Planning and administrative

- [`PHASE1-PLAN.md`](PHASE1-PLAN.md) — the original phase-one investigation plan and V1/V2/V3 claim structure.
- [`ms-project-plan.md`](ms-project-plan.md) — MS project planning doc and research roadmap.
- [`proposal-form.md`](proposal-form.md) — CS 698R master's project approval requirements.
- [`advisor-brief.md`](advisor-brief.md), [`project-ideas-advisor-brief.md`](project-ideas-advisor-brief.md) — talking docs.
- [`replication-uneven-ptq-mt-brief.md`](replication-uneven-ptq-mt-brief.md) — the replication brief.
- [`phase2-hypotheses.md`](phase2-hypotheses.md), [`ideas.md`](ideas.md) — candidate directions, mostly superseded by `OPEN-WORK.md`.
- [`skills-plan.md`](skills-plan.md) — career/interview plan through spring 2027 (DSA cadence, production toolkit, application calendar, visa checklist). **Overlaps** [`ROADMAP.md`](ROADMAP.md) and [`MATH-PLAN.md`](MATH-PLAN.md), which cover research skills over a similar horizon — worth reconciling into one plan rather than three.
- [`SESSION-HANDOFF.md`](SESSION-HANDOFF.md), [`claude-code-bootstrap.md`](claude-code-bootstrap.md) — working-process notes.
- [`systems-notes/`](systems-notes/) — cluster and environment notes.

## Papers (`papers/`)

- `pruning_project.pdf` — prior work applying IFR-guided pruning.
- `mmmc-multilingual-corpus.pdf` — MMMC: A Massively Multi-way-aligned Multilingual Corpus.
