# archive — accurate when written, not being pursued now

Nothing here is current. These documents are kept because they explain *why*
decisions were made, and because they are the paper trail for the MS project.
For what is true today, see [`../registry.md`](../registry.md). For what is being
worked on, see [`../../interlingua/`](../../interlingua/).

## The compression program — set aside, not refuted

These two are the direction being left behind. **Nothing in them was disproved**;
they are here because the research direction changed to `interlingua/`. If
compression ever resumes, this is where it resumes from.

- [`roadmap_compression_program.md`](roadmap_compression_program.md) — the
  six-month "sweet spot of compression for translation" program: a
  quantization × pruning × distillation frontier study, the statistical-rigor
  goals, and the speech↔text and cloud-deployment extensions. Captured
  2026-06-03 after an advisor meeting.
- [`open_work_compression.md`](open_work_compression.md) — the ranked backlog:
  open directions mined from the future-work / limitations sections of 95 cited
  papers and cross-referenced against what was already done or ruled out, with
  honest novelty labels against Q5/Q6. Tracks are mt-quant, pruning-structural,
  distill-recovery, nllb-encdec, tts-xtts, interp-bridge. Generated 2026-07-07.
  The most reusable document in this folder.

## The project as it was framed

- [`project_summary.md`](project_summary.md) — the thesis spine: prior work
  (IFR-guided layer pruning on Aya Expanse 8B), the model and language-pair
  choices, and the methodological discipline that phase one followed.
- [`phase1_plan.md`](phase1_plan.md) — the original phase-one investigation
  plan, including the V1/V2/V3 claim structure and the per-question
  satisfied-when criteria. Phase one is complete; results are in
  [`compression/docs/`](../../compression/docs/).
- [`ms_project_plan.md`](ms_project_plan.md) — the MS project planning document
  and research roadmap, from before the tracks split.
- [`phase2_hypotheses.md`](phase2_hypotheses.md) — the candidate directions for
  phase two, written while Q5 was still open and none was committed. Phase two
  ran; this is now history.

## Advisor and paperwork

- [`advisor_brief.md`](advisor_brief.md) — the talking document for explaining
  the project, honest about solid vs preliminary.
- [`project_ideas_advisor_brief.md`](project_ideas_advisor_brief.md) — the two
  candidate MS project ideas presented for advisor review (2026-06-19).
- [`proposal_form_cs698r.md`](proposal_form_cs698r.md) — the CS 698R master's
  project approval requirements.

## Process notes

- [`replication_uneven_ptq_brief.md`](replication_uneven_ptq_brief.md) — the
  brief that commissioned the independent replication of arXiv:2508.20893. The
  replication ran; its results are in
  [`compression/docs/replication_uneven_ptq_mt.md`](../../compression/docs/replication_uneven_ptq_mt.md).
- [`claude_code_bootstrap.md`](claude_code_bootstrap.md) — the original project
  bootstrap prompt and working agreement.
- [`session_handoff_2026_06_03.md`](session_handoff_2026_06_03.md) — a
  point-in-time resume note. Superseded by `../registry.md`.
