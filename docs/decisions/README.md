# Decision records

One file per decision, Nygard format, sequential number, never reused:
`NNNN_short_slug.md`.

Write one when a choice **constrains future work**, is **expensive to reverse**,
or was **non-obvious enough that you will re-litigate it in three months**. Not
for choices with an obvious default — a record per trivial choice makes this
directory unreadable and the practice dies.

Two rules make these worth keeping:

1. **Immutable.** A superseded record gets `Status: superseded by 0007`, never an
   edit. The reasoning that was true at the time stays readable — that is the
   entire value.
2. **Under a page.** Longer records stop getting written.

Use `/record-decision`. Template: [`../templates/decision_record.md`](../templates/decision_record.md).

Rationale and sourcing: [`../research_standards.md`](../research_standards.md) §20.6.

## Index

| # | Decision | Status | Date |
| --- | --- | --- | --- |
| [0001](0001_interlingua_model_implementation_substrate.md) | Model implementation substrate for interlingua from-scratch training | **proposed** — blocked on PI decision | 2026-08-06 |

A record stays `proposed` until the person who owns the decision has actually
made it. Marking a record `accepted` because it sounds settled is how a guess
acquires authority it never earned.
