---
name: record-decision
description: Write an architecture/method decision record in docs/decisions/. Use when a choice constrains future work, is expensive to reverse, or was non-obvious enough to be re-litigated later.
---

# Record a decision

Decisions made before code exists are the ones whose reasoning is lost fastest,
because nothing in the repository holds it. The interlingua choices being made
now — `LlamaForCausalLM` over a native `HookedTransformer`,
`transformer-lens==3.6.0`, WSD over cosine, the checkpoint grid — each have a
real rejected alternative and a real reason.

## When to write one

Write a record when the choice **constrains future work**, is **expensive to
reverse**, or was **non-obvious enough that you will re-litigate it in three
months**. Not for choices with an obvious default — a record per trivial choice
makes the directory unreadable and the practice dies.

## Format — Nygard, one page maximum

`docs/decisions/NNNN_short_slug.md`, sequential number, never reused:

```markdown
# NNNN — <title as a decision, not a topic>

- **Status:** accepted | superseded by [NNNN](NNNN_x.md) | deprecated
- **Date:** YYYY-MM-DD

## Context
The forces at play: constraints, what we know, what we do not. Written so it
still makes sense when the constraint has changed. State facts, not the
conclusion.

## Decision
"We will ..." — active voice, one paragraph.

## Alternatives considered
Each with the reason it was rejected. **This is the section you will come back
for.** A rejected option with no reason recorded gets re-proposed.

## Consequences
What becomes easier, what becomes harder, and what this forecloses. Include the
bad consequences — a record listing only benefits is advocacy, not a record.
```

## Two rules that make them worth keeping

1. **Immutable.** A superseded decision gets `Status: superseded by 0007`, never
   an edit. The reasoning that was true at the time stays readable — that is the
   entire value. Editing history is how you lose the ability to ask "why did we
   think that?"
2. **Under a page.** Longer records stop getting written.

## Related

- A decision that turned out wrong is also a **negative** — file it per
  `/close-experiment` and link the record from `docs/registry.md`.
- Method decisions with sourced backing belong in
  `docs/research_standards.md`; the record here should link to the section
  rather than restate it.
