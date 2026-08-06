# <Track> — <experiment name>

> <The question, in one sentence, ending in a question mark.>

<One paragraph: why this question is worth answering now, and what it unblocks.
If it is adjacent to something in docs/registry.md under "Ruled out", say what
is different about it.>

## Hypothesis

**Our hypothesis is that** <...>. <Why — the reasoning or prior result it rests
on.> ("Our hypothesis", never "the mechanism.")

## Satisfied when

<The falsifiable criterion, with the number in it. A deliverable is not a
criterion.>

- ✗ "A per-component importance × sensitivity map exists for Aya."
- ✓ "Spearman ρ between component importance and quantization sensitivity has a
  95% CI excluding 0, with n_seeds ≥ 5 and family size stated."

## Null / baseline

<What this is measured against, named explicitly. If the baseline has not been
run, it is not a baseline — schedule it here.>

- Random-init / step-0 null: <how it is computed>
- Comparison baseline: <what, and where its numbers will live>

Every measure must beat its random-init value. Probes, saliency maps and SAEs
all look interpretable on randomly initialised networks.

## Analysis plan

<Written before any data is seen.>

- **Statistic:** <e.g. Spearman ρ, paired bootstrap over seeds>
- **Seeds:** n = <N>. Vary weight init **and** data order/shuffling — randomising
  init alone caps effective n at ≈2.
- **Family size for multiplicity:** M = <N layers × N checkpoints × ...>;
  correction = <Holm / BH / …>
- **"No effect" means:** the 95% CI falls inside [<lo>, <hi>]. Never a bare
  negative.

## What would change my mind

<The concrete result that kills the hypothesis.>

## Out of scope

<What this run deliberately does not answer, so the writeup does not drift into
claiming it.>

## Runs

| Config | Model | Seed | Results dir | Manifest |
| --- | --- | --- | --- | --- |
| `configs/<x>.yaml` | | | `results/<...>` | ✓ |

One YAML per run. Seed declared in every one. No hidden defaults in code that
override config.

## Verdict

<Filled in by /close-experiment when the run completes. One of: **met** /
**not met** / **undecided** — with the number, the results path, and the git sha.
"Undecided" must say what would decide it.>
