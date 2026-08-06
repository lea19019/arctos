# compression/docs — the written record of this track

Everything written about *this* track: the results, the primers that explain the
methods, and the bibliography behind them. Cross-track material — the registry,
the research standards, the subject-matter foundations — lives in
[`../../docs/`](../../docs/).

**Start with [`reading_guide.md`](reading_guide.md)** if you want the whole track
in order, phase one → phase two, with a one-line reason each item matters.

**Read every result here against [`../../docs/registry.md`](../../docs/registry.md).**
An audit (2026-08-05) recomputed the headline numbers from the raw `.npz`/`.json`
and found the measurements reproduce exactly — but several *claims* written about
them do not, and the registry lists which.

## Phase one — how translation works inside an LLM

| File | Question | Coverage |
|---|---|---|
| [`q1_language_emergence.md`](q1_language_emergence.md) | When and where does the target language emerge? | 9 models |
| [`q4_architecture_comparison.md`](q4_architecture_comparison.md) | Does the depth signature generalize across architectures? | synthesis |
| [`q5_importance_vs_sensitivity.md`](q5_importance_vs_sensitivity.md) | **The pivotal negative** — is component importance correlated with quantization sensitivity? (ρ ≈ 0) | 6 models |

There is no `q2` or `q3` writeup. Q2 (MT-critical attention heads) ran only its
attention-pattern visualization; Q3 (MLPs and layers) never ran at all — both
runners are stubs. See the registry.

The full phase-one paper is [`../report/arctos-translation-report.pdf`](../report/arctos-translation-report.pdf).

## Phase two — compression grounded in phase one

Read in this order:

1. [`compression_primer.md`](compression_primer.md) — the
   **find / keep / shrink / prune** framework the whole phase was built on, plus
   its reading list (super weights, AWQ, Hessian methods, outliers). This is the
   explanation, not a result.
2. [`phase2_synthesis.md`](phase2_synthesis.md) — **start here for
   conclusions.** The honest consolidated read: what holds, what doesn't.
3. [`phase2_results.md`](phase2_results.md) — the cross-model tables
   (chrF++ / XCOMET-XL), gap-to-baseline.
4. [`phase2_method_primer.md`](phase2_method_primer.md) — the proposed
   MT-specific method and the literature-gap/novelty map behind it.
5. [`q6_compression.md`](q6_compression.md) — the find/keep/shrink/prune sweep
   writeup.
6. [`phase2_novel_direction.md`](phase2_novel_direction.md) — the pipeline-aware
   "protect the endpoints, crush the middle" idea. **It reads as a live
   proposal; it became a negative result.**

## Replication

[`replication_uneven_ptq_mt.md`](replication_uneven_ptq_mt.md) — independent
replication of arXiv:2508.20893 (*The Uneven Impact of PTQ in Machine
Translation*), COMET on WMT24++, n≈960/direction, 4 quantizers. The paper is a
non-peer-reviewed preprint and the writeup notes its red flags. The brief that
commissioned it is in
[`../../docs/archive/replication_uneven_ptq_brief.md`](../../docs/archive/replication_uneven_ptq_brief.md).

## Reference material

- [`annotated_bibliography.md`](annotated_bibliography.md) — the long-form
  survey: the four-stage compression pipeline (score → prune → recover →
  quantize) read against ~95 papers, plus three deep-research addenda. A
  reference to search, not to read straight through.
- [`deep_research_raw/`](deep_research_raw/) — unedited deep-research agent
  transcripts behind the synthesis documents.
- [`figures/`](figures/) — figures referenced by the writeups above.

## Where things go

Writeups land here when a question's satisfied-when criterion is met. Working
notes *while* a question is open live in `../experiments/<name>/notes.md`, not
here.
