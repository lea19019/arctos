# superweights — formation and behavior of super weights

**Status: docs only, no code.** Drafted 2026-08-06. Not yet a committed direction —
the next step is a `/new-experiment` spec for Phase 0 (see the program doc).

A super weight is a single scalar weight whose ablation is catastrophic for an LLM
(Yu et al., arXiv:2411.07191). This track asks **when they form during training,
whether they are shared or language-specific in multilingual models, and what their
formation costs at deployment (quantization)** — connecting the repo's two standing
interests, cross-lingual structure (`interlingua/`) and compression (`compression/`).

| Doc | What it is |
|---|---|
| [`docs/three_axis_program.md`](docs/three_axis_program.md) | The program: Phase 0 (re-verify q6 with a calibrated detector) → three axes, with gating logic and floor deliverables |
| [`docs/reading_list.md`](docs/reading_list.md) | Background reading, tiered, with verification status per paper |
| [`papers/README.md`](papers/README.md) | Index of local PDFs (PDFs themselves are gitignored, same policy as `interlingua/papers/`) |

Grounding: `docs/registry.md` (q6 super-weight section + ruled-out list),
`interlingua/docs/method_landscape.md` §5, `interlingua/docs/prior_work_map.md` §8.

**Before writing the first code here, read `docs/research_standards.md` §20.3** —
the provenance manifest writer, config-load tests, and invariant tests for the
detector are due at that moment, not later.
