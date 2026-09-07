# superweights — formation and behavior of super weights

**Status (2026-09-05):** Tier A replication done (`notes.md`); a 2026-09-02 joint-ablation
session is parked *unverified* in `parked_claude_2026-09-02/`; direction being re-scoped
around the collapse mechanism — see `docs/problem_search_2026_09.md` and
`docs/proposal_draft_v4.md`.

A super weight is a single scalar weight whose ablation is catastrophic for an LLM
(Yu et al., arXiv:2411.07191). This track asks **when they form during training,
whether they are shared or language-specific in multilingual models, and what their
formation costs at deployment (quantization)** — connecting the repo's two standing
interests, cross-lingual structure (`interlingua/`) and compression (`compression/`).

| Doc | What it is |
|---|---|
| [`docs/problem_search_2026_09.md`](docs/problem_search_2026_09.md) | **Read first.** 2026-09-05 literature sweep (≈130 new papers) synthesised against the project's results: nine footing-changing facts, ranked candidate problems, and a recommended re-scope |
| [`docs/proposal_draft.md`](docs/proposal_draft.md) | MS project proposal, v1 (Adrian, 2026-08): "Super Weights: Formation and Cross-Lingual Behavior" |
| [`docs/proposal_draft_v4.md`](docs/proposal_draft_v4.md) | **Current proposal (v4, 2026-09-05):** built-in-constant framing; RQ1 uniform table on multilingual/translation models (mandatory), RQ2 mechanism, RQ3 language dependence, RQ4 gated quantization arm in the lab's CT2 int8 regime |
| `docs/proposal_draft_v2.md`, `docs/proposal_draft_v3.md` | Superseded drafts (mechanism-first; problem-first) kept for the record |
| [`docs/reviews_2026_09_05/`](docs/reviews_2026_09_05/) | Six adversarial reviews (three of v2, three of v3) that produced v4 |
| [`docs/three_axis_program.md`](docs/three_axis_program.md) | The original 2026-08 program: Phase 0 → three axes. Partly superseded by the problem search |
| [`docs/reading_list.md`](docs/reading_list.md) | Background reading, tiered, with verification status per paper |
| [`docs/prior_experiments_and_ideas.md`](docs/prior_experiments_and_ideas.md) | Where the earlier q6 super-weight work lives (code, results, defects), which program claims rest on it, and tiered ideas for the fresh start |
| [`docs/literature/papers_index.md`](docs/literature/papers_index.md) | Index of the 384 local PDFs in flat `papers/` (gitignored, same policy as `interlingua/papers/`), organised by topic; the twelve `docs/literature/notes_<topic>.md` reading notes and `docs/literature/phenomenon_crosswalk.md` sit beside it |

Grounding: `docs/registry.md` (q6 super-weight section + ruled-out list),
`interlingua/docs/method_landscape.md` §5, `interlingua/docs/prior_work_map.md` §8.

**Before writing the first code here, read `docs/research_standards.md` §20.3** —
the provenance manifest writer, config-load tests, and invariant tests for the
detector are due at that moment, not later.
