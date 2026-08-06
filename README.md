# Arctos

MS project work (BYU CS) on **compressing multilingual translation and speech
models so they run on cheap hardware** — with an interpretability lens on *why*
compression hurts what it hurts.

The repo holds three research tracks plus their shared writing. Each top-level
folder is self-contained and has its own README explaining what's in it.

```
arctos/
├── interlingua/         # training dynamics of cross-lingual alignment (next up)
├── compression/         # LLM interpretability → quantization research (concluded)
├── speech-translation/  # NLLB-200 + XTTS v2 dubbing pipeline
├── notebooks/           # 8 runnable method tutorials on a tiny CPU model
└── docs/                # findings, reading lists, planning, papers
```

Nothing runs at the top level. `compression/` and `speech-translation/` each
carry their own `pyproject.toml` and build their own `.venv`.

## The three tracks

### `interlingua/` — when does the shared meaning space form? *(the direction being taken up)*

Where `compression/` asked what the language-neutral space looks like in a
*finished* model, this asks **when and how it forms during training** — and
whether the "emergence" everyone reports is real phase structure or an artifact
of the metric used to measure it.

The plan is Tier 1 of the Matrix Lab program *Does the Interlingua Grok?*:
train small (~36M) models from scratch on EN/FR/TR and track mechanistic and
behavioral measures across ~60 log-spaced checkpoints. **The architecture is not
settled** — the proposal specifies three arms (encoder-only mBERT-like,
encoder-only XLM-R-like, encoder-decoder NLLB-like) and makes the comparison
between two of them a hypothesis under test; the working plan argues for
decoder-only on tooling grounds and flags it as the PI's call. The sharp move is
refusing to report the headline claim as stated — a lag between a smooth
mechanistic curve and a jumpy accuracy curve is *guaranteed by construction*
when the behavioral metric is discontinuous, so the real experiment is whether
the lag survives a continuous metric. No outcome is empty.

**Tier 1 as written is not being built.** Two audits on 2026-08-06 — one of the
prior art, one of the program's premises — concluded that the plan's claimed
novelty does not survive and that two of its measures manufacture or invert
their own results. The verdict is *salvageable, smaller*, with a few-GPU-hour
pilot as the next step rather than the fifteen proposed runs. Both are linked
from the track README; read them before proposing anything here.

Docs only so far; nothing is built yet.

→ [`interlingua/README.md`](interlingua/README.md).

### `compression/` — how translation works inside an LLM, and what that means for quantization

An interpretability-led study of **how machine translation is carried out inside
open multilingual LLMs**, followed by a compression chapter grounded in what it
found. Eight models, three language pairs, phase one and phase two both complete.

The headline: *translation is depth-staged — understand the source early,
process in a language-neutral/pivot space in the middle, emit the target
language only in the last quarter.* The target language is a late conversion
step, not the medium the model computes in. This generalizes across lineage,
normalization, positional encoding, and generation, with Gemma-family the lone
exception. The encoder-decoder half of the claim is weaker than the rest — NLLB
was measured with a different metric and lacks two of the four methods (see
[`compression/README.md`](compression/README.md)).

And the load-bearing negative: **component importance is uncorrelated with
quantization sensitivity** (ρ ≈ 0). Where MT computation concentrates is *not*
where numerical precision matters — so the depth pipeline, however real, is not
a bit-allocation rule.

→ [`compression/README.md`](compression/README.md) for methods, results, and how to run it.

### `speech-translation/` — compressing the dubbing pipeline

The applied track: getting **NLLB-200 (translation) + XTTS v2 (voice cloning)**
to run together on a single cheap GPU at real-time latency, rather than an
A100. Includes `mobile-tts/`, a Swahili TTS fine-tuning experiment.

Two findings from the baseline sweep that shaped everything after:
**CTranslate2 INT8 is the only compression here that is actually faster**
(2.4× vs FP16 at equal quality), while `bitsandbytes` INT8 is **4× slower than
FP16** — smaller does not mean faster. And INT8 on the XTTS GPT core is quality-
neutral for English and Spanish but **triples French CER** (0.061 → 0.157), so
per-language evaluation is not optional.

→ [`speech-translation/README.md`](speech-translation/README.md) for the tables and pipeline.

## Where the ideas live

**A track's write-ups live with the track.** Each of the three has its own
`docs/`: [`compression/docs/`](compression/docs/) (q1/q4/q5/q6, phase two, the
primers, the bibliography), [`speech-translation/docs/`](speech-translation/docs/)
(the NLLB/XTTS surveys), [`interlingua/docs/`](interlingua/docs/) (the Tier 1
plan and the MS proposals, with its bibliography in
[`interlingua/papers/`](interlingua/papers/)).

Top-level [`docs/`](docs/) holds only what spans tracks or outlives them:

| | |
|---|---|
| [`docs/registry.md`](docs/registry.md) | **What has been done and what is ruled out, with the numbers.** Read before proposing anything. |
| [`docs/research_standards.md`](docs/research_standards.md) | How experiments are run here, sourced. |
| [`docs/learning/`](docs/learning/) | Explanations of the *subject* — quantization from scratch, the architectures, every paper worth reading, the math and skills plans. |
| [`docs/decisions/`](docs/decisions/) + [`docs/templates/`](docs/templates/) | Decision records, and the shapes a new experiment README / notes entry / decision record take. |
| [`docs/archive/`](docs/archive/) | Not being pursued. Pre-split history, **and the compression program itself** — its roadmap and its ranked backlog of open directions. |

**What's next is `interlingua/`, and its plan lives with the track.** There is no
top-level "plans" folder: the compression roadmap and the 95-paper backlog were
moved to `docs/archive/` because that direction is being left behind. Nothing in
them was refuted — they are where compression would resume from if it ever does.

→ [`docs/README.md`](docs/README.md) for the full index.

## Status

`compression/` is concluded — the phase-one report is written and phase two has
its result. `speech-translation/` has its baseline sweep done and is the applied
thread. `interlingua/` is the direction being taken up next and has no code yet.

Numbers in the phase-two writeups are directional (small n, generic prompts); a
paper-grade run would need larger n, chat templates, and a human spot-check.
