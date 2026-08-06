# Arctos

MS project work (BYU CS) on **compressing multilingual translation and speech
models so they run on cheap hardware** — with an interpretability lens on *why*
compression hurts what it hurts.

The repo holds two research tracks plus their shared writing. Each top-level
folder is self-contained and has its own README explaining what's in it.

```
arctos/
├── compression/         # LLM interpretability → quantization research (largely concluded)
├── speech-translation/  # NLLB-200 + XTTS v2 dubbing pipeline (the forward track)
├── notebooks/           # 8 runnable method tutorials on a tiny CPU model
└── docs/                # findings, reading lists, planning, papers
```

Nothing runs at the top level. `compression/` and `speech-translation/` each
carry their own `pyproject.toml` and build their own `.venv`.

## The two tracks

### `compression/` — how translation works inside an LLM, and what that means for quantization

An interpretability-led study of **how machine translation is carried out inside
open multilingual LLMs**, followed by a compression chapter grounded in what it
found. Eight models, three language pairs, phase one and phase two both complete.

The headline: *translation is depth-staged — understand the source early,
process in a language-neutral/pivot space in the middle, emit the target
language only in the last quarter.* The target language is a late conversion
step, not the medium the model computes in. This generalizes across lineage,
normalization, positional encoding, and the decoder-only ↔ encoder-decoder
divide.

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

`docs/` is the written record for both tracks — per-question findings, the
annotated bibliography, the reading and math plans, and the planning documents.

Most of what's in `docs/` describes directions that were explored and **parked**,
not active work. [`docs/OPEN-WORK.md`](docs/OPEN-WORK.md) is the useful index of
those: a ranked brief of open directions mined from the future-work sections of
95 cited papers, cross-referenced against what has already been done or ruled
out. Start there when picking up a thread.

→ [`docs/README.md`](docs/README.md) for the full index.

## Status

`compression/` is concluded — the phase-one report is written and phase two has
its result. `speech-translation/` is the active direction. Numbers in the
phase-two writeups are directional (small n, generic prompts); a paper-grade run
would need larger n, chat templates, and a human spot-check.
