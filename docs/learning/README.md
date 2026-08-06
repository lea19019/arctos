# learning — explanations, not results

The teaching material. Everything here exists to make something
*understandable*, and it's about the **subject**, not about any one track's
results. Track write-ups live with their track:
[`../../compression/docs/`](../../compression/docs/),
[`../../speech-translation/docs/`](../../speech-translation/docs/),
[`../../interlingua/docs/`](../../interlingua/docs/).

## Read in this order if you're new

1. [`project_foundations.md`](project_foundations.md) — **start here.** What
   quantization actually is, the NLLB and XTTS architectures, the compression
   landscape, and where interpretability fits. Assumes nothing.
2. [`llm_quant_foundations.md`](llm_quant_foundations.md) — the next layer,
   covering all three tracks: decoder-only LLM architecture and compression,
   NLLB's cross-attention problem, XTTS internals, and how a quantized model
   actually executes on the GPU. Assumes (1).
3. [`reading_list.md`](reading_list.md) — every paper cited, grouped by theme,
   with a one-line "why", plus videos and sites for the foundations. Has an "if
   you only read five" section — a good place to go after (2).

From there, the track primers:
[`compression_primer.md`](../../compression/docs/compression_primer.md) (the
find/keep/shrink/prune framework) and
[`phase2_method_primer.md`](../../compression/docs/phase2_method_primer.md).

## Personal

- [`math_plan.md`](math_plan.md) — the six-month, just-in-time math curriculum:
  getting to the point of re-deriving the core results in this area.
- [`skills_plan.md`](skills_plan.md) — the career track through spring 2027
  (interview cadence, production toolkit, application calendar). Independent of
  which research direction is live, which is why it survived the move away from
  the compression program. Overlaps `math_plan.md`; worth reconciling.
- [`learning_log.md`](learning_log.md) — dated running notes: dead ends,
  surprises, bookmarks. **Currently empty** — `CLAUDE.md` asks for dead ends to
  be recorded here and none have been.
- [`systems_notes.md`](systems_notes.md) — a list of hardware/systems notes that
  were planned (transformer math from scratch, GPU memory, KV cache, kernels,
  model storage formats, AWS deployment). **None were written.** Kept as a
  standing to-do, not as content.
