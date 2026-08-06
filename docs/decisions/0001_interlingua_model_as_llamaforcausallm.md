# 0001 — Define the interlingua model as an HF `LlamaForCausalLM`

- **Status:** accepted
- **Date:** 2026-08-06

## Context

The interlingua track trains small models from scratch and computes
interpretability analyses per checkpoint (~60 checkpoints × ~15 runs). Two
things have to be true of the model definition:

- The existing method code in `compression/src/` — roughly 900 lines of hooked
  forward passes, logit lens, patching, similarity measures — should transfer
  without a rewrite.
- The analyses must run on a **randomly initialised** model, because step 0 is
  the free null the rigor floor requires (`CLAUDE.md`, "every measure must beat
  its random-init value").

`transformer-lens`'s successor package has two open bugs that silently corrupt
from-scratch training and checkpoint reloading, so the native
`HookedTransformer` training path is not a safe base
(`docs/research_standards.md`, interlingua section).

## Decision

We will define the trained model as a standard HuggingFace `LlamaForCausalLM`,
and attach interpretability tooling to it via the existing `HookedModel`
wrapper rather than training a `HookedTransformer` directly. Where
`transformer-lens` is needed for analysis, we pin `transformer-lens==3.6.0` and
use the deprecated `HookedTransformer` path.

## Alternatives considered

- **Train a native `HookedTransformer`** — rejected: the successor package's two
  open bugs affect exactly from-scratch training and checkpoint reloading, which
  is the entire workload. Silent corruption is the worst failure mode available
  here, because it produces plausible checkpoints.
- **Write a bespoke minimal transformer** — rejected: forfeits the ~900 lines of
  method code, forfeits HF checkpoint/tokenizer tooling, and gains only
  transparency we do not need. It also breaks comparability with Pico's layer
  naming, which brings free CKA/PWCCA/effective-rank tooling and two public
  baselines bracketing 36M.
- **Use Pico-train directly as the trainer** — not rejected outright; still open.
  Its layer naming is the reason to match it. But it calls `wandb.Api()` at init
  and `evaluate.load(..., trust_remote_code=True)` at every eval, and both
  hard-fail on compute nodes with no internet. Adopting it means patching those
  two call sites first.

## Consequences

- Existing method code transfers unchanged; `HookedModel` provably works on a
  randomly initialised `LlamaForCausalLM`, so the step-0 null is available from
  day one.
- Pinning `transformer-lens==3.6.0` means living on a deprecated path. This will
  need revisiting when the successor's bugs are fixed; the pin should carry a
  comment pointing at this record.
- HF's `LlamaForCausalLM` is not hook-native, so any analysis needing internal
  activations goes through the wrapper. **`run_with_cache` raises on batch size
  > 1** — that has to be fixed before running 60 checkpoints × 15 runs, and it is
  a direct cost of this choice.
- Checkpoint format is HF-standard, which makes the artifacts reusable and
  publishable but larger than a minimal state dict.
