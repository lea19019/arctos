# Q6 notes / changelog

Running log of design decisions and fixes for the find/keep/shrink/prune
sandbox. Conceptual background + reading list:
[`compression/docs/compression_primer.md`](../../docs/compression_primer.md).

## 2026-06-02 — super-weight ranking fix (causal, not spike)

**Symptom.** First A100 run: the detected "top super weight" was correct on
Llama-3.1 (layer 1, ablation KL 0.24 — the genuine Apple effect) but wrong on
Aya (layer 31, activation spike 732 but ablation KL 4e-7 — causally inert).

**Cause.** `detect_super_weights` ranked candidates by the magnitude of the
`down_proj` output activation spike. The final layer's `down_proj` writes
straight into the residual that becomes the logits, so it always has a large
output magnitude — a false positive. The Apple paper's super weight is defined
by the super activation *propagating* (causal), not by raw spike size.

**Fix.** `experiment.py` now detects one spike candidate per layer
(`top_k=n_layers`), then **re-ranks by causal ablation KL** (zero the scalar,
measure next-token KL) and uses that order for `super_coords`, the KEEP
super-weight-preservation variant, and the PRUNE stress test. All candidates +
their ablation KL/top1-drop are logged in `q6_summary.json`
(`find_super_weights.candidates`).

**Cost.** Adds `n_layers` verification passes (~32 forwards × a few prompts) to
the find stage — ~2 min on an A100; negligible vs the 6 h job. (On CPU this is
what made the bloom-560m find-only smoke exceed a 10-min timeout — not a bug.)

**Validated.** CPU smoke (bloom-560m) prints the causal-ranked top weight;
unit tests in `tests/interp/test_compress.py` cover detect + verify.

## Scope

- NLLB (encoder-decoder) excluded — the quantizer targets decoder blocks.
- Quant/prune act on every `nn.Linear` in the decoder blocks (q/k/v/o +
  gate/up/down or per-arch equivalents); embedding + lm_head are left alone.
- Gemma-family is the known Q4 outlier; watch its shrink curve separately.

## 2026-06-02 — phase-two GEM run (deep-research-driven)

Deep-research (run wf_36650cc2-1b3) + primer #2
(`compression/docs/phase2_method_primer.md`) identified the verified novelty:
**multilingual / MT-conditional super-weight + salient-channel FP16 preservation
at the low-bit (W2/W3) cliff, causal-KL ranked, XCOMET-XL eval** — two
intersecting documented gaps (no multilingual super-weight study; no MT-quality
salient-FP16 low-bit recovery). Secondary: MT-conditional GPTQ at 2-3 bit
(unoccupied; 2508.20893 skipped GPTQ, Chimoto 2601.18306 = perplexity only).

New code: GPTQ (`compress.gptq_*`), Fisher mixed-precision (`bits_by_fisher`,
`quantize_mixed_precision`), COMET/XCOMET-XL (`eval/metrics.comet_score`,
default `Unbabel/XCOMET-XL`), runner stages `gptq` + `alloc` + COMET-aware
`_eval_q`. Validated by GPU canary (bloom-7b1).

Run: `bash experiments/q6-compression/slurm/submit_gem.sh` → `results/{model}/
q6gem/`; collect with `python scripts/q6gem_collect.py`.

Canary signals (n=4, directional): GPTQ-MT − generic helps **en-arz** (chrF++
+4.5 W4, +6.6 W3); Fisher mixed-precision (2/4 split) **underperformed** uniform
3-bit → allocator needs rework (kept exploratory, not headline). XCOMET-XL
working offline (needs `facebook/xlm-roberta-xl` encoder cached too).

CAVEAT (from deep-research): automatic MT metrics understate low-bit damage ~10x
vs human eval (arXiv:2407.03211) — add a stress/human check before strong 2-bit
claims.
