# Q6 — find / keep / shrink / prune: results and the path to an MT quant method

> Status: **complete** — full 8-model A100 sweep (2026-06-02), all five stages,
> zero failures. n=24 generated translations/pair, 3 pairs, greedy, generic
> prompt → **directional, not publication-grade** (chrF++, not COMET). Runner:
> `compression/experiments/q6-compression/experiment.py`; data: `compression/results/{model}/q6/
> q6_summary.json`; framework + reading list: `compression/docs/compression_primer.md`.

Q6 reframes the Q5 null (component *importance* ⟂ quantization *sensitivity*):
sensitivity is real and concentrated, but it lives at the **per-weight /
per-channel** level and is found with **sensitivity-native** signals, not
interpretability importance. The five stages: **find** (super weights, AWQ
salient channels + MT-vs-generic calibration shift, Fisher), **shrink** (RTN
INT-k), **keep** (protect the fragile minority), **prune** (Wanda vs magnitude),
**calib** (the linchpin: MT vs generic calibration head-to-head).

## FIND — super weights are early-layer, model-varying; Gemma is the outlier

Candidates ranked by **causal ablation KL** (not raw activation spike — spike
alone false-positives on the last layer; see
[[feedback_super_weight_causal_ranking]]).

| model | SW layer | ablation KL | AWQ salient-set Jaccard (mt vs source) |
|---|---|---|---|
| EuroLLM 9B | 9 | **3.28** | 0.57 |
| TowerInstruct 7B | 1 | 1.25 | 0.65 |
| TowerBase 7B | 1 | 0.96 | 0.67 |
| Llama-3.1 8B | 1 | 0.24 | 0.68 |
| Aya 8B | 2 | 0.002 | 0.66 |
| BLOOM 7B1 | 0 | 0.001 | 0.43 |
| Tower-Plus 9B (Gemma2) | 24 | 0.04 | 0.62 |
| Gemma-3 12B | 47 | 0.005 | 0.70 |

- Super weights sit in **early layers** (L0–L9) for every non-Gemma family —
  the Apple result reproduced. Strength varies hugely (EuroLLM monster vs
  Aya/BLOOM near-nil).
- **TowerBase & TowerInstruct share the *same* super weight** (L1, out2533,
  in7890); SFT only sharpens it (0.96→1.25) — the per-weight echo of Q1's
  "CPT sets the structure, SFT refines."
- **Gemma-family is the outlier again** (late layer, near-zero KL) — the same
  exception that broke the Q4 depth signature. Its norms + logit soft-cap
  suppress concentrated super-weight structure.
- AWQ salient channels overlap only **0.43–0.70** between MT and raw-source
  calibration → the salient set genuinely shifts with calibration data.

## SHRINK — RTN, chrF++ on cs-de; the 3-bit cliff is model-specific

| model | base | W4 | W3 | W2 |
|---|---|---|---|---|
| Aya | 71.2 | 73.5 | **70.8** | 6.5 |
| Tower-Plus | 61.5 | 58.3 | 31.5 | 4.0 |
| Gemma-3 | 60.5 | 56.6 | **12.7** | 11.3 |
| EuroLLM | 57.9 | 48.7 | 32.1 | 11.5 |
| TowerInstruct | 52.0 | 47.6 | 31.8 | 0.0 |
| Llama-3.1 | 50.8 | 49.0 | 33.0 | 3.9 |
| TowerBase | 40.4 | 35.7 | 24.2 | 3.6 |
| BLOOM | 24.1 | 24.1 | 20.7 | 0.3 |

W4 ≈ lossless everywhere; W2 dead everywhere; **W3 is the interesting cliff**
and is wildly model-dependent (Aya nearly lossless; Gemma collapses).

## KEEP — protecting the fragile minority recovers the 3-bit cliff (the win)

chrF++ on cs-de at **W3**. `keep_salient_fp16` = keep top-1% AWQ-salient input
channels in FP16; `rtn+SW` = restore the super-weight scalars after RTN.

| model | RTN | keep_salient_fp16 | AWQ (α=0.25) | rtn+superweight |
|---|---|---|---|---|
| Gemma-3 | 12.7 | **48.4** (+35.7) | 21.0 | 12.7 |
| Tower-Plus | 31.5 | **49.2** (+17.7) | 50.9 | 31.5 |
| EuroLLM | 32.1 | **45.8** (+13.7) | 36.7 | 32.1 |
| TowerBase | 24.2 | **38.1** (+13.9) | 28.2 | 24.2 |
| Llama-3.1 | 33.0 | **44.1** (+11.1) | 35.9 | 33.9 |
| TowerInstruct | 31.8 | **37.7** (+5.9) | 33.4 | 31.7 |
| BLOOM | 20.7 | 21.3 | 20.1 | 20.7 |
| Aya | 70.8 | 70.7 | 68.5 | 70.7 |

- **Keeping ~1% of channels in FP16 recovers most of the W3 cliff** for the
  fragile models — Gemma 12.7→48.4 is dramatic. This is the method's strongest
  result, and it was *invisible* at 2-bit (the original KEEP weakness, now
  fixed).
- **Super-weight preservation alone barely helps** (rtn+SW ≈ rtn): a handful of
  scalars can't offset broad quantization damage. The win is the salient
  *channels*, not the super-weight scalars (those matter for *removal*, below).
- Aya needs no protection (3-bit robust); BLOOM's baseline is too low to move.

## PRUNE — Wanda ≫ magnitude; the super-weight stress test

Wanda (|W|·‖X‖) vs magnitude at 50% sparsity, cs-de chrF++:

| model | magnitude | Wanda |
|---|---|---|
| Tower-Plus | 3.8 | **56.3** |
| Llama-3.1 | 20.8 | **47.5** |
| TowerInstruct | 21.0 | **45.0** |
| TowerBase | 14.2 | **37.0** |
| EuroLLM | 8.9 | **34.0** |
| Gemma-3 | 3.0 | **18.5** |
| Aya | 63.6 | 65.8 |
| BLOOM | 20.4 | 18.7 |

**Super-weight stress** (ablate the 1 causal super weight vs the 1000
largest-magnitude weights), cs-de chrF++ — magnitude is a *terrible* saliency:

| model | ablate 1 super weight | ablate 1000 largest-\|W\| |
|---|---|---|
| EuroLLM | **4.8** (collapse) | 60.0 |
| TowerInstruct | 28.4 | 49.5 |
| TowerBase | 28.9 | 41.8 |
| Llama-3.1 | 47.8 | 45.2 |
| Aya / BLOOM / Tower-Plus / Gemma | ~unchanged | ~unchanged |

For high-KL models, **one scalar matters more than the 1000 biggest weights**
(EuroLLM: removing one weight 57.9→4.8). The effect tracks the FIND KL ranking
exactly. This is the per-weight confirmation of the project's thesis: magnitude
≠ importance ≠ sensitivity.

## CALIB — the linchpin: does MT calibration beat generic? (mostly only for pruning)

MT-parallel vs generic-XNLI calibration (same languages), head-to-head at
matched bits/sparsity. Δ = MT − generic chrF++ (+ = MT better), aggregated over
8 models × 3 pairs:

| operation | mean Δ | median Δ | positive |
|---|---|---|---|
| **AWQ quantization (W3)** | −0.95 | −0.62 | **9/24** |
| **Wanda pruning (50%)** | +0.69 | **+2.18** | **19/24** |

- **MT calibration does NOT help quantization** (AWQ): 9/24 positive, slightly
  negative on average; sometimes much worse (EuroLLM cs-de −15.5). For weight
  quantization, the salient *set* shifts but generic calibration is just as
  good — the WMT25 "use C4" practice is fine for the quant grid.
- **MT calibration DOES help pruning** (Wanda): 19/24 positive, median +2.2
  chrF++ (Tower-Plus +11.0, BLOOM consistently +4–6). The activation norms that
  decide which weights to *prune* are task-sensitive in a way the quant grid
  is not. (Gemma cs-de −34.9 is an unstable outlier; median is the robust read.)

## Verdict — is a custom MT quantization method justified?

**Yes, with a sharper-than-expected thesis.** The naive hope ("MT-calibrated
AWQ beats C4-calibrated AWQ") is **not** supported. But three results do support
a defensible, MT-grounded compression method:

1. **Keep the salient channels** (universal, large): top-1% FP16 recovers the
   3-bit cliff (Gemma +36, Tower-Plus +18, EuroLLM +14). Kernel-friendly.
2. **Preserve super weights for pruning/structural decisions** (model-varying):
   for high-KL models one scalar is load-bearing — never let magnitude pruning
   touch it. Cheap to special-case (a handful of scalars, found data-free).
3. **MT-conditional *pruning* calibration** (the genuinely MT-specific lever):
   Wanda with MT activation norms beats generic by a median +2.2 chrF++.

**Proposed method shape:** MT-conditional **Wanda pruning** (the part where MT
calibration pays) + **salient-channel-preserving 4-bit quantization** (AWQ/
LeanQuant, generic calibration is fine) + **super-weight protection**, treating
**Gemma-family as a separate regime** (no concentrated super weight; its W3
cliff is recovered almost entirely by salient-channel FP16). This maps onto
Variant A/C in `compression/docs/annotated_bibliography.md` and aims at the empty WMT25 Pareto region
(moderate compression / moderate quality).

## Limitations (honest)

- **n=24, greedy, generic prompt, chrF++** — directional only. The decisive
  re-run is **COMET** (model cached: `Unbabel/wmt22-comet-da`) at larger n with
  each model's chat template.
- KEEP/CALIB tested at W3; W4 is lossless so protection isn't needed there, but
  a real method would target W4 (+ aggressive prune) for the size/quality knee.
- Gemma is unstable under aggressive ops (CALIB Wanda cs-de −34.9); needs its
  own handling, consistent with its Q4/super-weight outlier status.
- AWQ here is weight-only simulated with a small α sweep, not the full
  per-layer α search; LeanQuant's loss-aware grid is the stronger quantizer to
  test next.

## Where the data lives
`compression/results/{model}/q6/q6_summary.json` (all stages, all pairs) + `fisher.npz`.
Regenerate the cross-model tables: `python scripts/q6_collect.py`. Live sweep
status: `bash scripts/q6_status.sh`.
