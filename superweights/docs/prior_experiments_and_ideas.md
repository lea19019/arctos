# Prior super-weight experiments in this repo, and ideas for the fresh start

**Drafted 2026-08-08. Status: reference + idea list, nothing committed.** This
track is a fresh start: the earlier super-weight work was run in `compression/`
(the q6 experiment) and is treated here as **leads to re-verify, not results to
build on**. Everything below is pointers — deciding which ideas to actually run
is a PI/owner decision, and any idea that gets picked goes through
`/new-experiment` first.

---

## 1. Where the prior work lives

All of it is in the `compression/` track, from the 2026-06 q6 sweep:

| Artifact | Path | What it is |
|---|---|---|
| Detector + ablation verifier | `compression/src/interp/super_weights.py` | `detect_super_weights` (single forward pass, per-layer `down_proj` spike → candidate coordinate) and `verify_super_weight` (zero one scalar, measure KL / top-1 drop) |
| Runner | `compression/experiments/q6-compression/experiment.py` | q6's *find* stage (detection), *keep* stage (`rtn+SW` preservation arm), *prune* stage (1-super-weight vs 1000-largest-magnitude stress test) |
| Raw results | `compression/results/{model}/q6/q6_summary.json` | 8 decoder models, all stages |
| Collection scripts | `compression/scripts/q6_collect.py`, `q6gem_collect.py` | Regenerate the cross-model tables |
| Writeup | `compression/docs/q6_compression.md` | The findings doc (find/shrink/keep/prune/calib) |
| Canonical summary | `docs/registry.md` § "Super weights — priority section" | The audited version — read this one first |
| Tests | `compression/tests/interp/test_compress.py` | Smoke only; the detector has never been validated against a planted weight |

Two known defects of the prior detector, both recorded in `docs/registry.md`:
it is **greedy** (per-layer `argmax`, always returns a candidate, no null, no
threshold), and ranking by raw spike produced a **false positive on Aya L31**
(spike 732, ablation KL 4×10⁻⁷ — the last layer's `down_proj` writes straight
into the logits). Causal-KL ranking was the fix. Small erratum while we're
here: the module docstring cites "Yu, Bai, Jaiswal et al." — the actual authors
are Mengxia Yu, De Wang, Qi Shan, Colorado J Reed, Alvin Wan (already flagged
in `interlingua/docs/prior_work_map.md` §8).

## 2. What those experiments found (leads, not results)

Scale caveat on all of it: **n=24–32 translations, greedy decoding, chrF++,
generic prompt** — below this repo's own rigor floor ("directional, not
publication-grade"). Per-model detection, ranked by causal ablation KL:

| Model | SW layer | Ablation KL |
|---|---|---|
| EuroLLM-9B | 9 | 3.284 |
| TowerInstruct-7B | 1 | 1.252 |
| TowerBase-7B | 1 | 0.957 |
| Llama-3.1-8B | 1 | 0.243 |
| Tower-Plus-9B | 24 | 0.038 |
| Gemma-3-12B | 47 | 0.005 |
| Aya-Expanse-8B | 2 | 0.002 |
| BLOOM-7B1 | 0 | 0.001 |

Headline leads (full detail in `docs/registry.md`):

- Early layers (L0–L9) for every non-Gemma family; Gemma late and near-inert;
  strength spans 3.5 orders of magnitude.
- Ablating EuroLLM's one super weight: 57.9 → 4.8 chrF++. Ablating its 1000
  largest-magnitude weights: 60.0 (nothing).
- TowerBase and TowerInstruct share the same coordinate (L1, out 2533, in 7890,
  value 1.5390625) — **a replication of Yu et al.**, not a finding. The novel
  part is that SFT *sharpens* it (KL 0.96 → 1.25), and that is **n=1, no CI**.
- EuroLLM has a **second** load-bearing scalar — L1, out 750, in 9606, KL
  2.975; rank 3 drops to 0.0016. On disk, never written up anywhere.
- Negative: FP16-preserving super weights during quantization is a no-op
  (`rtn+SW ≈ rtn`, all 8 models).

## 3. ⚠️ Which claims in this track rest on those experiments

The three-axis program (`three_axis_program.md`) leans on q6 in three places,
which is exactly why Phase 0 re-verifies before anything else:

1. **The §2 leads table** is q6 verbatim — every coordinate, every KL, the
   sharpening datapoint, the second EuroLLM scalar.
2. **Constraint 1** ("preservation is a no-op for quantization") is q6's keep
   stage; **constraint 3** ("super weight ≠ salient channel") is q6's keep
   stage read against its salient-channel arm. (Constraint 2, importance ⟂
   sensitivity, is Q5, not q6.)
3. **Axis 3 presupposes re-verified coordinates** for EuroLLM, Aya, BLOOM, and
   the Towers.

If a fresh-start replication contradicts any q6 number, the program doc's §2
table and the registry both need back-annotation — that would be a finding,
not an embarrassment. Until Phase 0 runs, none of §2 should be cited outward.

## 4. Ideas

Tiered by commitment level. Tier A is scratch work (uncommitted, no
`/new-experiment` needed — it's learning, and its ground truth is published).
Tiers B–C are track work: spec first, and the first committed code fires the
`docs/research_standards.md` §20.3 triggers (provenance manifest, config-load
tests, detector invariant tests).

### Tier A — replication from scratch (own code, own understanding)

1. **Re-implement Yu et al.'s detection recipe from the paper alone** — no
   peeking at `super_weights.py`. One forward pass, max |input| and |output| of
   each `mlp.down_proj`, read the coordinate off the spikes. Check against the
   paper's Table 2 answer key (e.g. Llama-7B `layers[2].mlp.down_proj.weight
   [3968, 7003]`; OLMo-1B if hardware-constrained).
2. **The ablation moment.** Zero that one scalar, generate text, watch a 7B
   model produce gibberish; measure perplexity and stopword-probability shift
   (the paper's Figure 5 mechanism).
3. **Reproduce the Aya L31 false positive.** Rank candidates by raw spike and
   by ablation KL on one q6 model; see them disagree. This is the causal-vs-
   spike lesson from q6, re-derived independently.
4. **Break the detector on purpose.** Run it on a randomly initialized model
   and on Gemma-3: it will confidently return a "super weight" in both. This
   false positive, felt firsthand, is the entire motivation for Phase 0's
   calibrated null.
5. **Iterative detection.** Yu et al. remove the found super weight and repeat
   until spikes are suppressed. Does the recipe find EuroLLM's second scalar
   (L1, out 750, in 9606)? Its existence on disk is an in-repo prediction the
   fresh code can test blind.

### Tier B — re-verification of q6 leads (spec first; feeds Phase 0)

6. **Re-detect on the 8 q6 models with the fresh detector** and compare
   coordinates against `q6_summary.json`. Agreement table goes in the Phase 0
   spec as the "re-verified (or corrected) q6" deliverable.
7. **Re-measure ablation KL at proper n** — the q6 numbers are n=24–32; the
   replication harness in `compression/experiments/replication-uneven-ptq/`
   (n≈960, chat templates, COMET) is the protocol to reuse.
8. **The sharpening claim needs company.** TowerBase→TowerInstruct KL 0.96→1.25
   is one base/instruct pair. Other public pairs (Llama base/instruct,
   OLMo base/instruct, Mistral base/instruct) turn n=1 into a distribution —
   cheap, and it either generalizes or kills the repo's only training-dynamics
   datapoint.

### Tier C — the calibrated detector (Phase 0 proper; already specified in the program doc)

9. **Weight-shuffle null** (preserves the magnitude marginal, destroys learned
   coordinate structure), **max-statistic permutation logic** (Nichols &
   Holmes 2002), **planted-weight recovery test**, and an explicit
   **"no super weight found" outcome** — which no published method can emit.
   See `three_axis_program.md` Phase 0; don't re-derive the design here.

### Beyond — the axes

Formation traces (Pythia backward-trace, PolyPythias seeds, Ettin
encoder/decoder), per-language damage profiles, and the NLLB weight-level
search are Axes 1–3 in `three_axis_program.md`. Nothing to add here; they all
gate on Phase 0.

## 5. Standing constraints on whatever gets picked

- **No quantization-protection framing.** Ruled out twice (`docs/registry.md`,
  Ruled out list; the `rtn+SW ≈ rtn` negative). The legitimate compression link
  is formation of the outlier structure, not a protection recipe.
- **"Super weight" ≠ "salient channel."** They behave oppositely.
- **Magnitude ≠ importance ≠ sensitivity.**
- The claim, when multilingual, is always *single scalar weight* — the
  neuron-granularity version is occupied (arXiv:2402.16438, 2402.18815).
- Log what happens — including what breaks — in `../notes.md`, dated.
