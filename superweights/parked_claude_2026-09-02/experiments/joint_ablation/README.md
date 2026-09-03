# Superweights — joint_ablation

> When a model's super-weight candidates are individually inert, is the *set* jointly
> catastrophic — and is the super activation itself load-bearing regardless of which weights make it?

Every ablation in this track so far zeroed one scalar at a time (`notes.md`, 2026-09-02).
Subramanian et al. (COLM 2026, Table 8) zero Yu et al.'s coordinates jointly and report
Phi-3-mini ×374 on the six coordinates that are ×1.00–1.02 here individually. Our own JSON
shows every individually-inert model's candidates sharing one `down_proj` input column
(Phi-3: rows {525,1113,1693} × columns {808, 2723}; Llama-2-7B: [2533,7890]+[1415,7890];
Llama-3.1-8B: three rows on column 2427; Qwen3-8B: two on 5723). Until the set is ablated
as a set, "no super weight in this model" is unsupported. This experiment is the gate for
RQ2 (the modern-model negative) and RQ3 (whether Tower/EuroLLM/Aya/BLOOM have anything
strong enough to measure per language).

Adjacent registry entry: none ruled out. `registry.md`'s q6 table used single-scalar KL;
this changes the unit, not the metric.

## Hypothesis

**Our hypothesis is that** criticality is carried by a small set (2–6) of `down_proj`
weights fanning one intermediate neuron into several residual channels, so that the set is
jointly catastrophic in every model where the super activation exists, including the
individually-inert ones. Second, **our hypothesis is that** the super activation is
load-bearing independently of the weights: removing it directly at its onset layer
(Sun et al. 2024 Table 3 protocol, "set to zero") collapses every model with a persistent
massive activation, including Qwen3-8B, where no single weight is critical.

## Satisfied when

For each of the 12 models (9 Yu Table 2 + Llama-3.1-8B-Instruct, Qwen3-8B, TowerBase-7B):

1. **Joint criticality.** The paired-bootstrap 95% CI (over the 32 wikitext-2 windows) of
   the joint-ablation perplexity ratio has a lower bound above **the maximum** of N=50
   magnitude-matched random joint ablations of the same size drawn from the top-100 |W|
   of the same matrices (max-statistic null). A model where this holds is "jointly critical".
2. **Distributed vs. single.** A jointly-critical model is "distributed" if every
   individual candidate's ratio upper CI bound is < 2.0 and the joint lower bound is > 10.
   It is "single" if one leave-one-out (all but that weight) drops the ratio below 2.0.
3. **Published anchor.** Phi-3-mini on the six Table 2 coordinates jointly lands within a
   factor of 3 of Subramanian's ×374 (their windowing differs from ours; a factor of 3 is
   the OLMo-1B calibration: ×3,663 vs our ×3,667 shows the two protocols agree when the
   effect is large). Outside that factor is a real discrepancy to be reported as such.
4. **Direct super-activation removal.** Setting the detected super-activation channel(s)
   to zero at the onset layer, at every token where it exceeds half its detection-prompt
   peak, gives a ratio whose lower CI bound exceeds 10 in every model with a persistent
   super activation — or, if it does not, the model is named.

The experiment is satisfied if criteria 1–4 are each decided (met or not met) for all 12
models with CIs; it is *not* satisfied by any bare mean.

⚠️ Satisfied-when written by Claude on 2026-09-02 while Adrian was unavailable; **not yet
confirmed by Adrian.** Confirm or amend before citing any result outward.

## Null / baseline

- **Baseline:** the intact model on the same 32 × 2048-token wikitext-2 windows
  (`ablate_sw.py` default; every ratio is against this).
- **Magnitude-matched null (individual):** N=50 random single weights drawn from rank
  1–100 by |W| in each candidate layer's `down_proj`, excluding candidates. Reports the
  max ratio — the individual-scalar reference for "extreme".
- **Magnitude-matched null (joint):** N=50 random sets, same per-layer counts as the
  candidate set, drawn from the same top-100 pools. Reports max and 95th percentile.
- **Super-activation null:** zeroing the same number of *median-magnitude* channels at the
  same layer/tokens (Sun et al.'s control) — N=10 draws.
- Random-init null not applicable (no training here); the max-statistic null over
  magnitude-matched weights is the calibrated reference.

## Analysis plan

- **Statistic:** perplexity ratio ablated/intact, computed as exp(mean loss) over windows;
  95% CI from a paired bootstrap over windows (B=2000, log-space, seed 0).
- **Seeds:** the null draws use seed 0; the candidate coordinates are deterministic given
  the v5 detector output (`results/v5`, `results/modern`), one Hub revision per model.
  This is n=1 model revision per model — coverage is 12 of 12 models, not seeds.
- **Family size:** per model, family = (candidates + 3 joint sets + leave-one-outs);
  reported next to each table. The null max already controls the search over the top-100.
- **"No effect" means:** the joint ratio's 95% CI lies inside [0.9, 2.0] *and* the null
  max is also inside it.

## What would change my mind

- Phi-3's six Table 2 coordinates jointly inside the null (ratio < ~3) → our Phi-3 is a
  real discrepancy with Subramanian, not a unit-of-ablation artifact.
- Qwen3-8B's super activation removed directly and the ratio inside [0.9, 2.0] → the
  activation is not load-bearing there, and "super activations look universal" is a
  magnitude statement with no causal content for that model.

## Out of scope

- Detector thresholds (v5 is frozen; candidates are taken as given).
- Per-language damage (RQ3) — this run decides whether it is worth measuring.
- Encoder-decoder models (NLLB) and ≤1B models — separate experiment.
- Zero-shot accuracy; perplexity only.

## Runs

| Config | Model set | Seed | Candidates | Results dir | Manifest |
| --- | --- | --- | --- | --- | --- |
| `configs/table2.yaml` | `sw_models.MODELS` (9) | 0 | `results/v5` | `results/v6` | in every JSON |
| `configs/modern.yaml` | `sw_models.MODERN` (3) | 0 | `results/modern` | `results/modern_v6` | in every JSON |

## Verdict

_pending_
