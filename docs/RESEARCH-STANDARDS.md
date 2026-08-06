# Research standards — sourced

Backing detail for [`../CLAUDE.md`](../CLAUDE.md). Compiled 2026-08-05 from a
live literature sweep. **Every rule here carries a source.** Where a
recommendation is editorial it says so.

> **Known gap.** The general software-engineering layer — Hydra vs plain YAML,
> W&B vs MLflow, `cookiecutter-data-science`, architecture decision records, DVC
> data versioning, and the formal reproducibility checklists (NeurIPS,
> Papers-with-Code) — was **not** researched; that agent was cut before
> returning. Treat this document as covering *research* practice, especially
> checkpoint-suite and interpretability practice, and not general MLOps.

---

## 1. The research process

The field's shared process document is Neel Nanda's four-post sequence:
[Explore, Understand, Distill](https://www.lesswrong.com/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand),
[Key Mindsets](https://www.lesswrong.com/posts/cbBwwm4jW6AZctymL/my-research-process-key-mindsets-truth-seeking),
[Research Taste](https://www.lesswrong.com/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research),
[How to Write ML Papers](https://www.lesswrong.com/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers).

**The exploratory/confirmatory split is the field's core organizing convention.**

| Stage | North star | Rigor posture |
|---|---|---|
| Exploration | "gain information" | rigor *deliberately* deprioritized; breadth-first |
| Understanding | "gain evidence for and against" specific hypotheses | "every research result is false until proven otherwise" |
| Distillation | "concise, rigorous truth" | higher bar than convinced-yourself |

The named failure mode: researchers "think they are in the understanding stage,
but are actually in the exploration stage." Reverting is normal.

Adoptable directly:
- Keep a **highlights doc** during exploration to spot cross-experiment links.
- Ask **"am I getting enough information per unit time?"**; make new plots every
  few minutes.
- Per experiment: **"how surprised would I be if this turned out to be bullshit
  due to a bug?"** Re-implement critical results by a second pathway.
- **Exploratory p-threshold p < .001**, justified by replication rates (~28% for
  .005 < p < .05 vs ~74% for p ≤ .005).
- **Randomly sample qualitative examples** — "it's so easy to implicitly
  cherry-pick."
- Start writing **a month before the deadline**; writing reveals gaps.

Method-level standard: [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
(Heimersheim & Nanda) — uses exploratory vs confirmatory as explicit design
categories.

## 2. Statistics

- **Effective sample size.** [Bouthillier et al., MLSys 2021](https://arxiv.org/abs/2103.03098):
  randomizing **only weight init** — "the predominant approach used in the
  literature today" — converges to the quality of **k ≈ 2 ideal samples no matter
  how many seeds you run.** Randomizing all sources gives k ≈ 2–100.
- **Power.** [Colas et al.](https://arxiv.org/abs/1806.08295): at α=.05, N=5 gives
  β=0.51 (51% miss rate); N=10 gives β=0.19.
  [Reimers & Gurevych](https://aclanthology.org/D17-1035/): seed alone produced
  p < 10⁻⁴ differences and ~1 F₁ on NER.
- **Multi-Bootstrap.** [MultiBERTs](https://arxiv.org/abs/2106.16163)
  ([code](https://github.com/google-research/language/tree/master/language/multiberts)):
  resample **seeds and test examples jointly, both with replacement**; use the
  paired design. Bootstrapping test examples alone is overconfident because it
  treats the seed as fixed. Claims should be about the *training procedure*, not
  about run 7.
- **Multiplicity.** [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/)
  found **3 of 110** multi-dataset ACL papers corrected for it. Use BH/FDR for
  exploratory scans (report q and family size); **cluster-based permutation** for
  the checkpoint and layer axes, which are autocorrelated by construction —
  **permute seed labels, not checkpoints**. A cluster test licenses "an effect
  exists somewhere in layers 3–7"; it does **not** license "the onset is at
  checkpoint 22" — that needs [Cluster Depth Tests](https://arxiv.org/pdf/2105.07514).
- **Grouping.** Checkpoints within a run and heads within a layer are not
  independent — `GroupKFold` grouped by run for any probe CV.

## 3. Null models — the master rule

[The Dead Salmons of AI Interpretability](https://arxiv.org/abs/2512.18792)
(Méloux et al.): on **randomly initialized networks**, saliency maps stay
plausible, linear and structural probes reach surprisingly high accuracy, and
SAEs recover apparently interpretable components. Corroborated by
[Heap et al.](https://arxiv.org/abs/2501.17727) and
[Everything, Everywhere, All at Once](https://arxiv.org/abs/2502.20914).
The general form is [Adebayo et al., Sanity Checks for Saliency Maps](https://papers.neurips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf)
— model-parameter randomization and label-permutation tests generalize to *any*
interpretability measure.

> **Every measure must be shown to differ from its value on a randomly
> initialized model of the same architecture.** In a checkpoint study, step 0 is
> a free null. A measure that looks interesting at init is measuring your method.

This is also the fix for this repo's greedy `argmax` super-weight detector, which
returns a candidate whether or not one exists.

## 4. Interventions

- **Never zero-ablate** — off-distribution. Mean ablation destroys variation.
  **Resample/counterfactual ablation** is standard, confirmed empirically by
  [MIB, ICML 2025](https://arxiv.org/abs/2504.13151).
- **Noising and denoising are asymmetric** — denoising tests sufficiency, noising
  necessity. Run both; report disagreement.
- **Use a metric roughly linear in logits** (logit difference, KL). Probability is
  exponential in logits and **manufactures artificially sharp transitions**.
- **A successful intervention is not localization** ([Makelov et al., ICLR 2024](https://arxiv.org/abs/2311.17030)),
  and localization does not license editing ([Hase et al.](https://arxiv.org/abs/2301.04213)).
- **Do not use circuit overlap as a proxy for mechanism sharing** across
  languages, checkpoints, or seeds ([Hanna et al.](https://arxiv.org/abs/2403.17806)).

## 5. Representational similarity

The most exposed area for any cross-lingual alignment claim.

- [Ding, Denain, Steinhardt, NeurIPS 2021](https://arxiv.org/abs/2108.01661):
  **CKA fails sensitivity** — 97% of principal components had to be deleted before
  dissimilarity registered. PWCCA fails specificity. **Orthogonal Procrustes** is
  a strong classical baseline.
- [Davari et al., ICLR 2023](https://arxiv.org/abs/2210.16156): CKA can be
  manipulated without changing functional behavior; extreme outlier sensitivity.
- **Finite-sample bias is the killer for small models.** The naive CKA estimator
  **tends to 1 even for completely random unaligned representations** as the
  feature/sample ratio grows — exactly the small-model regime.
- [ReSi, ICLR 2025](https://arxiv.org/abs/2408.00531): **"No Free Lunch."** CKA
  ranked 1st in language, 8th in vision, 11th in graphs. Rankings do not transfer.
- **Calibration procedure to adopt:** [Revisiting the Platonic Representation Hypothesis: An Aristotelian View](https://arxiv.org/abs/2602.14486)
  ([code](https://github.com/mlbio-epfl/Aristotelian)). CKA has a non-zero null
  baseline scaling as O(d/n) — **wider models look more aligned for free.** Use
  permutation calibration; report `s_cal` and empirical p. **Aggregation-aware**
  calibration (same permutation across layers, null on the max) is the
  multiple-comparisons fix for layer scans.
- **The cross-lingual-specific confound:** [Libovický, Rosa, Fraser, Findings EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.150/)
  — mBERT sentence representations are approximately *language centroid +
  language-neutral component*. **An uncentered cross-lingual similarity measure is
  partly measuring "these are both sentences in language L."** Report centered and
  uncentered; if the effect vanishes under centering, it was language identity.

> Report ≥3 measures with different invariance classes; never a raw CKA number;
> state n, d, and the null baseline; pin one CKA variant; anchor every alignment
> curve to a functional co-measure; **never average alignment curves across runs
> whose transitions occur at different steps** — that smears a real transition
> into a fake ramp.

## 6. Phase transitions

[Schaeffer, Miranda, Koyejo, NeurIPS 2023 Outstanding Paper](https://arxiv.org/abs/2304.15004):
Accuracy ∝ p^L is geometric in target length (arbitrarily sharp) while Token Edit
Distance is quasi-linear. They *induce* emergence in vision models purely by
inventing a thresholded metric.

**The debate is live, and the rebuttals are substantive:**
[Du et al., NeurIPS 2024](https://arxiv.org/abs/2403.15796) — emergence appears
when pre-training loss crosses a threshold "regardless of the continuity of
metrics" (the strongest counter);
[Hu et al., ICLR 2024](https://arxiv.org/abs/2310.03262) — raising resolution does
not always dissolve sharpness;
[Wu & Lo, ICLR 2025](https://arxiv.org/abs/2410.01692) — emergence as composed
U-shaped curves, **so a jump in a pooled multilingual metric may just be
high-resource languages crossing a threshold**;
[Michaud et al.](https://arxiv.org/abs/2303.13506) — smooth aggregates and
discrete skill acquisition are compatible. **Smoothness is not evidence against a
transition any more than sharpness is evidence for one.**

Gold standard: [Olsson et al.'s six arguments](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/)
— co-occurrence, **co-perturbation**, ablation, generality, mechanistic
plausibility, continuity across scale. They admit their scalar metric is
"somewhat arbitrary" and show robustness to the choice. **Adopt that pattern.**

Only published stage-detection recipe: [Hoogland et al., TMLR 2025](https://arxiv.org/abs/2402.02364)
— LLC via SGLD, GP-smooth, plateaus as local minima of |derivative|. They candidly
report finding *no* change at one boundary. ⚠️ LLC is a weight-space quantity —
compute on unfolded, uncentered weights.

**To claim a transition:** sharpness survives ≥2 metrics of different continuity
class; evidence is a bump in d(measure)/d(log t) with the smoother stated;
**extrapolation failure** on held-out checkpoints (the field's weakest-covered
requirement, and cheap to satisfy); plot against **loss**, not only step; a
scatter of "step where A transitions" vs "step where B transitions" across runs
with slope ≈ 1 beats overlaid curves; disaggregate by language and frequency;
optimizer-artifact controls; **multi-seed transition-rate under a pinned
numerical environment**; never report a transition from a run-averaged curve.

⚠️ [Grokking Is Conditional and Fragile](https://arxiv.org/abs/2607.05104) (Jul
2026): **CPU thread count 1 vs 4 flips the grok outcome of 49 of 300 seeds (16%)**;
CPU→GPU flips 19 of 100. Protocol: *"Never report grokking from a single run.
Every claim is a grok-rate over ten or more seeds"* — pin and report thread count
and device. Grokking is also contested as a phenomenon:
[Omnigrok](https://arxiv.org/abs/2210.01117) induces and eliminates it by tuning
init scale and weight decay; [Slingshot](https://arxiv.org/abs/2206.04817) ties it
to optimizer instabilities.

## 7. Checkpoint suites

**Pythia** ([arXiv:2304.01373](https://arxiv.org/abs/2304.01373)) is the standard:
154 checkpoints, log-spaced to step 512 then uniform every 1000. **Its documented
flaw:** log-spacing stops at ~1B tokens, but induction heads form in a 2.5–5B
window ([Olsson et al.](https://arxiv.org/abs/2209.11895)) — the dense grid ends
right before the most-studied transition.
[When Do Attention Circuits Form?](https://arxiv.org/abs/2606.02378) reports
OLMo-1B's BOS-attractor fraction going **7% → 70% between two adjacent
checkpoints**. Prescription: **variable-density sampling, densifying near
predicted transitions.**

**Better schedule shape** — Stanford CRFM's four uniform regimes with
geometrically growing period bounds the maximum late-training gap, which pure
log-spacing does not (its last two points sit ~15% of the run apart).

⚠️ **A mid-run checkpoint is not a converged model.** OLMo 1 §4: intermediate
checkpoint performance "is influenced by where that checkpoint occurs in the
learning rate schedule."
[Scaling Law with LR Annealing](https://arxiv.org/abs/2408.11029) formalizes it.
**Fix: use a WSD/trapezoidal schedule** ([MiniCPM](https://arxiv.org/abs/2404.06395))
— every stable-phase checkpoint sits at the same LR. *This single choice does more
for comparability than any amount of extra density.*

**Conventions worth copying:** DataDecide's `step{N}-seed-{name}` single revision
string; OLMo encoding provenance in branch names; LLM360 Amber's data-order trick
(globally permute, split into as many chunks as checkpoint intervals — this is how
they found 4 chunks that produced NaN loss regardless of position). Amber publicly
regrets not saving optimizer state and writing BF16 checkpoints from an FP32 run.
⚠️ Stanford CRFM's checkpoint branches **no longer exist** — what survived is the
manifest, because it was in the code repo. **Publish a manifest in git; mirror
artifacts somewhere you control.**

**Closest template: [Pico](https://www.picolm.io/)** — `pico-decoder-tiny` (11M)
and `small` (65M) bracket 36M, LLaMA-style. Saves weights, config, eval JSON, full
optimizer state, and activations/gradients **plus the exact batch they came
from**, with a fixed held-out eval batch reused at every checkpoint.
[`pico-analyze`](https://github.com/pico-lm/pico-analyze) already implements CKA,
PWCCA, proportional effective rank, condition number, Gini, and Hoyer. **Matching
its layer naming gives free tooling and two public baselines.**

## 8. Tooling status (verified 2026-08-06)

⚠️ **TransformerLens 3.6.0 requires `transformers>=5.9.0`; circuit-tracer is
pinned `<=4.57.3`. They cannot coexist.** Plan two venvs.

⚠️ **`HookedTransformer.from_pretrained` now emits a `DeprecationWarning`**, with
the `Hooked*` classes "slated for removal" in favor of `TransformerBridge`. But
the bridge has open correctness bugs that name this exact use case: **#1568**,
from-scratch weight init drifts ("grokking, superposition become silently
irreproducible"), and **#1587**, `state_dict()`/`load_state_dict()` are not
inverses — *"you can reload a checkpoint into a random-init model, get no error,
and analyse garbage."*
**Pin `transformer-lens==3.6.0` and use the deprecated path; it is currently more
correct.**

⚠️ **Weight processing across checkpoints is a trap.** `fold_ln` is
function-preserving but rescales all weight norms by that checkpoint's γ, **and γ
grows over training** — so any weight-space progress measure computed on folded
weights measures a different trajectory. `center_writing_weights` is
function-preserving **only under LayerNorm** and is *silently skipped* under
RMSNorm. **Freeze one processing regime across all checkpoints, default to none,
and record the flags in every metric row.**

| Tool | Status | Verdict |
|---|---|---|
| SAELens | v6.47.1, active | Works for from-scratch via `override_model` |
| nnsight | v0.7.0, active | Best for arbitrary `nn.Module`; no cache/patching library — escape hatch |
| devinterp | v2.0.1 | *The* training-dynamics library; steal its Zarr/xarray data model |
| TorchLens | active | Use once, early, to verify the graph |
| penzai | **dormant ~14 months** | Do not adopt |
| transformer-debugger | **abandoned** | Ignore despite 4.1k stars |
| tuned-lens | PyPI stale since 2023 | Reimplement (~150 lines) |

**There is no library for tracking interp metrics across a checkpoint grid.** An
xarray `DataArray` indexed by `(run, step, layer, language, metric)` is the right
data model. Use W&B for training curves, JSONL/Zarr on disk for interp metrics —
a 900-point grid is a re-analyzable array, not a live stream.

## 9. SAEs

If used at all, know the state: seed-only differences share **~30%** of features
(Llama-3-8B) / ~42% (Pythia-160M) ([Paulo & Belrose](https://arxiv.org/abs/2501.16615))
— and orphan features are still interpretable, so **interpretability of a feature
is not evidence the feature is real**.
[Kantamneni et al., ICML 2025](https://arxiv.org/abs/2502.16681) found SAE win
rates of **2.2–8.7%**: *"We were initially tricked by promising results for SAEs,
but when we tried hard to find a strong baseline method, the improvements would
disappear."* SynthSAEBench: best SAE 0.88 F1 vs **0.974 for logistic regression**,
with reconstruction quality *anti-correlated* with feature quality. Feature
absorption worsens with wider SAEs, hedging with narrower — **no width avoids
both.**

⚠️ **Two citation corrections.** There is no Nanda post titled "SAEs are
disappointing." And **Anthropic did not abandon SAEs** — their
[June 2026 update](https://www.transformer-circuits.pub/2026/june-update/index.html)
still calls them "a valuable tool." What happened is *DeepMind* deprioritizing
fundamental SAE research (Mar 2025). "SAEs are not the peak tool" is defensible;
"Anthropic abandoned them" is an error a reviewer will catch.

## 10. Reporting and venues

- **X-axis = tokens seen, log-scaled**, secondary axis in steps; state checkpoint
  spacing explicitly.
- **Show all per-seed lines at low alpha, bold median, IQR band. Never show only
  the band.** Captions must name what the band is (NeurIPS checklist item 7).
- Mark phases with vertical dashed lines and numeric ranges in the caption.
- One shared `plotting.py` — what keeps hundreds of figures consistent.

**Venues:** [BlackboxNLP 2026](https://blackboxnlp.github.io/2026/call/) opened a
**Special Track on Reproducibility and Reliability in Interpretability Analyses**,
whose call cites the random-init-plausibility problem directly and asks for random
baselines and effect sizes — the natural target, and its
[Reproducibility Challenge](https://blackboxnlp.github.io/2026/reproducibility/)
"welcome[s] and encourage[s] negative results."
The [ICML 2026 Mech Interp Workshop](https://mechinterpworkshop.com/cfp/) is
**non-archival** and explicitly solicits "rigorous negative results" and
"compelling failed replications." ATTRIB @ NeurIPS has an Idea track accepting
"documentation of failed experiments."

**Two 2026 position papers** now push interp toward causal-inference disclosure:
an audit of 30 papers found **0/30 have an identification-assumptions section**,
and an ACL 2026 paper opens with two peer-reviewed papers reaching *contradictory
conclusions about the same mechanism*.

**Pre-registration, honestly:** the NeurIPS Pre-registration Workshop ran twice and
died — 22 proposals → 10 accepted → **3 results papers**. Interp-specific
pre-registration does not exist. It remains a good internal discipline; do not
expect venue credit.

## 11. Documented gaps — cheap contributions

1. **There is no accepted statistical test for a phase transition in a training
   curve.** The devinterp community triangulates with no formal threshold, no
   pre-specified sample size, no multiplicity correction.
2. **No interp paper has a multiple-comparison protocol** — not even
   BlackboxNLP's new reproducibility track mentions one.
3. **No standardized benchmark exists for training-dynamics or emergence claims.**

---

*Compiled from a live sweep on 2026-08-05. Some page extractions ran through a
summarizer — verify before quoting verbatim: the BIG-Bench counts in Schaeffer et
al., MultiBERTs' exact wording, and Hewitt & Liang's control-task accuracies.*
