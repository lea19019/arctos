# Research standards — sourced

Backing detail for [`../CLAUDE.md`](../CLAUDE.md). Compiled 2026-08-05 from a
live literature sweep. **Every rule here carries a source.** Where a
recommendation is editorial it says so.

**§§1–11** cover research method (statistics, null models, interventions,
similarity, transitions, checkpoint suites, tooling, reporting). **§§12–19**
cover research *engineering* — configuration, provenance, tracking, artifacts,
repo structure, testing, and how failure gets recorded — and were compiled
2026-08-06 by reading fifteen primary sources **as code**: the repositories
themselves, plus Anthropic's and GDM's published interpretability practice, plus
the formal reproducibility checklists. Every claim in §§12–19 points at a file
and line, a named post, or a quoted checklist item. Where it doesn't, the claim
isn't made.

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
| penzai | **dormant ~14 months** | Do not adopt — **and it is JAX-first** (`jax>=0.4.23`; torch only as a conversion dep), which disqualifies it here independently of dormancy |
| transformer-debugger | **abandoned** | Ignore despite 4.1k stars. ⚠️ Re-checked 2026-08-06: **not archived**, but the last commit is a pre-commit pin bump, and it calls `gpt-4o` — disqualifying on an offline node |
| tuned-lens | PyPI stale since 2023 | Reimplement (~150 lines). ⚠️ Also: `wandb` is still a **mandatory** dependency — a hard failure on an offline node |
| inseq | v0.7.1 (2026-03), active | **Encoder-decoder-*first*.** Config names `M2M100ForConditionalGeneration` (= NLLB-200) and `NllbMoe`. Added 2026-08-06 |
| pico-analyze | commit 2026-02 | **Never loads a model** — reads saved tensors only, so CKA/PWCCA/effective rank are architecture-agnostic for free. Added 2026-08-06 |
| pyvene | v0.1.8, semi-maintained | Architecture-general interventions; its ESM card maps verbatim BERT module paths, so a RoBERTa card is a file copy. Added 2026-08-06 |

⚠️ **The encoder / encoder-decoder picture changed in 2026.** Verified inside the
published `transformer_lens-3.6.0` wheel on 2026-08-06: `HookedEncoder` uses the
**same `blocks.{i}` hook names as `HookedTransformer`**, so
`transformer_lens.patching` runs unmodified on BERT; `HookedEncoderDecoder` plus
**12 encoder-decoder bridge adapters** ship, including `m2m100.py` whose
docstring names **NLLB-200**, with HF logit parity < 1e-5. Caveats: **no
RoBERTa/XLM-R adapter** (the BERT one is ~150 declarative lines), and the NLLB
adapter landed 2026-07-27 in a machine-generated 406-file PR with an open
correctness issue (#1611, seq2seq `return_type="loss"`). **What is still
genuinely decoder-only is SAEs and circuit tracing** — on tooling grounds *and*
on the absence of any published SAE precedent on a text encoder or MT
encoder-decoder. Detail: `interlingua/docs/method_landscape.md` §3.

⚠️ **Correction (2026-08-06): `run_with_cache` does *not* raise on batch size
> 1.** The `NotImplementedError` is scoped to `generate(return_cache=True)`;
the batched-`run_with_cache` issue (#1265) closed 2026-04-22. Bugs #1568 and
#1587 above **are** still open, but both live in the `boot_native`
train-inside-TransformerLens path — training in HF `transformers` and wrapping
with `boot_transformers(hf_model=…)` avoids both, which is what
[`decisions/0001`](decisions/0001_interlingua_model_implementation_substrate.md)
already proposes.

**There is no library for tracking interp metrics across a checkpoint grid.** An
xarray `DataArray` indexed by `(run, step, layer, language, metric)` is the right
data model. Use W&B for training curves, JSONL/Zarr on disk for interp metrics —
a 900-point grid is a re-analyzable array, not a live stream.

⚠️ **This may be superseded.** **TRACE** ([arXiv:2507.03668](https://arxiv.org/abs/2507.03668),
EMNLP 2025 demo `2025.emnlp-demos.62`) is a modular **in-training** analysis
toolkit — probing, intrinsic dimensionality, Hessian curvature, layer-wise
diagnostics — whose paper explicitly claims existing tools *"lack temporal
tracking."* **Unverified beyond the abstract**; evaluate before building the
xarray harness. Also **SAE-Track** ([arXiv:2412.17626](https://arxiv.org/abs/2412.17626))
for warm-started SAE series across checkpoints, and `reward-lens`
([arXiv:2604.26130](https://arxiv.org/abs/2604.26130)) for a ten-method adapter
protocol that isolates architecture-specific details so lens/patching/SAE modules
are written once. ⚠️ Note a **name collision**: arXiv:2505.17998 is a different
"TRACE" (a phase-transition detector).

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

# Part II — Research engineering

Sources read as code: [gpt-neox](https://github.com/EleutherAI/gpt-neox),
[pythia](https://github.com/EleutherAI/pythia),
[OLMo](https://github.com/allenai/OLMo) + [OLMo-core](https://github.com/allenai/OLMo-core),
[levanter](https://github.com/stanford-crfm/levanter),
[nanotron](https://github.com/huggingface/nanotron),
[open_lm](https://github.com/mlfoundations/open_lm),
[litgpt](https://github.com/Lightning-AI/litgpt),
[pico-train](https://github.com/pico-lm/pico-train) + [pico-analyze](https://github.com/pico-lm/pico-analyze),
[marin](https://github.com/marin-community/marin),
[metaseq OPT chronicles](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles),
[bigscience](https://github.com/bigscience-workshop/bigscience),
[amber-train](https://github.com/LLM360/amber-train),
[EleutherAI cookbook](https://github.com/EleutherAI/cookbook),
[chex](https://github.com/google-deepmind/chex),
[xmanager](https://github.com/google-deepmind/xmanager),
[maxtext](https://github.com/google/maxtext),
[penzai](https://github.com/google-deepmind/penzai),
[tracr](https://github.com/google-deepmind/tracr),
[circuit-tracer](https://github.com/safety-research/circuit-tracer).
Prose sources: transformer-circuits.pub, the GDM mech-interp team's progress
updates, the NeurIPS and Pineau checklists, `releasing-research-code`.

> **The single most important finding, up front.** *Almost none of these
> organizations write provenance into their results.* A `git rev-parse` grep
> returns **zero hits** across every Google DeepMind research repo, every
> Anthropic public research repo, OLMo v1, nanotron, open_lm, pico-train, and
> pico-analyze. Pythia has no run manifest at all. The `CLAUDE.md` rule "write
> `git_sha`, the resolved config, library versions, and the seed into every
> results directory" is **stricter than published practice at every lab
> surveyed**. Do not take the field's absence as license. Take it as the reason
> its results decay.

## 12. Configuration

### 12.1 The five designs actually in use

| Design | Example | Validation | Verdict |
|---|---|---|---|
| Flat argparse | open_lm `params.py:263-800` — **114** `add_argument` in one function | hand-written `check_args()` `params.py:189-260` | Avoid |
| Dataclass + YAML | nanotron `config/config.py:442-457`, `dacite.from_dict(strict=True)` `:647` | unknown key = hard error; `__post_init__` | **Adopt** |
| OmegaConf structured | OLMo v1 `olmo/config.py:128-146` | schema types only; semantics leak into `scripts/train.py` | Superseded |
| Dataclass, no OmegaConf | OLMo-core `config.py:56-327` | `validate()` hook; `_CLASS_` field round-trips polymorphism | Adopt if you need polymorphic configs |
| Signature-derived | litgpt `jsonargparse.CLI` + `attribute_docstrings=True` (`utils.py:525`) | jsonargparse types + `validate_args()` `pretrain.py:511-533` | Adopt the *validation* pattern |

**One line does most of the work.** Three independent projects landed on the
same defence against typo'd keys: `dacite(strict=True)` (nanotron
`config.py:647`), Pydantic `ConfigDict(extra="forbid")` (maxtext
`types.py:2774`), and gpt-neox's plain `cls(**config)` raising `TypeError`
(`arguments.py:284`). Pick one. maxtext's 4,219-line `types.py` is elaborate;
`extra="forbid"` catches more real bugs than the rest of the file.

Three validation patterns worth copying verbatim:

- **Accumulate all errors, raise once.** litgpt `pretrain.py:511-533` collects
  every problem into a list before raising, and *warns* on debug-only flags
  ("`train.max_steps` is intended for profiling or debug runs only").
- **Detect double-override.** maxtext `pyconfig.py:128-150` raises when a model
  config and the CLI set the same key to different values unless
  `override_model_config=True`. This is the mechanical form of `CLAUDE.md`
  rule 6.
- **Validate in `__post_init__`, next to the fields.** nanotron
  `config.py:485-547` (derived defaults, divisibility asserts, enum
  whitelists). open_lm's separated `check_args()` must be manually kept in sync
  and drifted.

### 12.2 The real disagreement: YAML-per-run vs experiments-as-Python

Both camps are serious and both are winning arguments about different things.

**Python-as-config.** Marin's experiments are `experiments/exp<ISSUE#>_<desc>.py`
(`docs/explanations/guidelines.md:108-113`), and sweeps are `itertools.product`
— `lib/marin/src/marin/experiment/sweep.py:6-10` states the position outright:
*"There is no framework `select`: selection is ordinary code over the resolved,
typed outputs."* OLMo-core's released runs are Python scripts
(`src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py`), not YAML. maxtext's
benchmark sweeps are a `MaxTextModel` dataclass registry expanded by nested
loops.

**YAML-per-run.** levanter commits `config/llama2_7b.yaml` as a *fully resolved
dump* of a real run, including `trainer.id: llama-8b-tootsie-0.001-19ad63` and
`seed: 0`. litgpt's `config_hub/pretrain/tinyllama.yaml` is the model artifact
here: every field present, each annotated `(type: ..., default: ...)`, nothing
implicit. gpt-neox commits the actual Pythia recipes (`configs/pythia/*.yml`).

**What the tradeoff turns on:** whether the unit you need to address is a *run*
or a *family*. YAML wins for runs — one file is greppable, diffable,
committable, and can be written into a results directory as literal evidence of
what ran. Python wins for families — a 3×4×2 sweep is six lines instead of
twenty-four files.

> **Resolution for this repo:** keep `CLAUDE.md`'s one-YAML-per-run rule, and
> commit the *Python sweep script that generates the YAMLs* alongside them.
> Marin's `grid()`/`sweep()` is 38 lines of `itertools.product`; its 199-line
> distributed `SweepTarget` machinery is used by **zero** files in
> `experiments/`. Take the 38 lines.

### 12.3 Anti-patterns, each observed

- **Merge that forbids layering.** gpt-neox `arguments.py:252` raises on any key
  present in two config files, making base+override composition impossible by
  design. Their reason is legible (all configs get flattened into one checkpoint
  dir) but the cost is real.
- **Type checks with a hole.** open_lm `check_replacement_type` (`params.py:138-149`)
  returns `True` whenever either side is `None` — every flag defaulting to
  `None` accepts any type from YAML.
- **Parsed args ≠ effective args.** open_lm mutates `args` after parsing
  (`main.py:477-486`) and writes `params.txt` *before* the mutation.
- **Write-once config.** pico-train writes `training_config.yaml` only
  `if not os.path.exists(config_path)` (`checkpointing/training.py:211-213`).
  Change the config, resume, and the recorded config is stale forever.
- **A config duplicate that silently disagrees.** pythia ships
  `utils/dummy_config.yml` with different `save-interval` and no
  `extra-save-iters` than the real `models/*/*.yml`.
- **Hyperparameters as module constants.** amber-train `main.py:33-47` has no
  config files at all. It is the flagship "we release everything" project.

## 13. Provenance — what identifies a run

### 13.1 Who actually does it, and how well

- **gpt-neox** — `git describe --always` (`megatron/neox_args.py:44-51`), stored
  as the `git_hash` dataclass field (`:731`). Caveats: runs in **cwd**, not the
  repo root; **no dirty check**; captured at import time; never written into the
  checkpoint — it reaches only W&B/Comet and stdout.
- **levanter** — `WandbConfig._get_git_sha` (`src/levanter/tracker/wandb.py:268-294`)
  tries `GIT_COMMIT` env, then GitPython, then shells out; plus
  `generate_pip_freeze()` (`tracker/helpers.py:76-80`). **W&B-only** — the
  noop/tensorboard/json trackers get the config artifact but not the SHA.
- **OLMo-core** — `GitConfig.from_env()` (`launch/utils.py:29-63`) with
  `is_dirty`, and the Beaker launcher **refuses to launch on a dirty tree**
  (`launch/beaker.py:591`). But that check lives only in the launcher: running
  `torchrun` directly gets **zero** provenance. `pip freeze` →
  `/results/olmo-core/requirements.txt` (`callbacks/beaker.py:98-109`).
- **marin — the best in the survey.** `Provenance`
  (`lib/rigging/src/rigging/provenance.py:45-67`) records `tree_hash` (a git
  *tree* hash obtained via `git stash create`, so **dirty state is captured
  without a timestamp**), `base_commit`, `dirty`, `branch`, `built_by`,
  `git_remote`, `created_at`, and `command_line` = `sys.argv`. It lands in
  `.artifact.json` beside the outputs, alongside the fingerprint and the
  materialized config.

`git stash create` is strictly better than `git rev-parse HEAD`: it gives a
content hash of the working tree including uncommitted edits, which is what you
actually ran.

### 13.2 Config contents, not paths

gpt-neox is the canonical implementation. `arguments.py:255-266` reads each
config file's **raw text** into `neox_args.config_files` (comments survive):

```python
config_files[filename] = open(conf_file_name).read()
config["config_files"] = config_files   # "used when saving checkpoints"
```

and `checkpointing.py:215-224` writes each one to
`<save>/global_step<N>/configs/<original_filename>.yml`. **Caveat: this captures
the files, not the CLI overrides**, so it is not by itself a record of what ran.

The three implementations that *do* record what ran:

- **litgpt** — `save_hyperparameters()` (`litgpt/parser_config.py:34-55`) uses
  `jsonargparse.capture_parser` to re-parse `sys.argv` **without running the
  function**, then `parser.save(...)` → `hyperparameters.yaml`. Defaults + file +
  CLI, fully merged. It is load-bearing, not decorative: `merge_lora.py:87-91`
  reads it back and errors if absent.
- **nanotron** — `config.save_as_yaml(root / "config.yaml")` inside the
  checkpoint (`serialize/main.py:67`), and `save_as_yaml` (`config.py:563-571`)
  **reloads the file it just wrote** as a round-trip check.
- **levanter** — `log_configuration()` (`tracker/tracker_fns.py:178-204`)
  `draccus.dump`s the resolved config and logs it as a run artifact of type
  `config`.

### 13.3 The minimal manifest

Composed from marin's `.artifact.json`, litgpt's resolved-config write, and
OLMo-core's `pip freeze`. Everything here is filesystem-only and costs ~40 lines:

```
results/<experiment>/<run-name>/
  manifest.json      # tree_hash, base_commit, dirty, branch, command_line,
                     #   created_at, seed(s), python/torch/transformers versions
  config.yaml        # the RESOLVED config — defaults + file + CLI overrides
  requirements.txt   # pip freeze / uv pip freeze
  metrics.jsonl      # append-only
  ...                # results
```

Two things the labs get wrong that this fixes: the git SHA living only in the
tracker (levanter, gpt-neox), and the dirty-tree check living only in the
launcher (OLMo-core).

### 13.4 Naming as provenance

**Gemma Scope's scheme is the one to copy**, and it costs nothing:
`google/gemma-scope-2b-pt-res` → `layer_20/width_16k/average_l0_71`. Every axis
that changes the artifact's behaviour is in the path, and the last coordinate is
the **measured** sparsity, not the requested hyperparameter.

**marin's is the other half:** `{prefix}/{name}/{version}`
(`lib/marin/src/marin/execution/lazy.py:225-235`), with versions CalVer-validated
and opaque `v1`-style tags **rejected** (`artifact.py:210-238`) because "an
artifact's version is the author's explicit statement of 'when this recipe was
frozen', not an opaque label."

**BigScience's is the best for experiment families:** `tr<N><letter>-<size>-<what-varies>`
— `tr3-1B3-baseline`, `tr4-1B3-rotary`, `tr6-1B3-prefix-lm`, `tr8-104B-wide`,
`tr8b-104B`. The number is the research question, the letter is a variant of it,
the size is embedded so scale-downs are legible. And corrections are written
into the name index itself: *"tr7a-1B3-alibi (not a real alibi pos embedding
experiment - the alibi matrix were not used in this experiment)"*.

### 13.5 How provenance rots

Pythia is the case study, and every failure is recoverable for free:

- The eval-harness commit is recorded in `README.md:303` as literally
  **`**to-do**`**, under a warning that the evals "may not be reproducible."
- `.gitmodules:1-4` pins `utils/gpt-neox` to a **branch**, not a SHA.
- The seed is in **no config** — README prose only ("the GPT-NeoX default: 1234").
- The eval JSONs embed their own provenance (`"model_args":
  "pretrained=EleutherAI/pythia-v1.1-70m,revision=step143000"`) — good practice —
  but that repo id no longer exists on the Hub.
- Its `*1024` step→sequence factor and its `2049` sequence length are magic
  numbers in `utils/batch_viewer.py:41-43` and `mmap_dataset.py:228`, read from
  no config, and silently wrong for the 4M-batch v0 models.

## 14. Experiment tracking on an offline cluster

### 14.1 The pattern: a tracker interface with a file backend

**levanter's `Tracker` ABC is the answer for compute nodes with no internet**
(`src/levanter/tracker/tracker.py:12-171`). Five methods —
`log_hyperparameters`, `log(metrics, *, step, commit)`, `log_summary`,
`log_artifact`, `finish` — plus `CompositeTracker` (fan-out), `NoopTracker` (all
`pass`), and **`JsonLoggerTracker`** (`tracker/json_logger.py:56-120`), which
writes JSON lines with a `_to_jsonable` coercion ladder ending in `str(value)`.
Backend selection is one config key (`tracker: {type: noop}`), and
`trainer.tracker` accepts a *tuple* of configs → auto-composite. About 150 lines
total, zero dependency on W&B.

Cheaper variants, both worth having anyway:

- **OLMo-core `MetricSaverCallback`** — `metrics_step{N}.json` + `metrics.json`
  in the save folder, filesystem-only, no network.
- **gpt-neox `Tee`** (`megatron/logging.py:29-64`) duplicates stdout/stderr to
  `<log_dir>/<hostname>_stdout.txt`, installed in `enable_logging()`
  (`arguments.py:746-755`) before anything else runs. Paired with
  `NeoXArgs.print()` (`:757-786`), which dumps every argument annotated
  `default`/`updated` and sorts updated-first. ~50 lines, no dependencies,
  survives W&B being down. This pair is the single most copyable thing in
  gpt-neox for a solo researcher.

### 14.2 Tracking must fail soft — four counterexamples

Each of these is a hard failure on a node with no internet:

- **pico-train** calls `wandb.Api().runs(...)` at init
  (`training/utils/initialization.py:547-560`) — network required to *start*.
  Worse, `run_paloma_evaluation` calls `evaluate.load("pico-lm/perplexity",
  trust_remote_code=True)` (`evaluation/tasks/paloma.py:136-149`) at **every**
  eval.
- **OLMo-core** `WandBCallback` raises `OLMoEnvironmentError` if
  `WANDB_API_KEY` is unset while enabled (`callbacks/wandb.py:122`).
- **gpt-neox** fails soft for W&B (`megatron/utils.py:157-186`) and TensorBoard
  (`arguments.py:167`) but **re-raises hard for Comet** (`arguments.py:182-217`)
  — an inconsistency, not a design.
- **nanotron** is W&B-only with **no offline fallback at all** (`trainer.py:397-411`).

open_lm gets this right by accident: both trackers are imported behind
`try/except` with `wandb = None` fallback (`main.py:38-45`) and asserted only at
use, and TensorBoard writes locally.

### 14.3 What to log, and where the grid lives

For training curves, a tracker. For the interp metric grid — many checkpoints ×
many layers × many languages — **JSONL or Zarr on disk, not a live stream**
(see §8): a 900-point grid is a re-analyzable array. Pythia's convention is the
practical version: one JSON per `(model, condition, step)` under
`evals/pythia-v1/<model>/<condition>/<size>_step<N>.json`, **each embedding its
own config block and shipping every metric with a `_stderr`**. Note that its
eval grid is ~5× sparser than its checkpoint grid — evaluate a subset, keep all
checkpoints.

## 15. Artifacts — checkpoints, results, and what is committed

### 15.1 What a checkpoint directory literally contains

| Project | Contents |
|---|---|
| gpt-neox `global_step<N>/` | DeepSpeed `mp_rank_XX_model_states.pt` / `_optim_states.pt`; `client_state` = `iteration` + an **8-field** arch subset + RNG states (python/np/torch/cuda/tracker) unless `no_save_rng`; `configs/<original>.yml` |
| OLMo-core `step<N>/` | `model_and_optim/` (DCP sharded) + `.metadata`; `train/rank{N}.pt`; `.metadata.json` (`{ephemeral, version}`); `config.json`; **`data_paths.txt`** listing every dataset file |
| nanotron `<step>/` | `model/` safetensors per tensor; `optimizer/*.pt` + `optimizer_config.json`; `lr_scheduler/`; **`random/tp-i_dp-j_pp-k.pt` — per-rank RNG as a first-class artifact**; `checkpoint_metadata.json` (versioned, with a load-time assert); `config.yaml`; `model_config.json`; `<root>/latest.txt` |
| levanter | the whole `TrainerState` pytree — `step`, `model`, `opt_state`, **`training_key`** (the PRNG *is* state), model-averaging state — via TensorStore, plus `metadata.json` = `{step, timestamp, is_temporary}` |
| litgpt `step-00000001/` | `lit_model.pth`, `model_config.yaml`, **`hyperparameters.yaml`** (the resolved config), copied tokenizer files |
| open_lm | `epoch_<N>.pt`, `optimizer_<N>.pt`, `stats_<N>.pt` — **split so the stats file can be read without loading weights**. No RNG state saved. |

Two conventions worth taking regardless of framework: **RNG state is part of the
checkpoint** (nanotron, gpt-neox, OLMo; open_lm and pico-train's DeepSpeed path
omit it), and **split the bookkeeping file from the weights** (open_lm) so
analysis can read step/metrics without a 30 GB load.

`latest` pointer: OLMo v1 uses a **symlink** (`train.py:496-505`), OLMo-core
**regex-scans `step(\d+)` dirs** (`train/checkpoint.py:358`) because symlinks
don't exist on object stores, nanotron writes **`latest.txt`**. On a POSIX
scratch filesystem the symlink is fine; the step-scan has no failure mode.

### 15.2 Data order and exact resume — three designs

1. **Save the loader state.** OLMo-core checkpoints `data_loader.state_dict()`;
   on resume, if the checkpoint's data seed differs from the config's, **the
   checkpoint's wins with a warning** (`data/data_loader.py:469-474`). OLMo v1
   restores RNG **only if world size matches** (`train.py:420`).
2. **Recompute from step — the best design.** levanter saves *no* loader state.
   Data order is a pure function of `(seed, global example index)`:
   `PermutationDataset` (`data/permutation.py:15-56`) applies an O(1)-memory
   pseudo-random permutation — a Feistel network (`data/_prp.py:151-243`, ~40
   lines of numpy) — and `BatchSchedule.global_data_offset_by_step`
   (`schedule.py:108`) maps step → index range independent of device count.
   Resume is `train_loader.iter_from_step(state.step)` (`main/train_lm.py:260-265`).
   **This is the single highest-leverage idea in the survey and it ports to
   PyTorch unchanged** — a `Dataset.__getitem__` that calls `perm(i)`.
3. **Recompute from consumed samples.** nanotron reconstructs from
   `consumed_train_samples` in the metadata — deterministic *only if the
   topology is identical* (`serialize/random.py:28`).

**Enforce the contract with an assert.** open_lm stores `shard_shuffle_seed` in
the checkpoint and hard-fails on resume with a mismatched seed
(`open_lm/main.py:108-118`): *"Since this seed affects shard shuffling, resuming
training must use the same seed."* Ten lines, and it is the best idea in that
repo.

**The bug this prevents**, live in amber-train: `fabric.seed_everything(SEED +
last_ckpt_idx + 1)` is called **once per process** (`main.py:144`), then
`np.random.permutation` is drawn sequentially from that global stream across
chunks (`main_utils.py:21`). The permutation for chunk *k* depends on how many
chunks *this process* has done, not on *k* — so a resumed run and an
uninterrupted run produce different intra-chunk orders. With 360 checkpoints,
resumption was near-certain. Intra-chunk data order in Amber is not reproducible.
The fix is one line: seed per chunk.

### 15.3 Checkpoint schedules

**levanter has the right primitive.** `CheckpointInterval(every, until)`
(`checkpoint.py:45-48`), a *list* of policies validated monotonic (`:101-112`),
with time-based *temporary* checkpoints kept separately and deleted:

```yaml
checkpointer:
  keep:
    - {every: 1000, until: 10000}
    - {every: 5000, until: 40000}
    - {every: 10000}
```

This is the piecewise-constant approximation of log spacing, and it directly
answers §7's requirement that the grid stay dense past the induction window —
make the first segment `every: 1`.

gpt-neox's equivalent is `is_save_iter` (`megatron/training.py:1408-1428`):
`checkpoint_factor` + `checkpoint_scale: "linear"|"log"` + an explicit
`extra_save_iters` list. **And the documented Pythia mistake is visible in the
committed config**: `configs/pythia/1B.yml` has
`"extra_save_iters": [0,1,2,4,...,512]` then `"checkpoint_factor": 1000` — the
log grid stops at 512 and jumps to 1000-spacing, exactly as §7 describes.

pico-train has **fixed interval only** (`save_every_n_steps: int = 1000`,
`checkpointing_config.py:81`), with weight saves, eval, and learning-dynamics
extraction all bolted to the same modulus. If you use pico-train as a template,
this is the first thing to replace.

### 15.4 Content-addressed output directories: marin built it, then removed it

This is the survey's most useful negative, because it is a *reversal by the
project that pioneered it*. Marin's old `ExecutorStep` hashed the step's config
and all transitive dependencies into the output path. HEAD no longer does.
`docs/explanations/lazy-artifacts.md:206-214`:

> *"The previous executor assigned each step an output path containing a hash of
> the step's config and all its transitive dependencies. A hyperparameter change
> silently re-addressed every downstream step, and the path gave no clue about
> what version of the data it held. […] Bumping a version is an explicit author
> decision, not an automatic consequence of any config change."*

What replaced it: explicit `{prefix}/{name}/{version}` paths, with the config
hash **demoted to an advisory drift warning**. `check_drift`
(`artifact.py:378-413`) on mismatch logs a field-level dotted-path diff
(`describe_drift`, `fingerprint.py:196`) **and serves the cached output anyway**;
escalation to `FingerprintMismatchError` is opt-in via `expected_fingerprint`.
The identity/execution split is what makes the fingerprint portable: literals in
the config enter the hash, anything pulled from the run context
(`ctx.output_path`, `ctx.prefix`, `ctx.region`) does not (`lazy.py:11-15`).

They traded silent-recompute for silent-staleness, and made the staleness loud.
Given `CLAUDE.md`'s claim-hygiene rules, that is the right ordering: a readable
path plus a printed diff beats an opaque hash. The whole mechanism is ~220 lines
of stdlib and works on a local filesystem.

### 15.5 What is committed

The consensus, verified across `.gitignore`s: **code, configs, and small result
files; never weights, corpora, logs, or tensorboard.** Specifics worth adopting:

- **litgpt enforces it mechanically** — pre-commit
  `check-added-large-files --maxkb=250 --enforce-all`. Cheap, and it makes
  `CLAUDE.md`'s "never commit weights" rule structural rather than aspirational.
- **Commit the run inventory as a small CSV.** OLMo ships
  `configs/official-1124/OLMo-2-1124-7B.csv` (`Step,Checkpoint Directory` for
  every public step) and `provenance.csv` (`Dataset,Data Directory` per shard).
  This is the "publish a manifest in git, mirror artifacts elsewhere" rule from
  §7, implemented.
- **Commit the eval JSONs.** Pythia commits **876** small result files under
  `evals/`, each carrying its own config block, and **keeps the superseded
  `evals/pythia-v0/` rather than deleting it**.
- **Keep data paths out of model configs.** gpt-neox splits `local_setup.yml`
  (cluster paths) from `configs/pythia/*.yml` (the recipe) and composes them at
  the CLI — which is what makes the model configs publishable at all. Pythia's
  committed configs carry absolute `/fsx/...` paths that no longer exist.
- **Hash your data.** pythia's `utils/shard_hashes.txt` + `checksum_shards.py` +
  `scrape.py`. On a pre-cache-then-run-offline cluster this is the whole of what
  "data versioning" needs to mean.
- **Traps:** pico-train's `.gitignore` contains `configs/` (line 174) even though
  its released configs are tracked — every *new* config is silently ignored.
  amber-train and the EleutherAI cookbook have **no `.gitignore` at all**.

## 16. Repository structure

**Library / experiment separation is the one structural decision that matters.**
Three coherent answers, in increasing order of research-orientation:

- **Library only, experiments elsewhere.** circuit-tracer has
  `circuit_tracer/` + `tests/` + `demos/`, and **no `experiments/` directory** —
  it ships a method, not a study. Ruff and pyright explicitly `exclude` `demos`,
  quarantining exploratory code from quality gates *by design*. That exclusion
  is worth copying: it lets notebook-grade code exist without pretending to be
  library-grade.
- **Numbered experiment scripts where the numbering is the changelog.**
  Anthropic's `introspection-mechanisms` uses `experiments/NN[a-z]_<name>.py`
  — `01_`…`17_`, with `b/c/d/e` suffixes for follow-ups — plus `src/` for shared
  utilities and `plotting/data/*.parquet` **committed so figures regenerate
  without a GPU**. It writes a `config.json` (name, ISO timestamp, models,
  sweeps, trial counts, seed) next to results (`03_behavioral_robustness.py:1507-1530`).
- **One directory per research question, with its own README and chronicle.**
  BigScience `train/tr11-176B-ml/` is the fullest example of a run directory
  from a major lab:

  ```
  README.md (37KB, the full spec)   chronicles.md (37KB)
  chronicles-prequel.md (65KB)      hang-debug.md
  reshaping-log.md                  backup-schedule.md
  tr11-176B-ml.slurm                start-tr11-176B-ml
  tr11-176B-ml-slurm-status.slurm   tr11-176B-ml-hub-sync-logs.slurm
  smaller_models/                   images/
  ```

  Note what is *absent*: no results files and no tensorboard in git. Logs and TB
  are pushed to Hub repos by the `hub-sync-*` SLURM jobs and the chronicle links
  to them.

This repo's existing `experiments/<name>/{README.md, configs/, slurm/, notes.md}`
is already the third pattern. It matches the best practice found. Keep it.

Two smaller structural points: **configs inside the installed package** (maxtext
`src/maxtext/configs/`, gemma) so `model_name=gemma3-4b` resolves without path
juggling; and **self-contained calculator scripts** — the EleutherAI cookbook's
`calc/README.md:11-13` deliberately refuses a shared utils module *"for the dual
purpose of (1) making them easily shared and untied to this repository (2)
clarity on which script arguments are relevant to the contained calculation."*
`calc/calc_transformer_mem.py` is directly applicable to the T4 VRAM problem in
the speech-translation track.

**If you plan to reuse pico-analyze, the integration surface is its on-disk
contract, not pico-train.** pico-analyze never loads a model or safetensors. In
local mode it needs exactly `<run>/training_config.yaml` plus
`<run>/checkpoints/step_<N>/learning_dynamics/{split}_{activations,weights,gradients}.pt`
as `torch.save`d `Dict[str, Tensor]` keyed by `named_modules()` names
(`src/utils/data.py:104,203-243`). Component addressing is
`<common-prefix><layer_index>.<suffix>` via `os.path.commonprefix`
(`components/base.py:487-507`) — HF Llama's `model.layers.0.self_attn.v_proj`
satisfies it natively, which confirms §7's "match Pico's layer naming" advice.
⚠️ `commonprefix` is character-wise, not path-aware: with a single layer, or
layers `[10,11]` only, the prefix over-matches and every lookup `KeyError`s.

## 17. Testing numerical and analysis code

### 17.1 The honest baseline — quantified

Across the fourteen repos read as code:

- **5 of 14 have zero test files**: pythia, pico-train, pico-analyze,
  amber-train, cookbook.
- **6 of 14 have no meaningful automated verification** — add metaseq, whose 154
  tests are backed by a CI that runs lint only. gpt-neox is a borderline
  seventh: `.github/workflows/pull_request.yml` is `on: workflow_dispatch`
  (**PR testing is commented out**), and what runs on push is `pytest -m cpu` on
  **torch 1.8.2**.
- **3 of 14 run tests on real accelerators in CI**: OLMo-core, nanotron, OLMo.
- **Zero repos use property-based testing. Zero use `gradcheck`.** Across roughly
  3,000 test functions, there is **one** statistical-property test
  (metaseq `tests/test_resampling_dataset.py:81`, an empirical sampling
  distribution within a documented 2% tolerance) and **zero** tests of any
  interpretability or representational-similarity measure.
- Test volume is bimodal, not graded: repos have either ~0 or 5,000–19,000 test
  LOC. The tested ones are the ones that became products.

**pico-analyze — the only pure-analysis repo in the set — has zero tests and no
CI.** It ships `cka.py`, `pwcca.py`, `per.py`, `condition_number.py`, `gini.py`,
`hoyer.py`, `norm.py`; `grep "assert "` over the whole repo returns two hits,
both argument validation. Its `per.py` docstring gives `-Σ p log₂ p` while the
code computes `exp(-Σ p ln p)/n` (`per.py:183-186` vs `:217-220`), and
`condition_number.py:257` divides by the smallest singular value with no rank
guard.

> The practical consequence: **there is no prevailing standard for analysis code
> to fall short of.** Eighty lines of closed-form invariant tests would put this
> repo above every repository surveyed.

### 17.2 The seven test types worth having, ranked by value per effort

1. **Differential test against an independent reference.** The dominant type in
   the field — 639 `allclose`/`assert_close` calls across the nine repos with
   tests. Two flavours:
   - *Against HF, with weights copied through the production converter.*
     litgpt builds HF's model, copies **its random init** into litgpt's model via
     `copy_weights_*`, runs both on the same tokens, `assert_close`
     (`tests/test_model.py:163-171`). This tests architecture *and* the
     converter in one shot, with no downloaded weights.
   - *Against a naive reference written in the test file.* OLMo
     `tests/grad_norm_test.py:142-156` defines `_naive_train_loop(...)` and runs
     the real optimizer + clipping against plain torch. **For analysis code this
     is the flavour you want** — a ten-line loop implementation of CKA catches
     vectorization bugs, wrong-axis reductions, and off-by-one windowing, which
     is the entire realistic bug class.
2. **Analytic-invariant tests.** Nobody in the survey does this for metrics, and
   it is the highest-yield thing available because the invariants are known in
   closed form: `CKA(X,X)==1`; CKA invariant under orthogonal rotation and
   isotropic scaling; effective rank of an identity `== dim`; Gini of a uniform
   vector `== 0`. The one exemplar of the *pattern* is circuit-tracer's
   `tests/test_freeze_points_hessian.py:44-141` — freeze all nonlinearities, the
   model becomes linear, therefore `d²(loss)/d(embed)² == 0`, asserted at
   `atol=1e-5`. Assert a property you can derive, not a number you recorded.
3. **"Every committed config still parses."** gpt-neox `test_neoxargs_load.py`
   instantiates every committed config; litgpt `tests/test_config_hub.py`
   auto-globs `config_hub/*/*.yaml` and asserts each validates against the script
   signature. ~60 lines, and it catches the most common research-repo rot.
   gpt-neox goes further with `test_neoxargs_usage.py:24`, which **greps the
   source for `args.*` reads and asserts each attribute exists** — a static check
   against typo'd config reads.
4. **Config round-trip.** `Config.from_dict(c.as_config_dict()).as_config_dict()
   == c.as_config_dict()` (OLMo-core `src/test/nn/transformer/config_test.py:25-27`).
   Three lines; directly enforces `CLAUDE.md`'s traceability rule.
5. **Determinism under a fixed seed.** open_lm
   `tests/test_training_simple.py:63` `test_training_deterministic` runs `main()`
   **twice end to end** with the same seed and different run names, reloads both
   checkpoints, and asserts `torch.allclose(p1, p2, atol=1e-6)` over all named
   parameters. This is the test that makes `n_seeds=N` mean N independent draws.
   Note that **levanter — whose headline claim is bitwise determinism — has no
   such test**; its coverage is a unit test on the permutation
   (`tests/test_prp.py:76-99`) plus a plot in the release post.
6. **Tiny golden values with a brittleness note.** OLMo-core
   `src/test/data/numpy_dataset_test.py:406-415`:
   `# NOTE: potentially brittle test here! / # Hard-coding exactly what the
   instances should be to ensure it's deterministic.` Committed fixture *data* is
   common; committed golden *outputs* essentially do not exist, and nobody uses
   snapshot testing.
7. **One gradient step on random input.** ML Test Score Infra 2: *"a simple unit
   test to generate random input data, and train the model for a single step of
   gradient descent is quite powerful for detecting a host of common library
   mistakes."* Plus "restore from a checkpoint after a mid-training crash."

Skip: gradient checks (nobody does them), distributed tests, GPU-marked CI.

### 17.3 Tolerances

Empirical distribution of literals across the nine tested repos: `atol=1e-5`
(177×), `rtol=1e-4` (153×), `atol=1e-4` (142×), `rtol=1e-5` (140×), `atol=1e-6`
(98×), `rtol=1e-3` (91×), exact `rtol=0/atol=0` (45×). The mode is **1e-4/1e-5
for fp32 module parity**, 1e-2 once fp16/bf16 or long chains are involved.

**Almost nobody documents why.** nanotron has three `# TODO @thomasw21: Tune
tolerance` comments sitting on live assertions. The one exemplar is OLMo-core
`src/test/nn/vision/parity_test.py:26-32`, and it is the template:

```python
# Float32 accumulation error across 24-27 transformer layers.  Two independent
# implementations of identical ops can differ by O(1e-4)-O(1e-3) due to kernel
# fusion differences and activation approximations - not from any precision loss.
# Measured worst-cases: CLIP 5e-4, SigLIP 2.8e-3, SigLIP2 3e-4.
# These tolerances verify fp32-level equivalence; they would NOT pass fp16
# inference (errors ~1e-2), which is intentional.
_RTOL = 1e-3
_ATOL = 3e-3
```

Three further rules, each sourced:

- **Bound absolute and relative error independently.** penzai
  `tests/models/transformer_consistency_test.py:180-185` asserts twice on the
  same comparison — once `atol=1e-3` (with default `rtol=1e-6`), once
  `rtol=3e-3` (with default `atol=0.0`). Both must pass.
- **Do not widen `atol` to hide a known divergence — `xfail` it with the
  mechanism.** litgpt `tests/test_model.py:66-73` marks the fp16/CUDA variant
  `xfail(raises=AssertionError, strict=False)` with the reason in the source:
  *"the reference does softmax upscaled to fp32 during attention. additionally,
  the final layernorm input is slightly different."*
- **Beware the `atol=0.0` default.** chex `assert_trees_all_close` defaults to
  `rtol=1e-06, atol=0.0` (`chex/_src/asserts.py:1661`), so near-zero values must
  match to 1e-6 *relative* — which fails constantly on quantization deltas. Its
  ULP-based alternative documents the underlying problem better than anything
  else found (`asserts.py:1764-1788`): *"with float32, the precision at 1 is
  ≈1e-7, but the precision at 5,000,000 is only 0.5… do you set the tolerance
  to…0.01? 0.001?"*

### 17.4 Gating and hygiene

- **Capability-based skip, not hardware-based.** OLMo-core
  `src/olmo_core/testing/utils.py:14-86` composes
  `pytest.mark.gpu` + `skipif(not has_cuda)` into a `requires_gpu` decorator,
  with per-library probes (`has_flash_attn_2/3/4`, `has_torchao`). levanter goes
  further with `skip_if_hf_model_not_accessible(model_id)`, which *attempts the
  load* and skips on failure — the right shape for an offline cluster.
- **Seed every test automatically.** circuit-tracer `tests/conftest.py:7-9`:
  `@pytest.fixture(autouse=True) def set_torch_seed(): torch.manual_seed(42)`.
- **Marin's bar, `TESTING.md:30-42`:** *"A test must fail when behavior is wrong.
  It should not fail only because an implementation detail […] changed. […] It is
  better to not have a test than to have 'slop' tests."*

⚠️ **This repo's 17 `pytest.skip("TODO")` stubs are worse than absent tests** —
they read as coverage in a count and provide none. gpt-neox demonstrates the
failure mode at scale: its checkpoint round-trip test and its loss-decreases test
are both `@pytest.mark.skip`, so the suite that looks like a guarantee is
largely theatre. Delete stubs or implement them.

## 18. Recording decisions, failures, and dead ends

This is the section with the most to steal and the least currently in place here
(`docs/learning/learning_log.md` has zero entries).

### 18.1 Anthropic: publish the failures, at lab-meeting rigor

**The vehicle is Circuits Updates**, a monthly post whose standing header sets
the bar explicitly and unchanged since 2023: *"Others are minor points we wish to
share, since we're unlikely to ever write a paper about them. We'd ask you to
treat these results like those of a colleague sharing some thoughts or
preliminary experiments for a few minutes at a lab meeting, rather than a mature
paper."* Each entry carries **its own byline distinct from the post's**, and a
named editor — that editor is the whole quality gate.

Four mechanisms, all free:

1. **A negative gets a full entry.** "Tanh Penalty in Dictionary Learning"
   (Feb 2024): the tanh penalty was "a Pareto improvement… often by a wide
   margin," but "the features in these autoencoders were much harder to
   interpret… Despite significant effort, we were unable to correct this issue.
   […] For now, we have put this direction to one side." It also reports the
   failure of its *own explanation*: "the simplest version of this theory… does
   not fit the evidence."
2. **Retraction banners edited into the original.** Feb 2024 entries carry
   *"Our views on ghost grads have changed. See the March 2024 update."* The
   March 2024 update then retracts three previously-announced wins: *"We no
   longer see ghost grads decreasing training loss even on 1L models… We have
   some evidence our implementation of ghost grads causes loss spikes."*
   Aug 2024 *un*-retracts tanh, honestly: *"We have not run ablations to
   determine exactly what caused the change."*
3. **Failed replications published as failures.** Aug 2024: *"We were unable to
   replicate JumpReLU being an improvement… We talked to the authors and
   attempted to replicate all the details of the paper on our infrastructure…
   There could be a bug in our infrastructure, a difference in the LLM we used,
   or some other issue."*
4. **Public errata.** Jan 2025 opens: *"An earlier version of this page
   incorrectly wrote the initialization as U(-1/n, 1/n) instead of…"*

Their rigor conventions worth importing wholesale: the **random-weights null**
(Towards Monosemanticity runs dictionary learning on a shuffled-weights model
and reports, against themselves, that *"for automated interpretability the
randomized features have a higher median score"*); **honest baseline critique**
(*"the performance with an ablated MLP may be an especially bad baseline, so this
percentage is considered an overestimate"*); and the **"(data not shown)"** tag
for unshown supporting evidence.

And their explicit **hedging vocabulary** — "we believe" / "we suspect" / "our
best guess" / "preliminary" — used consistently enough to carry information.
`CLAUDE.md`'s rule 8 ("say 'our hypothesis', not 'the mechanism'") is the same
rule.

### 18.2 GDM: set the bar before you run, then steelman the loser

The reference artifact is **"Negative Results for SAEs On Downstream Tasks and
Deprioritising SAE Research"** (Nanda, Smith, Rajamanoharan, Conmy, McDougall,
Lieberum, Kramár, Shah; 2025-03-26). Four transferable moves:

- **The bar was set before running:** *"if SAEs will eventually be useful for
  these ambitious tasks, they should enable us to do something new today"* — i.e.
  something *"we cannot currently easily do."* This is a satisfied-when that
  names the competitor, not just a threshold.
- **Baselines were resourced, and won:** *"Dense linear probes perform nearly
  perfectly, including out of distribution"* (AUROC 1.0/1.0/0.999). The writing
  advice is a resourcing instruction: on baselines, *"put meaningful effort into
  making them good."*
- **They argued against their own result:** *"this slightly stacks the deck
  against SAEs, since without SAE-based debugging, the linear probes may have
  latched onto these spurious correlations."* Steelmanning the loser costs
  nothing and is what makes a negative durable.
- **The conclusion is calibrated to the evidence:** *"we do not think that SAEs
  are useless… SAEs and SAE based techniques are not likely to be a gamechanger
  any time soon."* And the epistemic limit is stated: *"it's extremely hard to
  distinguish between fundamental issues and fixable issues."*

Progress Update #1 states the publishing norm plainly — negatives ship as short
numbered posts *"to help avoid wasted effort."*

The prescribed record set from the process sequence is three artifacts, and none
need infrastructure: a **highlights doc** ("this makes it easier to spot
connections"), a **research log** ("Ask why things worked or failed. Was it luck,
execution, or a fundamental judgment call?"), and a **weekly review** ("What
worked? What didn't? What surprised me? What would I do differently?").

### 18.3 Marin: the open lab notebook, as three separate artifacts

Marin runs an entire lab's process in public, and it separates three things this
repo currently conflates:

1. **A GitHub issue is the preregistration.** `docs/explanations/guidelines.md:78-101`
   requires an issue to state *"A **hypothesis or goal**, which is an a priori
   prediction of what the outcomes will be […] This serves as a
   [preregistration]"*. Experiment files are named `exp<ISSUE#>_<descriptor>.py`,
   so the code points back at the hypothesis.
2. **`docs/reports/index.md` is the conclusions index** — one line per
   experiment, ending in `Conclusion:`, with negatives in **exactly the same
   format as positives**: *"MuP for scaling laws […] Conclusion: not worth it
   compared to our heuristic version."* / three consecutive corpus ablations each
   *"Conclusion: No major improvement compared to control."* / *"NOTE: this seems
   like a loose end, we should pursue this further."*
3. **`docs/debug-log-*.md` is the incident log**, with a fixed template:
   `## Initial status` (real error strings, exit codes) → `## Hypothesis 1` →
   `## Changes to make` → `## Results` → `## Production validation`. One ends
   honestly at *"Pending production validation."* Another states a null plainly:
   *"The 0.6-billion-parameter smoke model solved neither task, so the recorded
   accuracy and mean reward are both 0.0. This validation measures plumbing
   rather than model quality."*

**`docs/reports/grug-archive.md` is the direct fix for an empty
`docs/learning/learning_log.md`.** It is a registry of *deleted* experiment code —
principle at `:6-9`, *"Prefer deletion over long-term maintenance of stale
experiment code"* — with a per-entry template: id, Path, Origin, Introduced
(SHA), **Last known-good (SHA)**, Status (`active|superseded|deleted`), Purpose,
Superseded by, Diff, Issue. Delete the code; keep the coordinates.

Their retrospectives set the honesty bar: *"We determined the root cause to be
contamination. That is, we (accidentally) cheated but we cheated badly"*
(`marin-32b-retro.md:340-353`), immediately followed by a scoping caveat — *"we
have no reason to believe [MATH] was contaminated. So, our generally poor
performance on math may have other causes."* Failed runs are carried in the
token-accounting table marked `† excluded from cumulative token totals`. And
`marin-8b-retro.md:422-460` documents a debugging dead end in full, ending
*"Nothing solved the problem. **What?**"*

One more habit: **issue numbers are cited in source as evidence.**
`datakit/decon.py:203-209` justifies a design choice with *"See PR #5656 for the
smoke-test finding (~18% phantom contamination on MMLU vs nemotron-math came from
the literal `"..."` short-paragraph artifact)."* 38 such references in the
library, 125 in the docs.

### 18.4 The chronicles: OPT and BigScience

**OPT states its own spec** (logbook p.1): *"To provide a source of truth of what
we did, when, and why… Add a dated entry for each log… For all launches, include:
Date… Context of why changes were necessary (Analysis of previous run)…
Checkpoint/log folder, Relevant commits, PR of a change to sweep script if
relevant."* Entry header: `DATE TIME TZ [Name] - title`. Entries are 3–15
bullets.

The structure of a launch entry is a three-part template, and it is the artifact
to copy: **Analysis of run N → Decisions for run N+1 → Launch steps for N+1**,
where "launch steps" pastes the literal `git checkout <sha>`, `RESTORE_FILE=`,
and full command.

**The single highest-value practice in either logbook is back-annotation.** When
the "fake SGD debacle" (2021-12-02) revealed that a beta1 bug invalidated a run
family, they wrote *"NOTE FOR FUTURE: BECAUSE OF THE BETA1 BUG […] any ablations
with 12.36/12.37 are no longer valid"* — **and edited a
`[WARNING: See 2021-12-02 17:16 ET: Debrief]` banner into every affected earlier
entry.** The invalidated results are marked *in place*, not only in the
postmortem. This matters more solo than in a team: no colleague will remember for
you. It is the same move as Anthropic's retraction banners, and it is what
`docs/registry.md` had to be built retroactively to do.

**BigScience's `train/lessons-learned.md` proves how low the bar can be.** Its
most valuable section, *"What was tried and it didn't work"*, is six bare lines
with no elaboration:

> - changing seed - the problem usually would just shift elsewhere…
> - a more numerically stable self-attention version…
> - lowering `beta2` to 0.95 (from 0.999)
> - changing width/depth ratio
> - longer lr warmup
> - tried Curriculum Learning

Five minutes of writing, and it is the most reused content in that repo. Its
positives carry the number: *"Setting `--init-method-std` to
`sqrt(2/(NHIDDEN*5))` has made a huge difference to the training stability."*
And the retrospective states its own limit: *"It's hard to tell if there is one
specific improvement that made the biggest impact w/o doing ablation studies."*

Four more conventions worth the keystrokes:

- **Record the artifacts bracketing every anomaly.** BigScience saved
  `checkpoints/spikes/global_step31200` and `global_step31259` around a loss
  spike. Small-run analogue: keep the exact inputs, seed, and checkpoint on
  either side of any surprising number.
- **Write "we don't know" when you don't.** *"So at the moment we don't know what
  happened."* This is exactly `CLAUDE.md` rule 7 — the alternative is inventing
  "transient CUDA faults" for a deterministic cholesky bug.
- **A pre-registered Plan B with a date.** OPT: *"We decided to follow through
  with our 'plan B' that we set for ourselves on October 18 before starting any
  of these runs"* — the abort decision was made calm, not at hour 60.
- **`train/sanity-checks.md`** — a pre-submit checklist of divisibility
  constraints and "what can't change on resume." Cheaper than one wasted queue
  slot.

And one methodological lesson from OPT's `10_percent_update.md` that generalizes
past training: *"Optimizing for the lowest loss/ppl early in training does not
seem to necessarily guarantee a uniform shift in the loss curve."*

### 18.5 Two more mechanisms, and one damning negative

**Pythia's errata section** is the model for correcting a released artifact: a
dated `## Changelog` (newest first), each entry naming what was wrong — *"released
a new version of all Pythia models, fixing various inconsistencies… The old
models ('v0') remain available"*; *"due to a typo one of our models was smaller
than we thought"* — plus a dedicated `#### Erata` section linking the 6.9B/12B
initialization bug to the exact config omission you can verify in
`models/12B/pythia-12b.yml`. Superseded results are **kept**, not deleted.

**OLMo-core's `CHANGELOG.md`** is Keep-a-Changelog form but with entries that
carry the *reason*, not just the change. And its one real mid-run decision
record is a comment inside a released config
(`official/OLMo3/OLMo-3-1025-7B-pretrain-1.py:100-102`):
`max_duration=Duration.tokens(int(5e12))  # Originally scheduled for 5T` /
`hard_stop=...  # But at this step we decided to extend schedule to 7T. See
...pretrain-2.py`.

**The negative: LLM360's amber-train — the flagship "we release the messy
middle" project — ships a repo with no failure record at all.** Grepping the
whole repository for nan / instability / restart / divergence / spike returns
**zero hits**. There is one commit, no issues, no notes file. The narrative of
what went wrong exists only in arXiv:2312.06550. This is the strongest empirical
argument for keeping the chronicle *in the repo*: the project that made
transparency its thesis still lost the process record.

### 18.6 The minimal version for this repo

Per `experiments/<name>/notes.md`, append-only, dated `YYYY-MM-DD`, **one entry
per config change — not per job**, four lines each:

```
## 2026-08-06
What the last run showed:   <one sentence, with the number>
What I'm changing and why:  <one sentence>
Provenance:                 <git sha> · <config path> · seed <N>
Verdict vs satisfied-when:  <met / not met / undecided>
```

Plus, repo-level:

- `docs/learning/learning_log.md` — a **"What was tried and it didn't work"** list you are
  allowed to add one bare line to. BigScience's format. No elaboration required.
- `docs/registry.md` — already the back-annotation surface. When a result is
  invalidated, **edit the original document**, do not only append.
- A `sanity-checks.md` pre-submit list for SLURM: what must divide what, what
  cannot change on resume, which caches must be warm before the node goes
  offline.

## 19. What we are deliberately not adopting

Each of these is real practice at a serious lab and wrong here. The reason
matters more than the verdict.

1. **Experiment launchers (xmanager, Beaker, Ray/`fray`).** xmanager's hard
   dependency is a container-scheduling control plane, not merely "GCP": a
   project, `gcloud auth`, `configure-docker`, a GCS bucket, and an image build
   per run. There is **no SLURM executor**, and adding one means implementing the
   `Executor`/`ExecutableSpec`/handle triad. Its entire abstraction stack exists
   to hide four backends; with one backend the load-bearing content is ~15
   lines — `itertools.product`, a dict of args, and its `identity` idea
   (`xm/core.py:986-988`: relaunching with the same identity is a no-op). The
   SLURM substitute already exists: `sbatch --dependency=afterok:`, `--array`,
   and `sacct -j <id> --json` for exit code, elapsed, and MaxRSS.
2. **Content-addressed (config-hash) output directories.** See §15.4 — marin
   built this and removed it. Use `name/version` paths with an advisory
   fingerprint-drift warning.
3. **Bitwise determinism as a goal.** levanter achieves it, and the price is
   visible: a whole random-access tokenized-cache subsystem built on TensorStore
   and four Ray actor types, an admitted latency hit, worse shuffle quality
   ("era shuffling… is not as good as full shuffling"), and one capability given
   up entirely ("I believe it's impossible to sample without replacement and have
   random access"). In PyTorch+CUDA you would additionally need
   `use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` and still
   lose it to any atomics-based kernel. **Adopt the cheap 80%** — data order as a
   pure function of `(seed, index)`, and global batch size in config so device
   count only changes microbatch count — and drop the guarantee. Note levanter's
   own scope: *"the same run with the same code on the same hardware
   configuration"*, and resuming on a different host count *"breaks
   reproducibility for now."*
4. **Multi-OS × multi-Python CI matrices and oldest-dependency jobs** (litgpt
   runs 4 OSes × Python 3.10–3.13 plus a minimum-pins job). This repo has one
   Python, one torch, one driver. Pin them and test that.
5. **GPU CI.** OLMo, OLMo-core, and nanotron dispatch tests to real GPU runners;
   levanter and marin to TPU VMs. There is no solo equivalent. Substitute: a
   `make test-gpu` you run manually before writing a claim, and a line in
   `notes.md` recording that you ran it.
6. **W&B (or any tracker) as required infrastructure.** Compute nodes have no
   internet. See §14.2 for four projects that hard-fail without a network. Write
   files first; make the tracker an optional, fail-soft add-on.
7. **Sharded / distributed checkpoint formats** (nanotron's rank-encoded
   filenames, DCP, ZeRO shard reconstruction, `merge20b.py`) and the base64
   config broadcast. All exist for multi-node topology. Single A100: one file.
8. **DVC / dataset-versioning tooling.** On a pre-cache-then-run-offline cluster
   the whole requirement is a sha256 manifest and a checker — pythia's
   `utils/shard_hashes.txt` + `checksum_shards.py`, about 40 lines.
9. **Hydra-style deep config composition.** gpt-neox's merge actively *forbids*
   key layering; marin's real sweeps are `itertools.product`; OLMo-core dropped
   OmegaConf. Nobody in the survey is getting value from composition complexity
   at experiment scale.
10. **Elaborate config schema frameworks.** maxtext's `types.py` is 4,219 lines
    of Pydantic; only the ~40 lines of cross-field validators and the one-line
    `extra="forbid"` are load-bearing. Take those two.
11. **Reproducibility-checklist ceremony beyond the load-bearing items.** Only
    **34%** of NeurIPS 2019 reviewers found checklist answers useful; **8.15%**
    of ACL submissions gave identical answers to every item; the
    `releasing-research-code` evidence is a **GitHub-stars correlation**, not a
    reproduction measurement. Pineau et al.'s own verdict: *"we do not have
    concluding evidence that these processes indeed have an impact on the quality
    of the work."* Do the eight items in §19.1 instead of the form.
12. **Preregistration as a venue play.** Already covered in §10 — the NeurIPS
    Pre-registration Workshop produced 3 results papers and died. Keep it as
    internal discipline (marin's issue-as-hypothesis is the working form).
13. **Anthropic's publication style, taken whole.** They publish no confidence
    intervals, no error bars, no significance tests, and no preregistration
    (grep across Towards Monosemanticity, Scaling Monosemanticity, and Circuit
    Tracing returns zero hits for all of those terms), and they withhold model
    size and compute. Their argument in *Reflections on Qualitative Research* is
    serious — rigorously measuring a metric that may not track the property of
    interest is *"a kind of Cargo-Cult Science… It can seem very rigorous with
    lots of line plots with standard deviation bars"* — but it is an argument for
    **validating the metric**, not for dropping CIs, and it runs on institutional
    trust this repo does not have. Keep `CLAUDE.md`'s rigor floor; add their
    metric-validation habit on top.
14. **GDM's substitute for statistics.** Their robustness check is *breadth* —
    "we have checked our results transfer across a range of model sizes and
    depths, from GELU-1L to Pythia-2.8B." That is compute-bought and unavailable
    on one A100. **Buy robustness with seeds and CIs instead.** `CLAUDE.md`'s
    rigor floor is stricter than GDM's published practice and should not be
    relaxed toward it.
15. **Anthropic-scale interactive interfaces.** They argue explicitly that these
    are load-bearing, not luxury (*"just as early chemistry depended on custom
    glassware… so too does interpretability depend on data visualization"*) — and
    they staffed ~10 people across "Infrastructure, Tooling" and "Interface" for
    one paper. Adopt the cheap half: fast disposable plots during exploration,
    and one shared `plotting.py` (§10). Skip the interactive interface.
16. **Anthropic's and GDM's *public code* as a hygiene model.** Blunt finding:
    `circuit-tracer` is good *software* (typed, pyright-clean, 26 test files,
    autouse seed fixture, real tolerances, the Hessian invariant test) but it
    ships a method, not a study — no `experiments/`, no config files, no results
    directory. The actual research repos (`persona_vectors`,
    `introspection-mechanisms`) and every GDM research repo have **zero git-sha
    capture and zero library-version capture in any output**;
    `deepmind-research` is self-described *"illustrative code"* whose paper
    numbers come from downloading a checkpoint, not running the code;
    `anthropics/evals` gitignores `*.py`. Take testing patterns from
    circuit-tracer and chex. Take provenance from marin and litgpt.

### 19.1 The eight checklist items that are actually load-bearing

Everything else on NeurIPS/Pineau/ACL is ceremony for a solo researcher's own
rigor. These eight have evidence behind them:

1. **Dependency lockfile + the exact command per table row.** NeurIPS item 5
   requires *"the exact command and environment needed to run to reproduce the
   results."* This is where reproduction actually dies.
2. **Hyperparameters fully specified**, including the range considered and the
   selection method — not just the final values. Raff 2019: p = 8.45×10⁻⁶.
3. **Write for readability; prefer tables to prose for headline numbers;
   minimize equations per page.** Raff 2019: readability p = 9.68×10⁻²⁵ (the
   strongest paper-intrinsic factor), tables p = 0.010 positive, equations/page
   p = 0.004 **negative**, mediated by readability (p = 0.001).
4. **Releasing code does not discharge the obligation.** Raff 2019 reproduced 255
   papers *without looking at author code* and found code availability was **not
   a significant predictor of independent reproducibility (p = 0.213)**. What
   predicted it was author responsiveness — **85% success when authors replied
   vs 4% when they didn't**. Solo corollary: write down what you would have said
   in the email, because there is no future you to answer it.
5. **Exact number of training and evaluation runs, and the metric's precise
   definition.** Pineau v2.0. Free to write; catches the "config said n=200, the
   run used n=20" class of error directly.
6. **Central tendency + variation, with the calculation method named.** NeurIPS
   item 7: *"It should be clear whether the error bar is the standard deviation
   or the standard error of the mean"* and *"It is OK to report 1-sigma error
   bars, but one should state it."* **Treat "N/A" as forbidden** — 36% of NeurIPS
   2019 authors used it, and that is the checklist's biggest measured hole.
7. **Compute disclosure including unreported runs.** NeurIPS item 8 asks whether
   *"the full research project required more compute than experiments
   reported."* This is the anti-cherry-picking clause and costs one sentence.
8. **Tool and metric implementation versions.** ACL C4 — *"the version number or
   reference to specific implementation."* Unique to ACL, absent from NeurIPS,
   and mandatory for MT work: sacreBLEU signature, COMET model version,
   chrF++ variant. ACL B5 (*"report the language of any language data"*) belongs
   with it.

Two further findings worth keeping in mind. **Gundersen & Kjensmo** surveyed 400
AI papers and found strict R1/R2/R3 reproducibility scores of **0.00 for all
papers**; a seeded rerun of your own code is **R1 and nothing more** — say so.
And **Kapoor & Narayanan** document leakage across 329 papers in 17 fields; their
remedy, a written argument per leakage type (L1.1–L3.3) that it does not apply,
is the cheap version and directly relevant to probing work where train/test
sentences share speakers, documents, or languages.

---

## 20. How this repo develops — turning the standards into mechanisms

§§1–19 describe what good practice is. This section is about the gap between a
standard and an enforced standard, because that gap is this repository's
documented failure mode: **every entry in `docs/registry.md` under "Process
failures worth not repeating" is a rule that existed as prose with nothing
enforcing it.** `docs/learning/learning_log.md` was created for dead ends and
has zero entries. Two questions sat in tables as experiments while their
runners raised `NotImplementedError`. Q5's config says COMET / n=200 /
σ∈{1e-3,1e-2,5e-2}; the run used chrF++ / n=20 / σ=0.1. None of these were
failures of intent. They were failures of mechanism.

### 20.1 The sorting rule

For every standard in this document, ask one question: **must this happen every
time, with zero exceptions?**

| Answer | Where it belongs | Why |
| --- | --- | --- |
| Yes, and it is checkable by a program | A test, or a hook | Deterministic. Anthropic's own framing: *"Unlike CLAUDE.md instructions which are advisory, hooks are deterministic and guarantee the action happens."* |
| Yes, but it needs judgment | A skill (`.claude/skills/`) | Loads on demand, walks the workflow, keeps the reasoning next to the step |
| No — it is context, not a rule | `docs/`, linked from CLAUDE.md | Reference material. Not every session needs it |
| It is already done correctly without being told | Delete it | A rule that changes no behaviour is noise that dilutes the rules that do |

That last row is the one people skip. The pruning test from the same source:
*"For each line, ask: would removing this cause Claude to make mistakes? If not,
cut it."* A bloated CLAUDE.md is **why** rules get ignored — the important ones
get lost among the ones that were never load-bearing. `CLAUDE.md` here is around
150 lines, which is at the edge. Every addition should displace something.

The corollary matters more than the rule: **when a standard is violated twice,
that is evidence it belongs one row higher in the table**, not evidence that it
needs restating more firmly. Restating a prose rule that already failed twice is
the definition of doing the same thing again.

### 20.2 Fitness functions, and what they govern here

A *fitness function* (Ford, Parsons & Kua, **Building Evolutionary
Architectures**, 2nd ed.) is an automated, objective test that the system still
satisfies a stated design goal, run continuously rather than at review time.
The standard examples govern module coupling and layering.

**That is the wrong dimension for this repo.** The 2026-08 audit found the
measurements sound and the *claims about them* wrong. The dimension worth
governing here is not code structure — it is:

1. **Traceability.** Does every result carry the commit, config, versions and
   seed that produced it?
2. **Spec compliance.** Does every experiment have a falsifiable satisfied-when
   that predates its runner, and a recorded verdict against it?
3. **Claim integrity.** Does every comparative sentence name the baseline it
   compares against, and state its coverage?
4. **Honest absence.** Does anything appear as done that did not run?

A linter that checks line length governs none of these. Checks that govern
these four are worth writing; nothing else is.

**Do not build them before the code they govern exists.** A gate written
against an empty track is tuned against nothing, and a gate whose failures are
all pre-existing debt is a gate that gets switched off within a week.

### 20.3 The trigger table — what to build, and when

Each row fires when the trigger becomes true, not before.

| Trigger | Build | Source for the design |
| --- | --- | --- |
| First runner writes to a results directory | `write_manifest()` — git sha, **tree hash via `git stash create`**, dirty flag, resolved config *contents*, seed, versions, SLURM job id | §13.3; marin `rigging/provenance.py:45-67` |
| Second committed config exists | A test that instantiates **every** committed config | §12; gpt-neox `tests/neox_args/test_neoxargs_load.py` — cheapest test in the survey |
| First config gets a CLI override | Record the **resolved** config, not the file | §13; litgpt `save_hyperparameters`, maxtext double-override detection |
| First training run launches | Checkpoint schedule as data (`CheckpointInterval(every, until)` list); data order a pure function of `(seed, global_index)`; a chronicle file | §15; levanter PRP, open_lm resume assert, the amber-train bug |
| First similarity / probe / SAE measure implemented | Closed-form invariant tests (`CKA(X,X)==1`, rotation and scale invariance, effective rank of identity) **and the step-0 random-init null** | §17.2 — across ~3,000 test functions in 14 flagship repos, **zero** test any interpretability measure. ~80 lines puts this repo above all of them |
| First findings document written | Claim linter for the greppable rules in §20.4 | §18; the 2026-08 audit |
| Any two of the above exist | Two hooks, and only two: refuse `sbatch` from a dirty tree; run the checks on `Stop` | §13 (OLMo-core `beaker.py:591` dirty refusal); Claude Code hooks reference |
| First long training run | Chronicle: timestamped prose keyed to step numbers — instabilities, restarts, fixes | §18.4; OPT and BigScience |

The `Stop` hook is the one that closes the agentic loop: *"a Stop hook runs your
check as a script and blocks the turn from ending until it passes."* Until
there are checks worth running, it has nothing to gate.

### 20.4 Which claim-hygiene rules can be mechanised

CLAUDE.md lists eight. They are not equally checkable, and pretending otherwise
produces a linter nobody trusts.

**Greppable — worth a linter once findings documents exist:**

| Rule | Pattern | Note |
| --- | --- | --- |
| 2 — unmeasured baseline | `worse than (not\|un)-(quantizing\|pruning\|…)` | The audited phrase shipped in three documents |
| 8 — hedging | `the mechanism (is\|behind\|by which)`; `proves`, `confirms that` | Cheap, high precision |
| 3 — coverage unstated | `across all`, `every model`, `universally` without a nearby `N of M` or `n=` | Some false positives; they cost seconds |
| 1 — comparison without baseline | `recovers/closes the <gap\|cliff>` without `than`, `vs`, `compared to` | Catches the exact audited failure shape |
| 7 — failure explained away | `transient/flaky/spurious` near `CUDA/OOM/error` | Forces the log to be read |

**Not greppable — these need `/audit-claim` and a human, and no linter should
pretend to cover them:**

- **Rule 4** (metric substitution) — requires knowing which quantity a column
  actually holds. Only a reader who opens the runner can tell.
- **Rule 5** (dropped data point) — requires knowing the intended denominator.
- **Rule 6** (config ≠ what ran) — **not a text problem at all.** This is solved
  by the manifest, mechanically, and cannot be solved by reading prose. It is
  the strongest argument in this document for provenance-first.

Scope any linter to documents that make *first-person claims about our own
results* — `<track>/docs/`, `docs/registry.md`, `paper/`, experiment READMEs.
A literature summary saying "across all models" is reporting someone else's
claim, and flagging it trains you to ignore the tool.

### 20.5 Conventions that hold from the first line of code

These need no trigger. They are true as soon as any code exists, and an agent
should apply them without being asked:

- **Spec before runner.** `README.md` with `## Satisfied when` exists before
  `experiment.py`. Use `/new-experiment`; it refuses to reorder these.
- **One YAML per run**, seed declared in every one, no hidden defaults in code
  that override config.
- **`from tools.provenance import write_manifest`** on the first line of every
  output path. Retrofitting provenance is how you get results you cannot cite.
- **Never write a number into a document without the path to the results
  directory that produced it.**
- **A verdict is recorded against every satisfied-when, including "not met."**
- **Back-annotate.** When a result supersedes an earlier one, edit the earlier
  document in place with a dated banner. Append-only correction leaves the wrong
  number as the one that gets read — this repo has already had a findings doc
  and a paper disagreeing. Use `/close-experiment`.
- **A negative gets filed the day it happens**, in `docs/registry.md` under
  Ruled out with the numbers that killed it, plus one dated line in
  `docs/learning/learning_log.md`.

### 20.6 Decisions

Design decisions made before code exists are the ones whose reasoning is lost
fastest, because nothing in the repository records them. The interlingua
decisions in CLAUDE.md — the training substrate, `transformer-lens==3.6.0`, WSD
over cosine, the checkpoint grid — are being made now and each has a real
alternative that was rejected for a reason.

A record also has to be honest about **who** decided. `docs/decisions/0001` was
first written asserting a decoder-only substrate and marked `accepted`; the
proposal in fact specifies three architecture arms and makes the architecture
contrast a hypothesis (H3a), and nobody had accepted anything. **A record stays
`proposed` until the person who owns the decision has made it** — marking it
`accepted` because it reads as settled is how a guess acquires authority it
never earned. The failure mode is the same one §20.4 is about: a claim drifting
from its evidence, here into a document that future work will treat as
foundational.

Use a decision record per choice in `docs/decisions/NNNN_slug.md`, Nygard
format: **Context / Decision / Status / Consequences**. Three properties make
them worth the five minutes:

- They are **immutable**. A superseded decision gets `Status: superseded by
  0007`, never an edit. The reasoning that was true at the time stays readable.
- They record **what was rejected and why**, which is the part you will want
  when the constraint changes.
- They are **cheap enough to actually write** — under a page, or they stop
  happening.

Write one when a choice constrains future work, is expensive to reverse, or was
non-obvious enough that you will re-litigate it in three months. Not for
choices with an obvious default.

---

*§20 compiled 2026-08-06 from the Claude Code hooks and best-practices
references, GitHub spec-kit's spec-driven methodology, and Ford, Parsons & Kua
on fitness functions, applied to the failure record in `docs/registry.md`. The
trigger table's right-hand column cites §§12–19 of this document; the
mechanisms themselves are unbuilt by design and should be built when their
trigger fires.*

---

*§§1–11 compiled from a live sweep on 2026-08-05. Some page extractions ran
through a summarizer — verify before quoting verbatim: the BIG-Bench counts in
Schaeffer et al., MultiBERTs' exact wording, and Hewitt & Liang's control-task
accuracies.*

*§§12–19 compiled 2026-08-06 by fifteen parallel agents reading primary sources.
Repository citations are file:line against the default branch as of that date and
were read from local clones. Quotations from transformer-circuits.pub and the GDM
progress updates were extracted via a summarizing fetcher — short quotes are
verbatim-probable, but re-check before quoting in a paper. Not verified: the
"200 Concrete Open Problems" post (no claim here rests on it), penzai's use by
the GDM interpretability team, and the ICSE artifact-reproduction percentages,
which came second-hand.*
