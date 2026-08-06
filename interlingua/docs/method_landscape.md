# Method and tooling landscape — Tier 1

**Compiled 2026-08-06** by a parallel literature sweep (nine area surveys + three
discovery passes). This is a **map of what exists**, not a plan and not a
recommendation. It does not choose an architecture, design an experiment, or
rank the directions. Where it contradicts [`tier1_plan.md`](tier1_plan.md), the
contradiction is stated and sourced.

**How to read the verification flags.** Most agents exhausted their web-search
budget partway through and finished by fetching primary sources directly. That
makes this document **stronger as an audit of named methods than as an
exhaustive sweep**. Flags:

- **[verified]** — a primary source (paper text, GitHub API, PyPI JSON, wheel
  contents) was opened in the sweep.
- **[discovery]** — surfaced in a search listing only; ID and venue are as they
  appeared. Not opened.
- **[UNVERIFIED]** — existence, ID, or attribution could not be confirmed. Do
  not cite without checking.

Nothing here should enter a findings doc, the registry, or a paper without the
confirmation pass its flag implies. Per `CLAUDE.md` claim hygiene, a citation
that has not been opened is not evidence.

---

## 0. The measurement regime, which determines most of what follows

Stated once, because it silently governs every entry below.

| Quantity | Value | Consequence |
|---|---|---|
| `n` paired samples | **1012** (FLORES-200 devtest) | The binding constraint |
| `d` model width | **512** | |
| **d/n** | **≈ 0.51** | Worst case for every spectral estimator |
| Analyses per measure | ~900 (60 ckpt × 15 runs), ×6 layers ×3 pairs ≈ 16,200 cells | |
| With a K=200 permutation null | ≈ 3.24M metric evaluations | Affordable **only** if re-evaluable in O(n²) from cached Grams |
| 1 forward pass, batch 32×512 | ~1.2e12 FLOPs ≈ **10–30 ms** on an A100 | Launch-latency bound, not FLOP bound |
| ⇒ a measure costing 100 FP/checkpoint | **~1 GPU-hour across the whole study** | Almost everything cheap is *free* |

Three consequences that reorganise the landscape:

1. **d/n ≈ 0.5 excludes the CCA family outright.** Canonical correlations are
   identically 1 when n ≤ d and severely inflated for n ≲ 10d. At n ≈ 2d,
   SVCCA/PWCCA numbers are determined by the truncation parameter. This is a
   stronger objection than the usual specificity critique. [verified]
2. **Sentence-level pooling caps n at 1012 and you cannot buy your way out**
   without more parallel *sentences* (FLORES dev adds ~997; NTREX-128; Tatoeba).
   Raising n to ~5000 does more for every measure than any choice among them.
3. **The permutation null is therefore cheap.** For CKA, RSA, distance
   correlation and mutual-kNN, a permutation acts on a *cached* matrix in O(n²),
   so calibrating the entire grid costs **minutes**, not 3.24M evaluations.
   Procrustes needs a fresh 512×512 SVD per permutation: a few GPU-hours. GW
   needs a full re-solve: infeasible. **There is no cost argument for shipping
   uncalibrated numbers.** [verified]

---

## 1. Measures of cross-lingual alignment, computable per checkpoint

### 1.1 The paired-representation family

Cost assumes cached, centered Gram matrices. "Perm-cheap" = the null can be
evaluated without recomputing from raw features.

| Measure | Invariance | Cost | Perm-cheap | Calibrated null | Note |
|---|---|---|---|---|---|
| Linear CKA (naive) | orthogonal, isotropic scale, translation | O(n²d) | yes, O(n²) | **no** — estimator → 1 as d/n grows | Excluded at d/n=0.5 |
| Debiased / unbiased-HSIC CKA | same | same | yes | E[HSIC₁]=0 under independence; the *ratio* is still biased | Use, still calibrate |
| Kernel (RBF) CKA | same | + bandwidth | yes | as above + a free parameter | ≈ linear CKA empirically |
| CCA / SVCCA / PWCCA | invertible linear (too much) | O(nd²+d³) | partial | useless at n≈2d | **Excluded by the regime** |
| Orthogonal Procrustes | orthogonal (+scale, +translation if centered) | O(nd²+d³) | moderate | permutation works; no closed form | Best sensitivity/specificity in Ding et al.; a true metric, so the 900 checkpoints can be embedded by MDS |
| RSA (RDM correlation) | depends on the RDM (see §1.3) | O(n²d)+O(n² log n) | **yes** | **best-developed inference in the field** | `rsatoolbox`, actively maintained |
| Mutual k-NN (m_NN) | any distance-preserving map per set | O(n²d)+O(nk) | yes, ~free | **closed form: hypergeometric.** E = k/(n−1) = **0.0099** at k=10 | The only analytic null |
| CKNNA | orthogonal, scale; local | O(n²d) | yes | permutation | Survives calibration in Gröger et al. |
| **Distance correlation** | orthogonal, isotropic scale, translation | O(n²d)+O(n²) | yes | **dCor = 0 iff independent** — a genuine population null | Best-in-language-domain in ReSi; the highest value-per-line addition found |
| Gromov–Wasserstein | needs **no item correspondence** | O(n³)/iter, nonconvex | no | matching rate vs known pairing; random = 1/n = 0.001 | Beautiful null, infeasible at 16,200 cells |
| Soft matching (OT over neurons) | soft permutation | O(d³ log d) | no | none published | Most discriminative, too slow |
| Model stitching | **functional**, not representational | one trained layer per pair | no | random/untrained stitch | Arbitrates when the geometric measures disagree |
| ANC (avg neuron-wise correlation) | requires neuron-index correspondence | O(nd) | yes | Fisher-z | **Within-arm only** — cannot compare arms |

**Single-set (unpaired) measures** — these behave identically on all four
architecture arms, which makes them the natural first smoke test: effective
rank, RankMe, participation ratio, spectral entropy, condition number, IsoScore,
intrinsic dimension (TwoNN). All share one activation pass and one
eigendecomposition, so the marginal cost of computing them all is zero. **Their
nulls are unusually good**: at random init the spectrum follows
Marchenko–Pastur, giving closed-form expectations at d/n = 0.506, and step 0 is
a free empirical null. Drop Gini and Hoyer — no null, no interpretation for
representations. [verified]

### 1.2 What the empirical literature says these measures miss

- **CKA sees only the top ~10 principal components.** Ding, Denain & Steinhardt
  (NeurIPS 2021, arXiv:2108.01661): 97% of PCs had to be deleted before
  dissimilarity registered. If cross-lingual structure lives in a low-variance
  subspace, CKA is blind to it. [verified]
- **CKA can be moved arbitrarily without functional change.** Davari et al.
  (ICLR 2023, arXiv:2210.16156) give an explicit optimisation, plus extreme
  outlier sensitivity. [verified]
- **Input population structure confounds CKA and RSA.** Cui et al. (NeurIPS 2022,
  arXiv:2202.00095) show spuriously high similarity **for completely random
  networks** when the stimulus set has cluster structure. In a cross-lingual
  study **language identity *is* the cluster** — so any design that pools
  languages into one matrix is measuring the confound. Keep languages as separate
  X and Y with row correspondence. [verified]
- **Superposition deflates alignment measures.** arXiv:2604.00208 [discovery]
  gives closed-form results that superposition systematically deflates RSA,
  linear CKA and linear regression, and that **more-compressed systems can look
  more aligned.** A 36M model at 512d is squarely the superposition regime.
- **Similarity metrics inflate with scale.** The Aristotelian result
  (arXiv:2602.14486) shows a width confounder scaling as **O(d/n)** for spectral
  measures but only **O(k/n)** for neighbourhood measures — 0.506 vs 0.0099 here,
  a 50× difference. Whether the same inflation operates along a *training
  trajectory* is not established and is worth knowing. [verified]

### 1.3 The centroid confound — which measures are immune

Building on `research_standards.md` §5. The operative property is **translation
invariance**: does adding a constant vector to every point in set Y change the
measure?

| Immune (translation-invariant) | Confounded |
|---|---|
| Linear/kernel CKA (the "C" is column-centering) | Raw cosine between paired sentence vectors |
| RSA with a **Euclidean** RDM | RSA with a **cosine/correlation** RDM |
| Mutual k-NN with **Euclidean** distance | Mutual k-NN with **cosine on L2-normalised** features — check what `platonic-rep` does before reuse |
| Procrustes **with** centering | Procrustes without centering |
| Distance correlation, GW, IsoScore | Effective rank on **uncentered** features (the centroid is a rank-1 direction that dominates the spectrum) |

Two things centering does **not** fix: the two languages have different
*covariance*, not just different means; and pooling reintroduces Cui et al.'s
confound. **The only thing controlling for both is the pairing permutation null**
— it preserves each language's centroid, covariance, anisotropy and token
statistics exactly, and destroys only translation-equivalence. [verified; the
per-measure invariance table is derivation from definitions, not a cited result]

### 1.4 The calibration procedure

Gröger, Wen & Brbić, ICML 2026 (arXiv:2602.14486), code `mlbio-epfl/aristotelian`,
`pip install calibrated-similarity` (v0.1.1, last push 2026-06-24). [verified]

- **Permute** the rows of Y — here, the EN↔FR sentence pairing.
- K = 200; p = (1 + #{s⁽ᵏ⁾ ≥ s_obs}) / (K+1), super-uniform under H₀.
- **Depth correction (the part usually skipped):** apply the *same* π to all
  layers, take τ from the distribution of the aggregate — the classical maxT
  procedure. Their depth confounder scales as **√log M**; at 6×6 = 36 layer
  pairs that is ≈1.89, a real inflation that would otherwise be reported as a
  finding.
- **Their headline:** after calibration, linear CKA, RBF CKA, SVCCA, RV,
  Procrustes and RSA **lose** the convergence trend; m_NN, cycle-kNN and CKNNA
  **retain** it. Calibrated CKA ≈ debiased CKA, an independent validation.
- ⚠️ With K=200 the smallest p is 1/201 ≈ 0.005. Per-cell significance after
  correction over 36 layer pairs needs K ≥ ~7,200 — still affordable for the
  O(n²) measures.

**Counterweight:** *Back into Plato's Cave* (arXiv:2604.18572) [verified] shows
m_NN alignment measured on ~1K samples degrades substantially at millions, and
what survives is coarse semantic overlap. Reconciliation: with n fixed at 1012
across all checkpoints, **within-study trends are valid; absolute values are not
portable.** Write n into the metric name.

### 1.5 Cross-lingual-specific measures (not generic similarity)

| Measure | What it needs | Arch. | Null | Note |
|---|---|---|---|---|
| Parallel-sentence retrieval (top-1, CSLS/margin) | parallel corpus | sentence-level pooling; awkward on decoder-only | 1/N, plus step 0 | ⚠️ **Tatoeba-36 is X↔English only — no FR↔TR. BUCC18 has no Turkish at all.** FLORES is the only resource covering FR↔TR [verified] |
| Language-centroid decomposition (Libovický et al., Findings EMNLP 2020) | parallel sentences | any | step 0 | Converts to a trajectory for free: track ‖centroid‖ / total variance, and the retrieval before/after **gap** is the confound size |
| Language-ID probing as a *negative* measure | language labels | any | ⚠️ **step 0 is likely at ceiling** — FR/TR subwords barely overlap, so a probe on a random-init model reads the embedding table | Almost no dynamic range unless computed on the centroid-removed residual. 900 probe fits — most expensive per unit information |
| LAPE / language-specific neurons (Tang et al. ACL 2024) | per-language corpus | FFN-based, all arms | — | ⚠️ Developed at multi-billion scale. At 512d/2048 FFN, superposition pressure is far higher and "language-specific neuron" may not be a well-defined object |
| Cross-lingual neuron overlap across checkpoints (Wang, Minervini & Ponti, ACL Findings 2024) | as above | — | none reported | The closest methodological precedent for a per-checkpoint cross-lingual measure. ⚠️ Reports **non-monotonic** dynamics: "degradation … in certain phases" |
| Structural-probe **transfer** (Chi et al. ACL 2020) | UD treebanks | encoder-developed; causal masking changes what a decoder can recover | random-init probe + control tasks + **right-branching baseline, which is language-dependent** (respectable in EN, near-zero in verb-final TR) | Refit per checkpoint per layer — dominant compute item |
| Anisotropy / cluster separability | — | any | — | ⚠️ Confounded: Timkey & van Schijndel (EMNLP 2021) show a handful of rogue dimensions dominate cosine similarity |
| ABX minimal-pair discrimination (de Seyssel et al., EMNLP 2025, arXiv:2505.17747) | parallel sentences | **encoder-native, training-free** | chance = 0.5 | Separates **form** (LD) from **meaning** (MD); LD ⟂ MD at ρ = −0.74; validated against retrieval at r = 0.77. ⚠️ Inside the paper: **MD predicts neither POS nor NLI** |

**A confound in the plan's Stage 2, flagged by the sweep as its own reasoning
rather than a citation.** Cross-lingual within-class JSD compares
`P(x | EN prefix)` against `P(x | FR prefix)` over a shared BPE vocabulary. Early
in training the model has not learned language identity, so these are similar →
low JSD. As training proceeds the model learns to emit French subwords after
French context → **JSD rises toward log 2, driven by language identity, not class
structure.** The identified quantity is the **contrast** (between-class minus
within-class at fixed language pair), where the identity component largely
cancels. Both terms are in the plan, so the contrast is available; claims about
the *level* are not supported.

### 1.6 The alignment→behaviour link is contested

Five independent results say cross-lingual alignment measures do **not** predict
behavioural capability:

- **arXiv:2506.16678** *Mechanisms vs. Outcomes: Probing for Syntax Fails to
  Explain Performance on Targeted Syntactic Evaluations* — across **32 models, no
  probe yielded a significant regression fit** against downstream syntactic
  accuracy. [discovery] This is a published negative aimed at exactly the H1
  comparison on exactly the syntactic targets the plan uses.
- **de Seyssel et al. (EMNLP 2025)** — MD has no significant effect on POS;
  neither ABX metric predicts NLI. [verified]
- **arXiv:2601.03168** — embedding similarity vs cross-lingual transfer on
  African languages; CKA ρ ≈ 0.1. [discovery]
- **arXiv:2505.21458** — latent-language consistency does not predict downstream
  performance. [discovery]
- **arXiv:2605.23315** *Convergence Without Understanding* — representational
  convergence without behavioural convergence. [discovery]

And one recent test of the affirmative case, one week old: **arXiv:2608.03446**,
*Predicting Multilingual Classification and Translation Performance of LLMs with
Cross-Lingual Alignment — Is English Enough?*, which surveys alignment measures
and tests whether they predict downstream performance in **encoder-only and
decoder-only** models. [discovery]

Counter-hypothesis worth knowing: **arXiv:2601.16390** finds cross-lingual
transfer works via *functional divergence*, with gains correlating with
**increased** language-cluster separation; and **arXiv:2605.23036** finds
effective steering occurs where alignment and separability *coexist*. "More
alignment ⇒ more transfer" is not the field's default in 2026. [discovery]

### 1.7 Measure families the plan never considered

- **Dynamical Similarity Analysis** (Ostrow et al., NeurIPS 2023) — compares
  systems by the *temporal structure of computation* (Koopman/Hankel delay
  embeddings) rather than representational geometry. arXiv:2410.24070 shows it
  detects structure **developing during training**. ⚠️ arXiv:2607.04493 (2026)
  argues DSA's orthogonal alignment is the wrong equivalence class;
  arXiv:2511.22828 is the speed fix. [discovery]
- **Common-vs-distinctive decomposition** — JIVE (arXiv:1102.4110), AJIVE, DIVAS
  (arXiv:2212.00703). Decomposes multiple blocks on the same subjects into
  **joint structure + block-specific structure + noise**. This is literally
  "interlingua + language-specific residual" as an *estimable decomposition with
  a significance test on the joint rank*, rather than a similarity score.
  arXiv:2501.09336 gives the theory on when shared-subspace estimation works at
  all. [discovery] **The single most structurally apt family found, and entirely
  absent from the ML alignment canon.**
- **Crossnobis (cross-validated Mahalanobis) distance** — an **unbiased**
  distance estimator whose null value is exactly zero rather than
  positive-biased. Fixes the standard problem that naive distances look nonzero
  on pure noise, i.e. at step 0. In `rsatoolbox`. [discovery]
- **Noise ceilings** (Nili et al.) — bounds on attainable similarity given
  measurement noise. Needed to say "alignment saturated" rather than "alignment
  stopped rising." [discovery]
- **Cross-lingual-native isomorphism measures** predating the CKA canon:
  Gromov–Hausdorff distance, Laplacian eigenvector similarity, relational
  similarity (arXiv:2001.11136, arXiv:2004.04070, IsoVec arXiv:2210.05098).
  [discovery]
- **Relative representations** (arXiv:2209.15430, arXiv:2605.30596) — anchor-based
  encodings invariant to latent isometry by construction, so comparability is
  built in rather than measured. [discovery]
- **Contrastive-difference CKA** (arXiv:2606.16897) — a CKA variant designed to
  compare **across architectures**, i.e. the H3a problem. [discovery]
- **Ordinal similarity indices (TSI/QSI)** (arXiv:2606.16379) — triplet/quadruplet
  ordinal consistency; claimed outlier-robust and cheap. [discovery]
- **Discriminative capacity of similarity measures** (arXiv:2509.04622) — scores
  measures by their ability to *separate model families* using d′ and ROC-AUC.
  A way to pre-screen which measure can distinguish four arms at all. [discovery]
- **Bures–Wasserstein / affine-invariant Riemannian metrics on covariance
  matrices** — handle rank-deficiency without regularisation. At 512d with modest
  batches the covariances will be singular; CKA absorbs this silently, these do
  not. [discovery]
- **MMD kernel two-sample tests** — distribution-level rather than geometry-level
  comparison, **with actual power theory** (arXiv:2109.14913). None of the
  measures in §1.1 come with a power analysis. [discovery]

### 1.8 Maintained implementations [verified 2026-08-06]

| Package | Repo | Last push | License | Note |
|---|---|---|---|---|
| `calibrated-similarity` | `mlbio-epfl/aristotelian` | 2026-06-24 | MIT | The calibration framework, on PyPI, with tests |
| `rsatoolbox` | `rsagroup/rsatoolbox` | 2026-07-31 | MIT | Only mature inference stack; crossnobis, noise ceilings, two-factor bootstrap |
| `POT` | `PythonOT/POT` | 2026-07-29 | MIT | GW backend |
| `repsim.measures` | `mklabunde/resi` | 2025-04-07 | CC-BY-4.0 | 24 measures, consistent implementations |
| `netrep` | `ahwillia/netrep` | 2024-07-30 | MIT | Shape metrics; **not on PyPI** |
| `platonic-rep` | `minyoungg/platonic-rep` | 2025-04-12 | **no license** | Blocker for reuse — reimplement |
| `pico-analyze` | `pico-lm/pico-analyze` | 2026-02-19 | Apache-2.0 | **Never loads a model** — see §3 |

⚠️ PyPI `repsim` (Kisung You) is a **different project** from Klabunde's `repsim`
module. Name the repo, not the module, in any config.

---

## 2. Mechanistic progress measures that survive cross-checkpoint comparison

### 2.1 The organising axis is not "refit-free"

The sweep's most useful correction. What kills per-checkpoint SAEs is
**non-identifiability of a non-convex overcomplete unsupervised decomposition**,
formalised in arXiv:2512.05534 (which covers SAEs, transcoders **and**
crosscoders as one piecewise-biconvex problem). That pathology does not transfer
to every "fit":

| Tier | Kind of fit | Comparable across checkpoints? | Examples |
|---|---|---|---|
| A | none (weights, or one forward pass) | **yes** | weight spectra, attention scores, logit lens, DLA, RankMe |
| B | **deterministic closed-form** — no optimizer, no seed | **yes** | J-lens Jacobian average, CKA, PWCCA, PCA, effective rank, LEACE |
| C | convex / near-convex supervised | mostly | linear probe (closed-form ridge), MDL probe |
| D | non-convex supervised | **no** — optimizer noise confounds with training stage | tuned lens, attention lens, future lens, edge-pruning masks |
| E | non-convex overcomplete unsupervised | **no** — non-identifiable | SAE, crosscoder, transcoder, CLT |

**Tier B is the load-bearing discovery.** Several measures the plan wrote off as
"requires a fit" are deterministic given the data and have no fitting noise at
all.

### 2.2 ⚠️ Correction to `tier1_plan.md` §2

The plan justifies dropping per-checkpoint SAEs with "cross-seed matched-feature
rates **as low as 1–4%** for large TopK dictionaries; feature reproducibility
21–30%." **The 1–4% figure is not in the cited paper.** Paulo & Belrose
(arXiv:2501.16615) report **30%** (Llama-3-8B, 131K latents) and **42%**
(Pythia-160M, 2^15 TopK); the abstract's only other quantitative claim is
directional. [verified — abstract fetched independently, full text read by the
survey agent] The direction of the argument holds; the number needs a source or
removal.

Adjacent, and more recent: **arXiv:2606.12138** *Unstable Features, Reproducible
Subspaces* — individual SAE features are seed-unstable but the spanned
**subspace** reproduces. That determines what may legitimately be averaged.
[discovery]

### 2.3 Crosscoders — EMERGING, and the closest prior art stops at 4 checkpoints

**"Crosscoding Through Time"** (Bayazit, Mueller & Bosselut; arXiv:2509.05291,
ACL 2026 `2026.acl-long.60`). PDF extracted and grepped: [verified]

- **4 checkpoints per model, never more than 3 inside one crosscoder.**
- 16,384 latents, **L1 with `l1_coeff = 2`**, 400M tokens, layer 8/12.
- Cost, verbatim: *"6 hours on an A100 80GB GPU for 2-way comparisons and 12
  hours for 3-way comparisons in 1B models."*
- 3 seeds trained, used **only** to average ℓ₀ / dead-feature counts. **No
  cross-seed latent-matching rate is reported anywhere.**
- Stated limitation: *"Our analysis is dependent on checkpoint selection"*; early
  checkpoints *"are particularly hard to interpret."*

⚠️ **The L1 artifact.** Minder, Dumas, Juang, Chugtai & Nanda (arXiv:2504.02922)
identify **Complete Shrinkage** — *"the L1 regularization term may force the norm
of the base decoder vector to be zero, even though it is present in the base
activation"* — and **Latent Decoupling**. On Gemma-2-2B, **18% of "chat-only"
latents fall inside the central 95% of the shared distribution.** BatchTopK shows
"minimal Complete Shrinkage." Translated: **on an L1 crosscoder, "this feature
appeared at checkpoint t" is confounded with "L1 zeroed the decoder for
checkpoints ≠ t."** Crosscoding Through Time cites this paper, thanks its first
author, and still trains L1 at λ=2. [verified]

**The better prior art**, missed by the plan: **arXiv:2509.17196** *Evolution of
Concepts in Language Model Pre-Training* (ICLR 2026) — **32 snapshots trained
jointly in one crosscoder**, snapshot-specific encoders/decoders with a shared
sparse code, **JumpReLU not L1**, 98,304 latents, 800M tokens, at **Pythia-160M**
— the smallest verified scale. Finds a statistical-learning phase then a
feature-learning phase with a turning point near **step 1,000**. Its stated cost
law: *"crosscoders require parameters that scale with the number of source
snapshots."* [verified]

**Cost arithmetic for this study** [survey agent's estimate, not a citation]: a
60-snapshot joint crosscoder at d=512 is ≈2×60×16384×512 ≈ **1.0B parameters —
larger than every model it describes combined** — at ~80 A100-hours per (arm,
seed), so ~1,600 A100-hours for 5 seeds × 4 arms.

**Practical alternative found in discovery: SAE-Track** (arXiv:2412.17626) —
warm-starts a continual SAE series across checkpoints instead of training each
independently. [discovery]

### 2.4 The lens family

| Lens | Tier | Cost | Null | Status |
|---|---|---|---|---|
| Logit lens | A | free | ⚠️ see below | MATURE |
| Tuned lens | **D** | ~1 A100-day for 900 | — | Method mature; **PyPI 0.2.0 dated 2023-07-18**; `wandb` still a mandatory dependency (blocker for offline nodes) |
| Attention lens | D | 48 heads × 900 = 43,200 trainings | — | Repo dead since 2024-08 |
| Future lens | D | probes per layer per offset | — | ONE-PAPER-OLD |
| DLA | A | free | — | ⚠️ direct path only; **self-repair** (Hydra effect, arXiv:2307.15771) means large DLA ≠ large counterfactual effect |
| **J-lens** | **B** | **~30 A100-hours for all 900** | **degenerates to the logit lens at step 0** | ONE-PAPER-OLD |

**Logit-lens failure modes are documented and land on this exact configuration.**
Belrose et al. (arXiv:2303.08112): it *"fails to elicit plausible predictions for
models like BLOOM and GPT-Neo"*; for **BLOOM-560M and OPT-125M the top-1
prediction is the *input token* in more than half the layers**; it is a biased
estimator (~4–5 bits KL on GPT-Neo-2.7B). ⚠️ And: *"Both the logit lens and the
tuned lens are designed primarily for the pre-LN architecture."* **BERT and
RoBERTa are post-LN.** [verified]

**J-lens** — arXiv:2607.15495 is titled ***Verbalizable Representations Form a
Global Workspace in Language Models*** (Gurnee et al., Anthropic, 16 Jul 2026),
not "the Jacobian lens"; J-lens is the technique inside it. [verified, fetched
directly] `J_ℓ = 𝔼[∂h_final,t′ / ∂h_ℓ,t]`, one d×d matrix per layer. It **is**
recomputed per checkpoint, but as a closed-form corpus average — no optimizer, no
seed — hence tier B. Cost is **linear in d_model**, so at 512 it is cheap: ~64
backward passes per prompt, ~100 prompts, ~1–3 min/checkpoint. Code
`anthropics/jacobian-lens` (`jlens` v0.1.0, Apache-2.0) but marked *"Reference
implementation. Not maintained."* ⚠️ Validated only on Claude-scale models; the
reference implementation is decoder-only and the `t′≥t` reduction is causal, so
encoder and encoder-decoder versions are ~200 lines you write and own.

**⚠️ Tied-embedding trap for the from-scratch setting** [survey agent's
derivation, not in the paper]: with tied embeddings at random init, `E·Eᵀx` peaks
at `x` by construction, so the step-0 logit lens is a copy-the-input detector. A
curve reading "copies input early, predicts later" would be substantially an
initialisation artifact. GPT-2 also ties and the lens works there, so tying is
not sufficient to cause it — but untying, or plotting the step-0 lens as the null
in every figure, resolves it.

**⚠️ The latent-language measure family is expected to return an uninformative
null on balanced data.** Wendler et al.'s measure detects pivoting to the
*training-dominant* language. On balanced EN/FR/TR there is none by construction,
and the published balanced-data precedent is BLOOM, where Wu et al. report *"we
do not see a clear dominating language in intermediate layers; … in most cases
they are symbols with no clear semantics."* Also: three phases resolved over 80
layers becomes 4 interior layers at 6L. What **does** transfer is **token
energy** — needs only `U` and hidden states, defined for a tied MLM head and for
the decoder side of an encoder-decoder, with a free analytic null (E² ≈ 1 for a
random direction). ⚠️ Absolute values are not comparable across models because
the denominator absorbs `U`'s anisotropy. [verified]

⚠️ **No published rebuttal of Wendler et al. exists** — ~200 citing papers
checked. Circulating "critique" text traces to an LLM-generated blog post and is
not citable.

### 2.5 Circuits and attribution graphs

- **circuit-tracer / CLTs — not viable here.** README, verbatim: *"Given a model
  **with pre-trained transcoders**, it finds the circuit."* The repo ships **no
  training code**. A transcoder is per-model hence per-checkpoint: 900 trainings,
  each on hundreds of millions of activation-tokens, producing tier-E objects
  that are not comparable anyway. [verified]
- **EAP / EAP-IG — cheap (2 forwards + 1 backward), but the comparison axis must
  change.** Hanna, Pezzelle & Belinkov, arXiv:2403.17806 — note the subtitle is
  *"Going Beyond Circuit Overlap When Finding Model **Mechanisms**"*, and the
  thesis is: *"when using circuits to compare the mechanisms models use to solve
  tasks, **faithfulness, not overlap, is what should be measured.**"* Comparing
  60 checkpoints by circuit Jaccard is precisely the comparison this paper says
  is unfaithful. [verified]
- **ACDC** — 10–20× cheaper than GPT-2-small at 6L, so possible once; weeks at
  900. **Edge Pruning** — an optimization per (model, task) with its own seed;
  900 runs would plot optimizer variance. [verified]
- **Circuits across checkpoints exists, and its finding is counter-intuitive.**
  Tigges, Hanna, Yu & Biderman (arXiv:2407.10827, NeurIPS 2024): *"the identity
  of components in each circuit is not constant … the name-mover head (4,6)
  suddenly stops exhibiting this behavior"* — yet *"in no model or metric are
  there dramatic shifts in algorithm."* **Component identity is unstable across
  checkpoints while the algorithm is stable**, so any "the circuit changed at
  step N" claim built on head-set overlap measures head-swapping. ⚠️ They
  **exclude Pythia-70M** — the exact configuration proposed here — because *"it
  does not learn the task."* [verified]
- New in discovery: **arXiv:2510.00845** (single-input causal-mediation scores
  are high-variance random variables), **arXiv:2602.22968** *Certified Circuits*
  (40 seeds, IoU stability guarantees), **arXiv:2606.06267** *Many Circuits, One
  Mechanism*. [discovery]

### 2.6 Parameter-free measures — where the affordable signal is

| Measure | Cost for all 900 | Null | Arch. |
|---|---|---|---|
| Weight spectra (stable rank, effective rank, SVD) | **minutes, on CPU, no data** | **analytic Marchenko–Pastur** | all four, identically |
| RankMe + PR + spectral entropy + anisotropy + intrinsic dimension | ~5–15 GPU-h, one activation pass + one eigendecomposition | step 0; MP | all four |
| CKA/PWCCA across checkpoints (60×60 matrix per seed per layer) | shares the above pass; pure linear algebra | CKA(step 0, step t) is a free drift curve | all four |
| Attention statistics (entropy, sink fraction, head concentration) | ~1–10 FP | **analytic uniform-attention null** | ⚠️ prefix-matching/copying is **decoder-only** — `head_detector.py` asserts a lower-triangular pattern and will not accept `HookedEncoder` |
| Logit lens + J-lens | free + ~30 A100-h | J-lens → logit lens at step 0 | J-lens needs porting for non-decoders |
| MDL probing | a few GPU-h | **built in** (uniform code) | **cross-arm comparable in bits** — almost nothing else is |
| Prequential code length | **free** — it is the area under the training loss curve | — | not cross-arm (MLM PLL is not a code length) |
| λ_max·η/2 (edge of stability) | ~1 GPU-h | — | **dimensionless, hence cross-arm comparable** |

**RankMe has a 2026 horse-race result behind it.** arXiv:2602.15997 [discovery]
tracked geometric measures and probes across six transformer sizes (405K–151M),
eight algorithmic tasks, and three Pythia LMs, and reports that *"of the
geometric measures tested, only rankme reliably precedes capability acquisition
for hard tasks"* — with a collapse-then-recover shape. ⚠️ Single-author preprint;
mostly algorithmic tasks; and its hard-vs-easy finding implies **no precursor at
all if the task is "easy" relative to capacity.** ⚠️ Sign conflict: arXiv:2606.19249
reports *monotone increase* in dimensional utilisation on a ViT. Do not assume the
direction.

⚠️ **CKA and attention sinks interact.** Davari et al. establish CKA's outlier
sensitivity; sink tokens are exactly high-norm outliers. Report CKA with and
without sink positions excluded, plus the token-norm distribution.

**LLC** — arXiv:2402.02364 was **retitled** *Loss Landscape Degeneracy and
Stagewise Development in Transformers* (TMLR 2025), and **arXiv:2402.03698 is
withdrawn** (merged into arXiv:2308.12108). [verified] Realistic cost: ~60–190
A100-hours for 900, **plus ~125 A100-hours of hyperparameter calibration before
measuring anything**. The authors' own guide states *"the question of how best to
choose hyperparameters for sampling is not resolved."* arXiv:2606.22389 (2026)
identifies a systematic minimisation bias in exactly the **transient,
off-equilibrium** regime — the early checkpoints. It is the only measure in this
survey with **no clean free null at step 0**, and it is judged **not comparable
across architecture arms** (different loss baselines, different parameter counts,
per-arm recalibration). ⚠️ Two variance components — seed and SGLD chain. Never
pool chain variance into a CI as if chains were seeds.

**Ruled out:** information-plane / mutual-information dynamics. Saxe et al. (ICLR
2018 / J. Stat. Mech. 2019) — *"none of these claims hold true in the general
case"*; the compression phase is a tanh-saturation and binning artifact. This is
the field's clearest worked example of sound measurement with an
estimator-dependent claim. **Nanda-style restricted/excluded loss** is also ruled
out: it requires a circuit basis handed over by algebraic task structure, and the
2026 "circuit-agnostic" successor (arXiv:2606.12966) never leaves modular
arithmetic. [verified]

### 2.7 Parameter-space decomposition — arm-agnostic by construction

Absent from the plan, and one of only two fully architecture-agnostic families
found: **APD** (arXiv:2501.14926, Apollo) and **SPD** (arXiv:2506.20790,
Goodfire/Apollo) decompose *weights* rather than activations into minimal,
faithful components; **arXiv:2511.08854** applies parameter decomposition
specifically at small scale. Weights are the same object at every step, so these
are inherently checkpoint-comparable and presuppose no decoder. [discovery]

Related: **arXiv:2601.22594** *Language Model Circuits Are Sparse in the Neuron
Basis* argues neurons suffice — consistent with **MIB**'s finding (arXiv:2504.13151,
ICML 2025) that **SAE features do not beat raw neurons** for circuit
localization. [discovery]

---

## 3. Tooling coverage by architecture family

**This section is the most decision-relevant and was verified hardest** — GitHub
API, PyPI JSON, shallow clones, and the contents of the published
`transformer_lens-3.6.0` wheel.

### 3.1 Tools

| Tool | Version / last commit | Decoder | Encoder-only | Enc-dec | Offline | Verdict |
|---|---|---|---|---|---|---|
| **TransformerLens** | **3.6.0** (2026-07-28); 182 commits/3mo | ✅ | ✅ `HookedEncoder` + BERT bridge | ✅ `HookedEncoderDecoder` + **12 adapters incl. M2M100** | ✅ | Claim of decoder-first is now largely false |
| **SAELens** | 6.47.1 (2026-08-05) | ✅ native | via `override_model` only | ❌ `HookedProxyLM.forward` passes one positional arg | ✅ | Decoder-first in fact |
| **circuit-tracer** | 0.5.0; 7 commits/3mo | ✅ only | ❌ | ❌ | transcoders download | **Genuinely decoder-only** |
| **nnsight** | 0.7.0 (2026-05-05) | ✅ | ✅ | ✅ | ✅ no network at import | The architecture-agnostic escape hatch |
| **inseq** | **0.7.1** (2026-03-06) | ✅ | n/a | ✅ **first-class** | ✅ | **Encoder-decoder-*first*** |
| **pyvene** | 0.1.8; commit 2026-03-05 | ✅ | ESM card = verbatim BERT paths | Whisper card | ✅ | ~40-line model card per arm |
| **pico-analyze** | commit 2026-02-19 | ✅ | ✅ | ✅ | ✅ | **Never loads a model** |
| **captum** | 0.9.0 (2026-04-17) | ✅ | ✅ | ✅ | ✅ | `nn.Module`-generic, active |
| **baukit** | 2024-02-22 | ✅ | ✅ | ✅ | ✅ | Frozen but architecture-blind |
| **tuned-lens** | **PyPI 0.2.0, 2023-07-18** | ✅ | ❌ | ❌ | ⚠️ `wandb` mandatory | Stale |
| **penzai** | 0.2.5; 2025-06-22 | JAX | JAX | JAX | ✅ | ⚠️ **Disqualified on JAX-only grounds**, independent of dormancy |
| **transformer-debugger** | last code change 2024-06 | GPT-2 | ❌ | ❌ | ❌ calls gpt-4o | ⚠️ **Not archived**, but effectively abandoned |
| **nnterp** | 1.3.0 | ✅ | ❌ | ❌ | ✅ | Markets agnosticism, delivers decoder-only |
| **ecco** | PyPI 2022-01-09 | — | — | — | ✅ | Dead |
| HF hooks (`output_hidden_states`) | transformers 5.14.1 | ✅ | ✅ | ✅ `cross_attentions` | ✅ | **Complete floor for all arms** |

### 3.2 What was verified inside the TransformerLens 3.6.0 wheel

- `HookedEncoder`, `HookedEncoderDecoder`, `model_bridge/supported_architectures/bert.py`
  and **`m2m100.py`** all ship. 139 adapter files, 152 registry entries.
- **`m2m100.py` docstring, verbatim: *"Covers Meta's M2M100 and NLLB-200
  translation families."*** Enc-dec adapters: T5, MT5, LongT5, T5Gemma,
  T5Gemma2, BART, mBART, **M2M100**, MarianMT, Pegasus, Blenderbot, LED,
  SwitchTransformers.
- **`HookedEncoder` uses `blocks.{i}.hook_resid_pre` — identical to
  `HookedTransformer`** — and `get_act_name` hardcodes `"blocks"`, so
  **`transformer_lens.patching` works unmodified on a BERT encoder.** Enc-dec
  uses `encoder_blocks.{i}` / `decoder_blocks.{i}`, which `get_act_name` does not
  match, so patching there means an explicit hook-name loop.
- `tests/integration/test_create_hooked_encoder.py` builds `HookedEncoder(cfg)`
  from a bare config — **from-scratch encoder instantiation works.**
- M2M100 test asserts bridge-vs-HF logit parity **< 1e-5**; `run_with_cache`
  populates both stacks.
- ⚠️ **`RobertaForMaskedLM` / XLM-R have no adapter.** The BERT adapter is ~150
  declarative lines, so this is hours of work, not weeks.
- ⚠️ **Maturity caveat, stated plainly:** the M2M100/NLLB adapter and its tests
  both landed 2026-07-27 in PR #1542, a 406-file / +59,410-line change the author
  describes as letting an agent run on `architecture-gaps.json` for a weekend.
  T5 (since 2025-10) and BERT (since 2025-11) are the hand-maintained ones.
  Issue #1611 (opened 2026-08-06): seq2seq `return_type="loss"` is wrong.

### 3.3 ⚠️ Two corrections to `CLAUDE.md` / `research_standards.md` §8

- **`run_with_cache` does *not* raise on batch size > 1.** The
  `NotImplementedError` at `bridge.py:3019` is scoped to
  `generate(return_cache=True)`; issue #1265 (batched `run_with_cache`) closed
  2026-04-22.
- **Bugs #1568 and #1587 are both still open** — but they live in the
  `boot_native` train-inside-TransformerLens path. Training in HF `transformers`
  and wrapping with `boot_transformers(hf_model=…)` avoids both, which is what
  [`decision 0001`](../../docs/decisions/0001_interlingua_model_implementation_substrate.md)
  already proposes.

Environment conflict stands: `transformer-lens` 3.6.0 needs `transformers>=5.9`;
`circuit-tracer` pins `<=4.57.3`; `jlens` needs `>=5.5`. **Separate venvs,
mandatory.**

### 3.4 Method-level coverage

| Method | Encoder-only | Enc-dec | Evidence |
|---|---|---|---|
| Probing | ✅ — **this literature *is* the encoder literature** | ✅ | Pires, Hewitt & Manning, Chi, Wu & Conneau |
| Activation patching / causal tracing | ✅ | ✅ | **Mueller, Xia & Linzen arXiv:2210.14328 — mBERT *vs* XGLM, multilingual counterfactual neuron interventions**; De Cao arXiv:2104.08164 (BERT + BART) |
| Logit lens | ✅ — **done on BERT first** (Kao et al. arXiv:2001.09309, Jan 2020) | ✅ via **DecoderLens** (arXiv:2310.03686, NAACL Findings 2024) | ⚠️ but see the post-LN caveat in §2.4 |
| CKA / CCA across objectives | ✅ | ✅ | **Voita, Sennrich & Titov arXiv:1909.01380 (EMNLP 2019) already ran MT vs LM vs MLM** |
| Representational similarity generally | ✅ | ✅ | `pico-analyze` never loads a model |
| Feature attribution | n/a | ✅ **better than decoders** | `inseq` config names `M2M100ForConditionalGeneration` and `NllbMoe` |
| **SAEs** | ⚠️ **no text-encoder precedent found** | ⚠️ **none found** | Only encoder-only SAE precedent is **MolFormer, a chemical LM** (arXiv:2606.23443). Three query phrasings returned nothing on BERT/RoBERTa/XLM-R/NLLB/mBART/M2M100 |
| **Attribution graphs / circuit tracing** | ✅ on ViTs (arXiv:2604.13304) | ❌ **none** | Method transfers to bidirectional encoders; the blocker is released *artifacts* |

### 3.5 Verdict

**The gap is partly closed, and the split is clean along method lines.** Probing,
activation patching, attention analysis and representational similarity are
closed — and were arguably never open, since the entire multilingual probing
canon the proposal rests on was measured on encoders. Feature attribution on
encoder-decoders is *better* supported than on decoders. **SAEs and circuit
tracing remain genuinely closed to non-decoders**, now on two independent
grounds: the tooling, and the total absence of published precedent on any text
encoder or MT encoder-decoder.

⚠️ **`tier1_plan.md` §3.1's premise is true as stated about the three tools it
names, but it surveys the wrong three tools** — it names one library that has
since shipped an NLLB adapter, and omits `inseq` (encoder-decoder-first) and
`pico-analyze` (architecture-agnostic by construction).

⚠️ **And §3.1's reason #1 does not survive at all.** See §7.1.

---

## 4. Statistics for transitions in training curves

### 4.1 The framing decides whether a CI exists

- A **changepoint** parameter is *non-regular*: the likelihood is not
  differentiable in the location, convergence is O(1) not O(n^-1/2), the limit
  law is a functional of a two-sided random walk — and **the ordinary
  nonparametric bootstrap is inconsistent for it** (Seijo & Sen, *AoS* 2011,
  arXiv:1101.1032: *"the standard bootstrap procedures in regression fail to
  provide valid confidence intervals for the change-point"*). Consistent
  alternatives: m-out-of-n, smoothed bootstrap, MOSUM bootstrap (Cho & Kirch,
  arXiv:2106.12844). [verified]
- A **sigmoid midpoint** is a *regular* parameter of a smooth nonlinear model:
  delta-method and profile-likelihood CIs are valid, the bootstrap is consistent,
  and it drops into a mixed model with a random effect per seed.

⚠️ **`tier1_plan.md` §5 proposes PELT/BOCPD followed by a bootstrap over 5 seeds.
That composition is not valid** — the bootstrap step is inconsistent for the
quantity the changepoint step produces.

**Is "changepoint on a smooth monotone curve" a category error?** Largely yes.
The change-in-mean cost assumes a piecewise-**constant** mean (false by
construction), iid residuals (false — within-run deviations are strongly
autocorrelated across adjacent checkpoints), and known constant variance (false —
early checkpoints are noisier, and constant-variance costs preferentially place
changepoints where variance is largest). **A falsifiable diagnostic, apparently
unreported in any ML training-dynamics paper:** fit at fixed penalty, then double
the checkpoint density and refit. A genuine changepoint keeps m=1 and localises
better; a smooth curve produces *more* changepoints tiling the steepest region.

**Which methods can return "no transition":** SMUCE (confidence set can be
empty), Davies/`pscore.test`, Andrews sup-Wald, Bayesian model comparison — yes.
PELT/binseg with BIC on a smooth sigmoid — nominally yes, practically never.
BOCPD — no.

### 4.2 The seed count is a hard resolution limit

With **n = 5 seeds**, the smallest attainable two-sided p from *any* exact
distribution-free paired test is **2/2⁵ = 0.0625**. This is arithmetic, not
conservatism.

| Correction family | p needed | Seeds needed |
|---|---|---|
| none | 0.05 | 6 |
| Bonferroni over 15 measures | 0.0033 | **10** |
| Bonferroni over 60 measures | 0.00083 | **12** |

A parametric paired t can go lower, at the cost of assuming Δ is approximately
normal across seeds — defensible for a difference of sigmoid midpoints,
indefensible for a difference of changepoint estimates (discrete, heavy-tailed).

**The paired design dominates.** Seed-to-seed variation in overall training speed
is large and **common to both measures**, so it cancels exactly in the
within-seed difference and dominates the variance in an unpaired comparison.
Note the identity: constraining the t_M-vs-t_B scatter slope to 1 and testing the
intercept **is** the paired test — the scatter plot checks the assumption, it
doesn't replace the test.

**An option that changes the achievable claim:** permuting the *labels* M and B
across measures makes the unit the measure (n = 15–60), not the seed. That tests
"M-type transitions precede B-type transitions on average" — a family-level claim
achievable at 5 seeds.

### 4.3 Multiplicity, and the confound that could manufacture the result

⚠️ **Cluster-based permutation tests do not license latency claims.**
Sassenhagen & Draschkow (*Psychophysiology* 2019), verbatim: *"the inferential
second stage does not ever 'see' first stage coordinates, only the cluster
size(s)"*, and *"cluster-based methods can underestimate the latency … this in
fact is more likely for the earliest and latest time points."* Maris himself:
*"one cannot quantify the uncertainty in the spatiotemporal localization."*

The consequence is specific and dangerous: **with more power, earlier points pass
the first-stage threshold, so the apparent onset moves earlier. A noisier measure
appears to transition later for that reason alone.** Mechanistic and behavioural
measures have different noise levels. **A naive onset comparison is confounded in
exactly the direction that would manufacture the H1 result.**
**Cluster-depth tests** (Frossard & Renaud, arXiv:2105.07514) give point-wise
strong FWER control and fix this; cluster-*mass* does not. Implementation
`permuco4brain` is **GitHub-only, not on CRAN**. [verified]

Rousselet's simulations (2023) partially rehabilitate cluster tests for onsets
while making the ranking clear (true onset 160 ms): changepoint bias 10 ms /
MAE 19; cluster-sum 30/30; FDR 30/42; MAX 40/44. **All methods are biased late.**
His recommendations transfer: estimate onsets **per run, not on the group
average**; report more than one method.

**Post-selection inference exists and is probably unknown here.** Jewell,
Fearnhead & Witten (*JRSS-B* 2022, arXiv:1910.04291) give valid p-values *after*
a changepoint is selected from the same data — but the test is on the
**magnitude** of the change in a ±h window under Gaussian iid errors with known
σ², **not a CI on the location**. Role: a gate, not the headline. For a location
CI with simultaneous coverage the right tool is **SMUCE / `stepR`** (Frick, Munk
& Sieling, *JRSS-B* 2014), which gives asymptotically honest confidence sets for
both the number and the locations. [verified]

### 4.4 What the grokking/emergence literature actually does

Blunt summary: **no paper in this literature establishes an ordering of
transitions with a defensible CI.** [verified — all four read]

- **Nanda et al. (ICLR 2023)** — phase boundaries read off by inspection. No
  changepoint procedure, no CI, no across-seed test.
- **Olsson et al. (2022)** — claims *simultaneity*, not precedence. Their
  strongest evidence is an **architectural intervention** (smeared keys) that
  moves the bump and moves both phenomena with it — good evidence for *coupling*,
  not a measured ordering. No seed counts, no timing error bars.
- **Hoogland et al. (TMLR 2025)** — GP-smooth then locate approximate zeros of
  the derivative w.r.t. **log** training time. Seeds: *"similar divisions arise
  with different training seeds"* — qualitative. **No CI on any stage boundary.**
  Their error bars are over 10 SGLD chains, not seeds.
- **Jian & Manning (EACL 2026)** — the closest match to this design, and its
  ordering claim has three problems not to reproduce: **the unit of independence
  is the verb, not the seed** (p < 10⁻⁸ from a single run — textbook
  pseudoreplication); the breakpoint is a **hard-coded absolute threshold**
  (0.01 above a 30-step baseline), so a noisier measure crosses later by
  construction; and the "50 steps before" claim carries **no interval at all**.

⚠️ **And a five-day-old paper attacks the measure directly.** Houghton &
Kapatsinski, *Exemplars in Disguise: Pure Exemplar Models Mimic Abstraction-First
Learning* (arXiv:2608.00821, 1 Aug 2026) [verified]: Jian & Manning's two onsets
are **asymmetric by construction** — between-class onset is a *relative* rank
test (first step where ≥10% of verbs pass a one-tailed Mann-Whitney U at
p<0.001), within-class onset is an *absolute* threshold. Their simulations show
**pure memorisers with no class representations and no information flow between
verbs reproduce the abstraction-first ordering** whenever observation sensitivity
is low. Conclusion: *"the ordering … is a function of observation sensitivity,
not of learning strategy."*

### 4.5 What physics offers, and what is unavailable

- **Finite-size scaling, data collapse, critical exponents, Binder cumulant —
  unavailable.** All require **system size as a varied axis**, with several
  well-separated sizes (the Binder literature routinely uses 6–10 lattice sizes
  spanning a decade). Seeds are replicates, not sizes. If asked whether a phase
  transition has been established in the physics sense, the answer is no, and the
  reason is the design.
- **Susceptibility / variance-peak — free, and apparently unused in ML.** At a
  continuous transition, χ ∝ N·Var(order parameter) **peaks at the critical
  point**. `Var_seeds[M(t)]` vs log t is a second, mechanistically independent
  onset estimate, it is a **maximum-finding** problem (better conditioned than
  changepoint detection on a monotone curve), and it does not inherit the
  "noisier looks later" confound. ⚠️ With 5 seeds the variance has 4 df and a
  χ²₄ sampling distribution, so heavy smoothing is required and the argmax moves
  with bandwidth. Corroborating: **arXiv:2602.09058** uses *collapse of
  across-realization variability* as a transition signature. [discovery]
- **Percolation model of emergence** (arXiv:2408.12578) — makes the ordering a
  *prediction* rather than an observation, which is a stronger position.

### 4.6 Imports the sweep found that ML has not made

**The single biggest gap: none of the covered changepoint methods produces a CI
on the *difference* of two changepoints, which is the headline quantity.**

- **Stress–strength reliability / stochastic precedence, R = P(X < Y)** — a large
  reliability-engineering literature estimating the probability that one random
  time precedes another, with parametric and nonparametric estimators and CIs.
  **This is the exact target of "does M emerge before B", expressed as an effect
  size in [0,1] rather than a significance test.** [discovery]
- **Doubly interval-censored estimation** (Reich et al., *Stat. Med.* 2009;
  arXiv:2310.04225) — a log-spaced checkpoint grid makes every emergence time
  interval-censored **by construction**, with geometrically widening intervals.
  This literature is built for exactly that. [discovery]
- **Lead–lag estimation** (Hoffmann–Rosenbaum–Yoshida, arXiv:1303.4871; noise-robust
  extensions arXiv:2601.01871) — consistent estimation of a *time shift* between
  asynchronously and noisily observed series, with a CI on the lag itself.
  **Thermal optimal causal path** (Sornette & Zhou) handles a *time-varying* lag.
  **Event coincidence analysis** (arXiv:1508.03534) is built for short/sparse
  event sets. [discovery]
- **Bayesian hierarchical change-point models with order constraints**
  (PMC8980247; arXiv:2603.14681 for **irregular designs and group structure**;
  `mcp`) — each unit gets its own changepoint from a population distribution,
  yielding a **posterior over the difference between two changepoints**. [discovery]
- **Systems Factorial Technology** (Townsend & Nozawa; R package `sft`) —
  psychophysics' rigorous answer to "did A finish before B started," diagnosed
  from the *sign* of a double-factorial interaction contrast. Works by design
  manipulation rather than curve-fitting. [discovery]
- **latenZy / latenZy2** (bioRxiv 2025.06.30.662308) — non-parametric, binning-free
  estimation of response onset *and the time two conditions diverge*, with
  permutation inference. The closest off-the-shelf estimator found for "at which
  checkpoint did measure A depart from baseline, with a CI." [discovery]
- **Temporal generalization matrices** (King & Dehaene, *TiCS* 2014) — train-at-t /
  test-at-t′ decoding, which distinguishes a change in the **ordering** of
  processing stages from a change in intensity or duration. Reframed as
  train-probe-at-checkpoint-i / test-at-j, a direct instrument for lead/lag.
  [discovery]
- **Anytime-valid / e-value inference** — Koning & van Meer (*JRSS-B* 2026,
  arXiv:2501.03982) construct an anytime-valid sequential version of *any*
  fixed-n test **at no power cost at the terminal sample size**, so peeking at
  every checkpoint and every added seed is free. **Bayes Factor Design Analysis**
  (Schönbrodt & Wagenmakers 2018) answers "is 5 seeds enough" quantitatively
  before running. [discovery]
- ⚠️ **MEG/EEG latency primer** (*Front. Neurosci.* 2018) states plainly that
  onset latency is **biased toward earliest onsets and unreliable under noise
  with slow rises** — applies verbatim to reading emergence times off smooth
  training curves. [discovery]

### 4.7 The psychometrics import — a mature literature on exactly this question

Measurement invariance asks whether two groups' latent constructs are comparable.
Languages = groups; interlingua = latent construct. The mapping is close to
direct and the field has litigated the problems ML is currently rediscovering.

- **Alignment optimization** (Asparouhov & Muthén, *SEM* 2014) — estimates
  group-specific loadings and intercepts under an approximate-invariance penalty
  so that **only configural invariance** is needed to compare latent means across
  many groups. Read with its skeptical evaluation (*Studies in Educational
  Evaluation* 2025, S0191491X25000768).
- **The configural / metric / scalar / strict invariance ladder** — a *graded*
  notion of comparability, which maps onto graded interlingua claims: same
  structure vs same loadings vs same intercepts.
- **Longitudinal measurement invariance / response-shift detection** — tests
  whether a measure means the same thing at time t as at t′. **Directly applies to
  "does CKA at step 100 measure the same construct as CKA at step 100,000" — a
  question a 60-checkpoint design assumes away.**
- **DIF detection** (Mantel–Haenszel, SIBTEST, IRT-LR, Rasch trees;
  regularized/LASSO DIF with built-in multiplicity handling) — a **per-item**
  diagnostic rather than one global scalar.
- **Mokken invariant item ordering** (`mokken`, `check.iio`; MIIO, MS-CPM, IT;
  HT coefficient) — nonparametric IRT machinery testing whether items are
  acquired in the **same order** regardless of who is measured. **A ready-made
  test for "do all 5 seeds acquire capabilities in the same order."**
- **Attribute Hierarchy Method / cognitive diagnostic models with learning
  progressions** — *infers* a prerequisite DAG over latent skills from response
  patterns rather than assuming the order.
- **Tucker's congruence coefficient** with published interpretive thresholds
  (0.95–1.00 equal, 0.85–0.94 fair) — ML similarity measures notoriously lack
  agreed thresholds; this field settled that decades ago.
- **IRT for benchmark evaluation** (arXiv:2505.15055, arXiv:2509.22888) — models
  item difficulty and discrimination instead of raw accuracy. A behavioural
  measure at 36M will be low and noisy; **IRT ability estimates are far more
  sensitive than mean accuracy**, which matters for a lead/lag comparison.

All [discovery] — none of these citations were opened.

---

## 5. Super weights and concentrated structure

### 5.1 State of the literature

**The entire super-weight corpus is three papers.** An arXiv title+abstract search
returns exactly three LLM-relevant hits; a Semantic Scholar pull of ~51 citing
papers found them to be almost entirely quantization/pruning/hardware — none on
detection methodology, none on formation. [verified]

**Yu et al. (arXiv:2411.07191, Apple)** detect by: one forward pass, one prompt;
plot outliers in `mlp.down_proj` input/output activations; where an input spike
at index k coincides with an output spike at j, read off `down_proj[j,k]`;
repeat **until "the magnitudes of large maximum activations are greatly
suppressed."** No threshold, no null, no significance test, and no stopping rule
beyond visual judgment. Validation is entirely by consequence (C4 perplexity
7.08 → 763.65). **No model is reported as lacking one** — which is what a
detector with no null would produce either way. Instruction-tuned variants have
super weights **at the same coordinates as their base models**, so the
TowerBase/TowerInstruct coordinate match in `registry.md` **replicates a
published observation**; the quantified sharpening (KL 0.96 → 1.25) is the new
part, at n=1 model pair. [verified]

**Subramanian et al. (arXiv:2607.08733, COLM 2026)** is the one substantive
follow-up, and it is skeptical in the direction the repo's own data points:
**super-weight pruning damage is not universal across LLMs**; and training only
the super weights drops OLMo to chance while training the same number of *random*
weights in the same layers succeeds. Importance ≠ trainability. [verified]

⚠️ **A correction for `registry.md`.** The claim "super weights emerge gradually,
beginning early in pre-training" traces through Yu et al.'s related-work section
to **Kovaleva et al. 2021, which is about BERT LayerNorm outliers, not super
weights.** Secondary sources have promoted it into a super-weight result. It is
not one. [verified]

### 5.2 Detection with a null — the blunt answer

**No published method for finding super weights, massive activations or attention
sinks has a calibrated null, a significance test, or a threshold derived from
anything but "where the perplexity stopped getting worse" or "where the plot
looks like a spike."** Three partial exceptions: [verified]

| Criterion | Value | Provenance |
|---|---|---|
| Massive activation | \|a\| > 100 **AND** ≥~1000× hidden-state median | authors call it *"a loose but broad definition"* |
| Attention sink | Sink₁^ε, **ε = 0.3** | chosen for length-insensitivity |
| Vision registers | token norm > **150** | authors call it a *"hand-picked cutoff"* |
| LLM.int8 outlier | α ≥ **6.0**, ≥25% layers, ≥6% seq dims | **reverse-engineered so the 125M model yields exactly one outlier** |
| BERT Busters | LayerNorm γ,β ≥ **3σ**, ≥½ layers | relaxed to 2σ per model until outliers appeared |
| Puccetti et al. | 2–3σ **AND ≥5× random-dimension damage** | **the only encoder paper with an empirical null** |
| **He et al. (arXiv:2405.19279, NeurIPS 2024)** | **kurtosis of neuron activation RMS** | **known minimum = 1 = no outlier features, and ≈1 at initialization — the null is built into the statistic** |

⚠️ **"6" means three different things** across these papers — an absolute
activation magnitude and a 6σ z-score. They coincide only where activation std ≈ 1.
Cross-paper agreement on "threshold 6" is notational coincidence.

⚠️ **Nearly every criterion is scale-dependent in the direction that makes small
models look clean.** arXiv:2508.03616 had to **relax** the massive-activation
definition because the original failed on small models. At 36M, "we found no
concentrated structure" and "our threshold was calibrated on 7B models" are
indistinguishable statements.

**What a null would look like.** The right framing is the **maximum statistic
under multiplicity** — searching ~19M weights for the largest ablation KL is the
neuroimaging voxel problem, and the standard answer is the max-statistic
permutation null (**Nichols & Holmes, *Human Brain Mapping* 15(1):1–25, 2002**).
Candidate null ensembles, increasing in cost: (1) the step-0 random-init network
run through the identical detector; (2) a **weight-shuffled** network preserving
the magnitude marginal while destroying learned coordinate structure — and the
repo's own finding that ablating the 1000 largest-magnitude weights does nothing
says this is the *right* null, because magnitude is not what carries the effect;
(3) matched random subsets extrapolated to the max order statistic. [verified]

⚠️ **Random matrix theory is the wrong tool, and it is worth recording why.**
Marchenko–Pastur and BBP are **spectral**; a single large weight is a rank-one
perturbation only degenerately, and the statistic here is a *functional of the
whole network*, not a weight magnitude. No work applies MP/BBP to entrywise
outlier detection, and inventing it is not recommended. **Extreme value theory
is the untapped option** — an arXiv search for EVT ∧ LLMs returns only papers on
*evaluation metrics*, nothing on network internals. GEV/Gumbel or peaks-over-
threshold on per-block maxima would also estimate the shape parameter ξ, which is
itself informative: ξ>0 says a "super weight" is the top of a heavy-tailed
continuum rather than a separate object.

Discovery found two candidates: **arXiv:2605.18898**, a two-parameter Weibull
framework for transformer weight distributions reporting max-to-99th-percentile
ratios up to 14.3×; and **arXiv:2603.27885**, Hill-estimator tail index with a
sharp threshold for when learned structure becomes spectrally visible. [discovery]

### 5.3 Planted-structure validation — the gap is precise

| Resource | What is planted | Useful for a planted super weight? |
|---|---|---|
| **Tracr** (arXiv:2301.05062) | a compiled RASP program | Partly — weights are compiled, not trained |
| **InterpBench** (arXiv:2407.14494, NeurIPS 2024 D&B) | transformers trained with **Strict IIT** so internal computation aligns to a known causal model | **Closest existing template** — SIIT is a way of *forcing* known structure into a trained network |
| **MIB** (arXiv:2504.13151, ICML 2025) | mostly faithfulness proxies; one InterpBench IOI model has a known circuit, scored by AUROC | Includes a **random-circuit null**. Its own admission: *"If we knew the ground-truth circuit, we could instead compute precision and recall."* |
| **SynthSAEBench** (arXiv:2602.14687) | planted **features** with realistic correlation and superposition | Template for the analysis side |
| **Pando** (arXiv:2604.11061) [discovery] | **planted decision trees** for feature-level causal ground truth | Newest entry; explicitly warns against rewarding plausible narratives that don't improve behavioural prediction |

⚠️ **Every planted ground truth in this literature is at the level of a circuit,
component, edge, or feature direction. Nothing plants a single scalar weight, and
nothing validates a weight-importance detector against a known answer.** Targeted
searches for planted single-weight validation returned zero. [verified]

**The prior to hold.** In the one adjacent subfield that finally ran the random
control, the flagship method largely failed it: *Sanity Checks for Sparse
Autoencoders* (arXiv:2602.14111) reports interpretability 0.90 vs 0.87 for random
baselines, sparse probing 0.72 vs 0.69, **causal editing 0.72 vs 0.73 (SAEs
lose)**, and on synthetic data with known ground truth **SAEs recover 9% of true
features at 71% explained variance.** [verified]

### 5.4 Does concentrated structure exist at 36M?

**Split, and the split lands on the question.**

- **Attention sinks and massive activations: yes, down to 14M.** Gu et al. (ICLR
  2025), verbatim: *"attention sink emerges in small LMs, even in Pythia-14M."*
  Massive activations at GPT-2 base (124M); LayerNorm outliers in a **BERT-medium
  8L/512d** retrained from scratch (~40M) — the closest published analogue to a
  6L/512d encoder. [verified]
- **The single-scalar super weight: never looked for below 1B.** Smallest reported
  is OLMo-1B. **Whether one scalar becomes causally load-bearing at 36M is
  unknown and appears never to have been asked.** [verified]
- **The absence results are about *conditions*, not scale**: context length 128 at
  120M → near-zero sink; <500M training tokens at 60M → sink disappears; weight
  decay γ=5.0 → Sink₁ = **0.01%**; sigmoid instead of softmax → no sink up to 1B;
  an explicit learnable attention bias → **no massive activations at 124M at
  identical perplexity**. The only genuine below-scale absence claim in the
  literature is **vision-only** (ViT-T/S/B show no artifacts).

⚠️ Gu et al.'s weight-decay sweep is the most actionable table in the area and it
is **strongly non-monotonic**: Sink₁ = 15.20 / 15.39 / 15.23 / 18.18 / **41.08** /
37.71 / 6.13 / **0.01** % for γ = 0 / .001 / .01 / .1 / .5 / 1 / 2 / 5. **You can
manufacture or abolish the sink by weight decay alone** — which means a cross-arm
comparison at unmatched regularization measures the recipe, not the architecture.
[verified]

### 5.5 When does it form?

| Source | Substrate | Timing |
|---|---|---|
| Gu et al. (ICLR 2025) | ~60M from scratch, 20k steps | Sink emerges **1k–2k steps** (5–10% of training) |
| arXiv:2510.06477 | Pythia 410M–12B | Massive activations, sinks and compression valleys *"emerge together around step 1k and remain synchronized"* |
| arXiv:2503.21718 | Pythia-12B, 15 checkpoints | Outlier dims appear **~steps 3000–4000** |
| arXiv:2508.03616 | **Pythia 14M–12B, 9 sizes, ~154 ckpts** | **Absent at step 0**; two regimes; fit `f(t)=A·e^{−λx}log(x)+K`, mean R² = 0.984. **Timing params are the ones you cannot predict from architecture** |
| Kovaleva et al. | **BERT-medium 8L/512d from scratch** | LayerNorm outliers diverge at **~50k steps**. ⚠️ Their run developed **one** outlier; the published model has **two** — **at ~40M the count is run-dependent** |
| Olsson et al. | — | Induction phase change: abrupt, **2.5B–5B tokens** |

⚠️ **Every timing number above is n_seeds = 1.** **PolyPythias** (arXiv:2503.09543,
ICLR 2025) — 45 runs, 9 seeds × 5 sizes (14M–410M), ~7k checkpoints — exists
precisely to fix this, and nobody has applied it to sinks, outliers, or super
weights. ⚠️ **Every checkpoint study is decoder-only**; Kovaleva 2021 is the only
encoder result and it has no checkpoint grid.

⚠️ **The field disagrees about whether these are one phenomenon or several**:
arXiv:2510.06477 says they emerge *together*; arXiv:2603.05498 says they are
functionally distinct and decouple without pre-norm; arXiv:2606.02378 says
induction circuits precede sinks by **10–20× in tokens**. These cannot all be
right in the same sense.

**Super-weight formation specifically: nothing exists.** Three independent
zero-result searches. Nobody knows when one forms, whether it exists at early
checkpoints, whether the same coordinate is selected across seeds, or whether it
appears abruptly. [verified]

⚠️ **A tension worth recording rather than resolving.** arXiv:2605.15572 reports
global activation maxima spanning nearly four orders of magnitude at comparable
parameter counts, with **Gemma3-27B-it at the extreme high end** — while
`registry.md` finds Gemma **near-inert** on single-weight ablation KL. These are
consistent only if massive-activation magnitude and single-weight causal
criticality are decoupled, which arXiv:2603.05498 argues on other grounds.

---

## 6. What comes for free

### 6.1 Public checkpoint suites

| Suite | Sizes | Intermediate checkpoints | Seeds | Languages |
|---|---|---|---|---|
| **Ettin** (arXiv:2507.11412, JHU-CLSP) | **17M–1B, paired encoder AND decoder, identical data and recipe** | **250+**, plus **batch-level training data per checkpoint** | 1 | EN |
| Pythia | 70M–12B | 154 (log to 512, then every 1000) | 1 | EN |
| **PolyPythias** (arXiv:2503.09543) | 14M–410M | ~7k total | **9** | EN |
| OLMo 1/2/3 | 1B–32B | 378 / 273 / 800+ branches | 1 | EN |
| Pico | 11.3M / 64.6M / 181M / 570M | ~200 as git **commits** on a run branch | 1 | EN |
| DataDecide | 4M–1B | `step{N}-seed-{S}`, >30k total | **3** | EN |
| Stanford CRFM GPT-2 | 124M, 355M | **401 tags** per repo | **5** | EN |
| MultiBERTs | BERT-base | 28 steps × 5 seeds = 140 | **25** | EN |
| SimpleStories V2 | 1.25M–134M | on W&B | 1 | EN |
| TinyModel | 44M, 4L | ships **trained SAEs and transcoders** | 1 | EN |
| mmBERT | small, base | full pretrain/mid/decay chain, Composer format | 1 | **1,833** |
| **Blevins et al. XLM-R replica** | ~270M | **39 fairseq checkpoints, live at `nlp.cs.washington.edu/xlmr-across-time/`, with an HF conversion script** | 1 | 59, incl. FR + TR |
| BLOOM-560m / -1b7 | — | **8 tags each** | 1 | FR; ⚠️ TR likely absent |
| EuroLLM | 1.7B–22B | **3–4 phase snapshots public**, not the 26 the paper used | 1 | FR |
| Apertus (arXiv:2509.14233) | 8B, 70B | branches | 1 | 1800+, ~40% non-English |

⚠️ **`bigscience/bloom-intermediate` (176B) now returns only `main`** — the
per-step tags its model card still documents are gone. [verified]

**The structural finding: the seed dimension and the multilingual dimension have
never been released together.** Every multilingual suite with intermediate
checkpoints is n=1; every suite with ≥3 seeds is English-only. [verified]

**Ettin is the largest single omission from the plan** — a public, controlled,
matched-data encoder-vs-decoder contrast at 17M–1B with data order recoverable
per checkpoint. [discovery]

**The proposed configuration is Pythia-70M's transformer body exactly** —
6L/512d/8h, **18,915,328 non-embedding parameters** [verified from the model
card]. Every scale question can be checked against 154 existing checkpoints
before any training. ⚠️ And Pythia-70M is the model Tigges et al. **excluded**
from circuit analysis after 300B tokens.

⚠️ **Compute-optimality tension.** At 18.9M non-embedding parameters,
Chinchilla-optimal is ~0.38B tokens (~0.72B on 36M total). The induction
transition is at **~2×10⁹ tokens — 3–5× past compute-optimal.** Pythia-70M is
trained to ~800× Chinchilla, and that is the regime its circuits were studied in.
Train compute-optimally and the mechanism may never form; train far past it and
"training dynamics" means the dynamics of an over-trained model. Both are
defensible; the choice is not currently named.

### 6.2 Harnesses

⚠️ **`research_standards.md` §8's "there is no library for tracking interp metrics
across a checkpoint grid" may no longer be true.** **TRACE** (arXiv:2507.03668,
EMNLP 2025 demo `2025.emnlp-demos.62`) is a modular **in-training** analysis
toolkit — probing, intrinsic dimensionality, Hessian curvature, layer-wise
diagnostics, convergence-based early stopping — whose paper explicitly claims
existing tools *"lack temporal tracking."* [discovery] Also **SAE-Track**
(arXiv:2412.17626) for warm-started SAE series across checkpoints, and
**`reward-lens`** (arXiv:2604.26130) whose **ten-method adapter protocol isolates
architecture-specific details** so lens/patching/SAE modules are written once —
the design pattern an unresolved-architecture study needs. [discovery]

Confirmed as the closest existing thing: **`pico-analyze`** — grepping the repo
for `AutoModel`/`from_pretrained` returns **nothing**; it reads saved
activation/weight tensors and computes CKA, PWCCA, proportional effective rank,
condition number, Gini, Hoyer. Architecture-agnostic by construction. [verified]
And **`devinterp` v2.0** stores per-token loss to **xarray/Zarr** — the only
working precedent found for a labelled multi-dimensional grid store. [verified]

**Checkpoint scheduling:** `gpt-neox` is the only mainstream trainer verified to
have native log spacing — `checkpoint_scale: log` + `checkpoint_factor` +
**`extra_save_iters`**. `pico-train` is **fixed-interval only** and
**decoder-only exclusively**. `levanter` **merged into the Marin monorepo in Nov
2025**. [verified]

**`lm-evaluation-harness` supports `AutoModelForSeq2SeqLM`** plus a `revision`
argument for partially-trained checkpoints — the only eval harness verified to
cover an encoder-decoder arm. [verified]

### 6.3 Data for EN/FR/TR — more is off-the-shelf than the plan assumes

| Asset | EN | FR | TR | Note |
|---|---|---|---|---|
| **FLORES+** `openlanguagedata/flores_plus` | ✅ | ✅ | ✅ | **v4.6**, CC BY-SA 4.0, 997 dev / 1012 devtest, no `trust_remote_code`; ⚠️ **gated** — accept terms on the login node |
| **UD 2.18** (released 2026-05-15) | EWT 16,622 | GSD 16,341 | BOUN 9,761 | ⚠️ TR-IMST is **CC BY-NC-SA** |
| **MultiBLiMP 1.0** (arXiv:2504.02768, TACL) | **770** | **2548** | **1742** | Subject-verb agreement, UD+UniMorph, all three languages [verified via HF datasets-server] |
| **TurBLiMP** (EMNLP 2025 `2025.emnlp-main.834`) | — | — | **16 phenomena × 1000 + 2000**, incl. explicit Subject Agreement, with human ratings | A **39M Goldfish model is above chance across phenomena** — a direct scale-validity datapoint |
| CLAMS | ✅ | ✅ | ❌ | A CLAMS-based design silently drops Turkish |
| XNLI / Tatoeba / FineWeb2 / HPLT / Glot500-c | ✅ | ✅ | ✅ | ⚠️ HPLT v2.0 `tur_Latn` path 404'd — verify |
| `minicons` | — | — | — | Exposes `MaskedLMScorer`, `IncrementalLMScorer` **and `Seq2SeqScorer`** — **one API for all four arms** |

⚠️ **`tier1_plan.md` §3.3 and §6 W2 budget "custom-built per language" agreement
minimal pairs as "real linguistic work."** MultiBLiMP ships all three languages
and TurBLiMP adds 18,000 Turkish items; both are HF-downloadable and
pre-cacheable. [verified]

⚠️ But see **Başar & Bisazza, SIGTURK 2026 (`2026.sigturk-1.9`)**: Turkish
minimal-pair benchmarks are **confounded by morpheme count, subword count and
sentence length** — from the author of TurBLiMP, about her own benchmark's use.
[discovery]

⚠️ **FLORES-200 is Wikipedia-derived.** If the pretraining corpus includes
Wikipedia, devtest sentences may be in training data, and "alignment emerged at
step k" becomes "the eval set was memorised at step k." *When Flores Bloomz
Wrong* (arXiv:2601.20858, EACL 2026) demonstrates this including cross-directional
leakage. A 13-gram bidirectional overlap audit is a few CPU-hours. [discovery]

---

## 7. Areas the brief did not name, that the sweep judged decision-relevant

### 7.1 ⚠️ The "JSD measures are decoder-only" objection does not hold

The metric requires exactly one thing: **a probability distribution over the
vocabulary at a designated slot, conditioned on context.** [verified from the
full text of `2026.eacl-long.32`: `P_v(x) = (1/N) Σ P(x|s_i)`, then pairwise
`D_JS`.] Nothing in that requires autoregression:

- **An MLM at `[MASK]` gives exactly that object**, same type, same dimension.
  Jian & Manning's footnote that smoothing is unnecessary because "next-token
  distributions do not contain true zeroes" holds identically for an MLM softmax.
- **The encoder-decoder arm needs no adaptation at all** — an NLLB-like decoder
  emits a genuine next-token distribution. **So §3.1's reason #1 does not touch
  the arm H3a is defined over.**
- The theoretical objection to MLM scoring — Torroba Hennigen & Kim (ACL 2023):
  MLM conditionals are not consistent with any single joint distribution — bites
  **sentence-level** scoring, which needs a joint. **It does not bite a
  single-slot divergence measure**, which never needs one. The JSD measure is on
  *firmer* footing for an MLM than pseudo-perplexity is.

**What is true is narrower: nobody has published the MLM version.** An arXiv
search for `"masked language model" AND "Jensen-Shannon" AND "training dynamics"`
returns **totalResults = 0**. The closest work, *Sudden Drops in the Loss*
(arXiv:2309.07311, ICLR 2024 Spotlight — BERT-Base, 3 seeds, syntactic attention
structure + BLiMP), reaches for a **scalar** (PLL) where Jian & Manning reach for
a **distribution**. That costs a citation, not a method. [verified]

Three real costs of the encoder version, none fatal: the **exemplar-first
baseline** needs a bidirectional context window instead of J&M's *"unidirectional
10 token context window"* (a parameter change, reportable as a deliberate
modification); **the step axis is not comparable across arms** because at equal
steps an MLM has received ~15% as many prediction targets (plot against
tokens-of-supervision or matched compute); and **bidirectional context changes
what the measure means**, mildly for subject-verb agreement since the controller
precedes the verb in all three languages — though Turkish is verb-final with more
intervening material, a confound already present in the plan.

### 7.2 The behavioural axis — "measure it continuously" is not a clean fix

`tier1_plan.md` §1 fixes the mirage problem by measuring behaviour continuously.
The rebuttal literature says that is a *negative control*, not a positive
instrument: [verified]

- **Du et al. (NeurIPS 2024, arXiv:2403.15796)** ran **Schaeffer's own Brier
  score** across checkpoints and still got a tipping point at pretraining loss
  **≈ 2.2**, *"surprisingly all around 2.2"* across four datasets in two
  languages. Verdict quote: *"continuous metrics cannot eliminate the observed
  tipping point."* ⚠️ And their own caveat is the trap: a context-free uniform
  predictor scores Brier 0.75, so all pre-threshold Brier improvement is
  calibration, not competence.
- **Hu et al. (ICLR 2024)** found *"accelerated emergence whose scaling curve
  cannot be fitted by standard scaling law function"* at effectively infinite
  resolution.
- **Schaeffer et al.'s own 2024 follow-up (arXiv:2406.04391)** concedes continuous
  surrogates do not restore predictability: the missing quantity is probability
  mass on the **incorrect** options, so `log p_gold` can climb smoothly while
  accuracy sits at chance.
- **The other direction is worse.** Michaud et al.'s quantization model says a
  smooth aggregate is *what you get* when many things transition abruptly at
  staggered times. **Smoothing the behavioural axis can manufacture the lag** by
  flattening a real behavioural transition. Two small-scale 2026 preprints report
  transitions the loss curve does not show at all, one a log-probability margin
  crossing zero inside 100 steps.
- ⚠️ **Aggregation is a separate confound from metric continuity.**
  arXiv:2510.24934 shows aggregate agreement-attraction curves hide non-monotonic
  per-condition dynamics during training. [discovery]

**And a sharper version of the mirage objection that the current fix does not
address:** mechanistic progress measures are *constructed* to be continuous while
behavioural metrics are thresholded, so **a continuous quantity crosses any
threshold before a thresholded one, by construction.** Without a null in which
the two are matched for smoothness *and noise*, "mechanistic leads behavioural"
is close to unfalsifiable. This is scale-independent.

**Probing costs, which favour one option strongly** [survey agent's arithmetic]:
at 16,200 probe fits, a **closed-form ridge/LDA probe is milliseconds and has
exactly zero refit variance** (all fits in well under an hour); an SGD probe is
~135 GPU-hours *and* introduces a probe seed. **MDL codelength** (Voita & Titov,
EMNLP 2020) costs ~2–3× one fit, is continuous in bits, has a built-in null, and
is **stable across hyperparameters where probe accuracy flips which layer wins**.
⚠️ Caching activations is not an option: ~1.6 PB.

⚠️ **The refit problem is essentially unacknowledged in the checkpoint-probing
literature.** Blevins et al. (39 checkpoints) mention "variance" once, meaning
across-language spread; Liu et al. (~62 checkpoints) use a fixed probe seed for
*reproducibility*, and the word "variance" appears **zero** times. The only
escape found is dropping the fit entirely — **ABX** (de Seyssel et al., EMNLP
2025), training-free, applied to XLM-R checkpoints. [verified]

⚠️ **Probe accuracy is not on the mechanistic side of the ledger.** Elazar et al.
(TACL 2021): *"conventional probing performance is **not correlated** to task
importance."*

### 7.3 Cross-seed comparability

Permutation symmetry does **not** break the plan's headline measures — CKA,
mutual-kNN, JSD and probe accuracy are permutation-invariant by construction. But:

- ⚠️ **Git Re-Basin does not transfer cleanly to transformers.** The only
  systematic treatment (arXiv:2310.05719, ICLR 2024) must handle multi-head
  attention, layer-norm and residuals individually and finds *"the significant
  role of **soft** alignment in the case of Transformers."* Hard permutation
  matching is not what works. Reference repos are dormant (2023-03, 2024-01).
  [verified]
- **Only 1–5% of neurons are universal across five GPT-2 seeds** (arXiv:2401.12181)
  — same architecture, same data, seed the only difference. [verified]
- **Model stitching** is the one item here worth its cost: functional rather than
  geometric, permutation-invariant automatically, one linear layer per pair, with
  a clean null. **No published cross-lingual stitching study was found.** [verified]
- **Multi-Bootstrap** (MultiBERTs, arXiv:2106.16163) is the published inference
  procedure for "several pretraining runs, limited test data" — the exact shape of
  a Δt-with-CI design at n≈1012. [verified]
- ⚠️ **Middle-layer attention heads are the least seed-stable** (arXiv:2602.16740)
  — and middle layers are where cross-lingual alignment is reported to live.
  [discovery]

### 7.4 Tokenizer confounds, measured on FLORES

Petrov et al. (NeurIPS 2023, arXiv:2305.15425), premium relative to English,
**measured on FLORES-200**: [verified]

| Encoding | EN | FR | TR |
|---|---|---|---|
| Characters | 1.00 | **1.19** | **1.03** |
| UTF-8 bytes | 1.00 | 1.24 | 1.12 |
| English-centric BPE (GPT-2) | 1.00 | 2.00 | **2.43** |
| Multilingual BPE (XLM-R) | 1.00 | 1.30 | **1.04** |

**Turkish is not intrinsically longer — in characters it is more compact than
French.** The entire penalty is manufactured by subword tokenization, and a
well-fitted multilingual tokenizer nearly erases it. Under an English-fitted BPE
the ordering **inverts**. This is a design variable, not a fact about Turkish.

⚠️ **The checkpoint-grid consequence, which the sweep flags as unreported
anywhere:** checkpoints are log-spaced in **optimization steps**; steps consume
**tokens**, not content. Under a shared English-leaning BPE, a step-*k* checkpoint
has seen **~2.4× less Turkish content** than English content. **A "lead" of a few
checkpoints could be entirely that.** Reporting cumulative content-bytes per
language per checkpoint costs nothing.

⚠️ **Vocabulary size is a dial on the dependent variable.** A 6L/512d body is
18.87M params; at V=32k tied, embeddings are **16.78M — 47% of the model** (which
suggests the "36M" budget already assumes ~32k tied). Dufter & Schütze (EMNLP
2020) find **overparameterization *reduces* multilinguality** (0.58 vs 0.70)
because the model can afford language-private capacity — so bigger vocabulary →
more private capacity → alignment emerges **later**. Tao et al. (NeurIPS 2024)
have their smallest IsoFLOPs group at **N_nv=33M, d=512** — this exact config —
and find optimal vocabulary *decreases* in the data-constrained regime.

⚠️ **MAFEX / "Tokenization–Morphology Misalignment"** [discovery, UNVERIFIED]:
interpretability methods assume subword tokens are independent semantic units,
which breaks for morphologically rich languages; evaluated on Turkish LMs. Every
probe, feature and attribution computed on Turkish sits in a basis that does not
align with Turkish morphology while EN/FR are far less affected — **which could
present as a spurious "Turkish aligns later" result, i.e. as the H4 finding.**

⚠️ **The anchor-point question is settled in the reassuring direction, with one
exception.** Wu et al. (ACL 2020, arXiv:1911.01464) ran the decisive ablation: a
language-prefixed vocabulary union with **zero shared subwords** costs only
**−1.1 average** (XNLI fr 73.6→72.1), vs −29.6 for separating embeddings and
layers 1–6. *"We have previously overestimated the contribution of anchor
points."* ⚠️ But anchors matter **more for distant pairs**, so EN–FR and EN–TR are
**differentially** sensitive. [verified]

### 7.5 Data-side and scale

- ⚠️ **A corpus-statistics null is the most likely reviewer objection.** Both
  mechanistic and behavioural curves are plausibly downstream of cumulative
  exposure to the same corpus statistics. Belrose et al. (ICML 2024,
  arXiv:2402.04362) show **token n-gram frequencies are formally equivalent to
  embedding-vector moments** and low-order moments are provably learned first;
  TrackStar found **BM25 beats gradient attribution** at fact tracing.
  **No multilingual analogue of distributional simplicity bias exists** — whether
  models learn within-language n-gram statistics before cross-language
  co-occurrence statistics is open, and answering it costs an n-gram fit plus a
  scoring pass per checkpoint. [verified]
- **At 36M, one training run is single-digit A100-hours**, so the
  "requires retraining" family — normally unusable — is affordable. A
  leave-one-group-out data ablation is ~20× one run, **cheaper than a single
  EK-FAC influence grid**. Template with a counterintuitive prediction to test:
  Shao et al. (arXiv:2601.00364, ACL 2026) find bilingual documents are **2% of
  the corpus** yet removing them drops translation BLEU **56%**, while on granular
  re-introduction **parallel data restores 91% and code-switching contributes
  minimally**. [verified/discovery]
- ⚠️ And **arXiv:2603.29026** finds parallel data has only **minimal** effect on
  cross-lingual alignment — it accelerates sharing early and reduces
  language-specific neuron count, but alignment emerges at similar levels
  without it. **That bears directly on H3a's premise.** [discovery]
- **Item-level learning curves** (Chang & Bergen, TACL 2022 — run on LSTM, **BERT
  and GPT-2**, so the method covers encoder and decoder arms; Chang, Tu & Bergen,
  TACL 2024 — five quantities per item including **age of acquisition**,
  **forgettability** and **cross-run variability**) cost a forward pass on a fixed
  probe set at checkpoints already being saved. **Best value-per-cost item found.**
  [verified]
- **Scale floors** [verified]: cross-lingual transfer at **29.65M params → XNLI
  Russian 61.4** (K et al., ICLR 2020; the cliff is between 4.23M and 11.83M);
  shared multilingual space at **~0.9M params** (Dufter & Schütze), with
  overparameterization *hurting*; induction heads need ≥2 layers and **~2×10⁹
  tokens**, roughly scale-invariant in absolute token count. ⚠️ For the encoder
  arms "induction head" is not a defined construct — prefix-matching presumes
  next-token prediction. The better-established encoder-side transition is
  **Syntactic Attention Structure** (Chen et al., ICLR 2024), acquired in a brief
  20k–30k-step window. **Putting "induction head strength" for decoders and
  something else for encoders in one table column and calling the difference an
  architecture effect is the substituted-metric failure `CLAUDE.md` rule 4 exists
  for.**

---

## 8. Claims in current repo documents that this survey contradicts

| Document | Claim | Status |
|---|---|---|
| `tier1_plan.md` §2 | SAE cross-seed matched features "as low as **1–4%**" | ⚠️ **Not in the cited paper.** Paulo & Belrose report 30% / 42%. Source it or remove it |
| `tier1_plan.md` §3.1 reason 1 | JSD measures are next-token, so encoders are out | ⚠️ **Does not hold.** An MLM at `[MASK]` gives the required object; the encoder-decoder arm needs no adaptation at all (§7.1) |
| `tier1_plan.md` §3.1 reason 2 | TransformerLens / SAELens / circuit-tracer are decoder-first | ⚠️ **True of those three tools, but it surveys the wrong three.** TL 3.6.0 ships an NLLB adapter; `inseq` is enc-dec-first; `pico-analyze` is architecture-agnostic (§3) |
| `tier1_plan.md` §3.3, §6 W2 | Agreement minimal pairs are "real linguistic work — budget for it" | ⚠️ **Largely a download.** MultiBLiMP covers EN/FR/TR; TurBLiMP adds 18,000 Turkish items (§6.3) |
| `tier1_plan.md` §3.6 | Log spacing justified by "Dumas et al. … first 10% of tokens" | ⚠️ **Misattributed** — arXiv:2601.22851 is **Körner, Müller-Eberstein, Korhonen & Plank**. The 10% figure itself is **unconfirmed against the primary source** |
| `tier1_plan.md` §5 | PELT/BOCPD then bootstrap over 5 seeds for a CI on Δt | ⚠️ **Not a valid composition** — the bootstrap is inconsistent for a changepoint location (§4.1). And 5 seeds caps any exact paired test at p = 0.0625 (§4.2) |
| `CLAUDE.md`, `research_standards.md` §8 | `run_with_cache` raises on batch size > 1 | ⚠️ **Stale.** Scoped to `generate(return_cache=True)`; the batching issue closed 2026-04-22 |
| `research_standards.md` §8 | penzai dormant | ✅ true, **and additionally JAX-only** — disqualifying for a PyTorch project |
| `research_standards.md` §8 | transformer-debugger abandoned | ✅ in substance; ⚠️ **not archived** — last commit is a pre-commit pin bump |
| `research_standards.md` §8 | "There is no library for tracking interp metrics across a checkpoint grid" | ⚠️ **Possibly superseded** by TRACE (arXiv:2507.03668) |
| `registry.md` super weights | (implied) super-weight formation is unstudied | ✅ **confirmed** by three independent zero-result searches |
| `registry.md` / secondary sources | "Super weights emerge gradually, beginning early in pre-training" | ⚠️ Traces to **Kovaleva 2021 on BERT LayerNorm outliers**, not super weights |
| `registry.md` super weights | TowerBase/TowerInstruct share the same super weight | ⚠️ **Replicates a published observation** (Yu et al. report identical coordinates for instruction-tuned variants). The quantified sharpening is the novel part, at n=1 |

---

## 9. Ranked reading list

Ordered by how likely each is to change a decision, not by importance to the
field. Every one was chosen because it undercuts something rather than confirming
it.

**1. Houghton & Kapatsinski, *Exemplars in Disguise: Pure Exemplar Models Mimic
Abstraction-First Learning*, arXiv:2608.00821 (1 Aug 2026).** [verified]
Read before anything else, because it is five days old and it attacks the measure
Stage 2 is built on. Look for: the asymmetry between the two onset criteria
(relative rank test vs absolute threshold), and the simulation showing pure
memorisers reproduce the abstraction-first ordering. If it holds, Stage 2 needs a
symmetric onset criterion before it can claim anything.

**2. Körner, Müller-Eberstein, Korhonen & Plank, *When Meanings Meet*,
arXiv:2601.22851, EACL 2026 Main.** [verified]
Your research question with a causal method and a published null structure
(`en_en` / `tgt` / `src_unpatched`), across 26 EuroLLM checkpoints. Look for three
things: that *"language-specific concept spaces do not strongly precede the
emergence of shared spaces"* — the plan's implicit ordering hypothesis, already
tested once and not supported; the checkpoint-granularity warning that the
phenomenon is over before 48B tokens; and the claim-hygiene warning that *"some
apparent gains in translation quality reflect shifts in behavior … rather than
improved translation ability."* Also confirm the "first 10% of tokens" figure
here or drop it.

**3. Gröger, Wen & Brbić, *Revisiting the Platonic Representation Hypothesis: An
Aristotelian View*, arXiv:2602.14486, ICML 2026.** [verified]
Look for the O(d/n) width confounder and the √log M depth confounder, and for the
result that after calibration CKA, RSA, Procrustes and SVCCA **lose** the
convergence trend while neighbourhood measures retain it. At d/n ≈ 0.5 with a
36-cell layer scan, this study sits in exactly the confounded regime. The drop-in
package is `pip install calibrated-similarity`. Read with *Back into Plato's
Cave* (arXiv:2604.18572), which disagrees about whether mutual-kNN is the good
metric or the fragile one — the disagreement is the most informative thing in the
current literature.

**4. Tigges, Hanna, Yu & Biderman, *LLM Circuit Analyses Are Consistent Across
Training and Scale*, arXiv:2407.10827, NeurIPS 2024.** [verified]
The only real study of circuits over training, and its finding is the opposite of
the intuitive one: **component identity is unstable across checkpoints while the
algorithm is stable.** Look for the name-mover head that "suddenly stops
exhibiting this behavior," and for the fact that they **exclude Pythia-70M** —
your exact configuration — because it "does not learn the task."

**5. Blevins, Gonen & Zettlemoyer, *Analyzing the Mono- and Cross-Lingual
Pretraining Dynamics of Multilingual Language Models*, EMNLP 2022,
arXiv:2205.11758.** [verified]
The closest existing study, and it is an **encoder**, which by itself refutes
"this line of work needs a decoder." 39 checkpoints still live at
`nlp.cs.washington.edu/xlmr-across-time/` with an HF conversion script. Look for
two design-changing findings: the step at which transfer emerges **differs by
language pair** (so a single Δt across EN–FR and EN–TR is the wrong estimand),
and **final-layer performance degrades over training while knowledge migrates to
lower layers** (so a fixed-layer readout across 60 checkpoints measures layer
drift as much as learning).

**6. Sassenhagen & Draschkow, *Cluster-based permutation tests of MEG/EEG data do
not establish significance of effect latency or location*, Psychophysiology 2019,
e13335.** [verified]
Eight pages, no mathematics, and it describes precisely the error the headline
claim is at risk of. Look for the mechanism by which the apparent onset moves
earlier as power increases — which means **a noisier measure appears to
transition later for that reason alone**, in exactly the direction that would
manufacture the H1 result. The authors admit to having made the error themselves.

**7. Du, Zeng, Dong & Tang, *Understanding Emergent Abilities from the Loss
Perspective*, arXiv:2403.15796, NeurIPS 2024.** [verified]
The strongest rebuttal to the plan's central move. Look for: they ran Schaeffer's
own Brier score across checkpoints and **still got a tipping point** at loss ≈ 2.2
on four datasets in two languages; and their caveat that a uniform predictor
scores Brier 0.75, so pre-threshold improvement is calibration, not competence.
Read with Schaeffer et al.'s own follow-up (arXiv:2406.04391), where the mirage
authors concede continuous surrogates do not restore predictability.

**8. *Mechanisms vs. Outcomes: Probing for Syntax Fails to Explain Performance on
Targeted Syntactic Evaluations*, arXiv:2506.16678.** [discovery — verify first]
A published negative aimed at H1: across **32 models, no probe yielded a
significant regression fit** against downstream syntactic accuracy. Look for what
exactly was regressed on what, and whether the null extends to the continuous
probe measures the plan uses rather than only to accuracy. If it holds, the
alignment→behaviour link that H1 presupposes is contested on the plan's own task.

**9. Minder, Dumas, Juang, Chugtai & Nanda, *Overcoming Sparsity Artifacts in
Crosscoders*, arXiv:2504.02922.** [verified]
Read only if crosscoders stay in scope — but then read it first. Look for
**Complete Shrinkage**: the L1 penalty can zero a decoder vector for a model in
which the feature is genuinely present, which means "this feature appeared at
checkpoint t" is partly an artifact of the penalty. 18% of "chat-only" latents
fall inside the central 95% of the shared distribution. Note that the closest
prior art (Crosscoding Through Time) cites this paper, thanks its first author,
and still trains L1 at λ=2.

**10. Gurnee et al., *Verbalizable Representations Form a Global Workspace in
Language Models*, arXiv:2607.15495 (16 Jul 2026).** [verified]
Not for the global-workspace claim but for the **J-lens** inside it: a
deterministic closed-form per-layer Jacobian average, cost **linear in d_model**,
so ~30 A100-hours for the whole grid at 512d. Look for the honest limitations —
"an imperfect tool, which we believe only approximately and incompletely captures
the model's underlying workspace structure," and that it "only identifies vectors
associated with concepts that correspond to single tokens," which is a real
problem for Turkish morphology. Zero published validation below Claude scale.

**11. Weiss/JHU-CLSP, *Seq vs Seq: An Open Suite of Paired Encoders and
Decoders* (Ettin), arXiv:2507.11412.** [discovery — verify first]
Because it may make part of the study unnecessary: paired encoder-only and
decoder-only models, **17M–1B, identical data and recipe, 250+ checkpoints, with
batch-level training data per checkpoint.** Look for whether the recipe controls
what H3a needs controlled, and whether the 17M and larger siblings bracket 36M
closely enough to serve as a pilot or an external validity check.

**12. Asparouhov & Muthén, *Multiple-Group Factor Analysis Alignment*,
Structural Equation Modeling 2014 — with the skeptical evaluation in Studies in
Educational Evaluation 2025 (S0191491X25000768).** [discovery — verify first]
The outside-field entry. Psychometrics has spent decades on "are two groups'
latent constructs comparable," with a graded invariance ladder
(configural/metric/scalar/strict), per-item DIF diagnostics, **published
interpretive thresholds** for similarity coefficients, and — via Mokken invariant
item ordering — a **direct test of whether items are acquired in the same order
across replicates**. Look for whether the invariance ladder gives a better-graded
vocabulary for interlingua claims than a single similarity scalar, and for
longitudinal invariance testing, which asks whether a measure means the same
thing at step 100 as at step 100,000 — a question a 60-checkpoint design
otherwise assumes away.

---

## 10. Verification debt

Items that must be confirmed before they enter any findings doc, per `CLAUDE.md`
claim hygiene:

1. **The "first 10% of tokens" figure** in `tier1_plan.md` §3.6 — attribution
   corrected, number still unconfirmed against the primary source.
2. **Every [discovery]-flagged arXiv ID** in this document. IDs are as they
   appeared in search listings; none of those papers were opened.
3. **The seven older cross-lingual citations** the sweep did not re-verify:
   Libovický et al. 2020, Tang et al. 2024, Wang/Minervini/Ponti 2024, Transfer
   Neurons 2025, Mondal et al. 2025, Timkey & van Schijndel 2021, Chi et al. 2020.
4. **MAFEX / Tokenization–Morphology Misalignment** — UNVERIFIED, and it is a
   load-bearing threat to the Turkish arm if real.
5. **Whether TRACE (arXiv:2507.03668) actually does what its abstract claims** —
   it would supersede a documented gap in `research_standards.md` §8.
6. **The 2025–2026 sweep for the retrieval / centroid / anisotropy areas is
   thin** — the responsible agent exhausted its budget before covering them.
   Treat as incomplete coverage, not evidence of absence.

*Compiled from nine area surveys and three discovery passes, 2026-08-06. The
surveys are stronger as audits of named methods than as exhaustive sweeps; six of
nine exhausted their web-search budget and finished by direct source fetching.*
