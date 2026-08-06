# Method landscape — unfiltered candidate list

**Companion to [`method_landscape.md`](method_landscape.md).** That document is
the synthesis; this one is the raw output of three **discovery-only** sweeps run
2026-08-06, preserved so nothing is lost. Roughly a third of these items made it
into the synthesis; the rest are recorded here because they were surfaced once
and would otherwise have to be found again.

⚠️ **Read the provenance caveat.** These agents were instructed to spend their
budget on *search breadth*, not verification: **do not read papers, do not verify
deeply, list candidates.** Consequently:

- **Every arXiv ID here is as it appeared in a search listing. None of these
  papers were opened.**
- One-line descriptions are paraphrases of abstracts or search snippets.
- Items marked **UNVERIFIED** are ones the agent itself doubted.
- Anything promoted from this file into a findings doc, the registry, or a paper
  **must be opened and confirmed first**, per `CLAUDE.md` claim hygiene.

Each sweep was given an explicit exclusion list of what the nine area surveys had
already covered, so this is deliberately biased toward things the main survey
missed.

---

## 1. Representational similarity and alignment

### 2026 work

- **Adjusted Similarity Measures and a Violation of Expectations** — arXiv:2601.10641 — generalizes the "adjustment operator" (as in adjusted Rand / Cohen's kappa) to arbitrary null models, with sufficient conditions for it to actually produce a mean-0-under-null measure. A permutation-calibrated similarity is an instance of this operator; this says when the calibration silently breaks.
- **Bridging Functional and Representational Similarity via Usable Information** — arXiv:2601.21568 — recasts the representational-vs-functional gap in terms of V-usable information. The cleanest formal framing found of "does the mechanistic measure track the behavioral one."
- **Logit Distance Bounds Representational Similarity** — arXiv:2602.15438 — proves a bound linking output-space distance to linear representational similarity, i.e. an analytic expectation for how tightly a behavioral measure *can* lag a representational one.
- **Measuring the Representational Alignment of Neural Systems in Superposition** — arXiv:2604.00208 — closed-form results that superposition systematically **deflates** RSA, linear CKA and linear regression, and that more-compressed systems can look *more* aligned. Promoted to the synthesis (§1.2).
- **Decoding Alignment without Encoding Alignment: a critique of similarity analysis in neuroscience** — arXiv:2605.05907 — decoding-based and encoding-based alignment come apart.
- **Beyond Activation Alignment: The Geometry of Neural Sensitivity** — arXiv:2605.03222 — compares systems by Jacobian/sensitivity geometry rather than activations. A genuinely different measure family.
- **Beyond Prediction Accuracy: Target-Space Recovery Profiles** — arXiv:2605.20127 — replaces a scalar alignment score with a per-dimension recoverability profile plus a reproducibility-based ceiling. The "which dimensions are consistently recoverable across independent runs" framing is a seed-level null.
- **A Unifying Framework for Concept-Based Representational Similarity** — arXiv:2606.09653.
- **Scalable and Interpretable Representation Alignment with Ordinal Similarity (TSI / QSI)** — arXiv:2606.16379 — triplet/quadruplet ordinal-consistency indices; claimed outlier-robust and cheap.
- **Discriminative Capacity of Representational Similarity Measures** — arXiv:2509.04622 — scores measures by their ability to *separate model families* (d′, silhouette, ROC-AUC). A way to pre-screen which measure can distinguish architecture arms at all.
- **What Representational Similarity Measures Imply about Decodable Information** — arXiv:2411.08197.
- **Evaluating Representational Similarity Measures from the Lens of Functional Correspondence** — arXiv:2411.14633 — eight measures scored on trained-vs-untrained discrimination and behavioral agreement; linear CKA and Procrustes win, linear predictivity is weak.
- **The Umwelt Representation Hypothesis: Rethinking Universality** — arXiv:2604.17960 — a 2026 counter-proposal to Platonic/universality convergence claims.
- **Normalized Space Alignment (NSA)** — arXiv:2411.04512 — differentiable metric-space-respecting alternative to CKA, usable as both measure and loss.
- **Representation Alignment Rests on Linear Structure** — arXiv:2605.28870.
- **Contrastive-Difference CKA** — arXiv:2606.16897 — a CKA variant designed to compare **across architectures**.
- **Objective drives the consistency of representational similarity across datasets** — arXiv:2411.05561 — similarity conclusions are dataset-dependent; relevant because three languages are three datasets.
- **Model Alignment Search** — arXiv:2501.06164 — causal counterpart to representational similarity (learned alignment subspaces validated by intervention).
- **Convergence Without Understanding: When LMs Agree on Representations but Disagree on Reasoning** — arXiv:2605.23315 — representational convergence without behavioral convergence.
- **ICLR 2026 Re-Align workshop + "Re-Align Challenge"** — representational-alignment.github.io/2026/ — ships reference implementations of comparison measures plus a leaderboard. Worth mining as a code source.
- **Relative representations enable zero-shot latent space communication** — arXiv:2209.15430; **Improving Relative Representations with Learned Anchors and Whitened Inner Products** — arXiv:2605.30596 — anchor-relative encodings invariant to latent isometry by construction.
- **CKA finite-sample bias** attributed in search results to **Murphy et al. (2024)** and **Cloos et al. (2025)** — ⚠️ **exact IDs unknown, worth chasing.**
- **Cross-lingual Similarity of Multilingual Representations Revisited** — arXiv:2212.01924 — argues common similarity indices are misleading for cross-lingual comparison; proposes ANI.

### Dynamics-based similarity (compare computation, not geometry)

- **Dynamical Similarity Analysis (DSA)** — Ostrow et al., NeurIPS 2023, likely arXiv:2306.10168 (UNVERIFIED) — compares systems via Hankel/Koopman delay-embedded linear operators.
- **DSA can identify compositional dynamics developing in RNNs** — arXiv:2410.24070 — detects structure *developing during training*.
- **Beyond DSA: Conjugacy-based Comparison of Dynamical Systems** — arXiv:2607.04493 — 2026 critique arguing DSA's orthogonal alignment is the wrong equivalence class.
- **Fast dynamical similarity analysis (fastDSA)** — arXiv:2511.22828.
- **KoopSTD** — OpenReview 29eZ8pWc8E — Koopman similarity with timescale decoupling, which matters when comparing early vs late checkpoints.
- **A Spectral-Grassmann Wasserstein metric for operator representations** — arXiv:2509.24920.
- **Comparing noisy neural population dynamics using optimal transport distances** — arXiv:2412.14421 — built for noisy trajectories.

### Cross-lingual-native isomorphism measures (predating the CKA canon)

- **The Secret is in the Spectra** — arXiv:2001.11136 — Laplacian-eigenvalue distance.
- **Are All Good Word Vector Spaces Isomorphic?** — arXiv:2004.04070.
- **IsoVec** — arXiv:2210.05098.
- Gromov–**Hausdorff** distance (distinct from Gromov-Wasserstein, which the main survey covers).

### Common-vs-distinctive decomposition (chemometrics / genomics)

- **JIVE — Joint and Individual Variation Explained** — arXiv:1102.4110 — decomposes multiple blocks on the same subjects into joint + block-specific + noise. Literally "interlingua + language-specific residual" as an estimable decomposition.
- **AJIVE** — ScienceDirect S0047259X1730204X; **DIVAS** — arXiv:2212.00703 — principal-angle successors with **significance testing on the joint rank**.
- **Estimating shared subspace with AJIVE: the power and limitation of multiple data matrices** — arXiv:2501.09336 — theory on when shared-subspace estimation works.
- **DISCO-SCA, OnPLS, RegularizedSCA** — PLOS One 10.1371/journal.pone.0037840; Behav Res Methods 10.3758/s13428-018-1163-z.

### Neuroscience beyond RSA

- **Crossnobis (cross-validated Mahalanobis) distance** — Walther et al. / Diedrichsen lab — an **unbiased** estimator whose null is exactly zero rather than positive-biased.
- **Pattern Component Modelling (PCM)** — PLOS Comp Biol 2017, 10.1371/journal.pcbi.1005508 — likelihood-based **model comparison** over representational hypotheses.
- **Noise ceilings** — Nili et al.; addendum bioRxiv 2020.03.23.003046 — needed to say "alignment saturated" rather than "alignment stopped rising."
- **Hyperalignment / connectivity hyperalignment / Shared Response Model** — Haxby et al., eLife 2020 (56601); hybrid hyperalignment S1053811921002524 — per-subject maps into a common space with **held-out generalization testing**.
- **Spectral Riemannian Alignment Score (S-RAS)** — UNVERIFIED, title/ID unknown — log-spectral distance on SPD manifolds used to recover *corresponding layers* across independently trained networks.

### Other cross-field imports

- **PROTEST — Procrustean randomization test** — Jackson, Écoscience 1995 — permutation test of Procrustes fit, shown more powerful than the Mantel test. See also "Much beyond Mantel", PLOS One 10.1371/journal.pone.0101238.
- **Baselga beta-diversity partitioning** (turnover vs nestedness) — Global Ecology and Biogeography 2010 — the right decomposition if you want to distinguish "French features replaced by Turkish ones" from "Turkish features are a subset."
- **Cophenetic correlation**; **Tanglegrams Are Misleading for Visual Evaluation of Tree Congruence** — MBE 2019, 10.1093/molbev/msy196.
- **An Information-Geometric Distance on the Space of Tasks** — Gao & Chaudhari, ICML 2021, PMLR v139 — Fisher-Rao length of the trajectory weights travel. **Trajectory-native**, unlike snapshot comparisons.
- **Fisher–Rao distance closed forms** — arXiv:2304.14885; Information Geometry 10.1007/s41884-024-00143-2.
- **Bures–Wasserstein / affine-invariant Riemannian metrics on covariance matrices** — MDPI Mathematics 13(13):2157; barycenters arXiv:2302.14618 — handle rank-deficiency without regularization.
- **MMD kernel two-sample testing in high dimension** — arXiv:2109.14913 (studentized MMD + power theory), arXiv:2105.03425, arXiv:2601.19755, arXiv:2605.12089 — **the only family found that comes with power theory.**
- **Specification curve / multiverse analysis** — Simonsohn et al.; arXiv:2605.19745; PMC12875576.

---

## 2. Psychometrics — measurement invariance, DIF, invariant item ordering

The highest-value cluster in the discovery round: a mature literature on "are two
groups' latent constructs comparable, and do they emerge in the same order."
**All [discovery] — none opened.**

- **Alignment optimization** — Asparouhov & Muthén, *Structural Equation Modeling* 2014 — group-specific loadings/intercepts under an approximate-invariance penalty, so only **configural** invariance is needed to compare latent means across many groups.
- **A critical evaluation of alignment optimization … international large-scale assessments** — Studies in Educational Evaluation 2025, S0191491X25000768 — read *with* the method.
- **Evaluating measurement invariance … PISA 2022: MGCFA vs alignment** — 10.1007/s10639-024-12921-7 — head-to-head across many language groups.
- **The configural / metric / scalar / strict invariance ladder** — standard MGCFA; a *graded* notion of comparability.
- **DIF detection family** — Mantel-Haenszel, SIBTEST, IRT-LR, logistic regression, Rasch trees — reviewed in Frontiers in Education 2025, 10.3389/feduc.2025.1595658. Per-item diagnostics rather than one global scalar.
- **Regularized / LASSO DIF (GPCMlasso)** — PMC12126468 — penalized simultaneous DIF across many items and grouping variables, **with built-in multiplicity handling.**
- **On the Complex Sources of Differential Item Functioning** — Educational & Psychological Measurement 2026, 10.1177/00131644251379802 — when the cause is multi-source (script, typology and corpus confound at once).
- **Longitudinal measurement invariance / response-shift detection** — e.g. 10.1111/jnp.12269 — does a measure mean the same thing at time t as at t′.
- **Second-order latent growth curve models with across-group invariance constraints** — 10.1007/s12564-023-09907-4 — models a growth trajectory of a *latent* construct while testing invariance across groups and occasions.
- **Tucker's congruence coefficient** + Lorenzo-Seva & ten Berge cutoffs — Methodology 2006 — a factor-similarity index **with published interpretive thresholds** (0.95–1.00 equal, 0.85–0.94 fair).
- **Mokken scale analysis / invariant item ordering** — R `mokken`, `check.iio`; methods MIIO, MS-CPM, IT; HT coefficient — nonparametric IRT test of whether items are acquired in the **same order** regardless of who is measured.
- **Attribute Hierarchy Method / cognitive diagnostic models with learning progressions** — Leighton, Gierl & Hunka — *infers* a prerequisite DAG over latent skills.
- **Network Comparison Test** — van Borkulo et al., CRAN `NetworkComparisonTest` — permutation test for whether two estimated networks differ.
- **A graph-theory based similarity metric for subpopulation psychometric networks** — Psychological Methods 2026, 10.1037/met0000625.
- **IRT for LLM/benchmark evaluation** — PSN-IRT arXiv:2505.15055; JE-IRT arXiv:2509.22888; DualEval arXiv:2606.26429; option-level psychometrics arXiv:2608.02966 — **IRT ability estimates are far more sensitive than mean accuracy**, which matters when the behavioral measure is low and noisy.

---

## 3. Ordering two noisy event times — imports from other fields

**The gap this addresses: none of the covered changepoint methods produces a CI
on the *difference* of two changepoints, which is the headline quantity.**

- **Stress–strength reliability, R = P(X < Y)** ("stochastic precedence") — reliability engineering — the exact target of "does M emerge before B", as an **effect size in [0,1]** rather than a significance test.
- **Doubly interval-censored / coarse-data incubation-period estimation** — Reich et al., *Stat. Med.* 2009; arXiv:2310.04225 — a log-spaced grid makes every emergence time interval-censored by construction.
- **Lead–lag estimation (Hoffmann–Rosenbaum–Yoshida)** — arXiv:1303.4871; noise-robust/non-synchronous extensions arXiv:2601.01871, arXiv:2002.00724 — point estimate **and CI for the lag itself**.
- **Thermal optimal causal path** — Sornette & Zhou, S0164070405000844 — non-parametric estimation of a **time-varying** lag.
- **Event coincidence analysis** — Donges et al., arXiv:1508.03534 — built for short/sparse event sets.
- **Convergent cross mapping / causalized CCM** — Sugihara et al.; NSF-PAR 10522387 — ⚠️ included with the caveat that CCM underperforms on stochastic series, which may disqualify it.
- **Systems Factorial Technology: Mean and Survivor Interaction Contrasts** — Townsend & Nozawa; tutorial TQMP 2016 12(1); R `sft` — psychophysics' answer to "did A finish before B started", diagnosed from the *sign* of a double-factorial interaction contrast.
- **Hierarchical Bayesian approach to distinguishing serial and parallel processing** — S0022249617301037 — pools across few subjects (≈ few seeds).
- **Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs** — arXiv:2603.14681 — **irregular observation grids and group structure**, which is the study's shape.
- **Bayesian hierarchical change-point model with parameter constraints** — PMC8980247 — per-unit change points with order constraints, yielding a **posterior over the difference between two change points**.
- **`mcp` — regression with multiple change points** — lindeloev.github.io/mcp/ — posteriors on locations and Bayes factors for the number of change points.
- **Kendall's W / Borda–Kendall rank aggregation with Friedman χ²** — tests whether an ordering is concordant across replicates before reporting a consensus order.
- **latenZy / latenZy2** — bioRxiv 2025.06.30.662308 — non-parametric, binning-free estimation of onset **and of the time two conditions diverge**, with permutation inference.
- **MEG/EEG latency-measure primer and toolbox** — Frontiers in Neuroscience 2018, 10.3389/fnins.2018.00765 — ⚠️ states plainly that onset latency is **biased toward earliest onsets and unreliable under noise with slow rises**.
- **Temporal generalization matrices** — King & Dehaene, *TiCS* 2014, PMID 24593982 — train-at-t / test-at-t′ decoding distinguishes a change in the **ordering** of stages from a change in intensity or duration.

### Sequential / evidence-accumulation designs

- **Bayes Factor Design Analysis** — Schönbrodt & Wagenmakers, 10.3758/s13423-017-1230-y; informed-prior tutorial 10.3758/s13428-018-01189-8 — answers "is 5 seeds enough" quantitatively *before* running.
- **Bayes Factor Group Sequential Designs** — arXiv:2601.02851 — interim analyses with Bayes factors; fits "run 2 seeds, look, decide."
- **Anytime-valid testing with e-values and confirmatory adaptive designs** — arXiv:2606.00878.
- **Anytime validity is free: inducing sequential tests** — Koning & van Meer, *JRSS-B* 2026 (qkag050), arXiv:2501.03982 — an anytime-valid version of **any** fixed-n test **at no power cost at the terminal sample size**.
- **Game-theoretic statistics and safe anytime-valid inference** — Ramdas et al., arXiv:2210.01948.
- **How Many Random Seeds? Statistical Power Analysis in Deep RL** — arXiv:1806.08295.
- **Covariance Correction for Permutation Statistics in Multiple Testing** — arXiv:2604.06915 — permutation multiplicity control under **non-exchangeability**.
- **Knockoffs-based FDR control for deep networks** — arXiv:2606.04404.

---

## 4. Emergence, phase transitions, developmental timing

- **When Do Attention Circuits Form?** — arXiv:2606.02378 — circuits form in the first **0.3–2.1%** of training; induction and BOS-attractor transitions separated by **10–20× in tokens**; per-head spectral signal *precedes* capability threshold crossing. ⚠️ Single-author, 16 solo 2026 preprints, no code.
- **Predicting the Emergence of Induction Heads in Language Model Pretraining** — arXiv:2511.16893.
- **Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns** — arXiv:2606.25010 — emergence timing is **stochastic across seeds**; if true it bounds what 5 seeds can resolve.
- **A Pre-Training Analogue of Grokking: Tracing Delayed Grammatical Generalization** — arXiv:2606.00230.
- **Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining** — arXiv:2606.26050 — rules can be **lost**, so alignment measures may be non-monotonic.
- **Evidence of Phase Transitions in Small Transformer-Based Language Models** — arXiv:2511.12768 — transitions in *small* models, visible via dispersion/KL/vocabulary probes but **not** in loss.
- **TRACE: Tracking the Emergence of Semantic Representations in Transformers** — arXiv:2505.17998 — combines loss-landscape spectral curvature, intrinsic dimensionality and linguistic-category alignment. ⚠️ Distinct from the TRACE toolkit at arXiv:2507.03668 — **name collision.**
- **Hidden Breakthroughs in Language Model Training** — arXiv:2506.15872 — unsupervised detection of breakthroughs **invisible in the aggregate loss curve**, by decomposing loss change along parameter-space directions.
- **Persistent Entropy as a Detector of Phase Transitions** — arXiv:2602.09058 — validated on Kuramoto, Vicsek *and* NN training; uses **collapse of across-realization variability** as the signature.
- **Tracking Representation Dynamics in LLMs with Persistent Homology** — arXiv:2606.19542 — dense-checkpoint TDA; different objectives give distinguishable topological trajectories **despite similar behavior**.
- **Emergence via Phase Transitions: Mechanism Landscapes and Universal Convergence** — arXiv:2606.07563.
- **Spectral Entropy Collapse as a Phase Transition in Delayed Generalisation** — arXiv:2604.13123 — claims a **predictive precursor**.
- **Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment** — arXiv:2508.20015 — template for defining an order parameter.
- **Feature Repulsion and Spectral Lock-in: Two-Layer Network Grokking** — arXiv:2605.08119.
- **Influence Dynamics and Stagewise Data Attribution** — arXiv:2510.12071.
- **Spectral Reach: Understanding Neural Scaling as Progress into the Spectral Tail** — arXiv:2605.31244.
- **Subspace Chronicles: How Linguistic Information Emerges, Shifts and Interacts during LM Training** — arXiv:2310.16484 — syntax probes saturate early, semantics keep improving.

---

## 5. Seeds, stability, reproducibility

- **Unstable Features, Reproducible Subspaces: Seed Dependence in SAEs** — arXiv:2606.12138 — individual features unstable, **subspaces reproduce**.
- **Quantifying LLM Attention-Head Stability: Implications for Circuit Universality** — arXiv:2602.16740 — ⚠️ **middle-layer heads are least stable yet most representationally distinct**; weight decay improves stability. Middle layers are where cross-lingual alignment is reported to live.
- **Certified Circuits: Stability Guarantees for Mechanistic Circuits** — arXiv:2602.22968 — 40 independent seeds, IoU stability guarantees.
- **Toward a Theory of Generalizability in LLM Mechanistic Interpretability Research** — arXiv:2509.22831.
- **Many Circuits, One Mechanism: Input Variation and Evaluation Granularity** — arXiv:2606.06267.
- **The Model Organism Lottery** — arXiv:2607.01033 — 54 model organisms, 7 training techniques; interpretability conclusions **flip with the training pipeline**. Direct threat to a from-scratch study.
- **Mechanistic Interpretability as Statistical Estimation: A Variance Analysis (EAP-IG)** — arXiv:2510.00845 — single-input causal-mediation scores are high-variance random variables.
- **On the Extreme Variance of Certified Local Robustness Across Model Seeds** — arXiv:2601.13303 — seed-only variance exceeding published effect sizes.
- **BlackboxNLP 2026 Special Track on Reproducibility and Reliability in Interpretability Analyses** — EMNLP 2026, Budapest, 29 Oct 2026.

---

## 6. Validity of interpretability measures

- **Mechanisms vs. Outcomes: Probing for Syntax Fails to Explain Performance on Targeted Syntactic Evaluations** — arXiv:2506.16678 — ⚠️ across **32 models, no probe yielded a significant regression fit** against downstream syntactic accuracy. Promoted to the synthesis (§1.6, reading list #8).
- **Can Interpretation Predict Behavior on Unseen Data?** — arXiv:2507.06445 — the affirmative pole.
- **Rigorous Interpretation Is a Form of Evaluation** — arXiv:2605.05508.
- **Pando: Do Interpretability Methods Work When Models Won't Explain Themselves?** — arXiv:2604.11061 — **planted decision trees** for feature-level causal ground truth.
- **Make Mechanistic Interpretability Auditable** — arXiv:2606.00033 — standards-layer paper, possibly useful for `research_standards.md`.
- **Statistical suggestions for mech interp research and beyond** — AI Alignment Forum, ID unknown — informal, but the only item found targeting nulls/CIs/multiplicity in mech interp directly.
- **MIB: A Mechanistic Interpretability Benchmark** — arXiv:2504.13151, ICML 2025 — ⚠️ **SAE features are not better than raw neurons**; attribution/mask-optimization wins circuit localization; supervised DAS wins causal-variable localization.
- **Findings of the BlackboxNLP 2025 Shared Task** — arXiv:2511.18409 — what actually worked when many teams attacked MIB.
- **Disentangling Polysemantic Neurons with a Null-Calibrated Polysemanticity Index** — arXiv:2508.16950 — an explicit null-calibrated index; the design pattern for "every measure beats its random-init value."

---

## 7. Dictionary learning, transcoders, model diffing

- **SAE-Track / Tracking the Feature Dynamics in LLM Training** — arXiv:2412.17626 — **warm-started continual SAE series across checkpoints**; the practical answer to 900 independent trainings.
- **Persistent Sparse Autoencoders: Learning Feature Timescales** — arXiv:2607.17117.
- **Matryoshka Sparse Autoencoders** — arXiv:2503.17547 — sharply lower feature absorption and splitting.
- **Kronecker Factorization Improves Efficiency and Interpretability of SAEs** — arXiv:2505.22255.
- **From Directions to Regions: Decomposing Activations via Local Geometry** — arXiv:2602.02464.
- **From Geometric Recovery to Causal Validation: A Reproducible Audit of SAE Features** — arXiv:2607.12166 — audits the stack through to "causal inertness."
- **CE-Bench** — arXiv:2509.00691; **Improving Robustness in SAEs via Masked Regularization** — arXiv:2604.06495.
- **SPARC: concept-aligned SAEs for cross-model and cross-modal interpretability** — 2026, ID unknown, **UNVERIFIED**.
- **Lorsa — Low-Rank Sparse Attention** — arXiv:2504.20938, ICLR 2026 — replaces MHSA with overcomplete single-dimension-OV heads; recovers cleaner induction heads and sinks.
- **Beyond Dense States: Elevating Sparse Transcoders to Active Operators** — arXiv:2602.01695.
- **A Unified Theory of Sparse Dictionary Learning** — arXiv:2512.05534 — piecewise biconvexity and spurious minima; covers SAEs, transcoders **and crosscoders**.
- **Reference Feature Atlases for Mechanistic Auditing** — arXiv:2607.22570 — train one sparse library on a reference panel, attach a new model by fitting **only a linear decoder** — demotes the per-target fit from non-convex-overcomplete to roughly convex.
- **Cross-Architecture Model Diffing with Crosscoders** — arXiv:2602.11729 — the missing method for comparing architecture arms with a shared dictionary.
- **Delta-Crosscoder** — arXiv:2603.04426 — fixes crosscoder failure under *narrow, asymmetric* change, which is what adjacent checkpoints are.
- **fmxcoders** — arXiv:2605.09438 — ⚠️ standard crosscoder latents are "functionally driven by only one or two layers", i.e. cross-layer sharing is largely cosmetic.
- **Sparse Crosscoders for diffing MoEs and Dense models** — arXiv:2603.05805; **Group Crosscoders** — arXiv:2410.24184.
- **Simple LLM Baselines are Competitive for Model Diffing** — arXiv:2602.10371 — a cheap null any crosscoder result must beat.
- **Beyond the Leaderboard: Performance Disparities via Model Diffing** — arXiv:2509.18792; **Localizing RL-Induced Tool Use to a Single Crosscoder Feature** — arXiv:2606.26474; **Med-SegLens** — arXiv:2602.10508; **Building and evaluating model diffing agents** — LessWrong 2026.

### Parameter/weight-space methods (arm-agnostic by construction)

- **Attribution-based Parameter Decomposition (APD)** — arXiv:2501.14926, Apollo.
- **Stochastic Parameter Decomposition (SPD)** — arXiv:2506.20790 — scalable, hyperparameter-robust successor; works on GPT-2-small.
- **Decomposition of Small Transformer Models** — arXiv:2511.08854 — parameter decomposition **specifically at small scale**.
- **Individual Parameters in Weight-Sparse Transformers Appear Interpretable** — arXiv:2607.02964.
- **Sparse Weight Decomposition for Efficient Circuit Extraction** — arXiv:2608.03913 — this month, **UNVERIFIED**.
- **Language Model Circuits Are Sparse in the Neuron Basis** — arXiv:2601.22594.

---

## 8. Tooling and harnesses (2025–2026)

- **TRACE: Training and Inference-Time Interpretability Analysis for Language Models** — arXiv:2507.03668, EMNLP 2025 demo `2025.emnlp-demos.62` — modular **in-training** analysis: probing, intrinsic dimensionality, Hessian curvature, layer-wise diagnostics, convergence-based early stopping. Paper claims existing tools *"lack temporal tracking."*
- **Interpreto: An Explainability Library for Transformers** — arXiv:2512.09730, ACL 2026 demo.
- **TDHook: A Lightweight Framework for Interpretability** — arXiv:2509.25475.
- **reward-lens** — arXiv:2604.26130 — notable for a **ten-method adapter protocol that isolates architecture-specific details**, so lens/patching/SAE modules are written once.
- **CLT-Forge** — arXiv:2603.21014, `LLM-Interp/CLT-Forge` — CLT training + auto-interp + attribution-graph visualization.
- **OpenMOSS Language-Model-SAEs** — distributed SAE/CLT/Crosscoder/Lorsa framework; 2026 adds **Complete Replacement Models**.
- **Neuronpedia + the `Attribute` library**; **Qwen-Scope** (open SAE suite over Qwen).
- **Mechanistic Interpretability Workshop @ ICML 2026** — mechinterpworkshop.com, 10 Jul 2026, Seoul — ⚠️ **the accepted-poster list including 23 spotlights is online and was never enumerated.** Highest-density unexplored source for 2026 methods.

---

## 9. Checkpoint suites and substrates

- **Ettin / "Seq vs Seq: An Open Suite of Paired Encoders and Decoders"** — arXiv:2507.11412, JHU-CLSP — **paired encoder-only and decoder-only, 17M–1B, identical data and recipe, 250+ checkpoints, batch-level training data per checkpoint.** Promoted to the synthesis.
- **SimpleStories V2** — arXiv:2504.09184 + HF `SimpleStories/*` — 1.25M–134M, checkpoints on W&B; the 35M model is essentially this size.
- **TinyModel** — 4-layer 44M on TinyStories V2, ReLU, **no layernorms**, ships trained **SAEs and transcoders**.
- **T5Gemma 2** — Google 2026 blog release — rare recent encoder-decoder release with pretrained checkpoints.
- **"Should We Still Pretrain Encoders?"** — ICLR 2026, OpenReview `jpz7e3jhRq`.
- **Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder LLMs** — arXiv:2510.26622 — controlled scaling comparison 150M–8B on matched data.
- **Return of the Encoder: Maximizing Parameter Efficiency for SLMs** — arXiv:2501.16273 — encoder-decoder advantages **at ≤1B scale**.
- **Encoder-Decoder Gemma** — arXiv:2504.06225 — adapting decoder-only into encoder-decoder.
- **Encodings of Source Syntax: Similarities in NMT Representations Across Target Languages** — arXiv:2005.08177 — ⚠️ encoder representations vary by **target** language, so an encoder-decoder arm's "source representation" is not target-independent.
- **BabyLM 2026 + BabyBabelLM** — call arXiv:2602.20092; BabyBabelLM arXiv:2510.10159 / EACL 2026 `2026.eacl-long.152`; eval repo `babylm-org/babylm-eval` — 2026 adds a multilingual track and **mandates checkpoint submission at 1M/10M/100M words**.
- **Bringing Up a Bilingual BabyLM** — arXiv:2603.29552 — small-scale controlled bilingual pretraining.
- **When Is Multilinguality a Curse?** — arXiv:2311.09205, EMNLP 2024 — **10,000+ models up to 45M params**; quantifies the capacity tradeoff at this exact scale.
- **ATLAS: Adaptive Transfer Scaling Laws** — ICLR 2026, ID unknown, UNVERIFIED.
- **Apertus** — arXiv:2509.14233 — intermediate checkpoints on separate HF branches, 1800+ languages, ~40% non-English.
- **EuroLLM-22B** — checkpoints every 10B tokens; **Salamandra** — 2/7/40B, 35 European languages, UNVERIFIED whether intermediates are released.
- **Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging** — arXiv:2602.03702, Kempner — 2026 evidence on WSD-style schedules.
- **WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging** — arXiv:2507.17634 — comparable intermediate checkpoints without decay branches.
- **On the impact of pretraining data ordering in transformer encoder- and decoder-only LMs** — Knowledge-Based Systems 2026, S0950705126005769.

---

## 10. Multilingual and cross-lingual, 2026

- **On the limited utility of parallel data for learning shared multilingual representations** — arXiv:2603.29026 — ⚠️ parallel data has only **minimal** effect on alignment. Bears on H3a's premise.
- **When Language Representations Interact: Separability and Cross-Lingual Effects** — arXiv:2606.14347 — causal-inner-product geometry over 28 bilingual contrasts; languages largely separable with simplex-like family structure.
- **A Shared Geometry of Difficulty in Multilingual Language Models** — arXiv:2601.12731, UNVERIFIED.
- **Neither Here Nor There: Cross-Lingual Representation Dynamics of Code-Mixed Text in Multilingual Encoders** — arXiv:2603.19771 — rare **encoder-focused** 2026 work.
- **Language-Switching Triggers Take a Latent Detour Through Language Models** — arXiv:2605.18646, UNVERIFIED.
- **Cross-Lingual Activation Steering** — arXiv:2601.16390 — ⚠️ transfer works via **functional divergence**, with gains correlating with **increased** language-cluster separation.
- **Multilingual Steering by Design: Multilingual SAEs and Principled Layer Selection** — arXiv:2605.23036 — effective steering occurs where alignment and separability **coexist**; gives an intersection-based layer criterion without a sweep.
- **Multilingual Language Models Encode Script Over Linguistic Structure** — arXiv:2604.05090 — ⚠️ script dominates; Latin-script EN/FR/TR makes this a live confound.
- **Explainability and Interpretability of Multilingual LLMs** (survey) — EMNLP 2025 `2025.emnlp-main.1033`.
- **Finding the Translation Switch** — arXiv:2601.11019; **Beyond Transfer Accuracy: Faithful Circuits for Controlled Low-Resource Adaptation** — arXiv:2601.08146; **Benchmarking Concept-Spilling Across Languages** — arXiv:2601.12549; **LangFIR** — arXiv:2604.03532; **SAE-LAPE** — arXiv:2507.11230.
- **CRANE** — 2026, ID unknown, UNVERIFIED — redefines language specificity as **functional necessity** via neuron-level intervention rather than activation statistics. Mirrors this repo's own causal-ranking lesson.
- **Multilinguality of LLMs From a Structural Perspective** — arXiv:2606.01800.
- **Predicting Multilingual Classification and Translation Performance with Cross-Lingual Alignment — Is English Enough?** — arXiv:2608.03446 — one week old; tests whether alignment predicts downstream performance in **encoder-only and decoder-only** models.
- **(title unknown) removing language-sensitive principal components improves cross-lingual alignment** — arXiv:2603.18863, UNVERIFIED title.
- **Mix, Don't Tune: Bilingual Pre-Training Outperforms Hyperparameter Search in Data-Constrained Settings** — arXiv:2605.13225.
- **Multilingual Knowledge Transfer under Data Constraints via Lexical Interventions** — arXiv:2605.23885.
- **The Role of Mixed-Language Documents for Multilingual LLM Pretraining** — arXiv:2601.00364, ACL 2026 — bilingual documents are 2% of the corpus yet removing them drops BLEU 56%; **parallel data restores 91%, code-switching contributes minimally.**
- **Brain–LLM Alignment Tracks Training Data, Not Typology** — arXiv:2605.23032 — ⚠️ an apparent typological effect explained by data volume. Same confound shape as "Turkish aligns later."
- **Can Embedding Similarity Predict Cross-Lingual Transfer? A Systematic Study on African Languages** — arXiv:2601.03168.
- **Modality Matching Matters: Calibrating Language Distances for Cross-Lingual Transfer in URIEL+** — EACL SRW 2026 `2026.eacl-srw.8` — calibrated typological distances as a covariate.
- **Cross-Attention is Half Explanation in Speech-to-Text Models** — arXiv:2509.18010 — ⚠️ cross-attention explains only ~50% of input relevance; bounds cross-attention as an alignment measure.
- **Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability** — Findings of ACL 2026 `2026.findings-acl.502`.

---

## 11. Behavioral targets, morphosyntax, tokenizers

- **MultiBLiMP 1.0** — arXiv:2504.02768, TACL — **EN 770 / FR 2548 / TR 1742** agreement pairs. Promoted to the synthesis.
- **A Morphology-Aware Evaluation of Turkish Syntax in Large Language Models** — Başar & Bisazza, SIGTURK 2026 `2026.sigturk-1.9` — ⚠️ Turkish minimal-pair benchmarks are confounded by **morpheme count, subword count and sentence length**; tests tokenizer-morphology alignment as a performance proxy.
- **MAFEX (Morpheme-Aligned Faithful Explanations)** — 2026, ID unknown, **UNVERIFIED** — names **"Tokenization–Morphology Misalignment"**; reprojects attributions from token basis to morpheme basis; evaluated on Turkish LMs.
- **Different types of syntactic agreement recruit the same units within LLMs** — arXiv:2512.03676.
- **Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction** — arXiv:2510.24934 — ⚠️ aggregate curves hide **non-monotonic per-condition** dynamics.
- **On the Similarity of Circuits across Languages: Subject-verb Agreement** — Ferrando & Costa-jussà, arXiv:2410.06496, EMNLP 2024 Findings — a **language-independent "subject number" direction, causally transferable English→Spanish.**
- **MORPHOGEN** — arXiv:2604.18914 — gender-aware morphological generation, French among three languages.
- **Crosslingual Structural Priming and the Pre-Training Dynamics of Bilingual Language Models** — arXiv:2310.07929 — structural priming as a **behavioral** alignment measure tracked over bilingual pretraining.
- **TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior** — arXiv:2512.20757 — controlled artifacts disentangling tokenizer from architecture and data.
- **Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer** — arXiv:2510.06128.
- **The Impact of Vocabulary Overlaps on Knowledge Transfer in Multilingual MT** — arXiv:2605.04196.
- **False Friends Are Not Foes: Vocabulary Overlap in Multilingual LMs** — arXiv:2509.18750.
- **Explaining and Mitigating Crosslingual Tokenizer Inequities** — arXiv:2510.21909, NeurIPS 2025 — token premiums persist after controlling data and vocabulary size; identifies **pre-tokenization** as a cause.
- **Exploring Anisotropy and Outliers in Multilingual LMs for Cross-Lingual Semantic Sentence Similarity** — arXiv:2306.00458.
- **Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability** — arXiv:2603.19427.

---

## 12. Concentrated structure — additional candidates

- **Hidden Dynamics of Massive Activations in Transformer Training** — arXiv:2508.03616 — five-parameter exponentially-modulated log fit giving a **predictive functional form** for onset timing across Pythia.
- **The Spike, the Sparse and the Sink** — arXiv:2603.05498 — disentangles three usually-conflated phenomena; pre-norm is what couples them.
- **A Unified View of Attention and Residual Sinks** — arXiv:2601.22966 — argues sinks are **functionally necessary**, not artifacts.
- **Attention Sinks and Outliers in Attention Residuals** — arXiv:2605.17887, UNVERIFIED title.
- **Not a nuisance but a useful heuristic: Outlier dimensions favor frequent tokens** — arXiv:2503.21718 — outlier dims emerge early and encode a frequency prior, tying onset to token statistics.
- **Measuring Maximum Activations in Open LLMs** — arXiv:2605.15572, UNVERIFIED.
- **A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions** — arXiv:2605.18898 — extreme-value-style fit to weight tails; max-to-99th-percentile ratios up to 14.3×. **Closest thing found to a calibrated null for super weights.**
- **Spectral Signatures of Data Quality: Eigenvalue Tail Index as a Diagnostic** — arXiv:2603.27885 — Hill estimator with a sharp threshold.
- **Accuracy estimation of neural networks by extreme value theory** — arXiv:2511.00490 — GPD fit to the error tail; the general EVT-on-networks template.

---

*Produced 2026-08-06 by three discovery-only agents with explicit exclusion lists.
Unverified by construction. See [`method_landscape.md`](method_landscape.md) §10
for the verification debt this file adds to.*
