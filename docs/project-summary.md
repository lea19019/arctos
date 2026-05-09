# Project summary

## Prior work

*Interpretability-Guided Layer Pruning for Efficient Multilingual Machine
Translation* (Castillo & Richardson, BYU) compressed **Aya Expanse 8B** for
cs→de and en→es with a four-stage pipeline: Information Flow Routes (IFR;
Ferrando & Voita 2024) for layer-importance scoring → layer pruning (fixed
count {8, 12, 16} and IFR-threshold variants τ = λ·μ) → LoRA recovery
optionally combined with sequence-level KD from Aya Expanse 32B → GPTQ 4-bit
quantization. With FT+KD recovery, IFR matched or beat the iterative
chrF++-ablation heuristic of Moslem et al. 2025 at 24 and 20 retained layers
on both pairs while being ~10× cheaper to compute (one forward pass over
200 examples vs ~392 held-out evaluations per configuration). The
IFR-threshold variant at λ=0.5 adaptively removed ~10 layers and reached
85.54 / 88.63 COMET — better than the matched fixed-count IFR runs. GPTQ
cost <1 COMET point in almost every condition, and the 16-layer GPTQ model
ran at 2671 tok/s vs 1308 baseline (2× faster, 4.54 GB on disk).

Two findings from that work motivate the new direction:

1. **IFR underperforms the iterative method at the most aggressive cs→de
   compression**, while still winning on en→es. IFR's averaged information
   flow does not capture some piece of late-layer importance that the
   iterative chrF++ ablation does — and that piece is *language-pair
   specific within a single model*. Whatever IFR is averaging away is the
   thing the new project most wants to see clearly.
2. **Both signals concentrate pruning in the middle of the network**
   (Jaccard 0.78 at 16 layers cs→de). This is the same convergence the
   broader depth-pruning literature reports across very different signals
   on generic calibration data — suggesting it may reflect task-agnostic
   redundancy more than MT-specific structure.

## New direction (not a pipeline extension)

The thesis spine: **before proposing a compression method, investigate how
the translation task is actually carried out inside several open-source
decoder-style LLMs used for MT.** Phase one is interpretability-led
understanding; phase two is a compression method whose hypotheses are
grounded in what phase one revealed. The annotated bibliography in
`research.md` is reference material *for phase two* — not the scaffold for
phase one. A pipeline-extension framing would skip the question this
project is built around.

## Phase one

**Three models — all decoder-only LLMs used for MT — chosen for contrast
in training intent and language scale.**

- *Aya Expanse 8B* (Cohere; multilingual-by-pretraining) — general
  multilingual LM in which translation is an embedded behavior, not a
  training target. Continuity with the prior paper.
- *omt-llama-8b* (Meta Omnilingual MT, Llama-class; ~1,600+ languages) —
  MT-purpose-built LLM at extreme language scale.
- *TowerInstruct-7B* (Unbabel, Llama-2 base; ~10 languages) —
  MT-purpose-built LLM at moderate scale. Well-documented mixed
  monolingual + bilingual CPT followed by MT-task SFT.

Holding architecture roughly constant (all Llama-class, all decoder-only)
removes one confound and lets Q4 isolate the effect of *training intent*
and *language scale* on the translation footprint.

**Three language pairs (WMT25 Model Compression round 2 set).**

- *cs→de* — sanity-check pair, carries continuity with the prior paper.
- *en→zh-Hans* — cross-script, cross-family, near-isolating target.
  Stresses cross-script methods and segmentation-sensitive evaluation.
- *en→ar-arz* — cross-script plus non-standard variety; the model has
  almost certainly seen far more MSA than arz, which adds a register-shift
  question on top of the language-shift question.

**Five investigative questions.**

- Q1 — Where does language identity emerge and target-language generation
  begin? (Logit/tuned lens, probing classifiers, IFR.)
- Q2 — Which attention heads are MT-critical, and what do they do?
  (Head-level activation patching, head ablation, attention pattern
  visualization.)
- Q3 — Which MLPs and layers carry the cross-lingual mapping?
  (Layer-level patching, IFR's MLP scoring, optional probing on MLP
  outputs.)
- Q4 — How does the MT footprint differ across the three architectures?
  Tested as an explicit *shared-depth hypothesis* with three claim
  strengths (V1: trivial endpoints; V2: characteristic depth signature;
  V3: depth-fraction is sufficient for compression decisions). The prior
  paper's own evidence — different layer rankings between cs→de and en→es
  *within Aya alone* — already cuts against V2/V3, and Q4 is designed to
  report findings honestly even if they falsify the strong claims.
- Q5 — Of MT-critical components, which carry numerically sensitive
  information vs information that's robust to perturbation? This is the
  bridge to phase two.

**Methodological discipline.** Each interpretability method is validated
against at least one other on shared examples before any single finding is
trusted; disagreement between methods is informative, not a failure.
Per-model adaptations (e.g., Cohere-specific attention scaling on Aya, any
tokenizer or position-encoding quirks on omt-llama) are flagged and
documented per method per model.

**Systems and hardware learning** is interleaved into Q1–Q5, with writeups
collected under `docs/systems-notes/` (transformer math, GPU memory,
attention at the hardware level, KV cache, kernels and deployment, model
storage, AWS notes). Each note is the result of *doing*, not just reading.

## Phase two

Deliberately deferred. `docs/phase2-hypotheses.md` is a seed document
listing four candidate directions — task-conditional layer-wise mixed
precision, component-level mixed precision, MT-specific calibration for
existing quantizers, and a depth-profile-driven model-agnostic prior — with
the phase-one findings that would support or kill each. Selection happens
once Q5 is satisfied; new candidates may emerge from phase-one findings.

## Constraints

A100 80GB hardware target; multi-model loading flagged where it would
exceed that. `uv` for Python env management. **TransformerLens** as the
hooking and intervention library across all three models (all are
Llama-class); HF Transformers underneath for tokenization and weight
loading. IFR is implemented from scratch in this repo (no port of the
prior paper's code). Quantization libraries and vLLM are phase-two
concerns. Investigation is gated by satisfied-when criteria, not calendar
dates.
