# Project bootstrap: Understanding Translation in Decoder LLMs as a Foundation for Compression

## Context: read this first

- `pruning_project.pdf` — my prior work applying IFR-guided
  pruning + LoRA/KD recovery + GPTQ to Aya Expanse 8B. Read this to
  understand my current capabilities and what I've already done.
  This work is **not** the scaffold for the new project — it's prior
  art I'll build on.

## What this project actually is

The end goal is a novel compression method for multilingual LLMs used
for machine translation, with a focus on quantization and possibly
pruning + recovery. **But the project does not start with the method.**
It starts with understanding.

The thesis spine is:

> Before proposing a compression method, deeply investigate **how the
> translation task is actually carried out** inside several open-source
> decoder-style LLMs used for MT. Use interpretability methods to
> identify which components, features, circuits, attention heads, and
> weights participate in translation, and how the task's footprint
> differs across model architectures. In parallel, build a strong
> systems-level understanding (transformer math, hardware, memory,
> kernels, deployment). Once both pictures are mature, propose
> compression methods whose hypotheses are grounded in what
> interpretability revealed about MT-critical structure, not in
> generic literature recipes.

Phase one is the investigation. Phase two is the method. The prompt
below is for phase one only — phase two will be designed once phase
one's findings stabilize.

## Models under investigation in phase one

Three open-source models, deliberately chosen for architectural contrast:

1. **Aya Expanse 8B** (Cohere) — decoder-only, multilingual-by-pretraining,
   generates translations as ordinary continuation. The model from my
   prior paper. Translation here is a *circuit embedded in a general
   LM*.
2. **NLLB** (Meta) — encoder-decoder, *trained specifically for
   translation*, covers ~200 languages. The "translation circuit" here
   is essentially the whole model. Pick the variant that's most
   tractable on A100 80GB (likely NLLB-200 1.3B distilled or 3.3B);
   document the choice.
3. **Omnilingual** (Meta) — recent multilingual model covering 1,600+
   languages. Use the text-to-text MT variant if available; if the
   only available checkpoints are speech-to-text, use the text decoder
   side and document the limitation. Alternative: a recent peer
   multilingual model if Omnilingual proves intractable.

The deliberate contrast: how does the same translation task look when
the model was *built around* translation (NLLB), *built to scale across
many languages* (Omnilingual), and *built as a general LM that happens
to translate* (Aya)? Each model is one data point; the comparison is
the value.

**Architectural note for the implementation.** Some interpretability
methods (IFR, logit lens) were developed on decoder-only models.
Adapting them to NLLB's encoder-decoder structure requires care — the
"decoding step" needs a clear definition, and signals computed on the
encoder vs decoder are different objects. Flag and document these
adaptations explicitly per method per model.

## Interpretability methods

A core set is implemented carefully and validated against each other on
shared examples before any single method's findings are trusted.
Additional methods are added when a specific investigative question
requires them.

**Core set (implement and validate first):**
- **Logit lens / tuned lens** — layer-by-layer "what would the model
  predict if we decoded from this hidden state?" Cheap diagnostic for
  where the model commits to a target-language token.
- **Activation patching** — replace activations from a clean MT
  prompt with those from a corrupt prompt at specific layers/heads;
  measure causal effect on the translation. Gold-standard causal
  method.
- **Information Flow Routes (IFR)** — Ferrando & Voita 2024.
  Familiar ground from my prior work. Extend to NLLB and Omnilingual
  with documented adaptations.
- **Probing classifiers** — train small classifiers on hidden states
  to detect language identity, source vs target representation, etc.
  Cheap, complements the heavier methods.

**Add when needed (do not implement preemptively):**
- Attribution patching (EAP, EAP-IG) — when a question needs
  edge-level attribution at scale.
- Sparse autoencoders — if pretrained SAEs become available for any
  of the three models, or if probing reveals features worth isolating.
- Direct logit attribution — when we need to decompose final logits
  by layer/component.
- Path patching — when activation patching's coarseness is the
  bottleneck.

**Validation discipline.** Before trusting any finding from a single
method, cross-check it with at least one other method on the same
examples. Document where methods agree and where they disagree —
disagreement is informative, not a failure.

## Investigative questions for phase one

Phase one is organized around questions, not tracks. Each question
has: methods used, expected artifacts (notebooks, plots, writeups),
the systems/hardware concept embedded in answering it, and a
"satisfied when" criterion that, taken together with the others,
gates phase two.

### Q1. Where in the model does the source language get represented as a
language-agnostic meaning, and where does target-language generation
begin?

The classic "concept space" question for multilingual models, applied
to MT specifically. Methods: probing classifiers for language ID across
layers, logit lens to find where target-language token probabilities
become dominant, IFR to trace flow.

Embedded learning: residual stream geometry, the "logit lens" view of
what each layer is computing, what "language identity" actually means
as a feature.

Satisfied when: I can produce, for each model, a layer-by-layer chart
showing language identity emergence and target-language commitment,
and explain the differences across the three architectures.

### Q2. Which attention heads are MT-critical, and what do they do?

Methods: head-level activation patching on clean (correct translation)
vs corrupt (lexically-similar wrong translation) prompts; head
ablation; attention pattern visualization.

Embedded learning: attention math from scratch (Q/K/V derivation,
multi-head structure, why softmax temperature matters), attention head
specialization findings in the literature (induction heads,
copy-suppression heads, etc.) and whether MT-critical heads fit any
known taxonomy.

Satisfied when: for each model, I have a ranked list of MT-critical
heads with at least an informal characterization of what each does
(source attender, target predictor, language router, etc.), and I
can explain why patching that head breaks translation.

### Q3. Which MLPs and layers carry the cross-lingual mapping?

Methods: layer-level activation patching, IFR's MLP scoring, and
optionally probing classifiers on MLP outputs.

Embedded learning: MLP-as-key-value-memory framing (Geva et al.),
why FFN width matters, how MLPs participate in next-token prediction
distinct from attention.

Satisfied when: for each model, I can characterize where in the depth
the cross-lingual translation work concentrates, and explain how this
differs between the dedicated MT model (NLLB) and the general LM (Aya).

### Q4. How does the translation task's footprint differ across the three
architectures?

The synthesis question. Methods: comparison across Q1–Q3 findings;
shared MT examples translated by all three models with parallel
interpretability traces.

Embedded learning: encoder-decoder vs decoder-only architecture
trade-offs, how training objective shapes internal structure, what
"the translation circuit" even means as a unit of analysis.

Satisfied when: I can write a short paper-style comparison
(`compression/docs/q4_architecture_comparison.md`) that someone unfamiliar
with the project could read and understand which structural facts about
MT inside LLMs are general vs architecture-specific.

### Q5. Of the components identified as MT-critical, which carry numerically
sensitive information, and which carry information that's robust to
perturbation?

This is the bridge to phase two. Methods: weight-perturbation studies
(add Gaussian noise of varying magnitude to MT-critical vs MT-irrelevant
components, measure quality drop); inverse Hessian-style sensitivity
analysis on MT calibration data; activation magnitude analysis on
MT-critical components.

Embedded learning: why some weights tolerate quantization and others
don't, what GPTQ/AWQ/LeanQuant are actually measuring when they
"protect" weights, the distinction between component *importance*
and component *quantization sensitivity* (these are not the same).

Satisfied when: I have a per-component map of (importance × sensitivity)
for at least Aya, and a hypothesis-shaped writeup
(`docs/archive/phase2_hypotheses.md`) about what compression strategies the
findings suggest. This document is the seed of phase two.

## Systems and hardware learning track

This is **not a separate track** — it is interleaved into Q1–Q5.
However, a `docs/systems-notes/` folder collects the writeups so the
learning is visible and reusable. Required notes by the time Q5 is
satisfied:

- `transformer-math.md` — attention, MLPs, residual stream, layer norm,
  position encodings, all derived from scratch with the notation I'd
  use to defend the thesis.
- `gpu-memory.md` — what lives in VRAM during a forward pass (weights,
  activations, KV cache); during training (add gradients, optimizer
  state); how this scales with batch and sequence length.
- `attention-at-the-hardware-level.md` — Q/K/V matmul → softmax →
  attention output, with memory access patterns; what FlashAttention
  changes and why.
- `kv-cache.md` — what the KV cache is, how it grows, why it dominates
  inference memory for long contexts, the implications for KV
  quantization.
- `kernels-and-deployment.md` — int4/int8 GEMM kernels (Marlin,
  CUTLASS), why some quantization methods have first-class kernels and
  others don't, the vLLM scheduler / PagedAttention.
- `model-storage.md` — safetensors, GGUF, HF format; what's actually in
  a model file at the byte level; how quantized formats encode scales
  and zero-points.
- `aws-deployment.md` — practical notes on running LLM inference on
  AWS (instance types with GPUs, EBS vs instance store, networking
  for multi-GPU). Lighter than the others; satisfies the "I'm learning
  AWS at work anyway" goal without bloating the project.

Each note should be the result of *doing*, not just reading — derived
from the actual experiments where possible.

## What to produce

### 1. Repo scaffold

A directory layout supporting investigation-first work:

```
.
├── README.md
├── archive/phase1_plan.md          # the investigation plan
├── docs/
│   ├── archive/project_summary.md  # 1-page synthesis of prior work + new direction
│   ├── findings/           # writeups per question (Q1–Q5)
│   ├── systems-notes/      # the systems/hardware notes track
│   └── learning/learning_log.md     # running personal notes
├── experiments/
│   ├── q1-language-emergence/
│   ├── q2-attention-heads/
│   ├── q3-mlps-and-layers/
│   ├── q4-architecture-comparison/
│   └── q5-importance-vs-sensitivity/
├── notebooks/              # exploratory work
├── src/
│   ├── models/             # loaders for Aya, NLLB, Omnilingual
│   ├── interp/             # core interpretability methods
│   │   ├── logit_lens.py
│   │   ├── activation_patching.py
│   │   ├── ifr.py
│   │   └── probing.py
│   ├── data/               # MT calibration data, clean/corrupt pair generators
│   └── eval/               # BLEU / chrF++ / COMET wrappers (reused from prior work)
├── data/                   # gitignored
├── models/                 # gitignored — checkpoints
├── configs/
└── scripts/
```

Justify any non-obvious choices in the README.

### 2. archive/phase1_plan.md

The investigation plan as a working document. Sections:

- Context (1 paragraph linking to archive/project_summary.md)
- The five investigative questions, in their own words
- For each question: methods, expected artifacts, the satisfied-when
  criterion, and which models the question will be answered for
- The systems-notes track and how it interleaves
- Risk register: what could make a question unanswerable, and what
  the fallback plan is per question
- Phase 2 placeholder — explicitly empty, "to be designed from
  Q5 findings"

### 3. Stub code for the core interpretability methods

`compression/src/interp/logit_lens.py`, `compression/src/interp/activation_patching.py`,
`compression/src/interp/ifr.py`, `compression/src/interp/probing.py` — stubs with clear
docstrings, type signatures, and TODOs marking the implementation work.
For IFR specifically, port what makes sense from my prior code (assume
I'll show you that code when ready); for the others, write stubs that
make the API obvious and let me fill in the math.

For each stub: a brief comment block at the top citing the canonical
paper and noting any architecture-specific adaptations needed (e.g.,
"NLLB encoder-decoder: define `position` as decoder step; encoder
contributions accessed via separate hook").

### 4. Per-question experiment scaffolds

Under `compression/experiments/qN-*/`, for each of Q1–Q5:

- `README.md` restating the question, the methods, the satisfied-when
  criterion, and the embedded learning
- `experiment.py` stub
- `notes.md` — empty, for as I work
- `configs/` — at least one config stub per model the question covers

### 5. Project summary doc

`docs/archive/project_summary.md` — written first, before any of the scaffold —
demonstrating that you've understood the prior work and the new
direction. One page. If your understanding here is wrong, I want to
catch it before the scaffold is built.

## Constraints and preferences

- **A100 80GB** is the assumed hardware target. Note when an experiment
  would need more (e.g., NLLB-3.3B is fine; loading multiple models
  simultaneously may not be).
- **uv** for Python env management. Lock file in repo.
- HuggingFace Transformers is the experimental scaffold. Use **TransformerLens**
  for Aya if it has support; document where TransformerLens does not
  cover NLLB/Omnilingual and what we use instead (likely raw HF hooks).
- Quantization libraries (gptqmodel, autoawq, etc.) are not needed in
  phase one — defer to phase two.
- vLLM is for deployment benchmarking later; not needed in phase one.
- **Embed learning into experiments.** Where you would normally write
  full implementations, write stubs with TODOs and design notes — I
  fill in the math myself for the learning value. Write full
  implementations for shared infrastructure where re-implementing
  offers no learning value (data loading, eval wrappers, plotting
  helpers).
- **Cite specific papers** in code comments when a design choice traces
  to one.
- When adapting interpretability methods across architectures, **flag
  and document the adaptation** rather than silently extending.
- **Validate methods against each other** before trusting findings.
  Build this discipline into the scaffold.
- If anything in this prompt conflicts with what you find in
  `pruning_project.pdf`, trust the paper and flag the conflict.

## What to do, in order

1. Read `pruning_project.pdf` end to end.
2. Write `docs/archive/project_summary.md` first. Do not produce anything else
   until this exists. If your understanding is off, I want to catch
   it here.
3. Stop and ask me to review (1)–(2) before producing the scaffold.
4. Once I confirm: produce the repo scaffold, archive/phase1_plan.md, the
   stub code, and the per-question experiment scaffolds.
5. Stop again before filling in any actual implementations.

Do not include timelines, week-by-week plans, or completion estimates
anywhere. The investigation is gated by satisfied-when criteria, not
calendar dates.

## Revisions to the original bootstrap prompt

### Language pairs

Replace the cs→de / en→es default with the WMT25 Model Compression
Shared Task round 2 set:

- **Czech → German (cs→de)** — shared script, shared family, heavy
  morphology on both sides. Carries continuity with my prior paper:
  use this pair as the sanity-check pair when validating new
  infrastructure against old results.
- **English → Chinese Simplified (en→zh-Hans)** — cross-script,
  cross-family, near-isolating target. Stresses cross-script
  interpretability methods and segmentation-sensitive evaluation.
- **English → Egyptian Arabic (en→ar-arz)** — cross-script + a
  non-standard variety. Particularly interesting because the model
  almost certainly saw far more MSA than arz in pretraining, so
  there's a register-shift question on top of the language-shift
  question. Where in the model does that distinction get made?

Evaluation caveats to document explicitly in `compression/src/eval/`:
- COMET wmt22-comet-da is trained mostly on standard varieties; it
  may misjudge Egyptian Arabic. Use the WMT25 task's prescribed
  metric protocol where available; report wmt22-comet-da alongside
  for continuity with my prior paper.
- For zh-Hans, segmentation choice affects BLEU/chrF++. Use
  sacreBLEU's `zh` tokenizer; document the choice.

### Q4 reframing — the shared-depth hypothesis

Q4 ("How does the translation task's footprint differ across the three
architectures?") needs an explicit hypothesis structure, because the
answer to this question is the bridge to phase two. The working
hypothesis is:

> Specific MT-critical components (which heads, which MLPs) will
> **not** generalize across Aya, NLLB, and Omnilingual — these are
> model-specific accidents of training. But the **depth profile** —
> where in the network MT-critical work concentrates relative to
> total depth — **may** generalize.

Three candidate strengths of this hypothesis to test:

- **V1 (weakest):** Early layers (~first 25%) are MT-irrelevant; final
  1–2 layers are protected; middle is where MT happens. Confirms
  general task-agnostic depth-pruning findings — not novel for MT.
- **V2 (medium):** MT has a characteristic depth signature (source
  understanding → language-agnostic semantics → target commitment)
  with similar relative depth fractions across architectures.
  Genuinely novel if true.
- **V3 (strongest):** The depth signature is consistent enough that
  bit-allocation or pruning decisions can be made from depth fraction
  alone, without per-model interpretability. Highest practical payoff.

A piece of evidence cutting *against* strong generalization: my prior
paper's IFR results showed somewhat different layer rankings between
cs→de and en→es within a single model. So even *within* one model,
the language pair affects the importance map. Q4 needs to be honest
about this and report findings that may falsify V2/V3 rather than
confirm them.

### docs/archive/phase2_hypotheses.md

Add this as a seed document listing candidate phase-two directions, so
the phase-one investigation knows what kinds of findings it's hunting
for. Do not commit to any of them — they are working candidates.

The four candidates:

- **Candidate A — Task-conditional mixed-precision allocation
  (layer-wise).** Use phase one's per-layer MT-importance findings
  to drive layer-wise bit-width assignment: MT-critical layers stay
  high-precision, MT-irrelevant layers go lower. Novelty is the
  signal (interpretability-derived, MT-specific), not the mechanism.
  Layer-wise mixed precision is kernel-friendly.

- **Candidate B — Component-level mixed-precision (head/MLP within
  layer).** More aggressive version of A. Bigger potential Pareto
  win, much harder kernel story. Likely a stretch goal or future
  work.

- **Candidate C — Task-specific calibration data for existing
  quantizers.** Run GPTQ / AWQ / LeanQuant with MT parallel
  calibration data (or activations from MT-critical components)
  instead of C4. Orthogonal to A and B, lower-risk, addresses the
  WMT25 gap from `learning/annotated_bibliography.md` directly.

- **Candidate D — Depth-profile-driven, model-agnostic prior.** If
  Q4 finds that depth profiles generalize across architectures
  (V2/V3), formulate bit-allocation or pruning candidacy as a
  function of depth fraction, validated across all three models.
  Strongest claim; cleanest fallback (degrades to Candidate A if
  generalization fails).

For each candidate, a paragraph noting:
- What phase-one findings would *support* this candidate
- What findings would *kill* it
- What baselines it would be evaluated against
- Approximate kernel/deployment risk

Phase two will pick from these (or invent new ones) once phase one's
findings are in.