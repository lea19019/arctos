# Phase one — investigation plan

## Context

Phase one is an interpretability-led investigation that precedes any compression method. Thesis spine, model choices, language pairs, and methodological discipline are in [`docs/project-summary.md`](docs/project-summary.md). Prior work is `docs/pruning_project.pdf`. Phase-two candidate directions are seeded in [`docs/phase2-hypotheses.md`](docs/phase2-hypotheses.md) and stay frozen until Q5 closes.

## Models

- **Aya Expanse 8B** (Cohere; decoder-only, multilingual-by-pretraining) — translation as embedded behavior in a general LM. Carries continuity with the prior paper.
- **omt-llama-8b** (Meta Omnilingual MT, Llama-class, ~1,600+ languages) — MT-purpose-built at extreme language scale.
- **TowerInstruct-7B** (Unbabel, Llama-2 base, ~10 languages) — MT-purpose-built at moderate scale via mixed monolingual + bilingual CPT then MT-task SFT.

All three are Llama-class decoder-only, so a single TransformerLens-based hooking layer applies across the three. Per-model adaptations (Cohere attention scaling on Aya, omt-llama's tokenizer, Tower's SFT prompt template) are flagged per method, per model.

## Language pairs (WMT25 round 2)

- **cs→de** — sanity-check pair; continuity with prior paper.
- **en→zh-Hans** — cross-script, cross-family, near-isolating target.
- **en→ar-arz** — cross-script, non-standard variety; register-shift on top of language-shift.

## The five questions

### Q1 — Where does language identity emerge, and where does target-language generation begin?

**Methods.** Probing classifiers across layers for source/target-language identity; logit lens / tuned lens for layer-by-layer next-token distributions; IFR to cross-check the layer-flow ranking.

**Models covered.** All three.

**Embedded learning.** Residual stream geometry; logit-lens math (final RMSNorm + lm_head applied to mid-layer hidden states); what "language identity" means as a probeable feature; selectivity vs raw probe accuracy (Hewitt & Liang 2019).

**Expected artifacts.** Per-model layer charts (probe accuracy + selectivity, target-token mass under logit lens) under `experiments/q1-language-emergence/`; synthesis in `docs/findings/q1.md`.

**Satisfied when.** For each model and language pair: a chart and a sentence describing (a) the layer at which source-language identity becomes linearly decodable, (b) the layer at which target-language token mass dominates under logit lens, (c) the relationship between (a) and (b). Disagreements between probing and logit lens are reported, not hidden.

### Q2 — Which attention heads are MT-critical, and what do they do?

**Methods.** Head-level activation patching on (clean, corrupt) MT prompt pairs (clean = source the model translates correctly; corrupt = source designed so the gold target differs lexically); head ablation as a coarser cross-check on top-ranked heads; attention-pattern visualization on top heads.

**Models covered.** All three.

**Embedded learning.** Q/K/V derivation from scratch; multi-head structure; why softmax temperature matters; attention-head taxonomies (induction heads, name-mover, copy-suppression) and whether MT-critical heads fit any of them.

**Expected artifacts.** Per-model ranked head list with patching effect size; informal characterization (source-attender, target-predictor, language-router, ...) for top ~10 heads; `docs/findings/q2.md`.

**Satisfied when.** For each model: a ranked list with informal characterizations for the top heads, and an explanation grounded in attention patterns of why patching each top head breaks the translation. Where multiple corrupt-prompt strategies disagree on the ranking, the instability is reported.

### Q3 — Which MLPs and layers carry the cross-lingual mapping?

**Methods.** Layer-level activation patching at `mlp_out` and `resid_post`; IFR's MLP scoring; optional probing on MLP outputs.

**Models covered.** All three.

**Embedded learning.** MLP-as-key-value-memory framing (Geva et al. 2021); FFN width and how it interacts with quantization; how MLPs participate in next-token prediction distinct from attention.

**Expected artifacts.** Per-model depth profile of layer-level MLP MT-importance (one column per method, so disagreement is visible); cross-model comparison in `docs/findings/q3.md`.

**Satisfied when.** Per-model characterization of where in the depth the cross-lingual translation work concentrates, with explicit reporting of any case where IFR and activation patching disagree on the layer ranking.

### Q4 — How does the translation footprint differ across the three architectures? (The shared-depth hypothesis.)

The synthesis question. **Working hypothesis:**

> Specific MT-critical components (which heads, which MLPs) will **not** generalize across Aya, omt-llama, and Tower — these are model-specific accidents of training. But the **depth profile** — where in the network MT-critical work concentrates relative to total depth — **may** generalize.

Three claim strengths to test:

- **V1.** First ~25% of layers MT-irrelevant; final 1–2 layers protected; middle is where MT happens. Confirms task-agnostic depth-pruning convergence — not novel for MT.
- **V2.** A characteristic depth signature (source understanding → language-agnostic semantics → target commitment) with similar relative depth fractions across architectures. Genuinely novel if true.
- **V3.** Depth signature is consistent enough that bit-allocation or pruning can be made from depth fraction alone, without per-model interpretability. Highest practical payoff.

**Methods.** Synthesis across Q1–Q3 on the same shared MT examples, with all three models traced apples-to-apples.

**Models covered.** All three jointly.

**Embedded learning.** How training intent (general LM vs MT-purpose-built) shapes internal structure; what "the translation circuit" even means as a unit of analysis when it spans the whole network; the difference between architectural similarity (all three are Llama-class) and functional similarity.

**Expected artifacts.** `docs/findings/architecture-comparison.md` — paper-style writeup pitched at someone unfamiliar with the project.

**Satisfied when.** The writeup exists and explicitly lands on V0 / V1 / V2 / V3 with evidence per language pair × model cell. **Falsifying evidence is required to be reported.** The prior paper's IFR results showed different layer rankings between cs→de and en→es within Aya alone — Q4 must be honest about findings that cut against V2/V3, not paper over them.

### Q5 — Importance vs quantization sensitivity

The bridge to phase two.

**Methods.** Weight-perturbation studies (Gaussian noise of varying magnitude on MT-critical vs MT-irrelevant components, measured by translation quality drop); inverse-Hessian-style sensitivity on MT calibration data (the diagonal SparseGPT/LeanQuant use, but computed on MT pairs rather than C4); activation magnitude analysis on MT-critical components (per AWQ).

**Models covered.** Aya at minimum (continuity with prior paper); the other two if compute permits — multi-model loading is the constraint.

**Embedded learning.** Why some weights tolerate quantization and others don't; what GPTQ/AWQ/LeanQuant actually measure when they "protect" weights; the **distinction between component importance and component quantization sensitivity** — these are not the same thing.

**Expected artifacts.** Per-component (importance × sensitivity) map for at least Aya; `docs/phase2-hypotheses.md` updated from a seed doc into a hypothesis-shaped writeup the phase-two design draws from.

**Satisfied when.** A per-component map exists, and `docs/phase2-hypotheses.md` has been updated with phase-one evidence supporting or killing each candidate.

## Systems and hardware notes

Interleaved into the questions; collected under `docs/systems-notes/`. Required by the time Q5 closes:

- `transformer-math.md` — attention, MLPs, residual, RMSNorm/LayerNorm, position encodings; derived from scratch.
- `gpu-memory.md` — what lives in VRAM during forward / training; KV cache scaling.
- `attention-at-the-hardware-level.md` — Q/K/V matmul → softmax → attention output, memory access patterns, what FlashAttention changes.
- `kv-cache.md` — growth, dominance at long contexts, KV-quantization implications.
- `kernels-and-deployment.md` — int4/int8 GEMM kernels (Marlin, CUTLASS), why some quantizers have first-class kernels, vLLM scheduler / PagedAttention.
- `model-storage.md` — safetensors, GGUF, HF format; quantized format encoding (scales, zero-points).
- `aws-deployment.md` — instance types, EBS vs instance store, multi-GPU networking.

Each note is the result of *doing*, not just reading.

## Testing discipline

Every method, loader, data generator, and metric wrapper that lands in `src/` ships with tests in `tests/` mirroring the source layout. Tests are first-class — pytest is a core dependency, not optional.

**Two levels, both required:**

- **CPU tests (`@pytest.mark.cpu`).** Run on a tiny dummy model — typically `gpt2` or a hand-built `HookedTransformer` config small enough to instantiate on CPU in a few seconds. Verify shape contracts, math invariants (e.g., the last-layer logit lens must equal the model's actual logits), and dataclass field contents. Must pass on a CPU-only machine; CI runs these by default.
- **GPU tests (`@pytest.mark.gpu`).** Run on a real target model (Aya / omt-llama / Tower) and check end-to-end behavior on a real MT prompt: e.g., that activation patching's effect direction is correct on a known clean/corrupt pair, that probe selectivity on the held-out split is non-zero, that IFR scores sum-to-one per token. Skipped automatically when CUDA is unavailable; opted into explicitly with `pytest -m gpu`.

A method is **not done** until both tiers exist. CPU tests catch shape and math bugs without burning GPU; GPU tests catch the real-model behavior the CPU dummies cannot exercise (Cohere attention scaling on Aya, omt-llama tokenizer quirks, Tower's prompt template).

The `cpu` / `gpu` / `slow` markers and conventions are configured in `pyproject.toml` and `tests/conftest.py`.

## Method validation discipline

Before trusting a finding from a single method, cross-check against at least one other on the same examples. Document agreement *and* disagreement; disagreement is informative.

Validation matrix (cell value = which question's findings are validated by the pair):

|                     | Logit lens | Activation patching | IFR | Probing |
|---------------------|:----------:|:-------------------:|:---:|:-------:|
| Logit lens          |     —      |          Q1         |  Q1 |   Q1    |
| Activation patching |     Q1     |           —         |  Q3 |   Q1    |
| IFR                 |     Q1     |          Q3         |  —  |    —    |
| Probing             |     Q1     |          Q1         |  —  |    —    |

## Risk register

| Risk | Question(s) | Fallback |
|------|-------------|----------|
| TransformerLens lacks coverage on a needed hook for one of the three models. | All | Drop to raw HF hooks for that model and intervention; document the adaptation per method. |
| Clean/corrupt prompt construction for MT does not produce a clean signal (paraphrased corrupt prompts often have the same gold translation). | Q2, Q3 | Use multiple corrupt-pair generators (lexical sub, language-id swap, target shuffle) and require activation-patching effect sizes to be stable across them before trusting any per-head ranking. |
| Multi-model loading exceeds 80 GB. | Q4 (some compositions), Q5 | Run sensitivity analyses sequentially per model; cache hidden states to disk to avoid simultaneous residency. |
| Probing classifiers overfit / leak — high accuracy without selectivity. | Q1 | Hewitt & Liang 2019 control-task setup; held-out language-pair test split; report selectivity, not raw accuracy. |
| Q2/Q3 patching effects don't agree with IFR ranking. | Q2, Q3, Q4 | Report the disagreement and investigate; do not paper over. The point of validation is to expose this. |
| Q4 V2/V3 hypothesis is falsified. | Q4 | Phase two collapses Candidate D back to Candidate A (per-model layer-wise) — the phase-two doc already documents this fallback. |
| Q5 sensitivity analysis budget too large for three models. | Q5 | Restrict the per-component map to Aya; do a coarser per-layer check on Tower / omt-llama only. Document the limitation. |
| COMET wmt22-comet-da misjudges en→ar-arz translations. | All | Report COMET alongside chrF++ and a small human-judgment spot check; document in `src/eval/`. |

## Phase 2

Empty by design. To be designed from Q5 findings against the candidates seeded in `docs/phase2-hypotheses.md`. New candidates may emerge from phase-one findings.
