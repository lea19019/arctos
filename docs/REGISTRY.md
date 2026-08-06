# Registry — everything done in this repo

**Audited 2026-08-05.** A reconciled record of what was attempted, what was
established, what was ruled out, and where the documentation and the data
disagree. Built by five agents reading every doc, every experiment runner, and
recomputing headline numbers from the raw `.npz`/`.json` on disk.

## How to use this

Before starting anything, check three things here:

1. **[Ruled out](#ruled-out--do-not-redo)** — the most valuable section. Several
   plausible-sounding ideas are already dead with evidence.
2. **[Open and under-credited](#open-and-under-credited)** — things the backlog
   lists as unexplored that are actually half-done, and things nobody has claimed.
3. **[Discrepancies](#discrepancies--claims-that-do-not-survive-audit)** — do not
   cite a finding from the docs without checking it isn't listed here.

The headline results **do** reproduce. An independent recomputation from the raw
arrays matched the published IFR quarter-shares, the pivot crossover table, and
all 18 per-head Spearman cells exactly. The problems are in coverage, framing,
and claims that drifted from their evidence — not in the core measurements.

## Status at a glance

| Track | State | Where |
|---|---|---|
| Interpretability of MT (phase one) | Complete, written up as a paper | `compression/experiments/q1…q5`, `docs/findings/` |
| Compression method (phase two) | Complete, directional results only | `compression/experiments/q6-compression/` |
| PTQ-MT replication | Complete, 31 of 35 planned units | `compression/experiments/replication-uneven-ptq/` |
| NLLB + XTTS dubbing | Baseline sweep only | `speech-translation/` |
| Interlingua / training dynamics | Docs only, no code | `interlingua/` |

**Everything in phase two is n=24–32, greedy decoding, generic (non-chat)
prompt.** The repo's own docs label these "directional, not publication-grade."
Nothing here is paper-ready without a rerun.

---

# Track: compression

## Phase one — how MT works inside an LLM

Nine models (Aya-Expanse-8B, TowerBase-7B, TowerInstruct-7B, Tower-Plus-9B,
BLOOM-7B1, EuroLLM-9B, Llama-3.1-8B, Gemma-3-12B, NLLB-200-3.3B) × three FLORES+
pairs (cs→de, en→zh, en→arz), n=200.

| Q | Question | Status | Coverage |
|---|---|---|---|
| Q1 | When/where does the target language emerge? | ✅ Complete | 9 models (NLLB partial) |
| Q2 | Which attention heads are MT-critical? | ⚠️ **Partial** — attention viz only | 6 models, 1 prompt/pair |
| Q3 | Which MLPs/layers carry the mapping? | ❌ **Never ran** | none |
| Q4 | Does the signature generalize? | ✅ Synthesis (hand-written) | 8 models |
| Q5 | Importance vs quantization sensitivity | ✅ Complete | 6 models |

**Q2 and Q3 runners are stubs that raise `NotImplementedError`.** Q3 has no
results and no writeup and never did. `docs/findings/README.md` lists `q2.md` and
`q3.md`; neither has ever existed in git history.

### Established (verified against raw data)

- **Depth-staged translation.** Target-script logit-lens mass ≈ 0 through ~80–95%
  of depth, rising only in the final layers. IFR quarter-shares: first ≤12%,
  middle ~40–52%, last ~42–58%, for six of seven models.
- **Gemma-family inverts it.** Gemma-3: **92% first-quarter / ~3% last-quarter**.
  Tower-Plus (Gemma 2 base): 31/50/18. Attributed to extra pre/post-FFN
  layernorms plus logit softcapping — a mechanism stated but never tested.
- **CPT sets the structure, SFT refines.** TowerBase ≈ TowerInstruct within ≤1pp
  per quarter.
- **Pivot trajectory.** Models pass through a Latin/pivot representation and
  convert late; crossover depths verified in six cells.
- **Q5 null — the load-bearing negative.** Per-head |DLA| vs logit drop:
  **ρ = −0.025** (18 cells, t = −0.55). The stronger whole-layer chrF++ rerun,
  built specifically to rule out "the metric was too weak": **ρ = −0.065 (IFR),
  −0.046 (DLA)**. Same answer. *Importance does not predict sensitivity.*

### Weak or hedged by the docs themselves

- Source-language probing: all models 0.40–0.55 selectivity at every layer;
  differences of 0.05–0.10 are "real but not load-bearing."
- Target-language probing is **leaky** — the prompt names the target language.
- Q2 head characterizations come from **one prompt per pair**, top-6 heads,
  heuristic labels, no causal test.
- Cross-model DLA magnitudes are not comparable (Aya's are 4–10× Tower's).

## Phase two — compression

Framework: **find / keep / shrink / prune**, plus `calib`, `gptq`, `alloc`,
`pipeline` stages. All in `compression/experiments/q6-compression/experiment.py`.

### Established

- **MT-calibrated GPTQ ≫ generic-calibrated GPTQ** at W3, all 6 models with data:
  +19.6 to +31.7 chrF++, +0.13 to +0.52 COMET. **This is the contribution.**
- **Salient-channel FP16 preservation recovers much of the 3-bit cliff.**
  Gemma 12.7 → 48.4 chrF; Tower-Plus 31.5 → 49.2; EuroLLM 32.1 → 45.8.
- **Wanda ≫ magnitude pruning** at 50% sparsity (Tower-Plus 3.8 → 56.3).
- **MT-conditional Wanda calibration helps**: median **+2.18 chrF++, 19/24
  positive**. Calibration is a *pruning* lever.
- **W4 near-lossless; W3 a model-varying cliff; W2 and below collapse.**

### Super weights — priority section

Detected by **causal KL**, not activation-spike magnitude. Spike ranking produced
a false positive on Aya L31 (spike 732, ablation KL 4×10⁻⁷ — causally inert)
because the final layer's `down_proj` writes straight into the logits.

| Model | Layer | Ablation KL |
|---|---|---|
| EuroLLM-9B | 9 | **3.284** |
| TowerInstruct-7B | 1 | 1.252 |
| TowerBase-7B | 1 | 0.957 |
| Llama-3.1-8B | 1 | 0.243 |
| Tower-Plus-9B | 24 | 0.038 |
| Gemma-3-12B | 47 | 0.005 |
| Aya-Expanse-8B | 2 | 0.002 |
| BLOOM-7B1 | 0 | 0.001 |

Four facts that matter for any future use:

1. **Early layers (L0–L9) for every non-Gemma family.** Gemma-family is late and
   near-inert. Strength spans **3.5 orders of magnitude**.
2. **TowerBase and TowerInstruct share the *same* super weight** — L1, out 2533,
   in 7890, identical weight value 1.5390625 — and SFT only sharpens it
   (KL 0.96 → 1.25). *This is the only training-dynamics datapoint in the repo.*
3. **Decisive for removal, irrelevant for quantization.** Ablating one super
   weight: EuroLLM 57.9 → **4.8** chrF++. Ablating the **1000 largest-magnitude**
   weights: 60.0 (nothing). But FP16-preserving them during quantization does
   nothing at all — `rtn+SW ≈ rtn` on all 8 models.
4. **EuroLLM has a second, nearly-equal super weight** — L1 (out 750, in 9606),
   KL **2.975**, unmentioned in any writeup. Rank 3 drops to 0.0016, so EuroLLM
   has exactly two load-bearing scalars.

⚠️ **Do not treat "super weight" and "salient channel" as the same thing.** They
behave oppositely. `docs/advisor-brief.md` conflates them; it is the drifted doc.

⚠️ **The detector is greedy.** It takes `argmax` per layer and always returns a
candidate, so on a model with no spike it still reports one. There is no null or
threshold criterion, and only a smoke test — the detection logic has never been
verified against a known planted super weight.

### Negatives (all measured here unless noted)

| Result | Evidence |
|---|---|
| Fisher/Hessian mixed precision < uniform | Negative on every model × pair, both metrics. Aya cs-de −62.9 chrF / −0.611 COMET |
| Depth-pipeline does not localize fragility | crush_middle vs crush_ends a wash: 3 models favor middle, 3 favor ends, 1 ties |
| MT calibration does *not* help AWQ-style quant | 9/24 positive, median −0.62 chrF++ |
| Super-weight FP16 preservation ≈ no-op for quant | `rtn+SW ≈ rtn`, all 8 models |
| Nothing rescues sub-2-bit | 1% FP16 cannot save a model destroyed at ternary/binary |
| COMET unreliable below 2 bits | Scores degenerate output 0.20–0.65 where chrF++ is 0.0 |
| No healing-free PTQ reaches FP16 at 3-bit | *Literature*, not measured here (deep-research `wco17ovot`) |

## Replication — arXiv:2508.20893

WMT24++, n=960/direction, 6 languages × both directions, per-model chat
templates, greedy, COMET. **This is the only rigorous evaluation in the repo.**

| Claim | Verdict |
|---|---|
| C1 4-bit preserves quality for high-resource/large | ✅ reproduced |
| C2 Low-resource degrades most | **✅ en→X / ❌ X→en** |
| C3 Language-matched calibration helps at 2-bit | ✅ reproduced (+1.8 vs paper's +3.1) |
| C4 GGUF most consistent | ✅ reproduced (4-bit data only) |
| C5 Small models lose most | ✅ reproduced |

**The direction split is the important part.** The paper's headline into-English
collapse does not reproduce: Llama-3.1 bn→en 4-bit deltas are *positive*
(+2.4 to +5.3 COMET). Off-target Indic output *falls* from 22% at baseline to
~5% at 4-bit — quantization reduces the failure mode, the opposite of the
paper's account. **Build on C2/C3 for en→X only.**

Two red flags found in the paper: Table 8's prose calls 71.5 vs 72.3 a "gain"
(it's a loss), and the stated English calibration config
(`HuggingFaceFW/fineweb-2`) has no English split, so it cannot be what they ran.

---

# Track: speech-translation

**NLLB-200-distilled-600M** (not 3.3B — that was never compressed) and **XTTS
v2**, en/es/fr only, n=100 and n=50.

- **Smaller is not faster.** BnB INT8 is **4.3× slower than FP16** at equal
  quality. Only CTranslate2 INT8 delivers a real speedup (2.4× wall-clock).
- **INT8 on the XTTS GPT core raises French CER 2.6×** (0.061 → 0.157) while
  being neutral for English and Spanish. A language-averaged metric would have
  hidden this completely.
- **mobile-tts**: Swahili MMS-TTS speaker fine-tune completed (30 epochs, best
  eval mel-loss 1.0848 at step 15000). **No quality evaluation of the text→speech
  path exists** — the objective and the eval are both posterior-encoder
  reconstruction. No CER/WER/MOS/speaker-similarity number.

Two survey documents (`docs/findings/compression-nllb-xtts-research.md`,
`interp-lrl-nllb-xtts.md`) specify ~11 experiments. **Essentially none were run**
— no LRL evaluation, no component ablation beyond the GPT core, no super-weight
or Fisher analysis on NLLB, no mixed precision.

---

# Track: interlingua

Docs only, no code. See [`../interlingua/README.md`](../interlingua/README.md).
Direction: Tier 1 of *Does the Interlingua Grok?* (Ringger, Matrix Lab).

---

# Ruled out — do not redo

1. **Importance-driven bit allocation.** Never wire IFR, DLA, Block Influence,
   Taylor, tuned-lens CBE, EAP/EAP-IG, HAWQ-V2 trace, or CoopQ Shapley to bit
   allocation. Q5, twice, at two granularities. *Legitimate as pruning criteria.*
2. **Depth/stage-aware bit allocation.** Wash at stage level. (The pruning
   layer-drop analog **is** still open — removal ≠ precision.)
3. **Fisher-diagonal mixed precision** as implemented. Negative everywhere.
4. **MT calibration as a general quantization lever.** True for GPTQ (which
   reconstructs on calibration activations), false for AWQ (which only derives a
   per-channel scale). Keep the quantizer qualifier — `OPEN-WORK.md` drops it.
5. **Magnitude as a saliency.** Magnitude ≠ importance ≠ sensitivity.
6. **Sub-2-bit rescue via salient/super-weight preservation.**
7. **`bitsandbytes` as a deployment path.** 4.3× slower than FP16.
8. **COMET below 2 bits.** Use chrF++ for collapse detection.

# Open and under-credited

**Under-credited** — `OPEN-WORK.md` lists these as unexplored; they are partly done:

- **MT-conditional pruning masks** (ranked #7, "partially-explored"). The core
  experiment is *done*: Wanda 50%, 8 models × 3 pairs, 19/24 positive, median
  +2.18 chrF++. What remains is SparseGPT, 25%/2:4 sparsity, and COMET.
- **Sensitivity-native signal bake-off** (ranked #1, "novel"). Three of five
  signals already have data — Fisher (failed), Wanda, causal-KL. Missing:
  Shapley, reconstruction-error, and the head-to-head against COMET drop.
  *`OPEN-WORK.md` labels this "Novel" in its top-8 table and
  "Partially-explored" in its track list — an internal contradiction.*

**Genuinely unclaimed:**

- **Super-weight formation across training.** Nothing tracks when super weights
  appear. On no backlog, in no proposal. The TowerBase→TowerInstruct datapoint is
  the only hint. Directly enabled by a checkpoint-suite study.
- Unreported results already on disk: the `crush_early`/`crush_late` pipeline
  arms (7 models × 5 bit-widths), the fact that both pipeline allocations beat
  uniform-W3 on every model, and EuroLLM's second super weight.

# Discrepancies — claims that do not survive audit

**Numbers that are wrong**

- `docs/findings/q1.md` reports head L26.H5 at **−0.18 on en→arz**. The data says
  **−0.058**; −0.177 is its *en→zh* value. The paper was corrected for this
  (commit `ff2fe04`); the findings doc was not. **`main.tex` and `q1.md` now
  contradict each other**, with the paper matching the data.
- `phase2-results.md` gives tower-base W3 COMET as .32; disk says 0.315.

**Coverage that is overstated**

- **Gemma and BLOOM have no GPTQ data at all.** `phase2-results.md` blames
  "transient CUDA faults." The logs show a **deterministic**
  `linalg.cholesky: not positive-definite` in this repo's own GPTQ
  (`compress.py:349/351`), identical on every retry. A fixable bug.
- **The pipeline negative rests on 7 models, not 8.** Gemma — the designated
  outlier — died with a CUDA launch failure and was never resubmitted. No doc
  says so.
- **Gemma-3 is absent from both findings docs** despite being the paper's most
  extreme data point.
- **NLLB is simultaneously "deferred" (`q1.md`) and a headline generalization
  result** (`architecture-comparison.md`, the paper).
- The replication findings doc is stale: 336 rows vs **372 on disk**; three
  completed units, including a large-model 2-bit result, never reached it.

**Claims whose evidence doesn't support them**

- *"Generic calibration is worse than not quantizing."* The comparison is
  GPTQ-generic vs **RTN-W3**, and RTN W3 *is* quantization. FP16 COMET was never
  measured. Repeated in three documents.
- *"MT-conditional GPTQ recovers the 3-bit cliff."* True vs GPTQ-generic; against
  plain RTN, **Aya is much worse** (66.3 → 49.3 chrF, −0.218 COMET).
- **NLLB's "IFR" is not IFR** — it's `‖resid[l] − resid[l−1]‖₁`, placed in the
  same table column as true IFR values without a note. NLLB also has no probing
  or DLA, so the enc-dec generalization rests on one lens curve and one proxy.
- **Q5's per-layer statistic is n=17 of 18 cells with no statement of which cell
  was dropped.** Reconstruction suggests `bloom-7b1 en-arz` (reproduces both
  means to 3 dp), but that is inference, not documentation.
- **Q5's config was not what ran** — config says COMET, n=200,
  σ ∈ {1e-3, 1e-2, 5e-2}; the run used chrF++, n=20, σ=0.1.
- The off-target diagnostic ("22% → ~5%") is a **hard-coded prose string** with no
  code. Recomputed: true for 3 of 4 quantizers, but **bnb-4 is 12.9%** — and bnb
  is the method the original paper headlines.
- `speech-translation` **VRAM is unmeasured** — the column is read after
  XCOMET-XL loads on the same GPU (hence 17.9 GB for a 600M model), and cannot
  see CTranslate2 allocations at all.
- `speech-translation` **tok/s is not comparable across backends** — HF counts
  padded tensor elements, CT2 counts real tokens. Wall-clock is the valid figure.

**Process failures worth not repeating**

- **The project's own headline claim was never hardened.** `ROADMAP.md` phase 1,
  item 1 was "harden the GPTQ-MT result (proper n + templates + XCOMET-XL +
  significance)." It never happened; `OPEN-WORK.md` re-lists it at **rank 8**.
  The rigor machinery was built for the replication and never back-applied.
- **Empty documents:** `docs/ideas.md` is 0 bytes. `docs/learning-log.md` — the
  intended record of dead ends — has no entries. The seven required
  `docs/systems-notes/` writeups were never written.
- **`phase2-hypotheses.md` was never updated**, though Q5's satisfied-when
  criterion required it. It still reads "None is committed," and its predicted
  kill conditions were inverted by events: Candidate C is listed as killed and is
  the one that won.
- **`phase2-novel-direction.md` still reads as a live proposal** with no results
  section, despite the idea having become a negative.

**Stale documents:** `SESSION-HANDOFF.md`, `advisor-brief.md` ("3 of 8 models"
— it's 6), `project-summary.md` ("phase two deliberately deferred"),
`phase2-hypotheses.md`, `docs/findings/README.md` (lists non-existent files),
`ROADMAP.md` (self-contradicts on depth/stage), `research.md` (carries a
hypothesis the W2 run refuted).

# Code assets

~4,700 lines under `compression/src/`. What is **reusable** for a from-scratch
training-dynamics study, and what is **missing**, is documented in
[`../interlingua/README.md`](../interlingua/README.md) and `CLAUDE.md`.

Short version: `logit_lens`, `probing` (with Hewitt–Liang selectivity), `ifr`,
`dla`, `super_weights`, and `_hooked.py` transfer — about 900 lines, and
`HookedModel` provably works on a randomly-initialized model. **Missing
entirely:** CKA/mutual-nearest-neighbor, changepoint detection, bootstrap CIs,
multi-seed orchestration, any checkpoint iteration. `run_with_cache` **raises on
batch size > 1.**

**Test reality:** 70 test functions, **17 are `pytest.skip("TODO")` stubs**.
`sensitivity.py` and `attribution_patching.py` have **zero tests** and are the
two methods the docs call load-bearing for Q5. `activation_patching.py` — the
"gold-standard causal method" other modules defer to — is entirely
`NotImplementedError`.

---

## Provenance

Produced 2026-08-05 by five parallel read-only agents over every document,
runner, and results directory, with headline numbers recomputed from raw
`.npz`/`.json`. Where an agent could not verify a claim it is marked as
inference. This document supersedes no findings doc — it annotates them.
