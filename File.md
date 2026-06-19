# MS Project — Compression Methods for LLMs & Speech Models

> Planning document and research roadmap. Working notes for an MS in CS final project (BYU).
> Focus: post-training compression (quantization, pruning, KD recovery) with a **multilingual / translation** evaluation lens, extending toward **speech translation**.

---

## 1. Project at a Glance

**Prior experience to build on:** GPTQ quantization, layer pruning, KD-based recovery (with both KD data and regular data), all applied to the AyaExpanse (multilingual) model.

**Core bet:** "Does compression disproportionately hurt multilingual / translation / low-resource behavior, and can we fix it?" is a **contested, open question** in the current literature — not a solved one. That contestedness is the opening for a real contribution.

**Strategic framing (avoid the false binary):** Don't frame the project as *"general vs. translation-specific quantization."* Instead, use **translation / multilingual data as the evaluation lens and the source of hard cases**, and let the method be as general as the evidence supports. Outcomes either way are publishable:
- a translation/multilingual-specialized method, **or**
- a general method validated on the hardest multilingual cases.

---

## 2. Personal Learning Outcomes (success criteria for *me*)

- [ ] Deeper, hands-on knowledge of deep learning internals.
- [ ] Train many models; understand every component, not just call APIs.
- [ ] Walk the full **ML engineering cycle**: data prep → training → evaluation → deployment → monitoring.
- [ ] Make a contribution that matters to BYU, the industry, and my own growth.

> Note: covering the *entire* cycle may not be feasible — maximize coverage, don't force completeness.

---

## 3. The Research Gap (why this is contestable)

The literature currently **disagrees with itself**:

- **"It hurts"** — Large multilingual studies find low-bit (esp. 2-bit) quantization disproportionately degrades low-resource and typologically diverse languages. One human-eval study found automatic metrics *severely underestimate* the damage (a ~1.7% automatic drop in Japanese corresponded to ~16% in human eval; non-Latin scripts hit worst).
- **"It doesn't"** — A Llama 3.3 70B study using importance matrices in three languages concluded current quantization practices do **not** disproportionately harm multilingual performance.
- **"It's about calibration"** — A Jan 2026 study argues the resolution lies in calibration: tailoring the calibration set to the evaluation language gives the largest per-language gains; failures trace to **differences in activation-range distributions across languages**.

**Key methodological insight:** much of the disagreement is *methodological* (automatic vs. human eval, which models, which bit-widths, which calibration languages). **Don't pick a side — design the study to explain *why* the disagreement exists.** That's a more durable contribution than another "method X beats Y."

---

## 4. Conceptual Background (answers to my starting questions)

### 4.1 Dynamic quantization (define it operationally — the term is overloaded)
- The classical distinction is **when activation quant params (scale, zero-point) are computed**. Weights are quantized offline in nearly all schemes.
  - **Static:** activation scales fixed ahead of time from a calibration set.
  - **Dynamic:** activations quantized on the fly at inference as data flows through.
- **Trade-off:** dynamic adapts to per-input activation ranges (helps because LLM activations have extreme outliers, ~100× typical values per SmoothQuant), but adds latency, and many accelerators are tuned for *pre-defined* scales.
- **Lit-review subtlety:** most recent LLM quant work *silently assumes dynamic activation scaling*, even though fixed-point accelerators traditionally used static — this affects token rate and hardware support.
- **Action:** never write the bare phrase "dynamic quantization." Specify, e.g., *"per-token dynamic activation quantization"* or *"adaptive per-layer bit allocation."*

### 4.2 How speech models work + how translation happens
- Pipeline: raw audio → frontend representation (log-mel spectrogram **or** learned self-supervised features à la wav2vec 2.0 / HuBERT) → encoder (transformer or **Conformer** = convolution-augmented transformer) → decoder (for generative tasks).
- **Whisper** = canonical example: ~1.5B seq2seq transformer, ~680k hours weakly supervised, generalizes across domains.
- **Speech translation, two paradigms:**
  - **Cascaded:** ASR (audio→source text) → MT (source→target text). Modular; lets me reuse LLM-quant skills on the MT stage.
  - **End-to-end:** audio → target text directly (Whisper's X→English mode). Harder, avoids error propagation, more open research.
- **Compression-critical structure:** the **encoder runs once** over the audio; the **decoder runs autoregressively many times** → decoder compression buys disproportionate latency.

### 4.3 How speech compression works
- Same toolkit I already know (quant / prune / KD) **plus** speech-specific structure:
  - **Distil-Whisper:** distill the decoder hard, keep the encoder mostly intact (encoder copied teacher→student and frozen during training).
  - **DQ-Whisper:** joint distillation + quantization (quantization-aware distillation); ~5.18× size reduction with marginal degradation.
  - **DistilWhisper:** parameter-efficient, adds **language-specific gated modules**, jointly optimizes ASR fine-tuning + KD to transfer robustness from whisper-large-v2 into whisper-small.
  - Ultra-low-bit PTQ for large speech models (Interspeech 2025).
- **What's different vs. text LLMs:** long input sequences, streaming/latency constraints, encoder/decoder asymmetry.

### 4.4 Is a translation-specific method viable?
**Yes** — best defensible version is **compression that is robust for translation because it's aware of where multilingual circuits live and how multilingual activations distribute.** Concrete levers:
- Calibration-language composition (sample calibration from all target languages for GPTQ; language-matched for AWQ).
- Per-layer / per-language **mixed-precision** bit allocation; target outlier-heavy layers for special treatment.
- KD recovery using translation data (already in my toolkit).

---

## 5. Prerequisite Questions (answer these *before* the headline questions)

1. **Measurement first.** Given that automatic metrics underestimate multilingual quant damage — *how will I know if my method works?* Lock the eval stack early (e.g., COMET / chrF / BLEU **plus** a human or LLM-as-judge component, across a resource spectrum). A wrong yardstick wastes every downstream experiment.
2. **Baseline lineage.** Extending PTQ (GPTQ / AWQ) or moving toward quantization-aware distillation (DQ-Whisper style)?
3. **Compute & size envelope.** 7B or smaller? Is QAT even feasible given my hardware?

> Only after these do "can there be a translation/speech-specific method?" become tractable — they're really *"given my eval and baseline, where's the measurable headroom?"*

---

## 6. Phased Roadmap

### Phase 0 — Baseline + eval harness *(now → summer; Fall-readiness)*
- Reproduce a clean **GPTQ + KD-recovery** baseline on a multilingual model across a **language-resource spectrum**.
- Build the **evaluation harness** (this is the single most important de-risking step).
- *Teaches:* most of the data-prep → training → eval cycle.

### Phase 1 — Characterize the failure (don't fix yet)
- Measure **per-language degradation**.
- Locate **outlier-heavy layers**.
- Inspect **activation-range distributions across languages**.
- *This is where the novel insight comes from.*

### Phase 2 — Intervene
- Calibration-language composition.
- Per-language / per-layer mixed precision.
- Translation-data KD recovery.
- Measure against Phase 0.

### Phase 3 — Stretch: port to speech translation
- Exploit the **encoder/decoder asymmetry** as a second axis.

---

## 7. Mapping to Learning Goals

| Goal | Where it's exercised |
|---|---|
| Full ML-eng cycle | Phase 0 (data/train/eval); Phase 3 (latency, deployment, monitoring on a quantized model) |
| Train many models, understand internals | Phase 1 *is* internal instrumentation — can't characterize failure without it |
| Contribution to BYU / industry | IWSLT shared task convergence (below) |

**Convergence point for the contribution goal:** the **IWSLT 2025 "Model Compression" shared task for speech translation** already exists (e.g., a team used iterative importance-based layer pruning + 4-bit QLoRA + KD on Qwen2-Audio-7B to reach ~50% parameter reduction while keeping 97–100% of translation quality). A recurring shared task = venue + leaderboard + baselines + a clear demonstration of contribution, sitting exactly at compression ∩ translation ∩ speech.

---

## 8. Open Decisions / TODO

- [ ] Pick model family + size for Phase 0.
- [ ] Choose language-resource spectrum (high / mid / low; Latin / non-Latin script).
- [ ] Finalize eval stack (metrics + human/LLM-judge component).
- [ ] Decide baseline lineage (PTQ vs. QAT/QAD).
- [ ] Confirm compute budget and whether QAT is feasible.
- [ ] Decide whether to target the IWSLT Model Compression track.
- [ ] Resolve the contradictory multilingual-quant papers: map exactly where their methodologies diverge.

---

## 9. Key References

| Topic | Title | Link |
|---|---|---|
| Multilingual quant harm (human eval) | How Does Quantization Affect Multilingual LLMs? | https://arxiv.org/abs/2407.03211 |
| MT across 55 language pairs | The Uneven Impact of Post-Training Quantization in Machine Translation | https://arxiv.org/pdf/2508.20893 |
| "No disproportionate harm" counterpoint | English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance | https://arxiv.org/pdf/2503.03592 |
| Calibration-language alignment (2026) | Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLM | https://arxiv.org/abs/2601.18306 |
| Static vs dynamic in practice | FPTQuant: Function-Preserving Transforms for LLM Quantization | https://arxiv.org/pdf/2506.04985 |
| Speech: joint distill + quant | DQ-Whisper | https://arxiv.org/abs/2305.10788 |
| Speech: parameter-efficient distill | DistilWhisper (Efficient Compression of Multitask Multilingual Speech Models) | https://arxiv.org/pdf/2405.00966 |
| Speech: robust KD | Distil-Whisper | https://arxiv.org/pdf/2311.00430 |
| Speech translation compression (IWSLT 2025) | Efficient Speech Translation through Model Compression and KD | https://aclanthology.org/2025.iwslt-1.40/ |
| Curated list | Awesome-LLM-Quantization | https://github.com/pprp/Awesome-LLM-Quantization |

> Reference snapshot as of June 2026 — verify latest versions and check for newer work before the Fall start.

---

## 10. One-Line Thesis (draft)

*Compression methods degrade multilingual/translation behavior unevenly; by characterizing where and why (activation distributions, outlier layers, calibration mismatch) we can design compression that stays robust for translation — validated on the hardest low-resource cases and extended to speech translation.*