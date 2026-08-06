# MS Final Project: Two Candidate Ideas for Advisor Review

**Student:** Adrian | **Program:** CS MS (CS 698R Final Project) | **Date:** 2026-06-19
**Scope:** ~150+ hours, summer semester | **Advisor meeting:** Monday

---

## Who I Am and What I'm Building

I am a research engineer, not a pure researcher. I am simultaneously:
- A graduate student completing my MS final project
- An SWE at my school developing a real-time dubbing app for non-profit organizations

The dubbing app translates spoken content from English into many languages (including low-resource languages / LRL) and produces dubbed audio output. It currently runs on the lab's A100 GPUs. The project is in the process of being acquired by a company, which means production deployment cost and real-time performance are real constraints, not academic abstractions.

I enjoy system-level thinking, cloud infrastructure (which I'm actively learning and deploying), and the intersection of ML research with engineering. I want work that has tangible impact for underserved language communities.

**Existing work context (Arctos project):** I have completed a substantial phase-one interpretability investigation of decoder-only LLMs for machine translation (Q1–Q6 across 8 models including NLLB-3.3B for architectural comparison). Key findings:
- IFR depth signature generalizes across architectures including encoder-decoder NLLB
- Component importance (IFR/DLA) does NOT predict quantization sensitivity (Q5 null)
- Super-weights, AWQ salient channels, Fisher/Hessian, and Wanda are the right sensitivity-native signals
- MT-conditional calibration (bilingual vs generic) matters — especially at 2-bit for LRL languages
- Confirmed via independent replication of arXiv:2508.20893 (PTQ-MT paper)
- Tools built: `compression/src/interp/super_weights.py`, `salient_channels.py`, `hessian_diag.py`, `compress.py`

---

## Project Idea 1: LRL-Preserving Quantization for Decoder-Only MT LLMs

### The Research Gap

Post-training quantization (AWQ, GPTQ, LeanQuant) is well-studied for decoder-only LLMs, but nobody has used mechanistic interpretability findings to drive per-layer precision allocation specifically for low-resource language (LRL) quality preservation. The existing methods use generic calibration data and uniform or heuristic bit allocation — they consistently underperform for LRL languages at low bit-widths (2–3 bit).

The PTQ-MT replication I completed (arXiv:2508.20893) confirmed: language-matched calibration helps primarily at 2-bit and primarily for divergent-script LRL languages (Bengali, Malayalam). This is the nexus the custom method addresses.

### What I Would Do

**The central question:** If we use interpretability to identify which weight channels are specifically critical for LRL quality (vs HRL quality), and protect those at higher precision, does LRL quality retention under compression improve meaningfully compared to uniform quantization?

**Experimental plan:**
1. **AWQ salient channel LRL vs HRL split** — Run AWQ on Aya Expanse 8B with two calibration sets: (A) English-French (HRL), (B) English-Bengali + English-Malayalam + English-Zulu (LRL). Compare salient channel masks. Channels in B but not A are LRL-specific. Evaluate both at 4-bit and 2-bit on LRL test set with COMET-22.

2. **Super-weight detection under LRL vs HRL inputs** — Adapt causal-KL super-weight ranking (`compression/src/interp/super_weights.py`) to compare which super-weight positions appear under LRL vs HRL activation traces. Non-overlapping positions = LRL-specific super-weights missed by standard calibration.

3. **Per-layer Fisher under LRL calibration** — Run `compression/src/interp/hessian_diag.py` with LRL calibration. Compare layer-wise Fisher rank order vs HRL calibration. Layers with largest rank-order difference = where LRL calibration changes what gets protected. Use this to drive per-layer bit-width decisions.

4. **Mixed-precision scheme assembly** — Combine signals from 1–3 into a per-layer bit-width map. Assign higher bits to LRL-critical layers. Evaluate on flores200 LRL/HRL test sets with COMET-22 vs uniform-precision baselines.

**Models:** Aya Expanse 8B (primary), Qwen3-1.7B/8B, Llama-3.1-8B-Instruct — all already cached on the cluster.

**Evaluation:** COMET-22 (cached offline), WMT24++ test sets (6 language directions including Bengali, Malayalam, Zulu), chrF++.

**Infrastructure:** Fully built. SLURM scripts, A100, all models cached, evaluation pipeline running. This is a continuation of existing Arctos infrastructure.

### Why This Is Strong

- All infrastructure exists and works — experiments can start immediately
- Q1–Q6 mechanistic grounding directly motivates the method
- Clear null hypothesis (LRL calibration makes no difference) to falsify
- The WMT25 shared task provides competitive context and baselines
- Clean story: "we investigated before compressing; now we compress based on what we found"

### Why This May Not Be The Right Choice

- It's the "safe" academic choice — well-established evaluation, known methods, incremental advance
- The impact is on decoder-only LLMs that translate text, not on the dubbing app the lab is actually building
- Does not develop new skills (cloud, deployment, TTS) — extends existing tools only
- Less personally motivating given the engineering direction I want to grow in

---

## Project Idea 2: Compression of NLLB-200 and XTTS v2 for Real-Time Dubbing Deployment

### The Deployment Problem

The dubbing app pipeline is: **English audio → [ASR] → English text → NLLB-200 (MT) → LRL text → XTTS v2 (TTS) → LRL dubbed audio**

Currently deployed on A100 GPUs (~$3–4/hr on AWS). Target: T4-class instances (~$0.50/hr) — a 6–8× cost reduction that makes the app viable for non-profit scale without the lab or acquiring company owning expensive GPU infrastructure.

**Memory math on a T4 (16GB VRAM):**
- NLLB-200-3.3B INT4: ~1.65GB
- XTTS v2 INT8: ~1.7GB
- Combined: ~3.4GB → fits on a T4 with substantial headroom for batching and activation memory

**The real-time constraint:** XTTS must run with real-time factor < 1.0 (generates audio faster than it plays). NLLB must have latency < 500ms per sentence for the dubbing pipeline to feel live.

### The Research Gap

**For NLLB (encoder-decoder MT):**
The dominant production path is CTranslate2 INT8 — uniform quantization with no published quality delta table. AWQ and GPTQ were designed for decoder-only LLMs and have never been applied to any NLLB variant. No published BLEU/COMET vs bit-width table exists for NLLB-200-3.3B under any PTQ method. This is a completely open area.

Key architectural complexity: NLLB has three distinct attention types (encoder self-attention, decoder self-attention, cross-attention) with fundamentally different activation statistics. Critically:
- Cross-attention is the dominant PTQ failure mode (analogous finding in CAR-SAM, CVPR 2026)
- Language tag tokens (e.g., `__ben_Beng__`) function as attention sinks absorbing 83–91% of all cross-attention mass — they must stay at high precision (arXiv:2605.01229, 2026)
- Decoder is more LRL-fragile than encoder (2× quality hit under compression for LRL pairs)
- LRL languages require more encoder layers to resolve — early encoder layers are critical for LRL (DecoderLens finding, NAACL 2024, arXiv:2310.03686)

**For XTTS v2 (TTS: GPT autoregressive + HiFi-GAN vocoder):**
Zero published academic papers on XTTS compression exist. GitHub issue requesting quantization support was closed as "won't fix." The field literally starts from zero. Closest published work: SPADE (KAIST/42dot, arXiv:2509.20802, 2026) — WER-based layer importance + distillation for a generic LLM-TTS system.

Architecture:
1. **GPT-2-style autoregressive module** — generates discrete audio tokens (VQ-VAE, 1024-code codebook). Decoder-only transformer — tools from Arctos transfer directly.
2. **HiFi-GAN vocoder** — convolutional, single-pass, fast. More robust to quantization.
3. **VQ-VAE codebook** — must stay at 16-bit (discrete lookup; one wrong index = audible artifact).

### What I Would Do

**Part 1 — XTTS compression (start here, 4–6 weeks)**

1. **Component ablation** — Four conditions: Q8 everything / GPT Q4 + vocoder Q8 / GPT Q8 + vocoder Q4 / Q4 everything. Evaluate CER + UTMOS (MOS proxy) on LRL utterances from flores200 (Bengali, Malayalam, Swahili, Zulu). Identifies empirically whether GPT or vocoder is the LRL bottleneck.

2. **Per-layer Wanda sensitivity in XTTS-GPT** — Apply `compression/src/interp/compress.py::wanda_mask` to XTTS-GPT using LRL text→audio token sequences as calibration. Identify highest-sensitivity layers.

3. **Super-weight detection in XTTS-GPT** — `compression/src/interp/super_weights.py` transfers directly (GPT module is decoder-only). Run with LRL audio token calibration vs HRL (English, Spanish). Non-overlapping positions = LRL-specific super-weights.

4. **Mixed-precision XTTS scheme** — GPT attention layers at 8-bit for LRL-sensitive layers, FFN middle at 4-bit, VQ-VAE codebook at 16-bit, HiFi-GAN at 4-bit. Evaluate real-time factor + CER on LRL.

**Part 2 — NLLB compression (4–6 weeks)**

1. **DecoderLens LRL layer mapping** — Freeze NLLB-3.3B encoder at layer k, measure chrF++ vs k for LRL and HRL inputs separately. Identifies minimum encoder protection depth for LRL. (arXiv:2310.03686 method)

2. **AWQ salient channel LRL vs HRL split** — Same A/B calibration design as Idea 1, adapted to NLLB with bilingual parallel pairs (required — generic text calibration misses cross-attention activations entirely). Evaluate at 4-bit and 2-bit on flores200 LRL.

3. **Attention sink analysis** — Replicate arXiv:2605.01229 filtering on LRL test pairs. Identify content-routing cross-attention heads. These heads + language tag embeddings get 8-bit or 16-bit.

4. **Mixed-precision NLLB scheme** — Early encoder: 8-bit. Middle encoder: 4-bit. Language tag/EOS embeddings: 16-bit. Content-routing cross-attention heads: 8-bit. Evaluate COMET-22 on flores200 LRL.

**Part 3 — Deployment validation (2–3 weeks)**

- Both models loaded simultaneously on a cloud T4 instance
- Measure: end-to-end latency (text → dubbed audio), GPU memory, cost per hour of dubbed content
- Compare compressed stack vs current A100 baseline: cost reduction × and quality retention %
- Document: the production deployment configuration for the dubbing app

**Priority reading list (before starting):**
1. arXiv:2406.04904 — XTTS v2 paper (architecture ground truth)
2. arXiv:2310.03686 — DecoderLens (first NLLB experiment)
3. arXiv:2605.01229 — Attention sinks in NLLB-200 (most actionable finding)
4. arXiv:2212.09811 — Memory-efficient NLLB (MoE expert pruning; decoder LRL fragility)
5. arXiv:2509.20802 — SPADE (LLM-TTS compression template)

### Why This Is The Right Choice

- **Direct impact:** Results immediately improve the dubbing app the lab is deploying for non-profits serving underserved language communities
- **Open territory:** Both NLLB PTQ and XTTS compression are completely unstudied — any result is the first result. Low bar to exceed, high novelty ceiling.
- **Research + engineering:** The project has a clear research component (interpretability-guided mixed-precision) AND a clear engineering component (cloud deployment, cost measurement, real-time validation)
- **Skill development:** Forces learning of TTS architecture, encoder-decoder quantization, cloud deployment of ML models — all directly relevant to the trajectory I want as a research engineer
- **Unified story:** The same "interpretability → precision allocation" method applies to both models, creating a coherent project rather than two disconnected experiments
- **Portfolio value:** A deployed, cost-characterized, quality-measured system is a stronger artifact than a benchmark improvement for a research engineer career

### Honest Risks

- More unknowns than Idea 1 — XTTS compression may hit unexpected blockers
- New infrastructure required (CTranslate2, XTTS loading, MOS evaluation)
- No prior literature to calibrate "is this result good?" against — you're setting the baseline
- Quality metrics (CER, UTMOS) are noisier than COMET for academic rigor

---

## Comparison Table

| Dimension | Idea 1: LLM + LRL Quant | Idea 2: NLLB + XTTS Compression |
|---|---|---|
| Infrastructure | Fully built, ready now | Needs setup (1–2 weeks) |
| Academic risk | Low | Medium |
| Novelty | Incremental over PTQ literature | First result for both models |
| Direct lab impact | Indirect (MT LLMs ≠ dubbing app) | Direct (these are the deployed models) |
| Personal motivation | Good | High |
| Skill growth | Extends existing | Cloud, TTS, enc-dec — new territory |
| Deployment validation | Optional | Core deliverable |
| Advisor appeal | Clean, well-framed | Broader, more ambitious |
| LRL impact | Demonstrated on academic benchmarks | Demonstrated on actual deployment |

---

## Related Documents in This Repo

- `speech-translation/docs/nllb_xtts_compression_survey.md` — Full compression survey for NLLB and XTTS (PTQ, pruning, distillation state of the art, open gaps, paper list)
- `speech-translation/docs/nllb_xtts_interp_map.md` — Interpretability map for LRL-preserving quantization: mechanistic findings, bit-allocation implications, full experimental roadmap
- `compression/docs/annotated_bibliography.md` — Arctos comprehensive survey of decoder-only LLM compression (Sections 1–3: pruning, quantization, recovery/distillation)
- `compression/docs/q5_importance_vs_sensitivity.md` — Q5 null result: importance ≠ quantization sensitivity
- `compression/docs/q6_compression.md` — Q6 phase-two results: super-weights, Wanda, 3-bit cliff, MT calibration
- `compression/docs/replication_uneven_ptq_mt.md` — PTQ-MT replication findings (C1–C5, LRL+2-bit nexus)
- `archive/phase1_plan.md` — Full phase-one plan and V1/V2/V3 claim structure

---

## What I'll Tell My Advisor

Both ideas address the same core question — how do we use interpretability to design quantization that doesn't destroy low-resource language quality? — but at different levels of risk and ambition.

Idea 1 is the safer academic choice: infrastructure exists, evaluation is established, results are directly comparable to existing literature. It would produce a clean, publishable result.

Idea 2 is the higher-impact engineering choice: it directly serves the app the lab is deploying, produces a working artifact rather than just a benchmark result, and occupies genuinely open research territory. The risk is that it requires building from scratch in areas with no published baseline.

My instinct is Idea 2, primarily because I care about the actual deployment problem and because I want to grow as a research engineer working across the full stack. But I want advisor input on whether Idea 2 is scoped appropriately for a 150-hour MS project, and whether the combination of NLLB + XTTS in one project is too broad or whether one model should be the primary focus.
