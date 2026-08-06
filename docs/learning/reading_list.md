# Reading & learning hub

Every paper we cited, grouped by theme so you can browse and pick what pulls you
— plus videos/sites to learn the foundations. Skim the one-line "why," click what
interests you. (arXiv IDs from `compression/docs/annotated_bibliography.md`, mostly verified.)

## ⭐ If you only read five
1. **The Super Weight in LLMs** — https://arxiv.org/abs/2411.07191 — a single weight can make-or-break a model. Wild, simple, deep.
2. **Do Llamas Work in English?** — https://arxiv.org/abs/2402.10588 — the "models think in a pivot language" result behind our phase one.
3. **GPTQ** — https://arxiv.org/abs/2210.17323 — the quantizer at the center of our result; the math is the Taylor/Hessian core.
4. **The Uneven Impact of PTQ in Machine Translation** — https://arxiv.org/abs/2508.20893 — the closest paper to our work; what we extend.
5. **QuIP** — https://arxiv.org/abs/2307.13304 — the cleanest "a math insight (incoherence/rotation) → a method" story; the level you want to reach.

---

## A. How translation works inside LLMs (the mechanism / interpretability)
- **Do Llamas Work in English?** (Wendler 2024) — https://arxiv.org/abs/2402.10588 — latent/pivot language.
- **Separating Tongue from Thought** (Dumas 2024) — https://arxiv.org/abs/2411.08745 — language-agnostic concept space, causal.
- **Language-Specific Neurons / LAPE** (Tang 2024) — https://arxiv.org/abs/2402.16438 — language machinery in top+bottom layers, shared middle (basis for our pipeline idea).
- **Information Flow Routes** (Ferrando & Voita 2024) — https://arxiv.org/abs/2403.00824 — our IFR method.
- **Tuned Lens** (Belrose 2023) — https://arxiv.org/abs/2303.08112 — de-biased logit lens.
- **Middle-layer cross-lingual alignment** — https://arxiv.org/abs/2502.14830 · **language→shared neurons / compression** — https://arxiv.org/abs/2506.01629 · **cross-layer transcoders, multilingual** — https://arxiv.org/abs/2511.10840.

## B. Super weights / massive activations / outliers
- **The Super Weight in LLMs** (Yu 2024) — https://arxiv.org/abs/2411.07191 (+ Apple writeup: https://machinelearning.apple.com/research/the-super-weight).
- **Massive Activations in LLMs** (Sun 2024) — https://arxiv.org/abs/2402.17762.
- **LLM.int8() / emergent outlier features** (Dettmers 2022) — https://arxiv.org/abs/2208.07339.

## C. Quantization methods (the core)
- **GPTQ** — https://arxiv.org/abs/2210.17323 · **AWQ** — https://arxiv.org/abs/2306.00978 · **SqueezeLLM** — https://arxiv.org/abs/2306.07629.
- **LeanQuant** (loss-aware grid; WMT25 quality winner) — https://arxiv.org/abs/2407.10032.
- **GPTVQ** (best healing-free codebook PTQ) — https://arxiv.org/abs/2402.15319 · **AQLM** — https://arxiv.org/abs/2401.06118 · **QuIP#** — https://arxiv.org/abs/2402.04396 · **QTIP** — https://arxiv.org/abs/2406.11235 · **VPTQ** — https://aclanthology.org/2024.emnlp-main.467/.
- **Rotation:** **QuaRot** — https://arxiv.org/abs/2404.00456 · **SpinQuant** — https://arxiv.org/abs/2405.16406 · **QuIP** (origin of incoherence) — https://arxiv.org/abs/2307.13304.
- **Mixed precision:** **HAWQ-V2** — https://arxiv.org/abs/1911.03852 · **CoopQ** (why naive per-layer fails) — https://arxiv.org/abs/2509.15455.
- **OmniQuant** — https://arxiv.org/abs/2308.13137. Classics behind the math: Optimal Brain Damage (LeCun 1990), Optimal Brain Surgeon (Hassibi & Stork 1993).

## D. Extreme low-bit (<2-bit)
- **BitNet b1.58** — https://arxiv.org/abs/2402.17764 (+ 2B4T: https://arxiv.org/abs/2504.12285) — 1.58-bit, but trained from scratch.
- **BiLLM** (1-bit PTQ, salient split) — https://arxiv.org/abs/2402.04291 · **PB-LLM** — https://arxiv.org/abs/2310.00034.
- **PTQTP** (ternary PTQ) — https://arxiv.org/abs/2509.16989 · **PT2-LLM** — https://arxiv.org/abs/2510.03267.

## E. MT / multilingual quantization + WMT25 (closest to our work)
- **Uneven Impact of PTQ in MT** — https://arxiv.org/abs/2508.20893 — the anchor (no GPTQ tested → our gap).
- **Calibrating Beyond English** (Chimoto, EACL 2026) — https://arxiv.org/abs/2601.18306 — multilingual GPTQ, perplexity-only.
- **How Does Quantization Affect Multilingual LLMs?** (Marchisio, EMNLP 2024) — https://arxiv.org/abs/2407.03211 — metrics understate damage ~10×.
- **WMT25 Model Compression Shared Task** (Gaido 2025) — https://aclanthology.org/2025.wmt-1.25/ — the benchmark we target.
- **Impact of Calibration Data in PTQ & Pruning** (Williams & Aletras 2024) — https://arxiv.org/abs/2311.09755 · **Outliers/calibration diminishing effect** — https://arxiv.org/abs/2405.20835 · **Self-calibration** — https://arxiv.org/abs/2410.17170.

## F. Pruning / structural compression
- **Wanda** — https://arxiv.org/abs/2306.11695 · **SparseGPT** — https://arxiv.org/abs/2301.00774 · **OWL** — https://arxiv.org/abs/2310.05175.
- **ShortGPT** — https://arxiv.org/abs/2403.03853 · **Unreasonable Ineffectiveness of Deeper Layers** (Gromov) — https://arxiv.org/abs/2403.17887 · **SLEB** — https://arxiv.org/abs/2402.09025 · **LaCo** — https://arxiv.org/abs/2402.11187 · **SliceGPT** — https://arxiv.org/abs/2401.15024 · **LLM-Pruner** — https://arxiv.org/abs/2305.11627 · **GeLaCo** (WMT25) — https://arxiv.org/abs/2507.10059.

## G. Distillation / recovery / MT training (the roadmap's distill axis)
- **GKD** — https://arxiv.org/abs/2306.13649 · **DistiLLM** — https://arxiv.org/abs/2402.03898 · **MiniLLM** — https://arxiv.org/abs/2306.08543.
- **QLoRA** — https://arxiv.org/abs/2305.14314 · **DoRA** — https://arxiv.org/abs/2402.09353 · **LoftQ** — https://arxiv.org/abs/2310.08659.
- **ALMA** — https://arxiv.org/abs/2309.11674 · **ALMA-R/CPO** — https://arxiv.org/abs/2401.08417 · **Tower** — https://arxiv.org/abs/2402.17733 · **X-ALMA** — https://arxiv.org/abs/2410.03115.

## H. Our own writeups (in this repo)
- `compression/docs/phase2_synthesis.md` (conclusions) · `phase2_results.md` (tables) · `compression/docs/compression_primer.md` · `compression/docs/phase2_method_primer.md`.
- `compression/docs/annotated_bibliography.md` — annotated bibliography + 3 deep-research syntheses; raw reports in `compression/docs/deep_research_raw/`.

---

## 🎓 Foundations — learn the background (videos + sites)

**Math (do these first — see `docs/learning/math_plan.md`):**
- **3Blue1Brown** — *Essence of Linear Algebra* + *Essence of Calculus* + *Neural Networks*: https://www.youtube.com/c/3blue1brown — the intuition exam courses skip.
- **Mathematics for Machine Learning** (free book): https://mml-book.github.io/ — your reference.
- **Seeing Theory** (visual probability): https://seeing-theory.brown.edu/.
- **StatQuest** (Josh Starmer — stats/ML, gentle): https://www.youtube.com/c/joshstarmer.

**Deep learning / building & training models (your "train more models" goal):**
- **Karpathy — Neural Networks: Zero to Hero**: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ — build nanoGPT from scratch. Best single resource.
- **fast.ai — Practical Deep Learning**: https://course.fast.ai/.
- **The Illustrated Transformer** (Jay Alammar): https://jalammar.github.io/illustrated-transformer/.

**Interpretability:**
- **Distill.pub**: https://distill.pub/ · **Transformer Circuits**: https://transformer-circuits.pub/ (the framework behind our methods).

**Quantization / compression specifically:**
- **Awesome-LLM-Compression** (curated paper list): https://github.com/HuangOwen/Awesome-LLM-Compression.
- **HuggingFace Quantization docs**: https://huggingface.co/docs/transformers/main/en/quantization/overview.

**Mindset / wealth (since you asked):**
- **Naval — How to Get Rich** (podcast/threads): https://nav.al/rich · *The Almanack of Naval Ravikant* (free): https://www.navalmanack.com/.

---

*How to use this: don't read top-to-bottom. Pick the section that excites you,
read 2–3, and notice which makes you lose track of time — that's your signal
(the Naval "specific knowledge" test). Then go deep there.*
