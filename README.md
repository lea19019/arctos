# Arctos

*Understanding translation in LLMs as a foundation for compression* — an
interpretability-led investigation into **how machine translation is carried
out inside open multilingual LLMs**, preceding a phase-two compression method
whose design is grounded in what phase one reveals.

**Phase one is complete.** The headline result, in one line: *translation is a
depth-staged "understand → process in a language-neutral/pivot space → emit
the target language last" computation; the target language is a late
conversion step, not the medium the model computes in* — and this structure
generalizes across architectures.

- **📖 Reading guide (start here):** [`docs/READING-GUIDE.md`](docs/READING-GUIDE.md) — an ordered path through every experiment, result, and paper, phase one → phase two.
- **📚 Papers + learning hub:** [`docs/READING-LIST.md`](docs/READING-LIST.md) — every cited paper (clickable, grouped by theme) + videos/sites to learn the foundations. [`docs/MATH-PLAN.md`](docs/MATH-PLAN.md) — 6-month math curriculum.
- **Phase-one report:** [`report/arctos-translation-report.pdf`](report/arctos-translation-report.pdf) — *What and how does the translation task work inside an LLM?* (methods, tables, figures, findings, citations).
- **Phase-two (compression):** see [Phase two](#phase-two--compression-for-translation) below — the find/keep/shrink/prune sandbox, the **MT-conditional GPTQ** result, and the honest negatives.
- **Findings per question:** [`docs/findings/`](docs/findings/) — `q1.md`, `q5.md`, `architecture-comparison.md` (phase one); `phase2-synthesis.md`, `phase2-results.md` (phase two).
- **Where it's going:** [`docs/ROADMAP.md`](docs/ROADMAP.md) — the multi-dimensional "sweet spot of compression for translation" program (quant × prune × distill across bit-scales).
- **Tutorials:** [`notebooks/`](notebooks/) — 8 runnable `# %%` notebooks (theory + mechanics) on a tiny CPU model.
- Thesis spine + plan: [`docs/project-summary.md`](docs/project-summary.md), [`PHASE1-PLAN.md`](PHASE1-PLAN.md).

## Models studied

Eight models spanning lineage, normalization, positional encoding, generation,
and the decoder-only ↔ encoder-decoder divide:

| Model | Architecture | Role |
|-------|--------------|------|
| Aya Expanse 8B | Cohere, RoPE, RMSNorm | general multilingual LM |
| TowerBase 7B → TowerInstruct 7B | Llama-2 | CPT vs CPT+MT-SFT ablation |
| Tower-Plus 9B | Gemma 2 | MT-specialist (2025) |
| BLOOM 7B1 | ALiBi, LayerNorm | old-gen multilingual (2022) |
| EuroLLM 9B | Llama-3 | European MT specialist |
| Llama-3.1 8B | Llama-3, GQA | general LM |
| Gemma-3 12B | Gemma 3 | baseline (Google QAT) |
| NLLB-200 3.3B | encoder–decoder | MT-purpose-built |

Language pairs (FLORES+): **cs→de** (same-script sanity), **en→zh** (Han),
**en→arz** (Egyptian Arabic). *(The original plan named omt-llama-8b, which
does not exist on HuggingFace; the set above is the realized substitute, with
Gemma as a baseline rather than a method target.)*

## Methods (`src/interp/`)

All read or intervene on the residual stream through one uniform wrapper
(`src/models/_hooked.py`, `HookedModel`) so the same code runs on every
architecture.

| File | Method | Answers |
|------|--------|---------|
| `logit_lens.py` | logit lens | when does the model commit to the target token? |
| `probing.py` | linear probes + control task | where is language identity decodable? |
| `ifr.py` | Information Flow Routes | which components are *loud* (magnitude)? |
| `dla.py` | direct logit attribution | which components push *toward* the target (signed)? |
| `attribution_patching.py` | attribution patching | which components, if damaged, *break* translation (causal)? |
| `sensitivity.py` | noise injection | where does numerical *precision* matter (quant proxy)? |
| `activation_stats.py` | AWQ-style activation magnitude | which weight channels see large activations? |
| `language_pivot.py` | **pivot trajectory** | does the model "think in a pivot script" then convert late? |
| `tuned_lens.py` | tuned lens (Belrose 2023) | de-biased logit lens (built, optional) |

## Layout

```
.
├── report/                   # the compiled PDF report + LaTeX source + figures
├── docs/findings/            # per-question writeups (q1, q5, architecture-comparison)
├── notebooks/                # 8 method tutorials (00 overview → 08 synthesis)
├── experiments/
│   ├── q1-language-emergence/   # logit lens, probing, IFR, DLA, pivot runners + SLURM
│   ├── q2-attention-heads/      # attention-pattern viz
│   ├── q4-architecture-comparison/  # combined analysis SLURM
│   └── q5-importance-vs-sensitivity/ # noise/attribution/AWQ + chrF++ quality runners
├── src/
│   ├── models/               # per-model loaders + _hooked.py (HookedModel) + nllb.py
│   ├── interp/               # the methods above
│   ├── data/                 # FLORES+ loader + clean/corrupt generators
│   └── eval/                 # chrF++/BLEU (sacrebleu)
├── results/                  # gitignored — per-model outputs (npz/json/charts)
├── tests/                    # cpu + gpu markers
└── scripts/                  # fetch_flores.py etc.
```

## Environment

```bash
uv sync                       # .venv from pyproject.toml (Python 3.11 pinned)
uv run pytest -m cpu          # method unit tests on a tiny CPU model
```

Cluster notes (BYU RC / SLURM): the env is pinned to **Python 3.11** (system
OpenSSL compatibility) and **torch cu128** (driver compatibility); SLURM
scripts set `OPENSSL_CONF=/dev/null` and `HF_HUB_OFFLINE=1` (compute nodes
have no internet — models are pre-cached on the login node). Hardware target
is a single A100 80GB.

## Key findings

1. **Translation does not happen in the target language until the end.** Under
   the logit lens, target-script probability mass is ~0 through ~80–95% of
   depth, rising only in the final layers (across 6 architectures). The middle
   of the network operates in a Latin/pivot representation.
2. **Depth-ordered pipeline:** source encoding (early) → language-neutral
   processing (middle) → target commitment (last quarter), where logit-lens
   target mass, signed DLA, and IFR magnitude all concentrate.
3. **The signature generalizes** across lineage, normalization, positional
   encoding, generation, and decoder-only ↔ encoder-decoder (NLLB) — the lone
   exception is Gemma-family.
4. **Importance ≠ quantization sensitivity** (ρ ≈ 0 across two metrics): where
   MT computation concentrates is *not* where numerical precision matters.

## Phase two — compression for translation

The interpretability-grounded compression chapter. Sandbox + experiments in
[`experiments/q6-compression/`](experiments/q6-compression/); code in
`src/interp/{compress,super_weights,hessian_diag,salient_channels}.py` +
`src/eval/metrics.py` (XCOMET-XL). Run: `submit_gem.sh` / `submit_extreme.sh`;
collect: `python scripts/q6gem_collect.py`; status: `bash scripts/q6_status.sh`.

**Docs (read in this order):**
1. [`docs/findings/phase2-synthesis.md`](docs/findings/phase2-synthesis.md) — **the honest consolidated read** (what holds, what doesn't).
2. [`docs/findings/phase2-results.md`](docs/findings/phase2-results.md) — cross-model tables (chrF++ / XCOMET-XL).
3. [`docs/findings/compression-primer.md`](docs/findings/compression-primer.md) — find/keep/shrink/prune framework + reading list.
4. [`docs/findings/phase2-method-primer.md`](docs/findings/phase2-method-primer.md) — the method + literature-gap/novelty map.
5. [`docs/findings/phase2-novel-direction.md`](docs/findings/phase2-novel-direction.md) — the pipeline-aware idea (now a reported **negative**).
6. [`docs/findings/q6.md`](docs/findings/q6.md) — the chrF++ sweep writeup.
7. [`docs/research.md`](docs/research.md) — annotated bibliography + 3 deep-research addenda; raw reports in [`docs/findings/deep-research-raw/`](docs/findings/deep-research-raw/).
8. [`docs/advisor-brief.md`](docs/advisor-brief.md) — talking doc. [`docs/replication-uneven-ptq-mt-brief.md`](docs/replication-uneven-ptq-mt-brief.md) — replication of arXiv:2508.20893.

**Phase-two key findings (directional; chrF++/XCOMET-XL, small n):**
- ✅ **MT-conditional GPTQ** recovers the 3-bit cliff (all 6 models, +0.13–0.52 COMET); *generic-text calibration is worse than not quantizing.* The contribution.
- ✅ **Salient-channel / super-weight FP16 preservation** independently recovers 3-bit (Gemma 12.7→48.4 chrF).
- ❌ **Depth-pipeline ≠ a compression rule:** protecting the language-specific endpoints vs the neutral middle is a wash (Q5 null reconfirmed at stage level).
- 🔭 **No healing-free PTQ reaches FP16 at 3-bit** (any method) → goal = *best healing-free MT-specific option at a given size.*

## Status

**Phase one complete; phase two has a result.** Headline: *for low-bit
translation, GPTQ must be calibrated on MT data — generic calibration actively
hurts — and salient/super-weight preservation recovers the 3-bit cliff, while
the depth-pipeline does **not** localize quantization fragility.* Next program
(6-month, fall start): the multi-dimensional **sweet-spot study** — quant ×
prune × distill across bit-scales and weight types, on MT (and speech↔text) —
see [`docs/ROADMAP.md`](docs/ROADMAP.md). All phase-two numbers are directional
(small n, generic prompt); the paper-grade run needs larger n + chat templates +
a human spot-check.
