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

- **Report (start here):** [`report/arctos-translation-report.pdf`](report/arctos-translation-report.pdf) — *What and how does the translation task work inside an LLM?* (methods, tables, figures, findings, citations).
- **Findings per question:** [`docs/findings/`](docs/findings/) — `q1.md` (language emergence), `q5.md` (importance vs sensitivity), `architecture-comparison.md` (Q4 synthesis).
- **Tutorials:** [`notebooks/`](notebooks/) — 8 runnable `# %%` notebooks explaining each method (theory + mechanics), runnable on a tiny CPU model.
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

## Status

**Phase one complete** — all methods implemented and run across the model set;
findings written; report compiled. Phase two (the quantization method itself)
is the next chapter: use the depth-signature as a coarse prior but allocate
bits with a sensitivity-native signal (finding #4), treating Gemma-family
separately.
