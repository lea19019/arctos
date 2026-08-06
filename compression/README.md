# compression — interpretability of MT in LLMs, and quantization built on it

*Understanding translation in LLMs as a foundation for compression.* An
interpretability-led investigation into **how machine translation is carried out
inside open multilingual LLMs** (phase one), followed by a compression method
chapter whose design is grounded in what phase one revealed (phase two).

**Both phases are complete.** This track is largely concluded; open directions
that came out of it are indexed in [`../docs/plans/open_work.md`](../docs/plans/open_work.md).

- **Start here:** [`docs/reading_guide.md`](docs/reading_guide.md) — an ordered path through every experiment and result.
- **The paper:** [`report/arctos-translation-report.pdf`](report/arctos-translation-report.pdf) — phase one, with methods, tables, figures, and citations.
- **All write-ups:** [`docs/`](docs/) — `q1_language_emergence.md`, `q5_importance_vs_sensitivity.md`, `q4_architecture_comparison.md` (phase one); `phase2_synthesis.md`, `phase2_results.md`, `q6_compression.md` (phase two); the primers and the bibliography. Index in [`docs/README.md`](docs/README.md).

## Layout

```
compression/
├── src/
│   ├── models/      # per-model loaders + _hooked.py (HookedModel), the uniform wrapper
│   ├── interp/      # the interpretability + compression methods (table below)
│   ├── quant/       # quantizer backends: bnb, AWQ, AutoRound, GGUF, GPTQ registry
│   ├── data/        # FLORES+ loader + clean/corrupt prompt generators
│   └── eval/        # chrF++/BLEU (sacrebleu) + XCOMET-XL
├── docs/            # this track's write-ups: findings, primers, bibliography
├── experiments/     # one folder per research question, each with configs/ + slurm/
├── tests/           # cpu + gpu markers
├── scripts/         # result collectors + job-status helpers
├── report/          # the compiled phase-one PDF + LaTeX source + figures
├── data/            # gitignored — FLORES+ pairs, regenerate via scripts/fetch_flores.py
└── results/         # gitignored — per-model outputs (json/npz/png)
```

## Models and languages

Nine models spanning lineage, normalization, positional encoding, generation,
and the decoder-only ↔ encoder-decoder divide (TowerBase and TowerInstruct share
a row below but are two separate models — other docs in this repo say "eight",
counting them once):

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
**en→arz** (Egyptian Arabic).

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
| `language_pivot.py` | pivot trajectory | does the model "think in a pivot script" then convert late? |
| `tuned_lens.py` | tuned lens (Belrose 2023) | de-biased logit lens (built, optional) |
| `compress.py` | RTN/AWQ/GPTQ/ternary/binary/prune/mixed-precision | the phase-two sandbox |
| `super_weights.py` | causal-KL super-weight detection | which individual weights are catastrophic to quantize? |
| `salient_channels.py`, `hessian_diag.py` | AWQ salience, Hessian diagonal | what to keep in FP16 |

## Experiments

| Folder | Question | Status |
|--------|----------|--------|
| `q1-language-emergence/` | when and where does the target language emerge? (logit lens, probing, IFR, DLA, pivot) | ✅ run, 9 models |
| `q2-attention-heads/` | which heads are MT-critical? | ⚠️ **partial** — only attention-pattern viz ran (6 models, one prompt per pair). `experiment.py` is a stub; the planned causal head ranking never ran, and there is no `q2.md` |
| `q3-mlps-and-layers/` | what do MLPs and layers contribute? | ❌ **never run.** `experiment.py` is a stub; no results, no writeup. Raw material exists (IFR `mlp_scores`, DLA `layer_mlp` in the q1 `.npz`) but was never analyzed |
| `q4-architecture-comparison/` | does the depth signature generalize across architectures? | ✅ synthesis (written by hand; the runner is a stub) |
| `q5-importance-vs-sensitivity/` | **the pivotal negative** — is importance correlated with quantization sensitivity? | ✅ run, 6 models |
| `q6-compression/` | the find/keep/shrink/prune compression sandbox (phase two) | ✅ run, 8 models (GPTQ stage: 6 — see below) |
| `replication-uneven-ptq/` | independent replication of arXiv:2508.20893 (uneven PTQ impact on MT) | ✅ run, 31 of 35 planned units |

## Key findings

1. **Translation does not happen in the target language until the end.** Under
   the logit lens, target-script probability mass is ~0 through ~80–95% of
   depth, rising only in the final layers (across 6 architectures). The middle
   of the network operates in a Latin/pivot representation.
2. **Depth-ordered pipeline:** source encoding (early) → language-neutral
   processing (middle) → target commitment (last quarter), where logit-lens
   target mass, signed DLA, and IFR magnitude all concentrate.
3. **The signature generalizes** across lineage, normalization, positional
   encoding, and generation — the lone exception is the Gemma family.
   *Caveat on the encoder-decoder half:* NLLB's number is **not IFR**. The
   enc-dec runner computes `‖resid[l] − resid[l−1]‖₁` (an "IFR-like" proxy, per
   its own docstring), and NLLB has no probing or DLA at all, so the
   decoder-only ↔ enc-dec generalization rests on one lens curve plus one
   magnitude proxy. The synthesis docs place it in the same table column as the
   true IFR values without flagging the substitution.
4. **Importance ≠ quantization sensitivity** (ρ ≈ 0 across two metrics): where
   MT computation concentrates is *not* where numerical precision matters. This
   is the negative that constrains everything downstream.

### Phase two (directional; chrF++/XCOMET-XL, small n)

- ✅ **MT-calibrated GPTQ ≫ generic-calibrated GPTQ** at W3 on all 6 models with
  data (+0.13–0.52 COMET). State it as this comparison, *not* as "GPTQ-MT beats
  not calibrating": against plain RTN, **Aya is much worse** (cs-de 66.3 → 49.3
  chrF, −0.218 COMET). And "generic calibration is worse than not quantizing" is
  **unsupported** — the comparison behind it is GPTQ-generic vs RTN-W3, and RTN
  W3 *is* quantization; FP16 COMET was never measured.
  *Coverage:* 6 of 8 models. Gemma and BLOOM are missing because this repo's own
  GPTQ dies on them with `linalg.cholesky: not positive-definite`
  (`compress.py:349/351`) — a deterministic bug, not the "transient CUDA faults"
  the findings doc claims.
- ✅ **Salient-channel FP16 preservation** recovers much of the 3-bit cliff
  (Gemma 12.7 → 48.4 chrF). **Super-weight preservation does not** — `rtn+SW ≈
  rtn` on every model. The two are different objects and behave oppositely;
  super weights matter for *removal* (ablating one takes EuroLLM 57.9 → 4.8
  chrF, while ablating the 1000 largest-magnitude weights costs nothing).
- ❌ **The depth pipeline is not a compression rule.** Protecting the
  language-specific endpoints vs the neutral middle is a wash — the Q5 null,
  reconfirmed at stage level. *On 7 models, not 8:* Gemma — the designated
  outlier family — is absent, its job having died with a CUDA launch failure
  that was never resubmitted. Unreported: both allocations beat uniform-W3 on
  every model, and the `crush_early`/`crush_late` arms ran on all 7 and appear
  in no document.
- 🔭 **No healing-free PTQ reaches FP16 at 3-bit** by any method, so the honest
  goal is *the best healing-free MT-specific option at a given size.*

Super weights are detected by **causal KL**, not activation-spike magnitude —
ranking by spike size produces last-layer false positives.

> **Read the findings docs against [`../docs/registry.md`](../docs/registry.md).**
> An audit of every claim against the raw results found the headline numbers
> reproduce exactly, but also several documented claims that don't — including a
> `q1_language_emergence.md` DLA value that contradicts the paper, Gemma-3 missing from both
> findings docs, and an undisclosed dropped cell in Q5's n=17 statistic.

## Running it

```bash
cd compression
uv sync                          # builds compression/.venv (Python 3.11 pinned)
uv sync --extra quant            # + bitsandbytes / autoawq / auto-round backends
uv run pytest -m cpu             # method unit tests on a tiny CPU model
```

SLURM jobs are submitted from anywhere — each script `cd`s to this directory
itself:

```bash
sbatch --job-name=aya-expanse-8b experiments/q6-compression/slurm/run_q6gem.sh aya-expanse-8b
bash experiments/q6-compression/slurm/submit_gem.sh      # all models
python scripts/q6gem_collect.py                          # collect results
bash scripts/q6_status.sh                                # job status
```

Cluster notes (BYU RC / SLURM): pinned to **Python 3.11** (system OpenSSL
compatibility) and **torch cu128** (driver only supports up to CUDA 12.8).
Scripts set `OPENSSL_CONF=/dev/null` and `HF_HUB_OFFLINE=1` — compute nodes have
no internet, so models must be pre-cached from the login node. Hardware target
is a single A100 80GB.

The GGUF quantization path additionally needs llama.cpp, built separately via
`experiments/replication-uneven-ptq/build_llama_cpp.sh` (`GGML_NATIVE=OFF` is
required on this cluster).

## A note on `results/`

`results/` is gitignored and holds the run outputs (json/npz/png) that back the
findings docs. Quantized model checkpoints and GGUF conversion intermediates are
**not** kept — they are large and fully regenerable by re-running the jobs above.
