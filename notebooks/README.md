# Arctos tutorials — understanding the methods and the theory

Eight runnable tutorials that explain every interpretability method in the
project: the mechanics, the math, and why each one matters for the goal
(interpretability-guided quantization of MT models).

## How to run

These are **percent-format scripts** (`# %%` cell markers), not `.ipynb`.
That means:

- **In VS Code**: open any file and click "Run Cell" above each `# %%` — it
  renders as an interactive notebook (you already have the Python extension).
- **As a plain script**: `python notebooks/01_logit_lens.py` runs top to
  bottom.
- **Convert to a real notebook** if you prefer: `pip install jupytext` then
  `jupytext --to notebook notebooks/01_logit_lens.py`.

Run from the repo root so `import src...` works:

```bash
cd ~/arctos
.venv/bin/python notebooks/01_logit_lens.py
```

Notebooks 01–07 use a **tiny CPU model** (gpt2 or bloom-560m) so they run in
seconds without a GPU — they teach the mechanics. Notebook 08 reads the
**real experiment results** under `results/` and walks the full
interpretability→quantization reasoning chain.

## Order

| File | Method | One-line |
|------|--------|----------|
| `00_overview.py` | — | The thesis, the model set, what each method answers |
| `01_logit_lens.py` | logit lens | "What token would the model emit from layer ℓ?" |
| `02_probing.py` | probing | "Is feature X linearly decodable at layer ℓ?" (+ control tasks) |
| `03_ifr.py` | Information Flow Routes | "How much does component c contribute (magnitude)?" |
| `04_dla.py` | direct logit attribution | "Does c push toward/away the target token?" (signed) |
| `05_attribution_patching.py` | attribution patching | "If I corrupt c, how much does output change?" (causal) |
| `06_noise_sensitivity.py` | Q5 noise sweep | "If I quantize c, how much quality drops?" (the bridge) |
| `07_awq_activation_stats.py` | AWQ stats | "Which channels see big activations?" (the baseline) |
| `08_putting_it_together.py` | synthesis | Reads real results; the quantization rule |

## The one mental model to keep

Everything reads or intervenes on the **residual stream** — the additive
running sum `resid = embed + Σ attn_out + Σ mlp_out` that flows down the
transformer. Logit lens *reads* it at each layer; probing *classifies* it;
IFR and DLA *decompose* it into per-component contributions (unsigned vs
signed); patching and noise *intervene* on it causally. If the residual
stream makes sense, all eight methods make sense.
