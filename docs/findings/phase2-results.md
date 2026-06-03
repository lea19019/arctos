# Phase-two results (gem + extreme sweeps)

> Run 2026-06-02/03 on A100, XCOMET-XL + chrF++. n=24–32 sentences, greedy,
> generic prompt → **directional, not final**. 6/8 gem, 7/8 extreme complete at
> time of writing (gemma/bloom gem + llama extreme resubmitted after transient
> CUDA faults). Collect: `scripts/q6gem_collect.py [--subdir q6gem|q6extreme]`.

## Headline: MT-conditional GPTQ recovers the 3-bit cliff

GPTQ calibrated on **MT data** vs **generic (XNLI)** text, **W3, cs-de**, Δ = MT − generic:

| model | Δ chrF++ | Δ COMET |
|---|---|---|
| aya-expanse-8b | +31.7 | +0.235 |
| eurollm-9b | +31.1 | **+0.524** |
| tower-plus-9b | +24.8 | +0.238 |
| tower-instruct-7b | +23.4 | +0.126 |
| llama-3.1-8b | +21.5 | +0.148 |
| tower-base-7b | +19.6 | +0.228 |

**All six positive** at W3 on cs-de (+20–32 chrF++, +0.13–0.52 COMET). Absolute
(EuroLLM cs-de W3): RTN 28.3/0.587, **GPTQ-MT 40.1/0.797**, GPTQ-generic
**9.0/0.273** — i.e. generic-calibrated GPTQ is *worse than not quantizing*.

**Caveats on the gem:** the win is concentrated at **W3** (at W2 both collapse,
Δ≈0) and on **cs-de** (same-script); en-zh/en-arz are smaller and sometimes
negative in COMET (e.g. aya en-zh −0.078, tower-base en-zh −0.127). So the clean
claim is: *MT-calibrated GPTQ recovers the 3-bit cliff, most reliably on the
high-quality same-script pair; generic calibration actively hurts.*

## SHRINK cliff (RTN, chrF++ / COMET, cs-de)

| model | W4 | W3 | W2 | ternary | binary |
|---|---|---|---|---|---|
| aya | 68.5/.880 | **66.3/.877** | 6.8/.20 | 0.1 | 3.0 |
| tower-plus | 57.3/.852 | 29.0/.32 | 3.7 | 3.2 | 0.8 |
| gemma-3 | 54.4/.871 | 13.3/.39 | 11.7 | 10.2 | 10.7 |
| eurollm | 48.1/.831 | 28.3/.59 | 11.5 | 0.9 | 0.0 |
| tower-inst | 46.6/.677 | 31.6/.30 | 0.0 | 0.0 | 0.0 |
| tower-base | 35.1/.747 | 23.6/.32 | 3.6 | 0.0 | 2.9 |
| bloom | 23.3/.32 | 20.8/.31 | 0.2 | 2.3 | 1.0 |

W4 ≈ lossless; **W3 is the model-varying cliff** (Aya nearly lossless at 3-bit!;
others fall hard); **W2 and below collapse** for all.

## KEEP — salient-FP16 preservation

- **At W3 it recovers the cliff** (chrF++): EuroLLM 28.3→40.5, Llama 32.1→42.1,
  Tower-Inst 31.6→37.2, Gemma 12.7→48.4. A second independent lever to the gem.
- **At W2 / ternary / binary it does NOT** — rescue is small/noisy and near the
  floor (e.g. ternary mostly 0–6 chrF either way). 1% FP16 can't save a model
  whose bulk is destroyed at sub-2-bit. Honest negative.

## Confirmed negatives (report these)

- **Fisher / Hessian mixed-precision allocation underperforms uniform** at equal
  average bits (mixed−uniform COMET negative on every model). Dead lever as
  implemented (2/4-bit split).
- **MT calibration does not help AWQ** (only GPTQ) — AWQ uses calibration only
  for a per-channel scale, so domain barely matters; GPTQ reconstructs *on* the
  calibration activations, so it does.
- **COMET is unreliable below 2-bit:** it scores degenerate (empty/garbage)
  outputs ~0.20–0.65 even when chrF++ is 0.0 (e.g. tower-* ternary: chrF 0.0,
  COMET 0.65). Use chrF++ for collapse detection at extreme bits; corroborates
  Marchisio et al. (EMNLP 2024) that automatic metrics understate low-bit damage.

## Reading: the sweet spot is 3-bit MT, not sub-2-bit

PTQ (round-to-nearest, even with salient preservation) **cannot rescue
ternary/binary** — consistent with the literature that 1.58/1-bit needs
from-scratch QAT (BitNet), not PTQ. The defensible, novel contribution lives at
the **3-bit cliff**: *MT-conditional GPTQ + salient-channel preservation recover
3-bit translation quality, where generic-calibration methods (the WMT25 norm)
fail* — and GPTQ-for-MT is exactly what the closest prior work (2508.20893,
Chimoto 2601.18306) left untested. Super-weight strength is multilingual-model-
varying (EuroLLM KL 3.28 ≫ Gemma ≈ 0), an unstudied phenomenon in itself.
