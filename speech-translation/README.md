# speech-translation — compressing the NLLB + XTTS dubbing pipeline

The applied track: getting a **translation → voice-cloning dubbing pipeline**
(NLLB-200 + XTTS v2) to run together on a single cheap GPU at real-time latency,
instead of an A100. The constraint that drives everything here is fitting both
models in ~16 GB while staying faster than real time.

This is the **active** direction. The research that precedes it is in
[`../compression/`](../compression/).

```
speech-translation/
├── nllb_experiment.py      # NLLB-200: FP16 vs BnB INT8 vs BnB NF4 vs CTranslate2 INT8
├── xtts_experiment.py      # XTTS v2: FP16 vs BnB INT8 on the GPT component
├── fetch_flores.py         # FLORES+ dev data for en/es/fr → data/
├── prepare_ref_audio.py    # one reference clip per language for speaker conditioning
├── configs/                # nllb.yaml, xtts.yaml
├── slurm/                  # precache (login node) + run/smoke jobs (compute nodes)
├── mobile-tts/             # separate: Swahili MMS-TTS speaker fine-tuning
├── data/                   # gitignored — regenerate with the two fetch scripts
└── results/                # summary.tsv + results.json committed; audio/models are not
```

## What was measured

**Languages:** English, Spanish, French (the overlap of NLLB coverage and XTTS v2
support). **Data:** FLORES+ dev.

### NLLB-200 (distilled-600M), 100 examples per direction

| Variant | chrF++ (en→es / en→fr / es→fr) | XCOMET-XL | wall-clock | latency |
|---|---|---|---|---|
| FP16 | 54.32 / 66.17 / 56.09 | .911 / .878 / .895 | 5.59–7.06 s | 56–71 ms |
| BnB INT8 | 53.95 / 65.82 / 56.20 | .909 / .871 / .898 | 23.9–29.6 s | 239–296 ms |
| BnB NF4 | 53.55 / 65.77 / 55.02 | .909 / .866 / .888 | 17.3–19.3 s | 173–193 ms |
| **CTranslate2 INT8** | 53.96 / 65.36 / 56.28 | .911 / .863 / .894 | **2.28–2.80 s** | **23–28 ms** |

**The load-bearing result: smaller is not faster.** `bitsandbytes` INT8 is
**4.3× slower than FP16** at essentially identical quality — its dequantization
overhead dominates at this model size. NF4 is also slower than FP16. Only
CTranslate2 INT8 delivers a real speedup: **2.4× faster than FP16 in
wall-clock**, quality within roughly a chrF++ point. Any deployment path here
goes through fused-kernel runtimes, not `bitsandbytes`.

> **Two columns from the raw `summary.tsv` are deliberately omitted, because
> they don't measure what they appear to.**
> *`peak_vram_gb`* is read **after** XCOMET-XL is loaded on the same GPU, which
> is why a 600M FP16 model reports 17.9 GB — it is mostly the metric model.
> `torch.cuda.max_memory_allocated()` also cannot see CTranslate2's allocations
> at all, so the CT2 row is not comparable even in principle. No VRAM claim in
> this repo is currently supported; measuring it properly needs an isolated
> process.
> *`tok_per_sec`* is defined differently per backend — the HF path counts padded
> tensor elements, the CT2 path counts real hypothesis tokens. Wall-clock is the
> comparable quantity, so that is what's shown.
>
> Also note the quality gap is a **single n=100 run with no confidence
> intervals**; CT2 loses 0.81 chrF++ and 0.015 XCOMET-XL on en→fr, the largest
> gap in the table, and "within noise" has not actually been established.

### XTTS v2, 50 examples per language

CER measured by Whisper-medium back-transcription; RTF < 1 means faster than
real time. Compression is applied to the GPT-2 core only — HiFi-GAN vocoder,
VQ-VAE codebook, and speaker perceiver stay FP16.

| Variant | lang | CER | WER | RTF | peak VRAM |
|---|---|---|---|---|---|
| FP16 | en / es / fr | .051 / .056 / .061 | .140 / .112 / .163 | ~0.30 | ~5.3 GB |
| BnB INT8 (GPT) | en / es / fr | .049 / **.042** / **.157** | .129 / .101 / .238 | ~0.31 | ~5.3 GB |

**INT8 on the GPT core is quality-neutral in English and Spanish — and breaks
French**, raising CER **2.6×** (0.0610 → 0.1565) and WER 1.5× (0.163 → 0.238).
It also buys nothing measurable: no RTF gain, and the VRAM column is unreliable
for the reason given above (English drops 0.08 GB, Spanish *rises* 0.14 GB).
Two lessons: evaluate per language, because an average would have hidden this
entirely; and the XTTS GPT core is not where the memory budget is won.

Note the config file disagrees with what ran — `configs/xtts.yaml` claims the
HiFi-GAN vocoder was "tested at INT8" and the perceiver is "always INT8", but
only two variants exist (`fp16`, `bnb_int8_gpt`) and no vocoder or perceiver
quantization was ever run.

### `mobile-tts/` — Swahili speaker fine-tuning

A separate experiment: fine-tuning `facebook/mms-tts-swh` on a single Swahili
speaker (13,146 training clips) for on-device TTS. Training completed — 30/30
epochs, 15,385 steps, ~48 min on an H200, best eval mel-loss **1.0848 at step
15000**, after three failed attempts (OOM, a `conv_pre` shape error, and a SLURM
cancellation). The loss curve is in `outputs/`, and `outputs/listen/listen.html`
is a curated listening set from checkpoint-15000.

**What is not established:** there is no quality evaluation of the fine-tuned
model's text→speech path. The training objective is a *posterior-encoder
reconstruction* loss (waveform → spectrogram → posterior encoder → decode → L1
log-mel), chosen deliberately to avoid the temporal-misalignment problem of the
text→audio forward pass — and the eval loss measures that same reconstruction.
No CER, WER, MOS/UTMOS, or speaker-similarity number exists for the fine-tune;
the only evidence is subjective listening. The RTF and footprint figures
sometimes quoted for this work (0.34, ~190 MB) are from a **pre-fine-tune smoke
test** of the base model on June 22, and that mean is dominated by a cold-start
outlier — the four warm calls average **0.043**.

Note `mobile-tts/config.py` points at group-scratch paths under
`/home/vacl2/groups/grp_mtlab/nobackup/autodelete/` — the corpus and checkpoints
live there, not in this repo, and that scratch space auto-deletes.

## Running it

```bash
cd speech-translation
uv sync                       # builds speech-translation/.venv (Python 3.11)
uv sync --extra ct2 --extra quant   # + CTranslate2 and bitsandbytes backends
```

Compute nodes have no internet, so caching happens on the login node first:

```bash
bash slurm/precache.sh        # FLORES+, models, reference audio, XCOMET-XL
```

Then, **submitted from the repo root**:

```bash
sbatch speech-translation/slurm/smoke_nllb_gpu.sh   # quick sanity check
sbatch speech-translation/slurm/run_nllb.sh
sbatch speech-translation/slurm/run_xtts.sh
sbatch speech-translation/slurm/run_nllb_ct2.sh     # converts + evaluates the CT2 INT8 model
```

The `mobile-tts/` jobs are self-contained and use the system Python:

```bash
sbatch speech-translation/mobile-tts/slurm/finetune.sh
```

## What's kept and what isn't

`results/*/summary.tsv` and `results/*/results.json` are committed — they are the
measurements, and they're small. **Not kept:** the converted CTranslate2 model
(`results/nllb/ct2_int8/`, 600 MB — regenerate with `run_nllb_ct2.sh`) and the
bulk synthesized audio. One FP16-vs-INT8 English pair is committed per XTTS
variant as a listening reference; the rest of the local audio is gitignored.

`data/` is gitignored and fully regenerable via `fetch_flores.py` and
`prepare_ref_audio.py`.

## Related reading

- [`../docs/findings/compression-nllb-xtts-research.md`](../docs/findings/compression-nllb-xtts-research.md) — survey of PTQ/pruning/distillation for both model families.
- [`../docs/findings/interp-lrl-nllb-xtts.md`](../docs/findings/interp-lrl-nllb-xtts.md) — the low-resource-language preservation angle.
- [`../docs/OPEN-WORK.md`](../docs/OPEN-WORK.md) — ranked open directions, including the `nllb-encdec` and `tts-xtts` tracks.
