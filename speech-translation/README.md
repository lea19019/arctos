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

| Variant | chrF++ (en→es / en→fr / es→fr) | XCOMET-XL | tok/s | latency | peak VRAM |
|---|---|---|---|---|---|
| FP16 | 54.32 / 66.17 / 56.09 | .911 / .878 / .895 | ~1150 | 56–71 ms | 17.9–18.5 GB |
| BnB INT8 | 53.95 / 65.82 / 56.20 | .909 / .871 / .898 | ~264 | 239–296 ms | 17.5–18.2 GB |
| BnB NF4 | 53.55 / 65.77 / 55.02 | .909 / .866 / .888 | ~393 | 173–193 ms | 17.3–18.0 GB |
| **CTranslate2 INT8** | 53.96 / 65.36 / 56.28 | .911 / .863 / .894 | **~1515** | **23–28 ms** | **16.6–17.3 GB** |

**The load-bearing result: smaller is not faster.** `bitsandbytes` INT8 is
**4.3× slower than FP16** at essentially identical quality — its dequantization
overhead dominates at this model size. NF4 is also slower than FP16. Only
CTranslate2 INT8 delivers what compression is supposed to deliver: **2.4× faster
than FP16, lowest VRAM, quality within noise.** Any deployment path here goes
through fused-kernel runtimes, not `bitsandbytes`.

### XTTS v2, 50 examples per language

CER measured by Whisper-medium back-transcription; RTF < 1 means faster than
real time. Compression is applied to the GPT-2 core only — HiFi-GAN vocoder,
VQ-VAE codebook, and speaker perceiver stay FP16.

| Variant | lang | CER | WER | RTF | peak VRAM |
|---|---|---|---|---|---|
| FP16 | en / es / fr | .051 / .056 / .061 | .140 / .112 / .163 | ~0.30 | ~5.3 GB |
| BnB INT8 (GPT) | en / es / fr | .049 / **.042** / **.157** | .129 / .101 / .238 | ~0.31 | ~5.3 GB |

**INT8 on the GPT core is quality-neutral in English and Spanish — and breaks
French**, tripling CER (0.061 → 0.157) and nearly doubling WER. It also buys
almost nothing: no RTF gain and ~0.05 GB of VRAM. Two lessons: evaluate per
language, because an average would have hidden this entirely; and the XTTS GPT
core is not where the memory budget is won.

### `mobile-tts/` — Swahili speaker fine-tuning

A separate experiment: fine-tuning `facebook/mms-tts-swh` on a single Swahili
speaker for on-device TTS. Mean RTF 0.34 on GPU with a ~190 MB footprint.
Fine-tuning ran to checkpoint-15000; the loss curve is in `outputs/`, and
`outputs/listen/listen.html` is a curated listening set for the checkpoint.

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
