"""XTTS v2 compression baseline: FP16 vs BnB INT8 on the GPT component.

Languages: English, Spanish, French (same 3-language set as NLLB experiment).
Metrics:
  CER       — character error rate via Whisper-medium ASR back-transcription
  RTF       — real-time factor (synthesis wall-clock / audio duration; <1 = faster than real-time)
  WER       — word error rate (Whisper, same pass)
  UTMOS     — neural MOS predictor (optional; requires utmos package or torch hub)

Component-level compression strategy:
  GPT-2 transformer  → FP16 baseline, then BnB INT8 (LLM.int8())
  HiFi-GAN vocoder   → always FP16 (fast single-pass conv; tested separately)
  VQ-VAE codebook    → always FP16 (lookup table; one wrong entry = audible artifact)
  Speaker Perceiver  → always FP16

Run from repo root:
    python speech-translation/xtts_experiment.py \\
        --config speech-translation/configs/xtts.yaml \\
        --output speech-translation/results/xtts \\
        [--n-examples 50] [--variants fp16 bnb_int8_gpt]

Prerequisites:
  1. python speech-translation/fetch_flores.py       (FLORES text)
  2. python speech-translation/prepare_ref_audio.py  (reference WAV clips)
  3. Run precache.sh on login node           (installs TTS library, downloads XTTS)

Outputs:
  results/xtts/results.json    — per-language per-variant scores
  results/xtts/summary.tsv     — human-readable table
  results/xtts/{variant}/{lang}/*.wav  — synthesized audio
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ── optional imports ──────────────────────────────────────────────────────────

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# ── torchaudio fallback (torchcodec needs libnvrtc which may be absent) ───────

def _patch_torchaudio_load() -> None:
    """Wrap torchaudio.load so torchcodec failures fall back to scipy.io.wavfile.

    torchcodec (the new torchaudio default backend in torch ≥ 2.9) links against
    libnvrtc.so.13 which is missing on some HPC compute nodes. Our reference audio
    files are plain WAV, so scipy is sufficient.
    """
    try:
        import functools
        import torchaudio
        original = torchaudio.load

        @functools.wraps(original)
        def _safe_load(filepath, *args, **kwargs):
            try:
                return original(filepath, *args, **kwargs)
            except (RuntimeError, OSError, ImportError):
                from scipy.io import wavfile
                sr, data = wavfile.read(str(filepath))
                if data.dtype == np.int16:
                    arr = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    arr = data.astype(np.float32) / 2147483648.0
                else:
                    arr = data.astype(np.float32)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                else:
                    arr = arr.T
                return torch.from_numpy(arr), sr

        torchaudio.load = _safe_load
    except ImportError:
        pass


_patch_torchaudio_load()

# ── CUDA probe ────────────────────────────────────────────────────────────────

def _cuda_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


_CUDA_OK: bool | None = None


def cuda_ok() -> bool:
    global _CUDA_OK
    if _CUDA_OK is None:
        _CUDA_OK = _cuda_ok()
    return _CUDA_OK

# ── FLORES text loading ───────────────────────────────────────────────────────

def load_mono_jsonl(data_dir: Path, flores_code: str, n: int) -> list[str]:
    path = data_dir / "mono" / f"{flores_code}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"FLORES data not found at {path}. "
            "Run: python speech-translation/fetch_flores.py"
        )
    texts = []
    with path.open() as f:
        for line in f:
            if len(texts) >= n:
                break
            texts.append(json.loads(line)["text"])
    return texts

# ── XTTS model loading ────────────────────────────────────────────────────────

def _find_xtts_checkpoint(checkpoint_dir: str | None) -> Path:
    if checkpoint_dir:
        p = Path(checkpoint_dir)
        if p.exists():
            return p

    # TTS library default location
    tts_cache = Path.home() / ".local" / "share" / "tts"
    candidates = list(tts_cache.glob("tts_models--multilingual--multi-dataset--xtts_v2"))
    if candidates:
        return candidates[0]

    # HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    hf_candidates = list(hf_cache.glob("models--coqui--XTTS-v2/snapshots/*"))
    if hf_candidates:
        return sorted(hf_candidates)[-1]

    raise FileNotFoundError(
        "XTTS v2 checkpoint not found. "
        "Run precache.sh to download, or set checkpoint_dir in xtts.yaml."
    )


def load_xtts_fp16(checkpoint_dir: str | None) -> "Xtts":
    if not HAS_TTS:
        raise ImportError("TTS library not installed. Run: pip install coqui-tts")

    ckpt_path = _find_xtts_checkpoint(checkpoint_dir)
    print(f"  Loading XTTS from {ckpt_path} …")

    config = XttsConfig()
    config.load_json(str(ckpt_path / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(ckpt_path), use_deepspeed=False)

    if cuda_ok():
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    return model


def _replace_linear_int8(module: torch.nn.Module, depth: int = 0) -> None:
    """Recursively replace nn.Linear with bnb.nn.Linear8bitLt in place."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            has_bias = child.bias is not None
            int8_layer = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=6.0,
            )
            int8_layer.weight = bnb.nn.Int8Params(
                child.weight.data.clone(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if has_bias:
                int8_layer.bias = torch.nn.Parameter(child.bias.data.clone())
            setattr(module, name, int8_layer)
        else:
            _replace_linear_int8(child, depth + 1)


def apply_bnb_int8_gpt(model: "Xtts", threshold: float = 6.0) -> "Xtts":
    if not HAS_BNB:
        raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")

    # The GPT component is at model.gpt.gpt (the actual nn.Module with transformer layers)
    # Quantize it; leave vocoder, speaker encoder, and mel encoder at FP16.
    target = model.gpt  # the XttsGPT wrapper; contains the decoder transformer

    n_linears = sum(1 for m in target.modules() if isinstance(m, torch.nn.Linear))
    print(f"  Quantizing {n_linears} Linear layers in GPT to INT8 …")

    _replace_linear_int8(target)

    # Move to CUDA to trigger BnB quantization (Int8Params quantizes on .cuda())
    model.gpt = target.cuda()
    print(f"  INT8 quantization applied.")
    return model

# ── synthesis ─────────────────────────────────────────────────────────────────

def _run_inference(model: "Xtts", text: str, lang: str, gpt_cond_latent, speaker_embedding) -> np.ndarray:
    out = model.inference(
        text,
        lang,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=True,
    )
    return np.array(out["wav"], dtype=np.float32)


def synthesize_batch(
    model: "Xtts",
    texts: list[str],
    lang: str,
    ref_audio_path: str,
    out_dir: Path,
    n_warmup: int = 1,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """Synthesize all texts with warmup. Returns (wav_list, sample_rate, timing_dict)."""
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[ref_audio_path],
        gpt_cond_len=30,
        gpt_cond_chunk_len=4,
        max_ref_length=60,
    )

    sample_rate = model.config.audio.output_sample_rate
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warmup: synthesize a short sentence, discard output
    if n_warmup > 0 and texts:
        for warm_text in texts[:min(n_warmup, len(texts))]:
            _run_inference(model, warm_text, lang, gpt_cond_latent, speaker_embedding)
        if cuda_ok():
            torch.cuda.synchronize()

    # Timed synthesis
    wavs: list[np.ndarray] = []
    total_audio_s = 0.0
    t_wall_start = time.perf_counter()

    for i, text in enumerate(texts):
        wav = _run_inference(model, text, lang, gpt_cond_latent, speaker_embedding)
        wavs.append(wav)
        total_audio_s += len(wav) / sample_rate
        _write_wav(out_dir / f"{i:04d}.wav", wav, sample_rate)

    if cuda_ok():
        torch.cuda.synchronize()
    total_synth_s = time.perf_counter() - t_wall_start

    rtf = total_synth_s / total_audio_s if total_audio_s > 0 else float("inf")
    timing = {
        "rtf":              round(rtf, 4),
        "elapsed_s":        round(total_synth_s, 3),
        "latency_ms":       round(total_synth_s / len(texts) * 1000, 1) if texts else 0.0,
        "audio_s_per_s":    round(total_audio_s / total_synth_s, 2) if total_synth_s > 0 else 0.0,
    }
    return wavs, sample_rate, timing


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import wave, struct
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(audio_int16)}h", *audio_int16))

# ── ASR back-transcription ────────────────────────────────────────────────────

_whisper_model = None
_whisper_processor = None


def load_whisper(hf_name: str = "openai/whisper-medium") -> None:
    global _whisper_model, _whisper_processor
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    print(f"  Loading Whisper from {hf_name} …")
    _whisper_processor = WhisperProcessor.from_pretrained(hf_name)
    _whisper_model = WhisperForConditionalGeneration.from_pretrained(hf_name)
    if cuda_ok():
        _whisper_model = _whisper_model.cuda()
    _whisper_model = _whisper_model.eval()


def transcribe(wav: np.ndarray, sr: int, lang: str) -> str:
    global _whisper_model, _whisper_processor
    if _whisper_model is None:
        raise RuntimeError("Whisper not loaded; call load_whisper() first")

    # Resample to 16kHz (Whisper requirement)
    if sr != 16000:
        from scipy.signal import resample as sp_resample
        wav = sp_resample(wav, int(len(wav) * 16000 / sr)).astype(np.float32)
        sr = 16000

    inputs = _whisper_processor(
        wav, sampling_rate=sr, return_tensors="pt", language=lang
    )
    device = next(_whisper_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pred_ids = _whisper_model.generate(
            inputs["input_features"],
            language=lang,
            task="transcribe",
        )
    return _whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

# ── CER / WER ────────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def compute_cer(hypothesis: str, reference: str) -> float:
    hyp = hypothesis.lower().replace(" ", "")
    ref = reference.lower().replace(" ", "")
    if not ref:
        return 0.0
    return _edit_distance(hyp, ref) / len(ref)


def compute_wer(hypothesis: str, reference: str) -> float:
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    if not ref_words:
        return 0.0
    return _edit_distance(hyp_words, ref_words) / len(ref_words)

# ── main ─────────────────────────────────────────────────────────────────────

def peak_vram_gb() -> float:
    if cuda_ok():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_peak_vram() -> None:
    if cuda_ok():
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config",     type=Path, default=Path("speech-translation/configs/xtts.yaml"))
    ap.add_argument("--output",     type=Path, default=Path("speech-translation/results/xtts"))
    ap.add_argument("--data-dir",   type=Path, default=Path("speech-translation/data"))
    ap.add_argument("--n-examples", type=int,  default=None)
    ap.add_argument("--n-warmup",   type=int,  default=1,
                    help="Warmup syntheses before timing (default 1)")
    ap.add_argument("--variants",   nargs="+", help="Subset of variant names to run")
    args = ap.parse_args()

    if not HAS_TTS:
        print("ERROR: TTS library not installed. Run: pip install coqui-tts", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    languages: list[str] = cfg["languages"]
    flores_codes: dict[str, str] = cfg["flores_codes"]
    variants: list[dict] = cfg["compression_variants"]
    n_examples: int = args.n_examples or cfg.get("n_examples", 50)
    ref_audio_dir = Path(cfg["reference_audio_dir"])
    ckpt_dir: str | None = cfg["model"].get("checkpoint_dir")

    whisper_name: str = cfg.get("eval", {}).get("whisper_model", "openai/whisper-medium")
    use_utmos: bool = cfg.get("eval", {}).get("use_utmos", False)

    print(f"Device: {'cuda' if cuda_ok() else 'cpu'}  |  CUDA usable: {cuda_ok()}\n")

    if args.variants:
        variants = [v for v in variants if v["name"] in args.variants]

    # Reference audio
    ref_paths: dict[str, str] = {}
    for lang in languages:
        p = ref_audio_dir / f"{lang}.wav"
        if not p.exists():
            raise FileNotFoundError(
                f"Reference audio not found at {p}. "
                "Run: python speech-translation/prepare_ref_audio.py"
            )
        ref_paths[lang] = str(p)

    # FLORES text
    lang_texts: dict[str, list[str]] = {}
    for lang in languages:
        lang_texts[lang] = load_mono_jsonl(args.data_dir, flores_codes[lang], n_examples)
    print(f"Loaded {n_examples} texts × {len(languages)} languages.\n")

    # Whisper loaded once; reused across variants
    load_whisper(whisper_name)

    all_results: dict[str, Any] = {}

    for variant in variants:
        vname = variant["name"]
        print(f"{'='*60}")
        print(f"Variant: {vname}  |  warmup={args.n_warmup}")
        print(f"{'='*60}")

        reset_peak_vram()

        try:
            model = load_xtts_fp16(ckpt_dir)
        except (FileNotFoundError, ImportError) as e:
            print(f"  [skip] {e}\n")
            continue

        if vname == "bnb_int8_gpt":
            if not HAS_BNB:
                print("  [skip] bitsandbytes not installed\n")
                del model
                continue
            if not cuda_ok():
                print("  [skip] BnB INT8 requires a functional CUDA GPU\n")
                del model
                continue
            model = apply_bnb_int8_gpt(model, threshold=variant.get("threshold", 6.0))

        load_vram = peak_vram_gb()
        print(f"  Load VRAM: {load_vram:.2f} GB\n")

        variant_results: dict[str, Any] = {"variant": vname}

        for lang in languages:
            n_sent = len(lang_texts[lang])
            print(f"  Language: {lang} ({n_sent} sentences + {args.n_warmup} warmup)")
            wav_out_dir = args.output / vname / lang

            reset_peak_vram()

            wavs, sr, timing = synthesize_batch(
                model,
                lang_texts[lang],
                lang,
                ref_paths[lang],
                wav_out_dir,
                n_warmup=args.n_warmup,
            )

            infer_vram = peak_vram_gb()

            # ASR back-transcription → CER / WER
            cer_scores, wer_scores = [], []
            for wav, ref_text in zip(wavs, lang_texts[lang]):
                hyp = transcribe(wav, sr, lang)
                cer_scores.append(compute_cer(hyp, ref_text))
                wer_scores.append(compute_wer(hyp, ref_text))

            mean_cer = float(np.mean(cer_scores))
            mean_wer = float(np.mean(wer_scores))

            lang_res: dict[str, Any] = {
                "cer":           round(mean_cer, 4),
                "wer":           round(mean_wer, 4),
                "rtf":           timing["rtf"],
                "elapsed_s":     timing["elapsed_s"],
                "latency_ms":    timing["latency_ms"],
                "audio_s_per_s": timing["audio_s_per_s"],
                "peak_vram_gb":  round(infer_vram, 2),
                "n_examples":    len(wavs),
                "n_warmup":      args.n_warmup,
            }

            if use_utmos:
                try:
                    predictor = torch.hub.load(
                        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
                    )
                    if cuda_ok():
                        predictor = predictor.cuda()
                    utmos_scores = []
                    for wav in wavs:
                        device = "cuda" if cuda_ok() else "cpu"
                        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
                        score = predictor(wav_t, sr).item()
                        utmos_scores.append(score)
                    lang_res["utmos"] = round(float(np.mean(utmos_scores)), 4)
                except Exception as e:
                    print(f"    [warn] UTMOS failed: {e}")
                    lang_res["utmos"] = None

            variant_results[lang] = lang_res
            print(f"    CER={mean_cer:.4f}  WER={mean_wer:.4f}  "
                  f"RTF={timing['rtf']:.3f}  ({timing['audio_s_per_s']:.1f}× real-time)  "
                  f"VRAM={infer_vram:.2f} GB")

        all_results[vname] = variant_results

        del model
        gc.collect()
        if cuda_ok():
            torch.cuda.empty_cache()
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = args.output / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Variant':<20} {'Lang':>5} {'CER':>7} {'WER':>7} {'RTF':>7} "
          f"{'ms/sent':>8} {'VRAM GB':>8}")
    print("-" * 80)
    for vname, vres in all_results.items():
        for lang, lres in vres.items():
            if not isinstance(lres, dict):
                continue
            print(f"{vname:<20} {lang:>5} {lres['cer']:>7.4f} {lres['wer']:>7.4f} "
                  f"{lres['rtf']:>7.3f} {lres['latency_ms']:>8.0f} "
                  f"{lres['peak_vram_gb']:>8.2f}")

    tsv_path = args.output / "summary.tsv"
    with tsv_path.open("w") as f:
        f.write("variant\tlang\tcer\twer\trtf\tlatency_ms\taudio_s_per_s\t"
                "peak_vram_gb\tn_examples\tutmos\n")
        for vname, vres in all_results.items():
            for lang, lres in vres.items():
                if not isinstance(lres, dict):
                    continue
                f.write(
                    f"{vname}\t{lang}\t{lres['cer']:.4f}\t{lres['wer']:.4f}\t"
                    f"{lres['rtf']:.4f}\t{lres['latency_ms']:.1f}\t"
                    f"{lres['audio_s_per_s']:.2f}\t{lres['peak_vram_gb']:.2f}\t"
                    f"{lres['n_examples']}\t{lres.get('utmos', '')}\n"
                )
    print(f"Summary TSV → {tsv_path}")


if __name__ == "__main__":
    main()
