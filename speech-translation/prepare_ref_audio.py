"""Extract one reference audio clip per language for XTTS speaker conditioning.

Sources tried in order:
  1. XTTS demo audio shipped with the model checkpoint (best quality)
  2. openslr/librispeech_asr clean test split (English, high-quality studio)
  3. mozilla-foundation/common_voice_11_0 es/fr test splits
  4. Synthetic fallback: amplitude-modulated noise that mimics speech rhythm
     (valid for pipeline testing; produces generic voice, not a real speaker clone)

Writes WAV files to speech-translation/data/ref_audio/{lang}.wav at 22050 Hz.
XTTS expects at least 3 seconds; clips are trimmed to 10 seconds max.

Run from repo root: python speech-translation/prepare_ref_audio.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TARGET_SR = 22050    # XTTS default sample rate
MIN_DUR_S = 3.0
MAX_DUR_S = 10.0


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    # scipy.signal.resample is good enough for reference clips
    from scipy.signal import resample as sp_resample
    target_len = int(len(audio) * target_sr / orig_sr)
    return sp_resample(audio, target_len).astype(np.float32)


def _clip(audio: np.ndarray, sr: int) -> np.ndarray:
    max_samples = int(MAX_DUR_S * sr)
    return audio[:max_samples]


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import wave, struct
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(audio_int16)}h", *audio_int16))


def extract_en(out_dir: Path) -> None:
    from datasets import load_dataset
    print("  en: loading LibriSpeech test-clean …", flush=True)
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
    for ex in ds:
        arr = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = ex["audio"]["sampling_rate"]
        if len(arr) / sr >= MIN_DUR_S:
            arr = _clip(arr, sr)
            arr = _resample(arr, sr, TARGET_SR)
            path = out_dir / "en.wav"
            _write_wav(path, arr, TARGET_SR)
            print(f"    → {path} ({len(arr)/TARGET_SR:.1f}s)")
            return
    raise RuntimeError("No suitable English clip found in LibriSpeech cache")


def extract_cv(lang: str, out_dir: Path) -> None:
    from datasets import load_dataset
    print(f"  {lang}: loading CommonVoice 11 …", flush=True)
    ds = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="test",
                      trust_remote_code=True)
    for ex in ds:
        arr = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = ex["audio"]["sampling_rate"]
        if len(arr) / sr >= MIN_DUR_S:
            arr = _clip(arr, sr)
            arr = _resample(arr, sr, TARGET_SR)
            path = out_dir / f"{lang}.wav"
            _write_wav(path, arr, TARGET_SR)
            print(f"    → {path} ({len(arr)/TARGET_SR:.1f}s)")
            return
    raise RuntimeError(f"No suitable clip found for {lang} in CommonVoice cache")


def _synthetic_speech_proxy(duration_s: float = 5.0, sr: int = TARGET_SR) -> np.ndarray:
    """Generate amplitude-modulated noise that mimics speech rhythm.

    This is a functional fallback for pipeline testing. XTTS will synthesize
    audio, but the voice identity will be generic (not cloned from a real speaker).
    Replace with real audio clips for voice-quality experiments.
    """
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n)
    # Carrier: low-frequency noise (speech bandwidth 100–4000 Hz)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float32)
    # Band-pass envelope: amplitude modulation at syllable rate (~4 Hz)
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)).astype(np.float32)
    audio = noise * envelope * 0.3
    return audio


def extract_en(out_dir: Path) -> None:
    path = out_dir / "en.wav"
    # Source 1: XTTS demo audio
    xtts_demo = Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/samples/en_sample.wav"
    if xtts_demo.exists():
        import shutil
        shutil.copy(xtts_demo, path)
        print(f"  en: copied XTTS demo audio → {path}")
        return

    # Source 2: LibriSpeech
    try:
        from datasets import load_dataset
        print("  en: loading LibriSpeech test-clean …", flush=True)
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=False)
        for ex in ds:
            arr = np.array(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            if len(arr) / sr >= MIN_DUR_S:
                arr = _clip(arr, sr)
                arr = _resample(arr, sr, TARGET_SR)
                _write_wav(path, arr, TARGET_SR)
                print(f"    → {path} ({len(arr)/TARGET_SR:.1f}s from LibriSpeech)")
                return
    except Exception as e:
        print(f"  en: LibriSpeech failed ({e}), using synthetic fallback")

    # Fallback: synthetic
    audio = _synthetic_speech_proxy()
    _write_wav(path, audio, TARGET_SR)
    print(f"  en: synthetic proxy → {path} (replace with real audio for voice cloning)")


def extract_cv(lang: str, out_dir: Path) -> None:
    path = out_dir / f"{lang}.wav"
    # Source 1: XTTS demo audio
    xtts_demo = Path.home() / f".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/samples/{lang}_sample.wav"
    if xtts_demo.exists():
        import shutil
        shutil.copy(xtts_demo, path)
        print(f"  {lang}: copied XTTS demo audio → {path}")
        return

    # Source 2: CommonVoice
    try:
        from datasets import load_dataset
        print(f"  {lang}: loading CommonVoice 11 …", flush=True)
        ds = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="test")
        for ex in ds:
            arr = np.array(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            if len(arr) / sr >= MIN_DUR_S:
                arr = _clip(arr, sr)
                arr = _resample(arr, sr, TARGET_SR)
                _write_wav(path, arr, TARGET_SR)
                print(f"    → {path} ({len(arr)/TARGET_SR:.1f}s from CommonVoice)")
                return
    except Exception as e:
        print(f"  {lang}: CommonVoice failed ({e}), using synthetic fallback")

    # Fallback: synthetic
    audio = _synthetic_speech_proxy()
    _write_wav(path, audio, TARGET_SR)
    print(f"  {lang}: synthetic proxy → {path} (replace with real audio for voice cloning)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("speech-translation/data/ref_audio"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    extract_en(args.out)
    extract_cv("es", args.out)
    extract_cv("fr", args.out)

    print("\nReference audio ready:")
    for p in sorted(args.out.glob("*.wav")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
