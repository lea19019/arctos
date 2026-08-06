#!/usr/bin/env python3
"""
test_inference.py — Swahili MMS-TTS inference benchmark
Model: facebook/mms-tts-swh (VITS-based, ~82M params)

Usage:
    python test_inference.py [--device cpu|cuda] [--model-cache PATH] [--output-dir PATH]
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import scipy.io.wavfile as wav_write
from transformers import VitsModel, AutoTokenizer


SENTENCES = [
    "Habari yako leo?",
    "Ninafurahi kukutana nawe.",
    "Watoto wanacheza mpira.",
    "Daktari anasema kwamba unahitaji kupumzika.",
    "Serikali imepanga mkutano mkubwa kesho asubuhi.",
]

SAMPLE_RATE = 16000  # MMS-TTS models output at 16 kHz
RTF_PASS_THRESHOLD = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description="MMS-TTS Swahili inference benchmark")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--model-cache",
        type=str,
        default="/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/models/mms-tts-swh",
        help="Path to model cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/vacl2/arctos/speech-translation/mobile-tts/outputs",
        help="Directory to save output WAV files and JSON summary",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_cache: str, device: str):
    """Load model + tokenizer from cache dir; fall back to HF Hub if not found."""
    cache_path = Path(model_cache)
    if cache_path.exists() and any(cache_path.iterdir()):
        print(f"Loading model from local cache: {cache_path}")
        model_id = str(cache_path)
    else:
        model_id = "facebook/mms-tts-swh"
        print(f"Cache not found or empty at {cache_path}. Downloading from HF Hub: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}.")
    return model, tokenizer


def run_inference(model, tokenizer, text: str, device: str):
    """
    Run TTS inference on a single text string.

    Returns:
        waveform (np.ndarray): 1-D float32 waveform
        infer_time (float): wall-clock seconds
        gpu_mem_mb (float | None): peak GPU memory in MB, or None if CPU
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    with torch.no_grad():
        output = model(input_ids)
    end = time.perf_counter()

    infer_time = end - start

    # output.waveform shape: [1, 1, T] or [1, T] — squeeze to 1-D
    waveform = output.waveform.squeeze().cpu().numpy().astype(np.float32)

    gpu_mem_mb = None
    if device == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    return waveform, infer_time, gpu_mem_mb


def save_wav(waveform: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Save float32 waveform as 16-bit PCM WAV via scipy."""
    # scipy.io.wavfile.write expects int16 or float32; float32 is fine
    wav_write.write(path, sample_rate, waveform)


def print_table(results: list, device: str):
    """Print a Unicode-box results table."""
    col_widths = {
        "idx": 3,
        "text": 18,
        "infer": 8,
        "audio": 8,
        "rtf": 7,
    }

    top    = "┌─────┬────────────────────┬──────────┬──────────┬─────────┐"
    header = "│  #  │ Text               │ Infer(s) │ Audio(s) │   RTF   │"
    sep    = "├─────┼────────────────────┼──────────┼──────────┼─────────┤"
    bottom = "└─────┴────────────────────┴──────────┴──────────┴─────────┘"

    print(f"\nDevice: {device}")
    print(top)
    print(header)
    print(sep)

    for r in results:
        idx_s    = str(r["sentence_index"]).center(3)
        text_s   = r["text"][:18].ljust(18)
        infer_s  = f"{r['infer_time_s']:.3f}".rjust(8)
        audio_s  = f"{r['audio_duration_s']:.3f}".rjust(8)
        rtf_s    = f"{r['rtf']:.3f}".rjust(7)
        print(f"│ {idx_s} │ {text_s} │ {infer_s} │ {audio_s} │ {rtf_s} │")

    print(bottom)

    mean_rtf = sum(r["rtf"] for r in results) / len(results)
    verdict = "PASS: RTF < 5.0 (usable)" if mean_rtf < RTF_PASS_THRESHOLD else "FAIL: RTF >= 5.0 (too slow)"
    print(f"\nMean RTF: {mean_rtf:.3f}")
    print(verdict)


def main():
    args = parse_args()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_cache, device)

    results = []

    for i, sentence in enumerate(SENTENCES, start=1):
        print(f"[{i}/{len(SENTENCES)}] Synthesising: {sentence!r}")
        waveform, infer_time, gpu_mem_mb = run_inference(model, tokenizer, sentence, device)

        audio_duration = len(waveform) / SAMPLE_RATE
        rtf = infer_time / audio_duration if audio_duration > 0 else float("inf")

        wav_path = output_dir / f"sentence_{i}_{device}.wav"
        save_wav(waveform, str(wav_path))

        entry = {
            "sentence_index": i,
            "text": sentence,
            "infer_time_s": round(infer_time, 6),
            "audio_duration_s": round(audio_duration, 6),
            "rtf": round(rtf, 6),
            "wav_path": str(wav_path),
        }
        if gpu_mem_mb is not None:
            entry["gpu_peak_mem_mb"] = round(gpu_mem_mb, 2)

        results.append(entry)
        print(f"    infer={infer_time:.3f}s  audio={audio_duration:.3f}s  RTF={rtf:.3f}")

    print_table(results, device)

    # Write JSON summary
    mean_rtf = sum(r["rtf"] for r in results) / len(results)
    summary = {
        "device": device,
        "model": "facebook/mms-tts-swh",
        "sample_rate": SAMPLE_RATE,
        "rtf_pass_threshold": RTF_PASS_THRESHOLD,
        "mean_rtf": round(mean_rtf, 6),
        "pass": mean_rtf < RTF_PASS_THRESHOLD,
        "sentences": results,
    }
    json_path = output_dir / f"inference_results_{device}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nJSON summary written to: {json_path}")


if __name__ == "__main__":
    main()
