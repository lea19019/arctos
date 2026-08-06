#!/usr/bin/env python3
"""
finetune.py — Speaker-adaptation fine-tuning for facebook/mms-tts-swh.

Approach: posterior-encoder reconstruction loss.
  - Encode reference waveform → linear spectrogram → posterior_encoder → z
  - Decode z → reconstructed waveform
  - L1 loss on log-mel between reconstructed and reference (same timing, clean signal)

This avoids the temporal-misalignment problem of the text→audio forward pass,
where predicted durations differ from reference durations making L1 loss noisy.

No librosa or soundfile required; uses scipy.io.wavfile + torch.stft.

Usage:
    python finetune.py \
        --train-csv  /path/to/spk7_train.csv \
        --eval-csv   /path/to/spk7_eval.csv \
        --output-dir /path/to/checkpoints \
        [--model-cache /local/mms-tts-swh] \
        [--epochs 30] [--batch-size 8] [--lr 5e-5] \
        [--warmup-steps 200] [--save-steps 500] [--max-duration 10.0]
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.io import wavfile
from transformers import VitsModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

# VITS spectrogram parameters (must match model config: spectrogram_bins=513)
N_FFT      = 1024   # 1024//2+1 = 513 bins
HOP_LENGTH = 256    # decoder upsample product = 8×8×2×2 = 256

# ---------------------------------------------------------------------------
# Linear spectrogram (input to posterior encoder)
# ---------------------------------------------------------------------------

def compute_linear_spec(waveform: torch.Tensor) -> torch.Tensor:
    """
    Magnitude spectrogram expected by the VITS posterior encoder.

    Args:
        waveform: (T,) or (B, T) float tensor in [-1, 1].
    Returns:
        (513, F) or (B, 513, F) magnitude spectrogram.
    """
    squeeze = waveform.dim() == 1
    if squeeze:
        waveform = waveform.unsqueeze(0)
    B, T = waveform.shape
    window = torch.hann_window(N_FFT, device=waveform.device)
    stft = torch.stft(
        waveform.reshape(-1, T),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=False,
        return_complex=True,
    )  # (B, 513, F)
    spec = torch.sqrt(stft.real ** 2 + stft.imag ** 2 + 1e-9)
    if squeeze:
        spec = spec.squeeze(0)
    return spec  # (B, 513, F)


# ---------------------------------------------------------------------------
# Mel-spectrogram helpers (pure torch, no librosa)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """Return (n_mels, n_fft//2+1) triangular filterbank matrix."""
    if fmax is None:
        fmax = float(sr) / 2.0

    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0.0, float(sr) / 2.0, n_freqs)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([_mel_to_hz(m.item()) for m in mel_points])

    # Indices of hz_points in the linear frequency grid
    bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()

    filters = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_center):
            if f_center != f_left:
                filters[m - 1, k] = float(k - f_left) / float(f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filters[m - 1, k] = float(f_right - k) / float(f_right - f_center)

    return filters  # (n_mels, n_freqs)


# Cache filterbanks to avoid rebuilding each call
_FILTERBANK_CACHE: Dict[Tuple, torch.Tensor] = {}


def compute_log_mel(
    waveform: torch.Tensor,
    sr: int = 16000,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute log-mel spectrogram.

    Args:
        waveform: (T,) or (B, T) float tensor, values in [-1, 1].
        sr: sample rate.
        n_fft: FFT size.
        hop: hop length.
        n_mels: number of mel bins.
        fmin/fmax: mel filterbank frequency bounds.
        eps: floor before log.

    Returns:
        (n_mels, frames) or (B, n_mels, frames) float tensor.
    """
    squeeze = waveform.dim() == 1
    if squeeze:
        waveform = waveform.unsqueeze(0)  # (1, T)

    B, T = waveform.shape
    device = waveform.device

    # Hann window
    window = torch.hann_window(n_fft, device=device)

    # STFT — output: (B, n_fft//2+1, frames, 2)
    stft = torch.stft(
        waveform.reshape(-1, T),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # (B, freq, frames)

    power = stft.abs() ** 2  # (B, freq, frames)

    # Filterbank
    cache_key = (sr, n_fft, n_mels, fmin, fmax if fmax is not None else -1)
    if cache_key not in _FILTERBANK_CACHE:
        fb = _mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax)
        _FILTERBANK_CACHE[cache_key] = fb
    fb = _FILTERBANK_CACHE[cache_key].to(device)  # (n_mels, freq)

    mel = torch.matmul(fb, power)  # (B, n_mels, frames)
    log_mel = torch.log(mel.clamp(min=eps))

    if squeeze:
        log_mel = log_mel.squeeze(0)  # (n_mels, frames)
    return log_mel


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_wav_scipy(path: str) -> Tuple[np.ndarray, int]:
    """Load a .wav file with scipy.io.wavfile; return (float32 array, sr)."""
    sr, data = wavfile.read(path)
    # Convert integer dtypes to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    # Stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, int(sr)


def resample_torch(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Simple linear interpolation resampler (torch only).
    For close sample-rate ratios this is fine; for large ratios use torchaudio.
    """
    if orig_sr == target_sr:
        return waveform
    ratio = target_sr / orig_sr
    n_out = int(len(waveform) * ratio)
    # Use torch.nn.functional.interpolate (needs batch + channel dims)
    wav = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    wav_resampled = F.interpolate(wav, size=n_out, mode="linear", align_corners=False)
    return wav_resampled.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SwahiliDataset(Dataset):
    """
    Loads pipe-delimited CSV with columns: audio_file|text|speaker_name.
    Returns tokenized text and waveform tensor.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        target_sr: int = 16000,
        max_duration: float = 10.0,
    ):
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.max_samples = int(max_duration * target_sr)

        self.samples: List[Dict] = []
        skipped = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                audio_path = row["audio_file"].strip()
                text = row["text"].strip()
                if not audio_path or not text:
                    skipped += 1
                    continue
                if not os.path.isfile(audio_path):
                    skipped += 1
                    continue
                self.samples.append({"audio_file": audio_path, "text": text})

        print(
            f"[Dataset] {csv_path}: loaded {len(self.samples)} clips, "
            f"skipped {skipped} (missing file or empty)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = self.samples[idx]
        try:
            audio_np, sr = load_wav_scipy(item["audio_file"])
        except Exception as e:
            print(f"[Dataset] Warning: failed to load {item['audio_file']}: {e}")
            return None

        waveform = torch.from_numpy(audio_np).float()

        # Resample if needed
        if sr != self.target_sr:
            waveform = resample_torch(waveform, sr, self.target_sr)

        # Drop clips exceeding max duration
        if len(waveform) > self.max_samples:
            return None

        # Tokenize text
        encoding = self.tokenizer(
            item["text"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].squeeze(0)        # (L,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (L,)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "waveform": waveform,
            "waveform_length": torch.tensor(len(waveform), dtype=torch.long),
        }


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """Pad input_ids and waveforms to batch-max length; drop None samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_text_len = max(b["input_ids"].shape[0] for b in batch)
    max_wav_len = max(b["waveform"].shape[0] for b in batch)

    input_ids_list = []
    attention_mask_list = []
    waveform_list = []
    waveform_lengths = []

    for b in batch:
        # Pad text (right-pad with 1 — VitsTokenizer padding token is typically 1)
        text_pad = max_text_len - b["input_ids"].shape[0]
        input_ids_list.append(
            F.pad(b["input_ids"], (0, text_pad), value=1)
        )
        attention_mask_list.append(
            F.pad(b["attention_mask"], (0, text_pad), value=0)
        )
        # Pad waveform (right-pad with zeros)
        wav_pad = max_wav_len - b["waveform"].shape[0]
        waveform_list.append(F.pad(b["waveform"], (0, wav_pad), value=0.0))
        waveform_lengths.append(b["waveform_length"])

    return {
        "input_ids": torch.stack(input_ids_list),           # (B, L)
        "attention_mask": torch.stack(attention_mask_list), # (B, L)
        "waveform": torch.stack(waveform_list),             # (B, T)
        "waveform_length": torch.stack(waveform_lengths),   # (B,)
    }


# ---------------------------------------------------------------------------
# Mel loss
# ---------------------------------------------------------------------------

def mel_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    sr: int = 16000,
) -> torch.Tensor:
    """
    L1 loss between log-mel spectrograms of generated and reference waveforms.

    generated: (T_gen,) or (B, T_gen)
    reference: (T_ref,) or (B, T_ref)

    Spectrograms may differ in frame count; we crop to the shorter one.
    """
    log_mel_gen = compute_log_mel(generated, sr=sr)   # (..., n_mels, F_gen)
    log_mel_ref = compute_log_mel(reference, sr=sr)   # (..., n_mels, F_ref)

    # Align frame dimension
    min_frames = min(log_mel_gen.shape[-1], log_mel_ref.shape[-1])
    log_mel_gen = log_mel_gen[..., :min_frames]
    log_mel_ref = log_mel_ref[..., :min_frames]

    return F.l1_loss(log_mel_gen, log_mel_ref)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ---- Load model --------------------------------------------------------
    model_path = args.model_cache
    if model_path and Path(model_path).exists():
        print(f"[Train] Loading model from local cache: {model_path}")
        model = VitsModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        hf_id = "facebook/mms-tts-swh"
        print(f"[Train] Local cache not found; loading from HF Hub: {hf_id}")
        model = VitsModel.from_pretrained(hf_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_id)

    model.to(device)

    # ---- Freeze everything except the HiFi-GAN decoder --------------------
    # Posterior-encoder path: reference → posterior_encoder → z → decoder → recon
    # Only the decoder needs to adapt to the target speaker's voice.
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if name.startswith("decoder"):
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"[Train] Frozen   {frozen/1e6:.1f}M params (text_encoder, flow, posterior_encoder, duration_predictor)")
    print(f"[Train] Trainable {trainable/1e6:.1f}M params (decoder / HiFi-GAN)")

    model.train()

    # ---- Datasets ----------------------------------------------------------
    train_ds = SwahiliDataset(
        args.train_csv,
        tokenizer,
        target_sr=16000,
        max_duration=args.max_duration,
    )
    eval_ds = SwahiliDataset(
        args.eval_csv,
        tokenizer,
        target_sr=16000,
        max_duration=args.max_duration,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Optimizer & scheduler --------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

    steps_per_epoch = len(train_loader)
    # Each optimizer step covers grad_accum_steps forward passes
    optimizer_steps_per_epoch = (steps_per_epoch + args.grad_accum_steps - 1) // args.grad_accum_steps
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    print(
        f"[Train] epochs={args.epochs}, batches/epoch={steps_per_epoch}, "
        f"grad_accum={args.grad_accum_steps}, optimizer_steps/epoch={optimizer_steps_per_epoch}, "
        f"total_optimizer_steps={total_optimizer_steps}, warmup={args.warmup_steps}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop -----------------------------------------------------
    global_step = 0       # counts optimizer steps (after grad accumulation)
    batch_idx = 0         # counts forward passes
    best_eval_loss = float("inf")
    running_loss = 0.0
    running_count = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        optimizer.zero_grad()

        for batch in train_loader:
            if batch is None:
                batch_idx += 1
                continue

            reference_wav = batch["waveform"].to(device)          # (B, T)
            waveform_lengths = batch["waveform_length"].to(device) # (B,)

            # --- Posterior-encoder reconstruction path ---
            # 1. Linear spectrogram (same timing as reference)
            linear_spec = compute_linear_spec(reference_wav)       # (B, 513, F)
            T_frames = linear_spec.shape[-1]

            # 2. Validity mask: float (B, 1, F), 1.0=valid 0.0=padding
            spec_lengths = ((waveform_lengths - N_FFT) // HOP_LENGTH + 1).clamp(1, T_frames)
            padding_mask = (
                torch.arange(T_frames, device=device).unsqueeze(0)
                < spec_lengths.unsqueeze(1)
            ).float().unsqueeze(1)  # (B, 1, F)

            # 3. Posterior encoder: spec → z (latent aligned with reference)
            z, _, _ = model.posterior_encoder(linear_spec, padding_mask)  # (B, 192, F)

            # 4. Decoder: z → reconstructed waveform (same length as reference)
            reconstructed_wav = model.decoder(z)                   # (B, 1, T) or (B, T)
            if reconstructed_wav.dim() == 3:
                reconstructed_wav = reconstructed_wav.squeeze(1)   # (B, T)

            # 5. Mel L1 loss — aligned comparison, clean signal
            loss = mel_loss(reconstructed_wav, reference_wav, sr=16000) / args.grad_accum_steps
            loss.backward()

            del reconstructed_wav, z, linear_spec
            if device.type == "cuda" and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            batch_idx += 1
            unscaled_loss = loss.item() * args.grad_accum_steps
            running_loss += unscaled_loss
            running_count += 1
            epoch_loss += unscaled_loss
            epoch_batches += 1

            # Optimizer step after accumulating grad_accum_steps batches
            if batch_idx % args.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = running_loss / running_count
                    elapsed = time.time() - t0
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"[Step {global_step:>6}] epoch={epoch}/{args.epochs}  "
                        f"loss={avg:.4f}  lr={lr_now:.2e}  elapsed={elapsed:.0f}s"
                    )
                    running_loss = 0.0
                    running_count = 0

                if global_step % args.save_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, max_batches=20)
                    print(
                        f"[Eval  step={global_step}] eval_mel_loss={eval_loss:.4f}"
                        + (" *** best ***" if eval_loss < best_eval_loss else "")
                    )
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss

                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[Ckpt ] Saved → {ckpt_dir}")
                    model.train()

        avg_epoch = epoch_loss / max(epoch_batches, 1)
        print(f"[Epoch {epoch}/{args.epochs}] avg_train_loss={avg_epoch:.4f}")

    # ---- Final save --------------------------------------------------------
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[Done] Training complete.")
    print(f"       Best eval mel-loss : {best_eval_loss:.4f}")
    print(f"       Total steps        : {global_step}")
    print(f"       Final checkpoint   : {final_dir}")
    print(f"       Total time         : {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: VitsModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue
        reference_wav = batch["waveform"].to(device)
        waveform_lengths = batch["waveform_length"].to(device)

        linear_spec = compute_linear_spec(reference_wav)
        T_frames = linear_spec.shape[-1]
        spec_lengths = ((waveform_lengths - N_FFT) // HOP_LENGTH + 1).clamp(1, T_frames)
        padding_mask = (
            torch.arange(T_frames, device=device).unsqueeze(0)
            < spec_lengths.unsqueeze(1)
        ).float().unsqueeze(1)  # (B, 1, F)
        z, _, _ = model.posterior_encoder(linear_spec, padding_mask)
        reconstructed_wav = model.decoder(z)
        if reconstructed_wav.dim() == 3:
            reconstructed_wav = reconstructed_wav.squeeze(1)

        loss = mel_loss(reconstructed_wav, reference_wav, sr=16000)
        total_loss += loss.item()
        n_batches += 1

        if n_batches >= max_batches:
            break

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune facebook/mms-tts-swh for speaker adaptation "
                    "via mel-spectrogram reconstruction loss."
    )
    p.add_argument(
        "--model-cache",
        default="/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/models/mms-tts-swh",
        help="Local path to cached mms-tts-swh model directory (falls back to HF Hub).",
    )
    p.add_argument(
        "--train-csv",
        default="/home/vacl2/arctos/speech-translation/mobile-tts/data/spk7_train.csv",
        help="Pipe-delimited CSV: audio_file|text|speaker_name (training split).",
    )
    p.add_argument(
        "--eval-csv",
        default="/home/vacl2/arctos/speech-translation/mobile-tts/data/spk7_eval.csv",
        help="Pipe-delimited CSV: audio_file|text|speaker_name (eval split).",
    )
    p.add_argument(
        "--output-dir",
        default="/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/mobile-tts-checkpoints",
        help="Directory where checkpoints are saved.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--grad-accum-steps", type=int, default=2,
                   help="Accumulate gradients over N batches before optimizer step.")
    p.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Drop audio clips longer than this many seconds.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate inputs
    for attr, name in [("train_csv", "--train-csv"), ("eval_csv", "--eval-csv")]:
        path = getattr(args, attr)
        if not Path(path).exists():
            print(f"[Error] {name} not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("MMS-TTS-SWH Speaker Adaptation Fine-tuning")
    print("=" * 60)
    print(f"  model-cache  : {args.model_cache}")
    print(f"  train-csv    : {args.train_csv}")
    print(f"  eval-csv     : {args.eval_csv}")
    print(f"  output-dir   : {args.output_dir}")
    print(f"  epochs       : {args.epochs}")
    print(f"  batch-size   : {args.batch_size}")
    print(f"  lr           : {args.lr}")
    print(f"  warmup-steps : {args.warmup_steps}")
    print(f"  save-steps   : {args.save_steps}")
    print(f"  grad-accum   : {args.grad_accum_steps}")
    print(f"  max-duration : {args.max_duration}s")
    print("=" * 60)

    train(args)
