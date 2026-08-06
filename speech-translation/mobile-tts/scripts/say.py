#!/usr/bin/env python3
"""
say.py — quick listen-test CLI for the finetuned Swahili MMS-TTS (VITS) model.

Give it Swahili text, it synthesizes a WAV and (by default) appends an audio
player to a single self-contained HTML page you can scp to your laptop and open
in a browser — no cluster audio playback needed.

Examples
--------
    # one sentence, latest checkpoint, CPU (login-node friendly)
    python scripts/say.py "Habari yako leo?"

    # pick a specific checkpoint and compare against the base model A/B
    python scripts/say.py "Watoto wanacheza mpira." --checkpoint 8500 --compare-base

    # interactive: type sentences, each is synthesized and added to the page
    python scripts/say.py -i

    # run the standard eval sentences from config.py
    python scripts/say.py --preset

The HTML page (outputs/listen.html by default) accumulates every clip you
generate across runs, so you can build up a listening set and scp it once.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav_write
import torch
from transformers import VitsModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = Path(
    "/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/mobile-tts-checkpoints"
)
BASE_MODEL_CACHE = Path(
    "/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/models/mms-tts-swh"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "listen"
SAMPLE_RATE = 16000

PRESET_SENTENCES = [
    "Habari yako leo?",
    "Ninafurahi kukutana nawe.",
    "Watoto wanacheza mpira.",
    "Daktari anasema kwamba unahitaji kupumzika.",
    "Serikali imepanga mkutano mkubwa kesho asubuhi.",
]


# --------------------------------------------------------------------------- #
# checkpoint resolution
# --------------------------------------------------------------------------- #
def _ckpt_num(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", p.name)
    return int(m.group(1)) if m else -1


def resolve_checkpoint(spec: str) -> tuple[str, str]:
    """Return (model_path, label) for a --checkpoint spec.

    spec may be:
      - "latest"           -> highest checkpoint-N under CKPT_ROOT
      - "best"             -> checkpoint with lowest eval loss (parsed from *.out logs)
      - "base"             -> the un-finetuned base model cache
      - a bare number "8500" -> CKPT_ROOT/checkpoint-8500
      - a full path to a checkpoint dir
    """
    if spec == "base":
        return str(BASE_MODEL_CACHE), "base"

    if spec == "latest":
        ckpts = [p for p in CKPT_ROOT.glob("checkpoint-*") if p.is_dir()]
        if not ckpts:
            sys.exit(f"No checkpoints found under {CKPT_ROOT}")
        p = max(ckpts, key=_ckpt_num)
        return str(p), p.name

    if spec == "best":
        p = _find_best_checkpoint()
        return str(p), f"{p.name} (best)"

    if spec.isdigit():
        p = CKPT_ROOT / f"checkpoint-{spec}"
        if not p.is_dir():
            sys.exit(f"Checkpoint not found: {p}")
        return str(p), p.name

    p = Path(spec)
    if not p.is_dir():
        sys.exit(f"Checkpoint path not found: {p}")
    return str(p), p.name


def _find_best_checkpoint() -> Path:
    """Parse training *.out logs for the best eval step, snap to nearest checkpoint dir."""
    best_step, best_loss = None, float("inf")
    pat = re.compile(r"\[Eval\s+step=(\d+)\]\s+eval_mel_loss=([\d.]+).*best", re.I)
    for log in PROJECT_ROOT.glob("*.out"):
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        for m in pat.finditer(text):
            step, loss = int(m.group(1)), float(m.group(2))
            if loss < best_loss:
                best_step, best_loss = step, loss
    if best_step is None:
        print("  (no '*** best ***' eval lines found in logs; falling back to latest)")
        ckpts = [p for p in CKPT_ROOT.glob("checkpoint-*") if p.is_dir()]
        return max(ckpts, key=_ckpt_num)
    p = CKPT_ROOT / f"checkpoint-{best_step}"
    if not p.is_dir():
        # snap to the nearest existing checkpoint
        ckpts = [c for c in CKPT_ROOT.glob("checkpoint-*") if c.is_dir()]
        p = min(ckpts, key=lambda c: abs(_ckpt_num(c) - best_step))
    print(f"  best eval_mel_loss={best_loss:.4f} at step {best_step} -> {p.name}")
    return p


# --------------------------------------------------------------------------- #
# synthesis
# --------------------------------------------------------------------------- #
def load(model_path: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path).to(device).eval()
    return model, tok


def synth(model, tok, text: str, device: str) -> tuple[np.ndarray, float]:
    inputs = tok(text, return_tensors="pt")
    ids = inputs["input_ids"].to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids)
    dt = time.perf_counter() - t0
    wav = out.waveform.squeeze().cpu().numpy().astype(np.float32)
    return wav, dt


def wav_to_int16(wav: np.ndarray) -> np.ndarray:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16)


# --------------------------------------------------------------------------- #
# HTML listening page (self-contained, base64-embedded audio)
# --------------------------------------------------------------------------- #
def _wav_b64(wav: np.ndarray) -> str:
    buf = io.BytesIO()
    wav_write.write(buf, SAMPLE_RATE, wav_to_int16(wav))
    return base64.b64encode(buf.getvalue()).decode("ascii")


CARD_MARK = "<!--CLIP-CARD-->"

HTML_HEADER = """<!doctype html>
<meta charset="utf-8">
<title>Swahili TTS — listen</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;
      background:#0f1115;color:#e6e6e6}
 h1{font-size:1.25rem}
 .clip{border:1px solid #2a2f3a;border-radius:10px;padding:.75rem 1rem;margin:.75rem 0;background:#161a22}
 .txt{font-size:1.05rem;margin:0 0 .5rem}
 .meta{font-size:.8rem;color:#8b93a7;margin:.25rem 0 .5rem}
 audio{width:100%}
 code{background:#222733;padding:.1rem .3rem;border-radius:4px}
</style>
<h1>Swahili MMS-TTS — listening page</h1>
<p class="meta">Newest clips at the top. Regenerate any time; this page accumulates.</p>
"""


def append_clip_html(html_path: Path, text: str, wav: np.ndarray, label: str, rtf: float):
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    card = f"""{CARD_MARK}
<div class="clip">
  <p class="txt">{_escape(text)}</p>
  <p class="meta">{_escape(label)} &middot; {stamp} &middot; RTF {rtf:.3f} &middot; {len(wav)/SAMPLE_RATE:.2f}s</p>
  <audio controls preload="none" src="data:audio/wav;base64,{_wav_b64(wav)}"></audio>
</div>
"""
    if html_path.exists():
        existing = html_path.read_text()
        # insert newest card right after the header (before the first existing card)
        idx = existing.find(CARD_MARK)
        if idx == -1:
            new = existing + card
        else:
            new = existing[:idx] + card + "\n" + existing[idx:]
    else:
        new = HTML_HEADER + card
    html_path.write_text(new)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description="Synthesize Swahili text with a finetuned MMS-TTS checkpoint and "
        "collect the audio into a self-contained HTML page you can scp and play.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("text", nargs="*", help="Swahili text to speak (omit for -i or --preset)")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="REPL: type a sentence per line; each is synthesized")
    ap.add_argument("--preset", action="store_true",
                    help="Synthesize the 5 standard eval sentences")
    ap.add_argument("--file", type=str, help="Read one sentence per line from this file")
    ap.add_argument("--checkpoint", default="latest",
                    help="latest | best | base | <step> | <path>  (default: latest)")
    ap.add_argument("--compare-base", action="store_true",
                    help="Also synthesize each sentence with the base model, for A/B")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"where WAVs + listen.html go (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--no-html", action="store_true", help="skip the HTML page, just write WAVs")
    return ap.parse_args()


def collect_texts(args) -> list[str]:
    if args.preset:
        return list(PRESET_SENTENCES)
    if args.file:
        return [ln.strip() for ln in Path(args.file).read_text().splitlines() if ln.strip()]
    if args.text:
        return [" ".join(args.text)]
    return []


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "listen.html"

    model_path, label = resolve_checkpoint(args.checkpoint)
    print(f"Loading {label}: {model_path}  (device={args.device})")
    model, tok = load(model_path, args.device)

    base_model = base_tok = None
    if args.compare_base:
        bp, _ = resolve_checkpoint("base")
        print(f"Loading base for A/B: {bp}")
        base_model, base_tok = load(bp, args.device)

    def render(text: str, n: int | None = None):
        prefix = f"[{n}] " if n else ""
        wav, dt = synth(model, tok, text, args.device)
        rtf = dt / (len(wav) / SAMPLE_RATE)
        stem = _slug(text)
        wav_path = out_dir / f"{stem}__{label.split()[0]}.wav"
        wav_write.write(str(wav_path), SAMPLE_RATE, wav_to_int16(wav))
        print(f"  {prefix}{label}: {wav_path.name}  ({dt:.2f}s, RTF {rtf:.3f})")
        if not args.no_html:
            append_clip_html(html_path, text, wav, label, rtf)

        if base_model is not None:
            bwav, bdt = synth(base_model, base_tok, text, args.device)
            brtf = bdt / (len(bwav) / SAMPLE_RATE)
            bpath = out_dir / f"{stem}__base.wav"
            wav_write.write(str(bpath), SAMPLE_RATE, wav_to_int16(bwav))
            print(f"  {prefix}base : {bpath.name}  ({bdt:.2f}s, RTF {brtf:.3f})")
            if not args.no_html:
                append_clip_html(html_path, text, bwav, "base", brtf)

    if args.interactive:
        print("\nType Swahili text and press Enter (empty line or Ctrl-D to quit):")
        try:
            while True:
                line = input("swh> ").strip()
                if not line:
                    break
                render(line)
        except (EOFError, KeyboardInterrupt):
            print()
    else:
        texts = collect_texts(args)
        if not texts:
            sys.exit("No text given. Pass a sentence, --preset, --file, or -i.")
        for i, t in enumerate(texts, 1):
            print(f'Synthesizing: "{t}"')
            render(t, i if len(texts) > 1 else None)

    if not args.no_html and html_path.exists():
        print(f"\nHTML listening page: {html_path}")
        print("Pull it to your laptop and open in a browser:")
        print(f"  scp {_login_host()}:{html_path} .")


def _slug(text: str, n: int = 32) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (s[:n] or "clip")


def _login_host() -> str:
    import socket
    host = socket.gethostname()
    return f"{host}" if "." in host else f"{host}.rc.byu.edu"


if __name__ == "__main__":
    main()
