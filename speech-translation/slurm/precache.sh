#!/usr/bin/env bash
# precache.sh — run on the LOGIN NODE before submitting SLURM jobs.
#
# What this does:
#   1. Install missing Python packages into the project venv
#   2. Download XTTS v2 via the TTS library (requires internet)
#   3. Optionally convert NLLB to CTranslate2 INT8 format
#   4. Fetch FLORES+ data for en/es/fr
#   5. Extract reference audio clips for XTTS
#
# Usage (from repo root):
#   bash speech-translation/slurm/precache.sh [--skip-ct2] [--skip-xtts]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SKIP_CT2=0
SKIP_XTTS=0
for arg in "$@"; do
    case $arg in
        --skip-ct2)  SKIP_CT2=1 ;;
        --skip-xtts) SKIP_XTTS=1 ;;
    esac
done

echo "=== [1/6] Installing Python packages ==="
# Use uv pip for the project venv (no pip binary; uv manages the venv)
UV_PYTHON="${UV_PYTHON:-$HOME/arctos/speech-translation/.venv/bin/python}"
uv pip install --quiet \
    bitsandbytes \
    ctranslate2 \
    scipy

echo "=== [2/6] Installing Coqui TTS library ==="
if [ "$SKIP_XTTS" -eq 0 ]; then
    # torchaudio must come from the CUDA 12.8 wheel index (matches project torch)
    uv pip install --quiet torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
    # coqui-tts[codec] is required for torch>=2.9 audio IO
    uv pip install --quiet "coqui-tts[codec]>=0.25"
    # spacy core (no language model needed — XTTS uses rule-based sentencizer only)
    uv pip install --quiet spacy
else
    echo "  [skip] --skip-xtts passed"
fi

echo "=== [3/6] Downloading XTTS v2 checkpoint ==="
if [ "$SKIP_XTTS" -eq 0 ]; then
    # This downloads to ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/
    # Requires accepting Coqui license once (pass --agree-to-license or set TTS_AGREE_TERMS=1)
    TTS_HOME="${TTS_HOME:-$HOME/.local/share/tts}"
    XTTS_DIR="$TTS_HOME/tts_models--multilingual--multi-dataset--xtts_v2"
    if [ -d "$XTTS_DIR" ]; then
        echo "  Already downloaded at $XTTS_DIR"
    else
        echo "  Downloading XTTS v2 (requires Coqui license agreement) ..."
        speech-translation/.venv/bin/python -c "
from TTS.api import TTS
import os; os.environ['COQUI_TOS_AGREED'] = '1'
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True)
print('XTTS v2 downloaded.')
"
    fi
else
    echo "  [skip] --skip-xtts passed"
fi

echo "=== [4/6] Converting NLLB-600M to CTranslate2 INT8 ==="
CT2_OUT="speech-translation/results/nllb/ct2_int8"
if [ "$SKIP_CT2" -eq 0 ]; then
    if [ -d "$CT2_OUT" ]; then
        echo "  Already exists at $CT2_OUT"
    else
        mkdir -p "$(dirname "$CT2_OUT")"
        HF_MODEL="facebook/nllb-200-distilled-600M"
        echo "  Converting $HF_MODEL → $CT2_OUT ..."
        speech-translation/.venv/bin/python -m ctranslate2.tools.ct2_transformers_converter \
            --help > /dev/null 2>&1 || true
        # Use the installed ct2 converter
        ct2-transformers-converter \
            --model "$HF_MODEL" \
            --output_dir "$CT2_OUT" \
            --quantization int8 \
            --force
        echo "  CTranslate2 model at $CT2_OUT"
    fi
else
    echo "  [skip] --skip-ct2 passed"
fi

echo "=== [5/6] Fetching FLORES+ data ==="
if [ -f "speech-translation/data/nllb/eng_Latn-spa_Latn.jsonl" ]; then
    echo "  Already fetched (speech-translation/data/)"
else
    speech-translation/.venv/bin/python speech-translation/fetch_flores.py --n 200
fi

echo "=== [6/6] Preparing reference audio clips ==="
if [ -f "speech-translation/data/ref_audio/en.wav" ]; then
    echo "  Already prepared (speech-translation/data/ref_audio/)"
else
    if [ "$SKIP_XTTS" -eq 0 ]; then
        speech-translation/.venv/bin/python speech-translation/prepare_ref_audio.py
    else
        echo "  [skip] --skip-xtts passed"
    fi
fi

echo ""
echo "=== precache complete — ready to submit SLURM jobs ==="
echo "  sbatch speech-translation/slurm/run_nllb.sh"
echo "  sbatch speech-translation/slurm/run_xtts.sh"
