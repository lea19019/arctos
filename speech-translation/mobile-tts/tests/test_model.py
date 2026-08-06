"""
pytest tests for MMS-TTS (facebook/mms-tts-swh) mobile TTS project.
All tests run on CPU and complete in < 2 minutes.
"""

import os
import time
import numpy as np
import pytest

MODEL_CACHE = "/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/models/mms-tts-swh"
MODEL_ID = MODEL_CACHE if os.path.isdir(MODEL_CACHE) else "facebook/mms-tts-swh"

TRAIN_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "spk7_train.csv"
)


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)


def _get_model():
    from transformers import VitsModel
    import torch
    return VitsModel.from_pretrained(MODEL_ID).to("cpu")


def _run_inference(text="Habari."):
    import torch
    from transformers import AutoTokenizer, VitsModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID).to("cpu")
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform[0].squeeze().numpy()
    return waveform, model.config.sampling_rate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tokenizer_loads():
    tokenizer = _get_tokenizer()
    assert len(tokenizer) > 0, "Vocab size must be > 0"


def test_model_loads():
    model = _get_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 50_000_000, f"Expected > 50M params, got {n_params:,}"


def test_single_inference_cpu():
    waveform, _ = _run_inference("Habari.")
    assert isinstance(waveform, np.ndarray), "Output must be a numpy array"
    assert len(waveform) > 0, "Waveform must be non-empty"
    assert not np.any(np.isnan(waveform)), "Waveform contains NaN"
    assert not np.any(np.isinf(waveform)), "Waveform contains Inf"


def test_output_sample_rate():
    from transformers import VitsModel
    model = VitsModel.from_pretrained(MODEL_ID)
    assert model.config.sampling_rate == 16000, (
        f"Expected sampling_rate=16000, got {model.config.sampling_rate}"
    )


def test_rtf_smoke():
    start = time.perf_counter()
    waveform, sr = _run_inference("Habari.")
    elapsed = time.perf_counter() - start

    audio_duration = len(waveform) / sr
    rtf = elapsed / audio_duration
    assert rtf < 20.0, f"RTF {rtf:.2f} exceeds generous CPU bound of 20.0"


@pytest.mark.skip(reason="spk7_train.csv requires data/prep_swahili.py to run first")
def test_prep_csv_exists():
    assert os.path.isfile(TRAIN_CSV), (
        f"spk7_train.csv not found at {TRAIN_CSV}. Run data/prep_swahili.py first."
    )


def test_prep_csv_format():
    if not os.path.isfile(TRAIN_CSV):
        pytest.skip("spk7_train.csv not found; run data/prep_swahili.py first")

    import csv

    required_cols = {"audio_file", "text", "speaker_name"}
    rows = []
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        header = set(reader.fieldnames or [])
        assert required_cols.issubset(header), (
            f"Missing columns. Expected {required_cols}, got {header}"
        )
        for row in reader:
            rows.append(row)

    assert len(rows) > 0, "CSV is empty"

    for i, row in enumerate(rows):
        assert row["speaker_name"] == "7", (
            f"Row {i}: expected speaker_name='7', got '{row['speaker_name']}'"
        )
        assert os.path.isabs(row["audio_file"]), (
            f"Row {i}: audio_file must be an absolute path, got '{row['audio_file']}'"
        )
