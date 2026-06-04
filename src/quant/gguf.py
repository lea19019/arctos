"""GGUF / llama.cpp quantization path — paper's "GGUF" method.

The paper uses llama.cpp K-quantization with an importance matrix (imatrix):
Q4_K_M (4-bit) and Q2_K (2-bit), imatrix estimated from 20k WikiText samples
at context length 512, batch 512. This is a *separate generation backend* from
HF transformers, so we are careful to keep it comparable:

  * The prompt fed to llama.cpp is rendered by the model's HF tokenizer
    ``apply_chat_template`` (via ``render_chat_prompt``) and passed verbatim —
    llama.cpp's own chat templating is never used.
  * Generation is greedy (temperature 0, top-k 1), same as the HF path.
  * The same WMT24++ sources, references, and COMET/chrF scorers are used.

Pipeline (all via the prebuilt llama.cpp binaries under ``$LLAMA_DIR``):
  convert_to_f16  ->  build_imatrix  ->  quantize  ->  (serve + translate)

The f16 base GGUF is as large as the model; callers delete it right after
quantizing (see the experiment driver's disk policy).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

from src.data.wmt24pp import TranslationExample
from src.models._chat_prompt import render_chat_prompt

SUPPORTED_BITS = (4, 2)

# GGUF quant type per bit-width (the paper's exact choices).
QTYPE = {4: "Q4_K_M", 2: "Q2_K"}

LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", str(Path.home() / "llama.cpp")))
_BIN = LLAMA_DIR / "build" / "bin"
_CONVERT = LLAMA_DIR / "convert_hf_to_gguf.py"

# imatrix estimation params (paper §4.3).
IMATRIX_CTX = 512
IMATRIX_BATCH = 512


def is_valid_gguf(path: str) -> bool:
    """True iff ``path`` exists and starts with the GGUF magic bytes.

    Guards against reusing a truncated/partial GGUF left by an interrupted
    convert/quantize (e.g. a job killed at the SLURM time wall, or a stale
    file from a cancelled run) — llama.cpp would otherwise fail to load it with
    'invalid magic characters'. Callers rebuild when this returns False.
    """
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _bin(name: str) -> str:
    p = _BIN / name
    if not p.exists():
        raise FileNotFoundError(
            f"llama.cpp binary {name} not found at {p}. "
            f"Build it via experiments/replication-uneven-ptq/build_llama_cpp.sh."
        )
    return str(p)


def _run(cmd: list[str], log_prefix: str) -> None:
    print(f">> [{log_prefix}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _resolve_local_dir(model: str) -> str:
    """Return a local model directory for ``model`` (an HF repo id or a path).

    ``convert_hf_to_gguf.py`` needs a directory, not a repo id, so resolve repo
    ids to their cached snapshot (offline-safe; weights are pre-cached)."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, local_files_only=True)


def convert_to_f16(base_dir: str, out_f16: str) -> str:
    """Convert an HF model (repo id or dir) to an f16 GGUF (base for quant)."""
    import sys

    local_dir = _resolve_local_dir(base_dir)
    os.makedirs(os.path.dirname(out_f16) or ".", exist_ok=True)
    _run(
        [sys.executable, str(_CONVERT), local_dir, "--outfile", out_f16, "--outtype", "f16"],
        "convert",
    )
    return out_f16


def build_imatrix(f16_path: str, calib_txt: str, out_imatrix: str) -> str:
    """Estimate an importance matrix from a calibration text file."""
    os.makedirs(os.path.dirname(out_imatrix) or ".", exist_ok=True)
    _run(
        [
            _bin("llama-imatrix"),
            "-m", f16_path,
            "-f", calib_txt,
            "-o", out_imatrix,
            "-c", str(IMATRIX_CTX),
            "-b", str(IMATRIX_BATCH),
            "-ngl", "99",
        ],
        "imatrix",
    )
    return out_imatrix


def quantize(f16_path: str, bits: int, out_path: str, imatrix: str | None = None) -> str:
    """Quantize an f16 GGUF to Q4_K_M/Q2_K, optionally guided by an imatrix."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"GGUF supports {SUPPORTED_BITS}, got {bits}-bit.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [_bin("llama-quantize")]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += [f16_path, out_path, QTYPE[bits]]
    _run(cmd, "quantize")
    return out_path


# ---- resident server for batched greedy generation --------------------------


@contextmanager
def serve(gguf_path: str, *, port: int = 8080, n_parallel: int = 4, ctx: int = 4096):
    """Start a llama-server with the model resident; yield its base URL.

    Loading the model once (vs reloading per sentence with llama-cli) is what
    makes GGUF inference over ~1000 segments tractable.
    """
    log = open(f"{gguf_path}.server.log", "w")
    proc = subprocess.Popen(
        [
            _bin("llama-server"),
            "-m", gguf_path,
            "-ngl", "99",
            "-c", str(ctx * n_parallel),
            "--parallel", str(n_parallel),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-webui",
            # Disable the RAM prompt cache: it accumulates ~60 MiB of saved
            # state per request and, under the ~1000-request-per-direction load,
            # eventually errors the server with HTTP 500. We never reuse prompts
            # across segments anyway (cache_prompt=False), so 0 = off is correct.
            "--cache-ram", "0",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(base, proc, timeout=900, log_path=log.name)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def _wait_healthy(base: str, proc, *, timeout: int, log_path: str | None = None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            if log_path:
                try:
                    tail = "\n".join(open(log_path).read().splitlines()[-15:])
                except OSError:
                    pass
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}).\n{tail}")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError("llama-server did not become healthy in time.")


def _completion(base: str, prompt: str, max_tokens: int, *, retries: int = 4) -> str:
    """One greedy completion via /completion, with retries on transient errors.

    Returns the generated text, or "" if the request keeps failing after
    ``retries`` attempts (a single un-translatable segment must not abort the
    whole ~1000-segment direction; an empty hypothesis scores honestly low).
    """
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 0,
            "cache_prompt": False,
            "stream": False,
        }
    ).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{base}/completion", data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["content"].strip()
        except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"   [gguf] completion failed after {retries} retries ({last_err}); "
          f"using empty hypothesis", flush=True)
    return ""


def translate(
    gguf_path: str,
    tokenizer,
    examples: Sequence[TranslationExample],
    *,
    chat_kwargs: dict | None = None,
    max_new_tokens: int = 512,
    port: int = 8080,
    n_parallel: int = 4,
) -> list[str]:
    """Greedy-decode translations for one direction via a resident server.

    ``tokenizer`` is used only to render the chat-templated prompt string
    (identical to the HF path); the GGUF weights do the generation.
    """
    from concurrent.futures import ThreadPoolExecutor

    prompts = [
        render_chat_prompt(
            tokenizer, ex.source, ex.src_lang, ex.tgt_lang, chat_kwargs=chat_kwargs
        )
        for ex in examples
    ]
    with serve(gguf_path, port=port, n_parallel=n_parallel) as base:
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            hyps = list(
                pool.map(lambda p: _completion(base, p, max_new_tokens), prompts)
            )
    return hyps
