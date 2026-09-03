"""Download every model in sw_models.MODELS into the local HF cache.

Run this on the LOGIN node — compute nodes have no internet, and the other
scripts load from this cache. Already-cached files are skipped, so re-running
is cheap and safe.

    uv run src/prefetch_models.py
"""

from huggingface_hub import snapshot_download

from sw_models import MODELS

# Skip files we never load (extra checkpoint formats, original Meta shards).
# "*.bin" is in here because every Table 2 model ships safetensors, and the
# older Llama/Mistral repos ship BOTH — downloading the .bin shards too cost
# 111 GB of pure duplicate before this line existed (cleaned 2026-09-02).
IGNORE = ["*.gguf", "*.pth", "*.h5", "*.msgpack", "original/*", "*.onnx",
          "*.bin", "*.bin.index.json"]


def main():
    failed = []
    for m in MODELS:
        print(f"--- {m}")
        try:
            path = snapshot_download(m, ignore_patterns=IGNORE)
            print(f"    ok: {path}")
        except Exception as e:                        # gated model, no token, ...
            print(f"    FAILED: {type(e).__name__}: {e}")
            failed.append(m)

    if failed:
        print("\nNot cached (fix access or comment them out in sw_models.py):")
        for m in failed:
            print(f"  {m}")
    else:
        print("\nAll models cached.")


if __name__ == "__main__":
    main()
