"""Super-weight detection — the motor.

Loads a model, hands it to a detector brain, writes what it found. All the
detection logic lives in src/detectors/<name>.py, each exposing one function:

    find(model, layers, inputs) -> [{"layer", "j", "k", "value"}, ...]

so a new idea is a new file there, and this file never changes.

    uv run src/detect_sw.py --detector v5 --model allenai/OLMo-1B-0724-hf
    uv run src/detect_sw.py --detector v1 --model huggyllama/llama-7b --out x.json
"""

import argparse
import datetime
import importlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "Language modeling is "


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detector", default="v5", help="a module in src/detectors/")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default=None,
                    help="JSON for ablate_sw.py (default: results/<detector>/<model>_found.json)")
    args = ap.parse_args()

    brain = importlib.import_module(f"detectors.{args.detector}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto" if device == "cuda" else torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer([args.prompt], return_tensors="pt",
                       return_token_type_ids=False).to(device)

    found = brain.find(model, model.model.layers, inputs)
    print("\nFound super weights:", found)

    out = Path(args.out) if args.out else (
        Path("results") / args.detector / (args.model.replace("/", "_") + "_found.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "detector": args.detector,
        "prompt": args.prompt,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "dtype": str(model.dtype),
        "found": found,
    }, indent=2))
    print("written to", out)


if __name__ == "__main__":
    main()
