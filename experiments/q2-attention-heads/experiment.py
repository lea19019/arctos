"""Q2 experiment runner. TODO: implement."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q2 (attention heads).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(
        f"TODO: implement Q2 runner; got config={args.config} output={args.output}"
    )


if __name__ == "__main__":
    main()
