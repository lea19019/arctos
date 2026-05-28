"""Q3 experiment runner. TODO: implement."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q3 (MLPs and layers).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(
        f"TODO: implement Q3 runner; got config={args.config} output={args.output}"
    )


if __name__ == "__main__":
    main()
