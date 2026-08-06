"""Q4 experiment runner.

Reads existing Q1–Q3 outputs and synthesizes them into the
`docs/findings/architecture-comparison.md` writeup. Optional re-run on a
fixed shared-example set across all three models.

TODO: implement.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q4 (architecture comparison).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(
        f"TODO: implement Q4 runner; got config={args.config} output={args.output}"
    )


if __name__ == "__main__":
    main()
