"""Q1 experiment runner.

Composes:
- src.interp.probing.probe_layers
- src.interp.logit_lens.logit_lens
- src.interp.ifr.ifr   (cross-check)

Reads configs/{model}.yaml; writes per-model results to results/{model}/q1/.

TODO: implement.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q1 (language emergence).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(
        f"TODO: implement Q1 runner; got config={args.config} output={args.output}"
    )


if __name__ == "__main__":
    main()
