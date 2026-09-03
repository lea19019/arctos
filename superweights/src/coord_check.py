"""Is a Table 2 coordinate pointing at an outlier weight at all?

An ablation that does nothing has two very different explanations:
  (a) the coordinate is right and the paper's causal claim does not hold, or
  (b) the coordinate is wrong (typo, transposed, off-by-one layer) and we
      ablated an unremarkable weight.
Rank tells them apart. A super weight should be among the very largest
|values| in its own down_proj matrix; an arbitrary coordinate sits near the
middle of ~10^8 entries.

Reads one weight matrix at a time straight out of the safetensors shard, so
this is CPU-only and never materialises a whole model.

    uv run src/coord_check.py
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.utils import cached_file

from ablate_sw import TABLE2
from sw_models import MODELS


def load_down_proj(model_id, layer):
    """The layer's down_proj weight, pulled from whichever shard holds it."""
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    index = json.loads(Path(cached_file(model_id, "model.safetensors.index.json")).read_text())
    shard = index["weight_map"][name]
    with safe_open(cached_file(model_id, shard), framework="pt") as f:
        return f.get_tensor(name)


def rank_of(W, j, k):
    """1-based rank of |W[j,k]| among all |W| entries (1 = largest)."""
    v = W[j, k].abs()
    return int((W.abs() > v).sum().item()) + 1


def main():
    print(f"{'model':<34} {'coord':<18} {'weight':>9} {'rank':>12} "
          f"{'of':>12} {'pctile':>8}  transposed reading")
    print("-" * 118)
    for m in MODELS:
        for (layer, j, k) in TABLE2.get(m, []):
            W = load_down_proj(m, layer).float()
            n = W.numel()
            r = rank_of(W, j, k)
            # the same numbers read as [k, j]; only valid if k < rows
            if k < W.shape[0] and j < W.shape[1]:
                rt = rank_of(W, k, j)
                trans = f"W[{k},{j}]={W[k, j]:+.4f} rank {rt:,}"
            else:
                trans = "out of bounds (so the reading is unambiguous)"
            print(f"{m:<34} L{layer}[{j},{k}]".ljust(53) +
                  f"{W[j, k]:>+9.4f} {r:>12,} {n:>12,} "
                  f"{100 * (1 - r / n):>7.4f}%  {trans}")
            del W


if __name__ == "__main__":
    main()
