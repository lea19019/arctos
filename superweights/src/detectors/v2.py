"""Detector v2 — v1's spikes, but several output channels per layer and a
contribution decomposition instead of a single argmax.

Why: v1 takes ONE argmax per layer, so a layer holding two super weights
(Llama-13B row 2231, Phi-3 column 808) can only ever surrender one. v2 looks
at the top TOP_J output channels of the spike token, decomposes each into
per-k contributions, and keeps the smallest prefix explaining EXPLAIN of it.

Known behaviour: recovers 20 of 21 Table 2 coordinates but returns ~1,900
extra candidates across nine models — a candidate generator, not a
detector. Its failure taught what v3 and v5 fix.
"""

import statistics

import torch

TOWERING_FACTOR = 10   # layer's max|Y| vs the median layer
TOP_J = 4              # output channels examined per layer
EXPLAIN = 0.80         # prefix must explain this share of Y[t,j]
MAX_K = 4              # cap on contributors per channel
MAX_ROUNDS = 15


def forward_pass(model, layers, inputs):
    """Keep every layer's down_proj input and output for the whole prompt."""
    store = {}

    def make_hook(i):
        def hook(module, inp, out):
            store[i] = (inp[0][0].float().cpu(), out[0].float().cpu())
        return hook

    handles = [l.mlp.down_proj.register_forward_hook(make_hook(i))
               for i, l in enumerate(layers)]
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return store


def candidates(X, Y, W):
    """(j, k, weight, share) for the top TOP_J channels of the spike token."""
    t = int(Y.abs().max(dim=-1).values.argmax())
    x, y = X[t], Y[t]
    out = []
    for j in y.abs().topk(min(TOP_J, y.numel())).indices.tolist():
        contrib = x * W[j]
        prefix = 0.0
        for k in contrib.abs().argsort(descending=True)[:MAX_K].tolist():
            share = contrib[k].item() / y[j].item()
            prefix += share
            out.append((j, k, W[j, k].item(), share))
            if abs(prefix) >= EXPLAIN:
                break
    return out


def find(model, layers, inputs):
    """Returns [{"layer", "j", "k", "value"}, ...]. Leaves the model as found."""
    found, seen = [], set()
    for rnd in range(MAX_ROUNDS):
        store = forward_pass(model, layers, inputs)
        max_y = {i: Y.abs().max().item() for i, (X, Y) in store.items()}
        median = statistics.median(max_y.values())
        towering = sorted((i for i in store if max_y[i] > TOWERING_FACTOR * median),
                          key=lambda i: -max_y[i])
        print(f"Round {rnd}: {len(towering)} of {len(store)} layers tower "
              f"(median max|Y| = {median:.2f})")

        fresh = []
        for L in towering:
            X, Y = store[L]
            W = layers[L].mlp.down_proj.weight.float().cpu()
            for j, k, w, share in candidates(X, Y, W):
                print(f"  layer {L:2d}  W[{j},{k}] = {w:+.4f}  max|Y|={max_y[L]:.1f}  "
                      f"share={share:+.3f}")
                if (L, j, k) not in seen:
                    fresh.append((L, j, k, w))
        if not fresh:
            print(f"Round {rnd}: nothing new — stopping.")
            break

        for L, j, k, w in fresh:
            seen.add((L, j, k))
            print(f"Round {rnd}: zeroing layer {L} W[{j},{k}] = {w:+.4f}")
            found.append({"layer": L, "j": j, "k": k, "value": w})
            with torch.no_grad():
                layers[L].mlp.down_proj.weight[j, k] = 0.0

    with torch.no_grad():
        for f in found:
            layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]
    return found
