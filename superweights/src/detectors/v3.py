"""Detector v3 — v2 plus the conditions the paper actually states.

Read Yu et al. 3.1 properly and four things v1/v2 never implemented:

  1. "If X_ik AND W_jk are both outliers" — both sides must be outliers,
     not just the product. (v2 let tiny weights sharing a column with a
     huge input spike look dominant.)
  2. "repeat ... until the magnitudes of large maximum activations are
     greatly suppressed" — stop when the spikes are gone, not when no
     candidate passes.
  3. "no more than three super weights" (six at most) — a plausibility bound.
  4. A super activation towers over the other CHANNELS at its token, not
     merely over other layers.

Known behaviour: 18 of 21 Table 2 coordinates, 124 extras across nine
models, 9 candidates on OLMo-1B. Still over-generates; v5 fixes that by
changing the tensor it measures.
"""

import statistics

import torch

TOWERING_FACTOR = 10   # layer's max|Y| vs the median layer
CHANNEL_FACTOR = 50    # |Y[t,j]| vs the median channel at that token
TOP_J = 4
TOP_W = 1000           # |W[j,k]| must rank this high within the layer
TOP_X = 10             # |X[t,k]| must rank this high at that token
EXPLAIN = 0.80         # prefix must explain at least this ...
OVERSHOOT = 1.25       # ... and at most this (a prefix over 1 cancels)
MAX_K = 4
SUPPRESSION = 0.10     # stop once max|Y| falls to this share of round 0's
MAX_SW = 8             # record, warn, and stop past this many
MAX_ROUNDS = 15


def forward_pass(model, layers, inputs):
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


def cut(v, n):
    """The n-th largest |entry| — the bar for 'is an outlier'."""
    flat = v.abs().flatten()
    return flat.topk(min(n, flat.numel())).values[-1].item()


def candidates(X, Y, W):
    """(j, k, weight, share) passing the paper's both-sides condition."""
    t = int(Y.abs().max(dim=-1).values.argmax())
    x, y = X[t], Y[t]
    w_cut, x_cut = cut(W, TOP_W), cut(x, TOP_X)
    y_cut = CHANNEL_FACTOR * y.abs().median().item()
    out = []
    for j in y.abs().topk(min(TOP_J, y.numel())).indices.tolist():
        if abs(y[j].item()) < y_cut:
            continue                                   # channel does not tower
        contrib = x * W[j]
        prefix, taken = 0.0, []
        for k in contrib.abs().argsort(descending=True)[:MAX_K].tolist():
            prefix += contrib[k].item()
            taken.append((j, k, W[j, k].item(), contrib[k].item() / y[j].item()))
            if abs(prefix) >= EXPLAIN * abs(y[j].item()):
                break
        ratio = abs(prefix / y[j].item())
        if not (EXPLAIN <= ratio <= OVERSHOOT):
            continue                                   # prefix cancels or falls short
        out += [c for c in taken
                if abs(W[c[0], c[1]]) >= w_cut and abs(x[c[1]]) >= x_cut]
    return out


def find(model, layers, inputs):
    """Returns [{"layer", "j", "k", "value"}, ...]. Leaves the model as found."""
    found, seen, max_y0 = [], set(), None
    for rnd in range(MAX_ROUNDS):
        store = forward_pass(model, layers, inputs)
        max_y = {i: Y.abs().max().item() for i, (X, Y) in store.items()}
        median = statistics.median(max_y.values())

        # the paper's stopping rule: spikes "greatly suppressed" vs round 0
        if max_y0 is None:
            max_y0 = max(max_y.values())
        elif max(max_y.values()) <= SUPPRESSION * max_y0:
            print(f"Round {rnd}: max|Y| suppressed {max_y0:.1f} -> "
                  f"{max(max_y.values()):.1f} — stopping.")
            break

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

        over = len(found) + len(fresh) > MAX_SW
        if over:
            print(f"Round {rnd}: exceeds MAX_SW={MAX_SW} (paper: at most six) — "
                  f"treat as over-generated")
        for L, j, k, w in fresh:
            seen.add((L, j, k))
            print(f"Round {rnd}: zeroing layer {L} W[{j},{k}] = {w:+.4f}")
            found.append({"layer": L, "j": j, "k": k, "value": w})
            with torch.no_grad():
                layers[L].mlp.down_proj.weight[j, k] = 0.0
        if over:
            break

    with torch.no_grad():
        for f in found:
            layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]
    return found
