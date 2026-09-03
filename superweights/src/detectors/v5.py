"""Detector v5 — residual-stream persistence (Yu et al. 2024, Figure 4).

The readable, standalone version of this brain is src/sw_v5.py; this file
is the same logic as a module so the motor (src/detect_sw.py) can call it.

Why a different signal than v1: v1 reads spikes off each layer's down_proj
OUTPUT. The paper's Figure 4 is about the RESIDUAL STREAM — the super
activation appears right after the super weight's layer and persists at
the same magnitude to the end. On OLMo-1B the residual max goes 0.2, 5.0,
267.7 (layer 2), ~420 for 13 layers, then 7.6 as the last layer removes it.
That fingerprint also kills the false positive every down_proj version hit:
OLMo-1B's LAST layer emits the biggest down_proj spike in the model (419.5,
above the real super weight's 266.9) and creates nothing that persists.

    1. find residual channels that are massive AND persist across depth
    2. the layer where each first appears is where its super weight lives
    3. only there, decompose Y[t, j] = sum_k X[t, k] * W[j, k] to find k
    4. zero, repeat until no super activation survives (the paper's rule)
"""

import torch

MASSIVE_FRAC = 0.10   # a super activation is >= 10% of the model's largest peak
PLATEAU = 0.5         # "on the plateau" = within half of the channel's own peak
MIN_PERSIST = 0.5     # must hold the plateau for at least half the later layers
TOP_W = 1000          # W[j,k] must rank this high by |W| within its matrix
TOP_X = 10            # X[t,k] must rank this high by |X| at that token
MIN_SHARE = 0.35      # a contributor must carry this share of Y[t,j]
EXPLAIN = 0.8         # take contributors until this share is explained
MAX_ROUNDS = 10


def forward_pass(model, layers, inputs):
    """One pass. Residual stream at every layer AND each down_proj's in/out."""
    store = {}

    def make_hook(i):
        def hook(module, inp, out):
            store[i] = (inp[0][0].float().cpu(), out[0].float().cpu())
        return hook

    handles = [l.mlp.down_proj.register_forward_hook(make_hook(i))
               for i, l in enumerate(layers)]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    for h in handles:
        h.remove()
    # H[i] is the residual stream AFTER decoder layer i-1 (H[0] = embeddings)
    H = torch.stack([h[0].float().cpu() for h in out.hidden_states])
    return H, store


def super_activations(H, bar):
    """(token, channel, decoder layer, peak) for every massive, persistent
    residual channel. The final hidden state is excluded on purpose: the
    last layer REMOVES the super activation on OLMo-1B (420 -> 7.6)."""
    last = H.shape[0] - 2
    mag = H.abs()
    peak = mag[:last + 1].max(dim=0).values
    out = []
    for t, j in (peak > bar).nonzero().tolist():
        above = [i for i in range(last + 1) if mag[i, t, j] >= PLATEAU * peak[t, j]]
        onset = above[0]
        if onset >= 1 and len(above) / (last - onset + 1) >= MIN_PERSIST:
            out.append((t, j, onset - 1, peak[t, j].item()))
    return out


def contributors(x, y, W, j):
    """Which (k, weight, share) produce y[j]? Paper 3.1: X_ik and W_jk must
    BOTH be outliers and their product must dominate. Take contributors in
    order until EXPLAIN of y[j] is covered — one weight usually, two when a
    pair shares the channel (Llama-13B)."""
    contrib = x * W[j]                      # signed; sums to y[j]
    w_cut = W.abs().flatten().topk(TOP_W).values[-1]
    x_cut = x.abs().topk(TOP_X).values[-1]
    explained, out = 0.0, []
    for k in contrib.abs().argsort(descending=True)[:4].tolist():
        share = contrib[k].item() / y[j].item()
        explained += share
        ok = (abs(share) >= MIN_SHARE
              and abs(x[k]) >= x_cut and abs(W[j, k]) >= w_cut)
        out.append((k, W[j, k].item(), share, ok))
        if abs(explained) >= EXPLAIN:
            break
    return out


def find(model, layers, inputs):
    """Returns [{"layer", "j", "k", "value"}, ...]. Leaves the model as found."""
    found, bar = [], None
    for rnd in range(MAX_ROUNDS):
        H, store = forward_pass(model, layers, inputs)
        # bar is fixed at round 0: relative to the CURRENT largest peak it
        # would re-normalise every round and the loop would never end
        if bar is None:
            bar = MASSIVE_FRAC * H[:-1].abs().max().item()
        sas = super_activations(H, bar)
        print(f"Round {rnd}: {len(sas)} super activation(s) (bar {bar:.1f})")
        if not sas:
            print(f"Round {rnd}: no super activation survives — stopping.")
            break

        fresh = []
        for t, j, L, pk in sas:
            X, Y = store[L]
            W = layers[L].mlp.down_proj.weight.float().cpu()
            for k, w, share, ok in contributors(X[t], Y[t], W, j):
                print(f"  h[t{t},{j}] peak {pk:.1f} -> layer {L}  W[{j},{k}] = "
                      f"{w:+.4f}  share {share:+.3f}{'' if ok else '  (rejected)'}")
                if ok and not any((f["layer"], f["j"], f["k"]) == (L, j, k) for f in found):
                    fresh.append((L, j, k, w))
        if not fresh:
            print(f"Round {rnd}: nothing new passes — stopping.")
            break

        for L, j, k, w in fresh:
            print(f"Round {rnd}: zeroing layer {L} weight[{j},{k}] = {w:.4f}")
            found.append({"layer": L, "j": j, "k": k, "value": w})
            with torch.no_grad():
                layers[L].mlp.down_proj.weight[j, k] = 0.0

    with torch.no_grad():
        for f in found:
            layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]
    return found
