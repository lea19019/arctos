"""Detector v1 — the brain from the original detect_sw.py (and olmo_sw.py).

Reads spikes off each layer's mlp.down_proj input and output:

    Y[i, j] ~= X[i, k] * W[j, k]
    X = down_proj input   -> its biggest |value| gives k
    Y = down_proj output  -> its biggest |value| gives j
    the candidate is W[j, k] of that layer

Three home-made checks decide whether a layer's candidate is real (the
paper has no criterion at all), then peel: zero the loudest survivor, run
again, until no layer survives.
"""

import statistics

import torch

DOMINANCE_BAND = (0.8, 1.2)   # X*W must explain Y within +-20%
TOWERING_FACTOR = 10          # spike must be >10x the median layer max
MAX_ROUNDS = 8


def make_hook(layer_idx, records):
    def hook(module, inp, out):
        X, Y, W = inp[0], out, module.weight
        token_x, k = divmod(X.abs().argmax().item(), X.shape[-1])
        token_y, j = divmod(Y.abs().argmax().item(), Y.shape[-1])
        records.append({
            "layer": layer_idx,
            "max_x": X.abs().max().item(), "token_x": token_x, "k": k,
            "max_y": Y.abs().max().item(), "token_y": token_y, "j": j,
            "sw_value": W[j, k].item(),
            "x_spike": X[0, token_x, k].item(),
            "y_spike": Y[0, token_y, j].item(),
        })
    return hook


def detection_pass(model, layers, inputs):
    """One forward pass with a hook on every down_proj; one record per layer."""
    records = []
    handles = [l.mlp.down_proj.register_forward_hook(make_hook(i, records))
               for i, l in enumerate(layers)]
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return records


def dominance(r):
    return (r["x_spike"] * r["sw_value"]) / r["y_spike"]


def passes(r, median_max_y):
    tokens_match = r["token_x"] == r["token_y"]
    dominant = DOMINANCE_BAND[0] < dominance(r) < DOMINANCE_BAND[1]
    towering = r["max_y"] > TOWERING_FACTOR * median_max_y
    return tokens_match and dominant and towering


def find(model, layers, inputs):
    """Returns [{"layer", "j", "k", "value"}, ...]. Leaves the model as found."""
    found = []
    for rnd in range(MAX_ROUNDS):
        records = detection_pass(model, layers, inputs)
        median_max_y = statistics.median(r["max_y"] for r in records)
        survivors = [r for r in records if passes(r, median_max_y)]

        print(f"Round {rnd}: {len(survivors)} of {len(records)} layers pass "
              f"(median max|Y| = {median_max_y:.2f})")
        for r in survivors:
            print(f"  layer {r['layer']:2d}  W[{r['j']},{r['k']}]  "
                  f"max|Y|={r['max_y']:.1f}  dominance={dominance(r):.3f}")
        if not survivors:
            print(f"Round {rnd}: no candidate passes — stopping.")
            break

        c = max(survivors, key=lambda r: r["max_y"])
        print(f"Round {rnd}: zeroing layer {c['layer']} "
              f"W[{c['j']},{c['k']}] = {c['sw_value']:.4f}")
        found.append({"layer": c["layer"], "j": c["j"], "k": c["k"],
                      "value": c["sw_value"]})
        with torch.no_grad():
            layers[c["layer"]].mlp.down_proj.weight[c["j"], c["k"]] = 0.0

    with torch.no_grad():
        for f in found:
            layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]
    return found
