"""Iterative super-weight detection on OLMo-1B, replicating Yu et al. 2024
(arXiv:2411.07191), section 3.1.

The idea: a super weight (SW) is one scalar in an early layer's
mlp.down_proj that turns a moderately large input activation into a huge
output activation (the "super activation"). Both spikes are observable, and
together they name the weight between them:

    Y[i, j] ~= X[i, k] * W[j, k]

    X = down_proj input  [batch, tokens, 8192]  -> its spike gives k
    Y = down_proj output [batch, tokens, 2048]  -> its spike gives j
    W = down_proj weight [2048, 8192]           -> the SW is W[j, k]

One detection pass finds only the LOUDEST super weight, so we peel greedily:
find one, zero it, run again, until no candidate survives verification.
"""

import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")

# One short prompt is enough: the super activation is a property of the
# weights, not the input — it fires on essentially any text.
inputs = tokenizer(["Language modeling is "], return_tensors="pt",
                   return_token_type_ids=False)


def make_hook(layer_idx, records):
    """Factory: build one hook that knows its own layer index.

    Each call creates a fresh scope, so each hook keeps its own layer_idx
    (a single function defined once would see only one shared value).
    """
    def hook(module, inp, out):
        # PyTorch calls this every time down_proj runs.
        # inp is a tuple of the module's inputs; inp[0] is our X.
        W = module.weight
        X = inp[0]
        Y = out

        # Where is the biggest |value| in each tensor?
        # argmax() returns a position in the FLATTENED tensor;
        # divmod by the row length converts it back to (token, channel).
        token_x, k = divmod(X.abs().argmax().item(), X.shape[-1])
        token_y, j = divmod(Y.abs().argmax().item(), Y.shape[-1])

        max_x = X.abs().max().item()      # spike sizes (sign stripped)
        max_y = Y.abs().max().item()
        x_spike = X[0, token_x, k].item()  # same values WITH their sign,
        y_spike = Y[0, token_y, j].item()  # needed for the dominance ratio
        sw_value = W[j, k].item()          # the weight the spikes point at

        records.append({
            "layer": layer_idx,
            "max_x": max_x, "token_x": token_x, "k": k,
            "max_y": max_y, "token_y": token_y, "j": j,
            "sw_value": sw_value,
            "x_spike": x_spike, "y_spike": y_spike,
        })
    return hook


def detection_pass(model, inputs):
    """One round: register hooks, run one forward pass, collect, clean up.

    Returns one record per layer (16 for OLMo-1B).
    """
    records = []
    handles = [
        layer.mlp.down_proj.register_forward_hook(make_hook(i, records))
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():          # measuring only — no gradients needed
        model(**inputs)
    for h in handles:              # always detach hooks when done, or they
        h.remove()                 # fire again on every later forward pass
    return records


found = []          # [{"layer": L, "j": j, "k": k, "value": v}, ...]
MAX_ROUNDS = 5      # safety cap so a bug cannot loop forever

for rnd in range(MAX_ROUNDS):
    records = detection_pass(olmo, inputs)

    # ---- brain: verify EVERY layer's candidate, THEN pick among survivors.
    # (Picking the biggest spike first and verifying after is a trap: the
    # last layer has a huge raw spike that is NOT a super weight, and it
    # would win the pick and wrongly stop the search.)
    #
    # The three checks below use made-up thresholds. The paper gives no
    # criterion at all (spikes are read off a plot by eye). Replacing these
    # numbers with a calibrated null distribution is Phase 0 of this track.

    # "Typical" spike size for this model right now = median across layers.
    median_max_y = statistics.median(r["max_y"] for r in records)

    def dominance(r):
        # If the SW story is true, one product should explain the whole
        # output spike: X[i,k] * W[j,k] ~= Y[i,j], so this ratio ~= 1.
        return (r["x_spike"] * r["sw_value"]) / r["y_spike"]

    def passes(r):
        # 1) Input spike and output spike must belong to the same token —
        #    one dot product happens within one token's vectors.
        tokens_match = r["token_x"] == r["token_y"]
        # 2) The single product must explain the output spike (within 20%).
        dominant = 0.8 < dominance(r) < 1.2
        # 3) The spike must be huge compared to a typical layer's biggest
        #    output — otherwise tiny spikes pass check 2 by luck.
        towering = r["max_y"] > 10 * median_max_y
        return tokens_match and dominant and towering

    survivors = [r for r in records if passes(r)]

    print(f"Round {rnd}: {len(survivors)} of {len(records)} layers pass verification "
          f"(median max|Y| = {median_max_y:.1f})")
    for r in survivors:
        print(f"  layer {r['layer']:2d}  W[{r['j']},{r['k']}]  "
              f"max|Y|={r['max_y']:.1f}  dominance={dominance(r):.3f}")

    # No survivors = nothing left to find. This is the stopping criterion —
    # and the paper has no equivalent of it ("until greatly suppressed").
    if not survivors:
        print(f"Round {rnd}: no candidate passes — stopping.")
        break

    # Greedy pick: the loudest VERIFIED spike wins this round.
    candidate = max(survivors, key=lambda r: r["max_y"])

    # ---- surgery: zero that one scalar, remember it so we can restore.
    L, j, k = candidate["layer"], candidate["j"], candidate["k"]
    print(f"Round {rnd}: zeroing layer {L} weight[{j},{k}] = {candidate['sw_value']:.4f}")
    found.append({"layer": L, "j": j, "k": k, "value": candidate["sw_value"]})
    with torch.no_grad():          # in-place edits on parameters need this
        olmo.model.layers[L].mlp.down_proj.weight[j, k] = 0.0

print("\nFound super weights:", found)

# ---- restore the patient: put every zeroed weight back.
with torch.no_grad():
    for f in found:
        olmo.model.layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]
