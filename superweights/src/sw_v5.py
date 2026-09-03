"""Super-weight detection by residual-stream persistence (Yu et al. 2024,
Figure 4). Same shape as olmo_sw.py; the brain is in the loop below.

Why a different signal than olmo_sw.py: that script reads spikes off each
layer's down_proj OUTPUT. But the paper's Figure 4 is about the RESIDUAL
STREAM — the super activation appears right after the super weight's layer
and then persists at the same magnitude to the end of the model. Measured
on OLMo-1B the residual max goes 0.2, 5.0, 267.7 (layer 2), ~420 for 13
layers, then 7.6 as the last layer removes it. That is the fingerprint.

It also explains the false positive every down_proj-based version hit:
OLMo-1B's LAST layer emits the biggest down_proj spike in the model (419.5,
above the real super weight's 266.9) because it writes almost straight into
the logits — and it creates nothing that persists. Persistence tells them
apart; magnitude cannot.

So the search is inverted:
    1. find residual channels that are massive AND persist across depth
    2. the layer where each first appears is where its super weight lives
    3. only there, decompose Y[t, j] = sum_k X[t, k] * W[j, k] to find k
    4. zero, repeat until no super activation survives (the paper's rule)
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMo-1B-0724-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else None    # JSON for ablate_sw.py

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto" if device == "cuda" else torch.float32).to(device) # type: ignore
tokenizer = AutoTokenizer.from_pretrained(MODEL)
layers = model.model.layers

# One short prompt is enough: the super activation is a property of the
# weights, not the input.
inputs = tokenizer(["Language modeling is "], return_tensors="pt",
                   return_token_type_ids=False).to(device)


def forward_pass(model, inputs):
    """One pass. Returns the residual stream at every layer AND every
    layer's down_proj input/output, since we need both."""
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


found = []
MAX_ROUNDS = 10
bar = None          # "massive" threshold, fixed at round 0 (see below)

for rnd in range(MAX_ROUNDS):
    H, store = forward_pass(model, inputs)
    n_layers = H.shape[0] - 1
    mag = H.abs()

    # ---- brain, step 1: which residual channels are super activations?
    #
    # The final hidden state is excluded: on OLMo-1B the last layer REMOVES
    # the super activation (420 -> 7.6), so including it would reject the
    # very thing we are looking for.
    last = n_layers - 1
    peak = mag[:last + 1].max(dim=0).values          # per (token, channel)

    # "Massive" = at least 10% of the largest peak in the model. Fixed after
    # round 0 on purpose: a bar relative to the CURRENT largest peak would
    # re-normalise every round, so after peeling the real super weight the
    # next activation becomes "massive" by construction and the loop never
    # ends. The paper's rule is suppression relative to the ORIGINAL.
    if bar is None:
        bar = 0.10 * peak.max().item()

    super_acts = []
    for t, j in (peak > bar).nonzero().tolist():
        plateau = 0.5 * peak[t, j].item()
        above = [i for i in range(last + 1) if mag[i, t, j] >= plateau]
        onset = above[0]
        # must hold the plateau for at least half the remaining depth —
        # that is what separates a persisting activation from a one-layer
        # spike (the last-layer false positive persists for 0 layers)
        if len(above) / (last - onset + 1) >= 0.5 and onset >= 1:
            super_acts.append((t, j, onset - 1, peak[t, j].item()))

    print(f"Round {rnd}: {len(super_acts)} super activation(s) "
          f"(bar {bar:.1f})")
    if not super_acts:
        print(f"Round {rnd}: no super activation survives — stopping.")
        break

    # ---- brain, step 2: at the onset layer, which weight(s) made it?
    fresh = []
    for t, j, L, pk in super_acts:
        X, Y = store[L]
        x, y = X[t], Y[t]
        W = layers[L].mlp.down_proj.weight.float().cpu()

        # every k's signed contribution to y[j]; they sum to y[j] exactly
        contrib = x * W[j]
        order = contrib.abs().argsort(descending=True)

        # The paper: "if X_ik and W_jk are BOTH outliers ... Y_ij will be
        # dominated by their product". So a contributor must (a) carry a
        # real share of y[j], (b) be an outlier on the X side, (c) be an
        # outlier on the W side. Take contributors in order until 80% of
        # y[j] is explained — one weight in the usual case, two when a
        # pair shares the channel (Llama-13B).
        w_cut = W.abs().flatten().topk(1000).values[-1]
        x_cut = x.abs().topk(10).values[-1]
        explained = 0.0
        for k in order[:4].tolist():
            share = contrib[k].item() / y[j].item()
            explained += share
            ok = (abs(share) >= 0.35
                  and abs(x[k]) >= x_cut and abs(W[j, k]) >= w_cut)
            print(f"  h[t{t},{j}] peak {pk:.1f} -> layer {L}  "
                  f"W[{j},{k}] = {W[j, k].item():+.4f}  share {share:+.3f}"
                  f"{'' if ok else '  (rejected)'}")
            if ok and (L, j, k) not in [(f['layer'], f['j'], f['k']) for f in found]:
                fresh.append((L, j, k, W[j, k].item()))
            if abs(explained) >= 0.8:
                break

    if not fresh:
        print(f"Round {rnd}: nothing new passes — stopping.")
        break

    # ---- surgery: zero every fresh candidate, remember it for restore.
    for L, j, k, v in fresh:
        print(f"Round {rnd}: zeroing layer {L} weight[{j},{k}] = {v:.4f}")
        found.append({"layer": L, "j": j, "k": k, "value": v})
        with torch.no_grad():
            layers[L].mlp.down_proj.weight[j, k] = 0.0

print("\nFound super weights:", found)

# ---- restore the patient: put every zeroed weight back.
with torch.no_grad():
    for f in found:
        layers[f["layer"]].mlp.down_proj.weight[f["j"], f["k"]] = f["value"]

if OUT:
    json.dump({"model": MODEL, "found": found}, open(OUT, "w"), indent=2)
    print("written to", OUT)
