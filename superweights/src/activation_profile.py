"""Does the super activation persist down the residual stream? (Yu et al. Fig 4)

Their Figure 4 plots max LAYER output across depth -- the residual stream --
and reports the super activation appearing after the super weight's layer and
then holding "exactly the same magnitude" to the end. Every detector we have
written measures down_proj output instead, which is why the last layer (whose
down_proj writes almost straight into the logits) outranks the real super
weight on OLMo-1B.

This dumps the residual-stream profile so the persistence criterion can be
designed against measured behaviour rather than assumed.

    uv run src/activation_profile.py --model allenai/OLMo-1B-0724-hf
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Language modeling is ")
    ap.add_argument("--top", type=int, default=3, help="channels to trace")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok([args.prompt], return_tensors="pt",
              return_token_type_ids=False).to(device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    # (n_layers+1, T, D); index 0 is the embedding output
    H = torch.stack([h[0].float().cpu() for h in out.hidden_states])
    n_layers, T, D = H.shape[0] - 1, H.shape[1], H.shape[2]
    print(f"{args.model}: {n_layers} layers, {T} tokens, d_model={D}\n")

    # the channels that dominate the residual stream at the LAST layer
    mag = H.abs()
    flat = mag[-1].flatten().topk(args.top).indices
    picks = [(int(i // D), int(i % D)) for i in flat]
    print("tracing the top residual channels at the final layer:", picks)
    print(f"\n{'layer':>5} {'max|h| any':>11}  " +
          "  ".join(f"h[t{t},{j}]".rjust(13) for t, j in picks))
    for i in range(n_layers + 1):
        row = "  ".join(f"{H[i, t, j].item():13.1f}" for t, j in picks)
        print(f"{i:>5} {mag[i].max().item():11.1f}  {row}")

    print("\nonset = first layer where the channel exceeds 10x its value at layer 0")
    for t, j in picks:
        base = max(abs(H[0, t, j].item()), 1e-6)
        onset = next((i for i in range(n_layers + 1)
                      if abs(H[i, t, j].item()) > 10 * base), None)
        persists = sum(1 for i in range(n_layers + 1)
                       if abs(H[i, t, j].item()) > 0.5 * abs(H[-1, t, j].item()))
        print(f"  h[t{t},{j}]: onset at layer {onset}, "
              f"above half final magnitude in {persists}/{n_layers + 1} layers")


if __name__ == "__main__":
    main()
