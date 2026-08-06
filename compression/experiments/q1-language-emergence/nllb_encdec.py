"""Q1 for NLLB (encoder-decoder) — decoder-side language emergence.

NLLB doesn't fit HookedModel, so this is a dedicated runner that answers the
same Q1 questions on the decoder stack using HF's `decoder_hidden_states`:

  - **logit lens**: apply the decoder's final norm + lm_head to each decoder
    layer's hidden state at the target positions, track gold-target-token
    mass per layer -> "when does the decoder commit to the target token".
  - **layer contribution (IFR-like)**: ||resid[l] - resid[l-1]||_1, L1-
    normalized per token, averaged -> per-decoder-layer importance profile,
    directly comparable to the decoder-only models' IFR depth profile.

We teacher-force the gold target (standard for measuring where the model
*would* commit), set the source language on the tokenizer, and force the
target-language BOS token, exactly as NLLB is meant to be run.

Output mirrors the decoder-only Q1 layout where it can:
  lens_{pair}.npz   (target_mass: (N, L_dec, K))
  ifr_{pair}.npz    (layer_scores: (L_dec,))
  summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.wmt import load_wmt_pairs
from src.models.nllb import PAIR_TO_NLLB_CODES, load_nllb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("results/nllb-200-3.3b/q1"))
    ap.add_argument("--n-examples", type=int, default=200)
    ap.add_argument("--target-prefix-tokens", type=int, default=8)
    ap.add_argument("--hf-name", default="facebook/nllb-200-3.3B")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[q1-nllb] loading {args.hf_name} ...", flush=True)
    t0 = time.time()
    model, tok = load_nllb(device="cuda", hf_name=args.hf_name)
    n_dec = model.config.decoder_layers
    print(f"[q1-nllb] loaded in {time.time()-t0:.1f}s; decoder_layers={n_dec}", flush=True)

    # decoder final norm + unembed
    dec_norm = model.model.decoder.layer_norm
    W_U = model.lm_head.weight.detach()  # (V, d)

    summary = {"model": "nllb-200-3.3b", "decoder_layers": n_dec, "per_pair": {}}

    for pair in PAIR_TO_NLLB_CODES:
        src_code, tgt_code = PAIR_TO_NLLB_CODES[pair]
        tok.src_lang = src_code
        tok.tgt_lang = tgt_code  # required for text_target= tokenization
        tgt_bos = tok.convert_tokens_to_ids(tgt_code)
        records = list(load_wmt_pairs(pair, n=args.n_examples))
        print(f"[q1-nllb] === {pair} ({src_code}->{tgt_code}), {len(records)} recs ===", flush=True)

        per_layer_mass, per_layer_contrib = [], []
        t0 = time.time()
        for rec in records:
            enc = tok(rec.source, return_tensors="pt").to(model.device)
            # teacher-forced decoder input: [tgt_lang_bos] + gold target tokens.
            # NLLB: tokenize the target with text_target= (as_target_tokenizer is
            # deprecated/broken in transformers 4.57).
            tgt_ids = tok(text_target=rec.target, return_tensors="pt").input_ids[0]
            # NLLB target starts with tgt_lang code; drop the trailing EOS for forcing
            dec_in = torch.cat([torch.tensor([tgt_bos]), tgt_ids[:-1]]).unsqueeze(0).to(model.device)
            gold = tgt_ids[: args.target_prefix_tokens].tolist()
            if not gold:
                continue
            with torch.no_grad():
                out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                            decoder_input_ids=dec_in, output_hidden_states=True)
            # decoder_hidden_states: tuple (n_dec+1) of (B, T_dec, d); [0]=embed
            hs = out.decoder_hidden_states
            last = dec_in.shape[-1] - 1  # read the position predicting the next gold token
            gold_ids = torch.tensor(gold, device=model.device)
            masses, contribs = [], []
            prev = hs[0][0, last]
            for l in range(1, len(hs)):
                r = hs[l][0, last]
                logits = dec_norm(r.unsqueeze(0)).squeeze(0) @ W_U.T.float() if r.dtype==torch.float32 \
                         else (dec_norm(r.unsqueeze(0)).squeeze(0).float() @ W_U.T.float())
                p = logits.softmax(-1)
                masses.append(p.index_select(0, gold_ids).float().mean().item())
                contribs.append(float((r - prev).abs().sum().item()))
                prev = r
            per_layer_mass.append(masses)
            # L1-normalize contributions per example
            c = np.array(contribs); c = c / c.sum() if c.sum() > 0 else c
            per_layer_contrib.append(c)

        mass = np.array(per_layer_mass)            # (N, L_dec)
        contrib = np.array(per_layer_contrib).mean(0)  # (L_dec,)
        np.savez(args.output / f"lens_{pair}.npz", target_mass=mass[:, :, None])
        np.savez(args.output / f"ifr_{pair}.npz", layer_scores=contrib)
        print(f"[q1-nllb] {pair}: {time.time()-t0:.0f}s  final-layer mass={mass.mean(0)[-1]:.4f}  "
              f"top contrib layer=L{int(contrib.argmax())}", flush=True)
        summary["per_pair"][pair] = {
            "n_records": int(mass.shape[0]),
            "final_layer_mass": float(mass.mean(0)[-1]),
            "ifr_top_layer": int(contrib.argmax()),
            "ifr_last_quarter_share": float(contrib[-(n_dec // 4):].sum()),
            "ifr_first_quarter_share": float(contrib[: n_dec // 4].sum()),
        }

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[q1-nllb] wrote summary.json", flush=True)


if __name__ == "__main__":
    main()
