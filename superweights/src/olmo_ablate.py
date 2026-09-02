"""Causal ablation of super-weight candidates in OLMo-1B (scratch version).

"Ablation" = remove one part and see what breaks. Here: set ONE weight to
zero (everything else untouched), measure how much the model got worse,
put the weight back. Repeat for each candidate. Whichever candidate causes
huge damage is a real super weight; whichever causes nothing is a false
positive of the detector.

Damage is measured three ways:
  - perplexity: how "surprised" the model is by a fixed paragraph of normal
    text. Higher = worse. A broken model's perplexity explodes.
  - KL: how much the model's next-token probabilities changed vs. the
    intact model, after a fixed prompt. 0 = no change at all.
  - a short greedy continuation, so we can SEE the damage with our eyes.

Candidates: the four found by src/olmo_sw.py (2026-09-02 run) plus the
paper's claimed second coordinate (Table 2 says layer 1, [1764, 8041] —
our detector said layer 2 instead). This script is the judge.

Result (2026-09-02): only found-1 is a real super weight. See notes.md.

NOTE: superseded by the model-agnostic pair detect_sw.py / ablate_sw.py.
Kept as the simple, readable version.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMo-1B-0724-hf"

# Each candidate is one scalar: layers[layer].mlp.down_proj.weight[j, k]
CANDIDATES = [
    {"name": "found-1 (= Table2 #1)",      "layer": 1,  "j": 1764, "k": 1710},
    {"name": "found-2 (dispute: layer 2)", "layer": 2,  "j": 1764, "k": 8041},
    {"name": "found-3 (orphan-spike FP?)", "layer": 1,  "j": 623,  "k": 1710},
    {"name": "found-4 (last-layer FP?)",   "layer": 15, "j": 1764, "k": 6840},
    {"name": "Table2 #2 (paper: layer 1)", "layer": 1,  "j": 1764, "k": 8041},
]

# Fixed texts: every condition sees the SAME text, so any difference in the
# numbers can only come from the weight we zeroed.
EVAL_TEXT = (
    "The quick brown fox jumps over the lazy dog. Language models predict "
    "the next token given the tokens that came before. When a model is "
    "damaged, its predictions become erratic and its perplexity rises "
    "sharply. The city of Paris is the capital of France, and water boils "
    "at one hundred degrees Celsius at sea level."
)
PROMPT = "The capital of France is"


def get_weight(model, layer, j, k):
    """Read one scalar out of a down_proj weight matrix."""
    return model.model.layers[layer].mlp.down_proj.weight[j, k].item()


def set_weight(model, layer, j, k, value):
    """Write one scalar. no_grad because in-place edits on trainable
    parameters are otherwise refused by autograd."""
    with torch.no_grad():
        model.model.layers[layer].mlp.down_proj.weight[j, k] = value


def perplexity(model, enc):
    """Perplexity of the model on this text. Passing labels=input_ids makes
    the model return its own average next-token loss; exp(loss) = ppl."""
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return out.loss.exp().item()


def next_token_logprobs(model, enc):
    """Log-probabilities for the token that would come after the prompt."""
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]     # last position -> [vocab]
    return F.log_softmax(logits, dim=-1)


def kl_from_baseline(base_logp, abl_logp):
    """KL(base || ablated), in nats. 0 = the distributions are identical."""
    return F.kl_div(abl_logp, base_logp, log_target=True, reduction="sum").item()


def greedy_continuation(model, tokenizer, enc, n_tokens=15):
    """Deterministic continuation (no sampling) — the eyeball test."""
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(PROMPT):].strip()


def verdict(ppl_ratio, kl):
    """Eyeball label from made-up cutoffs (display only, not science)."""
    if ppl_ratio > 10 or kl > 1:
        return "CATASTROPHIC"
    if ppl_ratio > 1.5 or kl > 0.1:
        return "damaged"
    return "no effect"


def measure(model, tokenizer, eval_enc, prompt_enc, base_logp=None):
    ppl = perplexity(model, eval_enc)
    logp = next_token_logprobs(model, prompt_enc)
    kl = kl_from_baseline(base_logp, logp) if base_logp is not None else 0.0
    gen = greedy_continuation(model, tokenizer, prompt_enc)
    return ppl, kl, logp, gen


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    eval_enc = tokenizer(EVAL_TEXT, return_tensors="pt", return_token_type_ids=False)
    prompt_enc = tokenizer(PROMPT, return_tensors="pt", return_token_type_ids=False)

    rows = []          # (name, weight_str, ppl, kl, gen) — printed at the end

    # ---- baseline: the intact model, measured first ----
    base_ppl, _, base_logp, base_gen = measure(model, tokenizer, eval_enc, prompt_enc)
    rows.append(("baseline (intact)", "—", base_ppl, None, base_gen))

    # ---- one candidate at a time: zero -> measure -> restore ----
    for c in CANDIDATES:
        L, j, k = c["layer"], c["j"], c["k"]
        original = get_weight(model, L, j, k)
        set_weight(model, L, j, k, 0.0)
        ppl, kl, _, gen = measure(model, tokenizer, eval_enc, prompt_enc, base_logp)
        set_weight(model, L, j, k, original)
        assert get_weight(model, L, j, k) == original   # surgery undone?
        rows.append((c["name"], f"{original:.4f}", ppl, kl, gen))

    # ---- the two big finds zeroed together (paper's "Prune SW" analog) ----
    combo = [CANDIDATES[0], CANDIDATES[1]]
    originals = [(c, get_weight(model, c["layer"], c["j"], c["k"])) for c in combo]
    for c, _ in originals:
        set_weight(model, c["layer"], c["j"], c["k"], 0.0)
    ppl, kl, _, gen = measure(model, tokenizer, eval_enc, prompt_enc, base_logp)
    for c, v in originals:
        set_weight(model, c["layer"], c["j"], c["k"], v)
    rows.append(("both zeroed (paper PruneSW)", "—", ppl, kl, gen))

    # ---- report: numbers table first, continuations below it ----
    print()
    print(f"{'condition':<28} {'weight':>8} {'ppl':>9} {'ppl x':>7} {'KL':>7}  verdict")
    print("-" * 76)
    for name, w, ppl, kl, _ in rows:
        ratio = ppl / base_ppl
        ratio_s = f"x{ratio:.1f}" if kl is not None else "—"
        kl_s = f"{kl:.3f}" if kl is not None else "—"
        verd = verdict(ratio, kl) if kl is not None else "—"
        print(f"{name:<28} {w:>8} {ppl:>9.2f} {ratio_s:>7} {kl_s:>7}  {verd}")

    print(f"\ncontinuations of {PROMPT!r} (greedy, 15 tokens, truncated):")
    for name, _, _, _, gen in rows:
        short = gen if len(gen) <= 46 else gen[:43] + "..."
        print(f"  {name:<28} | {short}")


if __name__ == "__main__":
    main()
