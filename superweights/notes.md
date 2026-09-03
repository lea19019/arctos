# Notes on Super Weights

Three questions I need to have clear answers for:

1- What is a super weight?
A scalar that has a deep impact in the performance of the model. If you remove or alter that scalar it breaks the model's ability to produce text.

2- How do we measure super weights?
We follow the massive activatoins which appear right after a a super weight.

3- How do we find super weights?
Through the massive activations

What is an activation outlier? What is an activation at all? This questions comes from the abstract of Yu-2024
What do they mean with spikes?

What is a softmax?

Why does not SA are the only key? What is the architecture and how are SW connected to the rest of the model?
If you remove the model seems like the semantics are lost, how does this affect the cross-lingual behavior?
Amplifying the weight can improve the model accuracy, what is 

---

## 2026-09-02 — Replicated Yu et al. detection + ablation on OLMo-1B

**What we did.** Wrote the spike detector from the paper alone (`src/olmo_sw.py`):
hooks on every layer's `mlp.down_proj`, one forward pass, find the biggest
|input| and |output| values per layer, read the weight coordinate off the two
spikes. Then peel: zero the found weight, run again, repeat. Because the raw
recipe always returns *something*, we added three home-made checks before
accepting a candidate (same token for both spikes; X·W explains Y within 20%;
spike >10× the median layer). **These thresholds are ours, not the paper's —
the paper has no criterion at all.** Then we judged every candidate by
ablation (`src/olmo_ablate.py`): zero that one scalar, measure damage, restore.

**Detection found 4 candidates. Ablation says only 1 is real.**

| candidate | coordinate | weight | ppl (base 13.66) | KL | verdict |
|---|---|---|---|---|---|
| found-1 (= paper's #1) | L1 [1764,1710] | 0.6323 | **1703.59** | **6.73** | real SW |
| found-2 | L2 [1764,8041] | −0.5924 | 13.95 | 0.002 | nothing |
| found-3 | L1 [623,1710] | 0.0586 | 14.05 | 0.005 | nothing |
| found-4 | L15 [1764,6840] | −0.7258 | 14.02 | 0.000 | nothing |
| paper's #2 | L1 [1764,8041] | **0.0018** | 13.66 | 0.000 | nothing |

With the real SW zeroed, greedy output collapses into stopwords ("We. We.
We.", "and, and, and") — exactly the paper's Figure 5 mechanism, seen live.

**The main finding: the paper's Table 2 lists a second super weight for
OLMo-1B at layer 1 [1764, 8041]. In our copy of the model there is
essentially no weight there (0.0018) and zeroing it does nothing.** The
same coordinates at layer 2 hold a big weight (−0.59) whose spike looks
super in detection — but zeroing it also does nothing. So OLMo-1B appears
to have ONE super weight, not two. Caveat: measured on one paragraph + one
prompt, one Hub revision, n=1 — enough for our notes, not enough to claim
publicly. The hardened re-run (several texts, pinned revision) is scripted
in `src/detect_sw.py` + `src/ablate_sw.py`.

**Two false-positive species we met (worth remembering, they motivate Phase 0):**

1. *Orphaned input spike* (found-3): zeroing a SW kills its output spike but
   NOT its input spike (the input is made upstream). The giant input then
   makes any ordinary weight in its column look "dominant" — 415 × 0.0586
   explains its tiny output perfectly. Dominance can be exactly 1.0 and
   mean nothing.
2. *Last-layer echo* (found-4): the final layer has big raw spikes because
   it writes almost directly into the logits — same false positive the old
   q6 detector hit on Aya L31. Size alone would have picked it first.

**Lesson in one line: spike size — even verified spike size — is not
importance. Only ablation separates them. A calibrated detector needs the
causal check built in.**

**Bugs that bit us on the way** (for the future test suite): argmax without
`.abs()` misses negative spikes; a variable named `max_activation` that held
a *position*, not a value, hid the layer-15 false positive for days;
pick-then-verify instead of verify-then-pick made the loop stop at the very
first (fake) candidate.

**Next:** run detect+ablate on a second model (Llama-2-7B or Mistral-7B,
also in Table 2), then open the Phase 0 spec (`/new-experiment`) — the
calibrated null that replaces our made-up thresholds.

## 2026-09-02 (later) — Tier A #1: the detector against all 9 Table 2 models

Ran `detect_sw.py` + `ablate_sw.py` on every model in Yu et al. Table 2
(9 models, 21 coordinates) on one GPU each: SLURM array `slurm/sweep.sh`,
~15 min total. Table 2 was completed first — `TABLE2` was missing Llama-30B
(3 coords) and Phi-3 (6 coords) while both were already in `MODELS`, so those
two models would have reported "0 of 0 confirmed" and looked fine.

Changes made before the run: `--dtype` defaulting to the checkpoint's own
`torch_dtype` (Llama/Mistral are fp16, **OLMo-1B and OLMo-7B are fp32**, only
Mistral/Phi-3 are bf16 — the old hardcoded bf16 was re-rounding every one of
them); `MAX_ROUNDS` 8 → 15 (Phi-3 has 6 SWs); `git_sha` into every results
JSON (`src/provenance.py`); per-layer records with per-criterion pass/fail in
the detection log. Also reclaimed 111 GB of duplicate `pytorch_model*.bin`
from the HF cache and added `*.bin` to `prefetch_models.py`'s ignore list.

### The paper's coordinates are real — this is the strongest result here

`src/coord_check.py` reads each Table 2 coordinate straight out of the
safetensors shard and ranks |W[j,k]| within its own `down_proj` matrix.
**20 of 21 coordinates rank 1–6 out of 16–119 million weights.** The
transposed reading `W[k,j]` was checked wherever both indices were in bounds
and always lands in the millions, so the row/column convention is
unambiguous too.

The 21st is the OLMo-1B entry from the earlier session, and it is now
explained. **Table 2's second OLMo-1B row has the wrong layer number:**

    L1[1764,8041] = +0.0018   rank 13,035,937 of 16,777,216
    L2[1764,8041] = -0.5924   rank           1 of 16,777,216
    L3[1764,8041] = +0.0008   rank 15,109,691

Also ruled out, since it was the standing hypothesis: **we are running the
weights they ran.** `model-*.safetensors` on `allenai/OLMo-1B-0724-hf` were
committed 2024-06-21 and never touched; the repo's newest commit of any kind
is 2024-08-05 (a model-card edit). Yu et al. went up 2024-11-07.

### Detection: the recipe replicates, our hardening does not

11 of 21 coordinates returned. But at round 0 — one forward pass on the
intact model, the paper's own setting — **20 of 21 coordinates sit at a layer
whose argmax is either that exact coordinate or a sibling Table 2 coordinate
in the same layer.** Localization is not the problem. Causes of the 10 misses:

| n | cause |
|---|---|
| 5 | **one argmax per layer** — the layer holds ≥2 Table 2 coordinates and a single `argmax` over Y can only return one (Llama-13B, Llama-30B, Phi-3 ×2, OLMo-1B) |
| 2 | **our dominance band** rejected an exact hit (Llama-13B 0.593, Llama-2-7B 0.758, band is 0.8–1.2) |
| 2 | **peel order** — passed all three checks but the loop takes only the loudest survivor per round (OLMo-7B L7, L24) |
| 1 | argmax landed elsewhere in the layer (Llama-30B L10, dominance 0.282) |

All four causes are **ours, not the paper's** — Yu et al. have no acceptance
criteria at all; they read spikes off a plot. This is the concrete argument
for Phase 0's calibrated criterion, arrived at from our own data.

**The dominance criterion is invalid by construction when two super weights
feed one output channel.** Llama-13B L2 has both Table 2 coordinates in row
2231: 570.5 × (−1.8223) = −1039.6 against an observed y_spike of −1754.0,
i.e. 0.593. The residual −714.4 is what the second weight (w = +1.8066)
would contribute from an input of ≈ −395. ⚠️ *That last step is arithmetic
inference — X[6939] is not logged. Verify before citing.*

**Super activations propagate, and the peel loop sees it.** OLMo-7B L24's
spike is 277.8 (222× the layer median) at rounds 0–1, then **collapses to
3.0 at round 2**, right after the L1 super weight is zeroed. L24 does not
create that spike; it receives it through the skip connections — the paper's
own Figure 2/4 mechanism, observed from the other side. Consistent with
ablation: L7 ×2.1, L24 ×1.2, i.e. little to nothing.

### Ablation: super weights exist, and 4 of them are ours

Zeroing one scalar, against the intact model, mean ppl over 4 eval texts:

| model | coordinate | weight | ppl | KL |
|---|---|---|---|---|
| Mistral-7B | L1[2070,7310] | −0.2734 | 5.56 → **4081** (×734) | 6.82 |
| OLMo-1B | L1[1764,1710] | +0.6323 | 8.54 → **709** (×83) | 5.10 |
| OLMo-7B | L6[269,9562] | −0.7771 | 6.82 → **401** (×59) | 2.36 |
| Llama-7B | L2[3968,7003] | −1.9268 | 5.39 → **32** (×6) | 1.06 |

Mistral collapses to `"the main. without without.  . . . ."`, OLMo-1B to
`"We. We. We."` — the Figure 5 mechanism.

**The other 17 coordinates are the control group, and that is what makes this
convincing.** They are *also* rank 1–6 magnitude outliers in the same
matrices, and zeroing them moves perplexity ×1.0–×1.4. Same matrices,
comparably extreme weights, opposite outcomes: `magnitude ≠ importance`,
reproduced from scratch. Mistral makes it sharpest — its super weight is the
*smallest* of the four (−0.27) and does the *most* damage.

⚠️ **OLMo-7B L6[269,9562] is not in Table 2.** It is our detector's find and
the strongest effect in that model, while all four of the paper's OLMo-7B
coordinates came back inert (×1.2, ×1.0, ×2.1, ×1.2). Unexplained.

### What this does NOT establish

- **Not a refutation of the paper's causal claim.** Our damage metric is
  perplexity on 4 paragraphs + KL on 3 prompts; theirs is zero-shot accuracy
  across benchmarks. 3-of-21 catastrophic *under our proxy* is not 3-of-21
  under theirs. Untested.
- **n=1 on the measurement** — one revision, one fixed text set, greedy so
  nothing to average. Coverage is all 9 models / all 21 coordinates.
- **"No super weight found" is still unstatable.** No null. Phase 0's job.
- The CATASTROPHIC/damaged/no-effect labels are made-up cutoffs. They happen
  not to matter here — the gap between the real ones and everything else is
  ~500× in ppl ratio — but they are not a criterion.

**Next (unstarted, needs a decision):** the four miss causes above are all
fixable — top-*n* per layer instead of argmax, drop or widen the dominance
band, peel all survivors per round. Whether to fix them here or fold them
into the Phase 0 calibrated detector is the open question.
