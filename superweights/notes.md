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

