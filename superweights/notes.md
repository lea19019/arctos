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

> ⚠️ **SUPERSEDED IN PART, same day — read this first.** Every perplexity
> ratio below was measured on four hand-written paragraphs. That metric is
> 20-70x under-powered against the paper's own (Wiki-2 / C4), and three of
> this entry's conclusions do not survive re-measuring on wikitext-2. The
> corrected numbers are in the next entry; the detection findings (Table 2
> coordinate ranks, the OLMo-1B layer typo, the four miss causes) are
> unaffected, because detection never touches the eval corpus.

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

---

## 2026-09-02 (evening) — the eval corpus was the confound, not the detector

**Correction to the entry above.** Its ablation numbers used four
hand-written paragraphs. Yu et al. Table 1 reports Llama-7B perplexity
7.08 → 763.65 (C4) and 5.67 → 1211.11 (Wiki-2) for the same coordinate we
measured at ×6. Re-scoring the **identical candidate set** on
wikitext-2-raw-v1 test (32 × 2048-token windows, `exp(mean loss)`) changes
the conclusions. Detection was untouched — it is one forward pass on one
prompt and never sees the eval corpus — so this isolates the metric.

| model | coordinate | 4 paragraphs | wikitext-2 |
|---|---|---|---|
| Llama-7B | L2[3968,7003] | ×6 | **×181** |
| Llama-30B | L3[5633,12817] | ×2.4 | **×16** |
| OLMo-7B | L1[269,7467] | ×1.2 | **×79** |
| Mistral-7B | L1[2070,7310] | ×745 | ×1430 |
| OLMo-1B | L1[1764,1710] | ×83 | ×3667 |

**Three things in the previous entry are wrong:**

1. **"3 of 21 Table 2 coordinates reproduce causally" → 5 of 21.** Llama-30B
   L3[5633,12817] and OLMo-7B L1[269,7467] were called "damaged" and "no
   effect"; on Wiki-2 they are ×16 and ×79. The OLMo-7B swing is ×69.
2. **"OLMo-7B L6[269,9562] is catastrophic and is not in Table 2" — retracted.**
   ×59 on the paragraphs, **×4** on wikitext-2. It was an artifact of the
   four texts. The paper is not missing a super weight there.
3. **Llama-7B is a replication, not a discrepancy.** Paper: Wiki-2 5.67 →
   1211.11 (×214). Ours: 6.12 → 1108 (×181), same coordinate, same corpus.

**What survives unchanged.** Everything about *detection*: the 20-of-21
rank-1-6 result from `coord_check.py`, the OLMo-1B layer typo
(L1[1764,8041] rank 13,035,937 vs L2[1764,8041] rank 1), the ruling-out of
the "different checkpoint" hypothesis, and the four miss causes.

**What is now a sharper puzzle.** 16 of 21 coordinates are still inert on
the paper's own corpus, and they cluster: **all six Phi-3 coordinates are
×1.0**, both Llama-2 models are flat, both Llama-13B coordinates are flat.
That is no longer explainable as metric weakness. Coverage: 9 of 9 models,
21 of 21 coordinates, n=1 revision, greedy/deterministic.

⚠️ **Standing lesson for this track:** a damage metric needs its own
sanity check against a published number before it is used to judge
anything. Ours disagreed with Yu et al.'s Llama-7B figure by 30× and we
read that as the paper being wrong for most of a day.

### Detector generations, and what each is for

| version | file | measures | Table 2 recall | causally real |
|---|---|---|---|---|
| v0 | `olmo_sw.py` | down_proj argmax, OLMo-1B only | — | 1 |
| v1 | `detect_sw_v1.py` | + model-agnostic, 3 thresholds | 11/21 | 4 (para) / 5 (wiki) |
| v2 | `detect_sw_v2.py` | + top-j, contribution prefix, peel-all | 20/21 | 4, with 1,945 extras |
| v3 | `detect_sw_v3.py` | + both-outliers, suppression stop, max-sw | 18/21 | 5, with 124 extras |
| v5 | `detect_sw.py` | **residual-stream persistence** (Fig 4) | 14/21 | 5, with **5** extras |

Scored on the same corpus (wikitext-2), so the columns are comparable:

| detector | recovered | extra candidates | causally real |
|---|---|---|---|
| v1 | 11/21 | 10 | 5 |
| v3 | 18/21 | 124 | 5 |
| **v5** | 14/21 | **5** | 5 |

**v3 answered its own question as a negative:** it recovers seven more Table 2
coordinates than v1 at 12x the false-positive cost and finds *no additional
real super weight*. Recall against Table 2 was never the bottleneck, because
most of Table 2 is causally inert on the paper's own corpus. v5 returns 19
candidates total across nine models and still catches all five real ones; it
also drops the OLMo-7B L6 artifact unprompted.

On OLMo-1B v5 returns exactly two candidates -- Table 2's two entries with the
layer typo corrected -- and stops on the paper's own criterion ("no super
activation survives"), not on a round cap or a guard.

"v4" is not a detector: it is v1's candidates re-scored on wikitext-2
(`slurm/reablate.sh`), i.e. the control that produced this entry.

**Why v5 changes tensor.** v0-v3 all read spikes off `mlp.down_proj`
output. The paper's Figure 4 plots *layer* output — the residual stream —
and claims the super activation persists at constant magnitude from the
super weight's layer to the end. Measured on OLMo-1B
(`activation_profile.py`):

    layer  0:   0.2      layer 3: 309.5
    layer  1:   5.0      ...       ~420  (constant, 13 layers)
    layer  2: 267.7 <-- onset       layer 15: 419.4
                                    layer 16:   7.6 <-- removed by last layer

This finally explains the false positive every version hit: OLMo-1B's
**last** layer emits the largest down_proj spike in the model (419.5 >
the real super weight's 266.9) because it writes almost straight into the
logits, and it creates nothing that persists. No magnitude rule separates
those two; persistence does.

**Bug worth remembering:** v3's `--max-sw` guard `break`ed before recording,
so a plausibility check meant to catch over-generation instead returned
zero candidates — and the probe passed it, because the assertion tested
only `n <= max_sw` and never `n >= 1`. Both fixed. Two detector generations
passed a probe they should have failed.

### ⚠️ Where Tier A ends

Five thresholds were adjusted today, each justified *after* seeing which
candidates were wrong. `--min-share 0.35` sits where it does partly because
we had measured Llama-13B's real shares (0.593 / 0.407); had the true value
been 0.30 somewhere we would have missed it and never known. That is fitting
to a published answer key by a slower route, and it is legitimate **only**
while the ground truth is public — which is exactly Tier A's premise
(`prior_experiments_and_ideas.md` §4).

It stops being legitimate the moment this detector is pointed at a model Yu
et al. never measured, which is what every axis of the program wants to do.
The remaining 16-of-21 inert coordinates — all six Phi-3, both Llama-2, both
Llama-13B — cannot be adjudicated by tuning either: with no null, "inert" and
"our threshold missed it" are indistinguishable.

**Next is Phase 0, not v6:** the weight-shuffle null, max-statistic
permutation logic (Nichols & Holmes 2002), planted-weight recovery, and an
explicit "no super weight found" outcome. Open it with `/new-experiment`.

**Also still open:** `verdict()` fires on `ppl_ratio > 10 OR kl > 1`, and its
KL is measured on three fixed short prompts that no corpus change touches.
OLMo-7B L6[269,9562] is still labelled CATASTROPHIC at x4 perplexity purely
on a KL of 2.36. The verdict function needs the scrutiny the perplexity
metric just received.

---

## 2026-09-02 (night) — outside Table 2: super activations without super weights

> ⚠️ **SUPERSEDED — the ablation UNIT is wrong, not the corpus this time.**
> Every ablation in this track zeroes **one scalar at a time**. Subramanian
> et al. (COLM 2026) Table 8 — a paper already in `papers/` and Tier 1 of our
> own reading list — zeroes Yu's coordinates **jointly**:
>
>     OLMo-1B         2 SWs   13.09 -> 47,951   (x3,663)
>     OLMo-7B         2 SWs    9.59 -> 42,024   (x4,400)
>     Phi-3-mini      6 SWs    9.48 ->  3,543   (x374)
>     Mistral-7B      1 SW     8.08 -> infinity
>
> Our Phi-3 numbers are x1.00-x1.02 for each of those same six coordinates
> individually. Both are right; they answer different questions. So the
> heading below is wrong: the evidence points to criticality **distributed
> over a small set**, not to super activations lacking a weight-level cause.
> Yu et al. say as much in §2.2 ("up to six weights and one activation").
>
> The signature is in our own JSON and we did not read it: every "inert"
> model's candidates share one `down_proj` INPUT COLUMN — Llama-2-7B
> [2533,7890]+[1415,7890]; Llama-3.1-8B three rows on column 2427; Qwen3-8B
> two on 5723; Llama-13B row 2231 at shares 0.593+0.407; Phi-3 three rows x
> two columns. One intermediate neuron fans one massive input into several
> residual channels; removing one weight removes one channel.
>
> **Read "inert" as "individually inert" everywhere below and in the two
> entries above.** Joint ablation is untested. Until it runs, no claim here
> about a model lacking super weights is supported.
>
> Also struck: "Qwen3-8B's is 28x larger than OLMo-1B's" (raw residual
> magnitudes are not comparable across models with different norms and
> dtypes); the `rtn+SW ~= rtn` mechanism sentence (q6 found that no-op on
> EuroLLM too, which HAS a strong single super weight, so the repo's own
> data contradicts it); and "three detectors agree" as independent evidence
> (they share prompt, thresholds and tensor).
>
> Two replications here ARE strong and should be foregrounded: our OLMo-1B
> x3,667 vs Subramanian's independent x3,663 on the same coordinate, and v5
> recovering q6's TowerBase L1[2533,7890] = 1.5391 vs 1.5390625.

First models with **no published answer key**. Three, chosen to answer
specific questions rather than to survey (`sw_models.MODERN`):

| model | why |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | the lineage: Llama-1-7B ablates ×181, Llama-2-7B only ×1.55 — does it come back? |
| `Qwen/Qwen3-8B` | a 2025 model from a family the paper never touched |
| `Unbabel/TowerBase-7B-v0.1` | a Llama-2-7B fine-tune this repo's q6 claims *does* have a super weight |

### No super weight in any of them — and three detectors agree

| detector | Llama-3.1-8B | Qwen3-8B | TowerBase-7B |
|---|---|---|---|
| v1 (strictest) | 5 cand → ×1.02 | **0 candidates** | **0 candidates** |
| v3 (loosest; 124 extras on Table 2) | 15 cand → ×1.02 | 9 cand → ×1.04 | 10 cand → ×1.58 |
| v5 | 3 cand → ×1.02 | 2 cand → ×1.04 | 2 cand → ×1.58 |

v3 over-generates by 12× on Table 2 models and still finds nothing
load-bearing here, so this is not detector sensitivity. Best result anywhere
is TowerBase at ×1.58 / KL 0.464 — "damaged", nowhere near the ×16–×3667 of
the five confirmed super weights.

### But the phenomenon is present in every one of them

`activation_profile.py`, max |residual activation| by depth:

| model | peak \|h\| | onset | persists | has a super weight? |
|---|---|---|---|---|
| OLMo-1B | 427 | L2 | 13/14 | **yes**, ×3667 |
| Llama-7B | 1,364 | L3 | 27/29 | **yes**, ×181 |
| Llama-2-7B | 893 | L2 | 28/30 | no (×1.55) |
| TowerBase-7B | 1,080 | L2 | 28/30 | no (×1.58) |
| Llama-3.1-8B | 322 | L2 | 29/30 | no (×1.02) |
| **Qwen3-8B** | **11,968** | L7 | 28/29 | no (×1.04) |
| Phi-3-mini | 3,776 | L5 | 24/27 | no (×1.0) |

Every model shows Yu et al.'s Figure 4 signature — a massive activation
appearing at one early layer and holding constant magnitude to the end.
Qwen3-8B's is **28× larger** than OLMo-1B's, the model with the strongest
super weight we measured.

**So the finding is not "newer models lack super activations". It is:
super activations look universal; a single scalar weight responsible for
one does not.** If that holds up it is a mechanism claim with a direct
consequence for the compression audience: where the structure producing the
massive activation is distributed, protecting a handful of scalars during
quantization cannot work — which is what q6's `rtn+SW ≈ rtn` negative found
empirically, now with a candidate mechanism.

### TowerBase reproduces this repo's own q6 coordinate exactly

Independent code, independent run: v5 returned `L1[2533,7890] = 1.5391`;
`docs/prior_experiments_and_ideas.md` §2 records q6 finding that same
coordinate at value `1.5390625`. It also supplies the base-model datapoint
q6's "SFT sharpens the super weight" claim never had, both on this harness:

| | weight | ppl | KL |
|---|---|---|---|
| Llama-2-7B (base) | +1.5625 | ×1.55 | 0.243 |
| TowerBase-7B (fine-tune of it) | +1.5391 | ×1.58 | **0.464** |

KL roughly doubles at the same coordinate. Directionally what q6 predicted.
⚠️ **One pair, no CI, n=1** — a lead, not a result, and q6's own numbers
(KL 0.957 there) came from a different protocol and are not comparable as
measured.

### Chronology does not explain it

Llama-1 (2023-02) ×181 · Llama-2 (2023-07) nothing · Mistral-v0.1 (2023-09)
×1425 · Phi-3 (2024-04) nothing · OLMo-0724 (2024-07) ×3667 · Llama-3.1
nothing · Qwen3 nothing. Not an era effect. Model- or recipe-specific.

### ⚠️ What would make this wrong — none of it tested

1. **One prompt.** Every detection run used `"Language modeling is "`. The
   paper claims a single prompt suffices; nobody has checked that outside
   its own table.
2. **Instruct variants.** Llama-3.1-8B-Instruct and Qwen3-8B are
   post-trained; TowerBase is a base model and shows the largest effect of
   the three.
3. **No null.** Three detectors agreeing on a negative is reassuring, not
   calibrated. Every threshold in all three was tuned against Table 2.
4. **n=3.** Not a result. The other cached models — EuroLLM-9B (q6's
   strongest, KL 3.284), Aya, BLOOM, Gemma-3 — are the obvious extension,
   and that is Tier B, which wants a `/new-experiment` spec first.

### Why this is a floor deliverable, not a failure

`three_axis_program.md` §5 already anticipated it: floor deliverable #2 is
the re-verified table *"including, if it comes to that, the first defensible
**absence** claims in this literature ('no super weight under a calibrated
null' is currently unstatable by anyone)."* Subramanian et al. (COLM 2026,
Tier 1 reading list) already report damage is not universal. And q6's own
eight-model table spans 3.5 orders of magnitude in ablation KL with five of
eight below 0.25 — this repo's data said the same thing a year ago.

The blocker is not evidence, it is calibration: **Phase 0 converts "we found
nothing" into "there is nothing", and nothing else does.**
