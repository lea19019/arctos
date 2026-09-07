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

---

## 2026-09-05 — Repetition pilot + lesion control triple: set up and submitted (Claude, autonomous; Adrian away)

**Context.** The project is being re-scoped around a question that does not depend on the
"super weight" term: the constant that a few early-layer `down_proj` weights write into the
residual stream (peer-reviewed as massive activations / attention sinks), what it does, and
whether the loops a lesion produces ("We. We. We.") are the same failure as the loops a
*healthy* model falls into on its own. Design and pre-registered definitions:
`docs/repetition_experiment_design.md`; literature: `docs/literature/notes_repetition_sink_link.md`,
`docs/literature/notes_multilingual_repetition.md`.

**Verified, zero compute — Yona et al. 2025's "sink neurons" are Yu et al.'s super-weight
columns.** Read side by side from the two PDFs (`papers/Interpreting-2025-Repeated-
Token-Phenomenon.pdf` Table 1; `Yu-2024-The-Super-Weight-in-LLMs.pdf` Table 2):

| model | Yona: sink layer / neuron ids | Yu Table 2 (or our v5 detection) |
|---|---|---|
| LLaMa-1-7B | 2 / **7003** | L2[3968, **7003**] |
| LLaMa-2-7B | 1 / **7890**, 10411 | L1[2533, **7890**] |
| Mistral-7B(-Instruct) | 1 / **7310**, 8572 | L1[2070, **7310**] (base) |
| Llama-3-8B-Instruct | 1 / 198, **2427** | our v5: three rows on column **2427** (Llama-3.1-8B) |

Yona's index is the MLP intermediate neuron; Yu's `k` is the `down_proj` input column, i.e.
the same neuron. Four independent four-digit matches with the layer agreeing each time. So the
"super weight" is one scalar of the fan-out of the neuron that builds the sink, which is also
why our 16 individually-inert Table 2 coordinates share one input column per model. Caveats:
Yona's checkpoints are the Instruct variants for Mistral and Llama-3; Yona's second neuron per
model (10411, 8572, 198) has no counterpart in Yu's table; Yona's K in TopK is not stated.

**Code (branch `loops-pilot`, commit ed3056c).** `src/rep_metrics.py` (L-onset rule g=4, k=3,
W=100, t_min=32; seq-rep-4; rep-r; TNG; six unit tests pass), `src/gen_loops.py` (greedy,
no repetition penalty, FLORES+ documents → 50-token prompts, 256 new tokens, batched with
left padding + a batch-invariance check, provenance in every JSON, bulky records to
gitignored `.records.jsonl`), `src/lesion_controls.py` (dose sweep α∈{1,.75,.5,.25,0} on a
known coordinate; **contribution-mean control** = zero the weight and add its calibration-mean
contribution back on the output channel, bucketed first-token vs rest, via a forward hook with
KV cache off; **magnitude-matched random null** = 50 single-weight zeroings from the top-100
|W| of the same matrix, max-statistic; paired-bootstrap CIs over 32×2048 wikitext-2 windows;
massive-activation |h₀[j]| after layer L as the dose check; 50 FLORES-eng continuations scored
with the repetition metrics). Configs: `configs/loops_pilot/*.yaml` (TowerBase-7B, EuroLLM-9B-
Instruct, Aya-Expanse-8B, BLOOM-7B1, OLMo-1B, Mistral-7B; 4–10 languages each, 200 prompts),
`configs/lesion/{olmo1b,mistral7b}.yaml`. SLURM: `slurm/loops_pilot.sh`, `slurm/lesion.sh`
(`--qos=cs --gpus=1`, no partition, per `BYU_ORC_AGENTS.md`, now copied to the repo top level).

**Smoke test (CPU, login node, OLMo-1B, 2 windows × 256 tokens, 2 prompts × 10 tokens):**
baseline ppl 12.86, |h₀[1764]| 267.7; dose 0 → ppl 20,086, |h₀| 5.3, loops from the first
tokens (`"\n mar,\n\n\n\n"`); **contribution-mean → ppl 12.87, |h₀| 239.6, no loops.** Two windows
is not a result, but it says the control does what Sun/Owen's set-to-mean did for activations:
the *presence* of the constant is what matters, not the weight. Matched-random ×1.000 (2 draws).

**Submitted:** lesion array 13593588 (OLMo-1B, Mistral-7B), loops array 13593589 (6 models).
Both pending at 21:21 on resources (cs nodes partly in maintenance). Results go to
`results/lesion/` and `results/loops_pilot/`; `src/loops_summary.py` prints the tables.

**Also learned today, not results:** FLORES+ devtest has 281 documents (URLs) per language,
median 4 sentences each, so 200 fifty-token prompts per language are available; the lab's
NLLB prediction files in `grp_mtlab` hold one sentence each, so no repetition can be read
off them; the ToAll incident doc reports 13.9% of Efik segments with ≥3× repeated tokens
from the lab's fine-tuned NLLB-600M in production — a natural-repetition observation in a
healthy translation model, decoding settings unknown.

**Added later the same evening — third experiment.** `src/constant_by_language.py` +
`configs/const_lang/*.yaml` + `slurm/const_lang.sh` (array 13593778, 7 models): per language,
20 FLORES+ sentences → where the massive activation appears (onset layer, channel, token
position), its magnitude with a bootstrap CI, and the v5-traced coordinate; plus a
cross-language agreement summary. CPU smoke on OLMo-1B with 2 sentences each of English and
Swahili: same channel 1764, same onset layer 1, same token position 0, same coordinate
L1[1764,1710], peaks 423 vs 418 — two sentences, a plumbing check only. First array
submission (13593777) was cancelled: it went out before a JSON-serialisation bug found by the
smoke test was fixed; nothing ran.

## 2026-09-06 — First results came back; one confound found before writing them up

All 15 array tasks completed (lesion 13593588, loops 13593589, const-lang 13593778; 3–40 min
each). **Before the numbers: a BOS confound.** In transformers 5.15 the tokenizer attribute
`add_bos_token` is `False` for Mistral, TowerBase, EuroLLM and Aya even though their default
call `tokenizer(text)` *does* prepend BOS. My runners keyed on the attribute, so those four
models ran **without BOS** in all three experiments (OLMo, BLOOM, Qwen3 are unaffected: their
default is no BOS). Without BOS the constant lands on the first delimiter token instead of
position 0 (visible in the per-language table: Mistral's sink on a comma at positions 3–21,
Aya's on ' ', ',', ' de'), and my contribution-mean control — which added the mean back at
position 0 — put the constant in the wrong place for Mistral: ×1196 [563, 2166], loops, while
for OLMo (sink genuinely at position 0) it was harmless, ×1.30 [1.00, 2.18], |h₀| 249 vs 268.
So the Mistral control result is an artefact of my bucketing, not a finding. Fixes (commit
below): BOS policy = the tokenizer's real default (`tokenizer("x").input_ids[0] == bos_id`),
recorded per run; contribution-mean bucketed by *sink positions detected from |x_k|* inside
the hook (per token, KV-cache safe) instead of by position 0; batch-invariance check now also
compares loop *rates* at bs=1 vs batched. The no-BOS results are kept as
`results/*_nobos/` and all three arrays are re-run with the default BOS policy.

**What the no-BOS run still established (unchanged by BOS for these models):**
- OLMo-1B dose sweep, 32×2048 wikitext-2 windows: α=0.75 ×1.03 [1.03,1.04]; 0.5 ×8.3 [7.7,9.0];
  0.25 ×279 [256,305]; 0 ×3667 [3425,3920] (replicates the established ×3667); |h₀[1764]| falls
  linearly with α (268→202→137→71→5). Matched-random null over 50 draws from the top-100 |W|:
  max ×1.000. Mistral-7B: 0.75 ×1.03; 0.5 ×1.39 [1.11,2.15]; 0.25 ×83 [68,110]; 0 ×1425
  [1229,1647] (replicates ×1430); null max ×1.001.
- **Same channel, same onset layer, same traced coordinate for every input language, in all 7
  models** (20 FLORES+ sentences per language, 6–11 languages each): OLMo-1B L1 ch1764
  [1764,1710]; Mistral L1 ch2070 [2070,7310]; TowerBase L1 ch2533 [2533,7890]; EuroLLM-9B L9
  ch1448 [1448,3575]; Aya-8B L2 ch2619 [2619,1079]; Qwen3-8B-Base L6 ch2276 [2276,5723];
  BLOOM-7b1 L7 ch1947 with no single weight passing the v5 share rule in any language. Per-
  language magnitudes have overlapping CIs within a model (e.g. EuroLLM 7026–7782 across 11
  languages; OLMo 407–423 across 6). Channel identity does not depend on BOS; the *position*
  does, so this table is re-run rather than cited.
- Greedy loop rates in healthy models are high everywhere (English, 200 prompts, 256 tokens):
  OLMo-1B 0.84, Mistral 0.83, BLOOM 0.90, TowerBase 0.79, EuroLLM 0.72, Aya 0.49 — so the
  natural-repetition arm is well powered, and the base rate is a model property before it is
  a language property. Rates vary within a model by language (Aya: zh/ko ≈0.75 vs es 0.40;
  BLOOM: zh 0.94 vs xho 0.36), but EOS rates vary too (OLMo emits EOS early in fr/de) and the
  batched-vs-bs1 exact-match rate for bf16 7–9B models is only 4–9/16, so per-language claims
  wait for the BOS-correct re-run and the rate-level invariance check.

## 2026-09-06 — BOS-corrected results (arrays 13594648 / 13594649 / 13594650, commit a1e2b82)

All 15 tasks completed. Provenance in every JSON (git sha, resolved config, versions, model
revision, dtype, SLURM job id, whether BOS was prepended). Coverage is stated per claim.

### 1. The lesion measures the constant's *presence*, not the weight's value (2 models)

Wikitext-2 test, 32×2048-token windows, paired bootstrap 95% CI; 50 FLORES-eng prompts,
greedy, 128 new tokens, no repetition penalty; matched-random null = 50 single weights from
the top-100 |W| of the same matrix, max-statistic.

| model | zero | contribution-mean (sink-bucketed) | random null max | |h₀| baseline → mean-ctrl |
|---|---|---|---|---|
| OLMo-1B L1[1764,1710] | ×3667 [3425, 3920] | **×1.00 [1.00, 1.00]** | ×1.000 | 268 → 249 |
| Mistral-7B L1[2070,7310] | ×1425 [1229, 1647] | **×1.52 [1.27, 1.86]** | ×1.001 | 264 → 312 |

Loop rate under the mean control equals baseline (OLMo 0.66 vs 0.62; Mistral 0.50 vs 0.54).
This is Sun et al. 2024 Table 3 / Owen et al. 2025 Table 1 (set-to-mean harmless, set-to-zero
catastrophic) reproduced at the **weight** level, which no published ablation has done: what
the model needs is the constant the neuron writes at the sink position, and a fixed value
serves. n = 2 models, 1 revision each; the earlier no-BOS Mistral control (×1196) was my
bucketing error, not a finding (previous entry).

**Dose sweep.** OLMo: α=0.75 ×1.03; 0.5 ×8.3; 0.25 ×279; 0 ×3667, |h₀| linear in α. Mistral:
0.75 ×1.03; **0.5 ×1.39 [1.11, 2.15] with loop rate 1.00 and seq-rep-4 0.99** ("bekan bekan
bekan…"); 0.25 ×83; 0 ×1425. So in Mistral generation collapses at a dose where perplexity
barely moves — perplexity is the wrong readout for this failure, as the design memo warned
(Jin 2025's looping generation at perplexity 2.99).

### 2. One constant per model, identical across input languages (7 models, 6–11 languages, 20 FLORES+ sentences each)

| model | onset L / channel / traced weight | languages | position | peak range across languages |
|---|---|---|---|---|
| OLMo-1B (no BOS) | 1 / 1764 / [1764,1710] | 6 | first token | 407–423 |
| Mistral-7B (BOS) | 1 / 2070 / [2070,7310] | 7 | `<s>` | 266.0–266.2 |
| TowerBase-7B (BOS) | 1 / 2533 / [2533,7890] | 10 | `<s>` (zh: '。' at pos 89) | 1101–1233 (zh 2423) |
| EuroLLM-9B-Instruct (BOS) | 9 / 1448 / [1448,3575] | 11 | `<s>` | 6858–6955 |
| Aya-Expanse-8B (BOS) | 2 / 2619 / [2619,1079] | 10 | `<BOS_TOKEN>` | 771.0–771.4 |
| Qwen3-8B-Base (no BOS) | 6 / 2276 / [2276,5723] | 9 | first token | 7418–10269 |
| BLOOM-7B1 (no BOS) | 7 / 1947 / no single weight passes the share rule | 10 | first token | 3513–3833 |

Same channel, same onset layer, and (BLOOM excepted) the same traced coordinate in 100% of
sentences in every language, for every model. Two honest qualifications. (a) With a BOS token,
position 0 attends only to itself, so h₀ is a fixed function of the BOS embedding and its
cross-language identity is guaranteed by causal masking — the value is literally the same
number (Mistral 266.0 in all seven languages). The informative cases are the no-BOS models,
where the constant lands on whatever the first token is (a quote mark, '他', 'U') and the
channel, layer and weight are still identical with magnitudes within ~5% (OLMo) to ~30% (Qwen);
and the no-BOS first run of the BOS models (`results/const_lang_nobos/`), where the constant
lands on the first delimiter (Mistral: ',' at positions 3–21) and the channel/weight are again
identical. (b) The Chinese exception in Tower is a text artefact: my sentence split on ". "
does not split Chinese, so the input was a whole paragraph and the largest activation was on
a mid-sequence '。' at 2423, more than twice the BOS constant — Sun et al.'s delimiter case,
worth a proper follow-up rather than a footnote. Fix the split before re-using this script.

Reading: the model builds one constant, from one neuron, through one weight (or, in BLOOM,
a distributed fan-out), and it is the same object for Swahili, Yoruba, Xhosa, Igbo, Arabic,
Chinese and English input. It is not a language-specific piece of machinery; whether the
*rest* of the sink circuit (which heads dump attention there, for which inputs) is
language-dependent is the open question this does not answer.

### 3. Healthy models loop constantly under greedy decoding (6 models × 4–10 languages, 200 prompts, 256 tokens, no penalty)

Loop rate (pre-registered L-onset rule), 95% Wilson CI, English: TowerBase 0.86 [0.81,0.90],
Mistral 0.84 [0.78,0.88], OLMo-1B 0.84 [0.78,0.88], BLOOM 0.90 [0.85,0.93], EuroLLM 0.63
[0.56,0.69], Aya 0.60 [0.53,0.67]. Full table: `src/loops_summary.py results/loops_pilot`.
Rate-level batch invariance holds (loops at bs=1 vs batched on the same 16 prompts agree within
one) even though exact token match is poor for bf16 7–9B models (2–8/16), so rates are usable
and per-sequence comparisons are not. Within-model language differences exist and are mostly
confounded: OLMo emits EOS early in French/German (EOS 0.82/0.63), which caps its loop rate
there at 0.07/0.11; BLOOM loops *earliest* in Igbo, Xhosa, Yoruba and Swahili (median onset
9–27 tokens vs 36–47 in English/Spanish), which leaves almost no pre-onset window for the
signature analysis (loop-with-window 0.02–0.23). Aya (zh/ko 0.68–0.77 vs es 0.48) and Tower
(en 0.86 vs fr 0.68) show real spread with CIs that separate, but with n=1 model per
architecture and greedy-only decoding this is a base-rate table, not a claim about languages.
The natural-repetition arm is well powered everywhere except BLOOM-in-African-languages, where
the loops start too early to window.

### What this changes

- The "super weight" is one scalar of the fan-out of the neuron that builds the sink, and its
  criticality is the constant's presence (§1) — consistent with the Yona = Yu index match.
- The constant is shared across languages in every model examined (§2); the interlingua
  question about this object has a first answer, with the BOS caveat stated.
- Repetition is abundant in healthy models under greedy decoding (§3) and appears in the lesion
  before perplexity moves (§1), so the pre-onset signature comparison is runnable; the next
  experiment is the position-split ablation and the pre-onset sink statistics, not more base rates.
