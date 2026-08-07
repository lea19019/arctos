# Super-weight formation — the three-axis program

**Drafted 2026-08-06** from a planning conversation, the day the three interlingua
audits landed. Status: **proposed, not committed.** Nothing here has a runner, a
config, or a satisfied-when yet — Phase 0 must go through `/new-experiment` before
any code lands.

---

## 1. Why this direction

Two standing interests — cross-lingual structure and compression — meet in one
phenomenon, and the meeting point is confirmed empty:

- **`docs/registry.md`, "Genuinely unclaimed":** *"Super-weight formation across
  training. Nothing tracks when super weights appear. On no backlog, in no
  proposal."* Confirmed still unclaimed 2026-08-06 by three independent
  zero-result searches (`interlingua/docs/method_landscape.md` §5).
- The prior-art audit ranked **causal super-weight (deletion-criticality)
  formation** the #2 genuinely open direction it found
  (`interlingua/docs/prior_work_map.md` §10).
- The link between **the super weight specifically** and cross-lingual behavior
  is empty in the literature (`prior_work_map.md` §8, and re-confirmed by the
  2026-08-06 spot-check — see §7). ⚠️ The *neighborhood* is occupied: per-language
  damage from tiny ablations is well studied at **neuron** granularity
  (arXiv:2402.16438; arXiv:2402.18815 — "as few as four neurons" collapse one
  language). Our claim must always say *single scalar weight*, never "tiny
  ablations."
- Unlike Tier 1 of the interlingua plan (audited: no consumer), this has
  identifiable consumers: the quantization/pruning community (~51 of the
  super-weight papers' citations are exactly that), and our own NLLB deployment
  (ToAll dubbing, 16 GB T4 target).

Why the field is empty is explainable, not mysterious: the phenomenon was
discovered by a compression team (whose actionable problem — protect the weights
when quantizing — is solved for their purposes), it fits neither the SAE/circuit
ontology of mech interp nor any standard statistical toolkit, the one follow-up
was deflationary, and until PolyPythias (2025) the multi-seed formation study
required training dozens of models. That last barrier fell.

## 2. What this repo already established (q6) — and why Phase 0 re-verifies it

From `docs/registry.md` (audited 2026-08-05). All of it **n=24–32, greedy
decoding, "directional, not publication-grade"** — treated here as leads, not
results.

| Finding | Status |
|---|---|
| Causal-KL detection beats spike detection (Aya L31 spike is causally inert: KL 4×10⁻⁷) | measured, 8 models |
| Super weights early-layer (L0–L9) in every non-Gemma family; strength spans 3.5 orders of magnitude | measured, 8 models |
| Ablating one super weight destroys MT (EuroLLM 57.9 → 4.8 chrF++); ablating the 1000 largest-magnitude weights does nothing | measured |
| EuroLLM has exactly **two** load-bearing scalars (second: L1, KL 2.975 — unreported in any writeup) | measured, unreported |
| TowerBase and TowerInstruct share the same super weight (same coordinate, same value); SFT sharpens it, KL 0.96 → 1.25 | coordinate match = replication of Yu et al.; **the sharpening is ours, n=1, no CI** |

**Why re-verify:** the detector is greedy — per-layer `argmax`, always returns a
candidate, no null, no threshold, never validated against a planted weight
(`registry.md`). And q6's scale is below this repo's own rigor floor. Trust is
rebuilt by rerunning, not by rereading.

## 3. Constraints from measured negatives — the program must not violate these

1. **Super-weight FP16 preservation is a no-op for quantization** (`rtn+SW ≈ rtn`,
   all 8 models). The bridge to compression is **not** "protect the super weight."
2. **Importance ⟂ quantization sensitivity** (Q5, twice, two granularities). No
   importance-driven bit allocation, ever (`registry.md`, Ruled out #1).
3. **Super weight ≠ salient channel.** They behave oppositely (preservation:
   no-op vs. rescues the 3-bit cliff). Never conflate them.
4. **Magnitude ≠ importance ≠ sensitivity.**

The legitimate compression link is the mechanistic chain **super weight (scalar) →
massive activations (channels) → outlier structure that quantization fights**
(Yu et al.), studied as *formation* — not as a protection recipe.

## 4. The program

### Phase 0 — calibrated detector + re-verification (gate for everything)

**Question:** do the q6 super-weight findings survive a detector with a null?

- Build the calibrated detector: weight-shuffle null (preserves the magnitude
  marginal, destroys learned coordinate structure — the right null *because* the
  repo's own data shows magnitude is not what carries the effect), max-statistic
  permutation logic (Nichols & Holmes 2002), planted-weight recovery test, and an
  explicit "no super weight found" outcome — which no published method can emit.
- Re-detect on the q6 models; re-measure ablation KL at proper n.
- §20.3 triggers fire here: provenance manifest, config-load tests, invariant
  tests for the detector.

Deliverable either way: the first super-weight detector with a calibrated null —
standalone methods contribution — plus a re-verified (or corrected) q6 table.

### Axis 1 — formation on public checkpoints (observational; no training)

**Question:** when does deletion-criticality form, gradually or abruptly, and do
seeds agree on the coordinate?

⚠️ **Novelty scope (post-verification):** a *magnitude-level* weight-formation
trace exists — Ding (arXiv:2605.18898, unrefereed, single seed, Pythia-70m, 14
checkpoints, no causal test): an isolated outlier reaches \|w\|≈1.0 by step
5,000. Axis 1's claim is therefore **causal criticality + seeds + architecture**,
not "first formation trace." Ding's curve is the free baseline to compare the
causal trace against — magnitude onset vs. criticality onset may themselves
diverge, which would be a finding.

- **Backward-trace:** find the super weight at the final checkpoint (Pythia,
  OLMo), then measure *that coordinate's* ablation-KL at every earlier checkpoint
  (154 for Pythia; one ablation + eval per checkpoint — cheap). Also scan whether
  a different coordinate was critical earlier.
- **Seeds:** PolyPythias (9 seeds × 5 sizes, 14M–410M, ~7k checkpoints) — same
  coordinate across seeds? Also answers whether criticality exists below 1B at
  all (never observed below OLMo-1B; nobody has looked).
- **Architecture:** Ettin (paired encoder/decoder, identical data and recipe,
  250+ checkpoints, batch-level data per checkpoint) — decoder phenomenon or
  transformer phenomenon? This carries ADR 0001's open architecture question for
  free. Ettin's per-checkpoint batch records also allow "what data preceded
  formation."
- **Multilingual timing, still free:** BLOOM public tags, mmBERT's full
  multilingual chain, the Blevins XLM-R replica (39 checkpoints, FR+TR).

### Axis 3 — behavior in known multilingual models (endpoint; runs after Phase 0)

**Question:** is the super weight shared machinery or language-specific?

- Per-language ablation damage profiles on models with (re-verified) coordinates:
  EuroLLM (both scalars), Aya, BLOOM, Towers — using the replication harness
  (6 languages × both directions, n=960, chat templates, COMET), the only
  rigorous protocol in the repo.
- Equal damage across languages ⇒ a causal shared-bottleneck (interlingua) claim.
  Unequal ⇒ a mechanistic lead on uneven quantization damage (replication C2,
  **en→X only** — the X→en headline did not reproduce).
- **NLLB-600M**: first encoder-decoder **weight-level** super-weight search
  anywhere (spot-check confirmed, §7) — but the activation level is occupied and
  must be cited: NLLB-200 attention sinks are published (arXiv:2605.01229 —
  sinks on language tags, content tokens get 17–20% of cross-attention mass),
  and T5-11B encoder outlier activations are published (2025.naacl-long.430).
  Listed as specified-but-never-run in the speech-translation survey; directly
  serves the ToAll deployment.

### Axis 2 — controlled training (interventional; launches gated on Axis 1)

**Question:** what causes formation — and does multilinguality change it?

- Small models (~10–100M), **monolingual vs. bilingual vs. multilingual at
  matched byte budget and identical recipe**, plus the knobs the literature says
  govern related phenomena: weight decay (Gu et al.: can create or abolish sinks
  outright), attention variant, learnable attention bias.
- **Pipeline is built in parallel with Axis 1** (data, provenance, configs, one
  smoke run — long-lead work with no dependence on Axis 1 answers). **Run
  launches wait**, because Axis 1 determines: whether criticality exists at this
  scale at all, the checkpoint density (related phenomena form at ~1k–4k steps;
  a wrong grid makes runs unusable — Pythia's documented mistake), and which
  knobs must be matched across arms.
- A quantization face of the same runs — **scope narrowed after verification
  (§7): the broad question is taken.** arXiv:2510.06213 already tracks PTQ
  degradation along pretraining trajectories up to 32B/15T tokens (error rises
  early, then **surges during LR decay**, decoupling from validation loss), and
  arXiv:2602.02047 tracks outlier dynamics during pretraining. What survives for
  us: linking quantization-sensitivity onset to **causal super-weight formation
  specifically**, multi-seed, at small scale, with the multilingual contrast —
  stated against those two papers by name. Their LR-decay finding is also a
  design input: our WSD schedule's decay phase is a predicted sensitivity event,
  so checkpoint density must cover it. Respects constraints 1–2 above: this is
  measurement of formation cost, not a bit-allocation scheme.

### Ordering and gates

```
Phase 0 (detector + re-verify)
  ├─→ Axis 1 (public checkpoints)  ──┬─→ gate: launch Axis 2 runs
  │      [Axis 2 pipeline build      │
  │       runs in parallel]          └─→ Axis 2 analysis
  └─→ Axis 3 (multilingual endpoint)
```

Gate logic: if PolyPythias shows no criticality ≤410M, Axis 2's "search at 36M"
arm is dead **before** the GPU-weeks are spent, and Axis 2 refocuses on the
smallest scale where Axis 1 found the phenomenon.

## 5. Floor deliverables (guaranteed even if every exciting outcome is null)

1. The calibrated detector + planted-weight validation (methods contribution;
   nothing in the literature has a null).
2. The re-verified q6 table at proper n — including, if it comes to that, the
   first defensible **absence** claims in this literature ("no super weight under
   a calibrated null" is currently unstatable by anyone).
3. The formation trace on free checkpoints (Pythia + PolyPythias + Ettin), which
   generalizes — or kills — the repo's own n=1 sharpening finding.

This is the lesson of the Tier 1 post-mortem applied in advance: the headline is
allowed to fail; the floor is not.

## 6. Risks and open questions

- **Existence below 1B is unknown** — never observed, never sought. This is a
  risk for Axis 2 and simultaneously the cheapest interesting question Axis 1
  answers.
- **Recipe confounds.** Weight decay alone manufactures or abolishes the related
  phenomena (Gu et al., strongly non-monotonic sweep). Any cross-arm or
  cross-model comparison at unmatched regularization measures the recipe.
- **Detection thresholds are scale-dependent** in the direction that makes small
  models look clean (`method_landscape.md` §5.2). At 36M, "nothing found" and
  "threshold calibrated on 7B" are indistinguishable without the Phase 0 null.
- **The statistics floor applies** (`CLAUDE.md`): the checkpoint × coordinate
  scan is a multiplicity family; seeds are the unit of independence; step 0 is a
  free null; report effect sizes with CIs.
- **Consumer question stays live.** The quantization community and ToAll are
  named consumers for Axes 2–3; the pure formation story (Axis 1) should be
  framed for them, not only for interp.

## 7. Blind-spot verification — results (2026-08-06, two agents)

Full details in `reading_list.md`; what changed in this document:

**Novelty spot-checks** (each claim attacked with multiple search angles;
negatives are auditable via the strategies logged in the agent reports):

| Claim this program leans on | Verdict | Consequence |
|---|---|---|
| Per-language damage from single-weight ablation is unmeasured | **PARTLY OCCUPIED** | Neuron-granularity per-language ablation is thoroughly studied (LAPE arXiv:2402.16438; arXiv:2402.18815, NeurIPS 2024). The *single-scalar-weight* form survives. Axis 3 phrasing updated. |
| Quantization sensitivity across pretraining checkpoints is unmeasured | **REFUTED** | arXiv:2510.06213 (≤32B/15T; LR-decay surge) and arXiv:2602.02047 own the broad question. Axis 2's quantization face narrowed to the causal-super-weight link, multi-seed, multilingual. |
| Super weights in encoder-decoder models are unexamined | **PARTLY OCCUPIED** | Weight level: holds (nothing found). Activation level: taken — NLLB sinks (arXiv:2605.01229), T5 outliers (2025.naacl-long.430). NLLB claim narrowed to weight level. |

**Citation verification** (the never-opened `[discovery]` flags):

- **Confirmed:** 2605.18898 (one paper, not two — the only weight-level
  formation trace; unrefereed, single seed, no causal test); 2510.06477
  (step-1k synchronization — ⚠️ rests on n=2 Pythia sizes); 2603.05498
  (sinks ≠ massive activations — in direct tension with 2510.06477; cite the
  disagreement, not either side); 2605.15572 (4-orders-of-magnitude activation
  spread, Gemma3 extreme); Puccetti = arXiv:2205.11380 (Findings EMNLP 2022);
  DataDecide = arXiv:2504.11393 (ICML 2025; the ">30k checkpoints" figure is
  unconfirmed).
- **Failed:** 2603.27885 contains **no transformers at all** (MNIST/CIFAR label
  noise) — dropped from all lists. The "15/15 model–task pairs" attribution on
  2606.02378 is **not in the paper**; its surviving claim (induction precedes
  sinks 10–20×) is scoped to DCLM models, n=2.
- **The "three-paper corpus" claim:** technically true; honestly **two
  substantive papers + one marginal** (arXiv:2606.19367 mentions super weights
  only in a robustness check; same single author as 2605.18898).
- ⚠️ Four of the six flagged IDs are unrefereed single-author preprints — under
  this repo's rules none can carry a claim alone.
- **Unopened leads** surfaced but not verified (do not cite): 2606.20743,
  2601.22966, 2602.07596, 2505.21670.

## 8. Next steps

1. `/new-experiment` spec for Phase 0 — question, null, satisfied-when, before
   any code.
2. `/record-decision` capturing the three-axis structure and its gates, so the
   rationale survives the next context loss.
3. PI conversation: this direction vs. the salvaged interlingua directions vs.
   the two MS proposal drafts (`interlingua/docs/ms_proposal_v*.md`) — they
   compete for the same semester.
