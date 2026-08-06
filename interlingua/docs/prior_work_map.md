# Prior-work map — Tier 1, "Does the Interlingua Grok?"

**Purpose.** Adjudicate what is already occupied, specifically the surviving-novelty
claim in [`tier1_plan.md`](tier1_plan.md) §7: *"The descriptive two-stage finding is
already scooped … The defensible novelty was never the phenomenon — it is the
metric-artifact adjudication and the statistical protocol."*

**Produced** 2026-08-06 by ten parallel read-only agents plus direct verification of
the four load-bearing citations. Papers were read in full where a claim turns on
detail. Every "UNVERIFIED" below is marked as such; nothing here is inferred from an
abstract where the body was reachable.

**Status vocabulary.** `CLOSED` — a published paper does this, at comparable or larger
scale, and a reviewer will say so. `PARTLY OPEN` — occupied in substance but with a
named, defensible residual. `OPEN` — nothing found, with the search stated so the
negative is auditable.

---

## 0. Citation errors to fix before this goes to the PI

These are not stylistic. Two of them misattribute a paper, and one of those is in the
proposal PDF itself.

| Where | Says | Actually |
|---|---|---|
| Proposal PDF, references p. 22; `tier1_plan.md` §1, §3.6 | "Dumas, T., et al. (2026). *When Meanings Meet* … arXiv:2601.22851" | **Körner, Müller-Eberstein, Korhonen & Plank**, EACL 2026 Main, pp. 3149–3169, [`2026.eacl-long.145`](https://aclanthology.org/2026.eacl-long.145/). Verified directly against both the ACL Anthology page and the arXiv abstract. Dumas et al. is a *different* paper; the ID and the author name have been welded together. |
| `tier1_plan.md` §7 | "Inaba EMNLP 2025" | **Findings** of EMNLP 2025, not main conference. [`2025.findings-emnlp.725`](https://aclanthology.org/2025.findings-emnlp.725/), arXiv:2503.06394. |
| `tier1_plan.md` §7 | "Riemenschneider & Frank ACL 2025" cited as scooping the two-stage finding | Real paper, correct venue ([`2025.acl-long.661`](https://aclanthology.org/2025.acl-long.661/)), **but it does not claim a two-stage dynamic** — see §1.2. Citing it this way misrepresents it. |
| `tier1_plan.md` §7 | "copy-first-translate-later" | Körner, Matveev, Eichin, Kutyniok, Plank & Hedderich, [arXiv:2604.17633](https://arxiv.org/abs/2604.17633) v2 (26 Jun 2026). **No venue** — cite as preprint. |
| `tier1_plan.md` §2 | "cross-seed matched-feature rates as low as 1–4% … feature reproducibility 21–30%" | "30%" verified (Paulo & Belrose, [arXiv:2501.16615](https://arxiv.org/abs/2501.16615): 30% at 131k latents on Llama-3-8B, 42% at 32k on Pythia-160M). **"21%" is mis-sourced** — the only match is [arXiv:2605.04072](https://arxiv.org/abs/2605.04072), a 14.5M-parameter *clinical sequence model*, whose authors attribute the low figure to that model's scale and narrow vocabulary. Drop the "21–". |
| `tier1_plan.md` §2 | "Three independent 2024–2026 papers document that EAP/EAP-IG circuits have high … variance" | The three are identifiable — Miller, Chughtai & Saunders (COLM 2024, [arXiv:2407.08734](https://arxiv.org/abs/2407.08734)); Méloux, Portet & Peyrard ([arXiv:2510.00845](https://arxiv.org/abs/2510.00845)); Wu, Tonin & Cevher ([arXiv:2606.16920](https://arxiv.org/abs/2606.16920)) — but **the table cites none of them**, and Wu et al. call sample-wise variance *"largely benign"*. Only rephrasing variance survives their analysis. |

Note also: `tier1_plan.md` §2's "Out, and why" table contains **zero citations** for any
of its numbers. Under this repo's own claim-hygiene rules that is the weakest-sourced
section of the plan.

---

## 1. The claim table

| # | Claim | Occupied by | Scale | Status |
|---|---|---|---|---|
| 1.1 | Cross-lingual alignment forms in two stages during training | Inaba et al., Findings EMNLP 2025 | LLM-jp 150M–3.7B, **2 languages**, n=1 run | **CLOSED** |
| 1.2 | — same, from Riemenschneider & Frank | R&F, ACL 2025 | own 257M, 16 langs, ~44 ckpts, n=1 pretraining run | **MISCITED** — they claim *gradual*, not two-stage |
| 1.3 | — same, from copy-first-translate-later | Körner et al., arXiv:2604.17633 | own **1.7B**, 9 langs, **200 ckpts**, n=1 | **CLOSED** |
| 2 | Mechanism forms before behaviour in multilingual training (H1's structure) | Körner et al. 2604.17633, layer-swapping | 1.7B, 9 langs | **CLOSED in substance** |
| 3 | Schaeffer continuous-vs-discontinuous metric pairing applied to **cross-lingual transfer/alignment** | nobody found | — | **OPEN** (~85% confidence, method stated in §3) |
| 3b | Metric pairing applied to non-English task emergence across checkpoints | Du et al., NeurIPS 2024 | 1.5B/6B/32B, C-Eval + GSM8K-Chinese | **CLOSED**, and emergence *survived* |
| 3c | Adjudication that apparent multilingual training gains are partly measurement artifacts | **Körner et al., EACL 2026** | EuroLLM-1.7B, 26 ckpts, 11 langs, causal activation patching | **PARTLY CLOSED** — different method, same headline |
| 4 | Prior work on cross-lingual training dynamics is single-run with no error bars | confirmed for all four candidates | — | **TRUE** |
| 5 | No accepted statistical test exists for a phase transition in a training curve | Chow 1960; Andrews 1993; Bai–Perron 1998/2003; Muggeo 2003; Jewell et al. JRSS-B 2022 | — | **FALSE as written** |
| 5b | No such test has been *validated on ML training curves* | Hu, Chen, Saphra & Cho, TMLR 2023 (HMM); Hoogland et al., TMLR (LLC) | — | **PARTLY OPEN** — direct competitors exist |
| 5c | The specific PELT/BOCPD-on-log(step) + 5-seed-bootstrap protocol | nobody | — | **OPEN, and unsound** — see §5 |
| 6 | No interpretability paper carries a multiple-comparison protocol | Gröger et al. 2026; Shi et al. NeurIPS 2024; Zhu et al. ICML 2024; the whole neuro-NLP literature | — | **FALSE** |
| 6b | Permutation-null calibration for CKA / mutual-kNN with BH-FDR | **Gröger, Wen & Brbić, [arXiv:2602.14486](https://arxiv.org/abs/2602.14486)** | the exact planned measures | **CLOSED** |
| 6c | No *checkpoint-scan* study controls error rate over the checkpoint × layer family | nobody found | — | **OPEN** — this is the narrow claim that survives |
| 7 | H3a: alignment **dynamics** compared across training objectives, matched from-scratch | nobody | — | **OPEN** |
| 7b | Objectives compared matched-from-scratch at **endpoint**, at ~this architecture | **Li et al., EMNLP 2024** ([arXiv:2407.15489](https://arxiv.org/abs/2407.15489)) | 12L/512d/8h/FFN2048, 6 langs, 5 objectives, n=1 seed | **CLOSED** for endpoints |
| 8 | Super-weight / massive-activation formation tracked across checkpoints | [arXiv:2508.03616](https://arxiv.org/abs/2508.03616); Kovaleva et al. 2021; Gu et al. ICLR 2025 | Pythia 14M–12B, ~154 ckpts | **CLOSED** |
| 8b | **Causal** super-weight (deletion-criticality) formation across checkpoints | nobody | — | **OPEN** |
| 8c | Do these phenomena exist at 36M/6L/512d? | Pythia-14M (6L/128d); BERT-small 4L/512d 29.1M | — | **YES** — not a scale risk |

---

## 2. The three named "scooping" papers — verified

### 1.1 Inaba et al. — says what the plan says, and more literally than the plan claims

**Inaba, Kamoda, Inui, Isonuma, Miyao, Oseki, Takagi & Heinzerling, "How a Bilingual LM
Becomes Bilingual: Tracing Internal Representations with Sparse Autoencoders."**
Findings of ACL: EMNLP 2025, pp. 13458–13470. [`2025.findings-emnlp.725`](https://aclanthology.org/2025.findings-emnlp.725/) · [arXiv:2503.06394](https://arxiv.org/abs/2503.06394)

Verbatim §4.1: *"These findings suggest that LLMs learn in two distinct stages. 1. During
the early to mid-training phase, they develop independent semantic representations within
each language. 2. In the subsequent mid-to-late training phase, they begin to align these
semantic representations across languages."*

- LLM-jp public checkpoints, 150M / 440M / 980M / 1.8B / 3.7B. They train no LM — only TopK SAEs.
- **Two languages only** (Japanese, English). Checkpoint count never stated; results binned into three token-count regimes.
- Full-text grep: **zero** occurrences of `seed`, `error bar`, `confidence interval`, `standard deviation`, `variance`.
- No changepoint test. No continuous-vs-discontinuous comparison. Never says "grokking" or "phase transition".
- Own limitation: *"the findings may not generalize to all language pairs."*

**Verdict: CLOSED.** This is the cleanest published statement of the two-stage finding.
Its weaknesses are exactly the ones the plan names.

### 1.2 Riemenschneider & Frank — the plan miscites this one

**Riemenschneider & Frank, "Cross-Lingual Generalization and Compression: From
Language-Specific to Shared Neurons."** ACL 2025 Long, pp. 13470–13491.
[`2025.acl-long.661`](https://aclanthology.org/2025.acl-long.661/) · [arXiv:2506.01629](https://arxiv.org/abs/2506.01629)

Verbatim: *"models initially form language-specific representations, which **gradually
converge** into cross-lingual abstractions as training progresses."* The word used
throughout is "gradually". The paper contains **no** "phase transition", "abrupt",
"sudden", or "two-stage" language applied to its own result. The "clear transition" in its
abstract is *across layers*, not across time.

- Own toy model: XGLM-564M architecture with `d_model` 1024→512, **257M params**, 2¹⁷ steps, 72h on one A100. Plus BLOOM-560M and 7B1 public checkpoints. 16 languages, 200 concepts.
- Checkpoints at powers of two {1…131072} plus 5000-step intervals ≈ 44.
- **The three seeds are probe-training seeds, not pretraining seeds:** *"we train a logistic regression classifier for each layer … we repeat this experiment with three different random seeds."* One pretraining run.
- Notable: they explicitly leave open *"syntactic phenomena shared across languages (such as **agreement** and word order)"* — which is Tier 1's minimal-pair target.

**Verdict: MISCITED.** This is not a scoop of the two-stage phenomenon; it is the
*gradual* counter-reading, which is the very question the plan says is its novelty.
Citing it as scooping the phenomenon hands a reviewer a free correction. Recite it as the
gradualist counterpoint and it becomes an asset.

### 1.3 Copy First, Translate Later — the closest competitor, and it is close

**Körner, Matveev, Eichin, Kutyniok, Plank & Hedderich, "Copy First, Translate Later:
Interpreting Translation Dynamics in Multilingual Pretraining."**
[arXiv:2604.17633](https://arxiv.org/abs/2604.17633) v2, 26 Jun 2026. **No venue — preprint.**

The arXiv ID in the plan is correct and the paper resolves.

- **Their own from-scratch run:** LLaMA-style causal decoder, 1,669,433,344 params, 24L/2048d/16h, ~37B tokens, WSD schedule, weight decay 0.01.
- **200 checkpoints, every 185M tokens — uniformly spaced, not log-spaced.** Claimed as *"more fine-grained … than any other publicly available multilingual checkpoints."* Checkpoints to be released.
- **9 languages, 72 directed pairs.**
- **Seeds: the string `seed` appears zero times in v1 and v2.** Every curve is one run. Error bars exist but are *"standard deviation across language-pair averages"* — dispersion across languages, not across runs. No CIs.
- **No changepoint test.** The phase boundary is eyeballed: *"At around 11B tokens processed, **we observe** the first successful predictions in WLT for word pairs without token overlap, **indicating** the development of generalizing mechanisms."* The only inferential statistic in the paper is a Shapley-value R² decomposition in Appendix E (three predictors explain 8.7% of variance).
- **`grok` appears zero times.** They describe the trajectory as *"WLT accuracy improves gradually during training."*
- Central claim: *"translation develops in two distinct phases: an initial phase dominated by copying and surface-level similarities, and a second phase in which more generalizing translation mechanisms are developed while copying is refined."*

**The part that hurts H1 specifically.** Their layer-swapping experiments establish that
*"the generalizing mechanisms in the intermediate layers are largely formed between 4B and
18.8B tokens processed"* — i.e. internal mechanism forms before behavioural translation
gains. That is H1's structure, already measured causally, at 47× the model scale and 3.3×
the checkpoint density.

Own limitations, verbatim: *"The training run has not converged"*; *"while we argue for the
emergence of shared representational spaces, **our methodology does not directly examine
the underlying activations**"*; ablations *"not performed exhaustively … This might occlude
other interesting dynamics."*

**Verdict: CLOSED for the phenomenon; H1's mechanism-before-behaviour structure is
CLOSED in substance.**

---

## 3. The mirage test — the load-bearing question

**Answer: nobody has run Schaeffer et al.'s continuous-vs-discontinuous metric pairing on
cross-lingual transfer or cross-lingual alignment — not across scale, not across training.
Confidence ≈ 85%. But the surrounding territory is much more occupied than the plan assumes,
and one paper takes the headline.**

### How the negative was established (so it is auditable)

- Enumerated **all 733 papers citing** Schaeffer et al. ([arXiv:2304.15004](https://arxiv.org/abs/2304.15004)) via the Semantic Scholar graph API; regex over title+abstract for multilingual/cross-lingual/translation terms → 48 hits, all applications, **none a mirage test**.
- Enumerated **all 3,737 papers citing** Wei et al. ([arXiv:2206.07682](https://arxiv.org/abs/2206.07682)) → 16 hits on the emergence ∩ multilingual filter, all irrelevant.
- **Co-citation intersection** — the strongest filter, since such a paper would have to cite both sides: citers of Wang/Minervini/Ponti ∩ mirage citers = **2** (neither relevant); citers of Blevins ∩ mirage citers = **3** (one relevant-adjacent); citers of mOthello ∩ mirage citers = **0**.
- Residual risk: the arXiv full-text API returned HTTP 429 on all 13 attempts, Google Scholar was unreachable, and S2 abstract coverage was 656/733. A very recent 2026 preprint could be under-indexed.

### What *is* occupied

**Du, Zeng, Dong & Tang, "Understanding Emergent Abilities of Language Models from the Loss
Perspective," NeurIPS 2024** ([arXiv:2403.15796](https://arxiv.org/abs/2403.15796)).
Ran exactly the metric triple — Accuracy vs. CorrectChoiceProb vs. **Brier Score**,
explicitly citing Schaeffer — on 1.5B/6B/32B **intermediate checkpoints**, on **C-Eval
(Chinese)** and **GSM8K-Chinese**. Result: *"All three metrics — accuracy, correct choice
probability, and Brier Score — show emergent performance improvements."* **The jump
survived.** This is not cross-lingual transfer (C-Eval is monolingual-Chinese task
performance, no alignment measure anywhere), so the gap is real — but it is narrower than
"nobody has done this on non-English", and Du et al. is the first thing a reviewer cites.

**Wei et al., TMLR 2022, Appendices A.1–A.2.** Already ran metric-robustness on **IPA
transliterate** (BLEU-scored, cross-script): *"For all three tasks, the emergent behavior
appears to be **independent of which evaluation metric is used**."*

⚠️ **Do not build a related-work paragraph on the Berti et al. survey**
([arXiv:2503.05788](https://arxiv.org/abs/2503.05788)). It claims Wei et al.'s
metric-robustness analysis covers "French-English translation." It does not — that analysis
covers Modified arithmetic, IPA transliterate, and Periodic elements only. The Fr-En claim
traces to a blog post where BLEU is the *only* metric reported.

**Isik et al., ICLR 2025** ([arXiv:2402.04177](https://arxiv.org/abs/2402.04177)) already
did BLEU/COMET vs. downstream cross-entropy on real MT across pretraining data size, and
found the metrics *diverge*: translation scores *"fluctuate or get worse"* while
cross-entropy *"monotonically improves."*

### The one that takes the headline — 3c

**Körner, Müller-Eberstein, Korhonen & Plank, "When Meanings Meet," EACL 2026 Main,
pp. 3149–3169** ([`2026.eacl-long.145`](https://aclanthology.org/2026.eacl-long.145/) ·
[arXiv:2601.22851](https://arxiv.org/abs/2601.22851)). Verified directly.

Verbatim from the abstract: *"in contrast to prior work, our fine-grained manual analysis
reveals that **some apparent gains in translation quality reflect shifts in behavior** —
like selecting senses for polysemous words or translating instead of copying cross-lingual
homographs — **rather than improved translation ability**."*

That is a published adjudication, on multilingual training checkpoints, with a *causal*
method (activation patching over 26 EuroLLM checkpoints, 11 languages), that apparent
gains in multilingual training curves are partly measurement artifacts.

**Be precise about what this does and does not close.** It is **not** the Schaeffer test —
there is no continuous/discontinuous metric pairing, and the mechanism is construct
validity (the metric measures a different behaviour than claimed) rather than metric
discontinuity (the metric thresholds a smooth quantity). Those are genuinely different
failure modes. But at the level a PI or a reviewer will hear it — *"someone already showed
that apparent multilingual training gains are measurement artifacts, causally, at EACL
2026, five months before this plan was written"* — the headline is taken. The plan's §7
does not cite this paper at all.

### The framing itself is contested

This matters, because "we adjudicate the mirage" is worth less if the mirage is already a
live dispute rather than a standing assumption:

- Schaeffer et al. themselves conceded ground ([arXiv:2406.04391](https://arxiv.org/abs/2406.04391)): *"Du et al. (2024) note that for many tasks, emergence remains despite the use of continuous metrics. Additionally, discontinuous metrics have been argued to often be the most reflective of real-world usefulness."*
- **Kangaslahti, Rosenfeld & Saphra, "Hidden Breakthroughs in Language Model Training," ICLR 2026** ([arXiv:2506.15872](https://arxiv.org/abs/2506.15872)) argues the *inverse*: breakthroughs *"occur frequently throughout training, but they are obscured by a loss metric that collapses all variation into a single scalar."* **A smooth continuous curve is not evidence of no transition.** This directly attacks the plan's outcome table, which reads "lag vanishes → mirage."
- Michaud et al. (NeurIPS 2023, quantization model); Berti et al. survey; Lu et al. (ACL 2024, ICL explanation) all offer competing accounts.

**Status: OPEN on the strict method, PARTLY CLOSED on the headline.**

---

## 4. Single-run confirmation — and a finding the plan can use

### Wang, Minervini & Ponti, Findings of ACL 2024 — confirmed, with caveats that cut both ways

[`2024.findings-acl.724`](https://aclanthology.org/2024.findings-acl.724/) · [arXiv:2406.13229](https://arxiv.org/abs/2406.13229)

**"Single-run, no error bars" — CONFIRMED without qualification.** Three BLOOM sizes
(560m/1b1/1b7), **18 checkpoints total** (6/8/4), essentially uniform 100k-step spacing,
one BigScience pretraining run each. Zero seed replication, zero error bars, zero CIs, zero
random-init null, zero changepoint test, zero multiplicity correction. The word `seed`
appears once, describing someone else's work. The only inferential statistics are 12
uncorrected Pearson p-values, six of which are computed on **pseudo-replicated**
checkpoint × language cells (n = 28–99 cells generated by 4–8 independent models).

Method: Stańczak et al.'s latent-variable intrinsic probe, **k = 50 informative neurons**
per (language, feature, layer), overlap = |C\*₁ ∩ C\*₂| / 50.

The non-monotonic finding, verbatim: *"a dramatic drop of overlap rates in two model
scales, which occurs at around 600k global steps for BLOOM560m and a bit earlier at 400k
steps for BLOOM1b1."* Their own caveat: *"While this phenomenon may be an artefact due to
the variance of overlap rates or **an error in checkpointing**, we remark that similar
drops were also observed in encoder-only multilingual LMs."* They rebut the artefact
hypothesis **only by analogy** — never by measurement, and they never report pretraining
loss at the anomalous checkpoints, which is the cheap decisive test.

**Three things recomputed from their own Appendix B that are not in the paper.** Treat as
this map's inference, not established fact, but they are checkable:

1. **The "drop" sits exactly at chance.** For two random k=50 subsets of d dimensions, expected overlap is k/d: 0.049 (560m, d=1024), 0.033 (1b1, d=1536), 0.024 (1b7, d=2048). Observed at the drop: 0.040 and 0.033. **The paper never prints the chance baseline.** Healthy checkpoints sit at only 2–5× chance.
2. **Cross-scale comparison is confounded by d.** Because chance = k/d, raw overlap is not comparable across model sizes. Chance-normalized, 560m is the *lowest* of the three at peak (3.50× vs 4.75× and 4.02×) — **reversing** the paper's stated conclusion that "the smallest model shows the highest overlap".
3. **The scaling-law claim rests on unmatched windows.** BLOOM-1b7 is observed only over 50k–200k steps; the drops occur at 400k and 600k. Over the window where 1b7 *was* measured, 560m and 1b1 are also monotonic.

Also unflagged: all downstream numbers are **4-bit QLoRA** measurements. FP16 was never scored.

**What this leaves open, concretely:** whether the drop is a property of multilingual
training or of that one BLOOM run; whether it is a corrupted checkpoint (their footnote 3
documents that the same release contains duplicate checkpoints); whether it is
cross-lingual at all, since **English collapses too** (XNLI 81.9 → 43.3), which makes it
look like general training instability rather than alignment-specific degradation.

### No follow-up has closed it

45 citing papers checked. Nobody has replicated or refuted the drop. The two components
exist separately — multi-seed controlled from-scratch training (**Cosma,
[arXiv:2605.26683](https://arxiv.org/abs/2605.26683)**, 700 runs, 4L/256d, *"three seeds
for the dataset creation and three seeds for the model training"* — the right design, but
it reports no error bars despite having them, and studies synthetic languages) and metric
adjudication (Körner et al. EACL 2026) — but nobody has combined them.

---

## 5. Claimed gap (a): "no accepted statistical test for a phase transition in a training curve"

### FALSE as written. Do not put this sentence in a proposal.

Accepted tests for a break in a curve are 30–65 years old and standard:

| Test | Source |
|---|---|
| Chow test (known break date) | Chow 1960 |
| sup-Wald / sup-LM (unknown break) | Andrews 1993 *(exact journal/pages UNVERIFIED — secondary sources only)* |
| Multiple structural change + sequential ℓ vs ℓ+1 test | Bai & Perron 1998 (Econometrica), 2003 ([J. Appl. Econometrics](https://onlinelibrary.wiley.com/doi/10.1002/jae.659)); R package `mbreaks` |
| CUSUM / CUSUM-SQ | Brown, Durbin & Evans |
| Segmented regression + **explicit breakpoint hypothesis tests** | Muggeo 2003; R `segmented`, with [`davies.test`](https://rdrr.io/cran/segmented/man/davies.test.html) and [`pscore.test`](https://rdrr.io/cran/segmented/man/pscore.test.html) |
| **Valid post-detection inference** | Jewell, Fearnhead & Witten, JRSS-B 84(4):1082–1104, 2022 ([arXiv:1910.04291](https://arxiv.org/abs/1910.04291)); Hyun et al., Biometrics 2021; Carrington & Fearnhead 2025 |
| Regime-shift detection | Rodionov 2004 (STARS), with a published validation study |
| Phase transitions in physics | finite-size scaling, Binder cumulants; Carrasquilla & Melko 2017; van Nieuwenburg et al. 2017 |

**The defensible version, with all three qualifiers, is:**

> Standard structural-break tests and post-detection changepoint inference are accepted for
> breaks in curves generally, but none has been validated on neural-network training curves,
> and no published procedure gives a calibrated interval for the **difference in transition
> time between two measures of the same run**.

The third clause is genuinely true as far as these searches go, and is the only part worth
claiming. It is a small, honest gap.

### 5b. And ML already has two competitors

- **Hu, Chen, Saphra & Cho, "Latent State Models of Training Dynamics," TMLR 2023** ([arXiv:2308.09543](https://arxiv.org/abs/2308.09543)) — fits an **HMM over per-checkpoint metrics across many seeds**, with latent-state transitions as the phase structure. This is the closest existing statistical model of training-curve phases and it directly competes with what the plan proposes. Read it first.
- **Hoogland et al., "Loss Landscape Degeneracy and Stagewise Development in Transformers," TMLR** ([arXiv:2402.02364](https://arxiv.org/abs/2402.02364)) — local learning coefficient as a theory-grounded stage detector. A reviewer will ask "why not LLC?"

Everything else in ML detects transitions **by eye**: Olsson et al. (loss-curve bump),
Nanda et al. (hand-designed restricted/excluded loss, phases split by inspection), Chen et
al. 2309.07311, Li/Fan/Zhou 2506.21551 (threshold criterion, grokking end read off Fig. 2).

Two 2026 preprints overlap the exact question and should be checked before anything is
frozen — [arXiv:2605.08237](https://arxiv.org/abs/2605.08237) (Hankel-DMD residual with an
explicit FPR/lead-time/AUROC protocol, AUROC ≈ 0.93 — the only *validated* transition
detector on training curves found) and
[arXiv:2606.02378](https://arxiv.org/abs/2606.02378) ("mechanism dates behavior" across
15/15 model–task pairs, no seeds, no CIs). ⚠️ Both are unvetted single/dual-author
preprints.

### 5c. The proposed method is unsound, in five separable ways

**Nobody has published PELT or BOCPD on a neural-network training curve.** Being first here
is a liability, because:

1. **Model misspecification — fatal, not a caveat.** PELT's standard cost assumes a **piecewise-constant mean with independent Gaussian noise** (Killick, Fearnhead & Eckley 2012, [arXiv:1101.1438](https://arxiv.org/pdf/1101.1438)). A training curve is a smooth monotone sigmoid, which **contains no changepoint at all** — the estimand PELT is consistent for does not exist in the data. Baranowski, Chen & Fryzlewicz (JRSS-B 2019) demonstrate exactly this (their Fig. 1): fitting a piecewise-constant model to a *continuous piecewise-linear* signal produces a spurious changepoint at the wrong location. A sigmoid is worse. What PELT returns is a staircase approximation whose step count and placement are set by the penalty and the SNR.
2. **Autocorrelation → overdetection.** Romano, Rigaill, Runge & Fearnhead, JASA 117(540), 2022 ([arXiv:2005.01379](https://arxiv.org/pdf/2005.01379)): autocorrelated noise causes **overestimation of the number of changes**. Checkpoint-evaluated metrics on a fixed eval set are strongly autocorrelated.
3. **Min-max normalization is an active confound, not neutral preprocessing.** It makes both endpoints data-dependent random variables (breaking the cost function's likelihood) *and* rescales each curve's noise by 1/(max−min), so a single fixed penalty means **a different effective detection threshold for every curve**. Since the behavioural and mechanistic measures will have different dynamic ranges, **this alone can manufacture a systematic nonzero Δt** — the exact result the plan is trying to establish.
4. **Bootstrapping a changepoint location is known to be inconsistent.** Seijo & Sen, Annals of Statistics 2011 ([arXiv:1101.1032](https://arxiv.org/abs/1101.1032)): *"standard bootstrap procedures in regression fail to provide valid confidence intervals for the change-point"*; Cattaneo, Jansson & Nagasawa, Econometrica 2020, generalize to all cube-root/argmax estimators. *(This kills the within-run bootstrap. The across-seed bootstrap is a different object and survives this objection — but not the next one.)*
5. **n = 5 caps the protocol at a sign test.** Resampling 5 seeds with replacement yields only **C(9,4) = 126 distinct multisets**, so the bootstrap distribution of mean Δt has at most 126 atoms — raising B from 10³ to 10⁶ adds literally zero information. The support is bounded by [min Δtᵢ, max Δtᵢ], so **if all 5 seeds agree in sign the percentile CI excludes zero with probability 1, by construction, regardless of how noisy each estimate is.** The decision rule *is* "did all 5 seeds agree", whose best attainable one-sided p is 2⁻⁵ = **0.031**. Even on clean normal data the percentile interval at n=5 is ~37% too narrow (≈85% actual coverage for nominal 95%).

**And the confound no statistics fixes.** Δt compares two *different instruments* with
different noise and smoothness. Any thresholded detector fires earlier on the higher-SNR,
smoother curve — so a nonzero Δt is confounded with the SNR gap between the mechanistic and
behavioural measures. **None of the ML work above runs a simultaneity null** (two curves
known to transition at the same time, corrupted with each measure's empirical noise, showing
the pipeline recovers Δt ≈ 0 with correct coverage). That calibration study is the genuine
methodological contribution available here — and it is a validation exercise, not a new test.

---

## 6. Claimed gap (b): "no interpretability paper carries a multiple-comparison protocol"

### FALSE. Verified by full-text grep of 36 papers.

**Counterexamples, worst first:**

**Gröger, Wen & Brbić, "Revisiting the Platonic Representation Hypothesis: An Aristotelian
View,"** [arXiv:2602.14486](https://arxiv.org/abs/2602.14486) (ICML 2026 per the arXiv
page), code at `mlbio-epfl/aristotelian`. This is the damaging one: a permutation-null
calibrated test over **exactly the plan's measures** — linear/kernel CKA, RV, CCA/SVCCA/PWCCA,
mutual-kNN, CKNNA, RSA, Procrustes. It derives the null baselines analytically (CKA:
`E_{H0}[‖C̃‖²_F] = d_x·d_y/(n−1)`; mutual-kNN: `k/(n−1)`), computes right-tail empirical
p-values, **names the layer-scan family explicitly** (*"the number of layer pairs searched
… an instance of the classical multiple comparisons problem (Benjamini & Hochberg, 1995;
Bonferroni, 1936)"*), and applies **Holm and BH-FDR**. Effect: calibration drops CKA's
correlation with model ranking from 0.86 → 0.45, and *"calibrated CKA shows no systematic
increase with model size."*

**If the plan's statistical-protocol novelty is "a calibrated null for representational
similarity", that contribution is spent.**

Others: **Shi et al., NeurIPS 2024** ([arXiv:2410.13032](https://arxiv.org/abs/2410.13032))
— Bonferroni over circuit edges, squarely mechanistic interp, cited by Sharkey et al.'s
*Open Problems* as *"a suite of formal statistical hypothesis tests for circuit efficacy."*
**Zhu, Zhang & Wang, ICML 2024** ([arXiv:2402.18496](https://arxiv.org/abs/2402.18496)) —
Bonferroni FWER over the top-10 attention heads. **Enkhbayar 2025**
([arXiv:2511.11711](https://arxiv.org/abs/2511.11711)) — Model-X knockoff FDR over 512 SAE
latents in Pythia-70M, with a section titled *"The Multiple Testing Problem in
Interpretability."* And the entire brain–LM literature does this routinely: Toneva & Wehbe
(NeurIPS 2019, BH-FDR + Barber–Candès knockoffs), Caucheteux & King (Comms Bio 2022),
Goldstein et al. (Nat. Neuro. 2022, FDR at q=0.01 over electrodes × lags).

### Where it *is* empty — and the claim that survives

Full-text grep returned **zero** correction terms in: Belinkov's *Probing Classifiers*
(CL 2022 — the field's canonical methodological review), Hewitt & Liang, Pimentel et al.,
Voita & Titov, ACDC, Miller et al., Sharkey et al., Dalvi et al., Antverg & Belinkov, Tang
et al. (LAPE), Pythia, Olsson et al., Crosscoding Through Time.

**The narrow claim that survives:**

> In the interpretability literature on **training dynamics** — checkpoint-scan studies of
> when representations or circuits form — we found no paper that controls family-wise error
> or FDR over the checkpoint × layer scan; the field's standard is an uncorrected null
> baseline (control tasks, random init) rather than error-rate control.

Say "we found no instance in N papers surveyed", not "no paper exists."

**Keep the distinction a PI will catch:** control tasks (Hewitt & Liang), MDL baselines
(Voita & Titov), and randomization sanity checks (Adebayo et al. 2018) are **nulls**, not
**multiple-comparison protocols**. They calibrate one statistic; they do not control error
rate over a family.

**Useful ammunition rather than a threat:** Dror et al., ACL 2018
([`P18-1128`](https://aclanthology.org/P18-1128/)) audits the field and reports *"of 110
papers that used multiple datasets only 3 corrected for multiplicity."* That is a citable,
quantified version of the gap the plan actually wants. The MC methodology itself is Dror et
al., TACL 2017 ([`Q17-1033`](https://aclanthology.org/Q17-1033/)).

---

## 7. H3a — cross-lingual alignment dynamics across training objectives

**Nobody has done it. The two axes exist separately and cleanly, and the intersection is
empty.**

### But the endpoint version exists, at almost exactly the planned architecture

**Li, Ji, Mickus, Segonne & Tiedemann, "A Comparison of Language Modeling and Translation as
Multilingual Pretraining Objectives," EMNLP 2024** ([arXiv:2407.15489](https://arxiv.org/abs/2407.15489)).

| | Li et al. 2024 | `tier1_plan.md` §3.1 |
|---|---|---|
| Layers | 12 | 6 |
| d_model | **512** | **512** |
| Heads | **8** | **8** |
| FFN | **2048** | **2048** |

They train **five objectives from scratch on matched data** — 2-LM (denoising
encoder-decoder), 2-MT (encoder-decoder translation), MLM, CLM, TLM — i.e. encoder-only vs.
decoder-only vs. encoder-decoder **and** MLM vs. causal vs. translation. That is the H3a
contrast. 6 languages, identical corpus, 600k steps.

- **Seeds, verbatim:** *"Owing to the computational requirements, we only train one seed for each of the five types of models considered."* (The five seeds they report are probing/fine-tuning heads.)
- **Dynamics: none.** Full-text grep: `checkpoint` 0 hits, `dynamic` 0 hits, `training step` 0 hits.
- **No representational alignment measure at all** — no CKA, no PWCCA, no retrieval. Downstream probing/fine-tuning only.
- Headline: *"the architecture dictates which pretraining objective is optimal."*

**Two consequences.** First, the proposal is exposed to *"this is Li et al. with
checkpoints"* unless the differentiator — trajectory, >1 pretraining seed, representational
measures — is stated against this paper **by name**. Second, and more useful: Li et al.
concede their own contrast may be noise — *"the relative ranks of the three single-stack
models fluctuate much more … owing to no little extent to the oftentimes momentous variation
across seeds for single-stack models."* That converts the multi-seed requirement from a
methodological nicety into a stated open problem left by the closest prior work. **It is
also a concrete prior that the effect at 36M may not clear the noise floor.**

### The dynamics axis, all single-objective

| Work | Models | Objective | Runs/seeds | Checkpoints |
|---|---|---|---|---|
| Blevins, Gonen & Zettlemoyer, EMNLP 2022 ([`2022.emnlp-main.234`](https://aclanthology.org/2022.emnlp-main.234/)) | own XLM-R replica, 270M, 94 langs | MLM only | **1** pretraining run (5 *probe* reruns) | **39**, public at `nlp.cs.washington.edu/xlmr-across-time` |
| Bayazit, Mueller & Bosselut, **ACL 2026** pp. 1353–1377 ([arXiv:2509.05291](https://arxiv.org/abs/2509.05291)) | Pythia/OLMo/BLOOM-1B | causal LM only | 3 *crosscoder* seeds, 1 pretraining run each | **4 per model** |
| Harrasse et al. ([arXiv:2511.10840](https://arxiv.org/abs/2511.10840)) | 8 purpose-trained, 68.5M & 177.6M | causal LM only | not stated | final models only |
| Körner et al., EACL 2026 (2601.22851) | EuroLLM-1.7B | causal LM only | 1 | 26 |
| Leino & Tiedemann ([arXiv:2603.29026](https://arxiv.org/abs/2603.29026)) | own 1.4B, EN–FI, 200B tokens | causal LM only | 1 per config | PWCCA across training |

Bayazit et al. state the reason the intersection is empty, in as many words: *"We compare
checkpoints from the same training run so that representational changes can be attributed to
pretraining dynamics rather than to differences in tokenizer, data mixture, or objective."*
**The field treats cross-objective dynamics comparison as a confound to be designed out.**

### Why unclaimed: mostly (c), with a real (a)

**(a) Hard, two ways.** Compute — the one group that tried could not afford a second seed
for 5 objectives; adding checkpoints × seeds makes it a 3-way product. And
**cross-objective measure comparability is the sharper problem**: MLM has no next-token
distribution, so the JSD measures the plan builds on (§4 Stage 2) **do not port to an
encoder arm at all**. Losses are not comparable across MLM/CLM/MT, so "the same point in
training" must be defined by tokens or FLOPs, never by loss. The objective-invariant measure
set is small — CKA, PWCCA, mutual-nearest-neighbor over a shared parallel probe set, effective
rank — and anything defined over the output distribution is out. This is the same point
ADR 0001 makes; it is a measurement problem, not a formatting detail.

**(b) Not uninteresting.** Li et al.'s endpoint answer is architecture-conditional, which
makes the trajectory question *more* interesting: a rank flip at the endpoint says nothing
about whether the onset differs. And two 2026 papers report alignment emerging **early**
(PWCCA rising by ~5k steps; concept spaces *"emerge early and continue to refine"*).

**(c) Unclaimed.** Different method stacks, different venues, and in the one case where both
exist in the same lab (Helsinki: Li et al. and Leino & Tiedemann), different papers two
years apart.

### The regularization confound

**No prior art either way.** Six targeted searches returned nothing on weight decay or
dropout as determinants of cross-lingual alignment formation. Two adjacent results
(snippet-level, **UNVERIFIED** in full text): [arXiv:2505.13090](https://arxiv.org/abs/2505.13090)
finds weight decay has minimal effect on MT fine-tuning and *harms* unsupervised directions;
[arXiv:2602.11137](https://arxiv.org/abs/2602.11137) reports higher weight decay *"encourages
linearly separable representations"* — i.e. it plausibly moves **exactly the quantity a linear
probe or CKA reads**, so the confound is not benign.

**Practical consequence:** because nothing is citable, the confound cannot be dismissed by
reference — it must be closed by design. `tier1_plan.md:102` already fixes AdamW, weight
decay 0.01. If that is applied **identically to all arms**, the confound is closed by
construction and the writeup should say so rather than conceding it. It only reappears if each
arm inherits its namesake's recipe (mBERT-style dropout, NLLB-style `--weight-decay 0.0`) —
which is exactly what the proposal PDF p. 8 proposes.

---

## 8. Super weights across training

### The angle is not DOA on scale — it is squeezed on novelty

**Q: has anyone tracked formation across checkpoints? Substantially yes, in four papers.**

**Gallego-Feliciano et al., "Hidden Dynamics of Massive Activations in Transformer
Training,"** [arXiv:2508.03616](https://arxiv.org/abs/2508.03616) (v2, Feb 2026). Opening
sentence: *"We present the **first comprehensive analysis of massive activation development
throughout transformer training**, using the Pythia model family as our testbed."* Nine
models, **Pythia-14M through 12B, ~154 checkpoints each**, fitting emergence with an
exponentially-modulated logarithmic function and predicting its parameters from architecture
alone. Limits: activations only (not weights), single seed, and their own limitations section
names encoder-decoder as untested. arXiv preprint, not peer-reviewed.

**Kovaleva, Kulshreshtha, Rogers & Rumshisky, "BERT Busters," Findings ACL-IJCNLP 2021**
([`2021.findings-acl.300`](https://aclanthology.org/2021.findings-acl.300/)). §5.2 is already
a from-scratch checkpoint study at almost the target config: *"a randomly initialized
BERT-medium configuration that has **8 layers with the hidden dimensionality of 512 units** …
We save checkpoints every 2000 steps … both scaling factors and biases begin to diverge from
their initialization values quite early (**after approximately 50k steps**)."*

**Gu et al., "When Attention Sink Emerges in Language Models," ICLR 2025 Spotlight**
([arXiv:2410.10781](https://arxiv.org/abs/2410.10781)). Pretrain LLaMA-style models at
**d=768, L=10, ≈60M params**, track the sink metric over training (*"emerges … between 1k and
2k steps"*), then **ablate LR, weight decay, batch size, data amount, loss function,
positional encoding, pre/post-norm, and softmax→sigmoid.** This is the paper that most
pre-empts a "what causes formation" design.

Also: Macocco et al., BlackboxNLP 2025 ([arXiv:2503.21718](https://arxiv.org/abs/2503.21718)),
*"A significant number of ODs appears around steps 3000/4000"*; Ding
([arXiv:2605.18898](https://arxiv.org/abs/2605.18898)), the only **weight-level** trajectory —
Pythia-70m, 14 log-spaced checkpoints, *"By step 5,000 an isolated outlier emerges at
|w|≈1.0"* — but magnitude only, **no causal deletion test**.

**Corrections to the premises in the task brief:**
- Yu et al. 2411.07191 authors are **Mengxia Yu, De Wang, Qi Shan, Colorado J Reed, Alvin Wan** — not "Yu, Cao, Kim". No checkpoint analysis.
- Sun et al. 2402.17762 is **COLM 2024**. Their "emerge" is **depth-wise, not temporal** (*"In LLaMA2-7B, massive activations first appear in layer 2"*). Smallest model examined: GPT-2, 124M.
- **Dettmers et al.'s "~6.7B" is about scale, not training, and about *extent*, not existence.** Their smallest model (125M) already has outliers. They walk the phase-transition framing back themselves: *"when measured by perplexity, the emergence … can be seen as emerging smoothly … **This indicates that there is nothing sudden about emergence**."* Do not cite 6.7B as a floor.

**What survives:** causal super-weight (deletion-criticality) formation, never tracked;
encoder-only and encoder-decoder arms (2508.03616 names them as future work, and it is the
H3a contrast anyway); seeds as the unit of independence (nobody); any link to cross-lingual
alignment (empty).

### Q: do they exist at 36M/6L/512d? Yes, decisively

**Pythia-14M — 6 layers, hidden 128, 4 heads** (verified against the HF config). Both
phenomena documented there: *"attention sink emerges in small LMs, even in Pythia-14M"*
(Gu et al. §3.4); *"in Pythia-14M … the top activation magnitudes at layer 3 clearly dominate
all others"* (2508.03616). **Smaller than 36M/6L/512d in every dimension.**

With a *causal* test, Kovaleva Table 2: **BERT-small (4L/512d, 29.1M)** — zeroing 16 outlier
LayerNorm weights moves WikiText CE 2.26 → 2.93 (vs 2.28 for 16 random). **BERT-medium
(8L/512d, 41.7M)** — 2.00 → 3.21 (vs 2.04 random). The target sits between these.

**Two caveats.** Effect size shrinks monotonically with scale (BERT-large 2.28→5.49, base
2.30→4.53, medium 2.00→3.21, small 2.26→2.93) — at 36M the phenomenon is present but *mild*,
so power and CIs matter more than they would at 7B. And **the binding constraint is token
budget and architecture, not parameters**: Gu et al. show sinks fail to emerge at 50–100M
training tokens (2B is comfortably above this), and vanish entirely under sigmoid attention
without normalization. **No paper reports absence below a parameter threshold.**

---

## 9. Threats not covered by the six questions

Found by the adversarial sweep; all verified to source.

**9.1 — The seed count is mis-specified, not merely small.**
**Zhao, Qin, Alvarez-Melis, Kakade & Saphra, "Random Scaling of Emergent Capabilities,"
ICML 2025** ([arXiv:2502.17356](https://arxiv.org/abs/2502.17356)). They ran **250 seeds**
and **200 seeds** and found performance is **bimodally distributed across seeds** — *"different
random seeds can produce either highly linear or highly emergent scaling trends"* — and
critically, *"random variation still leads to bimodal performance distributions, **even using
this continuous performance metric**."* A 5-seed bootstrap on a bimodal distribution does not
produce a valid CI. It also collapses the plan's outcome table, which admits only "real" or
"mirage": the true answer may be a third thing that n=5 cannot see.

Reference points for what the field uses: **MultiBERTs** (Sellam et al., ICLR 2022) — **25
pretraining seeds** varying init *and* data order, plus the **Multi-Bootstrap**, which is the
protocol §5 is a weaker instance of. **PolyPythias** (van der Wal et al., ICLR 2025,
[arXiv:2503.09543](https://arxiv.org/abs/2503.09543)) — **9 seeds × 5 sizes (14M–410M,
bracketing 36M), ~7,000 released checkpoints**, explicitly studying *"emergence of training
phases"*, concluding *"highly consistent training dynamics across both model sizes and initial
conditions."* That last finding is a vise: if dynamics are seed-stable, 5 seeds buy little over
Blevins; if they are not, 5 is below what the field's own reference study needed.

**9.2 — EN→FR may be saturated at the planned budget.**
**Deshpande, Talukdar & Narasimhan, NAACL 2022**
([`2022.naacl-main.264`](https://aclanthology.org/2022.naacl-main.264/)) already ran
**8 layers, 8 heads, hidden 512** on **English/French** with UD POS *and* embedding alignment
at **~100M tokens per language**, and got **zero-shot EN→FR POS of 97.2** with zero subword
overlap. The plan budgets **1B tokens per language** — 10× that. If the headline behavioural
metric is at ceiling for most of training, 60 checkpoints × 5 seeds measures noise near
saturation. This is a design problem, not a novelty problem, and it is fixable — but it is
not currently addressed anywhere in `tier1_plan.md`.

Relatedly, **XTREME** reports mBERT zero-shot POS at **fr 84.2 / tr 68.5**: EN–FR and EN–TR
are different regimes, so the three-language design is unbalanced rather than two samples of
one phenomenon. *(XTREME numbers single-sourced from ar5iv-rendered tables.)*

**9.3 — Every planned similarity measure has a published failure mode that generates H1 for free.**
- **Timkey & van Schijndel, EMNLP 2021** ([`2021.emnlp-main.372`](https://aclanthology.org/2021.emnlp-main.372/)): one to three dimensions carry 76–99% of expected cosine similarity, and anisotropy/norm growth **rises monotonically during training** — manufacturing the "smooth mechanistic rise" H1 predicts, with no shared space required.
- **Ding, Denain & Steinhardt, NeurIPS 2021** ([arXiv:2108.01661](https://arxiv.org/abs/2108.01661)): *"CKA requires 97% of a representation's principal components to be deleted for the dissimilarity to be detectable"*, and PWCCA fails specificity — *"random initialization affects this distance more than large changes in layer depth."* The two headline measures fail opposite halves of the only published sensitivity/specificity test.
- **Davari et al., ICLR 2023** ([arXiv:2210.16156](https://arxiv.org/abs/2210.16156)): translating a subset of points drives CKA *"to 0"* while an SVM still gets *">90% accuracy"*.
- **Idris, Mitra & Eiselen** ([arXiv:2601.03168](https://arxiv.org/abs/2601.03168)): 816 transfer runs, 272 language pairs — *"CKA achieves significant correlation in only 2 of 9 conditions"*, and pooling **flips the sign**. A direct measurement of the similarity ⇒ transfer link, and it fails.
- **Del & Fishel, AACL 2022** ([`2022.aacl-main.15`](https://aclanthology.org/2022.aacl-main.15/)): *"assumptions of CKA/CCA align poorly with one of the motivating goals of cross-lingual learning analysis, i.e., explaining zero-shot cross-lingual transfer."*
- **Koepke et al.** ([arXiv:2604.18572](https://arxiv.org/abs/2604.18572)): mutual-kNN alignment falls from 0.135 to 0.008 when the gallery scales 1K → 15M — largely a probe-set-size artifact.

**9.4 — Even a clean positive result may not license the conclusion.**
**Hua, Yun & Pavlick, "mOthello," Findings of NAACL 2024**
([arXiv:2404.12444](https://arxiv.org/abs/2404.12444)) — at **8 layers / 512 hidden / 8
heads**, essentially the target architecture: *"models trained with naive multilingual
pretraining fail to learn a language-neutral representation"* and *"the learning of a
language-neutral space alone is **not sufficient** to facilitate cross-lingual transfer."*

**9.5 — Nanda is on the other side of the grokking framing.**
[arXiv:2301.05217](https://arxiv.org/abs/2301.05217) abstract, verbatim: *"grokking, **rather
than being a sudden shift**, arises from the gradual amplification of structured mechanisms
encoded in the weights"*; limitations: *"our progress metrics are specific to small networks
on one particular algorithmic task."* Citing Nanda to support a *phase transition* claim
inverts the paper's thesis. Also: Liu et al. (Omnigrok, ICLR 2023) — *"training with
constrained weight norm can almost eliminate grokking"*, and their only *language*
demonstration is a 2-layer LSTM on 1,000 IMDb examples with init scaled ×6. Prieto et al.
([arXiv:2501.04697](https://arxiv.org/abs/2501.04697)) — grokking delay is partly a
softmax/floating-point stability effect that disappears with StableMax or ⊥Grad. Kumar et al.
([arXiv:2310.06110](https://arxiv.org/abs/2310.06110)) — grokking reduces to a continuously
tunable output-scale parameter; no order parameter, no criticality. Murty et al. (ACL 2023),
the strongest NL grokking result, uses as its headline metric *"the fraction of seeds (out of
10) where generalization accuracy eventually crosses 80%"* — at fixed architecture, some seeds
grok and some never do.

**9.6 — The counter-evidence, stated fairly.**
**Muckatira, Shivagunde, Deshpande & Rumshisky, "A Pre-Training Analogue of Grokking in
Language Models: Tracing Delayed Grammatical Generalization,"**
[arXiv:2606.00230](https://arxiv.org/abs/2606.00230) (29 May 2026). Verified: *"Across five
grammatical phenomena, we observe delayed generalization. Analyzing pre-training checkpoints
before and after generalization shows that grammatical concept vectors become more predictive
of grammatical acceptability and occupy a higher-dimensional subspace after generalization."*
Delayed generalization **does** appear in real single-pass pretraining at roughly the target
scale. This is the strongest evidence the premise is not dead. It is also monolingual, and it
is prior art for the "grokking analogue at ~36M" move.

**9.7 — Jian & Manning, the methodological anchor, is decoder-only by construction.**
Verified from the local PDF (`papers/EACL-2026-long-32.pdf` = EACL 2026 pp. 752–765,
[arXiv:2603.17475](https://arxiv.org/abs/2603.17475); Best Paper confirmed via the EACL
account, **not** flagged on the Anthology page).
- The "GPT-2" is **not OpenAI's** — it is Stanford CRFM Mistral's retrained GPT-2-small on OpenWebText (Karamcheti et al. 2021), 5 runs × 609 public checkpoints; the paper uses 450 of them.
- **Headline results are a single seed:** *"We report results for a single random seed for clarity, though all described behaviour was confirmed across three runs with different seeds."* All error bars are over items, never over seeds.
- Footnote 2 is load-bearing for ADR 0001: *"Smoothing is not required as **next-token distributions under autoregressive LMs do not contain true zeroes**."* The words "masked", "MLM", "bidirectional", "encoder" appear in the paper **only inside bibliography entries.** The authors give no guidance on porting the measures to an MLM.
- Onset detection is a per-step Mann–Whitney U at p < 0.001 plus a hard-coded heuristic (*"the step after which the average D_JS … is consistently at least 0.01 greater than the average of the first 30 steps"*). **No changepoint model, no multiplicity correction, no family size stated.** The plan's promised changepoint statistics are a genuine upgrade over the anchor — say so rather than inheriting credit.
- Their own open problem is the one Tier 1 targets: *"understanding how representations change to produce these behaviours is a core desideratum."* They also flag an unresolved **frequency confound** in acquisition order.

---

## 10. Verdict

**The metric-artifact adjudication is largely occupied. The statistical protocol is
occupied, unsound as specified, or both. Tier 1 as currently written is not worth doing.**

Taking the surviving-novelty claim in its two halves:

**The metric-artifact adjudication.** The strict Schaeffer pairing on cross-lingual
transfer is genuinely unrun (§3), and that negative is well-established — 733 + 3,737
citation-graph enumerations and three co-citation intersections. But the value of running
it has been eroded from three sides. Du et al. (NeurIPS 2024) already ran the exact metric
triple across checkpoints on a non-English benchmark and **emergence survived**, so the
prior on the interesting outcome is now against it. Körner et al. (EACL 2026) already
published a *causal* adjudication that apparent multilingual training gains partly reflect
behavioural shifts rather than capability — the same headline by a different route, five
months before the plan was written, and uncited in it. And Kangaslahti et al. (ICLR 2026)
establishes that a smooth continuous curve is **not** evidence of no transition, which
means the plan's "lag vanishes → mirage" outcome row does not follow.

**The statistical protocol.** Gap (a) is false as stated — accepted structural-break tests
are 30–65 years old — and the narrow true version ("no calibrated interval for the
*difference* in transition time between two measures") is a small claim, with two ML
competitors already in the space (Hu et al.'s HMM; Hoogland et al.'s LLC). Gap (b) is false
— Gröger et al. published permutation-null calibration with BH-FDR for **CKA and
mutual-kNN specifically**, the exact planned measures, in February 2026. And the proposed
machinery does not work: PELT on a smooth sigmoid is a model misspecification that returns
staircase artifacts; min-max normalization can manufacture a nonzero Δt from an SNR gap
alone; and the n=5 bootstrap reduces to a sign test capped at p = 0.031 one-sided / 0.0625 two-sided
(`method_landscape.md` §4.2 quotes the two-sided figure — same test, not a disagreement) —
against a quantity Zhao et al. (ICML 2025) showed to be **bimodal across seeds even under
continuous metrics**, which makes 5 seeds mis-specified rather than merely underpowered.

Add the design problem nobody has raised: at 1B tokens/language, EN→FR zero-shot POS is
plausibly saturated (Deshpande et al. got 97.2 at one-tenth the budget), so the primary
behavioural axis may be at ceiling for most of the trajectory being measured.

**What is genuinely, defensibly open**, in descending order:

1. **The simultaneity null** (§5c) — no published pipeline demonstrates it recovers Δt ≈ 0 with correct coverage on two curves known to transition together. This is the real methodological hole, and it is a validation study, not a new test.
2. **Causal super-weight (deletion-criticality) formation across training** (§8b) — magnitude has been tracked once, criticality never. Not DOA on scale: the phenomena are documented at Pythia-14M (6L/128d), smaller than the target in every dimension, and causally at BERT-small (4L/512d, 29.1M).
3. **Objective × dynamics, matched from-scratch** (§7) — genuinely empty, with Li et al. (EMNLP 2024) owning the endpoint version at 12L/512d/8h and conceding its own single-stack contrast may be seed noise.
4. **Error-rate control over a checkpoint × layer scan** (§6c) — the one form of the multiplicity claim that survives.
5. **Whether the Wang et al. non-monotonic drop is real or a corrupted checkpoint** (§4) — 45 citing papers, nobody has replicated or refuted it, and the authors themselves flag it as possibly artefactual.

Items 1, 3 and 5 are one coherent study. Item 2 is a different study that presupposes a
decoder and therefore presupposes the answer to ADR 0001's open question. Item 4 is a
paragraph in a methods section, not a contribution.

**None of that is what `tier1_plan.md` currently proposes to do**, and the gap between
"what is open" and "what is planned" is large enough that this is a rewrite, not an
amendment.

---

## 11. The strongest case against — stated at full strength

Written to be the argument a hostile reviewer makes, not a softened version.

**The composition is fully covered even though no single paper does the whole thing.** Three
papers between them own it: Blevins et al. (2022) owns cross-lingual checkpoint dynamics with
39 public checkpoints across 94 languages, and already found that *"the point in pretraining
when the model learns to transfer cross-lingually differs across language pairs"* — i.e. the
per-pair asynchrony the plan hopes to discover. Deshpande et al. (2022) owns the
8L/512d/8h EN–FR + UD-POS + embedding-alignment setup, including the alignment↔transfer
correlation. Körner et al. own from-scratch dense-checkpoint multilingual training (200
checkpoints, 1.7B, 9 languages, checkpoints to be released), the mechanism-before-behaviour
result, **and** the metric-artifact adjudication. A reviewer does not need one paper that does
everything; they need three that between them leave nothing.

**Both surviving contributions are spoken for.** The plan bet everything on "metric-artifact
adjudication and the statistical protocol" after conceding the phenomenon. The adjudication
was published at EACL 2026 with a causal method. The protocol is a weaker instance of
MultiBERTs' 2022 Multi-Bootstrap, its similarity-measure calibration was published by Gröger
et al. in February 2026, and its changepoint machinery is misspecified for smooth curves.

**The go/no-go gate cannot fire honestly.** §5's decision rule is "the bootstrap CI on Δt
excludes zero." Between Keung et al.'s 15-point run-to-run variance in zero-shot cross-lingual
evaluation, Blevins et al.'s finding that transfer onset ordering is inconsistent across
language pairs, and Zhao et al.'s bimodal across-seed distribution, n=5 has no realistic path
to a clean interval — and if it produces one, that is by construction, since with 5 seeds
agreeing in sign the percentile CI excludes zero with probability 1.

**Every measure has a published failure mode that produces H1 for free.** Anisotropy and
rogue-dimension norm growth rise monotonically through training; biased CKA inflates in the
low-sample/high-dimension regime; CKA's null scales as O(d/n). "Smooth mechanistic rise
preceding a thresholded behavioural jump" is the *predicted signature of the confounds*, with
no shared cross-lingual space required. A positive result is therefore weakly diagnostic and
a negative result is uninterpretable without the calibration work that is itself the only
real contribution.

**And the inference the whole design rests on has already been severed.** mOthello, at the
target's own architecture (8L/512d/8h), found that a language-neutral representation is *not
sufficient* for cross-lingual transfer. So even a clean, well-calibrated, multi-seed
demonstration that a shared space forms at step *t* does not license any claim about when
transfer becomes possible — which is what H1 is about.

**The honest counterweight.** Muckatira et al. (May 2026) do find delayed grammatical
generalization in real single-pass pretraining at 35M–130M, so the premise that
grokking-like dynamics exist at this scale is not dead. And the Wang et al. non-monotonic
drop remains genuinely unexplained after 45 citing papers, with a cheap decisive test (report
pretraining loss at the anomalous checkpoints) that nobody has run. There is a real study in
here. It is not this one.

---

## Provenance and confidence

Ten parallel read-only agents, 2026-08-06. Papers read in full (PDF or arXiv HTML) wherever a
claim turns on detail; full-text greps used for all seed/variance/multiplicity claims. Four
load-bearing citations re-verified directly outside the agents: `2026.eacl-long.145` /
arXiv:2601.22851 (authorship and abstract), arXiv:2606.00230, arXiv:2407.15489, and the
`2026.eacl-long.145` abstract text.

**Known coverage gaps.** The arXiv full-text search API returned HTTP 429 on all attempts in
the mirage sweep and Google Scholar was unreachable, so §3's negative rests on the Semantic
Scholar citation graph (656/733 abstracts covered) and co-citation intersections rather than
full-text search. Two agents exhausted their web-search budgets; both had already returned
their primary findings from the citation-graph and full-text-read channels. Several 2026
preprints cited here (2605.08237, 2606.02378, 2508.03616, 2605.18898, 2606.16920, 2607.05104)
are unrefereed and their quality is unvetted — they are cited as scoop risk, not as authority.
Items marked **UNVERIFIED** in the body were not confirmed to primary source and must be
checked before appearing in any writeup.
