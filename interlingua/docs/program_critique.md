# Adversarial audit of *Does the Interlingua Grok?*

**Produced 2026-08-06** by nine independent agents — eight attackers, one per load-bearing
premise, blind to each other, each instructed to break its premise and to default to
"unsupported"; plus one steelman whose only job was to make the strongest case *for* the
program. Sources were read as full texts, not abstracts. Load-bearing claims were
re-verified against local PDFs by hand where possible; those are marked ✓**self-verified**.

**Scope.** This audits the *premises*. Prior art and method tooling are handled elsewhere
(`prior_work_map.md`, `method_landscape.md`) and are deliberately out of scope here. No
experiments are designed and no alternative direction is proposed.

---

## Verdict

**Salvageable, but not as written — and the salvage is a much smaller project than the one
proposed.** The program has one genuinely good idea in it, and the idea is not grokking: every
published study of cross-lingual alignment dynamics has exactly **one seed**, several authors say
so in print, and at least one central claim (mid-training alignment degradation) is defended in
its own paper with *"may be an artefact due to the variance of overlap rates or an error in
checkpointing."* A seed-controlled replication settles that, and nothing else can. The
companion idea — measure the behavioral axis continuously — is real but **less novel than the
plan believes**: Ai2 already cites Schaeffer and ships bits-per-byte as its in-loop decision
metric, which is the same metric Tier 1 proposes as its novel control. Everything built on top
of these fails: the
grokking frame imports causal machinery (weight decay shedding memorizing components) whose
preconditions are absent from single-epoch pretraining and whose anchoring citation
contradicts its own numbers; "the interlingua" is not one object with one onset, and the
measures proposed to time it are documented to disagree with each other on identical
representations; H3b's capacity-pressure-as-weight-decay claim is not identifiable at fixed
capacity even in principle; and the Δt statistic as operationalized manufactures a positive
lag of the exact sign H1 predicts, at a 93–100% rate, from two curves with *no* true lag at
all. That last one is not a framing problem — it means the headline number would be an
artifact of the estimator. Two of these were established by direct measurement rather than by
argument: the Δt artifact by simulation, and the JSD measure — the plan's "cheapest real
signal" — by running the proposal's own contrast on a real model, where it returns
**+0.011 bits for EN–FR and the *wrong sign* for both Turkish pairs**, with the surviving
English–French effect carried by the comma and by English auxiliaries rather than by any French
token. Against this, the strongest fact in the program's favour is solid
and survived attack: 6-layer bilingual models *below* the proposed parameter budget do show
real structural cross-lingual transfer. So the behavioral axis probably exists. But every
such measurement is encoder-MLM with full fine-tuning, and the plan specifies a decoder with
frozen probes, for which no comparable evidence exists at any scale. **The single highest-value
action is a few-GPU-hour pilot establishing that the behavioral axis moves at all in the
actual architecture, before any of the fifteen runs are committed.** If it moves, a
well-powered measurement study is worth your months. If it does not, the project has no
dependent variable and you will have found that out for the price of an afternoon rather
than a semester.

Two things to carry into the PI conversation alongside that. First, **no consumer was found** —
nine model families' training reports read in full, zero take a representation measure as input
to any decision, and Tier 1 contains no low-resource language and no translation task, so the
NLLB motivation is four inferential steps away. The one surviving constituency is *methodological*:
anyone publishing a CKA-based multilingual result. Build for them. Second, the realistic build is
**14–20 part-time weeks, not 6–7**, and the only deliverable guaranteed to exist if the headline
fails is the released checkpoint suite — which is the proposal's Contribution 5 and should be
its Contribution 1.

---

## Load-bearing assumptions

Ranked by how much collapses if the assumption fails. Support levels:
**well-supported** / **thinly-supported** / **unsupported** / **contradicted**.

| # | Assumption | What rests on it | Support | Evidence |
|---|---|---|---|---|
| **A1** | A ~36M model on ~2B tokens shows measurable zero-shot cross-lingual transfer, so the behavioral axis moves | **Everything.** No behavioral axis ⇒ no t_behavioral ⇒ Δt undefined ⇒ H1–H4 all untestable | **thinly-supported, split by language and architecture** | ✓**self-verified** Karthikeyan et al., ICLR 2020, Table 6: 6L/13.37M bilingual model → **56.2** zero-shot XNLI Russian vs **33.3** chance, on Fake-English (zero lexical overlap). 6L/8.40M → 49.7. Dufter & Schütze EMNLP 2020: 3-language 1M-param BERT, multilinguality .70 vs **.00** untrained. **But**: all encoder-MLM, all *fine-tuned* not frozen-probe, 2M steps over ~1GB/lang. XTREME zero-shot POS: mBERT **tr 68.5**, XLM-R 76.3 at 178–270M. Computed majority-class + surface-rule baseline on **TR-BOUN = 50.0%** ⇒ only ~18pp of headroom at *mBERT* scale. **No decoder evidence found at this scale by any agent.** ⚠️ **Converges with the independent 2026-08-06 method survey**, which found that *neither* of `tier1_plan.md` §3.1's two stated reasons for dropping the encoder arms holds — an MLM at `[MASK]` gives the JSD measures what they need, and TL 3.6.0 ships an NLLB adapter. So two audits reach the encoder arm by different routes: this one because the *evidence* for the behavioral axis is encoder-only, that one because the *tooling* objection was false. |
| **A2** | "Cross-lingual alignment" is one object with one onset time | H1, H2, H3a/b/c, H4 — every hypothesis is phrased as *the* transition; the analysis computes a single Δt | **contradicted** | Bo et al. arXiv:2411.14633: 8 similarity measures incl. linear CKA and mutual-kNN on identical models, **mean pairwise r = 0.58** vs 0.85 for behavioral measures (p=2e-16); measures cluster by *mathematical family*, not by model. Del & Fishel AACL 2022: on one trained XLM-R variant, **CKA ≈ 0 for layers 6–12** while PWCCA/SVCCA report **0.5–0.8**, retrieval >50%, and XNLI transfer **unchanged** — replicated across restarts. Ding et al. NeurIPS 2021: CKA and PWCCA "disagree on which layers of different networks are most similar." ReSi ICLR 2025: CKA ranks **1st in language, 8th in vision, 11th in graphs** — "No Free Lunch." Gröger et al. arXiv:2602.14486: under permutation calibration CKA's signal drops **0.86→0.45** while mutual-kNN holds ~0.85 — *the two Tier-1 geometry measures give opposite verdicts on the same data.* |
| **A3** | Δt = t_behavioral − t_mechanistic, via PELT changepoints on min-max-normalized curves, measures **lag** | The single headline claim ("the CI on Δt excludes zero") | **contradicted** | Simulation (premise-7 agent; code in `critique_evidence/`): with **true lag identically zero**, varying only transition *width*, an L2/PELT changepoint detector returns Δt = **+0.084 → +0.262 log₁₀** (1.21×–1.83×) at a **93–100%** false-positive rate. Mechanism: the detector places the split early for a broad transition (bias −0.08 at w=0.30, −0.24 at w=0.50) and on time for a sharp one. **H1 stipulates exactly this width asymmetry** (mechanistic smooth, behavioral jumpy), so the hypothesis and its dominant artifact are perfectly confounded. A parametric sigmoid midpoint was **unbiased in every cell** (\|Δt\| ≤ 0.007) — and `tier1_plan.md` §5 explicitly *rejects* the sigmoid approach in favour of changepoints. That choice is backwards. Min-max normalization equalizes range, not slope. |
| **A4** | Grokking is the right frame — the phenomenon transfers to natural-language pretraining | Title, H1, H2, and the whole "progress measures" apparatus | **unsupported; H2 contradicted** | ✓**self-verified** Nanda et al. §3: full-batch AdamW, **weight decay λ=1**, ~3,830 fixed examples, **40,000 epochs**. The proposal calls mBERT's **λ=0.01** "the direct analog" — off by 100×. Nanda §5.3: "when networks are provided with enough data, **there is no longer a gap between the train and test losses**." Li, Fan & Zhou arXiv:2506.21551 (OLMoE-7B): "global grokking… **cannot be found in LLM pretraining**"; it is local and asynchronous across data groups. Omnigrok arXiv:2210.01117: "**no grokking is observed for standard initializations**" — the controlling variable is init scale, set to its no-grokking value here. Kumar et al. ICLR 2024: grokking is a corner case of lazy→rich, and occurs *without* weight decay. Nearest real case (arXiv:2606.00230, 35–130M from scratch on C4) had to **manufacture** a train/test split from verbatim lexical overlap and calls itself an "analogue." |
| **A5** | "Stronger regularization accelerates the phase transition" (H3b's sole anchor) | H3b, and the monotonicity test in §6.3 | **contradicted by its own source** | ✓**self-verified** Nanda et al. Appendix D.1, verbatim: *"larger amounts of weight decay lead to faster grokking — on average, it takes around **3k epochs** … with weight decay λ = 0.3, **5-10k epochs** … λ = 1.0, and **20k epochs** … λ = 3.0."* The three numbers increase monotonically in λ. The prose says the opposite of the data in the same sentence. Corroborated: arXiv:2607.05104 finds grok-rate **inverted-U** in weight decay (20%→27%→90%→**0%** as λ goes 0→0.01→0.1→1.0). Liu et al. NeurIPS 2022 is a bounded "Goldilocks zone," not a monotone relation. |
| **A6** | Capacity pressure (language count) plays the role of weight decay | H3b, Experiment A, and the A1/A2/A3 run matrix | **unsupported; not identifiable** | At fixed capacity, **T = L·D** — you can hold at most one of {tokens-per-language, total tokens} fixed while varying language count. A1↔A3 trades the data confound for **three others**: +50% total compute, 1.5× optimization steps, and a **WSD schedule whose decay phase lands at a different absolute step** — and the dependent variable *is a step number*. Also a tokenizer confound the plan flags then walks past: a 32k BPE trained on EN+FR ≠ one trained on EN+FR+TR, so "1B EN tokens" is different text in A1 and A3 (**match on bytes, not tokens**). "Capacity pressure" is never operationalized as a scalar — weight decay has 0.01; this has no number. Narrowed Tier 1 has **one** language-count contrast (2 vs 3); two points are monotone by definition. Contra the direction: XLM-R capacity dilution, XNLI **71.8→67.7** as languages go 7→100; Hua/Yun/Pavlick NAACL Findings 2024 (mOthello): naive multilingual pretraining **fails** to produce a language-neutral space — anchor tokens do it, not parameter sharing. |
| **A7** | Subject-verb agreement is the construction Jian & Manning's JSD measures are defined over | The JSD measures; circuit extraction; **and the exclusion of Chinese/Japanese, which guts H4** | **contradicted** | ✓**self-verified** from the local EACL PDF (`EACL-2026-long-32.pdf`, pp. 752–765). Their construct is **verb argument structure class** (to-Dative n=35, Motion n=36, Reciprocal n=16, spray-load n=16; target = the post-preposition argument slot), and they explicitly distinguish themselves from Evanson et al. on this point: *"our experimental setup differs as it directly measures argument predictions, rather than **using subject-verb agreement as a proxy**."* The paper contains **zero** occurrences of "cross-lingual" or "multilingual." **The proposal grafts the SVA example onto that machinery**: §4.4 says "conditioned on the same **verb class**," but SVA has no verb classes in that sense — its classes are subject number/person and its target is the verb. `tier1_plan.md` Stage 2 reproduces this verbatim ("agreement target position, same verb class"). **The ordering result the proposal predicts is also overstated**: J&M found class-before-item for only *one* of four class pairs, by 50 steps — "for the (iii) Reciprocal and (iv) spray-load classes, **the onset of change is concurrent**." Consequence: `tier1_plan.md` §3.2 drops ZH/JA because they lack subject-verb agreement, and calls a rebuild "a scope increase, not a language swap" — but ZH and JA **do** have verb argument structure classes. The exclusion that reduces H4 to a single typological contrast rests on a construct the source method does not use. |
| **A8** | Cross-lingual JSD over next-token distributions is well-posed and returns signal | The cheapest, highest-resolution, least-scooped measure — Stage 2 of the plan, and the measure the whole design leans on | **contradicted — measured directly** | The premise-6 agent **ran the proposal's own contrast** on Qwen3-1.7B (12 SVA prefixes/condition, JSD in bits at the agreement target, bootstrap over items; code at `critique_evidence/jsd_probe2.py`, `critique_evidence/decomp.py`). Signal = between-class − within-class: **EN–FR +0.0112 [+0.0050, +0.0247]; EN–TR −0.0049 [−0.0176, +0.0029]; FR–TR −0.0064 [−0.0151, +0.0008]**. The two pairs involving Turkish — the pair that carries H4 — have the **wrong sign** and CIs including zero. Sign pattern replicated on a disjoint item set. The cross-lingual pedestal is **0.64–0.73 bits**, so the surviving EN–FR effect is **1.7% of the pedestal** and **~4% of the within-language class signal** (EN sg-vs-pl = 0.2902). Overlap mass Σmin(P,Q) = 0.294 (EN–FR), 0.211 (EN–TR): **71–79% of probability mass is language-exclusive**. Structural reason: JSD reads the model *after* the unembedding — after it has committed to a language — which is the one point where any shared representation is guaranteed to have been converted back to surface form. Jian & Manning's onset logic requires a **zero floor** ("mean D_JS … is around 0"); cross-lingually the floor is ~0.7 and **drifts upward** as language ID sharpens, i.e. the opposite direction from "alignment emerging." Qwen3-1.7B is ~47× larger and far better trained than the Tier-1 target — a generous upper bound. |
| **A8b** | The agreement contrast is realized *at the next token*, so "next-token distributions at the agreement target position" is well-defined in all three languages | The JSD measure's operationalization; `tier1_plan.md` §3.2's claim that the construction is "identically operationalized everywhere" | **contradicted** | Tokenizer test across mBERT (119k), XLM-R (250k) and Qwen3 (151k): **Turkish agreement is never at the next-token position in any of the three.** `-lAr` is the final morpheme of a stem+TAM complex and lands 2–4 tokens downstream (koşuyor/koşuyorlar, geldi/geldiler, yazıyor/yazıyorlar — all identical at token 1 in all three tokenizers). French regular `-er` verbs (~90% of the French verb lexicon) differ only by orthographic `-nt` and often not at token 1 (marche/marchent identical in all three). English is the only language where it robustly works — and XLM-R breaks it there too (`walks` → `▁walk`+`s`). These are 119k–250k vocabularies; the plan specifies a **32k shared BPE over three languages** (~10k effective each), so fragmentation will be strictly worse. This is a **lower bound** on the problem, and it is checkable in an hour with no training. |
| **A8c** | H4 (typological distance modulates alignment timing) is testable in narrowed Tier 1 | RQ4, and one of the four hypotheses | **unsupported** | Proposal §6.4 specifies "correlate threshold-crossing time with typological distance (lang2vec, WALS)." With EN–FR and EN–TR there are **two points**: a correlation over two points is ±1 by construction or undefined. What remains is a two-group difference confounded with at least four covarying variables — cognate/lexical overlap, tokenizer fertility, FineWeb2 volume and quality, morphological richness. **And the measured decomposition makes this fatal rather than merely weak**: the JSD measure's own residual signal *is* a lexical-overlap quantity, so it would report "EN–FR aligns earlier than EN–TR" as a mechanical consequence of cognates — indistinguishable from H4 being true. Breaking the confound needs a distant pair *sharing* script and a close pair *not* sharing it; the three-language set cannot do this, the original eight-language set could have. |
| **A9** | Mechanistic measures can be leading indicators (Contribution 9) | Contribution 2's "early warning signals"; all of Phase 2 §8.3 | **thinly-supported** | ✓**self-verified** Nanda's own framing: *"**Based on this understanding**, we define progress measures"* — the measures were constructed **after** reverse-engineering the algorithm, and their limitations section calls task-independent measures a *prerequisite* for predicting emergence. Olsson et al. rate their own large-model evidence "**Medium, Correlational**." Exactly one prospective instance exists — arXiv:2607.27281, frozen constants, held-out models, preregistered ±25% band, 200-head placebo — and it is **eight days old, single-author, unreviewed, unreplicated**, covers induction heads only, and states outright "**no benchmark-suite prediction yet**." Counterexample from the proposal's own reference set: Olsson's ICL score **plateaus permanently** after the phase change while capability improves across three orders of magnitude — the §8.3 stopping rule would have halted every model in the suite. **Independently confirmed by the premise-8 agent across nine model families read as full texts: zero use an interpretability signal in a real-time decision** (see A12). |
| **A10** | 5 seeds × ~60 log-spaced checkpoints can detect the effect | The claim that this is the *rigorous* version of single-seed prior work | **unsupported** | Checkpoint quantization floor: log₁₀ spacing 0.0658 ⇒ **1.164× minimum resolvable ratio**; integer rounding collapses 60 nominal checkpoints to **53 unique steps**, only **14** of them above step 1000. Minimum detectable Δt at 80% power, n=5: **1.41×** (optimistic jitter) to **3.16×** (pessimistic); at the empirical transformer jitter (CV 0.444, from arXiv:2603.25009: ΔT = 50,800 ± 22,565 steps over 5 seeds) it is **2.00×**. Going to 10 seeds moves that to 1.58×, and at pessimistic jitter **not at all**. A nominal 95% percentile bootstrap over n=5 has measured coverage **83.4%** (72.1% if skewed). Family size implied by §4: **216 at the floor, ~2025 as specified**; the plan states neither and corrects for neither. And an n=5 bootstrap's finest representable tail is 5⁻⁵ = 3.2e-4, so it is **arithmetically incapable** of a Bonferroni correction past ~156. |
| **A11** | The design is falsifiable — "every outcome is non-empty" | The pitch that this is worth doing regardless of result | **unsupported** | The table omits every likely failure: behavioral axis flat (Δt *undefined*, not zero); measures disagreeing so t_mechanistic is a choice (near-certain, per A2); changepoints not robust to the unspecified PELT penalty; no changepoint or many. Outcome 3 ("lag survives for some pairs") is called "probably the true answer" — and is definitionally what an uncorrected scan over 216–2025 tests at a **6–7% measured per-test FPR** produces by default (~13–15 spurious hits with no true effect). `tier1_plan.md` §8 Q2 defers the *framing* until after W5 while §5 pre-registers the analysis — the narrative is selected post hoc regardless. §7 adds a fourth escape hatch (pivot to *where* not *when*). The plan contains **zero** occurrences of "power", "effect size", "family", "multiplicity", or "falsify". No stated result would make the authors say H1 is false. |
| **A12** | Someone needs the answer | Whether this is worth months | **unsupported** | No party was identified whose decision changes. Full training reports read for **nine model families** — OLMo 1/2/3, BLOOM + BigScience chronicles, EuroLLM, SmolLM2, Aya 101, NLLB-200, Pythia, Llama 3, Gemma 3, Apertus, Marin. The monitored set is uniformly loss, gradient norm, throughput, and benchmark/BPB scores. **Zero of nine** take an interpretability or representation-alignment signal as input to any in-training decision. EuroLLM — the closest multilingual comparator, whose stated purpose *is* cross-lingual transfer — measures no cross-lingual representation quantity at all. NLLB is the nearest miss: its phased curriculum *is* derived from a measurement, but the signal is per-direction **validation perplexity**, and Tier 1 produces nothing that plugs in because it has no translation task, no parallel data, and no MT eval. Turkish is not low-resource (in mBERT's 104 and XLM-R's 100, ample FineWeb2 volume, its own EMNLP 2025 main-conference benchmark), so Tier 1 contains **no low-resource language** and the NLLB motivation is four inferential steps away. |
| **A15b** | "Measure the behavioral axis continuously" is a contribution the field has not absorbed | The plan's claimed defensible novelty | **contradicted — it has already been adopted, in production** | ✓**self-verified by PDF grep**. OLMo 3 (arXiv:2512.13961) §3.3.2, verbatim: *"while many tasks appear emergent, continuous proxy metrics have been shown to be a better decision-making tool for model performance before we exit the noise floor **(Schaeffer et al., 2023;** …). We propose a Base Easy task suite which measures **bits-per-byte (BPB)**."* Ai2 cites Schaeffer, concludes discontinuous benchmarks are bad decision tools, and replaced them with bits-per-byte — **the same metric `tier1_plan.md` §4 Stage 1 proposes as its novel continuous control**, with data decisions running through ~80 microanneals scored on BPB via a fitted GLM. Term counts in the same 118-page PDF: `interpretab` **0**, `sparse autoencoder` **0**, `progress measure` **0**, `linear probe` **0**, `representation similarity` **0**, `grokking` **0**, `CKA` 0 (3 apparent hits are the substring in "pac**ka**ge"). The metric-artifact *question* remains worth asking (A15); the specific *answer* is already deployed by the constituency this program targets. What remains is scholarly value, not applied. |
| **A13** | Small controlled models from scratch are a legitimate instrument | The entire Tier 1 design | **well-supported** | Dufter & Schütze EMNLP 2020 is nearly a template: ~1M-param BERT, **5 seeds with mean±std**, random-init null, trained in <40 min on one GTX 1080Ti — and they verified transfer to scale ("our findings transfer from our small setup to larger scale settings"). Pythia's stated rationale is exactly this control. TinyStories: coherent generation below 10M params. Toy Models of Superposition made a small-model prediction that held at scale. |
| **A14** | Log-spaced checkpointing, dense early, is correct | The checkpoint schedule | **well-supported — but the cited justification is wrong** | The decision is right on independent arithmetic: Pythia's log grid is {1,2,…,512} then every 1,000 steps at 2,097,152 tokens/step, so **log-spacing ends at ~1.07B tokens** and resolution beyond is 2.1B tokens per interval. That is sufficient justification by itself. The "first 10% of tokens" figure attributed to Dumas et al. could not be located by two agents — see citation problems. |
| **A15** | The metric-artifact question is worth asking | The plan's claimed novelty | **well-supported** | Schaeffer et al. NeurIPS 2023: **>92%** of claimed emergent abilities appear under just two metrics (Multiple Choice Grade, Exact String Match); of 39 BIG-Bench preferred metrics **at most 5** show emergence; they *induce* emergence in LeNet/MNIST purely by choosing a discontinuous metric. Blevins et al.'s cross-lingual ordering (arc 115k / POS 200k / arc-class 209k / XNLI 274k steps) is defined by **98%-of-final threshold crossing on accuracy** — a discontinuous read-out nobody has re-measured continuously. The debate is live: Du et al. NeurIPS 2024 find emergence "regardless of the continuity of metrics." |

### Two cross-cutting notes on the table

**A3 is the one to lead with.** A1 is the project-killer, but it is checkable in an afternoon.
A3 is different: it says that if you run the experiment exactly as planned and it *works*,
the number you report is probably an artifact of the estimator rather than a property of the
model. It is also the cheapest to fix — the same simulation that found the bias found that a
parametric midpoint is unbiased under exactly the width mismatches that break the changepoint
detector.

**A2 does not kill the project; it relocates the contribution.** If the measures disagree,
"when does the interlingua emerge" has no single answer — but *nobody has ever measured the
disagreement*. Wang et al. use neuron overlap, Körner et al. patching success, Blevins probe
accuracy, Jian & Manning JSD; not one reports whether a second measure agrees. A quantified
disagreement matrix with seed-bootstrapped CIs on identical representations would be a first,
and it is robust to the phenomenon being boring. That is a promotion from the plan's §7 risk
table to a primary pre-registered outcome.

---

## Conditions under which the program is sound

Stated as falsifiable propositions, so you can check them rather than argue about them.
Ordered by when they can be checked and how much they decide. C1–C3 are cheap and decisive;
check them before committing the fifteen runs.

**C1 — The behavioral axis moves in the architecture you will actually use.**
A frozen-probe zero-shot EN→TR POS score, on the actual arm (decoder or encoder), exceeds the
computed **50.0%** TR-BOUN majority-plus-surface-rule baseline by a margin exceeding across-seed
spread; and EN→FR exceeds **32.2%** on FR-GSD. *If false, there is no dependent variable and
nothing downstream is testable.* Note that all supporting evidence in A1 is encoder-MLM with
fine-tuning, so this cannot be inferred from the literature for a decoder with frozen probes.

**C2 — Step 0 is measured and H2's null is corrected.** The random-init probe value is reported
as the null for every measure. H2 as written asserts transfer "remains at chance" during
memorization; there is no chance level for POS tagging, and punctuation and digits are shared
across EN/FR/TR and linearly decodable at initialization. *If the step-0 probe already scores
≥50% on Turkish, H2 is false by inspection before any training happens.*

**C3 — A genuine memorization phase exists.** Held-out multilingual loss on a disjoint shard is
flat-and-high while training loss is near zero, i.e. a real train/test gap. Free to log.
*If train and held-out loss track each other — the overwhelmingly likely outcome for
single-epoch pretraining — H2 is dead and the grokking framing has no referent.*

**C4 — Δt is estimated by a width-invariant method, and transition widths are reported.**
Either the two compared trajectories have widths within ~20% of each other, or Δt comes from a
parametric midpoint rather than a changepoint penalty. *If false, a spurious 1.2×–1.8× lag with
the H1-predicted sign appears at 93–100% rate and the headline number means nothing.*

**C5 — The true lag exceeds ~2× in steps and falls below ~8×.** Below 2× the design cannot
resolve it at n=5; above ~8× the behavioral transition falls past step 7600 and truncates
(a true 0.20 log₁₀ lag reads as 0.057 — 71% attenuation — when the mechanistic transition sits
at step 6310). *No published estimate of the multilingual lag magnitude exists, so this is
currently an assumption rather than a design parameter.*

**C6 — Across-seed jitter in transition location is ≤ 0.10 log₁₀ steps.** *If it exceeds this,
the n=5 bootstrap cannot work and the budget should move from 3 configs × 5 seeds to 1 config ×
15–20 seeds.* Directly measurable from the first five seeds of one config, before spending the
rest. The only published transformer estimate (CV 0.444) is well above this.

**C7 — A family size is stated and a correction applied.** Which requires n ≥ 10 seeds for the
bootstrap to represent the needed tail probability at all, and realistically pre-registering a
*single* primary (behavioral, mechanistic) pair with everything else labelled exploratory.

**C8 — A pre-specified result exists that you will call "H1 is false."** `tier1_plan.md` §8 Q2
must be resolved *before* W5, not after. *As written, every branch is a publication and none is
a refutation.*

**C9 — The JSD quantity is pre-registered as a difference of divergences,** not raw cross-lingual
JSD, and the shuffled-language-label control is run against it. *If raw JSD is used, the measure
is dominated by language identity.* **Necessary but no longer sufficient** — the difference *was*
taken in the measurement above and the Turkish pairs still came out with the wrong sign.

**C9b — The agreement contrast is realized at the next token under the study's own 32k tokenizer,
in all three languages.** Falsified for Turkish under three *larger* tokenizers; a 32k shared
vocab will be worse. **Costs an hour and no training.** *If false, the JSD measure has no
well-defined target position in the language that carries H4.*

**C9c — The residual JSD signal is carried by content tokens, not punctuation and
cross-language leakage.** Run the token-level decomposition on any candidate model; if the top
contributors are not verb forms, the measure is not measuring agreement. *Currently falsified on
a 1.7B model: top-5 contributors are `,`, ` is`, ` has`, ` are`, `:` — 52% of the signal, and no
French token among them.*

**C9d — Turkish 3rd-person-plural minimal pairs are genuinely minimal.** Turkish 3pl agreement is
reported to be optional under animacy and word-order conditions; if the "ungrammatical" member is
in fact grammatical, the class label is wrong in the language carrying the distant condition.
*Needs a Turkish linguist, not a literature search.*

**C10 — Claims name their construction.** "Agreement-JSD onset precedes POS-transfer onset" is
supportable; "the interlingua forms at step N" is not. This is `CLAUDE.md`'s claim-hygiene rule 1
applied to the thing this study will actually measure.

**Conditions under which the *grokking framing specifically* is sound: C3 must hold, and a
weight-norm signature (sum of squared weights, or per-layer ‖W‖) must inflect at the claimed
cleanup boundary.** Free to log, decisive if absent. Without it, "cleanup" is a label with no
mechanism — Nanda's cleanup is evidenced by a sharp drop in summed squared weights and a sharp
rise in the Gini coefficient of Fourier norms, and neither has a multilingual construction.

---

## The steelman, in full

Reported without softening. Where it concedes, the concession is its own; where it holds, it
held against the attacks above.

### The case

The published literature on when cross-lingual alignment emerges consists, without exception,
of single-run observational analyses of models somebody else trained for another purpose — and
its authors say so. The maximum number of independent training runs behind **any** published
claim in this area is **one**.

| Study | Trained by | Checkpoints | **Seeds** | Controls / CIs |
|---|---|---|---|---|
| Blevins et al., EMNLP 2022 | themselves (XLM-R replica, 270M, 786B tok) | 39 | **1** | no CIs on main findings |
| Wang, Minervini & Ponti, Findings ACL 2024 | public BLOOM checkpoints | **6 / 8 / 4** | **0** | no null model, no seed variance |
| Körner et al., EACL 2026 (EuroLLM) | public checkpoints | 26 | **0** | has `en_en` copying control — best in the literature |
| Jian & Manning, EACL 2026 | public GPT-2 checkpoints | 450 | 3 (reports 1) | exemplar-first baseline, 95% CIs |
| **This plan** | **themselves** | ~60 × 15 runs, log-spaced | **5 per condition** | step-0 null, shuffled-label null, positive + metric-pair controls |

Wang et al. state the blocker themselves, verbatim: *"our findings on the trend of alignment
might be not applicable if zooming in on a particular window of training with finer-grained
checkpoint models."* The plan is precisely the fix they name. And their headline
non-monotonic finding — alignment degrading mid-training — is defended in the paper with
*"this phenomenon **may be an artefact** due to the variance of overlap rates or an error in
checkpointing."* Two single-seed studies agreeing is consistent with both having drawn once
from a wide seed distribution. Only a seed-varied design separates these.

### What would be known afterwards that is not known now

1. **Whether the cross-lingual lag is a metric artifact.** Blevins' ordering is defined by a
   98%-of-final threshold crossing on an accuracy curve. Nobody has re-measured it
   continuously. If it survives bits-per-byte and mean log-prob it becomes much stronger; if
   it collapses, a widely cited ordering needs qualifying.
2. **Whether the mid-training alignment degradation is real or seed noise.** Three papers
   report it, none has a second seed. A 5-seed design either reproduces it with a CI or bounds
   it below noise. The actionable consequence — "select checkpoints per-language, don't ship
   the last one" — is something a practitioner would use.
3. **A quantified disagreement matrix between alignment measures on identical
   representations,** with seed-bootstrapped CIs. Currently nonexistent.
4. **A causal estimate of parameters-per-language on alignment,** from A1 vs A3. Observational
   checkpoint suites cannot do this at all.

### Why the multilingual setting is an unusually good testbed for the emergence question

This is the steelman's strongest original argument. Wu & Lo (composed U-shapes) and Michaud et
al. (quanta) converge on the claim that a sharp aggregate curve is a superposition over
subgroups — and **both had to manufacture their subgroups post hoc**, Wu & Lo by stratifying on
empirical difficulty, Michaud by clustering to discover quanta. Post-hoc strata are exactly
where an aggregation story becomes unfalsifiable. The multilingual case supplies a
disaggregation axis that is **pre-registerable** (language, pair, typological distance,
agreement feature), **externally ordered** (EN–FR vs EN–TR is a typological prediction, not a
data-derived split), and **equipped with a free null** (step 0; shuffled labels). The
scale-axis emergence studies have none of these, because you cannot ablate "scale."

Additionally, Du et al.'s reframing says the right x-axis is *pre-training loss*, not model
size — and sweeping loss with architecture, data and tokenizer held exactly fixed is what a
from-scratch training-time study does and what a cross-lab model family cannot.

### The steelman's concessions

- **(a) Grokking.** Concede almost entirely. "The word should come out of the title and the
  hypotheses." What survives is Nanda's *method* — build a measure that moves while the
  behavioral metric is flat — which is regime-independent and needs no weight decay and no
  memorization gap. **H2 should be demoted from hypothesis to motivation.**
- **(d) H3b.** Concede the analogy, keep a weaker mechanism. Dufter & Schütze state it without
  reference to grokking: *"BERT is multilingual because of a limited number of parameters. By
  forcing the model to use its parameters efficiently, it exploits common structures by
  aligning representations across languages."* The defensible restatement is
  **parameters-per-language modulates alignment strength** — testable by A1 vs A3, no analogy
  required. *(Note: the premise-4 attacker's identification argument still applies to the
  A1↔A3 comparison as specified; see A6.)*
- **(e) Contribution 9.** Concede entirely, but reframe: not "aspirational," **gated**. Tier 1
  tests a *necessary condition* and can kill it. "Strike Contribution 9 from the contributions
  list and re-enter it as the gate the experiment exists to open or close. A feasibility gate
  for a research program is a legitimate MS thesis. An aspirational contribution is not."
- **(g) Power.** Largely concede. But: Δt is measured on log(step), so the resolvable quantity
  is a *ratio*, and the literature's claims are ratios of the right magnitude — Blevins'
  20k monolingual vs 115k–274k cross-lingual is **6–14×**. The study is well matched to testing
  *the existing literature's claim*, not an arbitrary small effect. The concession that
  matters: **"go to 10 seeds if needed" is optional stopping** and directly undercuts the
  pre-registration. Fix n in advance at 8–10 and delete that sentence.
- **(f) Narrowness.** Partially answerable — but with a concrete saving. **MultiBLiMP 1.0**
  (Jumelet et al., TACL 2026) covers **101 languages, 128,321 balanced minimal pairs** on
  subject-verb and subject-participle agreement for number/person/gender, built from UD +
  UniMorph, and covers EN, FR and TR. The plan's W2 line item — *"custom-built per language…
  This is real linguistic work — budget for it explicitly"* — is largely unnecessary. That is a
  week recovered and a large reduction in the risk of building an idiosyncratic, unreviewable
  evaluation set.

### The steelman's own stated weak points

1. **The whole case assumes the mechanistic trajectories have identifiable changepoints at
   all.** If CKA, mutual-kNN and JSD all rise as smooth monotone curves, PELT returns arbitrary
   breakpoints and the result is a null about the *instrument*. **The positive control does not
   protect against this** — modular addition has a sharp transition by construction, so
   recovering Nanda's lag proves the pipeline works on sharp curves and says nothing about
   natural-language curves. The plan's control structure has this gap. *(This is independently
   corroborated: the premise-7 simulation found Δ = 1.00 log₁₀ has power 1.00 in all 27 cells
   including the worst — the positive control passes at 100% under maximally adverse
   assumptions and therefore carries zero information about the 1.2×–2× regime.)*
2. **Measurement papers are correct and get rejected.** "We bounded the lag and the CI includes
   zero" is a good thesis and a hard ACL submission. If publication is a hard requirement, the
   cross-measure disagreement result and the degradation reproduction are the two outcomes most
   likely to clear a venue bar, and the design should guarantee those regardless of Δt.
3. **The most reused evidence is six years old and architecturally off-target.** Karthikeyan
   (2020) and Dufter & Schütze (2020) are encoder-MLM, multi-epoch-over-1GB, pre-RoPE,
   pre-modern-tokenizer, extrapolated to a decoder-only single-pass FineWeb2 run.

### The single piece of evidence that would most damage the steelman's case

Its own words: *a W1 pilot showing that a 36M decoder-only model trained on ~2B tokens of
FineWeb2 EN/FR/TR yields zero-shot EN→TR POS transfer at or below the majority-class baseline.*
That eliminates the behavioral axis, leaves the mirage test with nothing to test, and reduces
the program to "the student built a training pipeline and measured CKA curves." **Every defense
of A1 rests on encoder-MLM evidence; none rests on decoder evidence, because none was found.**
If the pilot comes back flat, the recommended response is not more data or parameters — it is
to switch to the encoder arm, where transfer is documented at *smaller* parameter counts, and
absorb the tooling cost.

This converges with `tier1_plan.md` §3.1 and `CLAUDE.md` from an unexpected direction: the
architecture question is not a tooling matter. It is the question of whether your dependent
variable exists.

---

## Scale, timeline, and where this would publish

### Is it MS-scale?

**Too big, by roughly 2–3× — and the compute was never the constraint.** 60–90 A100-hours is
trivial. The build is not, against a track with **zero code** and unmeasured transfer from
`compression/src`. Underestimated items, ranked:

1. **FineWeb2 EN/FR/TR pipeline** — one of four W1 items. Needs streaming, dedup decisions, a
   **byte-matched** sampler (per A6, or A1/A3 aren't comparable), and full login-node
   pre-caching since compute nodes have no internet. Realistic: **1.5–2 weeks alone.**
2. **From-scratch training + log-spaced checkpointing, 130GB × 15 runs** — W3 budgets *one week
   to build it and run all fifteen*. Marin's public logbook is the reference class for what
   happens to competent teams: an LCG-shuffle-induced loss phase shift, a rotary-embedding bug
   found mid-run, z-loss needed because `lm_head` exploded.
   Realistic: **2–3 weeks to a trustworthy pipeline, then runs.**
   *(An earlier draft listed a `run_with_cache` batch-size-1 bug here. It does not exist —
   the `NotImplementedError` is scoped to `generate(return_cache=True)`, and issue #1265
   closed 2026-04-22. Corrected against the 2026-08-06 method survey.)*
3. **Measure stack** — debiased CKA with permutation calibration, MNN, JSD, PELT, bootstrap,
   disagreement analysis, plus the invariant tests `CLAUDE.md` §20.3 mandates for any similarity
   measure. W5 also contains the changepoint analysis, the negative control, *and* the go/no-go.
   Realistic: **1.5–2 weeks.**
4. **The positive control is a second training pipeline** (full-batch AdamW, λ=1, ~40k epochs),
   budgeted as one line inside W1 — **and it carries zero information**, since Δ = 1.00 log₁₀
   has power 1.00 in all 27 simulated cells. A week of work for a control that cannot fail and
   therefore says nothing about the 1.2×–2× regime the study operates in.
5. **Provenance** — manifest, config-load tests, per-run `git_sha`/resolved-config/versions/seed.
   Required by `CLAUDE.md` *before the first code lands*; appears nowhere in W1–W7.
6. **Minimal pairs — partly de-risked.** W2's "real linguistic work" is largely unnecessary:
   **MultiBLiMP 1.0** (TACL, 101 languages, >128,000 SV-agreement pairs) and **TurBLiMP**
   (EMNLP 2025 main, 16 phenomena × 1000 pairs, human judgments) both exist; **CLAMS** (ACL 2020)
   covers EN and FR but **not** Turkish. A week recovered — but this does **not** rescue A8b:
   having the items does not make the next-token target position well-defined in Turkish. Note
   TurBLiMP's own headline — *"cutting-edge Large LMs still struggle"* with Turkish phenomena.

**Realistic timeline: 14–20 part-time weeks** against a planned 6–7, including the 30–50% slip
standard for from-scratch training work. That is a semester to two — genuinely MS-scale *as
work* — but the plan puts its single go/no-go at W5 of 7, with one week of slack.

### The floor problem

**There is currently no guaranteed deliverable.** The headline is a conjunction of four serial
conditions, each independently attacked above: behavioral axis moves (A1) → changepoints exist
on both curves (steelman weak point #1) → Δt estimable without width bias (A3) → n=5 resolves it
(A10). The plan's "every outcome is non-empty" defence assumes the interesting null is
interesting — which is exactly what A12 denies.

Two candidate floors exist and neither is primary:

- **The cross-measure disagreement matrix** (A2) — robust to the phenomenon being boring,
  unclaimed in the multilingual setting.
- **The released 15-run, 5-seed, log-spaced 36M multilingual checkpoint suite.** This is the
  proposal's Contribution **5**, and it is the only Phase 1 contribution guaranteed to be
  delivered if the main result fails. **It is listed fifth. It should be first.**

### Venues

| Venue | Assessment |
|---|---|
| **BlackboxNLP 2026** (EMNLP 2026, Budapest, Oct 29) | Best realistic fit, especially for the disagreement-matrix framing. Dual track: 8-page **archival** + 2-page non-archival. 2025: 28 archival + 26 non-archival + 3 shared tasks from 99 submissions (denominator mixes tracks — **do not compute a rate**). Caveat: published scope does not name training dynamics. Ceiling: a workshop paper. |
| **CoNLL 2026** (ACL 2026, San Diego, Jul 3–4) | Best-fitting *archival* venue; SVA across three typologically varied languages is squarely its register. But **CoNLL 2025 accepted 40/217 = 18%**, stricter than ACL 2025 main. Moderate odds. |
| **TMLR** | The venue whose written criteria most accommodate a bounded null — significance and novelty are **explicitly excluded** as rejection grounds; the bar is "accurate, convincing and clear evidence" plus "some individuals in TMLR's audience would be interested." Good odds. Cost: less \*ACL hiring weight, and criterion 2 is Premise 8 wearing a hat. |
| **Findings of ACL/EMNLP** | Realistic target — **and where the closest predecessor landed** (Wang, Minervini & Ponti, Findings of ACL 2024). Rates: ACL 2025 16.7%, EMNLP 2025 17.34%, ACL 2026 17.8%. |
| **Insights from Negative Results in NLP** (EMNLP 2026) | Real and running, six editions 2020–2025, archival — but **4-page short papers**, which cannot hold a 15-run multi-measure study, and submitting pre-labels the work a negative. |
| **ACL / EMNLP / NAACL main** | Unlikely. ACL 2026 main **18.9%** of 12,148 unique submissions. |

**The bar, in ACL's own words.** There is **no negative-results track and no reproducibility
track** at \*ACL. ARR CFP, verbatim: *"Both positive and negative results for experimental
studies are welcome, and have the same challenge of justifying to the program committee why this
particular result is interesting and important."* **The venue question and the consumer question
are literally the same question, and ACL says so.**

**Artifact-adjudication precedent at \*ACL main exists** — *Tangled up in BLEU* (ACL 2020),
*Ties Matter* (EMNLP 2023, Outstanding), Hewitt & Liang's control tasks (EMNLP-IJCNLP 2019),
*We Need to Talk about Standard Splits* (ACL 2019, Outstanding), *A Call for More Rigor in
Unsupervised Cross-lingual Learning* (ACL 2020). **Every one invalidated a measure the field was
actively using, in the setting where it was used.** Tier 1 would adjudicate a lag reported in
three papers, at a scale none of them used, with a JSD measure that A8 shows returns the wrong
sign in the language carrying the key contrast.

**Empirical landing pattern** (exhaustive match over the ACL Anthology, 127,461 records):
descriptive checkpoint-tracking → Findings or workshop; main-conference acceptance empirically
required a **causal** contribution (Bayazit et al. crosscoders+RelIE → ACL 2026 main; Lesci et al.
causal memorisation → ACL 2024 main; Körner et al. activation patching → EACL 2026 main). Tier 1
as narrowed is correlational. Corroborating: **BabyLM's 75 Anthology papers split 55 workshop /
4 \*ACL main (all EMNLP) / 4 Findings / 3 CoNLL — zero at ACL, NAACL or EACL main.** And **Pico**
— the layer-naming baseline `CLAUDE.md` instructs this track to match — is **EMNLP 2025 System
Demonstrations**.

### The Schaeffer case study, which cuts both ways

*The debate did not settle* — three verified rebuttals in three years taking three different
positions (Du et al. NeurIPS 2024; Hu et al. ICLR 2024; Wu & Lo ICLR 2025), plus Lu et al. ACL
2024 main. That is the realistic model of a Tier 1 paper's discourse: cited and disputed.

*But it did change one practice — the exact one Tier 1 proposes.* Schaeffer's traceable effect on
the world is that Ai2 replaced task accuracy with bits-per-byte for in-loop data decisions,
citing it. That took a NeurIPS **Outstanding Paper**, a claim spanning all of BIG-Bench, a
constructive demonstration in a second domain, and **three years** — to move one metric choice at
one lab. Tier 1's methodological contribution *is* that metric choice, and it cannot produce the
change again because it has already happened.

---

## Citation problems found in the source documents

These are checkable and would be checked. Fix before the PI meeting.

| Where | Claim | Status |
|---|---|---|
| Proposal §5.1 | "the precedent of Blevins et al. (2022)… which found that training dynamics patterns replicate at small scale" | **Wrong.** Blevins trained an **XLM-R Base replica: 12L/768d, ~270M params, 100 languages, CC100, 1.5M updates ≈ 786B tokens**. It is a precedent for the *method* (frozen probes on log-spaced checkpoints) and for nothing about small scale. |
| Proposal §5.1 | "TinyStories-sized models show the same multilingual organization as larger ones" (arXiv:2511.10840) | **Contradicted by the source.** Their small model is **68.5M** (larger than proposed), and Finding 3 states: *"For the 4-layer TinyStories model, the up-and-down pattern is **absent**… suggesting a minimum model size for this behavior to emerge."* They also **never measured zero-shot cross-lingual transfer** — the analysis is representational only. |
| `tier1_plan.md` §1, §3.6 | "Dumas et al. found shared concept space forming within the **first 10% of tokens**" | **Could not be located** by two independent agents. The EuroLLM checkpoint study appears to be **Körner, Müller-Eberstein, Korhonen & Plank (EACL 2026, arXiv:2601.22851)**, who say "emerge early," not "first 10%." Dumas et al. (arXiv:2411.08745) is the concept-patching *method*. **§3.6 justifies the entire checkpoint schedule on this number** — but the schedule is independently justified by the Pythia resolution arithmetic (A14), so re-source or re-argue. |
| `tier1_plan.md` §4 | "Jian & Manning (EACL 2026 **Best Paper**)" | Venue and pages verified (pp. 752–765). **Best Paper designation unverified** by two agents. |
| `tier1_plan.md` §2 | "Foroutan et al. already established that curriculum order doesn't change the endpoint" | Attribution unverified. The relevant paper appears to be **Foroutan, Teiletche, Tarun, Bosselut, arXiv:2510.25947**, "Revisiting Multilingual Data Mixtures." The decision to drop Experiment B stands on its **second** ground (a grokking signature at a switch is not separable from forgetting/re-learning). |
| Proposal §4.1 + bibliography | "Adams, O., Cotterell, R., & Gonen, H. (2022). The geometry of multilingual representations in XLM-R." | **No such paper found.** The content described (mean-centering per language; stable language-neutral axes encoding token position and POS) matches **Chang, Tu & Bergen, "The Geometry of Multilingual Language Model Representations," EMNLP 2022** (arXiv:2205.10964). One of the four geometry measures currently has no verifiable operational definition. Check against the PI's bibliography before raising it as an error. |
| Proposal §5.1 "Regularization note" | mBERT/XLM-R's "AdamW with constant weight decay of 0.01 — **the direct analog** to the grokking setup" | **Off by 100×.** Nanda et al. use **λ = 1**. `tier1_plan.md:102` inherits this: *"weight decay 0.01 (the grokking-analog regularizer — keep it)."* |
| Proposal §4.3 / §5.4 | Circuit edge Jaccard across languages | Already dropped by the plan — correctly. Supporting detail: **no published cross-lingual circuit study uses edge Jaccard** (Ferrando & Costa-jussà use qualitative component matching; Zhang et al. ICLR 2025 use Pearson ρ on head activation frequency). Where exact edge overlap was tested against a size-matched random null it was **not significant** (0.142 [0.111, 0.177] vs 0.117 random, **p = 0.106**), and changing the prompt *template within one language* drives edge overlap to **~0**. At 6L/8h your models have **~3,628 edges** vs GPT-2 small's 32,491, so a 98%-sparse circuit is **~73 edges** — Jaccard over sets that small is high-variance, and the published seed-only noise floor (0.5–0.7 IoU) was measured on sets ~10× larger. |
| Proposal §4.4 | Jian & Manning's construction | See **A7** — it is verb argument structure, not subject-verb agreement. |
| `tier1_plan.md` §7 | "already scooped (**Inaba EMNLP 2025**; Riemenschneider & Frank ACL 2025; copy-first-translate-later)" | **"Inaba EMNLP 2025" does not exist.** The only Inaba at EMNLP 2025 published on conversational recommendation. "Copy First, Translate Later" is **Körner, Matveev, Eichin, Kutyniok & Plank, arXiv:2604.17633 — an unrefereed preprint**, 1.7B, nine languages, at *finer* checkpoint resolution than Tier 1 plans. |
| `tier1_plan.md` §7 | The scooping assessment generally | **Worse than the plan admits, and the adjudication niche is also occupied.** Riemenschneider & Frank (ACL 2025 **main**, arXiv:2506.01629) state the two-stage result verbatim across three MLLMs. Körner et al. (EACL 2026 **main**) have it with **causal** activation patching — *and already issue the artifact warning*: "some apparent gains in translation quality reflect shifts in behavior — like selecting senses for polysemous words or translating instead of copying cross-lingual homographs — rather than improved translation ability." |

---

## Status of this audit

All nine agents returned. Every premise was attacked; the steelman is reported unsoftened.

The steelman's three named constituencies were subsequently tested by the premise-8 attacker
and two of the three did not survive: practitioners do not consume representation measures
(0 of 9 model families), and the emergence debate cannot be adjudicated on Tier 1's axis
(Du et al. argue on pre-training loss across sizes; Hu et al. need decoding-sample resolution;
Wu & Lo disaggregate by difficulty — Tier 1 sweeps step within one architecture at one size).
**The third survives**: anyone publishing a CKA-based multilingual result is an identifiable and
numerous consumer for a measured disagreement matrix. That is the constituency to build for.

Three results here are measurements, not literature, and are the most reusable part of this
document:

- **The Δt sharpness artifact** (A3) — simulated changepoint pipeline, code in `critique_evidence/`, 250–400
  trials/cell. Also produced the bootstrap coverage figure (83.4% at n=5) and the resolution
  floor (53 unique checkpoints, 1.164× minimum ratio).
- **The JSD contrast on Qwen3-1.7B** (A8, A8b) — `critique_evidence/jsd_probe2.py`, `critique_evidence/decomp.py`,
  re-runnable. n=12 items/condition, bootstrap over items only, single model, single seed. By
  this repo's own standard (`CLAUDE.md`: "n=24–32 is exploratory") **this is exploratory, not a
  paper claim** — but the sign pattern replicated across two disjoint item sets, and the
  magnitude ratios are not marginal. The honest weak point, in that agent's own words: it
  measured an endpoint and is inferring about a trajectory.
- **The OLMo 3 grep** (A12, A15b) — full 118-page PDF read, decision-loop passage quoted
  verbatim, interpretability term counts all zero. Reproducible in one command.

Carried but not verified by this audit: the "~10M words" SVA-saturation figure behind the claim
that agreement is acquired in the first 2–4% of the run; the Turkish 3pl optionality claim
(second-hand, needs a linguist); the Edge Pruning 0.5–0.7 seed floor (read by a delegated agent,
not directly). Two literature sweeps commissioned by the premise-6 agent — SVA acquisition
timing, and the syntactic-vs-semantic cross-lingual dissociation including Anthropic's
multilingual circuit-tracing work — did not return, and nothing is cited from them.

Self-verified against local PDFs during composition: Nanda et al. §3 hyperparameters and
Appendix D.1 (A4, A5); Jian & Manning's construction and Schaeffer citation (A7);
Karthikeyan et al. Table 6 in full (A1). Baselines in A1 for FR-GSD and TR-BOUN were computed
by the premise-3 agent from the actual UD test files, not quoted from a paper.
