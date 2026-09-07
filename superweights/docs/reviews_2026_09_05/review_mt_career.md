# Adversarial review — MT relevance and RE-career value

Reviewed 2026-09-05 against PROJECT_STATEMENT.md, problem_search_2026_09.md, proposal_draft.md, notes.md, the two paper-notes files, the registry, and the uneven-PTQ replication. Web check: no WMT/EAMT/ACL-MT paper on super weights or massive activations in translation models exists; the nearest is an unreviewed preprint on NLLB cross-attention sinks (arXiv:2605.01229).

## (a) Verdict: SURVIVES WITH CHANGES

**MT researcher.** As framed, the MT arm is decoration. Zeroing a weight is a lesion no deployed system suffers, and the lab's own registry already shows the super weight is *irrelevant to the one real MT-compression problem*: `rtn+SW ≈ rtn` on 8 models, importance orthogonal to quantization sensitivity, and NVFP4 ranks `down_proj` least quantization-sensitive. The hallucination link is weaker still: Guerreiro's oscillatory hallucinations are rare, input-conditional failures of a *working* model, less prevalent in low-resource directions; "We. We. We." on every input is a broken model, and scoring it with TNG is characterisation, not discovery (the notes themselves reject (iv) as headline). What survives is the NLLB question, because target-language selection at decoder position 0 and off-target translation are problems MT people actually have.

**Hiring manager.** The notes are the best thing here: a metric validated against a published number, a retraction the same day, superseded banners, false-positive taxonomies — evidence of scientific honesty that most candidates cannot show. The code is not yet a portfolio: 1,532 lines, zero tests, zero YAML configs despite the repo's own rule, four detector versions kept as separate files, provenance = git sha only, and the notes record two detectors passing a probe they should have failed. Nothing trains a model, nothing is distributed, nothing is used by anyone else. Fixable cheaply; not fixed.

## (b) Strongest objections

1. **RQ2(b) will return a null the design cannot interpret.** Per-language ablation damage and per-language PTQ damage are both collinear with resource level; n = 20 languages; the PTQ covariate itself is setup-dependent (the lab's replication: X→en Indic collapse did *not* reproduce, Malayalam baseline off by +10 COMET). Under total collapse chrF++ floors, so there is no profile. *Fix:* demote to a dose–response on two models (EuroLLM, Tower joint set), scalar at {1, .5, .25, 0}, forced-decode KL per language; report the partial effect after resource level or drop it.
2. **Half the "multilingual" arm is already known empty.** Registry: Aya-Expanse KL 0.002, BLOOM 0.001, Gemma 0.005; TowerBase ×1.58 individually. Only EuroLLM (two scalars, KL 3.28/2.98) is a strong single-weight multilingual model. *Fix:* state coverage as "EuroLLM plus joint sets", and make RQ3 (unit = neuron) run *before* any MT table, or the tables measure the wrong object.
3. **The MT link is asserted, not built.** No real deployment quantity is measured. *Fix:* the lab runs NLLB in CTranslate2 int8 already (speech-translation). Quantize NLLB-3.3B (int8, 4-bit), score per-language COMET, and test whether per-language massive-activation magnitude predicts the drop after partialling resource level (compression notes P4). That converts an artificial lesion into a covariate for a real degradation.
4. **Scope is 2–3× the budget.** RQ1 (5 models × 7 measurements), RQ2 a/b/c, RQ3 at three granularities, RQ4 gate, plus an unbuilt encoder–decoder adapter, in 125–150 h. *Fix:* RQ3 → RQ1 on three models → NLLB detection/tag test → attractor-across-languages. Nothing else before week 9.
5. **RQ1 is the scoopable part.** Cancedda 2024, Yona 2025, Stolfo 2024 and the ICLR 2026 sink papers are big-lab, fast-moving. *Fix:* frame RQ1 as adjudicating Puccetti-vs-Macocco on the frequency direction, and keep the unscooped multilingual/NLLB pieces as the claim.
6. **Tainted term.** Building on a rejected preprint hurts only if the framing depends on it. Anchoring on Sun (COLM 2024), An (ICLR 2025), Subramanian (COLM 2026) fixes credibility; the ICLR rejection was on quantization claims this project does not make. Say so in one sentence.

## (c) Which MT question to keep

**Keep:** NLLB per stack (600M *and* 3.3B), tag-vs-arbitrary-token control, and the causal readout as **output language-ID and off-target rate under dose–response**, not chrF++ under zeroing. If the decoder's massive activation sits on the target tag and attenuating it produces off-target output, a single scalar is the mechanical implementation of target-language selection — an MT PI would care. If NLLB has no such structure, that is a modest architecture note; make it worth something by pairing it with the int8 per-language degradation measurement in (b)3.
**Keep, cheap:** attractor identity across forced target languages (fixed token vs target-language unigram) with the LAPE contrast arm — a mechanism claim for hours of compute.
**Drop as headline:** per-language damage profile from zeroing; TNG/ALTI+ characterisation of collapsed output.

## (d) Against the alternatives

- **(a) From-scratch formation study.** Better RE training (training loop, WSD schedule, checkpoints, seeds), and the registry marks formation "genuinely unclaimed". But 400–900 GPU-h for 12 arms × 3 seeds, no evidence a ≤1B model has a super weight at all (D0 unrun), and no MT component. Right project for a PhD student; wrong for 150 h with an MT lab.
- **(b) Pure MT-compression (Marie & Fujita line).** Highest MT relevance and the lab's harness (n≈960, COMET) exists — but 31/35 units are already run, novelty is incremental, and interpretability skills are zero. It is a better *paper* for an MT venue and a worse *learning* project.
- **This project, cut per (b)–(c):** the only option that is unscooped, MT-anchored (NLLB), interpretability-teaching, and finishable. Import the one week of (a) as a gate (Pythia/Ettin at final checkpoints) and (b)'s int8 NLLB per-language measurement as the real link.

## (e) Artifacts that must exist at the end

1. **A pip-installable, tested `superweights` package**: pytest with planted-weight recovery, a config-loads test for every committed YAML, the control triple and repetition metrics; one YAML per run; results carrying git sha, resolved config, library versions, seed. Two days now; it is what a hiring manager clones.
2. **One public table** — ~12 models × {scalar, neuron} × {zero, mean, matched-random} with CIs and coverage stated, plus the NLLB per-stack result — released as a small dataset with a reproduction command.
3. **A write-up in the notes.md register**: what was predicted, what was measured, what was retracted, with the negative results filed in the registry. The honesty is the differentiator; keep it visible.

Week-9 regret to pre-empt: writing a per-language table from the parked runs because the owned protocol was not finished. Do not.
