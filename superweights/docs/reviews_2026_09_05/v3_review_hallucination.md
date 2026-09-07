# Adversarial review — Arm B / H2 (hallucination signal in NLLB)

Reviewed 2026-09-05 against proposal v3 §§1–2, 4–5, 8; the three v2 reviews; `notes.md` (no NLLB or hallucination measurement exists yet); `docs/literature/notes_*`; and the PDFs of Dale 2023a, HalOmi, Binkowski 2026, arXiv:2605.01229. Web check: no sink- or massive-activation-based hallucination detector for NMT exists; the nearest are Binkowski 2026 (decoder-only QA) and HalOmi's own Appendix B.

## (a) Verdict: SURVIVES WITH CHANGES

v3 fixed the v2 complaint that TNG-on-a-broken-model is characterisation: Arm B now measures a *healthy* NLLB on human-annotated data, which is the right object. But the arm's mechanistic premise does not connect to its measurement, one of its two headline features is predicted by the literature to be constant, the baseline it names is from the wrong model, and per-direction intervals are infeasible. What survives is a well-posed, cheap, pre-registerable detector question — decoder-side sink mass and step entropy as features added to Seq-Logprob on pooled HalOmi — whose negative is informative and whose positive an MT researcher would use.

## (b) Objections

1. **The analogy is surface, and the dataset suppresses the category it rests on.** Removing the constant is a model-wide lesion that produces the loop on every input; an oscillatory hallucination is an input-conditional failure of a working model, and Guerreiro et al. 2023 find those are "less related to model uncertainty" and reverse under fallback. Worse, HalOmi's outputs were generated with **repetition of 4-grams forbidden** (HalOmi §2.2: beam 5, no-repeat-4-gram, length ≤3·src+5), so loops of the kind §2 describes are decoded away before annotation. *Fix:* drop the oscillation story from H2's justification; state a directional prediction that Arm C can supply (attenuating the constant lowers sink mass and raises TNG) and test only whether natural hallucinations sit on the same side (lower sink mass than clean). Anything more is not decidable in budget.

2. **"The constant's magnitude at the decoder's first positions" has no per-sentence variance to carry a signal.** Sun et al. 2024 and Owen et al. 2025 report the value is input-agnostic; the proposal itself calls it a bias. A near-constant feature contributes nothing to a logistic regression. *Fix:* cut it, or replace with what *can* vary — sink attention mass per query step and massive activations at later delimiter tokens (Sun 2024 §2), pre-registered.

3. **The proposal does not say where the sink is, and NLLB's known sink is the wrong kind.** In HF NLLB the decoder input is `[</s>, tgt_tag, …]` (position 0 is `</s>`, the tag is position 1); the tech report's "tag first" describes the target sequence before the EOS shift, and `notes_multilingual_mt_interpretability.md` currently asserts "tag at decoder position 0". Cross-attention mass sits ~75–87% on the source `</s>` (HalOmi App. B; arXiv:2605.01229 Table 1), and HalOmi notes that token's encoder state is an order of magnitude **smaller** than others — the opposite of a massive-activation sink. Decoder self-attention has never been measured. *Fix:* the Arm C NLLB gate (per-position residual norms, both stacks; sink rate in self-, cross- and encoder attention) must run before Arm B, and Arm B must pre-register which attention type and position is "the sink".

4. **A known negative is uncited.** HalOmi Appendix B tests cross-attention-derived detectors (Wass-to-Unif/Data/Combo/Mean) with and without the EOS token: 0.49–0.55 AUC, "not much better than chance", attributed to the EOS sink. Cross-attention sink statistics are therefore already dead; the proposal must say so and confine the claim to decoder self-attention and residual-stream features.

5. **Wrong baseline.** Dale 2023a's Table 1 (Seq-Logprob 83.0 / ALTI 84.9 AUC on all hallucinations; 93.5 / 98.7 on fully detached; LaBSE 91.7) is for a WMT18 de–en transformer with 323/3415 hallucinations, not NLLB. The NLLB bar is HalOmi Fig. 5 (Seq-Logprob "most stable", BLASER-2.0-QE best), reported per direction as a figure, not a table. *Fix:* recompute Seq-Logprob, ALTI+ (Ferrando's code is encoder-decoder-native) and TNG on the same teacher-forced pass; never copy numbers across models. Budget the ALTI+ install.

6. **n forbids per-direction AUC with intervals.** HalOmi has 144–197 sentences per direction with ≥3% hallucinated (1% full): as few as 5 positives in high-resource directions, ~40–60 in the worst-case-sampled low-resource ones. Per-direction AUC CIs will span ±0.1–0.2; a nested ΔAUC of 0.02–0.03 is undetectable there. *Fix:* pool with direction-stratified CV, report ΔAUC(logprob+features vs logprob) by DeLong or paired bootstrap, split only by resource group (family = 3), and pre-register the minimum detectable ΔAUC.

7. **The 3.3B arm measures a different question.** Teacher-forcing 600M's own outputs through 600M reproduces its decoding-time internals exactly (hidden states depend only on the prefix; Dale 2023a and HalOmi do the same for ALTI+). Forcing them through 3.3B measures a *second model's* opinion of text it did not produce — external QE, the BLASER-QE regime, not an internal detector. *Fix:* label it so, or drop it.

8. **Redundancy is only half decidable.** Step entropy and forced-token logprob are functions of the same logits; Guerreiro et al. 2022 already found logit-derived heuristics lose to Seq-Logprob. Sink mass comes from attention, so its increment is testable — but only if the comparison is against the *stacked* existing detectors (Seq-Logprob + ALTI+ + TNG; cf. Guerreiro et al. 2024, STARE aggregation), not against logprob alone.

## (c) Cut / keep / sharpen

**Cut:** the constant-magnitude feature; per-direction AUC tables; the 3.3B "transfer" framing; the oscillation analogy as motivation.
**Keep:** teacher-forced 600M on HalOmi with Seq-Logprob/ALTI+/TNG recomputed in-house; pooled nested ΔAUC with intervals.
**Sharpen (the interpretability content):** HalOmi has **word-level** hallucination spans. Ask whether decoder self-attention sink mass shifts *before* the first hallucinated token versus matched clean positions in the same sentence. That is token-level (n in the thousands), tests cause-versus-consequence, and is the only version of H2 that is about mechanism rather than a benchmark row. Binkowski's finding that the signal lives in sinks with large value norms gives a concrete feature to pre-register.

## (d) Most / least decidable

**Most decidable:** does decoder self-attention sink mass add AUC to Seq-Logprob on pooled HalOmi, with a DeLong interval. Either answer is a result and costs a few GPU-hours.
**Least decidable:** whether the lesion-induced function-word loop and natural oscillatory hallucinations share a mechanism — the dataset blocked the loops, the constant does not vary per input, and no intervention in the proposal produces a natural hallucination.

Sources checked online: [arXiv:2605.01229](https://arxiv.org/abs/2605.01229), [HalOmi](https://arxiv.org/html/2305.11746v2), [Binkowski 2026](https://arxiv.org/html/2604.10697v1), [Guerreiro et al. 2024 detector aggregation](https://arxiv.org/pdf/2402.13331).
