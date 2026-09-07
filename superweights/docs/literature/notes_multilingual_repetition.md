# Notes — does repetition / degeneration vary by language?

Compiled 2026-09-05 for the super-weight project's bridge question: **is the
"natural" repetition of a healthy multilingual model the same failure as the
repetition a super-weight lesion induces**, and in particular whether the folk
claim *"multilingual models loop more in low-resource languages"* is a measured
result.

Reading protocol, per the shared brief. Everything marked **[read]** was
extracted from a PDF with `pdftotext` and the section/table/figure numbers point
at what was actually on the page. **[unopened]** means I never opened it.
Sentences beginning *Inference:* are my reasoning past the papers, not their
claims. Venues are stated as printed on the PDF.

**Companion notes.** `notes_repetition_degeneration.md` covers the degeneration
mechanism literature (Holtzman, Welleck, Fu, Xu, Li, Hiraoka, Yao) and the MT
hallucination taxonomy; `notes_multilingual_mt_interpretability.md` covers the
multilingual/MT interpretability side. This file does not repeat them — it
covers only the **cross-lingual axis of repetition**, which neither had as its
question.

**Headline, stated up front so §1 can be read against it.** After going through
every paper below: *no published work measures the repetition rate of a healthy
multilingual model as a function of language, holding decoding fixed.* The
closest measurements are (a) Guerreiro et al. TACL 2023, which measures the
**share** of hallucinations that are oscillatory by resource level — a
composition, not an incidence; and (b) Barua et al. ICLR 2026, which measures an
error category that **conflates** repetition with format violations. Details in
§4.

---

## §1 Paper by paper

### 1.1 The one paper everyone cites for the resource-level asymmetry

**Guerreiro, Alves, Waldendorf, Haddow, Birch, Martins & Voita 2023,
*Hallucinations in Large Multilingual Translation Models*, TACL 2023.**
**[read]** §4.1–4.2, §5.1–5.3, §6.2, Tables 1–4, Figure 2 caption, Appendix D.

*Setup.* M2M-100 (418M / 1.2B / 12B, written S / M / L), SMaLL100, NLLB-3.3B and
ChatGPT. FLORES-101 in the main text, WMT and TICO in appendices. Three setups:
English-centric (64 LPs), non-English-centric (25 LPs), medical TICO (18 LPs).

*Decoding — load-bearing.* §3: "For decoding, we run **beam search with a beam
size of 4**. All experiments were run on fairseq." **No repetition penalty and no
n-gram blocking are mentioned anywhere in the paper.** *Inference:* this makes
their oscillatory numbers among the very few in the MT literature that measure a
decoder actually free to loop (contrast HalOmi, §1.3).

*Detectors.* ALTI+ (source contribution) for **detached** hallucinations; **TNG**
(top repeated n-gram: count of the top repeated *translation* n-gram exceeds the
count of the top repeated *source* n-gram by at least `t`) for **oscillatory**
ones, with **n = 4, t = 2**, following Raunak et al. 2021. TNG was validated on
human-annotated hallucinations at "perfect precision" (their words, citing
Guerreiro et al. 2022b). ALTI+ thresholds set at the lowest 0.02% of the score
distribution on high-resource WMT.

*Table 2 — English-centric, hallucination rate of ANY type (mean %, median in
subscript), by resource level:*

| Model | Low: LPs / rate | Mid: LPs / rate | High: LPs / rate |
|---|---|---|---|
| SMaLL100 | 14/16 · 2.352 (med 0.57) | 19/38 · 0.055 (0.02) | 1/10 · 0.005 (0.00) |
| M2M (S) | 15/16 · **15.20** (2.86) | 22/38 · 0.254 (0.05) | 3/10 · 0.025 (0.00) |
| M2M (M) | 14/16 · **12.53** (1.42) | 17/38 · 0.110 (0.00) | 2/10 · 0.010 (0.00) |
| M2M (L) | 14/16 · **11.22** (2.19) | 11/38 · 0.034 (0.00) | 0/10 · 0.000 (0.00) |

*Table 3 — by direction, average hallucination rate (%) over all LPs:*
SMaLL100 0.221 (xx→en) vs 1.022 (en→xx); M2M(S) 1.756 vs 6.152; M2M(M) 2.290 vs
4.110; M2M(L) 2.483 vs 3.169. "Models are significantly more prone to hallucinate
when translating **out of** English", and ALTI+ source contributions are lower in
that direction across all LPs.

*Table 4 — non-English-centric:* SMaLL100 5/10 · 2.160 | 6/13 · 0.054 | 1/2 ·
0.025; M2M(S) 10/10 · 12.61 | 12/13 · 0.467 | 1/2 · 0.075; M2M(M) 7/10 · 12.22 |
7/13 · 0.172 | 0/2 · 0.000; M2M(L) 6/10 · 6.580 | 4/13 · 0.077 | 0/2 · 0.000.

*The oscillatory claim itself.* §5.2: "in contrast to mid- and high-resource
language pairs, **oscillatory hallucinations are less prevalent, while detached
hallucinations occur more frequently in low-resource languages**." §5.3, for the
non-English-centric setup: "detached hallucinations are more prevalent in
low-resource settings, while **oscillatory hallucinations overwhelmingly dominate
in mid- and high-resource directions**." And §5.2, on direction: "**nearly all
hallucinations into English for mid- and high-resource language pairs are
oscillatory.**"

*Figure 2 is a heatmap and there is no numeric table behind it.* Its caption:
"Heatmap of the percentage of hallucinations detected with TNG (oscillatory
hallucinations) **among all hallucinations**." The same heatmap recurs as Fig. 9
(WMT), Fig. 11 (non-English-centric) and Fig. 13 (TICO), each time as a figure
only. **I could not extract a single per-language oscillatory percentage from
this paper's text.** Anyone citing a number from Fig. 2 is reading pixels.

*The trap, and it is a big one.* **[Inference — mine, and I believe correct]**
Fig. 2 measures a **conditional** quantity: oscillatory hallucinations *as a
fraction of hallucinations*. It is not the oscillatory rate per sentence. The
denominator varies by three orders of magnitude across resource levels (Table 2:
15.20% low vs 0.025% high for M2M(S)). So "oscillatory dominates at high
resource" is fully compatible with the **absolute** oscillatory rate being far
higher at low resource. Concretely, for M2M(S): even if only 10% of low-resource
hallucinations were oscillatory, that is ~1.5% of low-resource sentences; at high
resource "nearly all" of 0.025% is ~0.025% — a ~60× absolute gap the other way.
The exact factor is not computable from the paper's text because the numerator
share exists only as a heatmap. **The widely repeated statement "oscillatory
hallucinations dominate in mid/high-resource directions" is about composition and
is routinely mis-cited as being about incidence.** This is the single most
important thing in this file.

*Architecture, not language.* §4.2, on hallucinations under source perturbation:
"unlike traditional NMT models that frequently produce oscillatory character
hallucinations, **ChatGPT does not generate any such hallucinations under
perturbation**"; its errors are instead off-target translations, overgeneration,
or refusals. §6.2: used as a fallback system, "oscillatory hallucinations are
almost entirely eliminated… ChatGPT produces very few, if any, oscillations."
*Inference:* the largest documented difference in oscillation rate in this
literature is **encoder-decoder NMT vs decoder-only LLM**, not high- vs
low-resource. That is a confound Adrian's design must not walk into, since
NLLB-200 is enc-dec and Tower/EuroLLM/Aya/BLOOM are decoder-only.

*Limitations the paper states itself:* results are M2M-family; "It is unclear how
our findings generalize to other families of multilingual models (e.g., the NLLB
family)."

### 1.2 The original oscillatory-hallucination paper

**Raunak, Menezes & Junczys-Dowmunt 2021, *The Curious Case of Hallucinations in
NMT*, NAACL 2021.** **[read]** §3–5, Tables 5–6, Figures 4–5.

*Setup.* IWSLT-2014 **De-En** (160K pairs) plus WMT14 De-En as the noise source;
Transformer; validation BLEU with **beam = 5**. Four corpus-level noise patterns
(UU / UR / RU / RR) injected at ~12% of the corpus. Evaluated on an "invalid
reference set" (IRS) of **21 sentences** and a valid reference set.

*The only per-direction repetition numbers in this literature.* Same
architecture, same noise, both directions of the same pair:

| Noise | IRS-OH (De-En, Table 5) | IRS-OH (En-De, Table 6) | Unique-bigram frac De-En / En-De |
|---|---|---|---|
| U-U | 0.0% | 0.0% | 0.80 / 0.91 |
| U-R | 0.0% | 0.0% | 0.84 / 0.95 |
| R-U | **76.19%** | **47.61%** | **0.19 / 0.49** |
| R-R | 0.0% | 0.0% | 0.96 / 0.99 |
| none | — | — | 0.925 / 0.95 |

*Caveats.* n = 21, manually identified, so 76.19% = 16/21 and 47.61% = 10/21 —
a difference of six sentences. This is **not** a resource-level contrast (De and
En are both high-resource); it is a direction contrast within one pair, in a
bilingual model, under an artificial training-noise intervention. The unique-
bigram fraction (0.19 vs 0.49 under RU noise, against ~0.93–0.95 clean) is the
usable quantity: it is a decoding-agnostic repetition measure that *does* differ
by direction. Cite it as "direction-dependent under induced noise, n=21", nothing
stronger.

**Guerreiro, Voita & Martins 2023, *Looking for a Needle in a Haystack*, EACL
2023.** **[read]** §3–4, App. A. **WMT2018 DE-EN only**, beam 5. Confirms TNG
finds most oscillatory cases and that "Repeated targets" is a poor detector.
Single language pair — no cross-lingual content at all. It is the source of the
TNG validation everyone else cites.

### 1.3 HalOmi — and why it cannot see repetition

**Dale, Voita, Lam, Hansanti, Ropers, Kalbassi, Gao, Barrault & Costa-jussà 2023,
*HalOmi*, EMNLP 2023.** **[read]** §2.2, §2.3, §3.

18 translation directions, sentence- and word-level human annotations of
hallucination and omission on **NLLB-200-distilled-600M** output, over FLORES-200
dev plus out-of-domain Jigsaw/Wikipedia-talk text.

*Decoding — the reason this paper is in this file.* §2.2: "We translated these
texts with the 600M distilled NLLB model following the standard setting (**beam
size 5**, forbidden generation of the `<UNK>` token, **forbidden repetition of
4-grams**, limiting the translation length to 3·len(source)+5 tokens)."

**`no_repeat_ngram_size = 4` is on.** Oscillatory hallucination in Guerreiro's
sense (TNG at n=4) is *definitionally* suppressed by this decoder. HalOmi's
per-direction rates therefore say nothing about repetition, and it also cannot be
used as a null for a repetition study. It also discards outputs annotators mark
"incomprehensible / garbled", which is where a degenerate loop would land.
*Inference:* this is the standard fairseq/HF NLLB generation config, so **most
published NLLB numbers are from a repetition-blocked decoder** — including,
probably, whatever the MT lab's existing NLLB pipeline does. Check before
measuring anything.

### 1.4 A lesion in a multilingual MT model that does produce repetition

**Koishekenov, Berard & Nikoulina 2023, *Memory-efficient NLLB-200:
Language-specific Expert Pruning of a Massively Multilingual MoE Model*, ACL
2023.** **[read]** §6.4 and Appendix A.

Pruning up to 80% of the MoE experts of NLLB-200-54.5B. §6.4: "a rare but visible
phenomenon of **over-generation** (and sometimes hallucinations). In the majority
of cases, the translation is accurate initially but **subsequently includes
repetitions**, paraphrasing, or slight hallucinations."

Numbers, App. A: global-threshold pruning at 80% costs **1.75 spBLEU** vs 0.53
for fixed-per-layer; the **length-ratio delta** vs the full model is **+0.16**
(global) vs +0.04 (fixed-per-layer). Their metric calibration is directly
reusable: duplicating each FLORES en→fr output sentence (concatenating it with
itself) costs **47% spBLEU but only 13% chrF++**. They hypothesise the cause is
damage to experts specialised in emitting `</s>`, and leave it to future work.

*Why it matters here.* This is the one existence proof that a **structural lesion
in NLLB-200** produces the repetition/over-generation signature, in a
multilingual MT model, at scale. It reports **no per-language repetition rates**
and does not connect to sinks, massive activations or super weights. The
spBLEU-vs-chrF++ asymmetry is a free, well-calibrated repetition detector for
Adrian's runs.

**Kojima, Okimura, Iwasawa, Yanaka & Matsuo 2024, *Finding and Controlling
Language-Specific Neurons*, NAACL 2024.** **[read]** §5.2. Decoding: random
sampling, **temperature 0.8, top-p 0.9**, 100 seeds. Intervening on
language-specific neurons under an ambiguous translation prompt makes BLOOM
"generate text unrelated to the translation or simply **repeat the source text**
in English." That is source copying, not degenerate looping — a different failure.
No repetition rates.

### 1.5 Language confusion — adjacent, and not about repetition

**Marchisio, Ko, Bérard, Dehaze & Ruder 2024, *Understanding and Mitigating
Language Confusion in LLMs*, EMNLP 2024.** **[read]** §5, §6.1, Table 7, Table 8,
App. A.10.

**It does not report repetition anywhere.** Its only mention of the concept is
App. A.10's background sentence, "Previous work shows that greedy search
empirically leads to repetition and generally low-quality generations," citing
Holtzman. Metrics are LPR (line pass rate) and WPR (word pass rate). So: on the
brief's question, this paper is a negative.

What it *does* contribute is the **method template** for §5 of this file. §5
studies "confusion points" (CPs) — the first position where a wrong-language
token is emitted — on 15 Chinese prompts to Command R (k=0, p=0.75, t=0.7, 100
tokens each: 1500 sampling points, **9 CPs**), and compares the local
distribution at CPs against non-CPs in the same generations (Table 7):

| | avg nucleus size | avg entropy |
|---|---|---|
| @ CP | **3.56** | **1.228** |
| ¬@ CP | 1.61 | 0.356 |
| outputs with no CP at all | 1.61 | 0.365 |

i.e. the failure is preceded by a **flat, high-entropy** next-token distribution,
and outputs that never fail look identical on average to the clean parts of
outputs that do. That is exactly the "matched clean positions within the same
generation" control the super-weight hypothesis needs — see §5.

§6.1, Table 8: WPR degrades with temperature — at T=1 average monolingual WPR is
**83.5%**, and **72.0% for Japanese, 69.5% for Chinese**; T=0 is greedy. Shrinking
the nucleus helps less than lowering T.

**Nie, Schmid & Schütze 2025, *Mechanistic Understanding and Mitigation of
Language Confusion in English-Centric LLMs* (arXiv:2505.16538v2, 17 Sep 2025;
venue not printed on the PDF — treat as preprint).** **[read]** abstract + §
headers. TunedLens layer-wise analysis plus neuron attribution on the same LCB
confusion points; "transition failures in the final layers drive confusion";
editing a small critical neuron set mitigates it. **No repetition metric.** The
only occurrence of "repeat" in the PDF is "we repeat the analysis".

**Chen, Kong et al. 2023, Off-target in zero-shot multilingual NMT (Findings of
ACL 2023).** **[read]** grep of full text: **zero** occurrences of "repetit",
"repeat", "degenerat", "oscillat". Off-target only.

### 1.6 Repetition neurons and the one language contrast in that literature

**Hiraoka & Inui 2024, *Repetition Neurons* (NAACL 2025 short).** **[read]**
§2–3 and **Appendix D in full**.

Four models: Gemma-2B, Pythia-2.8B-deduped, Llama-3.2-3B (English) and
**LLM-jp-3-1.8B** (Japanese). Greedy decoding. |X| = 1,000 repetitive
generations, activation window r = 30.

Appendix D, the language result: deactivating repetition neurons costs the three
English models only moderate WikiText-2 perplexity, but "**Unlike the English
models, LLM-jp-1.8B's perplexity increases substantially even when a smaller
number of repetition neurons are deactivated.** This result implies that
repetition neurons may be **language-specific**." Symmetrically, activating them
sends Gemma-2B past PPL 100 at 500 neurons and Llama-3.2-3B past 100 at 800,
"whereas LLM-jp-1.8B's perplexity remains largely unaffected."

*Caveat the paper itself raises:* the test corpus is **English** WikiText-2 for
all four models, LLM-jp included. Their own reading is that "neurons identified as
repetition neurons in Japanese may serve a different role in English texts." So
the asymmetry may be a cross-lingual transfer artifact, not a property of
Japanese repetition. **No repetition rate by language is reported anywhere in the
paper**, and resource level is never mentioned. This is the only
language-contrastive result in the whole repetition-mechanism literature, and it
is about *neuron ablation sensitivity*, not about how often the model loops.

**Everything else in the degeneration literature is English-only.** Verified by
full-text grep: Holtzman 2019 (0 mentions of any non-English language), Xu et al.
2022 (Wikitext-103, book, random-token corpora), Fu et al. 2021, Li et al. 2023
(Wikitext), Yao et al. 2025 (repeat-curse SAE features). None has a cross-lingual
arm. There is no multilingual replication of self-reinforcement, high-inflow, or
repetition-feature results.

### 1.7 Neuron ablation by language: code-switching, not repetition

**Tang, Wang et al. 2024, *Language-Specific Neurons: The Key to Multilingual
Capabilities* (LAPE).** **[read]** §3.1–3.2, Table 2.

*Decoding:* "We utilize **greedy search with a repetition penalty of 1.1** to
generate output" (§3.1). **A repetition penalty is on**, so this setup cannot
observe looping either.

Table 2, LLaMA-2-70B answering a Simplified-Chinese prompt with zh-specific
neurons deactivated: the output becomes "a chaotic mixture of **Traditional
Chinese characters and redundant English phrases**" — script drift and
code-switching ("我是一個…I am a mountaineer who has climbed…當我站在珠my朗ma峰頂峰"),
**not** a repetition loop. Compare the super-weight lesion, which gives
"We. We. We." from the first token. *Inference:* language-specific-neuron
ablation and super-weight ablation produce **different** failure signatures —
consistent with the super weight being a model-wide sink/bias component rather
than a language-routing one. Worth stating explicitly in the writeup; it is a
free contrast and it is not confounded by decoding, because if anything the
repetition penalty biases Tang's setup *away* from finding repetition.

### 1.8 Technical reports of the models Adrian actually has

| Report | Repetition / degeneration content |
|---|---|
| **Alves et al. 2024, Tower (COLM 2024)** **[read]** §3.3, App. A | Decoding: "**All model generations are performed with greedy decoding**" (§3, App. F); App. A adds beam-5 and MBR (sampled at t=0.9, p=0.6), both of which beat greedy on quality. Only repetition mention in the paper: TowerInstruct as an APE post-editor improves NLLB-3.3B output "going as far as **converting oscillatory hallucinations into high-quality translations** (Figure 5)". No rates, no per-language breakdown, no measurement of Tower's own repetition. |
| **EuroLLM-9B technical report (2025)** **[read]** full-text grep | **Zero** occurrences of "repetit", "repeat", "degenerat", "oscillat". |
| **Martins et al. 2024, EuroLLM** **[read]** full-text grep | "Repeat" appears only about **repeating training data** (§2.2.2, Fig. 3: repeating Wikipedia improves Wikipedia test loss without degrading web test loss). Nothing about output repetition. |
| **NLLB-200 team 2022 (arXiv:2207.04672)** **[read]** full-text grep | "Repeated characters" appears once, as a **data-filtering** heuristic (§ on LID/filtering: "maximum number of repeated characters"). No output-repetition analysis, no per-language repetition rate, and the standard decode config is not discussed in repetition terms. |
| **BLOOM (arXiv:2211.05100)** **[read]** full-text grep | **Zero** occurrences of "repetit", "degenerat", "repetition penalty". |
| **Aya Expanse technical report (arXiv:2412.04261)** **[unopened]** — abstract and HF/blog summaries only | No repetition metric surfaced in any search result; evaluation is m-ArenaHard / Dolly win-rates against GPT-4o as judge. Treat as "not reported" pending an actual read. |
| **Apertus (2025)** **[read]** §5.4.1, Table 25 | Not multilingual on this axis (Gutenberg, English), but the cleanest decoding datum available: under **greedy** decoding, "significant degeneration occurs" — TTR 0.22–0.31 against a ground-truth ~0.539; under **nucleus (T=1.0, p=0.9)** TTR ≈ 0.500. Same model, same prompts: **decoding rule moves the repetition metric by ~2×**. |

**Üstün et al. 2024, *Aya Model*, ACL 2024.** **[read]** §5.4, App. E.6.
Decoding for the human/GPT-4 preference evaluation: **nucleus sampling, T = 0.9,
top-p = 0.8**, max 256 tokens. Human annotators over 7 languages (English,
Serbian, Spanish, Russian, Hindi, French, Arabic):

> "The most commonly reported issues were that Aya generations were **repetitive
> or contained hallucinated 'loops' or 'drifted off'**, were semantically
> incoherent or convoluted, contained grammar mistakes (**especially for Russian
> and Serbian**) and weird word choices…"

and §5.4's conclusion: generations "have **clear quality differences across
languages**, and can be expected to contain grammar and factuality errors,
**repetitions**, hallucinations and unnatural structures. We suspect that
translation errors in the finetuning data, especially due to their
language-specific systematicity, could be largely contributing to these issues."
App. E.6 repeats the observation for Serbian and Russian specifically.

*This is the single most-cited-adjacent piece of evidence for the folk claim, and
note what it actually says.* It is **qualitative annotator feedback, with no
rate**; the languages flagged are **Russian and Serbian**, which are not the
lowest-resource languages in the set; and the explanation offered is
**finetuning-data translation artifacts**, not resource level. It does not
support "worse in low-resource languages."

### 1.9 The two papers that come closest to a measurement

**Barua, Eisape, Yin & Suhr, *Long Chain-of-Thought Reasoning Across Languages*,
ICLR 2026 (arXiv:2508.14828v3, 20 Mar 2026).** **[read]** §7, App. A.4/A.9, B.1.
Downloaded as `Barua-2026-Long-CoT-Reasoning-Across-Languages.pdf`.

Nine non-English languages, three per resource tier; the low tier is **Marathi,
Telugu, Swahili**. Three settings: En-Only, En-CoT (non-English input, English
reasoning), Target-CoT (input and reasoning in the target language). Decoding:
**temperature 0.6, top-p 0.95, max 32,768 output tokens**, plus "language
forcing" (App. A.4). **No repetition penalty stated.**

Error analysis (Fig. 4): incorrect responses from DeepSeek-R1-Distill-Llama-70B
on AIME-Combined, classified into five categories by Gemini-2.5-Pro, validated by
native speakers on 20 samples each for Chinese, Telugu, French, Japanese.

| Error type | En-CoT | Target-CoT |
|---|---|---|
| Reasoning | 47.6% | 34.4% |
| Conceptual | 21.4% | 24.9% |
| **Output Generation** | **0.7%** | **11.3%** |

"Output Generation" is defined in the App. A.9 taxonomy as "Output violates
constraints (format) or **degenerates (stability)**", example: "Missing boxed
output; **repetitively generates 'The next step is'**." §7: "Target-CoT, however,
introduces additional barriers — **unstable generation (e.g., endless repetition)
in low-resource languages** — that often prevent models from reaching the
reasoning stage entirely." Per-language failure modes named: Telugu's conceptual
error rate doubles in Target-CoT (28% vs 14%); Latvian holds ~34% comprehension
errors either way.

*Three caveats, all fatal for citing this as "repetition is worse in low-resource
languages".* (i) The category **conflates a missing `\boxed{}` with a
degeneration loop**; repetition is never counted on its own. (ii) The denominator
is **incorrect responses**, so it is a share of errors, not a rate over
generations — if accuracy is lower in low-resource languages, the denominator
moves too. (iii) Figure 4 does have per-language panels (Japanese, Latvian,
Telugu, Swahili) but **the numbers are in the bars only**; the 16× En-CoT →
Target-CoT jump is the only figure in the text. What this paper genuinely
establishes is that *reasoning in the target language* rather than in English
shifts errors toward generation instability — a **prompt/generation-language**
effect, which is not the same as a resource effect.

**Liu, Zhao, Hedderich & Schütze 2026, *Crosslingual On-Policy Self-Distillation
for Multilingual Reasoning* (arXiv:2605.09548v1, 10 May 2026) — preprint,
unrefereed.** **[read]** §6.4, App. B. Downloaded as
`Liu-2026-Crosslingual-SelfDistillation-Repeat-Rate.pdf`.

Contributes the **metric definition** worth reusing:

> RepeatRate_n = 1 − |G_n^unique| / |G_n|

over all contiguous n-grams of a generation, computed for **n ∈ {2,3,4,5,6}**.
Qwen3-1.7B/4B/8B on 17 low-resource African languages (AfriMGSM) plus PolyMath in
8 languages spanning tiers (Swahili, Telugu / Thai, Russian, Bengali, Japanese,
Chinese, Spanish). Decoding: **temperature 1.0, top-p 0.95**, up to 8,192 tokens.

Fig. 5 and App. B Fig. 9 plot average repeat rate against training step for base
vs GRPO vs COPSD, at n = 1…6; COPSD is lowest throughout. **The repeat rate is
averaged over languages — there are no per-language numbers and no high-resource
comparison arm**, so this paper cannot and does not test the resource claim.

*It is, however, the clearest instance of the folklore citation chain.* §6.4:
"Prior work has identified repetition as a common failure mode in multilingual
reasoning, particularly in low-resource languages **(Barua et al., 2026; Tran et
al., 2025)**." Barua is §1.9 above — a conflated error category. Tran, O'Sullivan
& Nguyen 2025 (arXiv:2504.02890, "Reasoning transfer for an extremely low-resource
and endangered language") is **[unopened]** and is a single-language study by its
title. So the sentence rests on a paper that does not isolate repetition and a
paper with no cross-language comparison.

### 1.10 Explicitly checked and found to contain nothing usable

- **Wendler et al. 2024 (ACL), Schut et al. 2025, Trinley et al. 2025.** All three
  use a **"repetition task"** — prompt the model to repeat the last word — as a
  *probe* for the language of intermediate representations. **This is not
  degeneration** and must not be cited as such. Schut and Trinley contain no
  degeneration content at all. **[read]** greps.
- **Marchisio et al. 2024 quantization-multilingual; UnevenPTQ 2025.** Full-text
  grep: no repetition/degeneration mentions, despite both studying multilingual
  quality collapse.
- **Achilles et al. 2025, *Altering Neurons Cripples Language*.** Full-text grep:
  **zero** occurrences of repetition/degeneration/loop. Despite the title, the
  cripple signature is not characterised as repetition.
- **Lucie-7B (2025).** "Repetition removal" appears once, as a data filter.

---

## §2 Summary table

Columns: does the source contain *language-resolved* evidence about repetition;
what decoding it used (this determines whether repetition was even possible);
what to take from it.

| Model family / source | Language-resolved repetition evidence? | Decoding config | What it actually shows |
|---|---|---|---|
| **M2M-100 (S/M/L), SMaLL100** — Guerreiro TACL 2023 | **Partial, and conditional.** TNG-oscillatory *share* by LP, heatmap only (Figs. 2/9/11/13). Total hallucination rate by tier is tabulated (Table 2). | **Beam 4, fairseq. No repetition penalty, no n-gram blocking.** | Oscillatory *share* is lower at low resource; total hallucination rate is ~500× higher at low resource. **Absolute** oscillatory rate by tier is not reported by anyone. |
| **ChatGPT** — same paper, §4.2 / §6.2 | Yes, contrastive against NMT | Prompted, temperature 0 attempted | Produces essentially **no** oscillatory hallucinations. Architecture/training effect, not a language effect. |
| **Bilingual Transformer (IWSLT De-En)** — Raunak NAACL 2021 | **Yes, by direction, n=21.** IRS-OH 76.19% (De-En) vs 47.61% (En-De) under R-U noise; unique-bigram 0.19 vs 0.49. | Beam 5 | Direction changes the oscillation rate under induced training noise. Not a resource contrast; tiny n. |
| **NLLB-200-600M-distilled** — HalOmi, Dale EMNLP 2023 | **No — structurally blind.** | **Beam 5 + `no_repeat_ngram_size=4`** + length cap 3·len+5; garbled outputs discarded | Cannot observe oscillatory hallucination. Per-direction rates in HalOmi are not repetition evidence. |
| **NLLB-200-54.5B MoE (pruned)** — Koishekenov ACL 2023 | No per-language rates | Not stated in the sections read | **Lesion → repetition/over-generation** in a multilingual MT model. Length-ratio Δ +0.16 at 80% global pruning; spBLEU −1.75 vs chrF++ far less. |
| **LLaMA-2-70B, BLOOM, others** — Tang 2024 LAPE | Yes, but the failure is **code-switching**, not repetition (Table 2) | **Greedy + repetition penalty 1.1** | Language-neuron ablation ≠ super-weight ablation signature. |
| **XGLM / BLOOM / Llama-2** — Kojima 2024 | Source **copying** under intervention, not looping | Sampling, T=0.8, p=0.9 | Not degeneration. |
| **Command R / R+, Llama, Mistral…** — Marchisio EMNLP 2024 | **No repetition metric at all** | k=0, p=0.75, t=0.7 for the CP study; T swept 0–1 for Table 8 | Gives the CP-vs-¬CP within-generation control design (nucleus 3.56 / entropy 1.228 @CP vs 1.61 / 0.356 ¬@CP) and the T→confusion curve (WPR 83.5% avg, ja 72.0%, zh 69.5% at T=1). |
| **Gemma-2B / Pythia-2.8B / Llama-3.2-3B / LLM-jp-3-1.8B** — Hiraoka NAACL 2025 App. D | **Yes, one contrast:** Japanese model's repetition neurons behave differently under ablation/activation | Greedy | Repetition neurons "may be language-specific" — but measured as *English* WikiText-2 perplexity for all four models, and no repetition rate. |
| **DeepSeek-R1-Distill-Llama-70B + 8 others** — Barua ICLR 2026 | **Closest to yes.** "Output generation" errors 0.7% (En-CoT) → 11.3% (Target-CoT); per-language bars in Fig. 4 | T=0.6, p=0.95, 32,768 tokens; language forcing | Generation-language effect on a category that **conflates repetition with format violation**; denominator is incorrect answers only. |
| **Qwen3-1.7B/4B/8B, 17 African languages** — Liu 2026 (preprint) | **No** — repeat rate averaged over languages, no high-resource arm | T=1.0, p=0.95, 8,192 tokens | Supplies RepeatRate_n. Its "particularly in low-resource languages" sentence is a citation to Barua + an unopened single-language paper. |
| **Aya-101** — Üstün ACL 2024 | **Qualitative only.** Annotators report loops; the languages named are **Russian and Serbian** | Nucleus, T=0.9, p=0.8, 256 tokens | "Clear quality differences across languages", attributed to finetuning-data translation artifacts, not resource level. |
| **Aya Expanse** [unopened] | Not found | — | Not reported. |
| **Tower-7B/13B** — Alves COLM 2024 | No | **Greedy** (App. A: beam 5, MBR t=0.9/p=0.6) | Only that TowerInstruct *repairs* NLLB's oscillatory hallucinations. |
| **EuroLLM-9B**, **EuroLLM**, **BLOOM**, **NLLB-200** tech reports | **No — zero mentions** | — | The claim is absent from every technical report of every model in Adrian's set. |
| **Apertus-8B/70B** — 2025 | English only | Greedy vs nucleus (T=1.0, p=0.9) | Same model, same prompts: TTR 0.22–0.31 (greedy) vs ≈0.500 (nucleus). |

---

## §3 The explanations on offer, and whether any is mechanistic

Five candidate accounts appear in the literature. **None of them has been tested
against a per-language repetition measurement**, because no such measurement
exists (§4). Ordered by how much evidence actually stands behind them.

**(a) Data share / supervision per direction.** Guerreiro TACL §5.2 shows total
hallucination rate falling monotonically with resource level, and attributes it to
parallel-data volume; §5.3 shows the least-supervised pairs (ro-hy, af-zu) worst.
Li et al. NeurIPS 2023 shows, in English, that generated repetition tracks
**repetition in the training data**. Combining them is the natural hypothesis —
low-resource bitext is mined, noisy and duplicate-heavy, so it should carry more
repeated targets — and Raunak's R-U noise pattern (repeated source, unique target)
is precisely the mined-bitext pathology that produced 76% oscillation. **Status:
plausible, assembled from three papers, tested by none.** Nobody has measured
duplicate rate per language in NLLB/BLOOM/Aya training data against repetition
rate per language in their output. **Not mechanistic.**

**(b) Tokenizer fertility.** The intuitive story: high fertility (more subwords
per word) makes loops longer in token count, makes an n-gram-based detector fire
differently, and lengthens the horizon over which self-reinforcement can build.
**Status: I could not find a single paper connecting fertility to repetition.**
The fertility literature (Ács; ACL/arXiv 2508.06533, 2510.09947, 2606.15044 — all
**[unopened]**, from search listings) connects fertility to downstream accuracy
and inference cost only. This is an open, cheap, and genuinely unclaimed question.
**Not mechanistic; not even correlational yet.**

**(c) Script.** HalOmi varies script deliberately across its 18 directions and
reports that single-pair conclusions do not generalise, but does not report a
script effect on repetition (and could not, §1.3). Tang's Table 2 shows script
*mixing* under ablation. **Status: hypothesised, unmeasured.**

**(d) Off-target / language confusion as the same underlying failure.** Guerreiro
§5.2: "over 90% of off-target hallucinations occur when translating out of
English"; Marchisio locates confusion at flat, high-entropy positions.
Structurally this is attractive — both off-target drift and looping are the
model failing to keep the target distribution sharp — and Marchisio's CP analysis
is the only one that measures a **local, pre-failure** signal. But Marchisio never
measures repetition, and Guerreiro finds off-target and oscillatory to be
*different* categories with different resource profiles. **Status: two adjacent
literatures with no bridging measurement. Marchisio's is the most mechanistic
method in this file, applied to the wrong failure.**

**(e) Calibration / entropy collapse, and the sink route (this project's H).**
Marchisio's Table 7 is the calibration evidence: entropy is 3.5× higher at the
failure point. Apertus's Table 25 is the decoding evidence: whether the same model
degenerates at all is decided by the sampling rule. On the sink side, none of
these multilingual papers mentions attention sinks, massive activations, or
outlier weights **at all** — verified by grep on Guerreiro, HalOmi, Raunak, Tang,
Marchisio, Koishekenov. The sink→repetition link (Cancedda 2024; Yona et al. 2025;
Su et al. 2025) is entirely English/decoder-only. **Status: the only route with a
mechanism attached, and it has never been run across languages.** That gap is the
project's opening.

*Summary:* (a) is the standard story and is unmeasured; (b) and (c) are folklore;
(d) has the best method and the wrong dependent variable; (e) has the mechanism
and no multilingual data.

---

## §4 What is NOT documented

**Plainly: "multilingual models repeat more in low-resource languages" is
folklore.** I could not find a single published measurement of repetition rate
per language, in a healthy multilingual model, under a fixed decoding
configuration, with a high-resource comparison arm. What exists instead:

1. **A conditional share, routinely mis-cited as an incidence.** Guerreiro's
   Fig. 2 is oscillatory hallucinations *as a percentage of hallucinations*, and
   its per-cell values are not printed anywhere. The frequently repeated claim
   "oscillatory hallucinations dominate mid/high-resource directions" is true as
   written and false as usually understood.
2. **A conflated error category.** Barua's 0.7% → 11.3% "output generation"
   jump bundles missing `\boxed{}` with degeneration, over a denominator of wrong
   answers, and its axis is *reasoning language*, not resource level.
3. **Qualitative annotator complaints.** Aya's "loops", flagged for **Russian and
   Serbian** and attributed by the authors to finetuning-data artifacts.
4. **A citation chain that terminates in the above.** Liu et al. 2026 → Barua
   2026 + Tran 2025 **[unopened]**.
5. **Silence in every technical report.** BLOOM, NLLB-200, EuroLLM, EuroLLM-9B and
   Tower report nothing; Aya Expanse appears to report nothing.

Also not documented, specifically:

- **Repetition rate as a function of a language's training-data share.** Nothing.
  Not in the BLOOM/NLLB/EuroLLM reports (which publish the shares), not in the
  data-mixture literature, not in Li et al. 2023 (English only).
- **Tokenizer fertility vs repetition.** Nothing (§3b).
- **Any multilingual replication of the core degeneration results** —
  self-reinforcement (Xu 2022), high-inflow (Fu 2021), repetition features
  (Yao 2025), repetition neurons (Hiraoka, except the one Japanese ablation
  contrast). All English.
- **Any per-language repetition rate for NLLB-200 with `no_repeat_ngram_size`
  turned off.** The default config forbids it, so this has apparently never been
  looked at.
- **Any link between sinks / massive activations / super weights and repetition
  in a multilingual model.** Zero.

### The cheapest measurement that would settle it

**Q: holding decoding fixed, does a healthy multilingual model's spontaneous
repetition rate vary with the target language, and does it track resource level?**

Design, sized for a single GPU and a few hours:

- **Data.** FLORES-200 devtest, 1,012 sentences, aligned across all languages —
  so **the same semantic content** in every condition. This is the whole reason to
  use FLORES rather than per-language corpora: content is controlled by
  construction.
- **Languages.** ~12 target languages stratified into 4 tiers × 3 languages,
  crossed against script and fertility so the three explanations in §3 are not
  collinear. Include at least one high-resource high-fertility language (e.g.
  Finnish or Turkish) and one low-resource Latin-script language, which is what
  breaks the tier/fertility/script confound.
- **Models.** NLLB-200 (enc-dec) and two decoder-only models from
  {Tower-7B, EuroLLM-9B, Aya-Expanse-8B, BLOOM-7B1}. Given Guerreiro §4.2, the
  enc-dec/decoder-only contrast is expected to be **larger** than any language
  effect, so it must be a factor, not a nuisance.
- **Decoding.** Greedy, **`no_repeat_ngram_size=0`, `repetition_penalty=1.0`**,
  fixed `max_new_tokens`, identical for every language. Then re-run at
  nucleus (T=1.0, p=0.9) as a second level, because Apertus Table 25 says the
  decoding rule moves the metric ~2×. Two decoding conditions is the minimum that
  makes the result quotable.
- **Metric.** RepeatRate_n (Liu 2026) at n = 2…6 **plus** TNG (n=4, t=2, Guerreiro)
  **plus** length-ratio against the reference (Koishekenov's over-generation
  signal). Report all three; they disagree in informative ways.
- **Analysis.** Repetition rate ~ tier, with fertility and script as covariates,
  bootstrap CI over the 1,012 sentences; family size = languages × models ×
  decoding conditions, corrected. Per CLAUDE.md: report
  `effect [95% CI], n=1012 sentences, family=M`, and bound a null rather than
  reporting "no effect".

Cost: two forward passes per sentence per language per model per decoding
condition. This is a week-scale measurement that would produce the first actual
number behind a claim the field has been repeating for five years — and it is a
publishable negative if the effect is small.

---

## §5 Comparing natural repetition rates to a lesion signature without confounding

The comparison Adrian wants is: **does the sink/massive-activation signature that
a super-weight lesion produces also appear at the onset of natural repetition?**
Doing that across languages invites four confounds. Each has a specific control.

**Confound 1 — fertility.** A loop of *k* words is a longer token loop in a
high-fertility language, which inflates any token-level n-gram repetition metric
and lengthens the window over which self-reinforcement accumulates.

*Controls.* (i) Report every repetition metric **both** in tokens and in
whitespace/character units; if the tier effect survives only in tokens it is
fertility. (ii) Measure fertility per language on the same FLORES text with the
same tokenizer and enter it as a covariate. (iii) Cross fertility with tier
explicitly in the language selection (§4) so they are not collinear. (iv) For the
*mechanistic* quantities — sink attention mass, attention entropy, residual-stream
norm at the sink token — index them **per generated token relative to the first
repeated token**, not per word, which makes them fertility-invariant by
construction.

**Confound 2 — prompt language.** Barua's result is precisely that the language
you *reason in* changes the failure mode, at fixed input language. If the prompt
and the target language covary, "language effect" and "prompt effect" are the
same variable.

*Controls.* Cross them. Four cells: {English prompt, target-language prompt} ×
{English target, other target}, all on the same FLORES source content. For NLLB
this is the target language **tag**, not a prompt — a cleaner instrument, and one
whose ALTI+ contribution is already known to be anomalous. Report the
prompt-language main effect separately; if it dominates, say so.

**Confound 3 — decoding config.** Fatal if unstated. HalOmi's NLLB blocks 4-grams
(so oscillation is impossible); Tang runs repetition penalty 1.1 (so repetition is
damped); Guerreiro runs beam-4 unconstrained; Apertus shows greedy vs nucleus moves
TTR by 2×; Marchisio shows temperature drives language confusion (WPR 83.5% at
T=1).

*Controls.* (i) One config, written into the config file, applied identically to
every language and to lesioned and healthy runs alike:
`no_repeat_ngram_size=0, repetition_penalty=1.0`, fixed `max_new_tokens`, greedy.
(ii) **Audit the lab's existing NLLB pipeline before using it** — the fairseq/HF
default has `no_repeat_ngram_size=4` and would silently delete the phenomenon.
(iii) Run the second decoding level (nucleus) as a manipulation check, not as an
afterthought. (iv) Since the lesion produces repetition **from the first token**
under greedy, and natural repetition typically begins tens of tokens in, report
**onset position** as a first-class result: if the two failures differ in onset
distribution, that alone is a finding.

**Confound 4 — different denominators.** Guerreiro's oscillatory share, Barua's
share-of-errors, and a plain per-sentence repetition rate are three different
quantities, and the literature's confusion between them is exactly how the
folklore got established (§4). Fix one denominator — **generations**, not errors,
not hallucinations — and state it in every caption.

**The design that avoids all four: match positions within a generation, not
generations across languages.** This is Marchisio's CP/¬CP design (§1.5),
transplanted. For each *spontaneous* repetition onset in a healthy model, take the
position immediately before the first repeated token as the **onset position**,
and compare it against non-onset positions **drawn from the same generation, the
same language, the same prompt, the same decoding run**. Every language-level
confound — fertility, script, tier, prompt language, decoding — is held constant
by construction, because both sides of the contrast come from the same sequence.
Marchisio's control that outputs *without* any confusion look identical on average
to the clean parts of outputs *with* it (1.61/0.365 vs 1.61/0.356) is the exact
null this design gives for free.

Then: the language question becomes **"does the onset-vs-clean *effect size* vary
by language?"** — a difference of differences, which is immune to any per-language
baseline shift in the measured quantity. And the lesion question becomes **"does a
partial-dose lesion move clean positions toward the natural onset signature, in the
same direction and along the same axis?"** — measured on the same positions, in the
same model, so the two signatures are directly comparable.

Three further protections:

- **A within-model null before any cross-language claim.** Compare healthy-model
  onset signatures across languages *only after* establishing that the
  onset-vs-clean effect is larger than the between-seed / between-prompt variation
  within one language. CLAUDE.md's rule: seeds are the unit of independence, and a
  scan over 12 languages × N layers needs its family size stated.
- **Use the metric disagreement as a diagnostic, not a nuisance.** Koishekenov's
  calibration — duplicating an output costs 47% spBLEU and 13% chrF++ — means the
  spBLEU/chrF++ gap is itself a repetition detector that needs no generation-side
  instrumentation, and it is comparable across languages because both metrics are
  computed against the same reference.
- **Include the enc-dec/decoder-only contrast deliberately.** Guerreiro §4.2 says
  it is the largest oscillation effect anyone has measured. If NLLB and
  Tower/EuroLLM/Aya are pooled, that difference will masquerade as a language
  effect, since NLLB covers the low-resource languages the LLMs do not.

---

## §6 Verification debt

- **Guerreiro TACL Fig. 2 per-cell values:** heatmap only; the underlying numbers
  are in the released artifacts ("over a million translations and detection
  results") but not in the paper. Anyone needing an oscillatory rate for a specific
  LP must go to the release, not the PDF.
- **Barua Fig. 4 per-language bars:** in the figure only.
- **[unopened]** Tran, O'Sullivan & Nguyen 2025, arXiv:2504.02890 — one of the two
  citations behind Liu 2026's low-resource repetition claim.
- **[unopened]** Aya Expanse technical report, arXiv:2412.04261.
- **[unopened]** The tokenizer-fertility papers named in §3b (arXiv:2508.06533,
  2510.09947, 2606.15044) — surfaced from search listings only; confirm before
  citing that they do *not* discuss repetition.
- **[unopened]** Round-trip translation benchmark, arXiv:2604.12911 (Skorobogat,
  Prabhu & Bethge). I downloaded and grepped it: it documents catastrophic
  low-resource MQM collapse but contains **no** occurrence of "degener", "repeat"
  or "repetition". A search summary attributed per-model degeneration rates
  (Gemini 22.3%, GPT-4.1 5.8%) to it; **I could not confirm those numbers in the
  v1 PDF and they should not be cited.**
