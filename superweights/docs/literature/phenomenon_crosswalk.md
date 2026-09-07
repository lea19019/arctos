# One phenomenon, many names — a map across 47 papers (PDFs in `../../papers/`, flat and gitignored)

Written 2026-09-05 from the reading notes in `notes_*.md` (same folder) (which cite section/table numbers;
go there before quoting a paper). Venues are as printed on the PDF or in arXiv metadata; "preprint"
means no peer-reviewed venue could be confirmed. Papers are referred to by the filename stem.

## 1. The chain

Every paper here describes one or more links of the same causal chain. Reading them as one story
is the point of this map.

```
 (1) a few WEIGHTS                (2) one FFN NEURON              (3) a MASSIVE ACTIVATION
 in an early-layer FFN    ───►    fires enormously on one   ───►  written into 1–4 residual channels;
 (down_proj scalars, or           token (usually the first)        rides the residual stream unchanged
 the gate/up rows feeding                                          to the last layer; input-agnostic
 that neuron)                                                      → behaves like a fixed BIAS
        │                                                                  │
        │                                                                  ▼
        │                                              (4) an ATTENTION SINK forms on that token:
        │                                                  heads dump surplus attention there
        │                                                  (a learned "do nothing" / NOP gate)
        │                                                                  │
        ▼                                                                  ▼
 (6) REMOVE any link ⇒ COLLAPSE:                       (5) FUNCTION (disputed): implicit attention
     perplexity ×10²–10⁴, output degenerates               bias · NOP gate · anti-over-mixing ·
     into repeated function words                          gradient regulator · confidence/scale control
     ("We. We. We."), = an oscillatory hallucination
     in MT terms
```

Two facts organise everything below. **The value of the constant carries no information**
(Sun-2024 Table 3; Owen-2025 Table 1: set-to-mean is harmless in 16/16 models, set-to-zero is
catastrophic in 9). **Whether one weight or a set is critical is only how concentrated the fan-out
in link (1)→(2) is** (this project's own Tier A result; Su-2025 Table 8; Oh-2024 §2.2).

## 2. Terminology crosswalk

| Term | Link | Paper(s) that use it | What it denotes | Same object as |
|---|---|---|---|---|
| **super weight** | 1 | Yu-2024; Subramanian-2026 | one `down_proj` scalar whose zeroing collapses the model | a spoke of the massive-weight / super-neuron fan-out |
| **massive weights** | 1–2 | Oh-2024 | the `W_gate`/`W_up` rows producing the top-k intermediate activations | the neuron behind the super weight |
| **super neuron** | 2 | Li-2026 | FFN neuron with a channel-sparse down-projection that amplifies the first token | the same neuron, row basis |
| **super expert** | 2 | SuperExperts-2025 | MoE expert whose pruning collapses the model; mapped to `down_proj` weights (Table 8) | the same object in an MoE |
| **critical neurons** | 2 | Achilles-2025 | ultra-sparse neuron sets, mostly in `down_proj`, whose disruption collapses 72B models | the set-level generalisation |
| **supernodes** | 2 | Supernodes-2026 | top-1% FFN channels carrying ~59% of loss mass (Fisher proxy) — *weakly* overlapping with activation outliers | adjacent, not identical |
| **massive activation** | 3 | Sun-2024; Owen-2025; Chen-2026-Measuring; GallegoFeliciano-2025; SingleLayer-2026 | ≤4 scalars per hidden state, 10³–10⁴× the median, input-agnostic, early onset | link 3 |
| **super activation** | 3 | Yu-2024 | the massive activation created by the super weight | link 3 |
| **massive values** | 3 | Jin-2025 | the same, located in Q/K after RoPE | link 3 seen in attention |
| **activation spikes** | 3 | Yang-2024 | the same in GLU FFNs, token-dedicated, early and late layers | link 3, quantization view |
| **outlier features / dimensions** | 3′ | Dettmers-2022; Kovaleva-2021; Puccetti-2022; Macocco-2025; Amazon-2025 | a hidden *dimension* large across most tokens (per-channel, not per-token); zero overlap with massive activations in Llama-2 (Sun-2024 §2.3) | **a neighbour, not the same** — keep distinct |
| **rogue dimensions** | 3′ | Timkey-2021 | dimensions dominating cosine similarity, mismatched with behavioural importance | the outlier-dimension construct, methodological warning |
| **outlier channels (Mamba)** | 3′ | Pierro-2024 | absmax channels in an attention-free model | shows softmax is not necessary for channel outliers |
| **high-norm / register tokens** | 3–4 | Darcet-2023; Parodi-2026 | vision-transformer tokens with huge norm holding global information | link 3–4 in ViTs |
| **attention sink** | 4 | Xiao-2023; Gu-2025; Barbero-2025; Su-2026 survey; RanMilo-2026 ×2 | the token receiving most attention mass regardless of query | link 4 |
| **dark signals** | 3–4 | Cancedda-2024 | the near-null-space component of the BOS residual that implements sinking, written by an early MLP | link 3 in the spectral basis |
| **active-dormant heads / value-state drain** | 4 | Guo-2024 | heads that attend to the sink to output ~nothing | link 4–5 |
| **gradient sink** | 4 | Chen-2026-Attention-Sinks-Induce-Gradient-Sinks | the backward-pass mirror of the sink | link 4, training view |
| **compression valley** | 3–4 | QueipoDeLlano-2025 | the entropy/rank collapse of the representation matrix caused by the BOS norm | link 3–4 |
| **entropy / token-frequency neurons** | 5 | Stolfo-2024 | final-layer neurons that scale logits via the norm, or push toward the unigram prior | link 5, the readout side |
| **sink neurons** | 2 | Interpreting-2025 (Yona et al.) | MLP neurons selected by ‖MLP_j(BoS)‖ whose ablation kills the BOS norm | plausibly the super neuron (untested) |

## 3. Paper-by-paper: link, venue, evidence type, and how it connects

Legend: **C** = causal (intervention or from-scratch training), **O** = observational. "Feeds" = which
project question it bears on (Q1 what the constant does; Q2 which models need it; Q3 unit /
criterion; Q4 multilingual / encoder-decoder; Q5 formation).

### Link 1–2 — the weights and the neuron

| Paper | Venue | Ev. | What it shows | Connects to |
|---|---|---|---|---|
| Yu-2024 | preprint (rejected ICLR 2025, mainly on the quantization method) | C | one `down_proj` scalar → super activation; stopword shift (Fig. 5, descriptive); Table 2 coordinate directory | origin of link 1; Q1, Q3 |
| Subramanian-2026 | COLM 2026 | C | replicates Yu's coordinates; training only the SWs → chance; **top-magnitude weights inert in 4 models** (Table 8) | confirms link 1, refutes "magnitude = importance"; Q3 |
| Oh-2024 | preprint | C | criticality at the intermediate neuron; reports repetition after ablation; Gemma-2/Phi-3-medium insensitive | link 2; Q1, Q2, Q3 |
| SuperExperts-2025 | ICLR 2026 | C | 3 of 6,144 experts → collapse, repetitive output, ~90% sink decay; Table 8 maps to shared-neuron `down_proj` weights | link 2→4→6 in one paper; Q1, Q3 |
| Li-2026 | ICML 2026 (stated) | C/O | variance → `W_O` → super neuron → RMSNorm → sink; describes Yu's Llama-2-7B coordinate in row basis without citing Yu; head-wise RMSNorm removes both, 4 seeds | link 2→4; Q3, Q5 |
| Achilles-2025 | ICLR 2026 | C | ultra-sparse neuron sets in `down_proj` collapse 72B models; sharp phase transition; cites Yu | set-level version of link 1; Q3 |
| Supernodes-2026 | preprint | C | loss-critical FFN channels ≠ activation outliers | warns against equating links 2 and 3; Q3 |
| An-2025 | ICLR 2025 | C/O | weight, activation and attention outliers as one chain; 100% feature alignment `W_down`↔activations; "context-aware scaling factor" | the chain itself; Q1 |
| Ding-2026 ×2 | preprint (single author) | O | weight-tail statistics across checkpoints; random-init anchor k₀=1.205 (trims top decile, cannot see a SW) | Q3 null, Q5 |

### Link 3 — the massive activation

| Paper | Venue | Ev. | What it shows | Connects to |
|---|---|---|---|---|
| Sun-2024 | COLM 2024 | C | definition; **set-to-zero → ∞, set-to-mean → no change** (Table 3); implicit attention bias; explicit K/V biases remove them when training GPT-2 | the anchor for link 3 and 5; Q1 |
| Owen-2025 | preprint | C | 17 models: mean harmless in 16/16, zero catastrophic in 9; **OLMo-2-7B has none**; Gemma has them only with BOS; DyT removes them but attention concentration persists | Q2 is *this* paper's open problem; Q1 |
| Jin-2025 | ICML 2025 | C | massive values in Q/K; zero vs. mean agree when perturbation stays in-coordinates | Q1 control design |
| Chen-2026-Measuring | preprint | O | 27 checkpoints, 8 families; 4/24 fail Sun's binary criterion; continuous statistic | Q3 criterion |
| GallegoFeliciano-2025 | preprint | O | activation-ratio trajectories across all Pythia sizes; absent at init; Sun's criterion fails <1B | Q5 |
| SingleLayer-2026 | ICML 2026 (stated) | C (post-hoc) | the "ME layer" per model; masking intervention | Q1 where |
| Yang-2024 | preprint | C (quant) | GLU spikes in early/late layers, token-dedicated; routes around them | link 3 → compression |
| Darcet-2023 / Parodi-2026 | ICLR 2024 / preprint | C | ViT registers; **zero-ablation overstates dependence** (zero −36.6 pp; mean/noise/shuffle ~0) | Q1 control design |
| Pierro-2024 | ICML-W 2024 | C | outlier channels in Mamba | softmax not necessary for link 3′ |

### Link 3′ — outlier dimensions (the neighbour)

| Paper | Venue | Ev. | What it shows | Connects to |
|---|---|---|---|---|
| Dettmers-2022 | NeurIPS 2022 | C | ~6 dimensions carry emergent outliers; thresholds reverse-engineered; "sudden at 6.7B" softens vs perplexity | origin; Q3 null problem |
| Kovaleva-2021 | Findings ACL 2021 | C | 48 LayerNorm params break BERT; outlier identity differs between two BERT-base runs (fn. 1) | encoder precedent; Q5 seed question |
| Puccetti-2022 | Findings EMNLP 2022 | C/O | outliers driven by token frequency; ablation → *more frequent* tokens predicted | Q1 (H-prior), Q4 (frequency ⇒ language dependence) |
| Macocco-2025 | preprint | C | last-layer outlier dims in 8 decoders; ablation → *rarer* tokens; 11/38 survive across training | **contradicts Puccetti's direction**; Q1, Q5 |
| He-2024 | NeurIPS 2024 | C | kurtosis (init = 1) as threshold-free measure; Pre-Norm × Adam interaction causes outliers; SOAP/Shampoo reduce them | Q3 measure, Q5 causes |
| Amazon-2025 | NAACL 2025 | C | T5 encoder/decoder outlier dims disjoint; decoders have more; magnitude tracks depth not size | only encoder-decoder study; Q4 |
| Timkey-2021 / Rajaee-2022 / Hammerl-2023 | EMNLP 2021 / Findings ACL 2022 / preprint | O | rogue dims; **mBERT has no outlier dimension** (Rajaee) vs. cased mBERT has 3 (Hämmerl); XLM-R dominated by dim 588; 3σ has no null (§6.4) | Q4 risks and the only multilingual data |

### Link 4 — the attention sink

| Paper | Venue | Ev. | What it shows | Connects to |
|---|---|---|---|---|
| Xiao-2023 | (ICLR 2024 per common citation; arXiv in our index) | C | keeping 4 initial tokens' KV restores windowed attention; one learnable sink token suffices | link 4 named |
| Gu-2025 | ICLR 2025 | C | 60M from-scratch knob study: sinks by 1–2B tokens; LR, weight decay (inverted-U), data amount matter; positional encoding, FFN, LN placement don't; **key biases or sigmoid-without-normalisation ⇒ no sink and no massive activations** | Q5 causes; Q1 |
| Barbero-2025 | COLM 2025 | C | sinks prevent over-mixing; context length controls them; removing BOS from Gemma-7B collapses benchmarks | Q1 (H-sink) |
| QueipoDeLlano-2025 | ICLR 2026 | C | zeroing the layer-0 MLP→BOS contribution kills the sink and the compression valley; "all three emerge together ~step 1k" (n=2, 1 seed) | link 3→4 causal claim; Q1 dissociation, Q5 |
| Guo-2024 | preprint | C (toy) | mutual reinforcement of sink / value drain / residual peak; Adam→SGD removes peaks, sinks remain | Q5 optimizer |
| Kaul-2024 | preprint | C | softmax-1 removes first-token dominance but not outliers; **OrthoAdam removes outliers** | link 3 vs 4 dissociation |
| Chen-2026-Gradient-Sinks | preprint | C | V-scale suppresses massive activations, keeps sinks | dissociation, other direction |
| Qiu-2025 | preprint (Qwen) | C (15B scale) | gated attention: value gate cuts massive activations without cutting the sink — "MAs are not a prerequisite for sinks" | the strongest counter to link 3→4 |
| Zuhri-2025 | preprint | C | softpick removes both at 340M, costs accuracy at 1.8B | removal is not free |
| RanMilo-2026 ×2 | preprints | theory + C | sinks provably necessary under softmax; full GPT-2 sink circuit with 10 interventions and matched controls | link 4 mechanism; Q3 control design |
| Fesser-2026 | preprint | C (ViT) | two sink algorithms (NOP vs broadcast); massive activation as "the energetic cost of a reliable NOP gate" | most specific functional account; Q1 |
| Su-2026 | preprint (survey) | — | nine interpretations of sinks; names the missing benchmark and the unexplored training dynamics | map of Q1–Q5 |
| Sun-2026 | preprint | C (7B scale) | SwiGLU as directional amplifier; sandwich/QK-norm/DyT suppress spikes, sinks survive; "decoupled artifacts" | link 3 vs 4 |
| Bondarenko-2023 | NeurIPS 2023 | C | outliers exist because heads need a no-op; clipped softmax / gated attention prevent them at pretraining | origin of the NOP account; Q5 |

### Links 5–6 — function and collapse

| Paper | Venue | Ev. | What it shows | Connects to |
|---|---|---|---|---|
| Cancedda-2024 | arXiv (venue unconfirmed) | C | early-MLP "dark" vector in the BOS stream is the sink; filtering it yields `"the, the, the, …"` — the project's failure mode, attributed to over-copying | Q1 H-sink, the nearest precedent |
| Stolfo-2024 | NeurIPS 2024 | C | entropy neurons (final-norm scale) and token-frequency neurons (unigram direction); frozen-LN mediation ablation | Q1 tooling for H-prior and H-scale |
| Interpreting-2025 (Yona) | preprint | C | sparse "sink neurons" create the BOS norm; repeated tokens acquire BOS-like norms | Q1, Q3 identity test |
| SuperExperts-2025, Oh-2024, Jin-2025 | see above | C | all report repetitive output after removing the object, none analyse it | link 6 is reproduced, unexplained |

## 4. The live disputes (do not cite one side as settled)

| Dispute | For | Against |
|---|---|---|
| Massive activations **cause** sinks | Sun-2024 §4; QueipoDeLlano-2025 §3.3 (post-hoc ablation) | Qiu-2025; Chen-2026-Gradient; Sun-2026; Owen-2025 Table 2 (from-scratch: remove MAs, sinks survive). No one has removed sinks while keeping MAs. |
| The unit is the **scalar** | Yu-2024 Table 1 | Subramanian-2026 Table 8; Su-2025 Table 8; Oh-2024; Li-2026; this project's 16/21 inert |
| Every massive activation is **load-bearing** | Sun-2024; Yu-2024 | Owen-2025 (7/16 unaffected; OLMo-2-7B has none); Oh-2024 App. F |
| Ablating outliers shifts output toward **frequent** tokens | Puccetti-2022 (BERT); Yu-2024 Fig. 5 | Macocco-2025 (decoders → rarer). Possibly an encoder/decoder split. |
| Emergence is **sudden at 6.7B** | Dettmers-2022 Fig. 3a | Dettmers' own Fig. 3b; He-2024; Amazon-2025 (depth, not size) |
| Removing sinks is **free** | Gu-2025 §7.4 (1B); Qiu-2025; Li-2026 | Zuhri-2025 (costs 5 points at 1.8B); RanMilo-2026 (necessary under softmax) |
| Zero-ablation measures **dependence** | (implicit in most ablation papers) | Owen-2025, Sun-2024 Table 3, Parodi-2026: it is an upper bound; mean-replacement is the control |

## 5. What is established where (for the proposal's related-work section)

- **Existence and bias-like behaviour of the constant:** Sun-2024 (COLM), An-2025 (ICLR), Jin-2025 (ICML), Dettmers-2022 (NeurIPS).
- **The weight-level handle:** Subramanian-2026 (COLM), SuperExperts-2025 (ICLR), Achilles-2025 (ICLR), Li-2026 (ICML); Yu-2024 is the preprint that supplied the coordinates.
- **Sinks and their causes:** Gu-2025 (ICLR), Barbero-2025 (COLM), QueipoDeLlano-2025 (ICLR), Bondarenko-2023 (NeurIPS), Darcet-2023 (ICLR).
- **Outlier dimensions and their measurement:** He-2024 (NeurIPS), Kovaleva-2021 (ACL-F), Puccetti-2022 (EMNLP-F), Amazon-2025 (NAACL).
- **The readout side:** Stolfo-2024 (NeurIPS).
- **Not established anywhere:** what the constant does for *generation* (why loops, not noise); which models need it and why; anything multilingual beyond Hämmerl/Rajaee's encoder-embedding results; anything in a translation model; formation with more than one seed.

## 6. Suggested reading order

1. Sun-2024 (the object and the zero-vs-mean fact) → 2. Yu-2024 §3 + Subramanian-2026 Table 8 (the handle and its limits) → 3. Owen-2025 (what breaks the simple story) → 4. An-2025 and Oh-2024 (the chain and the neuron unit) → 5. Gu-2025 (causes) → 6. Cancedda-2024 and Stolfo-2024 (function and readout) → 7. QueipoDeLlano-2025 vs Qiu-2025 (the dispute) → 8. Puccetti-2022 vs Macocco-2025 (the other dispute) → 9. Amazon-2025, Hämmerl-2023, Rajaee-2022 (everything multilingual or encoder-decoder that exists) → 10. Su-2026 survey for the rest.
