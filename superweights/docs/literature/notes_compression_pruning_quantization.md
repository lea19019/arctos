# Reading notes — compression, pruning, quantization vs. super weights

Conventions used throughout:

- **[read]** — I opened the PDF and read the section cited. Section/table/figure
  refs point at the printed artifact.
- **[inferred]** — my reasoning from what I read; not stated by the authors.
- **[unopened]** — named in a search listing or another paper's bibliography and
  **not opened**. Confirm before citing.
- Venue = what the PDF prints. Where it prints nothing, the note says
  *preprint, publication status unverified from the PDF*. I did not fill venues
  in from memory.

---

## §1 Paper by paper

### 1.1 The four local PDFs

**`TrainingDynamics-2025-PTQ-Robustness.pdf`** — Catalan-Tatjer, Ajroldi, Geiping,
arXiv:2510.06213. *ICLR 2026, printed on every page — peer-reviewed.* [read: abstract,
§1–§6, App. C]
GPTQ 3-/4-bit error measured across hundreds of intermediate checkpoints from OLMo
(1B, 7B), OLMo2 (1B/7B/13B/32B), SmolLM3 (3B, 11T tokens), Apertus (8B, 15T),
OpenSci (1.3B), Amber (7B). Central result (Fig. 1a, Fig. 2, Fig. 4): quantization
error is roughly flat through the *entire* 11T-token constant-LR phase and then
spikes as the LR anneals, while validation loss goes the other way — so val loss and
quantization error **diverge** at decay. §4.2 argues Kumar et al. 2024 and Ouyang et
al. 2024's "more training data ⇒ worse PTQ" is confounded by cosine schedules: under
WSD at 160M the degradation does not increase with token budget at all (Fig. 5b).
§5: higher peak LR gives lower quantization error at matched val loss (Fig. 6);
model souping / weight averaging gives lower PTQ error than **any** individual
ingredient (Fig. 2 for OLMo2, Fig. 3 for SmolLM3). §6 attributes this to flatter
minima (Hessian trace + λ_max via PyHessian, Fig. 9).
**What matters here:** the paper is a checkpoint-scale causal story about PTQ
brittleness that contains **essentially no outlier analysis** — the words "massive
activation" and "super weight" do not appear, and "outlier" appears three times, all
in background or bibliography (one nod to Ahmadian et al. 2023 on weight decay
controlling outliers). That is the seam: the best-measured account of *when* models
become hard to quantize does not connect to the best-measured account of *what*
makes them hard to quantize.

**`NVFP4-2026-Outlier-Dynamics-Pretraining.pdf`** — Dong, Fan, Tao et al.,
arXiv:2602.02047. *"Preprint. February 3, 2026" — not peer-reviewed.* [read:
abstract, §1, §3, §4, §5, conclusion]
Longitudinal instrumentation (kurtosis, top-k magnitude, flush-to-zero) of NVFP4
4-bit pretraining for Gated Linear Attention, Gated Slot Attention, Gated DeltaNet
and Qwen3, 340M–7B. Three findings I can use: (i) linear-attention variants have
lower per-tensor kurtosis than softmax attention but still show **per-block** spikes
at 16×16 granularity (Fig. 4) — global smoothness does not imply local smoothness,
which is exactly the super-weight situation; (ii) **outliers migrate early and
freeze late** — transient drifting spikes in early training give way, by ~step 15k,
to a small fixed set of "hot channels" (§3, Fig. 5, Fig. 8); (iii) Table 3's
parameter-normalised quantization-sensitivity score puts `o_proj` (GLA, 0.058) and
`v_proj` (Qwen, 0.143) at the top and **`mlp.down_proj` at the bottom** (0.012 GLA /
0.020 Qwen). Their Hot-Channel Patch reinjects top-k channel residuals online and
cuts the NVFP4→BF16 loss gap from 0.939% to 0.588% on GLA-1.3B/60B tokens (Table 2).
**What matters here:** this is independent, pretraining-time evidence for the repo's
second measured negative — the module that *hosts* the super weight is, by their
sensitivity score, the *least* quantization-sensitive projection. Also note the unit:
they operate on **channels**, never on scalars.

**`UnevenPTQ-2025-Multilingual-MT-Quantization.pdf`** — Marie & Fujita (NICT),
arXiv:2508.20893. *No venue line — preprint.* [read: abstract, §1–§4.4, §5
calibration, Limitations]
WMT24++ (55 languages, 110 directions), COMET `wmt22-comet-da`, five models
(Qwen3-1.7B/8B/32B, Llama-3.1-8B-Instruct, Llama-3.3-70B), four PTQ methods (AWQ,
bitsandbytes NF4, GGUF `Q4_K_M`/`Q2_K` with imatrix, AutoRound). Headline numbers
(§4.4, Tables 1–4): Qwen3-8B at GGUF 2-bit en→X loses ~2 COMET on Japanese and
French but **−17 on Bengali and Malayalam**; Llama-3.1-8B under BnB 4-bit loses 1.5
(ja→en) / 0.3 (fr→en) but 7.7 (bn→en) / 9.7 (ml→en); Llama-3.3-70B loses 0.9
(pl→en) vs 6.2 (zu→en). They state an "inverse-performance law": the worse a
language's FP baseline, the more quantization costs it. GGUF is the most robust
method, BnB the least at 70B. Language-matched calibration helps **only at 2 bits**
(Bengali calibration: +0.8 COMET bn→en; larger gain en→bn). Methodologically strict:
they refuse to average COMET across language pairs (footnote 6).
**What matters here:** a clean, large per-language effect with **zero mechanistic
explanation** — no activation statistics, no outlier analysis, no per-language
anything beyond resource level and script.

**`Marchisio-2024-Quantization-Multilingual-LLMs.pdf`** — Marchisio, Dash, Chen,
Aumiller, Üstün, Hooker, Ruder (Cohere), arXiv:2407.03211. *No venue line —
preprint; commonly cited as EMNLP 2024 Findings, **unverified from this PDF**.*
[read: abstract, §1]
Four multilingual LLMs, 8B–103B, up to 23 languages; mMMLU, MGSM, FLORES-200,
Language Confusion, LLM-as-a-Judge, and **human** evaluation. (1) Automatic metrics
understate the damage by roughly an order of magnitude: −1.7% (Japanese) automatic
vs −16.0% human; −0.3% vs −16.6% for French (Fig. 1). (2) Languages are disparately
affected and non-Latin scripts worst: at 103B, Latin-script −0.7% vs non-Latin
−1.9%; at 8B, −3.0% vs −3.7%. (3) Hard tasks degrade fastest: math −13.1%,
LLM-judge −25.9%. (4) Occasionally quantization *helps* (W8A8 at 35B, +1.3% avg).
**What matters here:** the (2) result is the same phenomenon UnevenPTQ finds in MT,
measured differently, and again with no mechanism offered.

### 1.2 The three read-only companions

**`phenomenon/Dettmers-2022-LLM-int8.pdf`** — NeurIPS 2022
(printed) — peer-reviewed. [read: abstract, §1, §3, §4]
Emergent outlier features appear as a phase shift at ~6.7B parameters: at that scale
~150,000 outliers occur per sequence but are concentrated in **6 feature dimensions**
across the whole transformer; zeroing them collapses top-1 attention softmax mass and
perplexity. Fix: mixed-precision decomposition — ~0.1% of dimensions in FP16, the
rest in INT8. This is the **per-channel activation outlier** worldview, and every
"smooth/scale/rotate" method after it inherits that frame.

**`phenomenon/Yu-2024-The-Super-Weight-in-LLMs.pdf` §4**
[read: abstract, §1–§3 summary, §4.1–§4.2]
Table 1: pruning the single super weight collapses quality (accuracy to chance,
perplexity up orders of magnitude); pruning the *other top 7,000* largest-magnitude
weights costs a few points. Restoring the super activation after removing the super
weight recovers average accuracy from 35.14 to 49.94 — ~42% of the loss, so super
activations only *partially* mediate the effect. §4.1 activation quantization:
REPLACE the super activation with the tensor median → RTN quantize/dequantize →
RESTORE it in FP16. Table 3: this data-free trick retains **71–83%** of SmoothQuant's
improvement over naive W8A8 across Llama-7B/13B/30B on Wiki-2 and C4. §4.2 weight
quantization: preserving the super weight while clipping other weight outliers lets
plain RTN scale to much larger block sizes.
**Note for this project:** §4's *activation*-side result is the strong one and it is
about a **scalar activation**, not the weight. The in-repo measured negative — that
holding the super *weight* in FP16 while RTN-quantizing the rest is a no-op across 8
models — is not in tension with Table 3, because Table 3 protects the **super
activation**. Any write-up must keep those two apart. [inferred]

**`phenomenon/Sun-2024-COLM-Massive-Activations-in-LLMs.pdf`**
— COLM 2024 (per the PDF's own reference formatting; the header prints no venue).
[read: abstract, §2 "act as biases", §3 attention, §4 explicit-bias experiment]
Massive activations are input-agnostic constants — setting them to zero is
catastrophic, but *fixing them to their mean* is not, which is what licenses calling
them bias terms. They concentrate attention probability onto their token (the sink)
and thereby inject an implicit bias into the attention output. Most useful result for
compression: training GPT-2 with **explicit** learned attention bias k′/v′ makes
massive activations **disappear entirely** (Figs. 9, 10) — i.e. the phenomenon is a
training-time workaround, not an architectural necessity.

### 1.3 Layer pruning

**Gromov et al. 2024, `Gromov-2024-…pdf`**, arXiv:2403.17887.
*ICLR 2025 (printed) — peer-reviewed.* [read: abstract, §1, §3, discussion]
Angular-distance layer similarity picks the best contiguous block to delete, then
QLoRA "heals" it. Up to ~half the layers of common open models can go with minimal
benchmark degradation — but the deletable block is always deep, and the paper says
outright that "the shallow layers likely play a critical role" and that the final
layer cannot be pruned. No mechanism for *why* shallow layers are different.

**`Men-2024-ShortGPT-Layer-Redundancy.pdf`**, arXiv:2403.03853. *Preprint.* [read:
abstract, §1, §3, Table 1]
Block Influence = 1 − cosine(layer input, layer output). Redundancy is "primarily
manifested in the middle to later layers, with the **initial layers and the last
layer** often being more critical" (§3). Their Table 1 dissects the last layer:
removing the whole last layer takes PPL 7.60 → 13.37, removing only its FFN → 12.35,
removing only its attention → 7.65. They flag that the initial-layer result
*contradicts* their own appendix-A argument that deeper layers should be more
redundant, and offer no explanation for the early-layer half.

**`Yang-2024-LaCo-Layer-Collapse-Pruning.pdf`**, arXiv:2402.11187. *Preprint.*
[read: abstract, §1, Table 23]
Collapses rear layers into a prior layer by parameter differencing. Useful to this
project mainly for **Table 23**, a printed qualitative comparison on XSum where
baseline pruners (SliceGPT among them) emit *"of the 19900s of the 1900s of the 1900s
of the 1900s"* — one of the very few places the pruning literature shows the
degenerate-repetition failure mode rather than just a perplexity number.

**`Kim-2024-Shortened-LLaMA-Depth-Pruning.pdf`**, arXiv:2402.02834. *Preprint.*
[read: abstract, §1]
Depth (whole-block) vs width (head/channel) pruning head to head. Depth pruning wins
on wall-clock under memory-bound small-batch inference where width pruning does not
help at all; continued pretraining beats LoRA for recovery at severe ratios.

**`He-2024-Not-All-Attention-Is-Needed.pdf`**, arXiv:2406.15786. *Preprint.* [read:
abstract, §1]
Similarity-based drop of Blocks vs MLP vs Attention. Attention layers are
dramatically more redundant: half of Llama-2-70B's attention layers dropped gives
48.4% speedup for 2.4% quality loss; dropping MLP layers hurts. They also trace
checkpoints and find attention redundancy is present throughout training.

**`Siddiqui-2024-Deeper-Look-Depth-Pruning.pdf`**, arXiv:2407.16286. *ICML 2024 TF2M
workshop (printed) — workshop review.* [read: abstract, §1]
Adds Shapley-value block importance and splits blocks into attention vs FFN. Up to
33% of Mistral-7B's self-attention layers removable with no MMLU degradation; FFN is
not comparably removable. Also: lightweight additive-bias / low-rank "emulated
updates" recover most of the loss **for the initial blocks specifically** (up to +5
MMLU absolute) — an indirect hint that what early blocks contribute is close to a
constant offset. [inferred link to Sun et al.'s bias framing]

**`Lad-2024-Remarkable-Robustness-Stages-of-Inference.pdf`**, arXiv:2406.19384.
*"Preprint. Under review."* [read: abstract, §1, §3, §4 stage 1]
Deleting or swapping adjacent layers at inference keeps 72–95% of top-1 accuracy —
except at the ends. "Intervening on the first layer is **catastrophic** for model
performance for every model, regardless of size or model family" (§3). Their
explanation (§4, Stage 1): the first layer is not a normal layer but an extension of
the embedding, performing *detokenization* — without it the rest of the network is
blind to the current token. **They do not mention massive activations, attention
sinks, or super weights anywhere in that argument.** This is the single most direct
statement of the phenomenon the project could re-explain.

**`Ashkboos-2024-SliceGPT.pdf`**, arXiv:2401.15024. *ICLR 2024 (printed).* [read:
abstract]
Computational invariance (orthogonal transforms absorbed into adjacent weights) lets
you delete rows/columns and shrink the embedding dimension; up to 25% of Llama-2-70B.
Same mathematical device QuaRot later repurposes for outliers.

### 1.4 Weight / structured pruning metrics

**`Frantar-2023-SparseGPT.pdf`**, arXiv:2301.00774. *No venue printed.* [read:
abstract]
One-shot OBS-style pruning with layer-wise reconstruction against calibration
activations; 50–60% sparsity on 100B+ models without retraining. No outlier notion —
outlier handling is implicit in the Hessian.

**`Sun-2023-Wanda-Simple-Effective-Pruning.pdf`**, arXiv:2306.11695. *ICLR 2024
(printed).* [read: abstract, §2 metric, §3 motivation, related work]
`S_ij = |W_ij| · ‖X_j‖₂`, compared **per output row** (`G_ij = {W_uv : u = i}`), not
per layer or globally. §2's motivation cites Dettmers explicitly: because input
features can differ by 100×, magnitude alone picks the wrong weight. They note (§2)
that "when the input channel of the considered weight has large magnitude features,
the weight itself tends to be assigned a larger importance score even if its
magnitude is small."
**[inferred]** A super weight sits at `down_proj[i, j]` where column `j` is the
massive-activation-carrying intermediate channel. Both factors of Wanda's score are
extreme, so its score should be near the top of row `i` and Wanda should keep it at
any realistic sparsity. Nobody has measured this. See §5, problem 2.

**`Yin-2024-OWL-Outlier-Weighed-Layerwise-Sparsity.pdf`**, arXiv:2310.05175. *ICML
2024 (printed).* [read: abstract, §3.2 LOD definition, Empirical Studies I–III]
Defines the **Layerwise Outlier Distribution**: score `A_ij = ‖X_j‖₂·|W_ij|` (i.e.
Wanda's score), an element is an outlier if `A_ij > M·Ā^l` with M = 5 or 7, and
`D^l` is the fraction of such elements in layer `l`. Layer sparsity is then allocated
*inversely* — outlier-dense layers get pruned less. Empirical Study II asks whether
better pruners preserve more outliers, and finds they do; Study I shows LOD is far
from uniform across layers. Gains are large only at high sparsity (70%: +61.2 PPL
over Wanda, +6.8 over SparseGPT on LLaMA-13B).
**Note the unit:** OWL's "outlier" is a *weight-level score that folds in activation
norms*, aggregated to a per-layer scalar. It is neither a pure weight outlier nor a
pure activation outlier, and it is used only to set a per-layer budget.

**`Zhang-2024-RIA-Plug-and-Play-Pruning.pdf`**, ICLR 2024 proceedings. [read:
abstract, §1, §3, Fig. 1]
Relative Importance normalises each weight by its row *and* column sums before
multiplying by an activation term. Fig. 1's motivating case is a weight that global
magnitude would delete but relative importance retains. Plus channel permutation for
N:M sparsity. Relevant because the failure mode RIA fixes — a whole input channel
being uniformly deprioritised — is exactly what would endanger a super-weight column
under a badly-normalised metric. [inferred]

**`Dong-2024-Pruner-Zero.pdf`**, arXiv:2406.02924. *ICML 2024 (printed).* [read:
abstract, §1]
Genetic programming over a symbolic search space containing magnitude, Wanda, and
SparseGPT-like metrics, discovering better ones automatically. The nearest thing to
a systematic answer to "is `|W|·‖X‖` the right score", and a natural place to check
whether any discovered metric behaves differently on super weights. [inferred]

**`Lu-2024-AlphaPruning-HeavyTailed-Layerwise.pdf`**, arXiv:2410.10912. *Preprint.*
[read: abstract]
Layerwise sparsity ratios from the shape of each weight matrix's empirical spectral
density (heavy-tailed self-regularization α), not from outliers. Prunes LLaMA-7B to
80% sparsity. **This is the control condition for OWL:** if a spectral, non-outlier
allocator matches an outlier-based one, "outlier density explains prunability" is not
established.

**`phenomenon/Supernodes-2026-Loss-Critical-Hubs-FFN.pdf`**, arXiv:2604.23475. *Preprint.*
[read: abstract, §1]
Fisher-style loss proxy (activation × gradient second moments) at FFN **channel**
granularity. In Llama-3.1-8B the top 1% of channels per layer carry a median 58.7%
of loss-proxy mass (range 33.0–86.1%). Critically: *"LP-defined supernodes overlap
only weakly with activation-defined outliers and are not explained by activation
power or weight norms alone."* Structured FFN pruning at 50% sparsity: baselines that
prune supernodes hit PPL 989.2 (Wanda-channel), their protected variants 54.8 and
42.6. Replicates on Mistral-7B, Llama-2-7B, Qwen2-7B, Llama-3.1-70B, and the
concentration *increases over OLMo-2-7B pretraining*.
**This paper is the closest published relative of Adrian's joint-ablation finding**
(a set of `down_proj` columns sharing one intermediate neuron), and its
"loss-critical ≠ activation-outlier" result is a third-party corroboration of the
repo's measured negative that importance ⊥ quantization sensitivity.

**`GarbageAttention-2026-BOS-Sink-Heads-Sink-Aware-Pruning.pdf`**, arXiv:2601.06787.
*Preprint.* [read: abstract, §1, §2]
Heads with a high `<BOS>` sink score — especially in deeper layers — contribute
little to prediction and act as dumping grounds for surplus attention mass. The sink
score is then used directly as the structural pruning criterion for heads and layers.
An existence proof that an interpretability statistic can *be* a pruning metric.

### 1.5 Quantization and outliers

**`Frantar-2022-GPTQ.pdf`** (ICLR 2023, printed) — layer-wise second-order weight
quantization, column-by-column with Hessian error compensation; no explicit outlier
model. [read: abstract]

**`Xiao-2023-SmoothQuant.pdf`** (ICML 2023, printed) — migrates **per-channel
activation** outliers into the weights via a per-channel smoothing factor `s`, making
W8A8 tractable. [read: abstract]

**`Lin-2023-AWQ.pdf`** (MLSys 2024, printed) — identifies ~1% *salient weight
channels* using **activation** magnitude statistics and protects them by per-channel
scaling instead of mixed precision. The mainstream method conceptually closest to
"protect what matters", and it works at **channel** granularity, never scalar.
[read: abstract]

**`Shao-2024-OmniQuant.pdf`** (ICLR 2024, printed) — Learnable Weight Clipping
(weight outliers) + Learnable Equivalent Transformation (activation outliers shifted
into weights), both optimised by block-wise reconstruction. [read: abstract]

**`Wei-2023-Outlier-Suppression-Plus.pdf`** (no venue printed) — channel-wise
**shifting** for outlier asymmetry plus channel-wise **scaling** for concentration,
both absorbed into neighbouring modules. Pure per-channel-activation frame. [read:
abstract]

**`Ashkboos-2024-QuaRot.pdf`** (NeurIPS 2024, printed) [read: abstract, §1]
Randomized Hadamard rotations applied to the residual stream, FFN activations, parts
of attention, and the KV cache, using computational invariance so outputs are
unchanged. The rotated hidden state "has no outliers", enabling end-to-end 4-bit
weights, activations, and KV cache. Explicitly *per-channel* outliers.

**`Liu-2024-SpinQuant.pdf`** (ICLR 2025, printed) [read: abstract, §1, Fig. 2]
Same device, but the rotation is *learned* (Cayley-optimised). Their key empirical
observation: **random rotations differ by up to 13 points** in downstream accuracy,
and SpinQuant beats QuaRot by up to 45.1% of the remaining gap on Llama-3-8B. That
rotations are not interchangeable implies the thing being rotated has structure worth
respecting.

**`Lin-2024-DuQuant.pdf`** (NeurIPS 2024, printed) [read: abstract]
The one method here that names the distinction this project cares about: **Normal
Outliers** (moderate, present across all tokens — per-channel) vs **Massive
Outliers** (far larger, token-specific). They argue smoothing/rotation methods handle
the former and fail on the latter, and use outlier-dimension-informed block rotation
plus a zigzag permutation to spread both.

**`Chen-2024-PrefixQuant.pdf`** (no venue printed) [read: abstract, §1]
Cites Sun et al. 2024 by name and re-labels massive activations as **token-wise
outliers**: in one example just 2 outlier tokens out of a 2048-token context carry
values >1000. PrefixQuant *prefixes those outlier tokens into the KV cache*, so they
are held out of the quantization grid; the maximum outlier-token value drops from
>1000 to ~15 (Fig. 2b), though the remaining magnitude is still hundreds of times
normal. Training-free.

**`phenomenon/Bondarenko-2023-Quantizable-Transformers.pdf`** (NeurIPS 2023, printed) [read:
abstract, §1]
The causal story: strong outliers exist because attention heads that want a *no-op*
(or a partial residual update) must drive softmax inputs to extremes to get exact
zeros, and that pressure shows up as outliers elsewhere. Two architectural fixes —
clipped softmax and gated attention — produce models that never grow large outliers,
and quantize to full INT8 activations with no extra machinery, at equal or better FP
quality (BERT, OPT, ViT). This is the *pretraining-time* intervention; Sun et al.'s
explicit-attention-bias experiment is the same idea arrived at independently.

**`phenomenon/Yang-2024-Activation-Spikes-GLU-Variants.pdf`** ("Preprint. Under review.")
[read: abstract, §1]
Activation spikes in GLU/SwiGLU FFNs (1) occur in the FFN of **specific layers,
particularly early and late layers**, and (2) are **dedicated to a couple of tokens**
rather than shared across the sequence. Two fixes: QFeM leaves the identified modules
unquantized; QFeP prepends a fixed prefix so the spike lands on a known token. They
show SmoothQuant fails to control these.
**This is the closest quantization paper to the super-weight picture** — same module
family (GLU FFN), same layer range, same token-locality — but it treats the spikes as
things to route around, and never asks what *weight* produces them.

**`Wang-2025-Task-Circuit-Quantization.pdf`** (no venue printed) [read: abstract, §1]
TaCQ contrasts the unquantized and uniformly-quantized weights to estimate expected
weight change, multiplies by gradient information to predict task-performance impact,
and keeps the resulting "weight circuit" in 16 bits. 3.1 bits recovers 96% of
Llama-3-8B-Instruct's MMLU; +14.74% over the best baseline at 2 bits. It is the
existing template for "localise, then allocate bits", and it is *gradient*-based, not
activation-magnitude-based.

**`phenomenon/Achilles-2025-Altering-Neurons-Cripples-Language.pdf`** (ICLR 2026, printed)
[read: abstract, §1, §4–§5]
Perturbation-based causal search for critical neurons. (1) Ultra-sparse critical
sets: disrupting them makes a 72B model with 1.1B neurons collapse, perplexity up to
6.25×10²¹ — ~20 orders of magnitude. (2) They cluster in the **outer layers** and
overwhelmingly in **`mlp.down_proj`**. (3) Degradation is a **sharp phase
transition**, not a gradual decline. It cites Yu et al. and positions itself as the
set-level generalisation. Evaluation includes MGSM (multilingual math), but the paper
does *not* analyse per-language effects.
**This is the published paper closest to Adrian's Llama-3.1-8B joint-ablation result
(×47,488 jointly vs ×1.02 each).** Anything written up must situate against it.

### 1.6 Multilingual compression

**`Koishekenov-2023-NLLB200-Language-Specific-Expert-Pruning.pdf`**, arXiv:2212.09811.
*No venue printed.* [read: abstract, §1]
NLLB-200 54.5B MoE: up to 80% of experts removable without finetuning and with
negligible translation-quality loss, taking inference from 4×32GB GPUs to one. The
pruning metrics are shown to identify **language-specific experts**. The only
NLLB-specific compression paper in this folder.

**`Zhang-2024-Multilingual-Brain-Surgeon.pdf`**, arXiv:2404.04748. *No venue printed.*
[read: abstract, §1]
Points out that GPTQ / SparseGPT / Wanda all calibrate on a **single language**
(English) even for multilingual models, and that this is where the low-resource drop
comes from. MBS samples calibration data across languages proportionally to the
model's training language distribution. This is the field's current best answer to
per-language damage, and it is entirely a *data* answer, not a mechanism.

**`Ogueji-2022-Intriguing-Properties-Compression-Multilingual.pdf`**, arXiv:2211.02738.
*No venue printed.* [read: abstract, §1]
Sparsification of mBERT NER across 40 languages. The important counter-result:
compression can *improve* robustness over dense models, and under some sparsity
regimes **aids rather than disproportionately harms** low-resource languages. Any
claim of the form "compression hurts low-resource languages" has to survive this.

**`Williams-2024-Calibration-Data-Pruning-Quantization.pdf`**, arXiv:2311.09755.
*No venue printed.* [read: abstract]
First systematic study of calibration-set choice across quantization and pruning
methods, datasets, tasks, models; finds substantial downstream variation, contra
earlier claims of robustness. A mandatory confound control for anything that varies
calibration language.

---

## §2 Where interpretability findings are already inside compression methods

The literature already uses this machinery — but it is important to be exact about
**which quantity** each method targets, because they are three different objects and
the project's framing lives or dies on the distinction.

| Object | What it is | Where it lives |
|---|---|---|
| **(A) Per-channel activation outlier** | A hidden/feature *dimension* whose values are large across essentially all tokens. ~6 dims at 6.7B (Dettmers §3). | Residual stream, FFN inputs, attention inputs |
| **(B) Per-token activation outlier / massive activation** | A few *tokens* (often `<BOS>`/delimiters) whose hidden state is huge in one or a few dims; constant magnitude from onset layer to last (Sun 2024; Yu 2024 "super activation") | Residual stream from an early layer onward |
| **(C) Scalar weight outlier / super weight** | One element of `mlp.down_proj` (Yu 2024); or a whole `down_proj` column, i.e. one intermediate neuron (Adrian's joint result; "supernodes" 2026; "critical neurons" 2026) | Early-layer FFN |

**Methods that target (A), per-channel activation outliers:**
- **LLM.int8()** — mixed-precision decomposition: ~0.1% of dimensions kept in FP16,
  the rest INT8. The original and most literal "hold out the outlier".
- **SmoothQuant** — per-channel scale `s` migrating activation range into the weights.
- **Outlier Suppression+** — per-channel *shift* (for asymmetry) plus per-channel
  *scale* (for concentration), both absorbed into adjacent modules.
- **OmniQuant** — the same two moves, but learned (LET), plus learned weight clipping
  (LWC) for weight-side outliers.
- **QuaRot** — randomized Hadamard rotation of the residual stream, FFN activations,
  and KV cache. Rotation spreads any single channel's magnitude across all channels,
  so (A) genuinely disappears; computational invariance keeps outputs identical.
- **SpinQuant** — the same, with the rotation *learned*; the 13-point spread across
  random rotations says the structure being destroyed is not isotropic.
- **DuQuant** — block rotation seeded with known outlier dimensions plus a zigzag
  permutation, explicitly to catch what smoothing misses.

**Methods that target (B), per-token / massive activations:**
- **PrefixQuant** — names them token-wise outliers, cites Sun et al., and *prefixes
  the outlier tokens into the KV cache* so they are never quantized. Max outlier-token
  value >1000 → ~15.
- **Yang et al. 2024 (GLU activation spikes)** — QFeM skips quantizing the specific
  modules where spikes occur; QFeP prepends a fixed prefix so the spike is pinned to
  a known token position.
- **DuQuant** — its "Massive Outliers" category is exactly (B), and its claim is that
  every (A)-targeted method fails on it.
- **Yu et al. §4.1** — REPLACE the super activation with the median, RTN quantize,
  RESTORE in FP16. A one-scalar, data-free intervention that recovers 71–83% of
  SmoothQuant's benefit.

**Methods that target (C), scalar or channel-level weight importance:**
- **Wanda** — `|W_ij|·‖X_j‖₂` per output row. Uses (A)/(B) statistics to score
  *weights*. [inferred: this should automatically rank a super weight near the top of
  its row, but nobody has checked.]
- **OWL** — aggregates Wanda's score into a per-layer outlier density (LOD, threshold
  M·mean with M∈{5,7}) and allocates *less* sparsity to outlier-dense layers. It never
  protects an individual weight; it protects a *layer budget*.
- **AWQ** — per-channel salience from activation magnitude, protected by scaling
  rather than by precision.
- **RIA** — row/column-normalised importance × activations, to stop whole channels
  being uniformly deprioritised.
- **Supernodes / SCAR (2026)** — Fisher-style loss proxy at channel granularity, with
  explicit protection of the top-1% "supernode" channels and their read/write "halo".
  Directly reports that these channels **do not coincide** with activation outliers.
- **TaCQ** — gradient × expected-quantization-perturbation to choose which weights
  stay in 16 bits.
- **Garbage Attention (2026)** — `<BOS>` sink score as the head/layer pruning metric.

**Methods that remove the *cause* rather than route around it:**
- **Bondarenko et al. 2023** — clipped softmax / gated attention give heads a cheap
  no-op, so the softmax-input pressure that creates outliers never builds. Models
  pretrained this way quantize to INT8 with no extra machinery.
- **Sun et al. 2024 §4** — explicit learned attention bias `k′, v′` makes massive
  activations vanish in GPT-2.
- **NVFP4 / CHON 2026** — pretraining-time online residual patching of the top-k "hot
  channels", plus BF16 protection of post-QK ops.

**The gap, stated precisely.** Every method above operates on a **channel**, a
**token**, a **layer budget**, or an **architecture**. Not one operates on the scalar
weight that Yu et al. identify. The two methods that come closest — AWQ (channel
salience) and Yu's own §4 — either work at channel granularity or protect the
*activation*, not the weight. That is consistent with the repo's measured negative
that FP16-protecting the super weight under RTN is a no-op: there is no published
method that protects individual scalar weights either, and the reason may simply be
that scalar weight precision is not the binding constraint. [inferred]

---

## §3 What the literature knows about the failure mode of over-compressed models

**Does the degenerate output look the same?** Partially documented, never
systematically compared.
- **Yu et al. 2024** (read-only): ablating the super weight shifts "almost all logit
  probability mass to stopwords" — matching Adrian's *"We. We. We."* observation.
- **LaCo Table 23** [read]: over-pruned Llama2-7B under SliceGPT emits *"of the 19900s
  of the 1900s of the 1900s of the 1900s"* on XSum. This is the same attractor shape
  — high-frequency function words in a loop — reached by a completely different
  intervention. It is printed as a qualitative example, with no analysis.
- **`SignalDegradation-2026-Two-Failure-Modes-Quantization.pdf`**
  (arXiv:2604.19884, in-repo, preprint) [read: abstract, §1] is the closest thing to
  a systematic treatment: it separates **Signal Degradation** (patterns intact,
  precision eroded by cumulative error — repairable by training-free intervention)
  from **Computation Collapse** (key components stop functioning and *the signal is
  destroyed in the early layers* — not repairable by compensation). The 2-bit
  "performance cliff" is Computation Collapse. **This is the natural comparison
  partner for a super-weight-ablation attractor study** and it is already in the repo.
- Nobody, anywhere I read, compares the token distribution of an over-compressed
  model to that of an ablated model.

**Which layers collapse output vs merely degrade?** Consistent across six papers:
- Deep/middle layers: removable in bulk. Gromov (up to ~half, with healing); ShortGPT
  (middle-to-late redundant); He (half of Llama-2-70B's attention layers, 2.4% loss);
  Siddiqui (33% of Mistral-7B self-attention, no MMLU loss); Garbage Attention
  (deep sink-heavy heads).
- Last layer: **not** removable. ShortGPT Table 1 (7.60 → 13.37 PPL; the FFN half is
  the culprit at 12.35); Gromov says the same.
- Early layers: **not** removable, and this is where explanations run out.
  - Lad et al.: first-layer drop/swap is catastrophic *in every model family and
    size*, explained as **detokenization** — the first layer is an extension of the
    embedding, and without it the network is blind to the current token.
  - ShortGPT: initial layers "more critical", explicitly noted as contradicting their
    own theory, with no account offered.
  - Siddiqui: emulated-update repair works best precisely on the initial blocks (+5
    MMLU) — consistent with those blocks contributing something close to a constant
    offset. [inferred]
  - Yang et al. (GLU spikes): activation spikes occur in FFNs of "early and late
    layers" — the same two ends that resist pruning. Neither paper cites the other.

**Is early-layer unprunability explained by the massive-activation onset layer
anywhere?** **No — not in anything I opened.** The pieces exist in separate
literatures:
- Massive activations appear at one early layer (L1–L7) and persist at constant
  magnitude (Sun 2024; Adrian's replication across every model examined).
- Early layers cannot be pruned (Gromov, ShortGPT, Lad, Siddiqui).
- Queipo-de-Llano et al. 2025 (``, in-repo)
  ties massive-activation emergence to a depth-wise phase structure
  (Mix–Compress–Refine), which is the closest published bridge — **but it is not a
  pruning paper and makes no prunability prediction.** [read: title/abstract only via
  search listing — the PDF is in-repo but I did not open it; treat as [unopened]]
- The Signal-Degradation paper locates Computation Collapse "in the early layers",
  which is suggestive, but its mechanism is quantization error, not massive
  activations.

That conjunction — *the onset layer of the massive activation is the last layer you
cannot prune* — is, as far as I can tell from these PDFs, unclaimed and untested.

---

## §4 The multilingual / MT gap

**What is solidly known:**
1. **Per-language damage under quantization is large and uneven.** Marchisio: at 103B,
   Latin-script −0.7% vs non-Latin −1.9%; at 8B, −3.0% vs −3.7%; human evaluation
   shows ~10× the damage automatic metrics report. UnevenPTQ: Qwen3-8B GGUF 2-bit
   en→X, −2 COMET ja/fr vs −17 bn/ml; Llama-3.1-8B BnB 4-bit, −0.3 fr→en vs −9.7
   ml→en. Both report an "inverse-performance law".
2. **It is not monotone in "compression is bad for the tail".** Ogueji et al. found
   sparsification can *improve* robustness and can help low-resource languages under
   some regimes. Marchisio found a +1.3% average from W8A8 at 35B.
3. **Calibration data is a first-order lever and a first-order confound.** Williams &
   Aletras: calibration choice moves downstream performance substantially. MBS: making
   the calibration set proportional to the model's language distribution recovers
   low-resource performance. UnevenPTQ: language-matched calibration helps **only** at
   2-bit (Bengali calibration +0.8 COMET bn→en, more en→bn), not at 4-bit.
4. **Language-specific structure exists and has been used for compression once.**
   Koishekenov: NLLB-200's MoE pruning metrics identify language-specific experts, and
   80% of experts can go. That is the one place "language-specific component" and
   "compression" meet in this folder — and it is MoE-routing-specific, not applicable
   to dense models.

**What is not known — the gap:**
- **Nobody has tied per-language compression damage to outlier structure.** Neither
  Marchisio nor UnevenPTQ measures a single activation statistic. Their explanatory
  variables are training-data volume, script, and model size.
- **Nobody has connected language-specific neurons to compression damage.** The
  ingredients are in the repo (`Tang-2024-Language-Specific-Neurons-LAPE.pdf`,
  [unopened by me]) but the join is unmade in anything I read here.
- **No massive-activation / super-weight study is multilingual.** Yu et al., Sun et
  al., Achilles' Heel, Supernodes, PrefixQuant, DuQuant, NVFP4 — all English (or
  English + Chinese) perplexity and English benchmarks. Achilles uses MGSM but does
  not break results out per language.
- **No massive-activation study covers an encoder-decoder MT model.** NLLB, Tower and
  EuroLLM do not appear in any super-weight or massive-activation paper I opened. The
  in-repo `AttentionSinks-2026-Multilingual-NMT-NLLB.pdf`
  is the only NLLB×sink item anywhere in this repo [unopened by me].
- **Open empirical question with no data either way:** does a multilingual model have
  *one* super weight / massive-activation channel shared by all languages, or
  language-conditional ones? Sun et al.'s "input-agnostic constant bias" framing
  predicts one shared. Tang et al.'s language-specific neurons predict the opposite
  for at least part of the FFN. Nobody has looked. [inferred]

---

## §5 Open problems for a one-semester project

**Preamble on framing.** Two in-repo measured negatives bound this space and every
proposal below respects them: (i) FP16-protecting the super weight under RTN is a
no-op (rtn+SW ≈ rtn on 8 models, EuroLLM-9B included); (ii) ablation importance is
orthogonal to quantization sensitivity — and note that NVFP4's Table 3 independently
ranks `mlp.down_proj` as the *least* quantization-sensitive projection, which is
third-party support for (ii). So no proposal is "protect the super weight". Each is
either a **measurement** that the literature has left undone, or a **diagnostic** that
uses the super-weight detector as an instrument rather than as a thing to preserve.

---

### P1. Does the massive-activation onset layer predict the earliest prunable layer?

**Problem.** Every layer-pruning paper finds early layers unprunable and none explains
why in terms of activation structure. Every massive-activation paper finds an onset
layer and none connects it to prunability. Test whether onset layer L* predicts the
shallowest depth at which block deletion stops being catastrophic.

**Why it is open.** [read] Lad et al. explain first-layer catastrophe by
detokenization and never mention massive activations; ShortGPT reports the early-layer
result as contradicting its own theory and offers nothing; Gromov says "shallow layers
likely play a critical role" and stops. Queipo-de-Llano is the nearest bridge and
makes no prunability claim [unopened].

**Concrete experiment.** Models where the detector already runs and L* is already
known: OLMo-1B, Llama-7B, Mistral-7B, Llama-3.1-8B-Instruct, Phi-3-mini, plus 3–5
more to get n≥8 with spread in L* (the current harness already covers Table-2 models
and beyond). For each: (a) record L* from the existing detector; (b) sweep
single-block deletion over every layer, recording wikitext-2 PPL ratio and a
generation-degeneracy score; (c) sweep contiguous-block deletion at n∈{2,4,8} using
Gromov's angular-distance selection, restricted to blocks starting at each depth.
Primary test: rank correlation between L*/n_layers and the shallowest depth d where
PPL ratio < 1.5 (pre-registered threshold), across models, with a CI and family-size
correction over the depth sweep. Cost: one forward-pass detector run + O(n_layers)
perplexity evals per model — the expensive part is already built. Well inside a
7–13B single-GPU budget.

**What counts as an answer.** Either a correlation with a CI that excludes zero *and*
survives the obvious control (L*/n_layers vs. a constant fraction of depth — many
models put L* at a similar relative depth, which would make the correlation vacuous),
or a bounded null: "across n=8 models, r = x [CI], i.e. onset layer explains at most
y% of variance in earliest prunable depth."

**Confounds.** (1) The constant-relative-depth confound above is the serious one —
it must be pre-registered and reported. (2) Healing: Gromov's results are
*post*-QLoRA; without healing everything looks unprunable. Either skip healing and
say so, or budget for it. (3) Deleting the layer *containing* the super weight
conflates two mechanisms; report that layer separately. (4) Model families share
architectures — n is closer to the number of *families* than the number of models.

---

### P2. Does Wanda/OWL importance recover the super weight, and does it matter?

**Problem.** Wanda's score is `|W_ij|·‖X_j‖₂`; the super weight has both factors
extreme. Does the field's standard pruning metric already rank it top? Does OWL's LOD
peak at the super weight's layer? And — the part that decides whether it matters —
does forcing a super weight to be pruned at moderate sparsity produce collapse, or has
the surrounding sparse network already routed around it?

**Why it is open.** [read] Wanda's paper motivates the metric from Dettmers'
*channel* outliers and never discusses individual critical weights. OWL's LOD is a
per-layer aggregate; it is never asked whether its peak coincides with the super
weight's layer. The 2026 Supernodes paper reports that loss-critical channels overlap
only weakly with activation-defined outliers — which makes the answer genuinely
uncertain rather than obvious.

**Concrete experiment.** On the same model set: (a) compute Wanda scores and report
the super weight's **percentile rank within its output row** (Wanda's actual
comparison group) and within its layer; do the same for the joint-ablation candidate
set (the shared `down_proj` input column); (b) compute OWL's LOD per layer with M=5
and M=7 and check whether the peak layer is the super weight's layer; (c) the
decisive arm — prune with Wanda at 50%/60%/70% sparsity in three conditions: default,
super weight force-pruned, super weight force-kept; report PPL and generation
degeneracy. Cost: Wanda is one calibration pass; the whole thing is hours on one GPU.

**What counts as an answer.** A table of percentile ranks with the null explicitly
stated (a randomly chosen `down_proj` weight's rank distribution), plus the three-arm
sparsity result. A clean negative — "Wanda ranks it in the top 0.01% and force-pruning
it changes PPL by less than the seed variance at 50% sparsity" — is publishable
*within the project* as a boundary on the super weight's practical relevance to
pruning, and it directly extends the repo's existing rtn+SW negative from quantization
to sparsity.

**Confounds.** (1) Calibration data — use Williams & Aletras' finding as the reason to
run ≥2 calibration sets and report both. (2) Wanda's per-row grouping is the correct
comparison group; reporting a layer-global rank instead would be the classic "config ≠
what ran" error. (3) At high sparsity everything collapses; the effect must be read at
a sparsity where the default model still works.

---

### P3. What happens to the super activation under rotation-based quantizers?

**Problem.** QuaRot and SpinQuant claim to make the residual stream "outlier-free" via
rotation. A rotation cannot reduce a *token's* norm — it redistributes one channel's
magnitude across all channels. So what happens to the super activation, which is a
token-wise phenomenon? And does the super weight remain identifiable, or ablatable, in
a rotated model?

**Why it is open.** [read] QuaRot and SpinQuant frame outliers purely per-channel and
never mention massive activations or token-wise outliers. PrefixQuant and DuQuant both
argue explicitly that channel-oriented methods *fail* on massive/token outliers — but
neither measures what a rotation does to a super activation specifically, and neither
touches the super weight. Yu et al. cite rotation methods in related work and do not
evaluate against them.

**Concrete experiment.** Take 3–5 models with confirmed super weights (Llama-7B,
Mistral-7B, Llama-3.1-8B-Instruct, OLMo-1B, Phi-3-mini). Apply QuaRot's Hadamard
rotation (reference implementation; no learned rotation needed for the first pass).
Measure, before and after: (a) the per-token max hidden-state magnitude at the super
activation's token and layer — does the token norm survive, as theory says it must?
(b) the per-channel max — does the channel outlier flatten, as claimed? (c) run the
super-weight detector on the rotated model and ask whether a single scalar is still
identifiable in the rotated `down_proj`; (d) ablation: does ablating the corresponding
rotated structure still collapse the model? Cost: QuaRot is training-free and runs in
minutes; the detector is already built.

**What counts as an answer.** A before/after table separating per-channel from
per-token statistics, plus a yes/no on whether the super weight survives rotation as a
localised scalar. The interesting outcome either way: if rotation delocalises the
super weight across an entire row, that is a concrete statement that *rotation-based
quantizers destroy the locality that makes super weights detectable* — which is a real
result about the interaction of interpretability and compression, and requires
protecting nothing.

**Confounds.** (1) QuaRot's rotation is applied at specific points; be explicit about
which tensors are rotated and report only those. (2) "Still identifiable" needs a
pre-registered criterion — e.g. does a single scalar account for >X% of the row's
contribution to the super activation — not an eyeball judgment. (3) Random-rotation
seed variance: SpinQuant shows rotations differ by up to 13 points, so run ≥3 seeds
and report the interval.

---

### P4. Is per-language PTQ degradation predicted by per-language activation statistics?

**Problem.** The MT-side result (Adrian's replication; UnevenPTQ; Marchisio) is that
languages degrade unevenly under PTQ. The proposed explanatory variable everywhere is
training-data volume or script. Test an activation-structure variable instead: does a
language's massive-activation / outlier profile — measured on that language's inputs —
predict its COMET drop?

**Why it is open.** [read] Neither Marchisio nor UnevenPTQ measures any activation
statistic. No super-weight or massive-activation paper is multilingual, and none
covers an encoder-decoder MT model. MBS's answer is a calibration-data answer.
Koishekenov's language-specific experts are MoE-routing-specific.

**Concrete experiment.** Models: EuroLLM-9B (in-repo, has a strong super weight),
Tower, plus NLLB-200-1.3B/3.3B if the harness can be adapted to encoder-decoder
(non-trivial — budget for it or drop NLLB to a stretch goal). Languages: the six
UnevenPTQ reports on plus enough to reach n≈12, spanning script and resource level.
Per language, on FLORES/WMT24++ source text: (a) is the super activation's channel,
token position and magnitude the *same* across languages, or language-conditional?
(b) per-language kurtosis / max-magnitude of hidden states at the known massive-
activation layer; (c) fraction of tokens that are outlier tokens. Then regress
per-language COMET drop (4-bit and 2-bit, ≥2 methods) on those statistics **with
training-data proxy and script as covariates**, because without them the result is
uninterpretable.

**What counts as an answer.** An effect size with a CI for the activation statistic
*after* partialling out resource level. Either "outlier profile adds explanatory power
beyond resource level" — a genuinely new claim in this literature — or a bounded null.
The descriptive half (question (a): is the super activation shared or
language-conditional?) is worth reporting on its own regardless of the regression,
because nobody has measured it.

**Confounds.** (1) Resource level is correlated with everything; the regression must
report the partial effect and the collinearity, or it is worthless. (2) COMET is not
comparable across language pairs — UnevenPTQ's footnote 6 is right; use per-pair
*deltas*, never averages across pairs. (3) Calibration language must be held fixed or
crossed deliberately (Williams & Aletras). (4) Tokenizer fertility differs by
language and changes sequence length, which changes activation statistics — control
for it. (5) n = number of languages, not number of sentences; the CI must reflect
that.

---

### P5. Is the degenerate output of a 2-bit model the same attractor as an ablated model?

**Problem.** Super-weight ablation produces stopword repetition ("We. We. We.").
Over-pruned models produce "of the 1900s of the 1900s" (LaCo Table 23). 2-bit models
hit a "performance cliff". Are these the same attractor, and if so, is the shared
cause the loss of the massive activation's bias contribution?

**Why it is open.** [read] The Signal-Degradation paper (in-repo, arXiv:2604.19884)
already separates two quantization failure modes and locates the catastrophic one "in
the early layers" — but its unit of analysis is quantization error, and it does not
compare against ablation. LaCo prints one degenerate example and analyses nothing. Yu
et al. describe the stopword shift but compare it to nothing else.

**Concrete experiment.** One model family, three interventions producing comparable
degradation: (i) super-weight (or candidate-set) ablation; (ii) GPTQ at 2 bits; (iii)
Wanda at the sparsity that matches (ii)'s perplexity. Match on perplexity, then
compare *outputs*: KL between next-token distributions on a shared prompt set;
type-token ratio and repetition rate; the top-50 token overlap; and — the mechanistic
part — the magnitude of the massive activation under each intervention, since Yu et
al. showed ablation drops it by 75%. Cost: modest; the ablation and detector harnesses
exist, GPTQ/Wanda are off-the-shelf.

**What counts as an answer.** A distributional comparison with a stated null (two
*different* seeds of the same intervention, to establish how much distributional
distance is noise). If 2-bit and ablation land in the same place *and* both suppress
the massive activation, that is a mechanistic account of the 2-bit cliff. If they land
in different places, that is a clean negative that kills a tempting story — and it
should be filed as such.

**Confounds.** (1) Matching on perplexity is not matching on damage; report at least
two match points. (2) Decoding settings dominate repetition (Holtzman; UnevenPTQ finds
temperature dominates top-p) — fix and report them. (3) "Same attractor" needs a
pre-registered metric, not a vibe; pick it before running. (4) Repetition is a
well-studied phenomenon with its own literature already in
`` — do not re-derive it.

---

### Ranking, and what I would drop

**P3 (rotation) and P5 (attractor)** are the two I would put first: both are cheap,
both use the existing detector/ablation harness essentially unchanged, both have a
publishable outcome in either direction, and neither requires protecting anything.
**P4** is the one that satisfies the MT-lab constraint and is the most novel relative
to the literature, but it is also the most confound-heavy and the NLLB arm may not fit
the harness — scope it to decoder-only multilingual models (EuroLLM, Tower) first and
treat NLLB as a stretch. **P2** is the most likely to yield a clean negative, which is
valuable but modest. **P1** is the most attractive-sounding and, on reflection, the
weakest: the constant-relative-depth confound is severe, and n is effectively the
number of architecture families, which is small.

I would **not** pursue any variant of "allocate bits/sparsity by super-weight
proximity". OWL, AlphaPruning, AWQ, TaCQ and SCAR already occupy that design space
with better-developed methods, and the repo's two measured negatives predict it fails.
