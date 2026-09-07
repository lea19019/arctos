# NOTES — training dynamics and checkpoint suites, for RQ1 / RQ4

**Provenance convention used throughout.**
*Read* = taken from the PDF text in this folder (or the read-only companions named in
`README.md`), with a section/table/figure pointer. *Hub-verified* = queried live against
`https://huggingface.co/api/models/<id>/refs` on **2026-09-05**; the numbers are what the
API returned that day. *Inferred* = my arithmetic or my judgement, marked as such.
*[unopened]* = named but not read. Nothing here is cited from memory.

Two things to hold onto before reading further:

- **No paper in this folder measures when a super weight forms.** The closest are
  Ding-2026 (weight-side outliers in Pythia-70m at four checkpoints) and
  Queipo-de-Llano-2025 (massive activations across Pythia checkpoints, n=2 models). The
  causal criticality of a *single named coordinate* across checkpoints, across seeds,
  has not been measured by anyone. That is the hole RQ1 sits in.
- **The abrupt-vs-gradual question is not settled and appears to be
  configuration-dependent**, not a universal fact (Xu-2026, read; see §3).

---

## §1 Paper by paper

### 1.1 Instruments (checkpoint suites)

**Biderman et al. 2023, Pythia** — ICML 2023, peer-reviewed. arXiv:2304.01373v2.
*Read.* 8 sizes (70M–12B), each trained twice (Pile / deduped Pile). §2.4: checkpoints
at init and **every 1,000 steps (2,097,152,000 tokens)**, *plus* log-spaced checkpoints at
steps **{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}** — **154 per run**, 143,000 steps,
≈300B tokens, batch **1024 × 2048 = 2,097,152 tokens/step**, cosine LR, weight decay 0.01,
Adam β=(0.9, 0.95), rotary-pct 0.25, parallel attn+MLP (`gpt-j-residual`), untied
embeddings, LayerNorm. §2.7: a script reproduces the **exact dataloader**, so batch *n* is
recoverable. Apache 2.0. §3.2: training order has **little** effect on memorization (a
Poisson model fits). §3.3: a phase change at **step 65,000 (45% of training)** for
≥2.8B models on term-frequency-correlated arithmetic. n_seeds = **1** per (size, dataset).
14M and 31M are *not* in this PDF — they were added later.
*Causal?* No — all observational except the §3.1 counterfactual pronoun-swap retraining.

**van der Wal et al. 2025, PolyPythias** — ICLR 2025, peer-reviewed (stamped).
arXiv:2503.09543v2. *Read, fully.* **45 new runs = 9 extra seeds × 5 sizes
(14M, 31M, 70M, 160M, 410M)**, ≈7k new checkpoints, same **154-checkpoint** grid, same
hyperparameters/data as Pythia. §2 + footnote 3: a seed varies **both parameter
initialisation and batch composition** — and because GPT-NeoX shuffles documents *before*
packing, the sequences themselves differ across seeds, not just their order. Table 5 also
lists six 160M runs that **disentangle** the two: `pythia-160m-data-seed{1,2,3}` (data
only) and `pythia-160m-weight-seed{1,2,3}` (init only) — never discussed in the body, and
the only public way to attribute a formation event to init vs data.
Findings that matter here: only **2 of 50 runs** are outliers by a ±2σ final-accuracy
heuristic (**410M seeds 3 and 4**), and those two are the only ones with **loss spikes**
(Fig. 5). §5's HMM training maps are linear for stable runs and **fork** for the two
outliers; Table 1 says the outlier-only transitions are driven by **λ_max ↑1.41,
σ_b ↑1.79, σ_λ ↑1.71** — i.e. spectral events, exactly the signature a single dominant
scalar would produce, a connection the paper does not make. Table 2: state transitions land
at similar steps across seeds within a size (e.g. 160M: 2k, 18k, 61k, 100k), except 410M
where the outliers blow up the variance (0→1 at 18k ± **22.6k**). §3: metrics leave the
random baseline around **step 10³ ≈ 2B tokens** and converge around **step 10⁴ ≈ 20B**.
§4: representations are similar across seeds; subspace angles peak ~step 2k (end of LR
warm-up) then decline to ~20°; cross-size Pearson r = **0.99 / 0.98 / 0.94**. Footnote 8:
step-0 probes are used as an explicit random-init null.
*Causal?* No. **No explicit "you need N seeds" rule is stated.**
Checkpoint density: 154/run, dense early. n_seeds = **10 per size**.

**Sellam et al. 2022, MultiBERTs** — ICLR 2022, peer-reviewed (stamped).
arXiv:2106.16163v2. *Read (targeted).* **25 BERT-Base seeds**, 2M steps each, batch 256
sequences. Intermediate checkpoints for only **5 of the 25 runs, 28 each** — every 20,000
steps to 200,000, then every 100,000 (165 checkpoints total, ~68 GB). *Inferred*: at
256 × 512 tokens, the first intermediate checkpoint (step 20,000) is already
≈2.6B tokens — **no early-training resolution at all**. Encoder-only, one size.

**Weller et al. 2025, Ettin** — ICLR 2026, peer-reviewed (stamped). arXiv:2507.11412v2.
*Read.* Paired **encoder and decoder** at 17M/32M/68M/150M/400M/1B, differing **only** in
attention pattern (bidirectional vs causal) and objective (MLM 30% vs CLM) — §3.3, Table 1
caption. **236 checkpoints per model, every 8.5B tokens** (linear, no log-spaced early
grid), spanning pretraining 1.7T / mid-training 250B / decay 50B. Trapezoidal LR. Data
released **in batch order**. Also ships cross-objective continued-pretraining models
(50B tokens each direction). Caveat App. E: the **1B models were trained to only 1/3 the
data (667B)** and the paper does not say how many checkpoints they therefore have.
English + code (the word "multilingual" appears only in citations). No Hub path, no
license, and **no training-dynamics analysis** in the paper — all analyses are at the final
checkpoint. n_seeds = **1**.
*The suite's value here is the encoder-vs-decoder contrast, not the trajectory: its first
checkpoint is 8.5B tokens in, long after everything interesting has happened.*

**Magnusson et al. 2025, DataDecide** — ICML 2025, peer-reviewed. arXiv:2504.11393v2.
*Read (targeted).* 1,050 models = **25 data recipes × 14 sizes (4M–1B) × 3 seeds**,
"more than 30K model checkpoints". The catch, stated in §2.1 and the Table 2 caption:
**only the 1B models have 3 full reruns; at every smaller size seeds 2 and 3 are terminated
at 25% of target compute.** §3.2 notes final checkpoints "are not available for all seeds".
Token:param ratio fixed at 100. **Checkpoint spacing and Hub revision naming are not stated
in the PDF** — confirm from the collection. Useful numbers: the 1B seed-to-seed accuracy sd
"can be as high as 2% points"; §2.2 averages the **last 10% of checkpoints** to beat
step-to-step noise; §3.4/Fig. 5 formalises **noise (sd over seeds) vs spread (sd over
recipes)** as the decision criterion — a reusable framing for "is my criticality
difference real?".

**Groeneveld et al. 2024, OLMo-1** / **OLMo Team 2025, OLMo-2** — arXiv:2402.00838v4,
2501.00656v3. Preprint copies; ACL 2024 for OLMo-1 per publisher listing, not verified from
the PDF. *Skimmed; the load-bearing facts here are Hub-verified rather than read.*
OLMo-1B-0724-hf is where **Yu et al.'s OLMo super weight lives** (Table 2, read:
layer **1**, `mlp.down_proj[1764, 8041]`; the older OLMo-1B is layer 1 `[1764, 1710]` —
same row index, different column).

**Liu et al. 2023, LLM360 (Amber / CrystalCoder)** — arXiv:2312.06550v1, preprint copy
(COLM 2024 per publisher listing, unverified from the PDF). **360 checkpoints** plus the
full data sequence, Apache 2.0. *Hub-verified*: `IFM/Amber` has 360 branches
`ckpt_000`…`ckpt_359`; `IFM/Crystal` has 250 branches named
`CrystalCoder_phase{1,2,3}_checkpoint_NNNNNN`.

**SmolLM2 (arXiv:2502.02737v1, preprint "under review")**, **TinyLlama
(arXiv:2401.02385v2)**, **MAP-Neo (arXiv:2405.19327v4)** — skimmed as instruments only;
see §2 for what they actually publish. MAP-Neo's intermediate checkpoints are
*Hub-verified* to exist but only as raw **Megatron** shards
(`iter_NNNNNNN/mp_rank_XX/model_optim_rng.pt`), not HF revisions — a real conversion cost.

**Multilingual (the MT bridge).** *Read + Hub-verified.*
- **EuroLLM** (Martins et al. 2024, arXiv:2409.16235v1): 1.7B/9B, 35 languages, 4T tokens,
  **explicit parallel data**, batch 3,072 sequences ≈ 12M tokens, Apache 2.0.
  "When Meanings Meet" (arXiv:2601.22851v1, preprint) studies **26 EuroLLM-1.7B pretraining
  checkpoints**, first at **≈48B tokens**, and reports shared concept spaces arising early.
  **Caution:** those 26 revisions are *not* visible on `utter-project/EuroLLM-1.7B`
  (Hub-verified: 1 branch, `main`). Provenance unclear — ask the authors before planning
  around them.
- **Apertus** (arXiv:2509.14233v2, preprint): 8B/70B, **1811 languages**, ~40% non-English,
  15T tokens, seq 4096, batch 1024→2048 seqs (4.2→8.4M tokens), AdEMAMix + WSD, xIELU,
  QK-Norm. *Hub-verified*: `swiss-ai/Apertus-8B-2509` exposes **44 pretraining revisions**
  `step50000-tokens210B` … `step2627139-tokens15T` (every 50,000 steps ≈ 210B tokens) plus
  27 `longctx-step*`. The "When Meanings Meet" authors say outright that Apertus's
  **first checkpoint at 210B tokens is too late** and "appears to mask the development of
  shared concept spaces" — the same problem will bite a super-weight trace.
- **Lucie-7B** (arXiv:2503.12294v1, preprint): French-centric multilingual, 3T tokens,
  seq 4096, batch ramped 256→1024 sequences. Footnote 70 lists checkpoints at steps
  **5,000; 10,000; …; 25,000; 50,000; 75,000; 100,000; …; 750,000; 753,851** plus five
  context-extension checkpoints. *Hub-verified*: 35 `stepNNNNNNN` + 5 `extension_step*`
  branches on `OpenLLM-France/Lucie-7B`, and separate repos with **optimizer states**.
- **BLOOM** (multilingual, 46 languages): *Hub-verified* — `bigscience/bloom-{560m,1b1,
  7b1}-intermediate` expose intermediate weights as **git tags**, not branches, and there
  are only **8 of them** (`global_step1000`, `10000`, `100000`, then 100k increments).

### 1.2 What is known about formation (the phenomenon-side papers)

**Gu et al. 2025, "When Attention Sink Emerges in LMs"** — ICLR 2025, peer-reviewed.
``. *Read fully.* **The single most useful paper for
RQ4.** Metric (§3.2): a head has a sink if the first token's mean importance
α₁ > ε; `Sink^ε₁` = fraction of (layer, head) pairs above threshold, with **ε = 0.3, T = 64**.
Default from-scratch setup (§4): LLaMA-style, **d=768, L=10, H=8, FFN 1536, ≈60M
non-embedding params**, 5B Pile tokens, **context 2048, batch 1M tokens, 20k steps**
(100 warm-up), LR 4e-4 cosine, AdamW wd 0.1, measured on 100 held-out sequences.
Findings, all **causal** (they trained models with and without each knob):
- **Timing:** sinks emerge "between **1k and 2k steps**" (Fig. 4 middle) ⇒ **1–2B tokens**.
- **LR** (Table 9): 8e-4/20k → 32.23%; 4e-4/20k → 18.18%; 2e-4/20k → 11.21%;
  1e-4/20k → 2.90%. Compensating steps does **not** recover it: 2e-4/40k → 16.81%,
  1e-4/80k → 6.29%. Small LR both delays *and* weakens sinks.
- **Weight decay** (Table 2): γ = 0 → **15.20%** (sinks still emerge!); 0.001 → 15.39;
  0.01 → 15.23; 0.1 → 18.18; **0.5 → 41.08**; 1.0 → 37.71; 2.0 → 6.13 (valid loss 4.23);
  5.0 → 0.01 (valid loss 5.24). Non-monotone: more wd ⇒ more sinks until optimisation
  breaks.
- **Batch size:** no effect (Table 10 left).
- **Data amount:** at 5B tokens sinks grow; at 50M/100M they never appear (Sink < 1%),
  and Fig. 28 shows this is **not** an overfitting artifact.
- **Where the sink lands is set by the data/loss, not position:** re-sampling x₁ uniformly
  gives Sink₁ = 27.03%; randomising x₁ *and* x₂ moves it to token 2 (Sink₂ = 14.08%,
  Sink₁ = 1.98%); fixing a token at position 2 or 3 moves the sink there.
- **Positional embedding:** irrelevant — Rotary, NoPE, ALiBi, relative all show sinks (§7.1).
- **Pre-norm vs post-norm:** post-norm still has sinks (13.54%); the massive activations
  simply live **before** the LN instead of in h^l (Fig. 7 left).
- **Key biases abolish the first-token sink**: with learned k* (and v* = 0), Sink₁ →
  **0.00%** at equal validation loss 3.72 vs 3.73 (Table 4), and the model has **no massive
  activations** (Fig. 7 middle). Push ‖v*‖ up and the sink migrates back to token 1 (Table 5).
- **Softmax normalisation is the root cause**: sigmoid attention *without* normalisation
  gives no sink and **no massive activations**, and this holds **at 1B parameters**
  (valid loss 3.10 vs 3.07; Sink₁ 45.11% → 2.46%; Fig. 8 right).
n_seeds = **not stated** (single runs per configuration).

**Bondarenko et al. 2023, Quantizable Transformers** — NeurIPS 2023, peer-reviewed
(stamped). *Read (setup + results).* The prior existence proof that outliers are a *recipe*
property. Outlier = value beyond 6σ. Trained from scratch **with 2 seeds per config**:
BERT-base vanilla max-∞-norm **735 ± 55**, avg kurtosis **3076 ± 262** → clipped softmax
**21.5 ± 1.5 / 80 ± 6**; gated attention 39.2/201. OPT-125m: 340 → 8.7 (gated), kurtosis
1778 → 18.9. Scales: OPT-350m 253 → 65.4; OPT-1.3B 428 → 67.2.
**Cost, and this is the anchor for RQ4:** OPT-125m pretraining ran on **a single A100 80GB**
— batch 48 × 4 accumulation = effective 192, seq 512, **125,000 steps**, ≈**53.6 h**
(Table 11); BERT-base ≈92.8 h; whole paper ≈320 A100-days, project ≈1400.
Mechanism (§3): outliers come from attention heads trying to do a **no-op**, which pushes
softmax inputs up.

**Qiu et al. 2025, Gated Attention** — NeurIPS 2025 per publisher listing (this copy carries
no venue stamp). *Read (targeted).* Causal, at scale: 30 variants of 15B-MoE and 1.7B dense
models on subsets of 3.5T tokens. Table 4: baseline max hidden activation ("M-Act")
**1053**, first-token attention **0.467** → SDPA element-wise sigmoid gate **94 / 0.048**;
head-wise gate 98/0.073; head-*shared* gate 286/0.301; gating after v-projection
125/**0.297**. Their reading (§4.2): head-specific input-dependent gating is what kills the
sink, and **"massive activations are not a prerequisite for attention sinks"** — the v-gate
row cuts activations without cutting the sink. Also nearly eliminates loss spikes.

**Barbero et al. 2025, "Why do LLMs attend to the first token?"** — COLM 2025,
peer-reviewed (stamped). *Read (targeted).* Theory: the sink prevents over-mixing /
representational collapse, so **deeper models and longer contexts should sink harder**.
Causal small-scale test (§4.1, App. A.1): LLaMA2-style **≈120M params**, 5B tokens,
context ∈ {128, 256, 512, 1024, 2048} with **tokens-per-step held constant**. Result
(Fig. 5): the sink metric rises monotonically with context length and is "nearly
non-existent" at very short contexts, while validation losses are comparable (Fig. 7).
**Cost: "a single training run on 5B tokens takes up to 24 hours … on a single NVIDIA
H100."** Packing experiments used 30B tokens.

**Gallego-Feliciano et al. 2025, Hidden Dynamics of Massive Activations** —
``, arXiv:2508.03616v2. *Read fully.* The
activation-side formation trace: **all 9 Pythia sizes (14M–12B)**, ratio of top-1 activation
to layer median, over training. Because the |a| > 100 threshold of Sun et al. "does not
generalize well to smaller models" they relax it to the top-1/median ratio. Findings:
(1) MAs are **absent at initialisation** and learned; (2) trajectories fit
`f(t) = A e^{−λ x_t} log(x_t) + K` with mean **R² = 0.984** across **188 layers**
(per-size R²: 14M 0.9307 … 1.4B 0.9956); (3) **two regimes by depth** — shallow and deep
layers *peak early then decay*, middle layers climb logarithmically and are still rising at
step 143k; the pattern sharpens above 410M. Final top-1/median ratios: Pythia-14M **83**,
1.4B **2350**, 12B **3200** (Fig. 3).
**Limits that matter:** X = **10 RedPajama sequences**; Fig. 4 is a linear interpolation of
**37** checkpoints for Pythia-1B; **n_seeds = 1**; entirely **correlational** — no ablation,
no weights, no criticality. The ML step (Table 3) predicts fit parameters from architecture
with R² 0.85 (K) down to −1.57 (γ), on 188 samples from one architecture family, and the
paper itself warns it "could have trouble generalizing".

**Ding 2026, Weibull framework** — ``,
arXiv:2605.18898v1, **independent researcher, not peer-reviewed**. *Read (targeted).*
The only **weight-level** trace over checkpoints. §3.3: in Pythia-70m, at **step 1** the
right tail beyond q99 matches the Weibull body; **by step 5,000** an isolated outlier at
|w| ≈ 1.0 has appeared (max/q99 = **9.8×**); **by step 143,000** it reaches |w| = 1.2
(max/q99 = **14.3×**). Cross-family: every one of 8 terminal checkpoints contains isolated
"dragon-king" outliers, per-block max/q99 medians ≈**7–28×** (6.6× Pythia-160m to 27.7×
OLMo-1-7B), family maxima to **107×** (OLMo-1-7B); per-block kurtosis to 446.
Explicitly links these to Yu et al.'s super weights.
**Limits: four checkpoints, one model, one seed, no ablation, and no identification of
which coordinate.** §6 concedes it offers no controlled ablation. Treat as a hint, not
evidence.

**Queipo-de-Llano et al. 2025, Sinks and Compression Valleys** — ICLR 2026.
``. *Read (targeted).* §3.1: across
**six models** (Pythia 410M/6.9B, LLaMA3 8B, Qwen2 7B, Gemma 7B, Bloom 1.7B) BOS norm
spikes, entropy drops and sink rate surges **at the same layer**; Pearson on layer-wise
deltas r = **−0.9 ± 0.18** (norm vs entropy) and **0.58 ± 0.251** (norm vs sink).
The training claim (Fig. 2): "all three phenomena **emerge together around step 1k** and
remain synchronized" — **n = 2 models** (Pythia 410M and 6.9B), **1 seed each**, checkpoints
1–143k. Also: "the layer index where these phenomena emerge is **fixed for each model** …
in Pythia 410M, the transition consistently occurs at **layer 5** regardless of input."
**Causal**: ablating the MLP contribution to BOS at layer 0 of LLaMA3-8B keeps entropy at
~0.5 bits instead of 0.02, holds sink rate at 0, and stops the 10³× BOS-norm spike (Fig. 4).
Limitations section flags model-dependent exceptions and that the theory assumes a **single**
massive row.

**Macocco et al. 2025, outlier dimensions across checkpoints** —
``, arXiv:2503.21718v4. *Read via delegate.* Different
object: **last-hidden-layer** activation dimensions whose *median* |activation| is in the
top 1%. Footnote 1 states that An et al.'s massive-activation criterion finds **no massive
activation in the last layer** — so this is a neighbouring, not the same, phenomenon.
Checkpoint work is **pythia-12b only, 15 checkpoints** (500, 1000, …, 16000, 64000, 143000),
**no seeds**. OD count: 1 → 7 → 22 → 38 → 42 over steps 500→5000, then falls to 21–22 by
16k–64k and back to 36 at 143k — **a fast ramp over steps 2000–5000, then non-monotone**.
The `∩ final` column is the finding to carry forward: **only 11 of the 38 ODs at step 4000
are still ODs at step 143,000**. Causal at each checkpoint (ablating ODs vs matched random
dims; random ablation changes accuracy in the third significant digit). Weight-side link
(Fig. 4): ODs coincide with spikes in the **last-layer MLP down-projection**'s top singular
vectors, p ≈ 0 against a random-overlap null — same module family as the super weight, wrong
layer, and correlational.

**Xu 2026, "When Do Attention Circuits Form?"** — arXiv:2606.02378v2, single-author
preprint that leans on unpublished companions. *Read via delegate; treat with care.*
Three 1B-class models × 10 log-spaced revisions: **Pythia-1B** (step1…step143000),
**OLMo-1B-0724-hf** (step1000/2B … step1454000/3048B), **OLMoE-1B-7B**. The headline for us
(Table 4, whole-model BOS-class head fraction):
- Pythia-1B: **gradual** — 0% → 10.9% (~6B) → 46.1% (~80B) → 57.8% (300B).
- OLMo-1B: **sharp** — 0.0% through 52B, 7.4% at 117B, **70.3% at 264B**, 80.9% later.
- OLMoE: gradual — 3.5% (20B) → 75.8% (5117B).
**Same data (DCLM), opposite shapes under dense vs MoE.** Also: layers 0 and 1 never
contain a BOS-class head in any model at any revision; sinks emerge mid-layer-outward;
circuit membership turns over (Jaccard formation-vs-final **0.29–0.33**) but the **sink
population only accretes — "once a head becomes a sink it stays one."**
n_seeds = **0 replicates**; the Limitations say so explicitly. Pythia's early checkpoints
**require fp32** — at random-init scale, attention values fall below fp16 range.
Massive activations and super weights are **not measured**.

**NVFP4 2026, outlier dynamics in pretraining** — arXiv:2602.02047v1.
``. *Read via delegate.* Not a public suite — their own
from-scratch GLA-1.3B / Qwen-1.7B runs, batch **4M tokens**, 40–60B tokens. Tracks excess
kurtosis, top-k channel magnitude, FTZ ratio, pre/post-softmax statistics. Claim: outliers
go from **transient drifting spikes (steps 400–5,400) to spatially fixed "hot channels"
(steps 14,200–20,400)**; top-1 magnitude "stabilizes after approximately 10,000 steps".
Weight kurtosis rises **smoothly** and plateaus (Qwen > 2, GLA ≈ 1.1) while activation
kurtosis stays volatile. Per-block 16×16 spikes persist in **every** architecture tested
including Mamba and Gated DeltaNet. No seeds, no ablation of the outliers themselves.
Earliest observation is step 400 ≈ **1.6B tokens** — after the window Gu-2025 identifies.

**Catalan-Tatjer et al. 2026, Training Dynamics and PTQ Robustness** — ICLR 2026.
``. *Read via delegate.* Tracks GPTQ 3/4-bit error
across checkpoints of OLMo-1, OLMo-2, SmolLM3, Apertus, OpenSci and Amber. The finding is
about the **LR schedule**, not outliers: quantization error is flat through the stable phase
and **spikes when LR decays**; Hessian λ_max and trace surge in the same window. Their own
160M controls show token budget 10B–100B all land at comparable error after cooldown.
**"seed", "kurtosis", "massive activation", "super weight" return zero hits in the PDF.**
Useful as a warning: an outlier statistic measured on a *decayed* checkpoint is measuring
the LR schedule as much as anything else.

**Kaul et al. 2024, "From Attention to Activation"** — arXiv:2410.17174v1, **preprint**
(stamped). Downloaded here, then removed as a byte-identical duplicate of
`phenomenon/Kaul-2024-From-Attention-to-Activation.pdf`.
*Read (setup + all knob tables).* Measures two things it argues are **disjoint**:
`%First Attn` (fraction of (query, head) pairs whose argmax key is the first token) and
per-layer **kurtosis** of the residual stream (Eq. 2; Gaussian ≈ 3). All models trained from
scratch on C4 at **sequence length 256, batch 512**: GPT2 60M/130M/350M/1.4B and Llama2-130M,
for {21B, 42B, 126B, 79B} tokens; the knob ablations run **GPT2-130M for 40k steps = 5B
tokens** — the cheapest complete sweep published. Hardware **8× V100-32GB**; **GPU-hours,
weight decay and seed count are all not stated**. Headline (Table 4): the **optimizer** is
the causal knob for kurtosis — Adam **140.0**, RMSProp 70.5, SGD+momentum **5.0**, SGD 3.2
(but PPL +6.8), **OrthoAdam 3.0 at matched PPL**; **softmax-1** kills first-token dominance
(0.333 → 0.022) but leaves kurtosis at 244.7. Kurtosis grows monotonically with scale
(77.9 → 141.5 → 161.8 → 351.9 for 60M→1.4B). **Causal**, by matched retraining, including
"re-introduce and it comes back" controls. **Measures no training dynamics at all** —
Appendix H is loss curves only; no onset step is reported anywhere.

**Liao & Monz 2024, "Is It a Free Lunch for Removing Outliers during Pretraining?"** —
arXiv:2402.12102v1, preprint. *Read (setup + Tables 1, 5, 7).* The counterweight to
Bondarenko. Trains **BERT-small (35M), BERT-base (109M), BERT-large (335M)** from scratch on
a **single A6000 48GB**, 125K steps × batch 2048 × seq 128. Two findings that matter:
(i) **removing outliers is not free** — clipped softmax cuts BERT-base kurtosis
3931.9 → 160.3 but drops GLUE average **81.7 → 68.2** (CoLA 61.6 → 37.6); their NCS variant
recovers only to 73.8. (ii) **Pretraining sequence length is a strong, cheap knob**
(Table 7, BERT-small pretrained separately at each length): kurtosis **46.2 / 125.9 /
1575.5** for seq 64 / 128 / 256. Also a scale counter-example: BERT-large vanilla kurtosis
**285.0** is *lower* than BERT-base's **3931.9**. **Causal** for softmax variant and sequence
length; the OPT vanilla/CS rows are **borrowed from Bondarenko**, not run by them. Pretraining
seed count **not stated** (the three seeds are GLUE fine-tuning). The one timing datum in the
paper: the W8A8 gap "may be observed after **20K iterations** of BERT-base or OPT-125M" —
16% of budget.

**Shi et al. 2026, "A Single Layer to Explain Them All"** — arXiv:2605.08504v2, preprint
(ICML 2026 per the PDF). *Read via delegate.* Defines an **"ME Layer"** — the layer where
activations jump "several hundreds times" (no numeric threshold given) — and locates it per
model: Phi-3-mini **2**, Mistral-7B **2**, Qwen2.5-7B **4**, Llama3.1-8B **6**, Qwen3-8B **7**.
**Trains nothing from scratch; the smallest model anywhere in it is ~3.8B; it measures no
training dynamics and explicitly outsources that question to Gallego-Feliciano-2025.**
Its intervention (WeMask) is post-hoc on trained models. Useful here only as the current
mechanistic account of *where*, and for the citation trail to Oh et al. 2024.

**Chen, Lin & Yao 2026, "Attention Sinks Induce Gradient Sinks"** — arXiv:2603.17771v2,
preprint (stamped). *Read via delegate.* **The most useful from-scratch recipe in this set
for RQ4**, because it publishes a complete config at 0.1B. Defines a **gradient sink**
R_GS(s) = ‖∇ at token s‖ / mean‖∇ at other tokens‖ at the post-RMSNorm attention input, and
reports the **continuous ratio rather than a threshold**. From-scratch Llama-shaped models on
C4: **0.1B = 101M params, L=16, d=512, H=8, FFN 1408, 20,000 steps × batch 512 × seq 1024 =
10.49B tokens, AdamW wd 0.1, LR 2e-3→2e-4 cosine, A100 80GB** (GPU count and hours not
stated); also 0.3B and 1B. Median R_GS(0): **0.1B baseline 15.5 [5.2, 21.9]**, 1B 9.4,
Qwen3-8B 106.7. **The causal result is a dissociation**: their V-scale reparameterisation
*preserves* attention sinks while *suppressing* massive activations (gradient-sink median
**40.3 → 5.6**) at comparable downstream scores — matching Qiu-2025 from the other direction.
**Pretraining seed count not stated** (the 5 seeds are NIAH evaluation). Checkpoint-coloured
figures span steps 0–20k/25k but **no onset step is ever stated in text** — a gap this
project could fill directly.

### 1.3 Method papers

**Olsson et al. 2022, Induction Heads** — Transformer Circuits Thread, **not
peer-reviewed**. *Read (targeted).* Four signals co-occur in one window: in-context-learning
score (loss@50th token − loss@500th) jumping from **<0.15 to ≈0.4 nats**; prefix-matching
score; a **loss bump** — "the only place in training where the loss is not convex"; and a
pivot in the first two PCs of per-token losses. The robust statistic is
**dLoss/d ln(token index in context)** rendered as a heatmap over (training tokens ×
context index). Window: **~2.5–5 ×10⁹ tokens** for large models, **1–3B** for small ones.
Density: small models get **200 snapshots every 50 steps**; large models get **15
snapshots at 2× exponential spacing**. The sentence to quote: *"In large models, we have low
time resolution on our analysis over training. Co-occurrence when one only has 15 points in
time is less surprising and weaker evidence."* **n_seeds = 1 per model** (34 models).
**How the orange phase-change band's endpoints were computed is not stated** — treat it as
eyeballed.

**Nanda et al. 2023, Progress Measures for Grokking** — ICLR 2023, peer-reviewed.
*Read (targeted).* The template for the counter-hypothesis. A **progress measure**
"precedes and is causally linked to the phase transition, and varies more smoothly."
Restricted loss = keep only the 20 Fourier terms at the 5 key frequencies; excluded loss =
remove exactly those. Three phases (memorization 0–1.4k, **circuit formation 1.4k–9.4k
while train and test loss are flat**, cleanup 9.4k–14k). 5 seeds; **memorization completes
by ~1400 steps in all five**, but the later phase boundaries **differ by seed** (Fig. 18).
§6 concedes: *"we lack a general notion of criticality that would allow us to predict when
the phase transition will happen ex ante."* Checkpoint spacing **not stated**.

**Power et al. 2022, Grokking** — arXiv:2201.02177v1; no venue stamp on this copy.
*Read (targeted).* Train accuracy saturates at <10³ steps, validation at ~10⁶ — measured on
log axes (the log scaling is read off the figures, **not asserted in the text**).
3 seeds (7 for §3.1.1). Cautions worth copying: the budget determines whether you observe
"never" (some operations never generalise within budget), the LR window is ~1 order of
magnitude wide, and the effect only shows near the minimum viable dataset size.
App. A.5 reports **bimodality across seeds** — they trained until "approximately half"
reached high validation accuracy — with **n not stated**.

**Zhao et al. 2026, Random Scaling of Emergent Capabilities** — arXiv:2502.17356v5,
preprint. *Read (targeted).* **The reason not to trust a 3-seed abruptness claim.** Every
axis in the paper is model scale or data mixture, **not training step** — the transfer to a
step axis is by analogy, and the paper never claims it. With **200–250 seeds** (synthetic)
and **80** (Qwen2.5 partial-reinit), the **mode** accuracy jumps sharply while
**P(success)** and **mean | success** both move gradually (Fig. 3). Two seeds are named:
one gives an emergent curve, another a linear one, same task and hyperparameters. Bimodality
appears **before** most seeds break through. Toolkit: **Hartigan's dip test**
(p < 0.001 reported), **Wasserstein-L2** between per-scale distributions,
**95% CIs from 1000 bootstrap samples**, and Srivastava's breakthroughness B / linearity L
(B uses root-*median*-square in the denominator, L root-mean-square).
Two things that bite: **a continuous metric does not protect you** — NLL *exposed* clusters
that accuracy hid (§3.3); and their free null is that **final pretraining loss across the
same 80 seeds is unimodal** (mean = median = 11.62, Fig. 9), so multimodality tracks the
mechanism, not run quality. **No "you need N seeds" prescription is stated.**
Their **partial-reinitialisation trick** (reinit the last attention layer + LM head, then
continue-pretrain) is how they got 80 seeds on a real LLM without 80 pretraining runs.

**Schaeffer et al. 2023, Emergent Abilities a Mirage** — NeurIPS 2023 Outstanding Paper per
publisher listing; **this copy is stamped "Preprint"**. *Skimmed.* The other artifact route:
nonlinear/discontinuous metrics manufacture apparent abruptness. Forces you to state
whether the super-weight criticality metric is continuous **before** calling anything a
phase change. Note that Zhao's finding cuts the other way — continuity is necessary, not
sufficient.

**"Evidence of Phase Transitions in Small Transformer-Based LMs"** (arXiv:2511.12768v1,
IEEE Access template, acceptance not confirmed). *Skimmed.* A worked example of declaring a
transition when several independent metrics synchronise inside one narrow window. Method
reference only.

---

## §2 Instrument table — public checkpoint suites, verified

All Hub columns queried live on **2026-09-05** via `https://huggingface.co/api/models/<id>/refs`.
"tok/step" is *inferred* from seq × batch or from the suites' own `tokensNNNB` revision labels.
**"First real ckpt"** is the first non-zero checkpoint, in tokens — the column that decides
whether a suite can see formation at all.

| Suite / repo | Sizes | Ckpts per run | Spacing | tok/step | **First real ckpt** | Seeds | Data order | Multiling. | Revision naming |
|---|---|---|---|---|---|---|---|---|---|
| **Pythia** `EleutherAI/pythia-{14m…12b}[-deduped]` | 14M–12B (10) | **154** (Hub: 155 branches incl. `main`) | steps 0,1,2,4,…,512, then every 1,000 to 143,000 | 2.097M | **2.1M tokens (step 1)** | 1 per (size, dataset) | **yes** — exact dataloader replay script (§2.7) | no (English Pile) | `stepNNNNN` (branch) |
| **PolyPythias** `EleutherAI/pythia-{14m,31m,70m,160m,410m}-seed{1..9}` | 14M, 31M, 70M, 160M, 410M | **154** (Hub-verified on 410m-seed3, 14m-seed3) | identical to Pythia | 2.097M | **2.1M tokens** | **9 + original = 10 per size** | yes, per-seed pre-shuffled index datasets | no | `stepNNNNN` |
| ↳ *disentangling runs* `pythia-160m-{data,weight}-seed{1,2,3}` | 160M only | 154 each (Hub-verified) | identical | 2.097M | 2.1M | 3 data-only + 3 init-only | yes | no | `stepNNNNN` |
| **OLMo-1B-0724** `allenai/OLMo-1B-0724-hf` | 1B | **1,446** (Hub-verified) | **every 1,000 steps, uniform**, step 0 → 1,454,000 | 2.097M (from `step1000-tokens2B`) | **2B tokens** | 1 | yes (OLMo §3.3: order reconstructible) | no | `stepNNNNNNN-tokensNNNNB` |
| **OLMo-1B (0424)** `allenai/OLMo-1B-hf` | 1B | 351 (Hub-verified) | irregular; every 1,000 in places, gaps elsewhere | ~4.2M (from `step1000-tokens4B`) | 4B tokens | 1 | yes | no | `stepNNNNNNN-tokensNNNNB` |
| **OLMo-2-1B** `allenai/OLMo-2-0425-1B` | 1B | 267 (Hub-verified) | `stage1` coarse early (0, 300, 10k, 20k, 23.1k, 30k, 40k, 50k, 60k, 66.2k…) then `stage2-ingredientN` | ~2.1M | 1B tokens (`stage1-step300`) | 1 pretrain; **3–4 mid-train anneal orders** | yes | no ("not trained for multilingual tasks") | `stage1-stepN-tokensNB`, `stage2-ingredientK-…` |
| ↳ `allenai/OLMo-2-0425-1B-early-training` | 1B | **37** (Hub-verified) | **every 1,000 steps, 0 → 36,000** | ~2.1M | **3B tokens** (`stage1-step1000-tokens3B`) | 1 | yes | no | `stage1-stepN-tokensNB` |
| **OLMo-2-7B** `allenai/OLMo-2-1124-7B` | 7B | 964 (Hub-verified) | stage1 + stage2 ingredients | ~4.2M | not checked | 1 | yes | no | same |
| **Ettin** (JHU-CLSP) | 17M–1B, **encoder AND decoder** | **236** | **every 8.5B tokens** (linear) | — | **8.5B tokens** | 1 | yes, released in batch order | English + code | not stated in PDF; Hub id not printed |
| **MultiBERTs** | BERT-Base only | **28**, and only for **5 of 25 seeds** | every 20k steps to 200k, then every 100k | 0.131M (256×512) | **≈2.6B tokens** (*inferred*) | **25 final, 5 with trajectories** | not stated | no | not in PDF |
| **DataDecide** | 4M–1B (14) × 25 recipes | "30K+" total | **not stated in PDF** | varies | not stated | 3 — but **seeds 2–3 truncated at 25% of budget below 1B** | not stated | no | **not stated** |
| **LLM360 Amber** `IFM/Amber` | 6.7B | **359** (Hub-verified: `ckpt_000`…`ckpt_358`, plus `main`; the PDF says 360) | 1 per data chunk ≈ **3.5B tokens** | 4.59M | ≈3.5B tokens | 1 | **yes** — 360 released chunks in order | no | `ckpt_NNN` |
| **LLM360 CrystalCoder** `IFM/Crystal` | 6.7B | **250** branches (Hub) / 143 in PDF | per data chunk, 3 phases | — | — | 1 | yes | English + code | `CrystalCoder_phaseK_checkpoint_NNNNNN` |
| **SmolLM2-1.7B** `HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints` | 1.7B | **41** (Hub-verified) | every 125,000 steps, 125k → 5,125k | 2M (Table 6) | **250B tokens** | 1 | **no** (mixtures rebalanced by hand online) | no | `step-NNNNNNN` |
| **SmolLM2-135M** `…-135M-intermediate-checkpoints` | 135M | **8** (Hub-verified) | every 240,000 steps | 2M | 480B tokens | 1 | no | no | `step-NNNNNN` |
| **SmolLM3-3B** `HuggingFaceTB/SmolLM3-3B-checkpoints` | 3B | **118** stage ckpts + 10 longctx (Hub) | every 40,000 steps, from step 40,000 | not checked | — | 1 | no | **yes, 6 languages** (per model card, *not read here*) | `stageK-step-N` |
| **TinyLlama** | 1.1B (+ a Chinese variant) | released, **count/interval not stated in PDF**; Hub `TinyLlama-1.1B-intermediate-step-1431k-3T` is a single revision | — | 2M (v1.0) | — | 1 | no | Chinese variant | per-repo, not revisions |
| **MAP-Neo** `m-a-p/neo_7b_intermediate` | 7B (also 2B) | **186 `iter_*` directories** (Hub-verified) | uniform, `iter_0002384` increments (2,384 steps ≈ 20B tokens), `iter_0002384` → `iter_0444213` | ~8.4M | ≈20B tokens | 1 | no | **yes, En/Zh** | **raw Megatron shards** `iter_NNNNNNN/mp_rank_NN/model_optim_rng.pt` — **not HF-loadable revisions** |
| **BLOOM** `bigscience/bloom-{560m,1b1,7b1}-intermediate` | 560M, 1.1B, 7.1B | **8** (Hub-verified, git **tags** not branches) | `global_step1000`, `10000`, `100000`, then 100k | — | — | 1 | no | **yes, 46 languages** | `global_stepN` (tag) |
| **Apertus-8B** `swiss-ai/Apertus-8B-2509` | 8B (also 70B) | **44** pretrain + 27 longctx (Hub-verified) | **every 50,000 steps ≈ 210B tokens** | 4.2M→8.4M | **210B tokens** | 1 | no | **yes, 1811 languages, ~40% non-English** | `stepNNNNNNN-tokensNNNB` |
| **Lucie-7B** `OpenLLM-France/Lucie-7B` | 7B | **35** + 5 extension (Hub-verified) | 5k, 10k, …, 25k, 50k, 75k, then every 25k to 753,851 | ≤4.19M (batch ramps 256→1024 seqs) | **≈20B tokens** (step 5,000) | 1 | no | **yes** (Fr/En/De/Es/It + code); optimizer states also released | `stepNNNNNNN` |
| **EuroLLM-1.7B** `utter-project/EuroLLM-1.7B` | 1.7B, 9B, 22B | **Hub: 1 branch (`main`) — no revisions** | — | 12M | — | 1 | no | **yes, 35 langs, 20% parallel data, MT-evaluated** | — |

**Answering the four questions the task poses:**

**(a) Tracing a KNOWN super weight backward — OLMo-1B-0724.** *Verified: the checkpoints
exist.* `allenai/OLMo-1B-0724-hf` publishes **1,446 revisions**, step 0 to 1,454,000, uniform
every 1,000 steps, named `stepNNNNNNN-tokensNNNNB`, and Yu et al. Table 2 gives the
coordinate to trace: **layer 1, `mlp.down_proj[1764, 8041]`** (the older `OLMo-1B-hf` is
layer 1 `[1764, 1710]` — note the **shared row index 1764**, which is itself a datum).
**But the grid is the wrong shape for the question.** At 2.097M tokens/step, the entire
formation window that Gu-2025 identifies (1k–2k steps ⇒ 1–2B tokens) and that
Queipo-de-Llano-2025 places at "around step 1k" is covered by exactly **two** OLMo
checkpoints: step 0 and step 1,000 (= 2B tokens). *Inferred:* OLMo-1B-0724 can tell you
what the super weight looks like **after** it forms, and can answer "is the coordinate
stable once formed?" beautifully (1,445 post-formation points). It cannot resolve formation.
Xu-2026 hit exactly this wall on the same model.

**(b) Multi-seed formation — PolyPythias, unambiguously.** 10 seeds × 5 sizes × 154
checkpoints, with 11 checkpoints before step 1,000 (i.e. **before 2.1B tokens**), Apache-2.0,
Hub-verified. It is the only public instrument that can ask *whether the super-weight
coordinate is the same across seeds*, and the `-data-seed*` / `-weight-seed*` 160M runs are
the only way to attribute the answer to data order versus initialisation. The **prerequisite
risk**: nobody has published a super weight in any Pythia model. Yu et al. Table 2 lists
Llama, Mistral, OLMo and Phi-3 — **not Pythia**. Establishing existence at 410M is the gate.

**(c) Encoder vs decoder — Ettin** for a modern matched pair (identical data, order, recipe;
only objective and attention mask differ), **MultiBERTs** if you need encoder *seeds*.
Neither has early-training resolution: Ettin's first checkpoint is 8.5B tokens, MultiBERTs'
is ≈2.6B (*inferred*). Both are fine for "does an encoder have a super weight at all", which
is a cheaper and still-unanswered question.

**(d) Multilingual — the honest answer is: nothing good.** Ranked by usability:
**Lucie-7B** (35 revisions from ≈20B tokens, optimizer states released, genuinely
multilingual) > **Apertus-8B** (44 revisions but first at **210B tokens** — the "When
Meanings Meet" authors say outright this granularity *masks* the development they were
looking for) > **MAP-Neo** (bilingual, 186 checkpoints, but raw Megatron shards needing
conversion) > **BLOOM** (46 languages, but only 8 tags) > **EuroLLM** (the best MT model,
20% parallel data, WMT/FLORES/COMET evaluation — and **zero public intermediate
checkpoints**; the 26 studied by arXiv:2601.22851 are not on the Hub).
*Inferred, and the point worth acting on:* the MT angle and the formation-trace angle are
**disjoint in the public artifact space**. Any cross-lingual component has to attach to
final checkpoints (e.g. "where does the super weight sit in EuroLLM / NLLB / BLOOM, and does
ablating it damage translation more in low-resource directions?") rather than to a trajectory.

---

## §3 What is known about WHEN outlier structure forms — and what has not been measured

### 3.1 The timing evidence, by object

| Object | When | Where measured | Density | Seeds | Causal? |
|---|---|---|---|---|---|
| **Attention sink** (Sink^0.3₁) | **steps 1k–2k of a 20k-step run ⇒ 1–2B tokens** | Gu-2025 Fig. 4 middle, own 60M models | continuous logging | not stated | **yes** — knob sweeps |
| **Sink + massive activation + entropy collapse, jointly** | "**around step 1k**" ⇒ ≈2B tokens | Queipo-de-Llano-2025 Fig. 2, Pythia 410M & 6.9B | 154 ckpts available | **n=2 models, 1 seed each** | ablation is causal; the *timing* claim is correlational |
| **Massive activation ratio (top-1/median)** | absent at init; **shallow/deep layers peak then decay within 60k steps**, middle layers still climbing at 143k | Gallego-Feliciano-2025 Figs. 4–5, all 9 Pythia sizes | 37 ckpts for Pythia-1B | **1** | correlational only |
| **Isolated weight outlier in `mlp.down_proj`+FFN** | absent at **step 1**, present at **step 5,000** (max/q99 = 9.8×), grows to 14.3× by 143k | Ding-2026 §3.3, **Pythia-70m only** | **4 checkpoints** | **1** | correlational; coordinate not identified |
| **Outlier *dimensions* (last layer)** | ramp over **steps 2,000–5,000**, then non-monotone | Macocco-2025 Table 7, pythia-12b | 15 ckpts | **1** | causal per checkpoint (ablation vs random dims) |
| **BOS-sink head population** | **configuration-dependent**: gradual in Pythia-1B (0→57.8% over 300B), **sharp in OLMo-1B (7.4%→70.3% between adjacent checkpoints, 117B→264B)**, gradual in OLMoE | Xu-2026 Table 4 | 8–10 log-spaced revisions each | **0 replicates** | group ablation is causal; timing is not |
| **Channel-level activation outliers** | drifting spikes at steps 400–5,400 → **fixed hot channels by ~10k steps** | NVFP4-2026 §3.3 | ~10 snapshots, earliest 1.6B tokens | 1 | no |
| **Quantization error** | flat through the stable phase; **spikes when LR decays** | TrainingDynamics-2025 Figs. 1–2 | hundreds across 6 suites | 0 | correlational (Hessian) |
| **Induction heads / in-context learning** | **2.5–5 ×10⁹ tokens** (large), 1–3B (small) | Olsson-2022 | **200 snapshots every 50 steps** (small) | 1 per model | ablation-supported |
| **Circuits generally (Pythia)** | steps 1k–5k ⇒ **2–10B tokens** | Tigges et al., *as reported in* PolyPythias §3 | — | — | — [unopened: Tigges et al.] |
| **Whole-model metric departure from random baseline** | **step 10³ ≈ 2B tokens**; convergence ≈ step 10⁴ ≈ 20B | PolyPythias §6 | 154 × 10 seeds | **10 per size** | correlational |

*Inferred synthesis:* four independent measurements of four different objects — sink metric,
massive activation, weight tail, and generic learning — all land in a window around
**10⁹–10¹⁰ tokens (Pythia steps ~10³–5×10³)**. That is a real convergence, and it is exactly
where OLMo, Ettin, Apertus, SmolLM2 and Lucie have **zero to two** checkpoints.

### 3.2 How abrupt, and how seed-dependent

- **Abruptness is not a property of the phenomenon; it is a property of the run.** Xu-2026
  Table 4 is the cleanest statement: the *same* measure, on the *same* data (DCLM), is sharp
  under a dense 1B model and gradual under a MoE. Anyone claiming "super weights form
  abruptly" from one suite has to explain that table.
- **Checkpoint density determines what "abrupt" can even mean.** Olsson-2022's own
  concession — 15 log-spaced snapshots make co-occurrence "less surprising and weaker
  evidence" — applies verbatim to Xu-2026's 8–10 revisions, to Ding-2026's 4, and to any
  OLMo-based trace of an event that finishes inside its first inter-checkpoint gap.
- **Seed dependence is essentially unmeasured for outlier structure.** Across
  Gu-2025, Gallego-Feliciano-2025, Ding-2026, Queipo-de-Llano-2025, Macocco-2025, Xu-2026,
  NVFP4-2026, TrainingDynamics-2025, Kaul-2024, Liao-2024, SingleLayer-2026 and
  GradientSinks-2026, the number of papers stating a **pretraining** seed count is **zero**.
  Every mechanism figure in that list is n = 1 per configuration.
- **What multi-seed work exists says the boundaries move.** Nanda-2023 Fig. 18: memorization
  completes by ~1,400 steps in **all five** seeds, but the *later* phase boundaries differ by
  seed. PolyPythias Table 2: state transitions cluster tightly within a size (160M: 2k ± 0,
  18k ± 0.8k) except where instability intervenes (410M: 18k ± **22.6k**, driven by two
  outlier seeds). So: expect an early boundary to be seed-stable and a late one not to be —
  and expect ~2 runs in 50 to be pathological.
- **Zhao-2026 is the reason to distrust a 3-seed abruptness claim**, even though its axis is
  scale and not training step (a caveat the paper never asks you to ignore). With 80–250
  seeds, per-seed curves are abrupt while P(success) moves smoothly, and bimodality appears
  *before* most seeds break through. Their free null — final pretraining loss across 80 seeds
  is unimodal, mean = median = 11.62 — is directly copyable.
- **Schaeffer-2023 is the other half:** even with enough seeds, a discontinuous metric
  manufactures cliffs. "Perplexity blows up by 1000×" is an **Accuracy-family metric**
  (thresholded catastrophic failure), not a linear one. Zhao's finding that NLL *exposed*
  clusters that accuracy hid means continuity is necessary but not sufficient.

### 3.3 What the literature has NOT measured — the actual hole

1. **Causal criticality of a single named weight across checkpoints.** Nobody has taken a
   coordinate `layers[L].mlp.down_proj.weight[i, j]`, zeroed it at each of N checkpoints, and
   plotted the resulting perplexity ratio against training step. Ding-2026 comes closest and
   measures *magnitude*, not criticality, at four steps, without identifying the coordinate.
   The distinction matters: Macocco-2025 Table 7 shows outlier *count* and outlier *causal
   effect* on accuracy have different onsets (ODs appear at step 3,000; ablation starts
   hurting only around step 5,000).
2. **Whether the super-weight coordinate is the same across seeds.** Completely open. There
   is one suggestive datum: Yu et al. Table 2 gives OLMo-1B `[1764, 1710]` and
   OLMo-1B-0724 `[1764, 8041]` — **the same row across two different training runs of the
   same architecture**. *Inferred*: if the row index is architecture-determined and the column
   is run-determined, that is a publishable finding on its own, and PolyPythias can test it
   with n = 10.
3. **Whether the coordinate is stable once formed.** The two adjacent measurements
   **disagree**: Macocco-2025 finds outlier dimensions **turn over** (11 of 38 survive from
   step 4,000 to 143,000), while Xu-2026 finds sink heads **never** drop out ("once a head
   becomes a sink it stays one"). Which pattern super weights follow is a clean binary
   question with a cheap experiment.
4. **Whether any model ≤1B has a super weight at all.** The project's own gate. The
   evidence is only indirect and mixed: massive activations exist at Pythia-14M
   (top-1/median = **83**, Gallego-Feliciano Fig. 3, using a *relaxed* definition because
   Sun et al.'s |a| > 100 threshold "does not generalize well to smaller models"); weight
   outliers exist in Pythia-70m (max/q99 **9.8×** at step 5k, Ding-2026); activation
   kurtosis 313.8 and max activation 1856 at GPT2-60M (Kaul-2024 Table 2); ‖X‖_∞ = 801.5 at
   BERT-small **35M** (Liao-2024 Table 1). None of these is a *single-scalar-zeroing-collapses-
   the-model* result.
5. **Whether the phenomenon is objective-dependent** (encoder vs decoder). Ettin makes this
   answerable and nobody has asked.
6. **Anything multilingual.** No paper in this folder measures outlier structure in a
   multilingual model across training, and no paper asks whether ablating a super weight
   damages some languages more than others.
7. **The confound nobody controls: LR decay.** TrainingDynamics-2025 shows outlier-adjacent
   statistics move sharply at LR decay independent of tokens. Pythia's cosine decays to
   0.1× peak by step 143,000; a "the super weight grows late in training" finding measured on
   a decayed suite is partly measuring the schedule.

---

## §4 Formation CAUSES — the knobs with evidence, and what the smallest experiment costs

### 4.1 Knob table

Effect sizes are quoted as published. "Causal" means the authors trained matched models with
and without the knob. Objects differ across rows — sink rate, kurtosis, max activation — and
**none of these is a super weight**; treat every row as a prior about a neighbouring
phenomenon, not as evidence about the object of study.

| Knob | Varied by | Effect | Causal? | Scale it was shown at |
|---|---|---|---|---|
| **Softmax normalisation** | Gu-2025 §7.4 | sigmoid attention **without** normalisation ⇒ **no sink and no massive activations**, holds at **1B** (valid loss 3.10 vs 3.07; Sink₁ 45.11%→2.46%) | **yes** | 60M and 1B |
| **Clipped softmax / softmax-1** | Bondarenko-2023; Kaul-2024; Liao-2024 | BERT max-∞ **735→21.5**, kurtosis **3076→80**; Kaul: `%First Attn` 0.333→0.022 but **kurtosis unaffected** (244.7) | **yes** | 35M–1.3B |
| **Gated attention (SDPA head-specific sigmoid gate)** | Bondarenko-2023; Qiu-2025 | OPT-125m ‖X‖∞ 340→8.7, kurt 1778→18.9; Qiu Table 4 **M-Act 1053→94, first-token attn 0.467→0.048** | **yes** | 125M → 15B-MoE |
| **Key biases (learned k\*, v\*=0)** | Gu-2025 §7.3 Table 4 | Sink₁ → **0.00%** at matched loss; **no massive activations**. Raise ‖v\*‖ and the sink migrates back (Table 5) | **yes** | 60M |
| **Optimizer** | Kaul-2024 Table 4 | GPT2-130M/5B tokens: Adam κ **140.0**, RMSProp 70.5, **SGD+momentum 5.0**, SGD 3.2 (but PPL +6.8), **OrthoAdam 3.0 at matched PPL** | **yes** | 130M |
| **Learning rate** | Gu-2025 Table 9 | 8e-4→32.23%, 4e-4→18.18%, 2e-4→11.21%, 1e-4→**2.90%**; compensating steps does **not** recover (1e-4/80k → 6.29%) | **yes** | 60M |
| **Weight decay** | Gu-2025 Table 2 | **non-monotone**: γ=0 → 15.20% (sinks *still* form), 0.1 → 18.18, **0.5 → 41.08**, 1.0 → 37.71, 2.0 → 6.13, 5.0 → 0.01 (loss 5.24). Ding-2026 separately: Selection-class drift tracks **T/τ = T·η·λ_wd** monotonically across 5 Pythia sizes | **yes** (Gu) | 60M |
| **Sequence / context length** | Barbero-2025 Fig. 5; Liao-2024 Table 7 | Barbero: sink metric rises monotonically 128→2048 at fixed tokens-per-step, "nearly non-existent" at short context. Liao: pretraining seq 64/128/256 ⇒ kurtosis **46.2 / 125.9 / 1575.5** | **yes**, both | 120M / 35M |
| **Data amount** | Gu-2025 §5 | 5B tokens ⇒ sinks grow; **50M–100M tokens ⇒ Sink < 1%**, and Fig. 28 rules out overfitting | **yes** | 60M |
| **Which token is the sink** | Gu-2025 §5 | set by data/loss, not position: uniform-random x₁ ⇒ Sink₁=27.03%; random x₁ *and* x₂ ⇒ sink moves to token 2 (14.08% vs 1.98%); a fixed token at position 3 takes the sink | **yes** | 60M |
| **Pre- vs post-norm** | Gu-2025 §7.2 | post-norm still sinks (13.54%); massive activations just relocate to **before** the LN | **yes** | 60M |
| **Normalisation type** | Kaul-2024 Table 4 | LayerNorm κ 263.7, RMSNorm-M 230.4, **RMSNorm-S 140.0** (~40% cut, "still high") | **yes** | 130M |
| **QK-Norm** | Ding-2026 Fig. 7 | Qwen3-8B (with QK-Norm) shows "visibly tighter" Q/K tails than OLMo-1-7B and Qwen2.5-14B (without) | **no** — cross-family observation | 7–14B |
| **Value-path rescaling (V-scale)** | GradientSinks-2026 §5 | **dissociates the two**: preserves attention sinks, **suppresses massive activations** (residual + MLP output norms at token 0), gradient-sink median **40.3 → 5.6**, downstream comparable | **yes** | 0.3B and 1B |
| **Positional embedding** | Gu-2025 §7.1; Kaul-2024 Table 4 | irrelevant. Rotary/NoPE/ALiBi/relative all sink; Kaul: none κ=283.3, absolute 263.7, rotary 391.9 | **yes** | 60M / 130M |
| **Batch size** | Gu-2025 Table 10 | **no effect** | **yes** | 60M |
| **FFN biases, activation fn, multi-head design** | Gu-2025 App. D; Kaul-2024 | no meaningful effect (κ 291.7→263.7 for bias removal) | **yes** | 60M / 130M |

**Two published counter-results to hold onto.** (i) Qiu-2025 §4.2: the v-projection gate cuts
massive activations (125) *without* cutting the first-token sink (0.297) — "massive
activations are **not** a prerequisite for attention sinks", and GradientSinks-2026 shows the
same dissociation from the other side. So a super-weight result cannot be inferred from a
sink result. (ii) Liao-2024 Table 1: BERT-**large** (335M) has *lower* vanilla kurtosis
(285.0) and ‖X‖∞ (151.4) than BERT-**base** (3931.9 / 3857.0). Outlier magnitude is not
monotone in scale within an encoder family, contra Kaul-2024's monotone decoder trend. Whatever
direction a small run comes out, there is precedent for both.
And the negative that constrains ambition: Liao-2024's whole point is that removing outliers
is **not free** — clipped softmax cuts BERT-base kurtosis 3931.9→160.3 but drops GLUE average
**81.7 → 68.2** (CoLA 61.6→37.6).

### 4.2 The smallest from-scratch experiment, costed from published setups

Every number below is *read* from the paper cited; the per-run GPU-hour figures are the
papers' own, and my extrapolations are marked *inferred*.

| Published recipe | Params | Tokens | Steps × batch × seq | Hardware and time, **as reported** |
|---|---|---|---|---|
| **Gu-2025 default** (LLaMA-style, d=768, L=10, H=8, FFN 1536) — *the closest published match to a super-weight substrate* | ≈60M non-embed | **5B** | 20,000 × 1M tok × 2048 | **not stated** |
| **Barbero-2025** (LLaMA2-style) | ≈120M | **5B** | — × — × {128…2048} | **"up to 24 hours … on a single NVIDIA H100"** per run |
| **Bondarenko-2023 OPT-125m** | 125M | ≈12.3B (*inferred*: 125k × 192 × 512) | 125,000 × 192 × 512 | **single A100 80GB, ≈53.6 h** (Table 11) |
| **Kaul-2024 ablation** (GPT2-130M) — *the cheapest complete knob sweep published* | 130M | **5B** | 40,000 × 512 × 256 | 8× V100-32GB (hours not stated) |
| **GradientSinks-2026 0.1B** (Llama-shaped, L=16, d=512, H=8, FFN 1408) — *fully specified* | **101M** | 10.49B | 20,000 × 512 × 1024 | A100 80GB (count/hours not stated) |
| **Liao-2024 BERT-small** | **35M** | ≈32.8B (*inferred*) | 125,000 × 2048 × 128 | **single A6000 48GB** |
| **TinyLlama Table 2** — the one clean scaling anchor | 1.1B | 300B | — | **3,456 A100-hours** (Pythia-1.0B: 4,830; MPT-1.3B: 7,920) |

***Inferred* costing for RQ4.** Take Bondarenko's measured **≈54 A100-hours for 125M / 12.3B
tokens** as the unit. Gu-2025's 5B-token / 60M recipe is roughly **0.2× the FLOPs**, so
≈10–15 A100-hours per run; Barbero's independent measurement (**24 H100-hours for 120M /
5B**) brackets the same order. A 12-run single-knob sweep at Gu's scale with **3 seeds per
arm** is therefore ≈**400–900 single-GPU hours** — plausible on a SLURM cluster with
single-GPU jobs, and it would exceed the seed rigour of every paper in §4.1, all of which are
n = 1 per configuration.
Two ways to cut it further, both published: (1) run the sweep at Gu's 5B rather than
Bondarenko's 12.3B — Gu shows sinks form by **1–2B tokens**, so 5B is already ~3× past the
event; (2) Liao-2024's Limitations note that the quantization gap "manifests early … after
20K iterations" of a 125K-step run (**16% of budget**) — i.e. a screening run can be stopped
at ~20% if the readout is already separated.
**Two knobs to pre-register against, with published effect sizes and near-zero extra cost:**
optimizer (Kaul: Adam κ 140.0 → OrthoAdam 3.0 at 130M/5B) and **pretraining sequence length**
(Liao: 46.2 → 125.9 → 1575.5 for 64→128→256), which changes throughput rather than parameter
count. Budget separately for the fact that Kaul's **SGD arms needed 8× longer to converge**.

---

## §5 Open problems sized for one semester

A shared **measurable criterion for "abrupt vs gradual"** is defined once here and reused by
every problem below, because the two failure modes named in §3.2 (Olsson: too few points;
Zhao/Schaeffer: wrong metric or too few seeds) both bite unless it is fixed in advance.

> **Criterion.** Let `c(t) = log10( PPL(model_t with the coordinate zeroed) / PPL(model_t) )`
> — a **continuous, unbounded** criticality score, deliberately not a thresholded
> collapse/no-collapse label (Schaeffer-2023). Define `t_10` and `t_90` as the checkpoints at
> which `c` first reaches 10% and 90% of its own run maximum, and the **transition width**
> `W = log10(t_90) − log10(t_10)` in tokens.
> **Abrupt** := `W` is not distinguishable from the checkpoint-grid floor —
> i.e. the 95% CI over seeds for `W` contains `log10(t_next/t_prev)` for the relevant grid
> spacing. **Gradual** := the CI for `W` excludes that floor and spans ≥ 1 decade.
> Report `W [95% CI], n_seeds = N, family = M`. **Report the grid floor in the same
> sentence** — it is the whole reason Olsson-2022 called 15 snapshots weak evidence.

### P1 — Does super-weight criticality form abruptly? (RQ1, the core)

*Why open:* nobody has plotted causal criticality of a single named coordinate against
training step (§3.3.1). The adjacent measurements are magnitude, not criticality
(Ding-2026), or a different object (Macocco, Xu, Gallego-Feliciano), and all are n=1.

*Experiment:* **PolyPythias**, sizes 160M and 410M, **all 10 seeds each**, all **154**
checkpoints (or a 40-point log grid if throughput binds). At each (seed, checkpoint):
run the Yu et al. detection (max-magnitude `down_proj` input/output activation scan over
layers, one prompt) to locate a candidate coordinate; zero it; score `c(t)` on a fixed
wikitext-2 slice. **Null:** the same ablation applied to (i) the step-0 model — a free null,
per the rigor floor — and (ii) `k` random coordinates in the same matrix, matched for
magnitude, `k ≥ 20`. **Family size:** 154 checkpoints × 2 sizes × (1 super weight + 20
controls) — state it in the caption and use Benjamini–Hochberg.
*Answer looks like:* `W = 0.4 [0.2, 0.7] decades, n_seeds = 10, family = 6,468`, plus the
grid floor. Abrupt if `W` collapses to the floor in ≥8/10 seeds.
*Confounds:* (a) **Pythia may have no super weight at all** — run P0 below first;
(b) Pythia's cosine LR decays to 0.1× peak, so late `c(t)` movement is partly schedule
(TrainingDynamics-2025); (c) **Pythia's earliest checkpoints need fp32** — at random-init
scale, activations fall below fp16 range (Xu-2026); (d) 410M seeds 3 and 4 are the known
loss-spike outliers (PolyPythias §3) — **name them and report with and without**, do not
silently drop them.

### P0 — Gate: does any model ≤1B have a super weight? (must precede P1 and RQ4)

*Why open:* Yu et al. Table 2 covers Llama/Mistral/OLMo/Phi-3; **Pythia is absent**. Massive
activations exist at Pythia-14M but only under a **relaxed** definition
(Gallego-Feliciano: |a|>100 does not generalise down; top-1/median = 83 at 14M vs 3200 at 12B).
*Experiment:* run detection + single-scalar and small-SET ablation across
Pythia {70m, 160m, 410m, 1b} × 3 seeds, plus OLMo-2-0425-1B and SmolLM2-135M/1.7B, plus
**Ettin-Enc-150m vs Ettin-Dec-150m** for the encoder/decoder contrast, at final checkpoints.
*Answer looks like:* a table of `c(final)` per model, with the random-coordinate control
distribution beside it, and an explicit **size threshold** below which no single scalar and
no small SET produces `c > ` the control 99th percentile.
*This is the cheapest item here and it unblocks everything else.* If the answer is "no super
weight below ~1B", RQ4 as stated is dead and §5's P4 is the salvage.

### P2 — Is the coordinate the same across seeds, and stable once formed?

*Why open:* completely unmeasured (§3.3.2–3), and the two adjacent literatures **disagree**
about post-formation stability — Macocco-2025 finds outlier dimensions turn over (11/38
survive step 4k→143k); Xu-2026 finds sinks never drop out.
*Experiment:* the same P1 sweep, re-analysed. Two statistics: **(i)** across the 10
PolyPythia seeds at the final checkpoint, the distribution of `(layer, row, col)`; test row
and column agreement separately against a uniform-over-matrix null (Yu et al.'s
OLMo-1B `[1764, 1710]` vs OLMo-1B-0724 `[1764, 8041]` is the hint that **row may be
architecture-determined and column run-determined**). **(ii)** within a seed, Jaccard of the
top-`k` `down_proj` coordinates at formation vs final, to be compared directly against
Xu-2026's 0.29–0.33 and Macocco's 11/38. **Use the 160M `-data-seed*` / `-weight-seed*`
runs** to attribute any agreement to data order vs initialisation — the only public way.
*Answer looks like:* "the row index agrees in k/10 seeds (p = …, null = uniform over 3072
rows); the column does not", or the negation.

### P3 — Encoder vs decoder, at matched data and recipe

*Why open:* Ettin makes the objective a controlled variable and nobody has used it for this.
Gu-2025's mechanism (softmax normalisation forcing a key bias) is stated for causal LMs;
whether bidirectional MLM produces the same weight-level artifact is untested.
*Experiment:* Ettin encoder/decoder pairs at 68M/150M/400M/1B, all 236 checkpoints for the
150M pair. Same `c(t)` (with MLM pseudo-perplexity for the encoder — **and say so where the
number is used**, per the claim-hygiene rule about substituted metrics). n_seeds = 1, so this
is **exploratory** and must be labelled as such.
*Confound:* the first Ettin checkpoint is **8.5B tokens** — formation is already over.
This problem can answer *whether*, not *when*.

### P4 — From-scratch: can a super weight be prevented? (RQ4, budget-feasible)

*Why open:* every knob in §4.1 was shown to control sinks or kurtosis; **none was tested
against a super weight**, and Qiu-2025 + GradientSinks-2026 both show sinks and massive
activations dissociate — so the transfer is not automatic.
*Experiment:* Gu-2025's 60M LLaMA recipe (d=768, L=10, H=8, FFN 1536, 5B Pile tokens,
2048 ctx, 1M-token batch, 20k steps, LR 4e-4 cosine, AdamW wd 0.1), **3 seeds per arm**,
arms = {baseline, sigmoid-attention-without-normalisation, weight-decay ∈ {0, 0.1, 0.5},
LR ∈ {1e-4, 4e-4}, context ∈ {256, 2048}}. Checkpoint on a **log grid dense through
steps 1–2,000** (that is where Gu puts the event) and keep it dense past it — Pythia's
documented mistake was stopping the dense grid right before the window.
*Cost (inferred, §4.2):* ≈10–15 A100-h per 5B-token run ⇒ ≈**400–900 single-GPU hours** for
~12 arms × 3 seeds. Screen at 20% of budget (Liao-2024) before committing full runs.
*Answer looks like:* per arm, `c(final) [95% CI over 3 seeds]` against the random-coordinate
control, plus `W` per §5's criterion. "No effect" is **not** reportable — bound it.
*Confound:* at 60M the phenomenon may not exist (P0). Have the fallback ready: report
massive-activation ratio (Gallego-Feliciano's top-1/median, which *does* work at 14M) as a
secondary outcome so the sweep yields something either way.

### P5 — The MT bridge, honestly scoped

*Why open:* §2(d) — the multilingual and formation-trace instrument spaces are **disjoint**.
No trajectory study is possible; a final-checkpoint study is.
*Experiment:* locate super weights in EuroLLM-1.7B/9B, BLOOM-560m/1b1/7b1, Lucie-7B and
NLLB; ablate; measure **COMET and chrF++ per direction** across high- and low-resource pairs,
against the random-coordinate control. The question — *does removing one scalar cost
low-resource directions more?* — is unasked and connects to the lab's existing
uneven-PTQ-across-languages work. Use the **n≈960 / chat-template / COMET protocol already
in `compression/experiments/replication-uneven-ptq/`** rather than reinventing it.
*Optional trajectory add-on if budget allows:* Lucie-7B's 35 revisions from ≈20B tokens are
the densest genuinely multilingual grid that exists — enough for "is it stable once formed",
not for "when did it form".

### 5.1 Statistics floor (applies to all of the above)

- **Seeds are the unit of independence** — not checkpoints, not layers, not coordinates.
  A 154-point curve from one seed is n = 1.
- **Step 0 is a free null** and it is already in every Pythia/PolyPythia repo. Use it.
- **Multiplicity:** the family is checkpoints × coordinates × seeds × sizes. State the family
  size in every caption and correct (BH). Schaeffer-2023's ~10⁶-triplet argument is the
  cautionary case.
- **Report `effect [95% CI], n_seeds = N, family = M`.** Never a bare mean, never "no effect".
- **Do not trust a continuous metric to save you** — Zhao-2026 §3.3 found NLL *exposed*
  clusters accuracy hid. Report the per-seed distribution (histogram/KDE), not only its mean,
  and run **Hartigan's dip test** if it looks bimodal.
- **Free null for "is my bimodality real?"**: final training loss across the same seeds should
  be unimodal (Zhao Fig. 9, mean = median = 11.62). If loss is bimodal too, you are measuring
  run quality.
- **Name dropped points.** PolyPythias 410M seeds 3 and 4 will tempt you; say which and why.

### 5.2 Research engineering this needs (build before the first result, not after)

- **Provenance manifest per run**: `git_sha`, resolved config, library versions, seed,
  **and the exact Hub revision string** (`stepNNNNN` / `stageK-stepN-tokensNB`) plus the
  resolved commit SHA — revisions are mutable branches, SHAs are not.
- **One YAML per run**, no hidden defaults. Config ≠ what ran: log the values actually used.
- **A test that every committed config loads.**
- **Detector tests against a planted weight.** The single highest-value test here: take a
  small model, **write** a large value into a known `down_proj` coordinate, and assert the
  detector recovers exactly that `(layer, row, col)`. Add a negative control (uniform matrix
  ⇒ no detection) and a **precision test** (the detector must not silently pick a different
  coordinate at fp16 — Xu-2026 found Pythia's early checkpoints need fp32).
- **Checkpoint caching on the login node.** Compute nodes have no internet. 154 checkpoints ×
  10 seeds × 2 sizes is a large but finite prefetch; pin by commit SHA, verify by hash, and
  record the download date — Hub branch sets change.
- **A chronicle** (dated prose log) for any from-scratch run in P4, keyed to step numbers.
- **Log-spaced checkpointing that stays dense past the formation window** for P4 — the
  documented Pythia mistake is a dense grid that stops right before the event.

### 5.3 [unopened] — named but not read

Tigges et al. (circuit emergence in Pythia, cited by PolyPythias §3); Karamcheti et al. 2021
(10 GPT-2 runs × 600 checkpoints, cited by PolyPythias §2 — potentially a *second* multi-seed
trajectory suite); Madaan et al. 2024 (10 Llama-2-7B runs, 21 checkpoints, **stated to be
publicly unavailable**); Hu et al. 2023 (training maps / HMM method); Sun et al. 2024 Massive
Activations and An et al. 2025 (both are in `` and
`` but were not read for these notes); Fan et al. 2025
(AdamW steady-state √(η/λ_wd) scaling, the basis of Ding-2026's λ claim); Oh et al. 2024, *House of Cards: Massive Weights*
(large FFN weights driving massive activations — cited by SingleLayer-2026, the closest
citation to a weight-level cause; **the PDF is already at
`phenomenon/Oh-2024-House-of-Cards-Massive-Weights.pdf` and is the
single highest-priority unread item for this project**); Dettmers-2022 §on emergence with scale (in
``, not read here).
