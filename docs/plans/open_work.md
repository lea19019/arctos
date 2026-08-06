# What Adrian Could Work On Next — Prioritized Brief (Arctos)

> Generated 2026-07-07 by mining the future-work / limitations / open-problem
> sections of **95 cited papers** (learning/reading_list.md + research.md + the NLLB/XTTS
> findings docs), cross-referenced against what Arctos has already **done** or
> **ruled out**. Method: a fan-out workflow — one agent per paper (fetched the
> arXiv abstract + Conclusion/Limitations), then 6 per-track synthesis agents,
> then a merge. Novelty labels are honest against Q5/Q6.

## Executive summary

The open space sits at the intersection of four clusters, and the through-line is that Arctos's two headline wins (MT-conditional calibration; causal-KL super-weight FP16 preservation) plus its signature negative (Q5: importance ⟂ quant-sensitivity) have been established **only on decoder-only weight quantization** — leaving every other axis genuinely open. In **MT quant**, the sharpest opportunity is **rotation/incoherence PTQ**: the single biggest lever never pulled, orthogonal to the Q5 null, and the natural gateway to W4A4KV4 activation+KV compression the deployment actually needs. In **pruning/distill**, the sharpest move is porting the calibration/healing-data linchpin from quant to pruning masks and to an **on-policy KD healing stage** that attacks the "no healing-free PTQ hits FP16 at 3-bit" negative head-on. In **NLLB enc-dec**, the sharpest is **sink-aware, causal-KL-validated mixed precision** on the exact deployment model — the mostly-unbuilt cross-attention-sink / language-tag frontier. In **TTS/XTTS**, the sharpest is the **per-stage low-bit sensitivity Pareto** (GPT-AR vs VQ-VAE vs HiFi-GAN) — greenfield, cheap, and the map every other XTTS decision depends on. Overarching all of it, the thesis-flagship is turning Q5 from a negative into a positive by **validating a sensitivity-native bit-allocation signal** against ground-truth MT degradation.

## TOP 8 things to work on (ranked across all tracks)

| Rank | Idea | Track | Novelty | Effort | Key papers | Why now |
|---|---|---|---|---|---|---|
| 1 | Sensitivity-native bit-allocation signal bake-off (causal-KL vs Wanda vs Shapley vs recon-error vs Fisher) validated against ground-truth COMET drop | interp-bridge / mt-quant | Novel | Medium | 2509.15455, 2410.13056, 2411.07191, 2306.11695, 1911.03852 | Turns Q5's negative into the load-bearing positive; tools already in src/interp/; unlocks principled mixed precision the whole roadmap needs |
| 2 | Sink-aware + super-weight FP16 mixed precision for NLLB cross-attention, causal-KL-validated | nllb-encdec | Novel | Medium | 2605.01229, 2605.16901, 2411.07191, 2402.17762, 2508.20893 | The named-but-unbuilt frontier on the exact deployment model; converts two decoder-only wins to enc-dec while guarding the Q5 trap |
| 3 | Per-stage low-bit sensitivity Pareto for XTTS (GPT-AR / VQ-VAE / HiFi-GAN): where does the CER cliff live? | tts-xtts | Partially-explored | Medium | 2406.04904, 2509.20802, 2508.20893 | Foundational map that every XTTS bit-budget decision depends on; eval stack + 3-lang scaffold already exist; greenfield in the literature |
| 4 | Rotation-based PTQ for MT (QuaRot/SpinQuant/QuIP), MT-conditional rotation objective, and the W4A4KV4 unlock | mt-quant | Novel | Large | 2404.00456, 2405.16406, 2307.13304, 2402.04396, 2508.20893 | Biggest untried method-family; sidesteps Q5 by removing outliers; natively enables the activation+KV memory axis the T4 budget requires |
| 5 | On-policy KD (GKD/DistiLLM) healing of the 3-bit MT cliff from an FP16 self-teacher | distill-recovery | Novel | Medium | 2306.13649, 2402.03898, 2306.08543, 2210.17323 | Directly attacks Arctos's headline negative (no healing-free PTQ reaches FP16 at 3-bit); GKD validated on WMT but never as post-quant recovery |
| 6 | Transfer causal-KL super-weight / salient-channel FP16 preservation to the XTTS GPT-2 core | tts-xtts | Novel | Medium | 2411.07191, 2402.17762, 2306.00978, 2406.04904 | Lowest-effort high-value: reuses best text tooling nearly verbatim; tests whether the super-weight phenomenon is modality-universal |
| 7 | MT-conditional pruning masks: port the calibration linchpin to Wanda/SparseGPT | pruning-structural | Partially-explored | Small | 2306.11695, 2301.00774, 2311.09755, 2508.20893, 2601.18306 | Cheap analog of Arctos's biggest win; literature says pruning is 2–4× MORE calibration-sensitive than quant, so payoff likely larger |
| 8 | Rigorous statistical + Pareto-frontier evaluation protocol (bootstrap CIs, paired significance, metric-swap reordering, human spot-check) | interp-bridge / rigor | Partially-explored | Medium | 2508.20893, 2407.03211, 1911.03852, 2305.14314 | The #1 rigor item in the roadmap and most-cited open problem in the digest; retroactively hardens every other track's claims into a thesis |

---

## Track: mt-quant

- **Rotation-based PTQ for MT** — QuaRot/SpinQuant/QuIP-style incoherence on the 6 decoder-only models; test whether rotation moves the confirmed 3-bit cliff and whether an MT-conditional rotation objective helps LRL; natively enables W4A4KV4. Papers: 2404.00456, 2405.16406, 2307.13304, 2402.04396, 2406.11235, 2502.15779. *Novel / large.*
- **MT-conditional vector/codebook quantization (GPTVQ, AQLM, QuIP#)** — fit codebooks on MT-parallel Hessians as a better quantizer (not a fragility localizer) to cross the sub-3-bit frontier. Papers: 2402.15319, 2401.06118, 2402.04396, 2406.11235. *Novel / large.*
- **Sensitivity-native bit-allocation signal bake-off** — causal-KL vs Wanda vs Shapley (CoopQ) vs reconstruction-error (CMPQ), scored by predicted vs measured COMET drop, then driving channel-wise fractional allocation. Papers: 2509.15455, 2410.13056, 2411.07191, 2306.11695. *Partially-explored / medium.* (Shared flagship with interp track.)
- **Error-driven MT calibration-sample selection** — greedily pick parallel sentences that minimize GPTQ error / causal-KL vs random MT sets and self-generated calibration. Papers: 2601.18306, 2410.17170, 2311.09755, 2405.20835. *Novel / medium.*
- **Non-uniform MT-conditional quantization grids (LeanQuant, SqueezeLLM)** — learned k-means / inverse-Hessian LUT grids (distinct from the ruled-out Fisher bit-split) to bridge the 3-bit cliff. Papers: 2407.10032, 2306.07629, 2308.13137. *Partially-explored / medium.*
- **Joint W+A+KV quantization under the 16GB/T4 budget** — QuaRot + KIVI/KVQuant to push W4A4 + 2–4-bit KV; benchmark MXFP4/NVFP4 vs INT for LRL. Papers: 2508.20893, 2402.02750, 2401.18079, 2404.00456. *Partially-explored / large.*
- **Spread vs preserve: does rotation compose with super-weight FP16 preservation?** — all four cells of ±rotation × ±causal-KL FP16 islands at 2–3-bit. Papers: 2402.04396, 2404.00456, 2307.13304, 2411.07191. *Novel / medium.*

## Track: pruning-structural

- **MT-conditional pruning masks (Wanda/SparseGPT)** — build masks from MT-parallel vs generic data at 25–50% and 2:4 sparsity; measure calibration-source dispersion. Papers: 2306.11695, 2301.00774, 2311.09755, 2508.20893, 2601.18306. *Partially-explored / small.*
- **Is MT structural pruning depth-anisotropic?** — COMET-native block-drop curve vs phase-one depth stages; pit reverse-order "drop final 25%" against emit-late layers. Papers: 2403.03853, 2403.17887, 2402.02834, 2411.15558, 2306.11695. *Novel / medium.*
- **NASH-style asymmetric depth+width pruning of NLLB-200** — prune decoder depth, keep encoder width; interp-guided non-uniform budget beating AfriNLLB's greedy pruning. Papers: 2310.10054, 2602.09373, 2207.04672, 2310.03686, 2212.09811. *Novel / large.*
- **Structured pruning + MT-parallel healing** — test the healing-corpus linchpin (MT-parallel vs generic vs language-matched) to recover LRL quality depth-pruning destroys. Papers: 2411.15558, 2310.06694, 2305.18403, 2403.19135, 2507.21568. *Partially-explored / medium.*
- **SliceGPT width/embedding-dim pruning for MT** — does the PCA eigenvalue spectrum predict slice sensitivity, and does cutting shared width hurt LRLs? Papers: 2401.15024, 2311.09755, 2508.20893. *Partially-explored / medium.*
- **Attribution-guided structured pruning with MT reference sets** — LRP/EAP-IG head-relevance vs Wanda/causal-KL as pruning criteria (signal only, not bit allocation). Papers: 2506.13727, 2403.17806, 2310.10348, 2210.05709. *Novel / large.*
- **Moderate-compression prune × quant Pareto frontier** — structured pruning (10–40%) × 8/4/3-bit MT-conditional GPTQ, both orderings, with full rigor. Papers: 2301.00774, 2403.03853, 2402.02834, 2508.20893. *Partially-explored / large (depends on the above landing first).*

## Track: distill-recovery

- **On-policy KD healing of the 3-bit MT cliff (GKD/DistiLLM/MiniLLM)** — student samples, FP16 self-teacher scores, skew/reverse-KL on MT prompts. Papers: 2306.13649, 2402.03898, 2306.08543, 2210.17323. *Novel / medium.*
- **Resolve the LoRA-FT healing paradox on Llama-3.1** — QLoRA vs LoftQ vs partial-FT diagnostic (LLaMA3 study found LoRA healing can EXACERBATE degradation). Papers: 2404.14047, 2305.14314, 2310.08659, 2411.15558. *Novel / small — cheap de-risking gate for the whole track.*
- **QAT-lite with MT-conditional calibration (EfficientQAT Block-AP + BitDistiller CAKLD)** — MT-conditioned Block-AP and self-distill QAT to cross the 2-bit cliff. Papers: 2407.11062, 2402.10631, 2601.18306, 2508.20893. *Novel / medium.*
- **Interp-driven healing of off-target / language-confusion collapse** — use logit-lens/pivot/DLA to localize where quant damages target-language emission, then a language-tag-conditioned healing objective. Papers: 2407.03211, 2309.11674, 2401.08417, 2402.10588. *Novel / large.*
- **Causal-KL-guided FP16 split for low-rank healing (super-weight-aware LoftQ/PiSSA/DoRA)** — replace SVD-energy heuristics with causal-KL super-weight + AWQ salient-channel selection. Papers: 2310.08659, 2404.02948, 2402.09353, 2411.07191. *Novel / medium.*
- **Distill-heal a compressed NLLB-200 with an LRL-protecting heal mixture** — prune + low-bit quant then GKD/seq-KD, over-weighting divergent-script directions. Papers: 2207.04672, 2602.09373, 2310.06694, 2507.21568, 2306.13649. *Partially-explored / large.*
- **Joint prune→quant→heal Pareto cell** — sweep order-of-operations and healing corpus (MT-parallel vs generic). Papers: 2301.00774, 2306.11695, 2411.15558, 2508.20893, 2305.14314. *Partially-explored / large.*

## Track: nllb-encdec

- **Sink-aware, causal-KL-validated mixed precision for NLLB cross-attention** — re-rank the EOS/language-tag/punctuation sink weights (83–91% of cross-attn mass) by causal-KL, keep only the sensitive subset FP16 at 3-bit. Papers: 2605.01229, 2605.16901, 2411.07191, 2402.17762, 2508.20893. *Novel / medium.*
- **Localize NLLB super-weights / massive activations on language-tag/EOS embeddings** — test the enc-dec analogue of decoder-only early-layer super-weights on special tokens. Papers: 2411.07191, 2402.17762, 2605.01229, 2208.07339. *Novel / medium.*
- **DecoderLens depth map of NLLB-3.3B + direct enc-dec test of the depth/quant-fragility null** — either reproduce the Q5 stage-level null on enc-dec (publishable negative) or find a real LRL depth gain. Papers: 2310.03686, 2402.10588, 2603.02258, 2508.20893. *Partially-explored / medium.*
- **2D encoder × decoder structured pruning with interp-guided selection + MT-parallel healing** — NASH asymmetry with DecoderLens/causal-KL selection, healed by language-matched distillation, swept to a Pareto frontier. Papers: 2310.10054, 2602.09373, 2212.09811, 2507.21568, 2207.04672. *Partially-explored / large.*
- **Per-head cross-attention decomposition to protect language-specific heads** — causal/Shapley attribution to find language-specific heads, keep them at higher precision. Papers: 2603.02258, 2210.05709, 2402.16438, 2411.08745. *Partially-explored / medium.*
- **Attention-level reconstruction PTQ (aespa / CAR-SAM) with MT-conditional calibration** — port attention-wise reconstruction to NLLB cross-attention; check the dissipation/oscillation failure modes. Papers: 2402.08958, 2605.16901, 2508.20893, 2601.18306. *Novel / medium.*
- **Cross-attention KV-cache quantization with a sink-token carve-out** — 2–4-bit cross-attn KV keeping sink-token entries higher-precision (fills KVQuant/KIVI's explicit enc-dec gap). Papers: 2401.18079, 2402.02750, 2605.01229. *Partially-explored / medium (ranks below weight-side ideas; NLLB's short sequences make KV a smaller memory share).*

## Track: tts-xtts

- **Per-stage low-bit sensitivity Pareto (GPT-AR / VQ-VAE / HiFi-GAN)** — sweep each stage across bit-widths measuring CER/UTMOS/RTF; find where the CER cliff lives. Papers: 2406.04904, 2509.20802, 2508.20893. *Partially-explored / medium — the foundational entry point.*
- **Transfer causal-KL super-weight / salient-channel FP16 preservation to the XTTS GPT-2 core** — re-derive the ranking against CER/UTMOS; test if super-weights are modality-universal. Papers: 2411.07191, 2402.17762, 2306.00978, 2406.04904. *Novel / medium — lowest-effort high-value.*
- **SPADE-style CER-native structured pruning + KD ported to XTTS, with LRL eval** — WLI layer-importance recomputed on the GPT-2 core, depth pruning + multi-level KD, evaluated on Swahili + other LRLs. Papers: 2509.20802, 2406.04904, 2306.13649. *Novel / large.*
- **Usage-frequency / causal-ranked mixed-precision VQ-VAE codebook** — question the "codebook always FP16" assumption; protect rare/high-impact codes, quantize the bulk; map rare-code corruption to LRL/prosody failures. Papers: 2406.04904, 2411.07191, 2402.17762. *Novel / medium — most TTS-distinctive.*
- **KV-cache quantization for the XTTS AR decoder** — KIVI/KVQuant on the ~21.5 Hz AR core; CER/UTMOS at 4/2-bit KV + wall-clock RTF on T4. Papers: 2402.02750, 2401.18079, 2406.04904. *Novel / medium.*
- **On-policy / self-distillation healing of a compressed XTTS student (GKD-for-audio)** — GKD explicitly names audio as future work; heal on self-generated speech. Papers: 2306.13649, 2402.10631, 2509.20802. *Novel / large.*
- **Joint prune × quant × distill Pareto frontier for XTTS in 16GB** — compose the winners into the actual deployment deliverable with statistical rigor. Papers: 2509.20802, 2406.04904, 2508.20893. *Novel / large — the capstone.*

## Track: interp-bridge-and-rigor

- **Validate a sensitivity-native bit-allocation signal (thesis flagship)** — 5-signal bake-off against ground-truth MT degradation; find the signal with Spearman ρ ≫ 0 where importance gave ~0. Papers: 2509.15455, 1911.03852, 2306.07629, 2410.13056, 2407.10032, 2411.07191. *Novel / medium.*
- **Interp-driven, sensitivity-validated mixed precision for NLLB enc-dec** — DecoderLens + sink filtering + universal-geometry probes, each candidate causal-KL-validated; explicitly test whether the decoder-only Q5 null replicates or BREAKS on cross-attention. Papers: 2605.01229, 2603.02258, 2310.03686, 2605.16901, 2402.17762, 2411.07191, 2207.04672. *Novel / large.*
- **Rigorous statistical + Pareto-frontier evaluation protocol** — bootstrap CIs, approximate-randomization paired significance, multi-seed variance, chat-template + decoding sweeps, metric-swap reordering; show at least one prior "win" loses significance. Papers: 2508.20893, 2407.03211, 1911.03852, 2305.14314, 2401.08417. *Partially-explored / medium.*
- **Does Q5/depth-staging transfer to the pruning and enc-dec axes?** — (a) importance vs pruning-sensitivity correlation; (b) reverse-order pruning vs the emit-late target-emission stage. Papers: 2411.15558, 2403.17887, 2403.03853, 2402.02834, 2306.11695. *Novel / small — high insight-per-GPU-hour.*
- **End-to-end T4/16GB NLLB+XTTS serving stack** — realized latency/memory, KV-cache quant, and the super-weight-kernel vs uniform-GPTQ/rotation wall-clock tension on Turing hardware. Papers: 2402.02750, 2401.18079, 2404.00456, 2405.16406, 2306.00978, 2210.17323. *Novel / large.*
- **XTTS per-stage interp-driven bit allocation (GPT-2 vs HiFiGAN vs VQ-VAE), LRL-prosody-preserving** — per-stage sensitivity map + first check of whether the text 3-bit cliff and Q5 null hold for a TTS stack. Papers: 2406.04904, 2509.20802, 2411.07191, 2306.07629. *Novel / large.*

---

## Do NOT redo (already done or ruled out)

- **MT-conditional / language-matched GPTQ calibration** — DONE (Q6: recovers 3-bit cliff on all 6 models, +0.13–0.52 COMET; generic calibration worse than no quant). Use as baseline, not discovery. Also settled: "translation as a downstream calibration task" (2601.18306's headline) and "automatic metrics understate low-bit damage (~10×)."
- **Importance/Fisher/Hessian-trace-driven mixed-precision bit allocation** — RULED OUT. Q5 null reconfirmed across 2 metrics (ρ~0); Fisher-diagonal 2/4 split < uniform. Never wire raw importance (Block Influence, Taylor, IFR, tuned-lens CBE, EAP/EAP-IG, HAWQ-V2 trace, CoopQ Shapley) to bit allocation. Legit only as PRUNING criteria, or after re-validation as sensitivity-native.
- **Depth/pipeline-aware BIT allocation** — RULED OUT (wash for quantization at stage level). Protecting language-specific endpoints vs crushing the neutral middle does not work for precision. (The pruning layer-drop analog IS open — layer removal is a distinct operation.)
- **No healing-free PTQ reaches FP16 at 3-bit** — established negative; do not re-run healing-free 3-bit PTQ discovery. Open frontier is WITH-healing at 2-bit and on enc-dec/TTS.
- **Trivial FP16-vs-INT8 XTTS GPT baseline / straight SPADE replication on codec-LMs** — already exists / already published. Novelty starts one step past both (sub-INT8 with sensitivity-native protection, LRL eval, or the non-LM stages).

## If I were you, I'd start here

Start with a cheap, tool-reusing triad that de-risks the thesis and the dubbing deployment at the same time: run the **sensitivity-native signal bake-off** (interp #1 / mt-quant #3) to convert Q5's negative into the project's flagship positive, and in parallel spend the low-cost A100 time on the **XTTS per-stage sensitivity Pareto** (xtts #1) and **NLLB super-weight/sink localization** (nllb #1–2) — all three lean on `super_weights.py`/`salient_channels.py`/the existing eval stacks, require no training, and each answers a load-bearing question the 16GB/T4 deliverable can't move without (which signal to allocate bits by, which XTTS stage is the bottleneck, and which NLLB weights must stay FP16). The moment those three land, they compose directly into the deployment path: the validated signal drives principled mixed precision on both NLLB and the XTTS GPT core, the stage map tells you where to spend the bit budget, and you graduate to the two larger, higher-payoff levers the T4 budget genuinely needs — **rotation-based W4A4KV4** (mt-quant #1) for the activation+KV memory that weight-only quant leaves untouched, and **on-policy KD healing** (distill #1) to reclaim the sub-3-bit quality PTQ alone cannot. Wrap everything in the **rigor protocol** (interp #3) from day one so every Pareto point is thesis-defensible rather than a single-run point estimate — that is what turns a pile of deployment wins into the "sweet spot of compression for translation" roadmap.
