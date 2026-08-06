# Replication brief — "The Uneven Impact of Post-Training Quantization in Machine Translation"

**For:** a separate agent/effort, independent of the Arctos phase-two (q6) work.
**Goal:** critically **replicate** the core findings of arXiv:2508.20893 before we
build a method on top of them.

> ⚠️ **This paper is a preprint (arXiv, Aug 2025) — not (as far as we can tell)
> peer-reviewed/published in a conference.** Treat every claim as *unverified*.
> The job is genuine replication: reproduce the pipeline independently, report
> what holds and what doesn't, and flag anything that looks cherry-picked,
> under-specified, or sensitive to undisclosed choices (calibration set,
> decoding params, metric version). Do **not** assume the paper is correct.

---

## 0. Paper

- Title: *The Uneven Impact of Post-Training Quantization in Machine Translation*
- arXiv: https://arxiv.org/abs/2508.20893  · PDF: https://arxiv.org/pdf/2508.20893
- Companion worth checking: *Quantization and Machine Translation: When Do LLMs
  Forget Languages?* (https://arxiv.org/abs/2508.20893 neighborhood — search it).

**Step 0 (do this first):** fetch and read the FULL paper and its appendix.
The summary below is reconstructed from the abstract + secondary sources and is
**incomplete**. Extract the exact experimental setup into a table before running
anything:

| To extract from the paper | Value (fill in) |
|---|---|
| Exact model list + sizes (1.7B…70B) | ? |
| All 55 languages + their resource tiers / scripts | ? |
| Quantizers + libraries + versions (AWQ, BitsAndBytes, GGUF, AutoRound) | ? |
| Bit-widths per method (4-bit, 2-bit, others?) | ? |
| Calibration data: generic set + "language-matched" set (source, size, seq len) | ? |
| Evaluation metric(s) + exact version (COMET? XCOMET? which checkpoint) | ? |
| Test set (FLORES-200? WMT? n sentences) + translation direction(s) | ? |
| Decoding (greedy/beam, max tokens, prompt/template) | ? |
| Whether GPTQ / LeanQuant were tested (we believe NOT — confirm) | ? |

---

## 1. Claims to replicate (rank by importance)

State each as **reproduced / partially / not reproduced**, with numbers + the
deviation from the paper.

- **C1 (headline).** 4-bit PTQ largely preserves quality for **high-resource**
  languages and **large** models.
- **C2.** **Low-resource / typologically-diverse** languages degrade most,
  **especially at 2-bit**.
- **C3 (the one we care about most).** **Language-matched calibration** helps
  **primarily in the low-bit (2-bit) regime**, and most for low-resource /
  divergent-script languages. *This is the load-bearing claim for our phase-two
  direction — replicate it carefully.*
- **C4.** GGUF variants are the **most consistent**, even at 2-bit.
- **C5.** Small models (~1.7B) can lose up to **~5 COMET** at 4-bit; 32B/70B lose
  **≤1**.

---

## 2. Scaled-but-faithful replication design

Full 55-lang × 5-model (up to 70B) is more than our budget. Reproduce the
**claim structure**, not the exact scale. Minimum viable replication:

- **Models (≥3, cached locally, see §3):** one small (`Qwen2.5-3B` or
  `bloom-3b` as the "small" proxy), one mid (`Llama-3.1-8B-Instruct`,
  `aya-expanse-8b`, or `EuroLLM-9B`), and ideally one larger (`aya-expanse-32b`
  is cached). If the paper's exact models are cached, prefer those.
- **Languages / pairs (cover the resource spectrum + scripts):** at least one
  high-resource same-script (e.g. en→de or cs→de), one high-resource
  cross-script (en→zh, Han), one **low-resource / divergent-script** (en→arz
  Egyptian Arabic, or another low-resource FLORES language the paper uses).
  Pull more low-resource languages if feasible — C2/C3 live there.
- **Quantizers (use the OFFICIAL libraries, not our from-scratch q6 code):**
  AWQ (`autoawq`), BitsAndBytes (4-bit NF4 + 8-bit), GGUF (`llama.cpp`
  Q4_K_M / Q2_K), AutoRound (`auto-round`). Install what's missing on the login
  node (has internet). The point is to reproduce THEIR pipeline faithfully.
- **Bit-widths:** 4-bit and 2-bit at minimum (the contrast that drives C1–C3);
  add 3-bit/8-bit if cheap.
- **Calibration (the C3 test):** for each method that takes calibration, run
  **(a) generic** (C4 / Wikipedia / FineWeb snippets — match the paper) vs
  **(b) language-matched** (monolingual text in the *target* language; FLORES or
  a small monolingual sample). Hold size/seq-len identical between (a) and (b).
- **Data:** FLORES+ (`openlanguagedata/flores_plus`, cached) for the dev/test
  translations. Use the same split the paper uses if stated.
- **Metric:** reproduce the paper's metric exactly (confirm in Step 0), AND
  additionally report **XCOMET-XL** (`Unbabel/XCOMET-XL`, cached) so results are
  comparable to the WMT25 compression task and to our q6 work.
- **Decoding:** match the paper (greedy unless stated); use each model's chat
  template if the paper does.

Keep n (sentences) modest but honest (e.g. 500–1000 FLORES sentences/pair);
report n. Run on SLURM A100 (see §3).

---

## 3. Environment (BYU RC cluster)

- **Login node has internet** (download libraries + any missing models/datasets
  here). **Compute nodes are offline** → pre-cache everything, then set
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  OPENSSL_CONF=/dev/null`.
- **GPU:** login-node T4 is compute-**Prohibited**; submit GPU work to SLURM
  (`--partition cs --qos cs --account sdrich --gres gpu:a100:1`). Validate small
  things on CPU first; never run heavy GPU work on the login node.
- **Cached models** (`~/.cache/huggingface/hub/`): aya-expanse-8b/32b,
  Llama-3.1-8B-Instruct, EuroLLM-9B-Instruct, bloom-560m/3b/7b1, Qwen2.5-3B,
  gemma-3-4b/12b, Tower{Base,Instruct,Plus}. Download any others on the login
  node.
- **Cached datasets:** `flores_plus`, `xnli` (generic calibration, multilingual),
  gsm8k. **COMET:** `Unbabel/wmt22-comet-da` and `Unbabel/XCOMET-XL` (+ its
  encoder `facebook/xlm-roberta-xl`) are cached.
- **Reference only (do NOT reuse for replication):** the Arctos q6 harness
  (`compression/experiments/q6-compression/`, `compression/src/interp/compress.py`) is our *own*
  from-scratch quantizers; the replication must use the paper's actual tools.
  But `compression/src/data/wmt.py` (FLORES loader), `compression/src/eval/metrics.py` (chrF++ + COMET +
  XCOMET-XL wiring), and `compression/src/models/_prompt.py` are fine to reuse.
- This is the user's CLAUDE.md "orchestrator" territory if you want autonomy:
  Python orchestrator + `claude -p` headless in tmux submitting sbatch jobs.

---

## 4. Deliverables

Write `compression/docs/replication_uneven_ptq_mt.md` containing:
1. The extracted setup table (Step 0) + every deviation you had to make and why.
2. A per-claim verdict table (C1–C5): reproduced / partial / not, with your
   numbers next to the paper's.
3. The **C3 deep-dive**: generic vs language-matched calibration, per method ×
   bit-width × resource tier — does the benefit really concentrate at 2-bit /
   low-resource? Plot it.
4. A "robustness / red flags" section: how sensitive are the headline numbers to
   choices the paper under-specifies (calibration size, decoding, metric
   version, prompt)? Anything that looks cherry-picked?
5. A one-paragraph bottom line: which claims are solid enough to build on, which
   aren't.

Save raw outputs under `compression/results/replication/` (gitignored).

---

## 5. Pitfalls / be-careful list

- **Metric mismatch is the #1 reproducibility trap.** COMET versions disagree;
  the WMT25 task uses XCOMET-XL (0–100-ish), Vicomtech-style papers use
  wmt22-comet-da (0–1). Report the scale and checkpoint explicitly.
- **GGUF quantization** runs through llama.cpp, not HF — different tokenizer /
  generation path; control for that before comparing to AWQ/BnB.
- **"Language-matched calibration"** is under-specified in many papers — pin
  down exactly what text, how much, what sequence length, and keep it identical
  to the generic arm except for the language/domain.
- **Low-resource COMET is noisy and may be untrustworthy** (COMET is trained
  mostly on high-resource pairs) — corroborate en-arz-type results with chrF++
  and spot-check actual translations.
- **Small n** inflates variance — the effect sizes in C3 may be within noise;
  report confidence/spread, not just means.
- Keep this effort **separate** from Arctos q6 so the replication stays an
  independent check, not a self-confirmation.

---

## 6. Why we want this (context, do not let it bias the replication)

Our phase-two (q6) work plans to *extend* this paper — MT-conditional GPTQ/
LeanQuant (quantizers it didn't test), super-weight + salient-channel
protection, and a mechanistic explanation — specifically to rescue the
low-resource / 2-bit collapse (en-arz). All of that rests on **C2 and C3 being
real.** So replicate them honestly: if C3 doesn't hold, we need to know *before*
building on it. A negative replication is a valuable result here.
