# Math plan — from "basic" to research-functional in ~6 months

Goal: be able to **read, understand, and manipulate the math behind quantization
/ compression research**, and re-derive its core results. NOT to become a
mathematician — to reach the *functional* bar for proposing novel ideas.

## Principles (read once)
- **Just-in-time, pull not push.** Learn each piece *because a paper/experiment
  needs it*, not by grinding a textbook front-to-back.
- **Code-first.** Every concept → do it in `numpy`; check derivatives with
  `sympy` / `torch.autograd`; plot it. Math-by-experiment plays to your strength.
- **Conceptual > mechanical.** You already understand ideas; you don't need
  exam-speed symbol pushing. Slow, with code, with lookups, is real research math.
- **~30–45 min/day** beats weekend cramming. Consistency compounds.

## Core resources
- **3Blue1Brown** — *Essence of Linear Algebra* + *Essence of Calculus* (YouTube).
  Intuition/pictures — the thing exam courses skipped.
- **"Mathematics for Machine Learning"** (Deisenroth, Faisal, Ong) — free PDF at
  mml-book.github.io. Chapters 2 (linear algebra), 5 (calculus/gradients),
  6 (probability). Your reference, not a cover-to-cover read.
- **Papers as the pull:** GPTQ (2210.17323), Optimal Brain Surgeon/Damage,
  super-weights (2411.07191), QuIP (2402.04396), AWQ (2306.00978).

## The whole toolkit (this is the finish line — it's small)
Linear algebra: matrix = linear map, matmul, transpose, norms (L1/L2/∞), inner &
outer products, rank, eigen/SVD, symmetric positive-definite matrices, **quadratic
forms xᵀAx**, orthogonal matrices = rotations, projection. · Calculus: derivative,
partial derivative, **gradient**, **chain rule** (=backprop), **2nd-order Taylor
ΔL ≈ gᵀΔw + ½Δwᵀ H Δw**, optimum (g=0, H≽0). · Probability: expectation, variance,
Gaussian vs heavy-tailed, quantiles/outliers, sampling; bootstrap, confidence
intervals, significance tests.

---

## Phase 0 — intuition (weeks 1–3)
Build the pictures. ~30 min/day.
- Watch 3B1B *Essence of Linear Algebra* (all). After each video, reproduce it in
  numpy: build a matrix, apply it to vectors, see the transformation; compute a
  dot product, a norm, an outer product.
- Watch 3B1B *Essence of Calculus* (through the chain rule).
- **Rep:** in numpy, take a 2×2 matrix, compute its eigenvectors, and *show* it
  stretches along them. Plot a quadratic form xᵀAx as a surface.
- **You're ready when:** you can explain, in words + a picture, what an
  eigenvector, a norm, and a gradient *are*.

## Phase 1 — the quantization core (weeks 4–8)
The single most valuable stretch. Tie everything to GPTQ/super-weights.
- **Gradients & Taylor:** derive the 2nd-order Taylor expansion of a loss in the
  weights by hand; check each term with `torch.autograd`. Understand why at a
  trained min the gradient term vanishes → ΔL ≈ ½Δwᵀ H Δw.
- **Quadratic forms & the Hessian:** what xᵀHx means; why H being
  positive-definite means "bowl-shaped"; diagonal vs full H.
- **Pull-paper: GPTQ + Optimal Brain Surgeon.** Read with the goal of seeing
  ΔL ≈ ½Δwᵀ H Δw *inside the method*. Re-derive the OBS single-weight update in
  numpy on a toy linear layer; confirm it beats round-to-nearest at fixed bits
  (you already have this experiment in `compression/src/interp/compress.py::gptq_quantize`).
- **Probability/outliers:** distributions, variance, quantiles; why a few
  heavy-tailed weights/activations (super-weights) dominate — read the
  super-weight paper with the stats in hand.
- **You're ready when:** you can read GPTQ's equations and explain each symbol,
  and re-derive the core update on paper + numpy.

## Phase 2 — evaluation statistics + more geometry (weeks 9–16)
Aimed at hardening the GPTQ-MT result (the roadmap's phase 1) — learn the stats
by *using* them on real numbers.
- **Stats for ML eval:** expectation/variance of a metric, sampling error, the
  **paired bootstrap** for COMET deltas, confidence intervals, significance
  (why "method A > B by 0.3 COMET" needs a test). Implement a bootstrap CI in
  numpy on the q6 results. This is your "settle stats/probability" goal, done.
- **SVD, rotations, incoherence:** read **QuIP** for the cleanest "math idea from
  understanding" — incoherence = a rotation makes the max matrix entry small.
  Reproduce: rotate a matrix with a random orthogonal Q, watch the max-abs entry
  drop. That's the whole insight, in ~10 lines of numpy.
- **You're ready when:** you can put a confidence interval on a result and say
  whether a difference is real; and you can explain why rotation helps quant.

## Phase 3 — apply, chase an anomaly, propose (weeks 17–26)
Math now serves your own questions.
- **Re-derive one full paper** end-to-end (GPTQ or QuIP) — every equation,
  reproduced in code. This is the threshold to "I can do this."
- **Chase an anomaly** from the project (why is generic-GPTQ worse than nothing?
  why does Gemma have no super-weights?) using the tools above. The novel idea
  lives here, at *your* intersection (mechanism × compression).
- **Write** the math up in your own words (a short note). Explaining is the test
  of understanding.

## Milestones / self-checks
- End of month 1: explain eigenvector/gradient/norm with pictures + numpy.
- End of month 2: read GPTQ's equations; re-derive OBS update in numpy.
- End of month 4: bootstrap CI on a real result; explain QuIP's rotation idea.
- End of month 6: re-derived one paper fully; written a short math note on an
  anomaly.

## How it plugs into the project
Each math skill is used on a real experiment (so it sticks):
quadratic form/Hessian → GPTQ/`compress.py`; outliers/quantiles → super-weights
& AWQ salience; bootstrap/CI → hardening GPTQ-MT (`phase2-results.md`); SVD/
rotation → trying QuaRot/QuIP as a base quantizer (roadmap). See `ROADMAP.md`.

*(Calibration: 6 months of this gets you research-functional, not a
mathematician — and that's the bar. Maturity keeps compounding for years; the
program deadline is not the deadline for the skill.)*
