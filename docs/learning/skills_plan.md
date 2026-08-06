# Skills Plan — Summer 2026 → Final Semester

Goal: MLE offers by spring 2027. The thesis/ToAll work covers differentiation;
this plan covers the generic gates that decide whether the differentiated profile ever gets seen.

**Calendar anchor: new-grad 2027 applications open August–September 2026. Apply before feeling ready.**

---

## Tier 1 — Interview gates (non-negotiable, start now)

### Coding / DSA
- [ ] Cadence: 3–4 problems/week, every week through fall (consistency > volume)
- [ ] Patterns to cover: arrays/hashmaps, two pointers, sliding window, BFS/DFS,
      heaps, binary search, intervals, basic DP
- [ ] **Bar:** solve a random medium in ~25 min while explaining out loud
- [ ] From September: 1 timed mock/week (peer or platform), out-loud always

### Classic ML fundamentals (review — rust removal, not new learning)
- [ ] Bias/variance, over/underfitting, regularization (L1/L2, dropout, early stopping)
- [ ] Metrics: precision/recall/F1/AUC, when accuracy lies, class imbalance
- [ ] Cross-validation, data leakage, train/serve skew
- [ ] Optimization: SGD/Adam, learning-rate schedules, why training diverges
- [ ] Transformers/attention mechanics — explainable on a whiteboard, no warm-up
- [ ] Embeddings: how trained, cosine vs dot product, failure modes (thesis = depth here)
- [ ] **Bar:** any topic above, 3-minute whiteboard explanation, cold

### ML system design
- [ ] Master the loop: data → features/model → serving → monitoring → feedback/retraining
- [ ] Prepare 4 narrated designs: (1) translation-quality monitor (= thesis, lived),
      (2) recommendation feed, (3) spam/abuse classifier, (4) search ranking
- [ ] **Bar:** 35-minute structured walkthrough of an unfamiliar domain without stalling

---

## Tier 2 — Production toolkit (build through the thesis, deliberately)

- [ ] **PyTorch beyond forward():** non-bottlenecking DataLoaders, mixed precision,
      GPU profiling, OOM debugging → exercised by the Phase 1 census pipeline
- [ ] **Docker + FastAPI:** QE tool as a containerized service a stranger could run
- [ ] **Experiment tracking (W&B or MLflow)** from day one of Phase 1
      → bar: "I can reproduce any number in my thesis from a logged run"
- [ ] **SQL:** Phase 1 results in a real database; query in SQL, not pandas
      → bar: joins, GROUP BY aggregations, window functions without googling
- [ ] **AWS legibility:** diagram ToAll's architecture from memory; justify each choice
- [ ] **Monitoring:** score-distribution dashboards + drift alerting on the QE service

## Tier 3 — Differentiators (small time, high leverage)

- [ ] Project pitch: 90-second and 5-minute spoken versions, tested on a non-ML listener
- [ ] Behavioral story log: weekly note of what broke / what I did / what changed
      (mine it in spring for interview stories — memory alone produces mush)
- [ ] Present thesis work to the African-NLP / Masakhane community (network = referrals)

## Explicitly NOT this year

Kubernetes depth · Rust · second ML framework · more interp theory · new coursework-shaped learning.
Real skills; none gate the next 12 months. Every hour there is an hour off Tier 1.

---

## Weekly template (≈8–10 hrs alongside project/work)

| Day | Block |
|---|---|
| Mon/Wed/Fri | 1 DSA problem each (timed, out loud) |
| Tue | ML fundamentals review (1 topic, whiteboard test) |
| Thu | SQL or system-design rep (alternate weeks) |
| Sat | 2-hr deep block: mock interview (fall) or Tier 2 skill (summer) |
| Sun | Off. Protected. |

## Season map (December 2026 graduation — compressed)

- **Now–mid-Aug:** DSA cadence up; ML review pass 1; Phase 0 coverage audit; résumé final;
  (bounded break if taken: before mid-Aug)
- **Aug 1–Sep 15:** applications out HARD — big tech + AI-adjacent + mid-tier, wide net;
  target "Dec 2026 grads welcome" postings explicitly; referrals hunted
- **Sep–Nov:** interview loops + thesis Phases 1–3 in parallel; system-design reps weekly;
  offers typically land Nov–Dec
- **Oct:** file OPT paperwork (can file up to 90 days before program end — confirm dates with ISS)
- **Dec:** graduate; decision point — external MLE offer vs. staying SWE with negotiated ML scope
- **Fallback is not failure:** stay SWE, own the ML surfaces (QE service, matcher seam,
  MatrixLab providers), jump as experienced MLE in 12–18 months

## Visa checklist (verify everything with ISS / immigration attorney — not legal advice)

- [ ] OPT filing window and 90-day unemployment clock after OPT start date
- [ ] STEM OPT extension eligibility for the degree program
- [ ] **TN status (USMCA)** — as a Mexican citizen, a major structural advantage:
      no lottery, fast processing; check degree/job-title fit (Engineer / Computer Systems Analyst)
- [ ] If staying university-affiliated: H-1B **cap-exempt** employer question
- [ ] Any international travel: visa stamp validity + re-entry before booking
