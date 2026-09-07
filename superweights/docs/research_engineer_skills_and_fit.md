# Research Engineer: what the job is, whether you would like it, and what a 698R project can prove

**Written 2026-09-07, before choosing the project.** Purpose: decide what the project
is *for* before deciding what it is *about*. Part 1 describes the role next to its
neighbours. Part 2 is the skill map. Part 3 is a self-assessment to fill in
directly in this file. Part 4 maps the skills onto what a 125–150 hour project can
and cannot demonstrate, and what the proposal form's "what will be learned" line
should say. Part 5 is the job-market timing that constrains all of it.

Where a statement rests on current job postings, it is marked `[postings]` and
sourced in §1.4. Everything else is my synthesis; treat it as a well-informed
opinion, not a fact.

---

## Part 1 — The role, next to its neighbours

### 1.1 Four jobs that get confused with each other

Titles are inconsistent across employers, so read the *day-to-day* in a posting,
never the title. But the underlying jobs are distinct:

| | Research Scientist | **Research Engineer** | ML Engineer (product) | ML infra / platform |
|---|---|---|---|---|
| Owns | the *question* | the *experiment* | the *model in production* | the *system that trains or serves* |
| Typical output | paper, research direction | a running experiment, a trustworthy result, a reusable tool, co-authorship | deployed model, pipeline, dashboard | framework, cluster tooling, throughput |
| Judged on | novelty and correctness of findings | how fast **and how trustworthily** questions get answered | reliability, latency, a business metric | scale, efficiency, uptime |
| Code lifetime | days | days to months; some tools live for years | years | years |
| Failure mode of the job | wrong question | wrong answer that looks right | outage, drift | slow, expensive |
| Credential norm | PhD usual | PhD *not* required at most labs; evidence required | BS/MS | BS/MS |

Your description of MLE, "SWE but dealing with the infrastructure and maintenance
of ML models", is right for the product-company version. At research
organisations "MLE" and "Research Engineer" often mean the same job, and at frontier
labs everyone is "Member of Technical Staff" and the RS/RE line is deliberately
blurred. So the question is not which title, but which *column* above you want to
be judged on.

### 1.2 What the Research Engineer actually does

The one-sentence version: **a Research Engineer turns a half-formed question into a
running experiment, makes sure the answer can be trusted, and communicates it so
the next question can be chosen.** The engineering is in service of the answer,
not the other way round.

Three things follow from that sentence, and they are what make the job different
from SWE:

1. **Most of what you build is thrown away.** The experiment code that answers the
   question has done its job. What survives is the *result* and occasionally a tool.
   People who need to see their code used for years find this demoralising.
2. **The enemy is not the bug that crashes; it is the bug that makes the result
   look right.** A wrong shape, a leaked test set, a BOS token handled differently in
   two paths. Debugging is mostly "this number is too good, why?" You did exactly this
   in commit `a1e2b82` (the BOS-policy fix that changed the lesion results). That
   week was a Research Engineer week.
3. **The goalposts move, on purpose.** The result of Tuesday's experiment changes
   what Wednesday's should be. There is no spec. Someone who needs a stable
   definition of done will find this exhausting; someone who finds specs confining
   will find it freeing.

### 1.3 A representative week

Composite of the first-person accounts in §1.4, not one person's week:

- **Mon.** A researcher's idea from Friday: "does the effect survive if we only
  ablate at position 0?" You write the intervention, hit a shape mismatch, fix it,
  launch on four GPUs, have numbers by evening. They look too clean.
- **Tue.** Find why: an off-by-one at the sequence start. Rerun. The effect halves
  but is still there. Make five plots, post them with two sentences of caveat.
- **Wed.** Reading group. Refactor the activation-caching code because three people
  now depend on it. Review a teammate's PR that reimplements a paper's baseline.
- **Thu.** The eval loop is the bottleneck; profile it, batch it, four times faster.
  Start a six-seed sweep.
- **Fri.** Sweep done. Write a one-page memo with confidence intervals. Two of the
  week's five ideas are dead. The group decides the next question.

Roughly: a third writing experiment code, a third debugging and making results
trustworthy, a sixth reading and discussing, a sixth writing up and tooling. The
proportion of *systems* work (distributed training, kernels, data pipelines) varies
from near zero on an interpretability team to most of the job on a pretraining team.

### 1.4 What postings and first-person accounts say (2026-09)

*Filled from a survey of current postings; see the sourced section at the end of
this document (§6).*

---

## Part 2 — The skill map

Twelve skills in four clusters. For each: what it means concretely, what "good"
looks like, and how a hiring manager would *see* it. The "where you stand" column
is my read from working with you since August; correct it in Part 3.

### Cluster A — Making experiments happen

**A1. Sketch to running experiment in a day.** Given a verbal idea, produce code
that runs end to end today, correct but not polished. *Good:* you know when to
write throwaway code and when not to. *Seen via:* commit history of a research
repo; how fast a take-home is turned around. *You:* strong on the SWE half; the ML
half (knowing which tensor to hook, which library does what) is what this year is
building.

**A2. Compute fluency.** Running on a cluster without babysitting: job arrays,
checkpoint/resume, pre-caching models on a login node, knowing what fits on one
GPU. *Good:* a sweep of 30 runs launches from one config file and you find out by
morning which ones died and why. *Seen via:* SLURM scripts, provenance manifests,
a results directory someone else can read. *You:* already doing this on the ORC
cluster; the provenance-manifest habit from `research_standards.md` §20.3 is the
next step.

**A3. Framework depth (PyTorch, and increasingly JAX).** Hooks, autograd, memory
behaviour, dtype pitfalls, `torch.compile`, what `no_grad` does and does not save.
*Good:* you can read a model's forward pass and say where the activation you want
lives. *Seen via:* interviews ask this directly ("implement attention, then
KV-cache it"). *You:* the gap named in August (attention "only vaguely") is this
skill. It is the most interview-tested skill on this list.

### Cluster B — Making experiments trustworthy

**B1. Experimental design.** Baseline and null stated before running; controls
that would falsify the hypothesis; seeds as the unit of independence; effect size
with an interval. *Good:* "zero / mean / magnitude-matched random" is reflexive.
*Seen via:* an experiment README with a satisfied-when; a write-up whose caption
names the family size. *You:* the repo's rules are exactly this, and you have been
applying them. This is the most *underrated* skill in hiring: many candidates can
code, few can design a control.

**B2. Paranoia about results that look right.** Sanity checks, ablations of the
pipeline itself, "the result is wrong until it survives my attempts to break it".
*Good:* you reproduce a known number before trusting your pipeline on a new one.
*Seen via:* notes that record what broke; a replication of a published result.
*You:* present, and the adversarial-review habit is a version of it.

**B3. Evaluation literacy.** Knowing what a benchmark measures, its contamination
and variance, when a metric substitution is silent. *Good:* you can say why chrF++
and COMET disagree and which to trust for which claim. *Seen via:* a write-up that
states coverage next to every cross-model claim. *You:* building; the MT
evaluation stack in `speech-translation/` is a real asset here.

**B4. Statistics that fit the data.** Bootstrap CIs, multiplicity correction,
knowing that n=24 is exploratory. Not advanced statistics; the *habit* of bounding
every claim. *Seen via:* any table you produce.

### Cluster C — Systems and performance

**C1. Profiling and throughput.** Finding the bottleneck (data loading, host-device
copies, Python overhead) and removing it. *Good:* you know your run's tokens per
second and why. *Seen via:* "made X 4× faster" lines on a résumé; interview
questions on memory arithmetic. *You:* transferable SWE skill, not yet exercised on
GPU code.

**C2. Distributed training.** DDP, FSDP, tensor/pipeline parallelism, sharded
checkpoints. *Good:* you can take a single-GPU script to eight GPUs and explain the
memory budget. *Seen via:* a from-scratch training run at any scale. **This is the
skill a 698R project is least likely to produce**, and the one pretraining-adjacent
postings weight most. Interp and eval teams weight it much less.

**C3. Data pipelines.** Tokenization, dedup, streaming, filtering, contamination
checks. *Seen via:* a dataset card you wrote. *You:* strong general-engineering
transfer (AWS, batch systems).

### Cluster D — Research literacy and communication

**D1. Reading and reimplementing papers.** Turning a paper's method section into
code that reproduces its headline number. *Good:* you do this in days, notice when
the paper omits a detail, and find it in the appendix or the released code. *Seen
via:* a public reimplementation with a "reproduced Table 2 within 0.3" line. **This
is the single most-requested evidence in postings and interviews.** *You:* the
Table 2 coordinate-rank replication in this track is one; make it visible.

**D2. Knowing the literature well enough to place a result.** Not encyclopaedic;
enough to say "that is Sun et al.'s finding with a different control". *Seen via:*
a related-work paragraph that is specific. *You:* the September sweep (130 PDFs,
47 mapped) is unusually deep for a master's project; the risk is breadth without
the anchoring result.

**D3. Writing results people can act on.** One page, the comparison stated (not the
impression), plots that answer one question each, what you did *not* find. *Seen
via:* a blog post or a short paper; the week-9 draft. *You:* the claim-hygiene
rules are training for this. Most engineers write badly about results; this is a
differentiator.

**D4. Research taste.** Choosing the experiment that most reduces uncertainty per
GPU-hour; knowing when to stop. Learned only by doing it many times with feedback.
*Seen via:* the sequence of experiments in a notes file — did each one follow from
the last? *You:* early; that is what the advisor relationship is for.

### The trait underneath: tolerance for a high failure rate

Most experiments produce "no effect" or "the idea was wrong". A Research Engineer
who takes that as personal failure burns out; one who takes it as information
thrives. The self-assessment below probes this directly, because no amount of
skill compensates for it.

---

## Part 3 — Self-assessment (fill in here)

Answer in the file. Replace `__` with a number. Be honest rather than aspirational;
the point is to find out, not to pass.

### 3.1 Temperament: which column do you want to be judged on?

Rate each 1 (strongly disagree) to 5 (strongly agree).

**RE-leaning**
- R1. I am satisfied when I *find out* whether something is true, even if the code gets deleted afterwards. `__`
- R2. "This number is too good; something is wrong" excites me more than it annoys me. `__`
- R3. I would rather have a vague question and a week than a precise spec and a week. `__`
- R4. Reading a paper and rebuilding its result sounds like a good Saturday. `__`
- R5. I can spend a day making a plot say exactly one thing and consider that day well spent. `__`
- R6. I am fine being second author on someone else's idea if my experiment made it credible. `__`

**MLE / product-leaning**
- M1. I get most satisfaction from something I built being used, reliably, by many people. `__`
- M2. I want a clear definition of done and to be measured against it. `__`
- M3. On-call, monitoring, and latency budgets sound like real engineering, not distraction. `__`
- M4. I would rather improve a shipped model by 2% than discover why it behaves the way it does. `__`

**Research Scientist-leaning**
- S1. I want to choose the questions, not just answer them. `__`
- S2. I would do a PhD if the job required it. `__`
- S3. Publishing under my own name matters to me. `__`

**ML infra-leaning**
- I1. Making a training run 30% faster is more satisfying than analysing its output. `__`
- I2. I enjoy reading profiler traces and memory dumps. `__`
- I3. I want to work on the system, not on the questions the system answers. `__`

Score: sum each block, divide by item count. A gap of ≥1.0 between your top block
and the next is a real signal; below that, the survey has told you the roles are
close for you and the decision should be made on market grounds (Part 5).

### 3.2 Gut-check scenarios

Write two or three sentences each.

- G1. You spent two weeks on an experiment. The answer is "no effect, and the
  hypothesis was wrong". The code will not be reused. How do you feel on the Friday?
  `__`
- G2. A researcher changes what they want measured for the third time this week.
  Each change is reasonable in light of the previous result. Honest reaction? `__`
- G3. You are asked to reproduce a paper's Table 3. After three days you are 2
  points off and cannot find why. Do you (a) report the gap and move on, (b) spend
  three more days, (c) something else? What did you actually do the last time this
  happened? `__`
- G4. Think of the best week of your three years as a SWE. What made it good? Does
  the RE week in §1.3 contain that ingredient? `__`

### 3.3 Skill inventory

Rate 0–3: **0** never done it; **1** done it with help or following a tutorial;
**2** done it alone on a real problem; **3** could teach it and know its failure
modes. Then mark my read (from Part 2) as `agree`/`too high`/`too low`.

| Skill | Self | My read | Agree? |
|---|---|---|---|
| A1 sketch → running experiment in a day | `__` | 2 (SWE half) / 1 (ML half) | `__` |
| A2 compute fluency (SLURM, checkpoint/resume, provenance) | `__` | 2 | `__` |
| A3 PyTorch/transformer internals | `__` | 1 | `__` |
| B1 experimental design (nulls, controls, seeds) | `__` | 2, recently | `__` |
| B2 paranoia about results that look right | `__` | 2 | `__` |
| B3 evaluation literacy (benchmarks, metrics, contamination) | `__` | 1–2 | `__` |
| B4 statistics habit (CIs, multiplicity) | `__` | 1 | `__` |
| C1 profiling and throughput on GPU code | `__` | 1 | `__` |
| C2 distributed training | `__` | 0 | `__` |
| C3 data pipelines | `__` | 2 (general), 1 (ML-specific) | `__` |
| D1 reimplementing a paper to its number | `__` | 1–2 | `__` |
| D2 placing a result in the literature | `__` | 2 (this topic), 1 (elsewhere) | `__` |
| D3 writing results people can act on | `__` | 2 | `__` |
| D4 research taste | `__` | 1 | `__` |

### 3.4 Constraints that decide the market, not the fit

- K1. Must have an offer by: `__` (December graduation; when does the *search* have to end?)
- K2. Geography: relocate anywhere / specific cities / remote only: `__`
- K3. Work-authorisation constraints, if any (sponsorship needed?): `__`
- K4. Would you take "ML Engineer on a research team" or "Research Engineer at a
  university lab / nonprofit" at lower pay as the entry point, then move? `__`
- K5. PhD: closed / open in 2–3 years / actively considering: `__`
- K6. Is there a domain you would *refuse* to work in (e.g. ads, defence, trading)? `__`

### 3.5 Signals that this is not the job

Any two of these is worth taking seriously:

- R1 ≤ 2 and M1 ≥ 4 (you need your work to be *used*, not *known*).
- G1 reads as frustration rather than curiosity.
- G2 reads as "unprofessional" rather than "that is how it goes".
- You skipped the reading in this repo's `papers/` and went straight to code, every time.
- A3 self-rating is 0–1 *and* the thought of spending 40 hours on transformer
  internals is unappealing (the interview loop will test it regardless).

None of these says you would be bad at it. They say you would not *enjoy* it, and
the failure rate of the work makes enjoyment load-bearing.

---

## Part 4 — What a 698R project can prove, and what it cannot

### 4.1 The form's constraints

From `698R_Proposal_Form.pdf`: a one-page description that states "what will be
learned"; a weekly-meeting plan; a deliverables list used for grading; 125–150
tracked hours split by category; a week-9 draft report; a final report and
presentation. **"What will be learned" is the line where the skill map goes**, in
the advisor's language: experimental design, reproducible research engineering,
evaluation, scientific writing. Advisors approve projects that name what the
student will learn; so do hiring managers reading a résumé line.

### 4.2 Skill → project feature → what a hiring manager sees

| Skill | Build this into the project | The visible artifact |
|---|---|---|
| B1 design | spec-first experiments (`/new-experiment`): question, satisfied-when, null, analysis plan **before** the runner | `experiments/*/README.md` with a verdict line |
| A2 compute, reproducibility | provenance manifest (git SHA, resolved config, seed, library versions) in every results dir; one YAML per run | a results tree a stranger can rerun |
| B2, D1 replication | **reproduce one published number before extending anything** | "reproduced X within Y" in the README, first line |
| B4 statistics | control triple, ≥3 seeds where seeds exist, bootstrap CIs, family size in captions | every table in the write-up |
| B3 evaluation | one metric substitution documented where used, not only in a docstring | the methods section |
| C1 performance | profile the activation-caching or eval loop once; record the before/after | a one-line notes entry with numbers |
| D3 writing | a four-page write-up in the style of a workshop paper, plus a 600-word blog post | the thing you link in applications |
| A1, tooling | the detector as an installable tool with a CLI and a test that every committed config loads | a repo with a README that runs in five minutes |
| D4 taste | `notes.md` as a dated changelog where each experiment follows from the last, including what broke | the notes file, read by anyone who looks past the README |

Everything in the middle column is already a rule in `CLAUDE.md` or
`research_standards.md`. The project does not need extra process; it needs the
existing process **made visible**: a public repo, a write-up, and a README that
leads with the replicated number.

### 4.3 What the project cannot demonstrate, and where to get it

- **C2 distributed training.** Not in scope of a 150-hour inference-time study. A
  weekend side exercise covers the interview version: fine-tune a small model on
  one GPU, then on four with FSDP, and write 300 words on the memory budget. Do it
  once, put it on GitHub, do not make it the project.
- **Working inside a research team.** Only partly; the advisor is the researcher
  you serve. A small contribution to a tool you use (TransformerLens, nnsight,
  lm-eval) demonstrates the collaboration part cheaply.
- **Scale.** Nobody expects a master's project at scale; they expect the *habits*
  that scale.

### 4.4 Three project shapes that maximise RE evidence, regardless of topic

1. **Replicate-then-extend.** Reproduce a published finding across models the
   paper did not test; report where it holds, where it fails, with controls. Highest
   evidence-per-hour for D1, B1, B2, D3. The superweights track is already this
   shape.
2. **Build the measurement, not the finding.** A detector, an eval harness, or a
   benchmark with a calibrated null; the "finding" is what the tool reveals across a
   model set. Strong for A1, A2, B3; weaker for D4 unless the tool answers a question.
3. **A from-scratch training study.** Small models, checkpoint suite, a dynamics
   question. Only shape that produces C2; riskiest in 150 hours, and the
   `interlingua/` audits say why.

Whatever the topic, the deliverable that gets read is the same: a README with a
replicated number, a short write-up with controls and intervals, and a notes file
that shows judgment.

---

## Part 5 — The December constraint

A job by December means the search runs *during* the project, not after it. That
imposes a schedule on the deliverables that the form does not:

- **By week 6:** a public repo with the replication and a README; this is what
  goes in applications sent in October.
- **By week 9 (the draft report):** the write-up in near-final form; a blog post
  version of it.
- **Finals week:** the presentation is interview practice.

The realistic target set for MS + 3 years SWE, no PhD `[postings, §6 confirms or
corrects]`: Research Engineer at nonprofit and academic-adjacent labs, at applied
research groups in industry, and at startups; "ML Engineer, research" at product
companies; residencies. Frontier-lab RE is reachable from a strong SWE background,
and several REs there came that way, but the loop is long and the odds per
application are low; apply, do not plan around it.

**The project's job in that search is one thing:** to be the artifact a hiring
manager can read in ten minutes and conclude "this person designs controls, checks
their results, and writes clearly". Choose the topic for that, then for interest.

---

## Part 6 — Sources: postings and first-person accounts

*(Filled from the 2026-09-07 survey.)*
