# Skills to build for ML / software engineering roles (2026-09-07)

You will apply to every ML engineer and software engineer role that fits. These are the
skills that showed up across ~30 engineering postings at AI companies and big tech,
ranked by how often they were required. Build these; they transfer to any of the roles.

## What postings require, most common first
1. Python, including reading and changing large codebases you did not write.
2. Distributed systems you built *and ran* in production. The interview wants numbers:
   traffic, data size, what broke, what you did. Your ToAll and Symplsoft work is this;
   write the story down.
3. Docker, Kubernetes, a cloud (you have AWS), CI/CD.
4. Debugging and profiling: finding why something is slow or wrong in an unfamiliar stack.
5. PyTorch, and how a transformer actually runs: attention, KV cache, batching.
6. Building tools other engineers depend on (this is what "next to research" jobs are).
7. Evaluation: how you know a model got better or worse. Regression detection, test sets.
8. Data pipelines at scale.
9. Working from a vague goal without waiting for a spec.

Not required anywhere: PhD, publications, research experience.

## What interviews test
- Coding. AI companies: practical building under a timer (an in-memory database in
  stages, an LRU cache then make it thread-safe). Big companies: two algorithm problems
  in 45 minutes. Meta has an AI-assisted coding round; Anthropic bans AI in interviews.
- System design: a service under growing load; at AI companies, model serving.
- ML system design (Meta, Amazon): data, training, evaluation, serving, monitoring.
- A 45-minute walkthrough of one project. This is the only place the MS project matters.

## What to do this semester, outside project hours
1. Interview coding practice, both kinds. 40+ hours. This is the gate.
2. Write the ToAll story with numbers.
3. Implement attention with a KV cache by hand, once. Then a simple quantizer.
4. One weekend: fine-tune a small model on 1 GPU, then on 4 with FSDP. Note the memory
   difference. That is your distributed-training résumé line.
5. Run vLLM once and explain where the time goes.
6. Make the Kubernetes line on the résumé true.
7. Rehearse the project walkthrough.

## Two immigration facts
- File OPT the day the window opens (mid-September); the permit takes 1–3 months.
- As a Mexican citizen you can use TN status (no lottery). Ask every recruiter on the
  first call whether they file TN. A proposed August 2026 rule may add a ~$100k fee to
  H-1B for students changing status; verify with the international office.
