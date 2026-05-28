# Arctos

Phase-one scaffold for *Understanding translation in decoder LLMs as a foundation for compression* — an interpretability-led investigation into how machine translation is carried out inside three open-source decoder-only LLMs (Aya Expanse 8B, omt-llama-8b, TowerInstruct-7B), preceding a phase-two compression method whose hypotheses will be grounded in what phase one reveals.

Thesis spine, model choices, language pairs, and methodological discipline: [`docs/project-summary.md`](docs/project-summary.md).
Investigation plan: [`PHASE1-PLAN.md`](PHASE1-PLAN.md).
Phase-two seed candidates: [`docs/phase2-hypotheses.md`](docs/phase2-hypotheses.md).

## Layout

```
.
├── README.md
├── PHASE1-PLAN.md            # the investigation plan
├── pyproject.toml            # uv-managed env (see "Environment" below)
├── docs/
│   ├── project-summary.md    # 1-page synthesis of prior work + new direction
│   ├── phase2-hypotheses.md  # seed candidates for phase two
│   ├── claude-code-bootstrap.md
│   ├── research.md           # annotated bibliography (phase-two reference)
│   ├── pruning_project.pdf   # prior paper (Castillo & Richardson, BYU)
│   ├── findings/             # writeups per question (Q1–Q5)
│   ├── systems-notes/        # transformer math, GPU memory, kernels, etc.
│   └── learning-log.md       # running personal notes
├── experiments/
│   ├── q1-language-emergence/
│   ├── q2-attention-heads/
│   ├── q3-mlps-and-layers/
│   ├── q4-architecture-comparison/
│   └── q5-importance-vs-sensitivity/
├── notebooks/                # exploratory work
├── src/
│   ├── models/               # loaders for Aya, omt-llama, Tower
│   ├── interp/               # core interpretability methods
│   │   ├── logit_lens.py
│   │   ├── activation_patching.py
│   │   ├── ifr.py
│   │   └── probing.py
│   ├── data/                 # MT calibration data + clean/corrupt pair generators
│   └── eval/                 # BLEU / chrF++ / COMET wrappers
├── tests/                    # mirrors src/ layout; cpu + gpu markers
├── data/                     # gitignored — datasets
├── models/                   # gitignored — checkpoints
├── configs/                  # shared / global run configs
└── scripts/                  # one-off CLIs
```

### Non-obvious choices

- **`experiments/qN-*/` mirrors the five investigative questions, not pipeline stages.** Phase one is organized around questions; each question's directory holds its own configs and notes so the work stays reviewable per question, not per method.
- **`src/interp/` holds methods, not models.** A method (logit lens, IFR) is implemented once with per-model adaptations flagged inside it. Per-model loaders live in `src/models/`.
- **`docs/findings/` vs `experiments/qN-*/notes.md`.** `notes.md` is working memory while a question is open. `docs/findings/qN.md` is the satisfied-when artifact when the question closes. The split makes the gating between phase one and phase two visible.
- **`docs/systems-notes/`** is interleaved into the questions, not run as a parallel track. The folder exists so the systems learning is collected and reusable, but each note is the result of doing one of the experiments.
- **No phase-two scaffold yet.** Phase two will be designed from Q5 findings; pre-building its directories would invite premature commitment.

## Environment

```bash
uv sync          # creates .venv from pyproject.toml; commit uv.lock when ready
uv run pytest    # runs anything inside the env
```

Hardware target is A100 80GB; experiments that would exceed that (e.g., loading multiple models simultaneously) are flagged in the per-question configs and risk register.

## Status

Scaffold only. No interpretability methods are implemented yet — see TODOs in `src/interp/*.py` and `experiments/qN-*/experiment.py`. Per the bootstrap, phase two stays empty until Q5 closes.
