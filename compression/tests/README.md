# Tests

Mirrors `src/` layout. Two markers, both required for any new method:

- `@pytest.mark.cpu` — runs on CPU using a tiny dummy model (typically `gpt2` via TransformerLens, or a hand-built minimal HookedTransformer config). Tests shape contracts, math invariants, dataclass fields. Must pass on a CPU-only machine.
- `@pytest.mark.gpu` — runs on a real target model (Aya / omt-llama / Tower) on CUDA. Tests end-to-end behavior on real MT prompts. Skipped automatically when CUDA is unavailable.

`@pytest.mark.slow` exists for tests over ~30s; opt in with `-m slow`.

Per `archive/phase1_plan.md` "Testing discipline": a method is not done until both CPU and GPU tests exist.

## Running

```bash
uv run pytest                     # CPU tier only (default; GPU skipped without CUDA)
uv run pytest -m gpu              # GPU tier only (requires CUDA + model checkpoints)
uv run pytest -m "cpu or gpu"     # both
uv run pytest -m "not slow"       # skip slow tests
uv run pytest tests/interp/       # one module
uv run pytest -n auto             # parallel via pytest-xdist
```

## Why both tiers

- **CPU tests** catch shape and math bugs cheaply. They cannot exercise model-specific quirks (Aya's Cohere attention scaling, omt-llama's tokenizer, Tower's SFT prompt template) — those need real models.
- **GPU tests** catch real-model behavior the dummies cannot. They are slower and require checkpoints; gated behind `-m gpu`.

If you write a method and only ship CPU tests, the GPU-side bug surfaces during a Q-experiment, where it is much more expensive to debug.
