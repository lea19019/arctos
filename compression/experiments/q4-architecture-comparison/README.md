# Q4 — Architecture comparison: the shared-depth hypothesis

> How does the translation footprint differ across the three architectures? Specifically: do MT-critical components generalize across Aya, omt-llama, and Tower, or only the depth profile?

## Working hypothesis

> Specific MT-critical components (which heads, which MLPs) will **not** generalize across Aya, omt-llama, and Tower — these are model-specific accidents of training. But the **depth profile** — where in the network MT-critical work concentrates relative to total depth — **may** generalize.

Three claim strengths to test:

- **V1 (weakest).** Early layers (~first 25%) are MT-irrelevant; final 1–2 layers protected; middle is where MT happens. Confirms task-agnostic depth-pruning convergence — not novel for MT.
- **V2 (medium).** A characteristic depth signature (source understanding → language-agnostic semantics → target commitment) with similar relative depth fractions across architectures. Genuinely novel if true.
- **V3 (strongest).** The depth signature is consistent enough that bit-allocation or pruning can be made from depth fraction alone, without per-model interpretability. Highest practical payoff.

## Methods

Synthesis across Q1–Q3 for the three models on the same shared MT examples (same source sentences; all three models' interpretability traces). No new method here — the value is the apples-to-apples comparison.

## Models covered

All three jointly.

## Embedded learning

- How training intent (general LM vs MT-purpose-built) shapes internal structure.
- What "the translation circuit" even means as a unit of analysis when it spans the whole network.
- The difference between architectural similarity (all three are Llama-class) and functional similarity.

## Expected artifacts

- `docs/findings/architecture-comparison.md` — paper-style writeup pitched at someone unfamiliar with the project.

## Satisfied when

The writeup exists and explicitly lands on V0 / V1 / V2 / V3 with evidence per language pair × model cell. **Falsifying evidence is required to be reported.** The prior paper's IFR results showed different layer rankings between cs→de and en→es within Aya alone — Q4 must be honest about findings that cut against V2/V3, not paper over them.

## Tests

No new methods → no new method tests. The cross-method agreement tests (`tests/interp/test_ifr.py::test_ifr_agrees_with_patching_on_top_layer`) gate the inputs Q4 synthesizes.

## Working notes

See `notes.md`.
