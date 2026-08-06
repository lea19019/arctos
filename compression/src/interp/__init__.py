"""Core interpretability methods.

Each method has its own module. Per-model adaptations are flagged inside the
module — the API stays uniform across Aya / omt-llama / Tower as long as the
model loader returns a TransformerLens HookedTransformer. Where TL coverage
is incomplete, the method module documents the HF-hook fallback.

Method validation discipline (per `PHASE1-PLAN.md`): before trusting any
finding from a single method, cross-check it against at least one other on
the same examples. The validation matrix is in PHASE1-PLAN.md.
"""
