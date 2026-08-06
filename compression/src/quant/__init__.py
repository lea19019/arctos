"""Official-library quantizers for the PTQ-MT replication (arXiv:2508.20893).

This package wraps the *paper's own tools* — AutoAWQ, bitsandbytes, AutoRound,
and llama.cpp/GGUF — so the replication reproduces their pipeline faithfully.
It is intentionally independent of Arctos's from-scratch quantizers in
``src/interp/compress.py`` (those are for the q6 line of work); nothing here
imports them.

See ``registry.py`` for the method table and supported bit-widths.
"""
