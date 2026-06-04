"""Method registry for the PTQ-MT replication.

Maps each paper method to its backend and supported bit-widths so the
experiment driver can iterate uniformly. Two backends:

  * ``hf``   — AWQ, BnB, AutoRound: quantize (maybe to disk) then run through
               the shared ``hf_generate`` greedy loop.
  * ``gguf`` — llama.cpp: convert/imatrix/quantize then serve + generate.

The paper's bit-width coverage:
  4-bit: AWQ, BnB, GGUF (Q4_K_M), AutoRound
  2-bit:           GGUF (Q2_K),  AutoRound      (AWQ & BnB have no 2-bit)
"""

from __future__ import annotations

from dataclasses import dataclass

from . import hf_autoround, hf_awq, hf_bnb


@dataclass(frozen=True)
class Method:
    name: str
    backend: str            # "hf" | "gguf"
    supported_bits: tuple[int, ...]
    needs_artifact: bool    # writes a quantized copy to disk (False = load-time)
    module: object          # the implementing module


# GGUF is imported lazily inside the property to avoid importing llama.cpp glue
# (and its server deps) when only HF methods are used.
def _gguf_module():
    from . import gguf
    return gguf


REGISTRY: dict[str, Method] = {
    "awq": Method("awq", "hf", hf_awq.SUPPORTED_BITS, True, hf_awq),
    "bnb": Method("bnb", "hf", hf_bnb.SUPPORTED_BITS, False, hf_bnb),
    "autoround": Method("autoround", "hf", hf_autoround.SUPPORTED_BITS, True, hf_autoround),
    # gguf.module is resolved lazily; supported bits are (4, 2).
    "gguf": Method("gguf", "gguf", (4, 2), True, None),
}

ALL_METHODS = tuple(REGISTRY)


def get(name: str) -> Method:
    if name not in REGISTRY:
        raise ValueError(f"Unknown method {name!r}; known: {ALL_METHODS}.")
    m = REGISTRY[name]
    if name == "gguf" and m.module is None:
        return Method(m.name, m.backend, m.supported_bits, m.needs_artifact, _gguf_module())
    return m


def methods_for_bits(bits: int, methods: tuple[str, ...] | None = None) -> list[str]:
    """Which requested methods support the given bit-width."""
    names = methods or ALL_METHODS
    return [n for n in names if bits in REGISTRY[n].supported_bits]
