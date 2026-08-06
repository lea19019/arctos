"""Tests for the quant method registry and chat-prompt builder."""

from __future__ import annotations

from src.models._chat_prompt import build_mt_instruction
from src.quant import registry


def test_bit_width_method_coverage():
    # 4-bit: all four methods; 2-bit: only GGUF + AutoRound (AWQ/BnB have none).
    assert set(registry.methods_for_bits(4)) == {"awq", "bnb", "autoround", "gguf"}
    assert set(registry.methods_for_bits(2)) == {"autoround", "gguf"}


def test_method_backends():
    assert registry.get("bnb").backend == "hf"
    assert registry.get("gguf").backend == "gguf"
    assert registry.get("awq").needs_artifact is True
    assert registry.get("bnb").needs_artifact is False


def test_gguf_module_resolves_lazily():
    m = registry.get("gguf")
    assert m.module is not None  # lazily imported on get()
    assert hasattr(m.module, "QTYPE")


def test_mt_instruction_mentions_both_languages():
    instr = build_mt_instruction("Bonjour le monde.", "fr", "en")
    assert "French" in instr and "English" in instr
    assert "Bonjour le monde." in instr
