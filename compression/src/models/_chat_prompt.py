"""Chat-template MT prompt for the PTQ-MT replication (arXiv:2508.20893).

The paper uses "the same simple translation prompt for all models ... while
using each model's chat template" and greedy decoding. To keep the HF path
(AWQ/BnB/AutoRound) and the llama.cpp/GGUF path comparable, *both* render the
prompt through the model's HF tokenizer ``apply_chat_template`` and feed the
resulting string verbatim to generation. Never let llama.cpp apply its own
built-in template — that would silently diverge from the HF runs.

Qwen3 are reasoning models; the paper disables reasoning. That is a
template-level flag (``enable_thinking=False``) passed per-model via
``chat_kwargs`` in the config.
"""

from __future__ import annotations

# Short code -> full language name used in the instruction. Region-neutral
# names match how the paper phrases the task (e.g. "Bengali", not "Bengali
# (India)").
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "ja": "Japanese",
    "fr": "French",
    "pl": "Polish",
    "bn": "Bengali",
    "ml": "Malayalam",
    "zu": "Zulu",
}


def build_mt_instruction(source: str, src_lang: str, tgt_lang: str) -> str:
    """The single direct-translation instruction, identical across models.

    Phrased to elicit only the translation (no preamble), matching the
    paper's "simple translation prompt" intent.
    """
    src_name = LANG_NAMES[src_lang]
    tgt_name = LANG_NAMES[tgt_lang]
    return (
        f"Translate the following {src_name} text into {tgt_name}. "
        f"Output only the translation, with no explanations or notes.\n\n"
        f"{source}"
    )


def render_chat_prompt(
    tokenizer,
    source: str,
    src_lang: str,
    tgt_lang: str,
    *,
    chat_kwargs: dict | None = None,
) -> str:
    """Render the instruction as a templated prompt string ready to generate.

    Returns the *string* (``tokenize=False``) with the generation prompt
    appended, so the identical text can be tokenized by HF or passed to
    ``llama-cli -p``. ``chat_kwargs`` carries per-model template flags such as
    ``{"enable_thinking": False}`` for Qwen3.
    """
    instruction = build_mt_instruction(source, src_lang, tgt_lang)
    messages = [{"role": "user", "content": instruction}]
    kwargs = dict(chat_kwargs or {})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    except TypeError:
        # Model's template doesn't accept a passed flag (e.g. enable_thinking
        # on a non-Qwen3 tokenizer): retry without the extra kwargs.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
