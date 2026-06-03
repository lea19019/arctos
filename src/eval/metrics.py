"""Translation-quality metrics with caveats documented in-line.

chrF++ is the workhorse here: it's reference-based, tokenization-light
(operates on character n-grams + word n-grams), and works across scripts
(Latin, Han, Arabic) without per-language tokenizer setup — which matters
for our cs-de / en-zh / en-arz mix. BLEU is reported alongside but needs
the right tokenizer per target (`zh` for Chinese); COMET is optional and
heavy (a neural model), off by default.

We use sacrebleu directly — no reimplemented metric math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sacrebleu.metrics import BLEU, CHRF

# sacrebleu BLEU tokenizer per target language.
_BLEU_TOKENIZER = {"cs-de": "intl", "en-zh": "zh", "en-arz": "intl"}


@dataclass(frozen=True)
class MetricScores:
    bleu: float
    chrfpp: float
    comet: float | None  # None if COMET was disabled


def chrfpp(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus chrF++ (word_order=2). Script-agnostic; our primary metric."""
    metric = CHRF(word_order=2)  # word_order=2 is the "++" in chrF++
    return float(metric.corpus_score(list(hypotheses), [list(references)]).score)


def sentence_chrfpp(hypothesis: str, reference: str) -> float:
    """Per-sentence chrF++ — used for per-example sensitivity deltas."""
    return float(CHRF(word_order=2).sentence_score(hypothesis, [reference]).score)


# ---- COMET (WMT-grade semantic MT metric) ------------------------------------
# DEFAULT = XCOMET-XL, the primary metric of the WMT25 Model Compression Shared
# Task (Gaido et al., 2025). Falls back to wmt22-comet-da if XCOMET-XL isn't
# cached. COMET is source-aware (src+mt+ref) — it can reveal adequacy gains
# that chrF++ (character overlap) misses, which is exactly the worry about our
# chrF++-only verdict.
WMT25_COMET_MODEL = "Unbabel/XCOMET-XL"
_COMET_MODELS: dict[str, object] = {}


def _comet_ckpt(model_name: str) -> str:
    import glob, os
    slug = "models--" + model_name.replace("/", "--")
    hits = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/{slug}/**/checkpoints/model.ckpt"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{model_name} checkpoint not in HF cache")
    return hits[0]


def comet_score(sources, hypotheses, references, *,
                model_name: str = WMT25_COMET_MODEL,
                batch_size: int = 16, gpus: int = 1, fallback: bool = True):
    """Corpus + per-segment COMET. Returns (system_score, [scores]).

    Defaults to XCOMET-XL (the WMT25 compression-task metric). gpus=1 uses GPU;
    gpus=0 runs on CPU (smoke tests). If XCOMET-XL is unavailable and
    fallback=True, uses wmt22-comet-da instead.
    """
    global _COMET_MODELS
    if model_name not in _COMET_MODELS:
        from comet import load_from_checkpoint
        try:
            ckpt = _comet_ckpt(model_name)
        except FileNotFoundError:
            if not fallback:
                raise
            model_name = "Unbabel/wmt22-comet-da"
            ckpt = _comet_ckpt(model_name)
        _COMET_MODELS[model_name] = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)]
    out = _COMET_MODELS[model_name].predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    return float(out.system_score), [float(x) for x in out.scores]


def evaluate(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    pair: str,
    use_comet: bool = False,
) -> MetricScores:
    """Compute BLEU + chrF++ (+ optional COMET) for one batch.

    Tokenizer choices (sacrebleu): cs-de -> intl, en-zh -> zh (required for
    meaningful BLEU on zh-Hans), en-arz -> intl (Arabic script; chrF++ is
    the trustworthy metric here, BLEU is a rough companion).

    COMET is off by default (heavy neural model; for en-arz it's also
    trained mostly on standard Arabic varieties, so chrF++ is the sanity
    check). Enable with use_comet=True only when the COMET model is cached.
    """
    bleu = BLEU(tokenize=_BLEU_TOKENIZER.get(pair, "intl"))
    bleu_score = float(bleu.corpus_score(list(hypotheses), [list(references)]).score)
    chrf_score = chrfpp(hypotheses, references)

    comet_score = None
    if use_comet:
        from comet import download_model, load_from_checkpoint  # lazy import

        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        # COMET needs source too; caller must pass it via a richer API. For
        # phase one we only wire reference-based metrics; COMET stays a
        # documented opt-in until a source-carrying path is needed.
        raise NotImplementedError(
            "COMET requires source sentences; use chrF++/BLEU for now."
        )

    return MetricScores(bleu=bleu_score, chrfpp=chrf_score, comet=comet_score)
