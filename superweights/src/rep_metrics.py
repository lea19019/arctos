"""Repetition metrics for generated token sequences. Pure Python, CPU only.

Definitions are pre-registered in docs/repetition_experiment_design.md §1.1.
Every function takes a list of token ids (or strings) for the GENERATED part
only — the prompt is excluded by the caller.

    loop_onset(tokens)      -> (t_star or None, gram)   L-onset rule
    seq_rep_n(tokens, n=4)  -> float                     Welleck et al. 2019, Eq. 10
    rep_r(tokens)           -> float                     Fu et al. 2021, §4.1
    tng(hyp_tokens, src_tokens, n=4) -> int              Guerreiro et al. 2023, §5.1

L-onset (g=4, k=3, W_scan=100, t_min=32): t* is the smallest t such that the
g-gram starting at t already occurred at some t' < t AND that g-gram occurs at
least k times within [t, t + W_scan]. A generation loops iff such t* exists
with t* >= t_min. k and W_scan are Hiraoka & Inui 2024 §2.2; g=4 is the shared
n of seq-rep-4 and TNG; the equal-interval constraint of Hiraoka is dropped
(the lesion loops drift). t_min guarantees a pre-onset window.
"""

from collections import Counter

G, K, W_SCAN, T_MIN = 4, 3, 100, 32


def _grams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def loop_onset(tokens, g=G, k=K, w_scan=W_SCAN, t_min=T_MIN):
    """Return (t_star, gram) under the L-onset rule, or (None, None).

    t_star indexes the GENERATED tokens (0 = first generated token) and is the
    start of the second occurrence of the offending g-gram.
    """
    grams = _grams(tokens, g)
    seen = set()
    for t, gram in enumerate(grams):
        if gram in seen:
            window = grams[t:t + w_scan]
            if sum(1 for x in window if x == gram) >= k:
                if t >= t_min:
                    return t, gram
                # an early loop (t < t_min) is still a loop, but has no
                # pre-onset window; report it so the caller can count it
                return t, gram
        seen.add(gram)
    return None, None


def seq_rep_n(tokens, n=4):
    """Welleck et al. 2019 Eq. 10: 1 - |unique n-grams| / |n-grams|. 0 = none."""
    grams = _grams(tokens, n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def rep_r(tokens):
    """Fu et al. 2021 §4.1: fraction of token positions covered by some
    bigram that occurs more than once in the sequence."""
    if len(tokens) < 2:
        return 0.0
    bigrams = _grams(tokens, 2)
    counts = Counter(bigrams)
    covered = set()
    for i, bg in enumerate(bigrams):
        if counts[bg] > 1:
            covered.add(i)
            covered.add(i + 1)
    return len(covered) / len(tokens)


def tng(hyp_tokens, src_tokens, n=4):
    """Guerreiro et al. 2023 §5.1 top n-gram score: max n-gram count in the
    hypothesis minus max n-gram count in the source. They flag >= 2."""
    def top(tokens):
        grams = _grams(tokens, n)
        return max(Counter(grams).values()) if grams else 0
    return top(hyp_tokens) - top(src_tokens)


def summarize(tokens, src_tokens=None, ended_with_eos=False):
    t_star, gram = loop_onset(tokens)
    out = {
        "n_tokens": len(tokens),
        "ended_with_eos": ended_with_eos,
        "loop": t_star is not None,
        "loop_with_window": t_star is not None and t_star >= T_MIN,
        "onset": t_star,
        "seq_rep_4": seq_rep_n(tokens, 4),
        "rep_r": rep_r(tokens),
    }
    if src_tokens is not None:
        out["tng"] = tng(tokens, src_tokens)
    return out
