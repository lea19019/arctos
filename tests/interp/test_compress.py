"""CPU tests for the phase-two compression + salience modules.

Pure-tensor invariants (no model needed) plus in-place-mutator round-trip
checks on the tiny gpt2 fixture. Marked cpu so they run in CI without CUDA.
"""

from __future__ import annotations

import torch
import pytest


@pytest.fixture(scope="module")
def tiny_llama():
    """A tiny randomly-initialised Llama (nn.Linear path — the study models).

    gpt2 (the global `tiny_model`) uses Conv1D, which the study set never does,
    so the compression primitives are validated against a real Llama layout.
    """
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from src.models._hooked import LLAMA_PATHS, HookedModel

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = LlamaConfig(
        vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    hf = LlamaForCausalLM(cfg).to("cpu").eval()
    return HookedModel(hf, tok, LLAMA_PATHS)


from src.interp.compress import (
    absmax_quantize,
    awq_scale,
    magnitude_mask,
    wanda_mask,
    quantize_linears,
    prune_linears,
    ablate_weights,
)


@pytest.mark.cpu
def test_absmax_more_bits_less_error():
    torch.manual_seed(0)
    W = torch.randn(64, 128)
    e8 = (W - absmax_quantize(W, 8)).abs().mean()
    e4 = (W - absmax_quantize(W, 4)).abs().mean()
    e2 = (W - absmax_quantize(W, 2)).abs().mean()
    assert e8 < e4 < e2


@pytest.mark.cpu
def test_keep_cols_are_exact():
    torch.manual_seed(0)
    W = torch.randn(32, 48)
    keep = [0, 5, 17]
    Wq = absmax_quantize(W, 3, keep_cols=keep)
    for c in keep:
        assert torch.allclose(Wq[:, c], W[:, c], atol=1e-6)
    # a non-kept column should generally differ at 3 bits
    assert not torch.allclose(Wq[:, 1], W[:, 1], atol=1e-6)


@pytest.mark.cpu
def test_awq_scale_normalized():
    s = awq_scale(torch.rand(100) + 0.1, alpha=0.5)
    assert abs(float(s.mean()) - 1.0) < 1e-5
    assert (s > 0).all()


@pytest.mark.cpu
def test_masks_hit_target_sparsity():
    torch.manual_seed(0)
    W = torch.randn(16, 100)
    m = magnitude_mask(W, 0.3)
    assert abs((~m).float().mean().item() - 0.3) < 0.02
    act = torch.rand(100) + 0.1
    mw = wanda_mask(W, act, 0.5)
    # per-row 50% pruned
    assert abs((~mw).float().mean().item() - 0.5) < 0.02


@pytest.mark.cpu
def test_quantize_linears_restores(tiny_llama):
    tiny_model = tiny_llama
    w0 = next(tiny_model.iter_layer_linears(0)).detach().clone()
    with quantize_linears(tiny_model, 4, layers=[0], group_size=64):
        w_in = next(tiny_model.iter_layer_linears(0)).detach().clone()
        assert not torch.allclose(w_in, w0)          # actually changed
    w1 = next(tiny_model.iter_layer_linears(0)).detach().clone()
    assert torch.allclose(w1, w0)                     # exactly restored


@pytest.mark.cpu
def test_prune_linears_restores(tiny_llama):
    tiny_model = tiny_llama
    w0 = next(tiny_model.iter_layer_linears(0)).detach().clone()
    with prune_linears(tiny_model, 0.5, method="magnitude", layers=[0]):
        w_in = next(tiny_model.iter_layer_linears(0)).detach()
        assert (w_in == 0).float().mean() > 0.4
    w1 = next(tiny_model.iter_layer_linears(0)).detach()
    assert torch.allclose(w1, w0)


@pytest.mark.cpu
def test_ablate_weights_restores(tiny_llama):
    tiny_model = tiny_llama
    from src.interp.super_weights import _mlp_out_linear
    blocks = tiny_model.arch.get_blocks(tiny_model.hf_model)
    W = _mlp_out_linear(tiny_model, blocks[0]).weight
    v0 = float(W[0, 0])
    with ablate_weights(tiny_model, [(0, 0, 0)]):
        assert float(W[0, 0]) == 0.0
    assert float(W[0, 0]) == v0


@pytest.mark.cpu
def test_super_weight_detect_runs(tiny_llama):
    tiny_model = tiny_llama
    from src.interp.super_weights import detect_super_weights, verify_super_weight
    res = detect_super_weights(tiny_model, ["The capital of France is"], top_k=3)
    assert len(res.candidates) >= 1
    assert len(res.massive_resid) >= 1
    v = verify_super_weight(tiny_model, ["The capital of France is"], res.candidates[0])
    assert "mean_kl_clean_vs_ablated" in v


@pytest.mark.cpu
def test_fisher_and_second_moment(tiny_llama):
    tiny_model = tiny_llama
    from src.interp.hessian_diag import fisher_diagonal, channel_second_moment
    toks = tiny_model.to_tokens("The capital of France is")
    tgt = tiny_model.tokenizer(" Paris", add_special_tokens=False)["input_ids"][:2]
    res = fisher_diagonal(tiny_model, [(toks, tgt)])
    assert len(res.layer_fisher) == tiny_model.cfg.n_layers
    assert res.n_examples == 1
    sm = channel_second_moment(tiny_model, ["The capital of France is"])
    assert all(v.ndim == 1 for v in sm.values())


@pytest.mark.cpu
def test_ternary_binary_levels():
    from src.interp.compress import ternary_quantize, binary_quantize, quant_weight, parse_spec
    torch.manual_seed(0)
    W = torch.randn(32, 128)
    # ternary has exactly 3 levels per group (incl 0); binary has 2 (no 0)
    t = ternary_quantize(W, group_size=None)
    assert len(torch.unique(t[0])) <= 3          # per-row {-s,0,+s}
    b = binary_quantize(W, group_size=None)
    assert (b != 0).all()                        # binary has no zeros
    assert len(torch.unique(b[0])) == 2          # per-row {-s,+s}
    # error ordering: more levels -> less error (4bit < ternary < binary)
    from src.interp.compress import absmax_quantize
    e4 = (W - absmax_quantize(W, 4, group_size=None)).abs().mean()
    et = (W - t).abs().mean()
    eb = (W - b).abs().mean()
    assert e4 < et < eb
    # dispatch
    assert parse_spec("ternary") == ("ternary", 1.58)
    assert parse_spec("binary") == ("binary", 1.0)
    assert parse_spec("3") == ("int", 3)
    assert torch.allclose(quant_weight(W, "ternary", group_size=None), t)
    assert torch.allclose(quant_weight(W, "binary", group_size=None), b)


@pytest.mark.cpu
def test_quantize_linears_accepts_specs(tiny_llama):
    from src.interp.compress import quantize_linears
    tiny_model = tiny_llama
    for spec in ("2", "ternary", "binary"):
        w0 = next(tiny_model.iter_layer_linears(0)).detach().clone()
        with quantize_linears(tiny_model, spec, layers=[0], group_size=None):
            assert not torch.allclose(next(tiny_model.iter_layer_linears(0)).detach(), w0)
        assert torch.allclose(next(tiny_model.iter_layer_linears(0)).detach(), w0)


@pytest.mark.cpu
def test_gptq_beats_rtn_on_reconstruction():
    """GPTQ should reconstruct WX better than RTN at the same bits (its objective)."""
    from src.interp.compress import gptq_quantize, absmax_quantize
    torch.manual_seed(0)
    n, in_f, out_f = 512, 96, 48
    # correlated activations so the off-diagonal Hessian actually matters
    A = torch.randn(in_f, in_f)
    X = torch.randn(n, in_f) @ A
    W = torch.randn(out_f, in_f)
    H = (X.t() @ X) / n
    Wq_gptq = gptq_quantize(W, H, bits=3, group_size=None)
    Wq_rtn = absmax_quantize(W, 3, group_size=None)
    err_gptq = ((W - Wq_gptq) @ X.t()).pow(2).mean()
    err_rtn = ((W - Wq_rtn) @ X.t()).pow(2).mean()
    assert err_gptq < err_rtn  # GPTQ minimizes exactly this


@pytest.mark.cpu
def test_bits_by_fisher_hits_average():
    from src.interp.compress import bits_by_fisher
    fish = {f"m{i}": float(i) for i in range(100)}
    b = bits_by_fisher(fish, avg_bits=3.0, bit_choices=(2, 4))
    avg = sum(b.values()) / len(b)
    assert abs(avg - 3.0) < 0.1
    # highest-Fisher module gets the high bits, lowest gets low
    assert b["m99"] == 4 and b["m0"] == 2


@pytest.mark.cpu
def test_allocate_layer_bits():
    from src.interp.compress import allocate_layer_bits
    drops = {i: float(i) for i in range(20)}  # layer 19 most sensitive
    lb = allocate_layer_bits(drops, avg_bits=3.5, low=3, high=4)
    assert abs(sum(lb.values()) / len(lb) - 3.5) < 0.1
    assert lb[19] == 4 and lb[0] == 3            # most-sensitive gets high bits
    # avg=low -> all low; avg=high -> all high
    assert set(allocate_layer_bits(drops, 3.0, 3, 4).values()) == {3}
    assert set(allocate_layer_bits(drops, 4.0, 3, 4).values()) == {4}


@pytest.mark.cpu
def test_module_bits_from_layer_bits(tiny_llama):
    from src.interp.compress import module_bits_from_layer_bits
    mb = module_bits_from_layer_bits(tiny_llama, {0: 4, 1: 3}, default=4)
    assert all(v in (3, 4) for v in mb.values())
    assert any(k.startswith("blocks.0.") for k in mb) and any(k.startswith("blocks.1.") for k in mb)


@pytest.mark.cpu
def test_mixed_precision_restores(tiny_llama):
    from src.interp.compress import quantize_mixed_precision
    tiny_model = tiny_llama
    names = [n for n, _ in __import__("src.interp.compress", fromlist=["_iter_named_linears"])
             ._iter_named_linears(tiny_model, [0])]
    bbm = {n: (4 if i % 2 else 2) for i, n in enumerate(names)}
    w0 = next(tiny_model.iter_layer_linears(0)).detach().clone()
    with quantize_mixed_precision(tiny_model, bbm, layers=[0]):
        assert not torch.allclose(next(tiny_model.iter_layer_linears(0)).detach(), w0)
    assert torch.allclose(next(tiny_model.iter_layer_linears(0)).detach(), w0)


@pytest.mark.cpu
def test_gptq_quantize_linears_restores(tiny_llama):
    from src.interp.compress import collect_hessians, gptq_quantize_linears
    tiny_model = tiny_llama
    H = collect_hessians(tiny_model, ["The capital of France is Paris."], layers=[0])
    assert len(H) > 0
    w0 = next(tiny_model.iter_layer_linears(0)).detach().clone()
    with gptq_quantize_linears(tiny_model, 3, H, layers=[0], group_size=None):
        assert not torch.allclose(next(tiny_model.iter_layer_linears(0)).detach(), w0)
    assert torch.allclose(next(tiny_model.iter_layer_linears(0)).detach(), w0)


@pytest.mark.cpu
def test_salience_comparison(tiny_llama):
    tiny_model = tiny_llama
    from src.interp.salient_channels import salience_by_regime, compare_salience
    regimes = {
        "a": ["The capital of France is Paris.", "Cats sit on the mat."],
        "b": ["Le chat est sur le tapis.", "La capitale de la France."],
    }
    sal = salience_by_regime(tiny_model, regimes)
    cmp = compare_salience(sal, "a", "b", top_frac=0.05)
    assert 0.0 <= cmp.mean_jaccard <= 1.0
    assert -1.0 <= cmp.mean_spearman <= 1.0
