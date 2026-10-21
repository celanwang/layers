"""L2 checks against hand calculations and public cross-framework oracles."""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import pytest
import torch

from layers import BPETokenizer, CausalSelfAttention, RotaryEmbedding, TokenEmbedding
from layers.attention import _stable_softmax, attention_scores


ROOT = Path(__file__).resolve().parents[1]


def as_numpy(value):
    return np.asarray(value, dtype=np.float32)


def numpy_rope(x, base):
    """Independent small-array RoPE definition for the hand calculation."""
    result = np.array(x, dtype=np.float32, copy=True)
    dims = result.shape[-1]
    for position in range(result.shape[-2]):
        for pair in range(dims // 2):
            angle = position * base ** (-2 * pair / dims)
            even = result[..., position, 2 * pair].copy()
            odd = result[..., position, 2 * pair + 1].copy()
            result[..., position, 2 * pair] = even * math.cos(angle) - odd * math.sin(angle)
            result[..., position, 2 * pair + 1] = even * math.sin(angle) + odd * math.cos(angle)
    return result


def test_hand_computed_three_token_attention_full_chain():
    model = CausalSelfAttention(d_model=2, n_head=1, d_head=2, seed=42)
    identity = mx.eye(2)
    model.wq = identity
    model.wk = identity
    model.wv = identity
    model.wo = identity
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    out, weights, actual = model.forward_with_intermediates(mx.array(x))

    q = x[None, :, :]
    k = x[None, :, :]
    v = x[None, :, :]
    rotary_q = numpy_rope(q, 10000.0)
    rotary_k = numpy_rope(k, 10000.0)
    scores = rotary_q @ np.swapaxes(rotary_k, -1, -2) / math.sqrt(2.0)
    scores = scores[0]
    scores[np.triu_indices(3, k=1)] = -np.inf
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    expected_out = (probs @ v[0])

    np.testing.assert_allclose(as_numpy(actual["q"]), q, rtol=0, atol=0)
    np.testing.assert_allclose(as_numpy(actual["k"]), k, rtol=0, atol=0)
    np.testing.assert_allclose(as_numpy(actual["v"]), v, rtol=0, atol=0)
    np.testing.assert_allclose(as_numpy(actual["rotary_q"]), rotary_q, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(as_numpy(actual["rotary_k"]), rotary_k, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(as_numpy(actual["scaled_scores"]), scores[None, ...], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(as_numpy(actual["probs"]), probs[None, ...], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(as_numpy(weights), probs[None, ...], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(as_numpy(actual["out"]), expected_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(as_numpy(out), expected_out, rtol=1e-6, atol=1e-6)


def test_causal_future_change_does_not_change_earlier_outputs():
    mx.random.seed(7)
    model = CausalSelfAttention(d_model=16, n_head=4, d_head=4, seed=42)
    x = mx.random.normal((7, 16))
    changed = x.at[5].add(mx.ones((16,)) * 100.0)
    original_out, _ = model(x)
    changed_out, _ = model(changed)
    mx.eval(original_out, changed_out)
    np.testing.assert_array_equal(as_numpy(original_out[:5]), as_numpy(changed_out[:5]))


def test_rope_quarter_turn_and_relative_position_identity():
    rope = RotaryEmbedding(dims=2, base=10000.0)
    quarter_turn = rope(mx.array([[1.0, 0.0]]), offset=math.pi / 2)
    np.testing.assert_allclose(as_numpy(quarter_turn), [[0.0, 1.0]], rtol=0, atol=1e-6)

    dims = 8
    basis = mx.eye(dims)[:, None, :]
    m, n = 7, 19
    at_m = as_numpy(RotaryEmbedding(dims=dims)(basis, offset=m)[:, 0, :])
    at_n = as_numpy(RotaryEmbedding(dims=dims)(basis, offset=n)[:, 0, :])
    at_difference = as_numpy(RotaryEmbedding(dims=dims)(basis, offset=n - m)[:, 0, :])
    np.testing.assert_allclose(at_m @ at_n.T, at_difference.T, rtol=1e-6, atol=1e-6)


def torch_rope(x, base):
    dims = x.shape[-1]
    positions = torch.arange(x.shape[-2], dtype=torch.float32)
    pair_indices = torch.arange(0, dims, 2, dtype=torch.float32)
    inv_freq = torch.exp(-pair_indices * (math.log(base) / dims))
    angles = positions[:, None] * inv_freq[None, :]
    angles = angles.reshape((1,) * (x.ndim - 2) + (x.shape[-2], dims // 2))
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * torch.cos(angles) - odd * torch.sin(angles),
                        even * torch.sin(angles) + odd * torch.cos(angles)), dim=-1).reshape(x.shape)


def test_pytorch_oracle_matches_each_attention_intermediate():
    d_model, n_head, d_head, length = 24, 3, 8, 9
    mx.random.seed(123)
    model = CausalSelfAttention(d_model=d_model, n_head=n_head, d_head=d_head, seed=42)
    x_np = np.random.default_rng(20261021).normal(size=(length, d_model)).astype(np.float32)
    out, _, mlx_values = model.forward_with_intermediates(mx.array(x_np))

    weights = {
        name: torch.from_numpy(np.array(getattr(model, name), dtype=np.float32, copy=True))
        for name in ("wq", "wk", "wv", "wo")
    }
    x_torch = torch.from_numpy(x_np[None, ...])
    q = (x_torch @ weights["wq"]).reshape(1, length, n_head, d_head).transpose(1, 2)
    k = (x_torch @ weights["wk"]).reshape(1, length, n_head, d_head).transpose(1, 2)
    v = (x_torch @ weights["wv"]).reshape(1, length, n_head, d_head).transpose(1, 2)
    rotary_q, rotary_k = torch_rope(q, 10000.0), torch_rope(k, 10000.0)
    scaled_scores = rotary_q @ rotary_k.transpose(-1, -2) / math.sqrt(d_head)
    future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    scaled_scores = scaled_scores.masked_fill(future, -torch.inf)
    probs = torch.softmax(scaled_scores, dim=-1)
    context = (probs @ v).transpose(1, 2).contiguous().reshape(1, length, d_model)
    expected_out = context @ weights["wo"]

    torch_values = {
        "q": q[0],
        "k": k[0],
        "v": v[0],
        "scaled_scores": scaled_scores[0],
        "probs": probs[0],
        "out": expected_out[0],
    }
    for name, expected in torch_values.items():
        np.testing.assert_allclose(
            as_numpy(mlx_values[name]), expected.detach().numpy(), rtol=1e-5, atol=1e-5
        )
    np.testing.assert_allclose(as_numpy(out), expected_out[0].detach().numpy(), rtol=1e-5, atol=1e-5)


def test_mlx_fast_attention_and_rope_public_oracles():
    mx.random.seed(2026)
    q = mx.random.normal((1, 2, 31, 16))
    k = mx.random.normal((1, 2, 31, 16))
    v = mx.random.normal((1, 2, 31, 16))
    own_scores = attention_scores(q, k)
    own = _stable_softmax(own_scores) @ v
    fast = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask="causal"
    )
    mx.eval(own, fast)
    np.testing.assert_allclose(as_numpy(own), as_numpy(fast), rtol=1e-5, atol=1e-5)

    rope_input = mx.random.normal((2, 31, 16))
    own_rope = RotaryEmbedding(dims=16, base=10000.0)(rope_input)
    oracle_rope = mlx_nn.RoPE(dims=16, traditional=True, base=10000.0)(rope_input)
    mx.eval(own_rope, oracle_rope)
    np.testing.assert_allclose(as_numpy(own_rope), as_numpy(oracle_rope), rtol=1e-5, atol=1e-5)


def test_l1_tokenizer_embedding_attention_end_to_end():
    tokenizer = BPETokenizer.load(str(ROOT / "results/l1/tokenizer-1024.json"))
    token_ids = tokenizer.encode("Attention connects tokens using their context.")
    embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size + 4, d_model=512, seed=42)
    attention = CausalSelfAttention(d_model=512, n_head=8, d_head=64, seed=42)
    embedded = embedding(mx.array(token_ids))
    out, weights = attention(embedded)
    mx.eval(embedded, out, weights)
    assert embedded.shape == (len(token_ids), 512)
    assert out.shape == (len(token_ids), 512)
    assert weights.shape == (8, len(token_ids), len(token_ids))
    assert np.isfinite(as_numpy(embedded)).all()
    assert np.isfinite(as_numpy(out)).all()
    assert np.isfinite(as_numpy(weights)).all()


def test_embedding_is_a_seeded_lookup_table():
    first = TokenEmbedding(vocab_size=13, d_model=5, seed=42)
    second = TokenEmbedding(vocab_size=13, d_model=5, seed=42)
    ids = mx.array([9, 2, 9])
    values = first(ids)
    mx.eval(values, second.weight)
    np.testing.assert_array_equal(as_numpy(values), as_numpy(first.weight[[9, 2, 9]]))
    np.testing.assert_array_equal(as_numpy(first.weight), as_numpy(second.weight))


@pytest.mark.parametrize("dims", [0, 3])
def test_rope_rejects_invalid_rotary_dimensions(dims):
    with pytest.raises(ValueError):
        RotaryEmbedding(dims=dims)
