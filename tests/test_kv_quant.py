"""Per-token KV grid, append invariants, and L4 injection behavior."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

from bench.l5_kv_bf16_bench import logit_comparison
from layers import BPETokenizer, GPT, Sampler
from layers.inference import KVCache, QuantKVCache, generate
from layers.quant import dequantize_kv, quantize_kv

ROOT = Path(__file__).resolve().parents[1]


def test_hand_computed_kv_grid_all_heads_share_one_scale():
    x = mx.array([[[[127., -63.5], [2., -1.]], [[0.5, -127.], [-254., 0.]]]])
    q, scales = quantize_kv(x)
    np.testing.assert_array_equal(np.asarray(scales), [[[[1.], [2.]]]])
    np.testing.assert_array_equal(np.asarray(q), [[[[127, -64], [1, 0]], [[0, -127], [-127, 0]]]])
    reconstructed = dequantize_kv(q, scales)
    assert reconstructed.dtype == mx.float32
    assert bool(mx.all(mx.abs(reconstructed - x) <= scales.astype(mx.float32) / 2).item())


def test_kv_roundtrip_bound_and_pytorch_public_api_oracle():
    values = np.random.default_rng(7).normal(size=(2, 3, 7, 4)).astype(np.float32)
    values[0, :, 1, :] = 0
    values[1, 2, 4, 0] = 100
    q, scales = quantize_kv(mx.array(values))
    restored = np.asarray(dequantize_kv(q, scales))
    s32 = np.asarray(scales).astype(np.float32)
    assert np.all(np.abs(restored - values) <= s32 / 2 + 1e-7)
    tokens = values.transpose(0, 2, 1, 3).reshape(14, 12)
    oracle = torch.quantize_per_channel(
        torch.from_numpy(tokens.copy()), torch.from_numpy(s32.reshape(-1).copy()),
        torch.zeros(14, dtype=torch.int64), axis=0, dtype=torch.qint8,
    )
    actual = np.asarray(q).transpose(0, 2, 1, 3).reshape(14, 12)
    np.testing.assert_array_equal(actual, oracle.int_repr().numpy())
    np.testing.assert_allclose(restored.transpose(0, 2, 1, 3).reshape(14, 12),
                               oracle.dequantize().numpy(), rtol=0, atol=0)


def test_append_get_lengths_storage_and_old_tokens_are_immutable():
    fp, quant = KVCache(2), QuantKVCache(2)
    assert fp.seq_len == quant.seq_len == 0
    assert quant.get(0) == (None, None)
    rng = np.random.default_rng(13)
    old_keys = None
    for length in (3, 1, 2):
        for layer in range(2):
            keys = mx.array(rng.normal(size=(2, 3, length, 4)).astype(np.float32))
            values = mx.array(rng.normal(size=(2, 3, length, 6)).astype(np.float32))
            a, b = fp.append(layer, keys, values), quant.append(layer, keys, values)
            for raw, restored, scale in zip(a, b, (quant._key_scales[layer], quant._value_scales[layer])):
                assert raw.shape == restored.shape
                assert bool(mx.all(mx.abs(raw - restored) <= scale.astype(mx.float32) / 2 + 1e-7).item())
            np.testing.assert_array_equal(np.asarray(b[0]), np.asarray(quant.get(layer)[0]))
        if old_keys is not None:
            np.testing.assert_array_equal(np.asarray(quant.get(0)[0])[:, :, :old_keys.shape[2]], old_keys)
        old_keys = np.asarray(quant.get(0)[0]).copy()
        assert quant.seq_len == fp.seq_len
        if length == 3:
            quant.append(0, keys[:, :, :1], values[:, :, :1])
            with pytest.raises(RuntimeError, match="different sequence lengths"):
                _ = quant.seq_len
            # Restore alignment by the same additional append in both caches.
            quant.append(1, keys[:, :, :1], values[:, :, :1])
            fp.append(0, keys[:, :, :1], values[:, :, :1])
            fp.append(1, keys[:, :, :1], values[:, :, :1])
            old_keys = np.asarray(quant.get(0)[0]).copy()
    expected = 2 * 2 * quant.seq_len * (3 * (4 + 6) + 4)
    assert quant.storage_nbytes == expected
    assert all(x.dtype == mx.int8 for x in quant._keys + quant._values)


@pytest.mark.parametrize("cache_type", [KVCache, QuantKVCache])
def test_cache_rejects_shape_changes_and_bad_layer_without_mutation(cache_type):
    cache = cache_type(1)
    keys = mx.ones((1, 2, 3, 4))
    cache.append(0, keys, keys)
    for bad in (mx.ones((2, 2, 1, 4)), mx.ones((1, 3, 1, 4)), mx.ones((1, 2, 1, 6))):
        with pytest.raises(ValueError):
            cache.append(0, bad, bad)
        assert cache.seq_len == 3
    for layer in (-1, 1, True):
        with pytest.raises(ValueError):
            cache.get(layer)
    with pytest.raises(ValueError):
        cache.append(0, keys[0], keys)


def test_invalid_kv_values_do_not_partially_append():
    cache = QuantKVCache(1)
    with pytest.raises(ValueError, match="finite"):
        cache.append(0, mx.ones((1, 2, 1, 4)), mx.full((1, 2, 1, 4), float("nan")))
    assert cache.get(0) == (None, None)


def test_generate_accepts_either_cache_and_default_fp32_is_bitwise_equal():
    model = GPT(vocab_size=256, n_layer=2, d_model=16, n_head=2, ctx=32, seed=42)
    tokenizer = BPETokenizer()
    options = dict(model=model, tokenizer=tokenizer, prompt=[65, 66], max_new=8)
    default = "".join(generate(**options, sampler=Sampler(temperature=0)))
    fp = KVCache(2)
    assert "".join(generate(**options, sampler=Sampler(temperature=0), cache=fp)) == default
    quant = QuantKVCache(2)
    assert isinstance("".join(generate(**options, sampler=Sampler(temperature=0), cache=quant)), str)
    assert fp.seq_len == quant.seq_len == 9


def test_logit_error_curve_finite_and_recorded():
    model = GPT(vocab_size=32, n_layer=2, d_model=16, n_head=2, ctx=32, seed=42)
    record = logit_comparison(model, [1, 4, 9, 2], steps=16)
    errors = [row["max_abs_diff"] for row in record["curve"]]
    assert len(errors) == 17 and np.isfinite(errors).all()
    envelope = [row["running_max_abs_diff"] for row in record["curve"]]
    np.testing.assert_array_equal(envelope, np.maximum.accumulate(errors))
    path = ROOT / "results/l5/t2-kv-test-logits.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
