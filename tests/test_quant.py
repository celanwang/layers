"""Hand calculations, independent Oracles, scope checks, and heldout gates."""

import mlx.core as mx
import numpy as np
import pytest
import torch

from bench.l5_quant_bench import heldout_loss, load_model, mlx_api_oracle, named_arrays, storage_stats
from layers import GPT, KVCache, QuantizedLinear, dequantize, quantize, quantize_model_
from scripts.train import make_corpus


def test_hand_computed_int4_packing_signed_values_and_group_boundaries():
    # Scales 1 and 2; signed q = [-7,-2,0,7, -7,2,0,7].
    w = mx.array([[-7.0, -2.0, 0.0, 7.0, -14.0, 4.0, 0.0, 14.0]])
    q, scales = quantize(w, 4, 4)
    np.testing.assert_array_equal(np.asarray(q), [[0xE9, 0x70, 0x29, 0x70]])
    np.testing.assert_array_equal(np.asarray(scales), [[1.0, 2.0]])
    np.testing.assert_array_equal(np.asarray(dequantize(q, scales, 4, 4)), np.asarray(w))
    assert q.dtype == mx.uint8 and scales.dtype == mx.float16


def test_hand_computed_int8_per_output_channel_and_even_ties():
    w = mx.array([[-127.0, -1.5, 0.5, 127.0], [-254.0, -3.0, 1.0, 254.0]])
    q, scales = quantize(w, 8)
    np.testing.assert_array_equal(np.asarray(q), [[-127, -2, 0, 127], [-127, -2, 0, 127]])
    np.testing.assert_array_equal(np.asarray(scales), [[1.0], [2.0]])
    np.testing.assert_array_equal(
        np.asarray(dequantize(q, scales, 8)), [[-127, -2, 0, 127], [-254, -4, 0, 254]]
    )
    assert q.dtype == mx.int8


@pytest.mark.parametrize("bits,group", [(8, 128), (8, 32), (4, 64), (4, 2)])
def test_roundtrip_each_group_half_step_bound_and_independent_numpy(bits, group):
    w = np.random.default_rng(19).normal(size=(5, 128)).astype(np.float32)
    w[0] = 0
    w[1, :group] *= 1e-7  # includes fp16 scale underflow/clamping
    w[2, group - 1] = 12.345  # outlier exactly on a group boundary
    q, scales = quantize(mx.array(w), bits, group)
    actual = np.asarray(dequantize(q, scales, bits, group))
    s = np.asarray(scales).astype(np.float32)
    qmax = 2 ** (bits - 1) - 1
    grouped = w.reshape(5, -1, group)
    expected_q = np.clip(np.rint(grouped / s[..., None]), -qmax, qmax)
    expected = (expected_q * s[..., None]).reshape(w.shape)
    np.testing.assert_array_equal(actual, expected)
    # Delta is the stored grid step, not range/(2**bits-1): symmetric uses
    # [-qmax,+qmax] and leaves the most-negative two's-complement code unused.
    errors = np.max(np.abs(w - actual).reshape(5, -1, group), axis=-1)
    assert np.all(errors <= s / 2 + 2e-6)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()


def test_int8_pytorch_public_api_oracle():
    w = np.random.default_rng(4).normal(size=(3, 128)).astype(np.float32)
    q, scales = quantize(mx.array(w), 8)
    oracle = torch.quantize_per_channel(
        torch.from_numpy(w), torch.from_numpy(np.asarray(scales)[:, 0].astype(np.float64)),
        torch.zeros(3, dtype=torch.int64), axis=0, dtype=torch.qint8,
    )
    np.testing.assert_array_equal(np.asarray(q), oracle.int_repr().numpy())
    np.testing.assert_array_equal(np.asarray(dequantize(q, scales, 8)), oracle.dequantize().numpy())


@pytest.mark.parametrize("bits", [8, 4])
def test_subnormal_scale_rounds_up_to_preserve_half_step_endpoint_bound(bits):
    qmax = 2 ** (bits - 1) - 1
    w = np.array([[-qmax * 1.4e-7, qmax * 1.4e-7]], dtype=np.float32)
    q, scales = quantize(mx.array(w), bits, 2)
    actual = np.asarray(dequantize(q, scales, bits, 2))
    step = float(np.asarray(scales)[0, 0])
    assert step >= 1.4e-7
    assert np.max(np.abs(w - actual)) <= step / 2 + 1e-12


def test_mlx_public_api_oracle_and_native_affine_difference():
    result = mlx_api_oracle()
    if result["available"]:
        assert len(result["comparisons"]) == 2
        for row in result["comparisons"]:
            assert row["our_grid_vs_mlx_dequantize_max_abs"] < 1e-6
            assert row["our_vs_native_affine_max_abs"] > 0


@pytest.mark.parametrize("bits", [8, 4])
def test_linear_non_square_orientation_fp32_and_exact_storage(bits):
    weight = mx.array(np.random.default_rng(7).normal(size=(64, 9)).astype(np.float32))
    projection = QuantizedLinear(weight, bits)
    inputs = mx.ones((2, 3, 64), dtype=mx.float16)
    actual = projection(inputs)
    expected = inputs.astype(mx.float32) @ dequantize(
        projection.q, projection.scales, bits, projection.group_size
    ).T
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.shape == (2, 3, 9) and actual.dtype == mx.float32
    assert set(projection.parameters()) == {"q", "scales"}
    assert projection.q.nbytes == 64 * 9 * bits // 8
    assert projection.scales.nbytes == 9 * 2


@pytest.mark.parametrize("bits", [8, 4])
def test_model_scope_protected_arrays_bitwise_unchanged_and_no_fp32_projections(bits):
    model = GPT(vocab_size=19, n_layer=2, d_model=64, n_head=4, ctx=16)
    before = dict(named_arrays(model.parameters()))
    protected = {name: value for name, value in before.items()
                 if ".attn.w" not in name and ".mlp.w" not in name}
    assert quantize_model_(model, bits) is model
    after = dict(named_arrays(model.parameters()))
    for name, value in protected.items():
        assert after[name] is value and after[name].dtype == mx.float32
        np.testing.assert_array_equal(np.asarray(after[name]), np.asarray(value))
    quant_names = [name for name in after if name.endswith(".q")]
    scale_names = [name for name in after if name.endswith(".scales")]
    assert len(quant_names) == len(scale_names) == 2 * 6
    assert set(after) == set(protected) | set(quant_names) | set(scale_names)
    for block in model.blocks:
        for projection in (block.attn.q_proj, block.attn.k_proj, block.attn.v_proj,
                           block.attn.o_proj, block.mlp.up_proj, block.mlp.down_proj):
            assert isinstance(projection, QuantizedLinear)
            assert projection.scales.shape[1] == (1 if bits == 8 else projection.q.shape[1] * 2 // 64)
    with pytest.raises(ValueError, match="already quantized"):
        quantize_model_(model, bits)


@pytest.mark.parametrize("bits,group", [(3, 64), (8, 0), (4, 63), (4, True)])
def test_invalid_configuration_leaves_model_intact(bits, group):
    model = GPT(vocab_size=9, n_layer=1, d_model=64, n_head=4, ctx=8)
    original = model.blocks[0].attn
    with pytest.raises(ValueError):
        quantize_model_(model, bits, group)
    assert model.blocks[0].attn is original


def test_invalid_arrays_are_rejected_and_zero_groups_remain_zero():
    for values in (mx.array([[float("nan"), 1.0]]), mx.array([[float("inf"), 1.0]]),
                   mx.array([[1e10, 1.0]]), mx.ones((2,)), mx.ones((2, 2), dtype=mx.int32)):
        with pytest.raises(ValueError):
            quantize(values, 8)
    for bits in (4, 8):
        q, scales = quantize(mx.zeros((2, 64)), bits)
        np.testing.assert_array_equal(np.asarray(dequantize(q, scales, bits)), np.zeros((2, 64)))
        with pytest.raises(ValueError, match="scales"):
            dequantize(q, scales.astype(mx.float32), bits)
        with pytest.raises(ValueError, match="dtype"):
            dequantize(q.astype(mx.int32), scales, bits)


def test_checkpoint_logits_error_order_and_quantized_cached_forward():
    ids = mx.array([1, 14, 21, 99, 72, 3, 5, 7], dtype=mx.int32)
    reference = np.asarray(load_model()(ids))
    errors = []
    for bits in (8, 4):
        model = quantize_model_(load_model(), bits)
        logits = np.asarray(model(ids))
        assert np.isfinite(logits).all()
        errors.append(float(np.sqrt(np.mean((logits - reference) ** 2))))
        cache = KVCache(model.n_layer)
        for position in range(len(ids)):
            cached = np.asarray(model(ids[position:position + 1], cache=cache))
            assert np.isfinite(cached).all()
            np.testing.assert_allclose(cached[-1], logits[position], atol=1e-5, rtol=1e-5)
    assert 0 < errors[0] < errors[1] < 0.5


def test_checkpoint_entire_heldout_quality_thresholds_and_storage_accounting():
    _tokenizer, train, heldout, corpus = make_corpus()
    assert len(train) == corpus["total_tokens"] * 9 // 10
    assert len(heldout) == corpus["total_tokens"] - len(train)
    losses, storage = [], []
    for bits in (None, 8, 4):
        model = load_model()
        if bits is not None:
            quantize_model_(model, bits)
        losses.append(heldout_loss(model, heldout))
        storage.append(storage_stats(model, bits is not None))
    assert np.isfinite(losses).all()
    assert losses[1] / losses[0] - 1 < 0.01, losses
    assert losses[2] / losses[0] - 1 < 0.05, losses
    assert storage[0]["serialized_npz_bytes"] > storage[1]["serialized_npz_bytes"] > storage[2]["serialized_npz_bytes"]
    assert len({row["protected_fp32_bytes"] for row in storage}) == 1
    projection_fp32 = storage[0]["projection_payload_bytes"]
    assert storage[1]["integer_bytes"] == projection_fp32 // 4
    assert storage[2]["integer_bytes"] == projection_fp32 // 8
    assert storage[2]["fp16_scale_bytes"] == projection_fp32 // 4 // 64 * 2
