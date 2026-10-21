"""L7 page materialization and engine equivalence to the immutable L6 oracle."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from layers import BatchedEngine, GPT, PagedAttentionEngine, RequestCapacityError, Sampler
from layers.engine import _BatchedPagedKVCache, _DirectPagedKVCache, _decode_paged_batch
from layers.paged_kv import PagedKVCache, PagedKVPool


def numpy(value):
    mx.eval(value)
    return np.asarray(value, dtype=np.float32)


def model():
    return GPT(vocab_size=67, n_layer=2, d_model=32, n_head=4, ctx=128, seed=83)


class RecordingSampler:
    def __init__(self, seed, temperature=0.8):
        self.sampler = Sampler(temperature=temperature, top_k=23, top_p=0.95, seed=seed)
        self.logits = []

    def __call__(self, logits):
        self.logits.append(numpy(logits))
        return self.sampler(logits)


def test_materialization_is_one_index_for_both_kv_and_masks_stale_fork_tails(monkeypatch):
    pool = PagedKVPool(2, block_size=4, max_context=32, max_sequences=4)
    parent, short = pool.alloc(), pool.alloc()
    keys = mx.arange(9 * 6, dtype=mx.float32).reshape(1, 2, 9, 3)
    pool.append_pages(parent, keys, -keys, 0)
    pool.append_pages(short, keys[..., :1, :], -keys[..., :1, :], 0)
    child = pool.fork(parent, prefix_len=0)
    # fork requires all layers to contain the prefix.
    pool.append_pages(parent, keys, -keys, 1)
    pool.free(child)
    child = pool.fork(parent, prefix_len=5)
    empty = pool.alloc()
    order = [child, short, parent, empty]
    references = [pool.gather(seq_id, 0) for seq_id in order]
    original_take, calls = mx.take, []

    def take(*args, **kwargs):
        calls.append(1)
        return original_take(*args, **kwargs)

    monkeypatch.setattr(mx, "take", take)
    monkeypatch.setattr(pool, "gather", lambda *args: pytest.fail("row gather called"))
    actual_k, actual_v, lengths = pool.materialize_batch(order, 0)
    assert calls == [1]
    assert lengths == (5, 1, 9, 0)
    assert actual_k.shape == actual_v.shape == (4, 2, 12, 3)
    for row, (reference_k, reference_v) in enumerate(references[:-1]):
        np.testing.assert_array_equal(numpy(actual_k)[row:row + 1, :, :lengths[row]], numpy(reference_k))
        np.testing.assert_array_equal(numpy(actual_v)[row:row + 1, :, :lengths[row]], numpy(reference_v))
    assert pool.allocated_blocks == 4  # shared pages never duplicated by materialization
    for seq_id in order:
        pool.free(seq_id)
    assert pool.allocated_blocks == 0


def test_write_only_append_preserves_cow_and_lazy_layer_lifecycle():
    pool = PagedKVPool(2, block_size=4, max_context=16, max_sequences=3)
    seq_id = pool.alloc()
    assert pool.materialize_batch([seq_id], 1) == (None, None, (0,))
    k = mx.ones((1, 2, 5, 4))
    for layer in range(2):
        pool.append_pages(seq_id, k, k * 2, layer)
    child = pool.fork(seq_id, prefix_len=3)
    shared = pool.block_table(seq_id)[0]
    pool.append_pages(child, k[..., :1, :] * 7, k[..., :1, :] * 9, 0)
    assert pool.refcount(shared) == 1
    assert pool.block_table(child)[0] != shared
    assert pool._state(child).layer_lengths == [4, 3]
    actual_k, actual_v, lengths = pool.materialize_batch([child, seq_id], 0)
    assert lengths == (4, 5)
    np.testing.assert_array_equal(numpy(actual_k)[0, :, 3], 7)
    np.testing.assert_array_equal(numpy(actual_v)[0, :, 3], 9)
    np.testing.assert_array_equal(numpy(pool.gather(seq_id, 0)[0]), numpy(k))
    # The CoW copied other layers too; no eager layer-1 write took place.
    _, _, lengths = pool.materialize_batch([child, seq_id], 1)
    assert lengths == (3, 5)
    pool.free(seq_id)
    pool.free(child)
    assert pool.active_sequences == pool.allocated_blocks == 0


def test_new_interfaces_reject_invalid_inputs_and_missing_payloads():
    pool = PagedKVPool(1, block_size=4)
    seq_id = pool.alloc()
    k = mx.ones((1, 2, 1, 4))
    for layer in (-1, 1):
        with pytest.raises(ValueError, match="layer_idx"):
            pool.append_pages(seq_id, k, k, layer)
        with pytest.raises(ValueError, match="layer_idx"):
            pool.materialize_batch([seq_id], layer)
    for ids in ([], [seq_id, seq_id]):
        with pytest.raises(ValueError, match="distinct"):
            pool.materialize_batch(ids)
        with pytest.raises(ValueError, match="distinct"):
            _DirectPagedKVCache(pool, ids)
    with pytest.raises(KeyError, match="unknown"):
        pool.materialize_batch([999])
    with pytest.raises(ValueError, match="shape"):
        pool.append_pages(seq_id, mx.ones((2, 2)), mx.ones((2, 2)))
    assert pool.seq_len(seq_id) == 0
    pool.append_pages(seq_id, k, k)
    with pytest.raises(ValueError, match="changed"):
        pool.append_pages(seq_id, mx.ones((1, 3, 1, 4)), mx.ones((1, 3, 1, 4)))
    page = pool._pages[pool.block_table(seq_id)[0]]
    page.layers.clear()
    with pytest.raises(RuntimeError, match="missing"):
        pool.materialize_batch([seq_id])


@pytest.mark.parametrize("temperature", [0, 0.8])
@pytest.mark.parametrize("lengths", [(7,), (7,) * 8, (1, 2, 4, 7, 11, 16, 23, 31)])
def test_equal_unequal_and_singleton_tokens_logits_match_l6(temperature, lengths):
    gpt = model()
    left = PagedAttentionEngine(gpt, block_size=4, max_concurrent=len(lengths), share_prefixes=False)
    right = BatchedEngine(gpt, block_size=4, max_concurrent=len(lengths), share_prefixes=False)
    samplers = []
    for row, length in enumerate(lengths):
        a, b = RecordingSampler(100 + row, temperature), RecordingSampler(100 + row, temperature)
        prompt = ((np.arange(length) + row * 3) % gpt.vocab_size).tolist()
        assert left.add_request(prompt, a, 33) == right.add_request(prompt, b, 33)
        samplers.append((a, b))
    for _ in range(33):
        assert left.step() == right.step()
        for a, b in samplers:
            np.testing.assert_allclose(a.logits[-1], b.logits[-1], atol=1e-5, rtol=0)
    assert left.step() == right.step() == []
    assert left.pool.allocated_blocks == right.pool.allocated_blocks == 0


def test_dynamic_arrivals_fork_cancel_completion_and_reuse_match_l6():
    gpt = model()
    engines = [PagedAttentionEngine(gpt, block_size=4, max_concurrent=4),
               BatchedEngine(gpt, block_size=4, max_concurrent=4)]
    samplers, live = {}, set()

    def arrive(prompt, count, seed):
        pair = [RecordingSampler(seed), RecordingSampler(seed)]
        ids = [engine.add_request(prompt, sampler, count) for engine, sampler in zip(engines, pair)]
        assert ids[0] == ids[1]
        samplers[ids[0]] = pair
        live.add(ids[0])
        return ids[0]

    parent = arrive([1, 2, 3, 4, 5], 9, 31)
    child = arrive([1, 2, 3, 4, 5], 2, 32)
    for engine in engines:
        assert engine.pool.refcount(engine.pool.block_table(parent)[0]) == 2
    for iteration in range(13):
        if iteration == 2:
            arrive([1, 2, 3, 4, 6, 7], 5, 33)
            cancelled = arrive([51], 8, 34)
        if iteration == 3:
            for engine in engines:
                engine.cancel_request(cancelled)
                engine.cancel_request(cancelled)
                engine.cancel_request(child)  # completed history can also be removed
            live.remove(cancelled)
        if iteration == 5:
            arrive([41, 42, 43], 6, 35)
        a, b = (engine.step() for engine in engines)
        assert a == b
        for event in a:
            left, right = samplers[event.seq_id]
            np.testing.assert_allclose(left.logits[-1], right.logits[-1], atol=1e-5, rtol=0)
            if event.finished:
                assert engines[0].generated_tokens(event.seq_id) == engines[1].generated_tokens(event.seq_id)
                live.remove(event.seq_id)
        assert all(engine.pool.active_sequences == len(live) for engine in engines)
    assert not live
    assert all(engine.pool.allocated_blocks == 0 for engine in engines)


def test_direct_attention_masks_all_padding_and_permuted_rows_match_l6(monkeypatch):
    gpt = model()
    pool = PagedKVPool(gpt.n_layer, block_size=4, max_context=128, max_sequences=6)
    seq_ids = [pool.alloc() for _ in range(3)]
    lengths = (1, 5, 13)
    for seq_id, length in zip(seq_ids, lengths):
        mx.eval(gpt(mx.array(list(range(length))), cache=PagedKVCache(pool, seq_id)))
    clones = [pool.fork(seq_id) for seq_id in seq_ids]
    ids = mx.array([[2], [9], [17]])
    x = gpt.blocks[0].ln_attn(gpt.token_embedding(ids))
    expected, old_probs, _ = gpt.blocks[0].attn.decode_batch_with_intermediates(
        x, _BatchedPagedKVCache(pool, seq_ids), 0
    )
    order = [2, 0, 1]
    monkeypatch.setattr(pool, "gather", lambda *args: pytest.fail("legacy gather called"))
    actual, probs, details = gpt.blocks[0].attn.decode_paged_batch_with_intermediates(
        x[mx.array(order)], _DirectPagedKVCache(pool, [clones[row] for row in order]), 0
    )
    np.testing.assert_allclose(numpy(actual), numpy(expected)[order], atol=1e-5, rtol=0)
    assert probs.shape == (3, gpt.n_head, 1, 16)
    for row, source in enumerate(order):
        length = lengths[source] + 1
        np.testing.assert_allclose(numpy(probs)[row, ..., :length], numpy(old_probs)[source, ..., :length], atol=1e-7, rtol=0)
        assert np.count_nonzero(numpy(probs)[row, ..., length:]) == 0
        assert np.all(np.isneginf(numpy(details["scaled_scores"])[row, ..., length:]))
    for seq_id in seq_ids + clones:
        pool.free(seq_id)
    assert pool.allocated_blocks == 0


def test_new_engine_decode_never_calls_legacy_append_or_gather(monkeypatch):
    engine = PagedAttentionEngine(model(), block_size=4, max_concurrent=2)
    for prompt in ([1, 2, 3], [9]):
        engine.add_request(prompt, Sampler(temperature=0), 5)
    monkeypatch.setattr(engine.pool, "append", lambda *args, **kwargs: pytest.fail("legacy append called"))
    monkeypatch.setattr(engine.pool, "gather", lambda *args: pytest.fail("legacy gather called"))
    assert len(engine.run()) == 10
    assert engine.pool.allocated_blocks == 0


def test_capacity_context_invalid_batch_and_reclamation_match_l6():
    gpt = model()
    for engine_type in (BatchedEngine, PagedAttentionEngine):
        pool = PagedKVPool(gpt.n_layer, block_size=4, max_context=8, max_sequences=2, num_blocks=2)
        engine = engine_type(gpt, pool=pool, max_context=8, max_concurrent=2)
        seq_id = engine.add_request([1, 2, 3, 4, 5, 6, 7], Sampler(temperature=0), 12)
        with pytest.raises(RequestCapacityError, match="reservation"):
            engine.add_request([2], Sampler(), 2)
        event = engine.step()[0]
        assert event.finished and len(engine.generated_tokens(seq_id)) == 1
        assert pool.allocated_blocks == 0
    engine = PagedAttentionEngine(gpt, max_concurrent=1)
    seq_id = engine.add_request([3], Sampler(), 2)
    with pytest.raises(RequestCapacityError, match="active request limit"):
        engine.add_request([4], Sampler(), 2)
    cache = _DirectPagedKVCache(engine.pool, [seq_id])
    with pytest.raises(ValueError, match="shape"):
        _decode_paged_batch(gpt, mx.array([[1, 2]]), cache)
    with pytest.raises(ValueError, match="shape"):
        gpt.blocks[0].attn.decode_paged_batch(mx.zeros((1, 2, gpt.d_model)), cache, 0)
    cache.lengths = (gpt.ctx,)
    with pytest.raises(ValueError, match="exceed ctx"):
        _decode_paged_batch(gpt, mx.array([[1]]), cache)
    assert engine.pool.seq_len(seq_id) == 1
    engine.cancel_request(seq_id)
    assert engine.pool.allocated_blocks == 0


def test_trained_d512_checkpoint_logits_and_seeded_tokens_match_l6():
    from bench.l6_batching_bench import load_model

    gpt = load_model()
    engines = [PagedAttentionEngine(gpt, block_size=4, max_concurrent=8),
               BatchedEngine(gpt, block_size=4, max_concurrent=8)]
    samplers = []
    for row, length in enumerate((1, 2, 4, 7, 11, 16, 23, 31)):
        pair = [RecordingSampler(500 + row), RecordingSampler(500 + row)]
        prompt = ((np.arange(length) + row * 29) % gpt.vocab_size).tolist()
        for engine, sampler in zip(engines, pair):
            engine.add_request(prompt, sampler, 33)
        samplers.append(pair)
    max_error = 0.0
    for _ in range(33):
        assert engines[0].step() == engines[1].step()
        for a, b in samplers:
            max_error = max(max_error, float(np.max(np.abs(a.logits[-1] - b.logits[-1]))))
            np.testing.assert_allclose(a.logits[-1], b.logits[-1], atol=1e-5, rtol=0)
    assert all(engine.pool.allocated_blocks == 0 for engine in engines)
    path = Path(__file__).resolve().parents[1] / "results/l7/checkpoint-oracle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "oracle": "unchanged L6 BatchedEngine", "batch_size": 8,
        "sampled_tokens_per_request": 33, "decode_forwards_per_request": 32,
        "atol": 1e-5, "rtol": 0, "max_absolute_logit_error": max_error,
        "seeded_tokens_equal": True, "all_pages_reclaimed": True,
    }, indent=2) + "\n")
