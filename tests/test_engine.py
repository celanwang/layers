"""L4-T2 page-pool invariants and iteration-level scheduler checks."""

import mlx.core as mx
import numpy as np
import pytest

from layers import Engine, GPT, KVCache, RequestCapacityError, Sampler
from layers.paged_kv import PagedKVCache, PagedKVPool, PoolExhaustedError


def tiny_model(ctx=64):
    return GPT(vocab_size=32, n_layer=2, d_model=16, n_head=4, ctx=ctx, seed=71)


def numpy(value):
    mx.eval(value)
    return np.asarray(value, dtype=np.float32)


def kv(values, width=4):
    keys = np.asarray(values, dtype=np.float32).reshape(1, 1, -1, 1)
    values = np.asarray(values, dtype=np.float32).reshape(1, 1, -1, 1) + 100
    return mx.array(np.repeat(keys, width, axis=-1)), mx.array(
        np.repeat(values, width, axis=-1)
    )


def test_interleaved_paged_logits_match_t1_cache_without_cross_talk():
    model = tiny_model()
    pool = PagedKVPool(num_layers=model.n_layer, block_size=4, max_context=64)
    prompts = ([1, 2, 3, 4, 5], [9, 8, 7])
    decode_tokens = ([12, 13, 14], [20, 21, 22])
    sequence_ids = [pool.alloc(), pool.alloc()]
    direct_caches = [KVCache(model.n_layer), KVCache(model.n_layer)]

    for index, prompt in enumerate(prompts):
        token_ids = mx.array(prompt, dtype=mx.int32)
        oracle = model(token_ids, cache=direct_caches[index])
        paged = model(
            token_ids,
            cache=PagedKVCache(pool, sequence_ids[index]),
        )
        np.testing.assert_allclose(numpy(paged), numpy(oracle), rtol=1e-5, atol=1e-5)

    for step in range(3):
        for index in (0, 1):
            token = mx.array([decode_tokens[index][step]], dtype=mx.int32)
            oracle = model(token, cache=direct_caches[index])
            paged = model(token, cache=PagedKVCache(pool, sequence_ids[index]))
            np.testing.assert_allclose(
                numpy(paged[-1]), numpy(oracle[-1]), rtol=1e-5, atol=1e-5
            )

    assert pool.seq_len(sequence_ids[0]) == len(prompts[0]) + 3
    assert pool.seq_len(sequence_ids[1]) == len(prompts[1]) + 3
    pool.free(sequence_ids[0])
    pool.free(sequence_ids[1])
    assert pool.num_free_blocks == pool.capacity


def test_fork_copy_on_write_keeps_parent_and_sibling_branches_isolated():
    pool = PagedKVPool(
        num_layers=1, block_size=4, max_context=16, max_sequences=4, num_blocks=8
    )
    parent = pool.alloc()
    keys, values = kv([1, 2, 3], width=2)
    pool.append(parent, keys, values)
    parent_before = tuple(numpy(value) for value in pool.gather(parent))
    child_a = pool.fork(parent)
    child_b = pool.fork(parent)
    shared_block = pool.block_table(parent)[0]
    assert pool.refcount(shared_block) == 3

    keys_a, values_a = kv([11], width=2)
    keys_b, values_b = kv([21], width=2)
    pool.append(child_a, keys_a, values_a)
    pool.append(child_b, keys_b, values_b)

    parent_after = tuple(numpy(value) for value in pool.gather(parent))
    np.testing.assert_array_equal(parent_after[0], parent_before[0])
    np.testing.assert_array_equal(parent_after[1], parent_before[1])
    gathered_a = tuple(numpy(value) for value in pool.gather(child_a))
    gathered_b = tuple(numpy(value) for value in pool.gather(child_b))
    np.testing.assert_array_equal(gathered_a[0][..., :3, :], parent_before[0])
    np.testing.assert_array_equal(gathered_b[0][..., :3, :], parent_before[0])
    assert not np.array_equal(gathered_a[0][..., -1, :], gathered_b[0][..., -1, :])
    assert pool.block_table(child_a)[0] != shared_block
    assert pool.block_table(child_b)[0] != shared_block

    pool.free(child_a)
    pool.free(child_b)
    pool.free(parent)
    assert pool.num_free_blocks == pool.capacity


def test_repeated_alloc_append_fork_free_returns_every_page():
    pool = PagedKVPool(
        num_layers=1, block_size=4, max_context=32, max_sequences=4, num_blocks=8
    )
    initial_free = pool.num_free_blocks
    keys, values = kv(range(20), width=2)
    tail_k, tail_v = kv([90], width=2)
    for _ in range(12):
        parent = pool.alloc()
        pool.append(parent, keys, values)
        child = pool.fork(parent)
        pool.append(child, tail_k, tail_v)
        pool.free(parent)
        pool.free(child)
        assert pool.num_free_blocks == initial_free
    assert pool.active_sequences == 0


def test_fragmentation_stays_within_average_half_block_bound():
    rng = np.random.default_rng(20261021)
    lengths = rng.integers(32, 257, size=16).tolist()
    pool = PagedKVPool(
        num_layers=1, block_size=16, max_context=256, max_sequences=16
    )
    for length in lengths:
        seq_id = pool.alloc()
        keys = mx.zeros((1, 2, length, 4), dtype=mx.float32)
        values = mx.ones((1, 2, length, 4), dtype=mx.float32)
        pool.append(seq_id, keys, values)
    upper_bound = len(lengths) * (pool.block_size // 2) * pool.bytes_per_token
    assert pool.wasted_bytes <= upper_bound


def test_continuous_schedule_matches_independent_request_tokens():
    model = tiny_model(ctx=48)
    prompts = ([1, 2, 3, 4],) * 4
    continuous = Engine(
        model,
        max_context=48,
        max_concurrent=4,
        max_new_tokens=7,
        block_size=4,
        share_prefixes=True,
    )
    continuous_ids = [
        continuous.add_request(prompt, Sampler(temperature=0, seed=120 + index))
        for index, prompt in enumerate(prompts)
    ]
    while continuous.active_requests:
        continuous.step()
    continuous_outputs = [continuous.generated_tokens(seq_id) for seq_id in continuous_ids]

    independent_outputs = []
    for index, prompt in enumerate(prompts):
        engine = Engine(
            model,
            max_context=48,
            max_concurrent=1,
            max_new_tokens=7,
            block_size=4,
            share_prefixes=False,
        )
        seq_id = engine.add_request(prompt, Sampler(temperature=0, seed=120 + index))
        engine.run()
        independent_outputs.append(engine.generated_tokens(seq_id))
    assert continuous_outputs == independent_outputs


def test_pool_exhaustion_is_explicit_and_recoverable():
    pool = PagedKVPool(
        num_layers=1, block_size=4, max_context=8, max_sequences=2, num_blocks=1
    )
    first = pool.alloc()
    keys, values = kv(range(5), width=2)
    try:
        pool.append(first, keys, values)
    except PoolExhaustedError:
        pass
    else:
        raise AssertionError("append beyond fixed pool capacity should fail")
    pool.free(first)
    assert pool.num_free_blocks == pool.capacity

    model = tiny_model(ctx=64)
    undersized_pool = PagedKVPool(
        num_layers=model.n_layer,
        block_size=4,
        max_context=16,
        max_sequences=2,
        num_blocks=2,
    )
    engine = Engine(
        model,
        pool=undersized_pool,
        max_context=16,
        max_concurrent=2,
        max_new_tokens=8,
    )
    with pytest.raises(RequestCapacityError, match="worst-case page reservation"):
        engine.add_request([1, 2, 3, 4], Sampler(temperature=0))
