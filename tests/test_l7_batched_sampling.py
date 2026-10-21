"""Token-exact L7-T2 sampling against unchanged scalar and engine oracles."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from layers import (
    BatchedEngine, BatchedSamplingEngine, GPT, PagedAttentionEngine,
    RequestCapacityError, Sampler, sample_batch,
)
from layers.paged_kv import PagedKVPool


CONFIGS = (
    {"temperature": 0},
    {"temperature": 0.7, "top_k": 1},
    {"temperature": 1},
    {"temperature": 0.35, "top_k": 7},
    {"temperature": 1.7, "top_p": 0.8},
    {"temperature": 0.9, "top_k": 13, "top_p": 0.95},
    {"temperature": 1, "top_k": 999, "top_p": 1},
    {"temperature": 2, "top_p": 1e-6},
)
ENGINES = (BatchedEngine, PagedAttentionEngine, BatchedSamplingEngine)


def model():
    return GPT(vocab_size=67, n_layer=2, d_model=32, n_head=4, ctx=128, seed=83)


def config(row, mode="mixed"):
    if mode == "greedy":
        return {"temperature": 0}
    if mode == "random":
        return {"temperature": 0.8, "top_k": 37, "top_p": 0.95}
    return CONFIGS[row % len(CONFIGS)]


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_mixed_rows_previous_draws_permutations_and_global_rng_are_bit_exact(dtype):
    left = [Sampler(**params, seed=91 + row % 3) for row, params in enumerate(CONFIGS)]
    right = [Sampler(**params, seed=91 + row % 3) for row, params in enumerate(CONFIGS)]
    initial = mx.array(np.linspace(-2, 2, 67), dtype=dtype)
    for row, (a, b) in enumerate(zip(left, right)):
        for _ in range(row % 4):
            assert a(initial) == b(initial)
    rng = np.random.default_rng(67)
    for step in range(48):
        # Include ties, uniform rows and changing request order/batch membership.
        values = rng.normal(size=(8, 67)).astype(np.float32)
        values[0] = 0
        values[1, :9] = 3
        logits = mx.array(values, dtype=dtype)
        order = rng.permutation(8)[: 1 + step % 8].tolist()
        expected = [right[row](logits[row]) for row in order]
        expected_rng = mx.random.uniform(shape=(9,)).tolist()
        mx.random.seed(8000 + step)  # unrelated users of the global generator
        actual = sample_batch([left[row] for row in order], logits[mx.array(order)])
        actual_rng = mx.random.uniform(shape=(9,)).tolist()
        assert actual == expected
        assert actual_rng == expected_rng
        assert [a._draws for a in left] == [b._draws for b in right]
    # Greedy also consumes a draw. Changing parameters must use its real history.
    for a, b in zip(left, right):
        a.temperature = b.temperature = 0.9
        a.top_k = b.top_k = None
        a.top_p = b.top_p = None
    assert sample_batch(left, logits) == [b(logits[row]) for row, b in enumerate(right)]


def test_reused_sampler_object_preserves_sequential_draw_order():
    logits = mx.array(np.random.default_rng(1).normal(size=(4, 67)))
    left, right = Sampler(seed=71), Sampler(seed=71)
    for _ in range(16):
        assert sample_batch([left] * 4, logits) == [right(row) for row in logits]
    assert left._draws == right._draws == 64


def test_batch_has_one_eval_one_transfer_and_no_scalar_host_reads(monkeypatch):
    logits = mx.array(np.random.default_rng(2).normal(size=(8, 67)))
    samplers = [Sampler(**params, seed=row) for row, params in enumerate(CONFIGS)]
    calls = {"eval": [], "tolist": []}
    original_eval, original_tolist = mx.eval, mx.array.tolist

    def evaluate(*arrays):
        calls["eval"].append([array.shape for array in arrays])
        return original_eval(*arrays)

    def transfer(array):
        calls["tolist"].append(array.shape)
        return original_tolist(array)

    def forbidden(*args, **kwargs):
        pytest.fail("scalar sampling or additional synchronization/host read")

    monkeypatch.setattr(mx, "eval", evaluate)
    monkeypatch.setattr(mx.array, "tolist", transfer)
    monkeypatch.setattr(mx.array, "item", forbidden)
    monkeypatch.setattr(mx.array, "__array__", forbidden)
    monkeypatch.setattr(mx, "synchronize", forbidden)
    monkeypatch.setattr(Sampler, "__call__", forbidden)
    assert len(sample_batch(samplers, logits)) == 8
    assert calls == {"eval": [[(8,)]], "tolist": [(8,)]}
    assert all(sampler._draws == 1 for sampler in samplers)


@pytest.mark.parametrize("shape,count", [((7,), 1), ((1, 0), 1), ((2, 7), 1)])
def test_invalid_shape_or_count_does_not_advance_rng(shape, count):
    samplers = [Sampler() for _ in range(count)]
    with pytest.raises(ValueError):
        sample_batch(samplers, mx.zeros(shape))
    assert all(sampler._draws == 0 for sampler in samplers)


def test_empty_batch_and_unsupported_callable_validate_before_draws(monkeypatch):
    monkeypatch.setattr(mx, "eval", lambda *args: pytest.fail("empty batch eval"))
    assert sample_batch([], mx.zeros((0, 7))) == []
    sampler = Sampler()
    with pytest.raises(TypeError, match="standard Sampler"):
        sample_batch([sampler, lambda logits: 0], mx.zeros((2, 7)))
    assert sampler._draws == 0


@pytest.mark.parametrize("mode", ["greedy", "random", "mixed"])
@pytest.mark.parametrize("lengths", [(7,), (7,) * 8, (1, 2, 4, 7, 11, 16, 23, 31)])
def test_equal_unequal_singleton_sequences_are_token_exact(mode, lengths):
    gpt = model()
    engines = [kind(gpt, block_size=4, max_concurrent=len(lengths), share_prefixes=False)
               for kind in (PagedAttentionEngine, BatchedSamplingEngine)]
    samplers = [[], []]
    for row, length in enumerate(lengths):
        prompt = ((np.arange(length) + row * 3) % gpt.vocab_size).tolist()
        for index, engine in enumerate(engines):
            sampler = Sampler(**config(row, mode), seed=100 + row)
            samplers[index].append(sampler)
            assert engine.add_request(prompt, sampler, 33) == row
    for _ in range(33):
        for seq_id in engines[0].active_seq_ids:
            np.testing.assert_array_equal(
                np.asarray(engines[0]._requests[seq_id].next_logits),
                np.asarray(engines[1]._requests[seq_id].next_logits),
            )
        assert engines[0].step() == engines[1].step()
    for seq_id in range(len(lengths)):
        assert engines[0].generated_tokens(seq_id) == engines[1].generated_tokens(seq_id)
    assert all(sampler._draws == 33 for group in samplers for sampler in group)
    assert all(engine.pool.allocated_blocks == engine.pool.active_sequences == 0 for engine in engines)
    assert engines[0].step() == engines[1].step() == []


def test_dynamic_mixed_arrivals_fork_cancel_completion_and_reuse_match_three_stages():
    gpt = model()
    engines = [kind(gpt, block_size=4, max_concurrent=4) for kind in ENGINES]
    samplers, live = {}, set()

    def arrive(prompt, count, row):
        group = [Sampler(**config(row), seed=31 + row) for _ in engines]
        # A newly admitted request need not have an unused sampler.
        for sampler in group:
            for _ in range(row % 3):
                sampler(mx.zeros(gpt.vocab_size))
        ids = [engine.add_request(prompt, sampler, count) for engine, sampler in zip(engines, group)]
        assert len(set(ids)) == 1
        samplers[ids[0]] = group
        live.add(ids[0])
        return ids[0]

    parent = arrive([1, 2, 3, 4, 5], 9, 2)
    child = arrive([1, 2, 3, 4, 5], 2, 3)
    assert all(engine.pool.refcount(engine.pool.block_table(parent)[0]) == 2 for engine in engines)
    for iteration in range(13):
        if iteration == 2:
            arrive([1, 2, 3, 4, 6, 7], 5, 4)
            cancelled = arrive([51], 8, 5)
        if iteration == 3:
            for engine in engines:
                engine.cancel_request(cancelled)
                engine.cancel_request(cancelled)
                engine.cancel_request(child)
            live.remove(cancelled)
        if iteration == 5:
            arrive([41, 42, 43], 6, 6)
        for seq_id in live:
            values = [np.asarray(engine._requests[seq_id].next_logits) for engine in engines]
            np.testing.assert_allclose(values[1], values[0], atol=1e-5, rtol=0)
            np.testing.assert_array_equal(values[2], values[1])
        events = [engine.step() for engine in engines]
        assert events[0] == events[1] == events[2]
        for event in events[0]:
            assert len({sampler._draws for sampler in samplers[event.seq_id]}) == 1
            if event.finished:
                assert len({engine.generated_tokens(event.seq_id) for engine in engines}) == 1
                live.remove(event.seq_id)
        assert all(engine.pool.active_sequences == len(live) for engine in engines)
    assert not live
    assert all(engine.pool.allocated_blocks == 0 for engine in engines)


def test_engine_uses_batch_entry_and_keeps_t1_kv_calls(monkeypatch):
    import layers.engine as implementation
    import layers.inference as inference

    engine = BatchedSamplingEngine(model(), block_size=4, max_concurrent=8)
    for row in range(8):
        engine.add_request([row + 1], Sampler(**config(row)), 3)
    real_sample, real_decode = inference.sample_batch, implementation._decode_paged_batch
    calls = []

    def sample(samplers, logits):
        calls.append(("sample", logits.shape))
        return real_sample(samplers, logits)

    def decode(gpt, ids, cache):
        assert isinstance(cache, implementation._DirectPagedKVCache)
        calls.append(("decode", ids.shape))
        return real_decode(gpt, ids, cache)

    def forbidden(*args, **kwargs):
        pytest.fail("legacy sampling or KV path called")

    monkeypatch.setattr(inference, "sample_batch", sample)
    monkeypatch.setattr(implementation, "_decode_paged_batch", decode)
    monkeypatch.setattr(Sampler, "__call__", forbidden)
    monkeypatch.setattr(engine.pool, "append", forbidden)
    monkeypatch.setattr(engine.pool, "gather", forbidden)
    assert len(engine.run()) == 24
    assert calls == [("sample", (8, 67)), ("decode", (8, 1))] * 2 + [("sample", (8, 67))]
    assert engine.pool.allocated_blocks == 0


def test_arbitrary_callable_and_sampler_subclass_keep_original_call_order(monkeypatch):
    import layers.inference as inference

    class CustomSampler(Sampler):
        def __call__(self, logits):
            return (super().__call__(logits) + 1) % logits.shape[0]

    gpt = model()
    engines = [kind(gpt, block_size=4, max_concurrent=3)
               for kind in (PagedAttentionEngine, BatchedSamplingEngine)]
    seen = [[], []]
    for index, engine in enumerate(engines):
        def custom(logits, index=index):
            seen[index].append(np.asarray(logits).copy())
            return 2
        engine.add_request([1], Sampler(seed=11), 3)
        engine.add_request([2], custom, 3)
        engine.add_request([3], CustomSampler(seed=13), 3)
    monkeypatch.setattr(inference, "sample_batch", lambda *args: pytest.fail("custom batch"))
    assert engines[0].run() == engines[1].run()
    np.testing.assert_array_equal(seen[0], seen[1])
    assert all(engine.pool.allocated_blocks == 0 for engine in engines)


def test_context_capacity_and_finished_iteration_reclaim_pages():
    gpt = model()
    for kind in ENGINES:
        pool = PagedKVPool(gpt.n_layer, block_size=4, max_context=8, max_sequences=2, num_blocks=2)
        engine = kind(gpt, pool=pool, max_context=8, max_concurrent=2)
        seq_id = engine.add_request([1, 2, 3, 4, 5, 6, 7], Sampler(temperature=0), 12)
        with pytest.raises(RequestCapacityError, match="reservation"):
            engine.add_request([2], Sampler(), 2)
        event = engine.step()[0]
        assert event.finished and len(engine.generated_tokens(seq_id)) == 1
        assert pool.allocated_blocks == pool.active_sequences == 0
        assert engine.step() == []


def test_trained_checkpoint_mixed_seeded_tokens_and_logits_match_three_stages():
    from bench.l6_batching_bench import load_model

    gpt = load_model()
    engines = [kind(gpt, block_size=4, max_concurrent=8, share_prefixes=False) for kind in ENGINES]
    samplers = [[], [], []]
    for row, length in enumerate((1, 2, 4, 7, 11, 16, 23, 31)):
        prompt = ((np.arange(length) + row * 29) % gpt.vocab_size).tolist()
        for index, engine in enumerate(engines):
            sampler = Sampler(**config(row), seed=500 + row)
            samplers[index].append(sampler)
            engine.add_request(prompt, sampler, 33)
    max_error = 0.0
    for _ in range(33):
        for seq_id in engines[0].active_seq_ids:
            arrays = [np.asarray(engine._requests[seq_id].next_logits) for engine in engines]
            max_error = max(max_error, float(np.max(np.abs(arrays[2] - arrays[0]))))
            np.testing.assert_allclose(arrays[2], arrays[0], atol=1e-5, rtol=0)
            np.testing.assert_array_equal(arrays[2], arrays[1])
        events = [engine.step() for engine in engines]
        assert events[0] == events[1] == events[2]
    assert all(sampler._draws == 33 for group in samplers for sampler in group)
    assert all(engine.pool.allocated_blocks == engine.pool.active_sequences == 0 for engine in engines)
    path = Path(__file__).resolve().parents[1] / "results/l7/t2-checkpoint-oracle.json"
    path.write_text(json.dumps({
        "stages": [kind.__name__ for kind in ENGINES], "batch_size": 8,
        "sampled_tokens_per_request": 33, "decode_forwards_per_request": 32,
        "sampler_configs": CONFIGS, "atol": 1e-5, "rtol": 0,
        "max_absolute_logit_error_vs_l6": max_error,
        "t2_vs_t1_logits_bit_exact": True, "seeded_tokens_equal": True,
        "all_pages_reclaimed": True,
    }, indent=2) + "\n")
