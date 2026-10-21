"""L6 kernel batching against the unchanged L4 single-request oracle."""

import mlx.core as mx
import numpy as np
import pytest

from layers import BatchedEngine, Engine, GPT, RequestCapacityError, Sampler
from layers.engine import _BatchedPagedKVCache
from layers.paged_kv import PagedKVCache, PagedKVPool


def numpy(value):
    mx.eval(value)
    return np.asarray(value, dtype=np.float32)


def model():
    return GPT(vocab_size=67, n_layer=2, d_model=32, n_head=4, ctx=128, seed=83)


class RecordingSampler:
    def __init__(self, seed, temperature=0.8):
        self.sampler = Sampler(temperature=temperature, top_p=0.95, seed=seed)
        self.logits = []

    def __call__(self, logits):
        self.logits.append(numpy(logits))
        return self.sampler(logits)


@pytest.mark.parametrize("temperature", [0, 0.8])
def test_eight_variable_prefixes_match_single_request_logits_for_33_steps(temperature):
    gpt = model()
    batched = BatchedEngine(gpt, block_size=4, max_concurrent=8)
    prompts = [list(range(1, length + 1)) for length in (1, 2, 4, 7, 11, 16, 23, 31)]
    references, samplers, seq_ids = [], [], []
    for row, prompt in enumerate(prompts):
        oracle = Engine(gpt, block_size=4, max_concurrent=1)
        left = RecordingSampler(row + 100, temperature)
        right = RecordingSampler(row + 100, temperature)
        seq_ids.append(batched.add_request(prompt, left, max_new_tokens=33))
        oracle.add_request(prompt, right, max_new_tokens=33)
        references.append(oracle)
        samplers.append((left, right))

    for _ in range(33):
        events = batched.step()
        for row, oracle in enumerate(references):
            reference_event = oracle.step()[0]
            assert events[row].token_id == reference_event.token_id
            assert events[row].finished == reference_event.finished
            np.testing.assert_allclose(
                samplers[row][0].logits[-1], samplers[row][1].logits[-1],
                atol=1e-5, rtol=0,
            )
    assert all(len(left.logits) == 33 for left, _ in samplers)
    for row, oracle in enumerate(references):
        assert batched.generated_tokens(seq_ids[row]) == oracle.generated_tokens(0)
    assert batched.pool.active_sequences == batched.pool.allocated_blocks == 0
    assert batched.step() == []


def test_row_permutation_permutes_logits_with_cache_and_inputs():
    gpt = model()
    pool = PagedKVPool(gpt.n_layer, block_size=4, max_context=128, max_sequences=6)
    seq_ids = [pool.alloc() for _ in range(3)]
    for seq_id, prompt in zip(seq_ids, ([1, 2], [7, 8, 9, 10, 11], [42])):
        mx.eval(gpt(mx.array(prompt), cache=PagedKVCache(pool, seq_id)))
    clones = [pool.fork(seq_id) for seq_id in seq_ids]
    ids = mx.array([[3], [12], [43]], dtype=mx.int32)
    expected = gpt.decode_batch(ids, _BatchedPagedKVCache(pool, seq_ids))
    order = [1, 0, 2]
    actual = gpt.decode_batch(
        ids[mx.array(order)], _BatchedPagedKVCache(pool, [clones[row] for row in order])
    )
    np.testing.assert_allclose(numpy(actual), numpy(expected)[order], atol=1e-5, rtol=0)
    for seq_id in seq_ids + clones:
        pool.free(seq_id)
    assert pool.allocated_blocks == 0


def test_pad_attention_probabilities_are_exactly_zero_after_softmax():
    gpt = model()
    pool = PagedKVPool(gpt.n_layer, block_size=4, max_context=128, max_sequences=3)
    seq_ids = [pool.alloc() for _ in range(3)]
    lengths = [1, 5, 13]
    for seq_id, length in zip(seq_ids, lengths):
        mx.eval(gpt(mx.array(list(range(length))), cache=PagedKVCache(pool, seq_id)))
    cache = _BatchedPagedKVCache(pool, seq_ids)
    x = gpt.blocks[0].ln_attn(gpt.token_embedding(mx.array([[2], [9], [17]])))
    _, probs, intermediates = gpt.blocks[0].attn.decode_batch_with_intermediates(x, cache, 0)
    probabilities = numpy(probs)
    scores = numpy(intermediates["scaled_scores"])
    assert probabilities.shape == (3, gpt.n_head, 1, max(lengths) + 1)
    for row, length in enumerate(lengths):
        pad = probabilities[row, :, :, length + 1:]
        np.testing.assert_allclose(pad, 0, atol=1e-7, rtol=0)
        assert np.count_nonzero(pad) == 0
        assert np.all(np.isneginf(scores[row, :, :, length + 1:]))
        np.testing.assert_allclose(probabilities[row].sum(axis=-1), 1, atol=1e-7)
        assert pool.gather(seq_ids[row], 0)[0].shape[-2] == length + 1


def test_dynamic_arrivals_completions_and_shared_page_release_preserve_logits():
    gpt = model()
    batched = BatchedEngine(gpt, block_size=4, max_concurrent=4)
    live, completed = {}, []

    def arrive(prompt, count, seed):
        oracle = Engine(gpt, block_size=4, max_concurrent=1)
        left, right = RecordingSampler(seed), RecordingSampler(seed)
        seq_id = batched.add_request(prompt, left, max_new_tokens=count)
        oracle.add_request(prompt, right, max_new_tokens=count)
        live[seq_id] = (oracle, left, right)

    arrive([1, 2, 3, 4, 5], 9, 31)
    arrive([1, 2, 3, 4, 5], 2, 32)
    assert batched.pool.refcount(batched.pool.block_table(0)[0]) == 2
    for iteration in range(13):
        if iteration == 2:
            arrive([1, 2, 3, 4, 6, 7], 5, 33)
            arrive([51], 1, 34)
        if iteration == 5:
            arrive([41, 42, 43], 6, 35)
        for event in batched.step():
            oracle, left, right = live[event.seq_id]
            reference = oracle.step()[0]
            assert (event.token_id, event.finished) == (reference.token_id, reference.finished)
            np.testing.assert_allclose(left.logits[-1], right.logits[-1], atol=1e-5, rtol=0)
            if event.finished:
                assert batched.generated_tokens(event.seq_id) == oracle.generated_tokens(0)
                completed.append(event.seq_id)
                del live[event.seq_id]
        assert batched.pool.active_sequences == len(live)
    assert len(completed) == 5
    assert batched.active_requests == batched.pool.allocated_blocks == 0


def test_singleton_context_limit_capacity_and_invalid_batch_fail_before_append():
    gpt = model()
    batched = BatchedEngine(gpt, block_size=4, max_concurrent=1, max_context=8)
    seq_id = batched.add_request([1, 2, 3, 4, 5, 6, 7], Sampler(temperature=0), 12)
    with pytest.raises(RequestCapacityError):
        batched.add_request([2], Sampler())
    events = batched.step()
    assert len(events) == 1 and events[0].finished
    assert len(batched.generated_tokens(seq_id)) == 1
    assert batched.pool.allocated_blocks == 0
    seq_id = batched.add_request([3], Sampler(), 2)
    cache = _BatchedPagedKVCache(batched.pool, [seq_id])
    with pytest.raises(ValueError, match="shape"):
        gpt.decode_batch(mx.array([[1, 2]]), cache)
    with pytest.raises(ValueError, match="distinct"):
        _BatchedPagedKVCache(batched.pool, [seq_id, seq_id])
    assert batched.pool.seq_len(seq_id) == 1
    batched.run()
    assert batched.pool.allocated_blocks == 0


def test_trained_checkpoint_variable_prefix_logits_stay_within_absolute_tolerance():
    # Width 512 exposes GEMM accumulation error hidden by the tiny fixture.
    from bench.l6_batching_bench import load_model

    gpt = load_model()
    batched = BatchedEngine(gpt, block_size=4, max_concurrent=8)
    references, samplers = [], []
    for row, length in enumerate((1, 2, 4, 7, 11, 16, 23, 31)):
        prompt = ((np.arange(length) + row * 29) % gpt.vocab_size).tolist()
        oracle = Engine(gpt, block_size=4, max_concurrent=1)
        left, right = RecordingSampler(500 + row), RecordingSampler(500 + row)
        batched.add_request(prompt, left, max_new_tokens=33)
        oracle.add_request(prompt, right, max_new_tokens=33)
        references.append(oracle)
        samplers.append((left, right))
    for _ in range(33):
        events = batched.step()
        for row, oracle in enumerate(references):
            event = oracle.step()[0]
            assert (events[row].token_id, events[row].finished) == (event.token_id, event.finished)
            np.testing.assert_allclose(
                samplers[row][0].logits[-1], samplers[row][1].logits[-1],
                atol=1e-5, rtol=0,
            )
    assert batched.pool.allocated_blocks == batched.pool.active_sequences == 0
