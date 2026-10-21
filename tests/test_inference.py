"""Inference path checks against the L3 checkpoint and independent behavior."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from layers import BPETokenizer, GPT, KVCache, Sampler, generate


ROOT = Path(__file__).resolve().parents[1]


def as_numpy(value):
    return np.asarray(value, dtype=np.float32)


def load_l3_model():
    model = GPT(vocab_size=1028, n_layer=4, d_model=512, n_head=8, ctx=256)
    with np.load(ROOT / "results/l3/ckpt-final-500.npz") as checkpoint:
        for name in checkpoint.files:
            parts = name.split(".")
            current = model
            for part in parts[:-1]:
                current = current[int(part)] if part.isdigit() else getattr(current, part)
            setattr(current, parts[-1], mx.array(checkpoint[name]))
    return model


def test_l3_checkpoint_cache_matches_full_prefix_for_32_decode_steps():
    model = load_l3_model()
    ids = mx.array((list(range(8)) + list(range(8, 40))), dtype=mx.int32)
    cache = KVCache(model.n_layer)

    prefill_logits = model(ids[:8], cache=cache)
    mx.eval(prefill_logits)
    np.testing.assert_allclose(
        as_numpy(prefill_logits[-1]), as_numpy(model(ids[:8])[-1]), rtol=1e-5, atol=1e-5
    )
    for position in range(8, 40):
        cached_logits = model(ids[position : position + 1], cache=cache)[-1]
        full_logits = model(ids[: position + 1])[-1]
        mx.eval(cached_logits, full_logits)
        np.testing.assert_allclose(
            as_numpy(cached_logits),
            as_numpy(full_logits),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"cached decode diverged at absolute token position {position}",
        )


def test_sampler_greedy_top_k_top_p_and_seeded_reproducibility():
    logits = mx.array([0.25, 0.5, 2.0, -1.0])
    greedy = int(mx.argmax(logits).item())
    assert Sampler(temperature=0)(logits) == greedy
    assert Sampler(top_k=1)(logits) == greedy

    probabilities = Sampler(temperature=0.8, top_k=3, top_p=0.8).probabilities(logits)
    mx.eval(probabilities)
    values = as_numpy(probabilities)
    assert np.isclose(values.sum(), 1.0, atol=1e-6)
    assert np.count_nonzero(values) <= 3
    assert np.all(values >= 0)

    first = Sampler(temperature=0.9, top_k=3, top_p=0.95, seed=2026)
    second = Sampler(temperature=0.9, top_k=3, top_p=0.95, seed=2026)
    sequence_a = [first(logits) for _ in range(12)]
    sequence_b = [second(logits) for _ in range(12)]
    assert sequence_a == sequence_b


class _ByteSequenceModel:
    """Tiny deterministic model that emits a prescribed token byte sequence."""

    n_layer = 1

    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.position = 0

    def __call__(self, ids, cache=None):
        logits = np.full((ids.shape[-1], 256), -100.0, dtype=np.float32)
        logits[-1, self.token_ids[self.position]] = 100.0
        self.position += 1
        return mx.array(logits)


class _FiniteLogitsModel:
    def __init__(self, model):
        self.model = model
        self.n_layer = model.n_layer

    def __call__(self, ids, cache=None):
        logits = self.model(ids, cache=cache)
        mx.eval(logits)
        assert np.isfinite(as_numpy(logits)).all()
        return logits


def test_streaming_utf8_buffers_split_cjk_and_emoji_bytes():
    tokenizer = BPETokenizer()
    expected = "中🙂"
    token_ids = list(expected.encode("utf-8"))
    chunks = list(
        generate(
            _ByteSequenceModel(token_ids),
            tokenizer,
            prompt=[ord("x")],
            max_new=len(token_ids),
            sampler=Sampler(temperature=0),
        )
    )
    streamed = "".join(chunks)
    assert streamed == tokenizer.decode(token_ids)
    assert "\ufffd" not in streamed
    assert len(chunks) < len(token_ids)


def test_l3_checkpoint_greedy_generation_smoke_200_tokens():
    model = load_l3_model()
    tokenizer = BPETokenizer.load(str(ROOT / "results/l1/tokenizer-1024.json"))
    prompt = "ROMEO: "
    prompt_ids = tokenizer.encode(prompt)
    initial_logits = model(mx.array(prompt_ids, dtype=mx.int32))
    mx.eval(initial_logits)
    assert np.isfinite(as_numpy(initial_logits)).all()

    chunks = list(
        generate(
            _FiniteLogitsModel(model),
            tokenizer,
            prompt,
            max_new=200,
            sampler=Sampler(temperature=0),
        )
    )
    generated = "".join(chunks)
    assert generated
    assert "\ufffd" not in generated
    assert len(set(generated)) > 1

