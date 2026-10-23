"""Independent full-prefix stop Oracle, Unicode boundaries and HTTP semantics."""
import random

import pytest

from server.completion import CompletionFilter, validate_stop
from tests.test_api_server import engine, serving, payload, completion, reclaimed


def oracle(text, stops):
    for end in range(1, len(text) + 1):
        matches = [end - len(stop) for stop in stops if text[:end].endswith(stop)]
        if matches:
            return text[:min(matches)], True
    return text, False


def test_stream_filter_matches_independent_character_oracle_for_byte_partitions():
    rng = random.Random(42)
    for text, stops in [
        ('你好🙂before<END>after', ('<END>',)),
        ('prefix abcd tail', ('abcd', 'bc')),
        ('abababac trailing', ('abac',)),
        ('no stop he', ('hello',)),
        ('🙂你结束余文', ('你结束',)),
    ]:
        expected, stopped = oracle(text, stops)
        for _ in range(25):
            filt = CompletionFilter(stops)
            data, parts = text.encode(), []
            while data:
                size = rng.randrange(1, 6)
                part, _matched = filt.feed(data[:size])
                parts.append(part)
                data = data[size:]
            parts.append(filt.feed(b'', final=True)[0])
            assert ''.join(parts) == expected
            assert filt.stopped == stopped


@pytest.mark.parametrize('value', ['', [], ['a'] * 5, [1], ['a' * 257], True])
def test_stop_validation(value):
    with pytest.raises(ValueError):
        validate_stop(value)


def scripted_engine(tokens, backend='l6'):
    e = engine()
    if backend == 'l7':
        from layers import BatchedSamplingEngine
        e = BatchedSamplingEngine(e.model, e.tokenizer, max_concurrent=8, block_size=4)
    original = e.add_request
    def add(prompt, sampler, max_new_tokens):
        iterator = iter(tokens)
        return original(prompt, lambda logits: next(iterator), max_new_tokens)
    e.add_request = add
    return e


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('backend', ['l6', 'l7'])
def test_stop_and_eos_suppress_control_text_and_release_early(stream, backend):
    for tokens, stops, expected, count in [
        (list('你🙂STOP尾'.encode()), 'STOP', '你🙂', len('你🙂STOP'.encode())),
        ([ord('a'), 257, ord('z')], None, 'a', 2),
    ]:
        with serving(scripted_engine(tokens, backend)) as server:
            text, data = completion(server, payload(stream=stream, stop=stops,
                max_tokens=len(tokens), stream_options={'include_usage': True})
                if stream else payload(stop=stops, max_tokens=len(tokens)))
            assert text == expected
            if stream:
                assert data[-2]['choices'][0]['finish_reason'] == 'stop'
                assert data[-1]['usage']['completion_tokens'] == count
            else:
                assert data['choices'][0]['finish_reason'] == 'stop'
                assert data['usage']['completion_tokens'] == count
            reclaimed(server)


def test_length_flushes_incomplete_stop_prefix():
    with serving(scripted_engine(list(b'hello ST'))) as server:
        text, data = completion(server, payload(stop='STOP', max_tokens=8))
        assert text == 'hello ST'
        assert data['choices'][0]['finish_reason'] == 'length'
        reclaimed(server)
