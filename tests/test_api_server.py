"""HTTP wire tests against the real blind-built tokenizer/model/engine.

Schema oracle (public API documentation only, checked 2026-10-21):
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
https://developers.openai.com/api/reference/resources/models/methods/list
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import json
import socket
import struct
import threading
import time

import pytest

from layers import BatchedEngine, BPETokenizer, GPT, Sampler
from server.app import APIError, EngineBridge, chat_template, make_server


def engine():
    return BatchedEngine(
        GPT(vocab_size=260, n_layer=1, d_model=32, n_head=4, ctx=256, seed=49),
        BPETokenizer(), max_concurrent=8, block_size=4,
    )


@contextmanager
def serving(gpt_engine=None, **options):
    server = make_server("127.0.0.1", 0, gpt_engine or engine(), **options)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
        assert not server.bridge.thread.is_alive()


def payload(row=0, **options):
    return {"model": "layers-gpt", "messages": [{"role": "user", "content": f"Prompt {row}"}],
            "max_tokens": 24, "temperature": 0.8, "top_p": 0.9, "seed": 100 + row, **options}


def post(server, body):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    return connection, response


def read_sse(response):
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    assert response.getheader("Cache-Control") == "no-cache"
    assert response.getheader("x-request-id").startswith("req-")
    chunks = []
    while True:
        line = response.readline()
        assert line.startswith(b"data: "), line
        assert response.readline() == b"\n"
        if line == b"data: [DONE]\n":
            assert response.read() == b""
            return chunks
        chunks.append(json.loads(line[6:]))


def completion(server, body):
    connection, response = post(server, body)
    try:
        if body.get("stream"):
            chunks = read_sse(response)
            text = "".join(choice["delta"].get("content", "")
                           for chunk in chunks for choice in chunk["choices"])
            return text, chunks
        assert response.status == 200
        data = json.loads(response.read())
        return data["choices"][0]["message"]["content"], data
    finally:
        connection.close()


def direct(gpt_engine, body):
    # This independent oracle bypasses HTTP, bridge, and incremental decoder.
    prompt_ids = gpt_engine.tokenizer.encode(chat_template(body["messages"]))
    seq = gpt_engine.add_request(prompt_ids, Sampler(
        temperature=body["temperature"], top_p=body["top_p"], seed=body["seed"]
    ), body["max_tokens"])
    gpt_engine.run()
    text = gpt_engine.tokenizer.decode(list(gpt_engine.generated_tokens(seq)))
    gpt_engine.cancel_request(seq)
    return text


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "resource reclamation timed out"
        time.sleep(0.005)


def reclaimed(server):
    gpt_engine = server.bridge.engine
    eventually(lambda: gpt_engine.pool.allocated_blocks == gpt_engine.pool.active_sequences == 0
               and not gpt_engine._requests)


def test_schema_models_json_and_incremental_sse_match_direct_engine():
    gpt_engine = engine()
    body = payload(max_tokens=40)
    expected = direct(gpt_engine, body)
    with serving(gpt_engine) as server:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/v1/models")
        response = conn.getresponse()
        models = json.loads(response.read())
        conn.close()
        assert response.status == 200 and models["object"] == "list"
        model = models["data"][0]
        assert {"id", "created", "object", "owned_by"} <= model.keys()
        assert model["id"] == "layers-gpt" and model["object"] == "model"
        assert type(model["created"]) is int and "base model" in model["description"]
        assert "not instruction tuned" in model["description"]

        text, data = completion(server, body)
        assert text == expected
        assert {"id", "object", "created", "model", "choices", "usage"} <= data.keys()
        assert data["id"].startswith("chatcmpl-") and type(data["created"]) is int
        assert data["object"] == "chat.completion" and data["model"] == "layers-gpt"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "length"
        prompt_count = len(gpt_engine.tokenizer.encode(chat_template(body["messages"])))
        assert data["usage"] == {"prompt_tokens": prompt_count, "completion_tokens": 40,
                                 "total_tokens": prompt_count + 40}

        streamed, chunks = completion(server, {**body, "stream": True,
                                               "stream_options": {"include_usage": True}})
        assert streamed == expected
        assert len({c["id"] for c in chunks}) == len({c["created"] for c in chunks}) == 1
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk" and chunk["model"] == "layers-gpt"
            assert type(chunk["created"]) is int
            for choice in chunk["choices"]:
                assert choice["index"] == 0 and isinstance(choice["delta"], dict)
                assert "message" not in choice
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert chunks[-2]["choices"][0]["finish_reason"] == "length"
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == data["usage"]
        assert all(c["usage"] is None for c in chunks[:-1])
        assert len([c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]) > 1
        reclaimed(server)


def test_eight_concurrent_prompts_and_sampler_settings_match_isolated_runs():
    gpt_engine = engine()
    bodies = [payload(row, temperature=(0, 0.3, 0.8, 1.4)[row % 4], top_p=0.7 + row / 30)
              for row in range(8)]
    expected = [direct(gpt_engine, body) for body in bodies]
    barrier = threading.Barrier(8)
    with serving(gpt_engine) as server:
        def run(body):
            barrier.wait(5)
            return completion(server, {**body, "stream": True})
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, bodies))
        assert [text for text, _ in results] == expected
        assert len({chunks[0]["id"] for _, chunks in results}) == 8
        assert all("usage" not in c for _, chunks in results for c in chunks)
        reclaimed(server)


@pytest.mark.parametrize("stream", [True, False])
def test_disconnect_reclaims_live_pages_before_generation_finishes(stream):
    gpt_engine = engine()
    original = gpt_engine.step
    paused, resume = threading.Event(), threading.Event()
    steps = []
    def gated_step():
        steps.append(threading.get_ident())
        if len(steps) == 2:
            paused.set()
            assert resume.wait(5)
        return original()
    gpt_engine.step = gated_step
    with serving(gpt_engine) as server:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        body = payload(stream=stream, max_tokens=200)
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        # Set this while HTTPConnection still owns the socket. Reading headers
        # with Connection: close transfers ownership to HTTPResponse's file.
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        try:
            assert paused.wait(5)
            assert gpt_engine.pool.allocated_blocks > 0
            if stream:
                response = connection.getresponse()
                assert response.readline().startswith(b"data: ")
                assert response.readline() == b"\n"
                response.close()
            # RST closes both buffered HTTP and raw sockets, also for JSON mode
            # where no response headers have been emitted yet.
            connection.close()
        finally:
            resume.set()
        reclaimed(server)
        assert len(steps) < 200
        assert set(steps) == {server.bridge.thread.ident}


def test_output_queue_overflow_aborts_only_slow_request_and_frees_pages():
    gpt_engine = engine()
    bridge = EngineBridge(gpt_engine, output_size=1)
    disconnected = threading.Event()
    try:
        job = bridge.submit([1, 2, 3], 200, {"temperature": 0, "seed": 11}, disconnected.set)
        assert job.done.wait(5)
        assert job.cancelled.is_set() and disconnected.is_set()
        assert "fell behind" in str(job.error)
        assert job.count == 2 and job.output.qsize() == 1
        assert gpt_engine.pool.allocated_blocks == gpt_engine.pool.active_sequences == 0
        # Capacity remains usable after cancelling the slow stream.
        healthy = bridge.submit([4, 5], 1, {"temperature": 0, "seed": 12})
        assert healthy.done.wait(5)
        assert healthy.output.get_nowait()[1] and not healthy.cancelled.is_set()
    finally:
        bridge.close()


def test_pending_capacity_is_bounded_and_shutdown_cancels_pending_and_live_jobs():
    gpt_engine = engine()
    original = gpt_engine.step
    paused, resume = threading.Event(), threading.Event()
    def gated_step():
        paused.set()
        assert resume.wait(5)
        return original()
    gpt_engine.step = gated_step
    bridge = EngineBridge(gpt_engine, pending_size=1)
    try:
        live = bridge.submit([1], 100, {"temperature": 0, "seed": 1})
        assert paused.wait(5)
        pending = bridge.submit([2], 100, {"temperature": 0, "seed": 2})
        with pytest.raises(APIError) as exc:
            bridge.submit([3], 100, {"temperature": 0, "seed": 3})
        assert exc.value.status == 429
        bridge.stopping.set()
    finally:
        resume.set()
        bridge.close()
    assert live.done.is_set() and pending.done.is_set()
    assert live.cancelled.is_set() and pending.cancelled.is_set()
    assert gpt_engine.pool.allocated_blocks == gpt_engine.pool.active_sequences == 0
    with pytest.raises(APIError) as exc:
        bridge.submit([4], 1, {"temperature": 0})
    assert exc.value.status == 503


@pytest.mark.parametrize("change,status,param", [
    ({"model": "missing"}, 404, "model"), ({"stream": "true"}, 400, "stream"),
    ({"messages": []}, 400, "messages"), ({"messages": [{"role": "tool", "content": "a"}]}, 400, "messages"),
    ({"temperature": float("nan")}, 400, None), ({"temperature": -1}, 400, "temperature"),
    ({"top_p": 0}, 400, "top_p"), ({"seed": True}, 400, "seed"),
    ({"max_tokens": True}, 400, "max_tokens"), ({"max_tokens": 0}, 400, "max_tokens"),
    ({"max_tokens": 1000}, 400, "max_tokens"), ({"tools": []}, 400, "tools"),
    ({"stream_options": {"include_usage": True}}, 400, "stream_options"),
])
def test_invalid_requests_return_openai_error_envelope(change, status, param):
    with serving() as server:
        connection, response = post(server, payload(**change))
        data = json.loads(response.read())
        connection.close()
        assert response.status == status
        assert set(data["error"]) == {"message", "type", "param", "code"}
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == param
        reclaimed(server)


def test_engine_capacity_returns_429_and_cancel_is_idempotent():
    gpt_engine = engine()
    bridge = EngineBridge(gpt_engine)
    # Queue 9 admissions before a single step, without racing a fast tiny model.
    original = gpt_engine.add_request
    entered, resume = threading.Event(), threading.Event()
    def gated_add(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return original(*args, **kwargs)
    gpt_engine.add_request = gated_add
    try:
        jobs = [bridge.submit([1], 100, {"temperature": 0})]
        assert entered.wait(5)
        jobs.extend(bridge.submit([row], 100, {"temperature": 0}) for row in range(2, 10))
        resume.set()
        assert jobs[-1].ready.wait(5)
        assert jobs[-1].error.status == 429 and jobs[-1].done.is_set()
        for job in jobs:
            job.cancelled.set()
        assert all(job.done.wait(5) for job in jobs)
    finally:
        resume.set()
        bridge.close()
    assert gpt_engine.pool.allocated_blocks == 0
    gpt_engine.cancel_request(jobs[0].seq_id)


def test_utf8_split_tokens_are_buffered_until_complete():
    # A controlled sampler probes the byte-decoding boundary, not model output.
    gpt_engine = engine()
    original = gpt_engine.add_request
    token_ids = list("你🙂".encode("utf-8")) + [0xE4]
    def admit(prompt, sampler, max_new_tokens):
        iterator = iter(token_ids)
        return original(prompt, lambda logits: next(iterator), max_new_tokens)
    gpt_engine.add_request = admit
    with serving(gpt_engine) as server:
        text, chunks = completion(server, payload(stream=True, max_tokens=len(token_ids)))
        assert text == bytes(token_ids).decode("utf-8", "replace")
        contents = [choice["delta"].get("content", "") for c in chunks for choice in c["choices"]]
        assert [part for part in contents if part] == ["你", "🙂", "�"]
        reclaimed(server)
