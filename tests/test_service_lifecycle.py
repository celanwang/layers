"""Real engine lifecycle and HTTP state probes with explicit synchronization."""
import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from server.app import APIError, EngineBridge
from server.observability import Metrics
from tests.test_api_server import engine, serving, payload, completion, reclaimed, direct


def get(server, path):
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    conn.request('GET', path)
    response = conn.getresponse()
    result = response.status, json.loads(response.read())
    conn.close()
    return result


def test_health_metrics_and_l7_random_api_results_match_l6_oracle():
    from layers import BatchedSamplingEngine
    reference = engine()
    requests = [payload(i, max_tokens=8, stream=bool(i % 2)) for i in range(8)]
    expected = [direct(reference, body) for body in requests]
    optimized = BatchedSamplingEngine(reference.model, reference.tokenizer,
                                      max_concurrent=8, block_size=4)
    with serving(optimized) as server:
        assert get(server, '/healthz')[0] == get(server, '/readyz')[0] == 200
        with ThreadPoolExecutor(8) as pool:
            outputs = list(pool.map(lambda body: completion(server, body)[0], requests))
        assert outputs == expected
        reclaimed(server)
        status, metrics = get(server, '/metrics')
        assert status == 200 and metrics['counters']['completed'] == 8
        assert metrics['counters']['generated_tokens'] == 64
        assert metrics['gauges']['allocated_pages'] == 0
        assert metrics['latencies']['itl_seconds']['window_count'] == 56
        server.bridge.begin_drain(1)
        assert get(server, '/readyz')[0] == 503


def test_drain_completes_previously_accepted_jobs_and_rejects_new_jobs():
    e = engine()
    entered, resume = threading.Event(), threading.Event()
    original = e.step
    def gated():
        entered.set()
        assert resume.wait(5)
        return original()
    e.step = gated
    bridge = EngineBridge(e, log_requests=False)
    try:
        live = bridge.submit([1], 3, {'temperature': 0})
        assert entered.wait(5)
        queued = bridge.submit([2], 2, {'temperature': 0})
        bridge.begin_drain(2)
        with pytest.raises(APIError) as error:
            bridge.submit([3], 1, {'temperature': 0})
        assert error.value.status == 503
        resume.set()
        bridge.close(drain_timeout=2)
        assert live.done.is_set() and queued.done.is_set()
        assert not live.cancelled.is_set() and not queued.cancelled.is_set()
        assert e.pool.allocated_blocks == e.active_requests == 0
        assert not bridge.ready and not bridge.thread.is_alive()
    finally:
        resume.set()
        bridge.close()


def test_expired_drain_cancels_at_next_iteration_and_reclaims_everything():
    e = engine()
    entered, resume = threading.Event(), threading.Event()
    original = e.step
    def gated():
        entered.set()
        assert resume.wait(5)
        return original()
    e.step = gated
    bridge = EngineBridge(e, log_requests=False)
    try:
        live = bridge.submit([1], 100, {'temperature': 0})
        assert entered.wait(5)
        queued = bridge.submit([2], 100, {'temperature': 0})
        bridge.begin_drain(0)
        resume.set()
        bridge.close(drain_timeout=0)
        assert live.done.is_set() and queued.done.is_set()
        assert live.cancelled.is_set() and queued.cancelled.is_set()
        assert e.pool.allocated_blocks == e.pool.active_sequences == 0
        assert bridge.snapshot()['gauges']['waiting_requests'] == 0
    finally:
        resume.set()
        bridge.close()


def test_owner_failure_marks_not_ready_and_cleans_pages():
    e = engine()
    def fail(): raise RuntimeError('controlled failure')
    e.step = fail
    bridge = EngineBridge(e, log_requests=False)
    job = bridge.submit([1], 10, {'temperature': 0})
    assert job.done.wait(5)
    bridge.close()
    assert bridge.failed.is_set() and not bridge.ready
    assert e.pool.allocated_blocks == 0
    assert bridge.snapshot()['counters']['errors'] == 1


def test_metric_sample_memory_is_bounded():
    metrics = Metrics()
    for value in range(2000): metrics.observe('itl_seconds', value)
    assert metrics.snapshot()['latencies']['itl_seconds']['window_count'] == 1024
