"""Diagnostic I/O failures must not change model request outcomes."""
import pytest

from layers import RequestCapacityError
from server.app import EngineBridge
from tests.test_api_server import engine
import server.app as app


@pytest.fixture(params=[BrokenPipeError, ValueError], ids=['broken-pipe', 'closed-stream'])
def failed_diagnostics(monkeypatch, request):
    def fail(*args, **kwargs):
        raise request.param('controlled diagnostic write failure')
    # Isolate the service's writer. Keep pytest's stderr/traceback channel usable.
    monkeypatch.setattr(app, 'print', fail, raising=False)


def test_log_failure_preserves_success_and_subsequent_requests(failed_diagnostics):
    e = engine()
    bridge = EngineBridge(e)
    try:
        for token in (1, 2):
            job = bridge.submit([token], 1, {'temperature': 0})
            assert job.ready.wait(2)
            assert job.done.wait(2)
            assert job.error is None and not job.cancelled.is_set()
            assert job.output.get_nowait()[1]
            assert bridge.ready
        assert e.pool.allocated_blocks == e.pool.active_sequences == 0
        counters = bridge.snapshot()['counters']
        assert counters['completed'] == 2 and counters['log_write_failures'] == 2
    finally:
        bridge.close()


def test_log_failure_preserves_rejection_notifications_and_reuse(failed_diagnostics):
    e = engine()
    original = e.add_request
    def admit(prompt, sampler, max_new_tokens):
        if prompt == [1]:
            raise RequestCapacityError('controlled rejection')
        return original(prompt, sampler, max_new_tokens)
    e.add_request = admit
    bridge = EngineBridge(e)
    try:
        rejected = bridge.submit([1], 1, {'temperature': 0})
        assert rejected.ready.wait(2), 'rejected request must wake its HTTP waiter'
        assert rejected.done.is_set() and rejected.error.status == 429
        assert bridge.ready
        accepted = bridge.submit([2], 1, {'temperature': 0})
        assert accepted.done.wait(2) and accepted.error is None
        assert e.pool.allocated_blocks == e.pool.active_sequences == 0
        assert bridge.snapshot()['counters']['rejected'] == 1
    finally:
        bridge.close()


def test_log_failure_during_engine_error_still_reclaims_every_page(failed_diagnostics):
    e = engine()
    def fail():
        raise RuntimeError('controlled model failure')
    e.step = fail
    bridge = EngineBridge(e)
    try:
        job = bridge.submit([1], 10, {'temperature': 0})
        assert job.ready.wait(2) and job.done.wait(2)
        bridge.close()
        assert job.error.status == 500 and job.cancelled.is_set()
        assert bridge.failed.is_set() and not bridge.ready
        assert e.pool.allocated_blocks == e.pool.active_sequences == 0
        counters = bridge.snapshot()['counters']
        assert counters['errors'] == 1 and counters['log_write_failures'] >= 1
    finally:
        bridge.close()
