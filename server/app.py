"""OpenAI-compatible text chat subset, implemented without a web framework.

Run ``.venv/bin/python server/app.py --port 8611``. The L3 checkpoint is a
500-step base model, not instruction tuned. Messages use ``role: content``
lines, without an invented assistant suffix. Only the documented text subset
is accepted; unsupported generation options are rejected rather than ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import queue
import select
import signal
import socket
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layers import BatchedEngine, BPETokenizer, GPT, RequestCapacityError, Sampler
from layers.inference import _token_bytes
from server.completion import CompletionFilter, validate_stop
from server.observability import Metrics
from server.runtime import MODEL_MANIFEST, load_assets

MODEL_ID = "layers-gpt"
DESCRIPTION = (
    "L3 500-step base model; not instruction tuned. "
    "Messages are joined as role: content lines. Text-only chat subset."
)
CHECKPOINT = ROOT / "results/l3/ckpt-final-500.npz"
TOKENIZER = ROOT / "results/l1/tokenizer-1024.json"


class APIError(Exception):
    def __init__(self, message, status=400, param=None, code=None):
        super().__init__(message)
        self.status, self.param, self.code = status, param, code

    def body(self):
        kind = "server_error" if self.status >= 500 else "invalid_request_error"
        if self.status == 429:
            kind = "rate_limit_error"
        return {"error": {"message": str(self), "type": kind,
                          "param": self.param, "code": self.code}}


def chat_template(messages):
    """The deliberately minimal, inspectable base-model input format."""
    if not isinstance(messages, list) or not messages:
        raise APIError("messages must be a non-empty list", param="messages")
    lines = []
    for item in messages:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise APIError("each message needs only role and text content", param="messages")
        if item["role"] not in ("system", "developer", "user", "assistant"):
            raise APIError("unsupported message role", param="messages")
        if not isinstance(item["content"], str):
            raise APIError("message content must be text", param="messages")
        try:
            item["content"].encode("utf-8")
        except UnicodeEncodeError as exc:
            raise APIError("message content must be valid Unicode", param="messages") from exc
        lines.append(f"{item['role']}: {item['content']}")
    return "\n".join(lines)


def parse_request(body, engine):
    if not isinstance(body, dict):
        raise APIError("request must be a JSON object")
    supported = {"model", "messages", "stream", "max_tokens", "temperature",
                 "top_p", "seed", "stream_options", "stop"}
    unknown = set(body) - supported
    if unknown:
        raise APIError("unsupported parameter", param=sorted(unknown)[0])
    if body.get("model") != MODEL_ID:
        raise APIError("model must be layers-gpt", 404, "model", "model_not_found")
    stream = body.get("stream", False)
    if type(stream) is not bool:
        raise APIError("stream must be a boolean", param="stream")
    for name, default, low, high in (("temperature", 1.0, 0, 2), ("top_p", 1.0, 0, 1)):
        value = body.get(name, default)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high or (name == "top_p" and value == 0)):
            raise APIError(f"invalid {name}", param=name)
    seed = body.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= 2**32 - engine.max_context:
        raise APIError("seed must be a non-negative 32-bit integer with room for context", param="seed")
    options = body.get("stream_options", {})
    if (not isinstance(options, dict) or set(options) - {"include_usage"}
            or type(options.get("include_usage", False)) is not bool
            or ("stream_options" in body and not stream)):
        raise APIError("stream_options supports only include_usage with stream=true", param="stream_options")
    try:
        validate_stop(body.get("stop"))
    except ValueError as exc:
        raise APIError(str(exc), param="stop") from exc
    prompt_ids = engine.tokenizer.encode(chat_template(body.get("messages")))
    remaining = engine.max_context - len(prompt_ids)
    count = body.get("max_tokens", min(32, remaining))
    if type(count) is not int or count <= 0 or count > remaining:
        raise APIError("max_tokens must be positive and fit the remaining context", param="max_tokens")
    return prompt_ids, count, {
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0), "seed": seed,
    }, stream, options.get("include_usage", False)


@dataclass
class Generation:
    prompt_ids: list[int]
    max_tokens: int
    sampling: dict
    output: queue.Queue
    disconnect: object = None
    ready: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: APIError | None = None
    seq_id: int | None = None
    count: int = 0

    completion: object = field(default_factory=CompletionFilter)
    finish_reason: str = "length"
    request_id: str = field(default_factory=lambda: "req-" + uuid.uuid4().hex)
    submitted_at: float = field(default_factory=time.monotonic)
    last_token_at: float | None = None
    first_text_at: float | None = None
    reported: bool = False

    def abort(self, error):
        self.error = error
        self.cancelled.set()
        self.ready.set()
        if self.disconnect is not None:
            self.disconnect()


class EngineBridge:
    """Single engine owner; bounded ingress and per-request token queues.

    Cancellation is an Event, so it still reaches the owner when ingress is
    full. Queue overflow never blocks other requests: abort just that stream,
    close its socket, and release its pages at the iteration boundary.
    """

    def __init__(self, engine, *, pending_size=32, output_size=128, log_requests=True):
        if pending_size <= 0 or output_size <= 0:
            raise ValueError("queue capacities must be positive")
        import mlx.core as mx

        # A lazy graph may refer to its creator's thread-local stream. Evaluate
        # weights before handing sole inference ownership to another thread.
        mx.eval(engine.model.parameters())
        self.engine, self.output_size = engine, output_size
        self.pending = queue.Queue(pending_size)
        self.stopping = threading.Event()
        self.draining = threading.Event()
        self.failed = threading.Event()
        self.drain_deadline = None
        self.metrics = Metrics()
        self.log_requests = log_requests
        self.eos_id = engine.tokenizer._special_ids.get("<|endoftext|>") if engine.tokenizer else None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="layers-engine", daemon=True)
        self.thread.start()

    def submit(self, prompt_ids, max_tokens, sampling, disconnect=None, *, stops=(), request_id=None):
        job = Generation(prompt_ids, max_tokens, sampling, queue.Queue(self.output_size), disconnect)
        job.completion = CompletionFilter(stops)
        if request_id is not None:
            job.request_id = request_id
        self.metrics.count("received")
        with self.lock:
            if not self.ready:
                self.metrics.count("rejected")
                raise APIError("server is stopping", 503)
            try:
                self.pending.put_nowait(job)
            except queue.Full as exc:
                self.metrics.count("rejected")
                raise APIError("request queue is full", 429, code="capacity_exceeded") from exc
        return job

    @property
    def ready(self):
        return self.thread.is_alive() and not (
            self.stopping.is_set() or self.draining.is_set() or self.failed.is_set())

    def snapshot(self):
        result = self.metrics.snapshot()
        result.update({"ready": self.ready, "draining": self.draining.is_set(),
                       "owner_alive": self.thread.is_alive(), "failed": self.failed.is_set()})
        return result

    def _publish(self):
        self.metrics.publish(active_requests=self.engine.active_requests,
                             waiting_requests=self.pending.qsize(),
                             allocated_pages=self.engine.pool.allocated_blocks,
                             capacity_pages=self.engine.pool.capacity,
                             free_pages=self.engine.pool.num_free_blocks)

    def _diagnostic(self, message):
        """A broken/closed log sink must never interrupt terminal notifications."""
        try:
            print(message, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            # Do not recursively report a failure through the same broken sink.
            self.metrics.count("log_write_failures")

    def _report(self, job):
        if job.reported:
            return
        job.reported = True
        outcome = ("rejected" if job.seq_id is None and job.error and job.error.status in (429, 503)
                   else "errors" if job.error else "cancelled" if job.cancelled.is_set() else "completed")
        self.metrics.count(outcome)
        if self.log_requests:
            self._diagnostic(json.dumps({"event": "request_end", "request_id": job.request_id,
                              "outcome": outcome, "finish_reason": job.finish_reason,
                              "completion_tokens": job.count,
                              "duration_seconds": time.monotonic() - job.submitted_at}))

    def _release(self, job):
        if job.seq_id is not None:
            self.engine.cancel_request(job.seq_id)
        self._report(job)
        job.done.set()

    def _admit(self, job, live):
        try:
            if self.draining.is_set() and time.monotonic() >= self.drain_deadline:
                job.abort(APIError("drain deadline expired", 503))
            if not job.cancelled.is_set():
                job.seq_id = self.engine.add_request(
                    job.prompt_ids, Sampler(**job.sampling), job.max_tokens
                )
                live[job.seq_id] = job
        except RequestCapacityError as exc:
            job.error = APIError(str(exc), 429, code="capacity_exceeded")
        except Exception:
            self._diagnostic(traceback.format_exc())
            job.error = APIError("engine admission failed", 500)
        finally:
            if job.seq_id is None:
                self._report(job)
                job.done.set()
            job.ready.set()

    def _run(self):
        live = {}
        try:
            while not self.stopping.is_set():
                if self.draining.is_set() and time.monotonic() >= self.drain_deadline:
                    break
                # Cancel before admitting, so reclaimed capacity is immediately reusable.
                for seq_id, job in list(live.items()):
                    if job.cancelled.is_set():
                        self._release(job)
                        del live[seq_id]
                self._publish()
                if self.draining.is_set() and not live and self.pending.empty():
                    break
                if not live:
                    try:
                        self._admit(self.pending.get(timeout=0.01), live)
                    except queue.Empty:
                        continue
                # A finite drain keeps incoming traffic from starving decode.
                for _ in range(self.pending.maxsize):
                    try:
                        job = self.pending.get_nowait()
                    except queue.Empty:
                        break
                    self._admit(job, live)
                try:
                    events = self.engine.step()
                    for event in events:
                        job = live[event.seq_id]
                        job.count += 1
                        now = time.monotonic()
                        self.metrics.count("generated_tokens")
                        if job.last_token_at is not None:
                            self.metrics.observe("itl_seconds", now - job.last_token_at)
                        job.last_token_at = now
                        eos = event.token_id == self.eos_id
                        content, matched = job.completion.feed(
                            b"" if eos else _token_bytes(self.engine.tokenizer, event.token_id),
                            final=event.finished or eos,
                        )
                        finished = event.finished or eos or matched
                        if eos or matched:
                            job.finish_reason = "stop"
                        if content and job.first_text_at is None:
                            job.first_text_at = now
                            self.metrics.observe("ttft_seconds", now - job.submitted_at)
                        try:
                            if not job.cancelled.is_set():
                                job.output.put_nowait((content, finished, job.count))
                        except queue.Full:
                            job.abort(APIError("client fell behind the output queue", 503))
                        if finished or job.cancelled.is_set():
                            self._release(job)
                            del live[event.seq_id]
                except Exception:
                    self.failed.set()
                    self.stopping.set()
                    self._diagnostic(traceback.format_exc())
                    for job in live.values():
                        job.abort(APIError("engine generation failed", 500))
                        self._release(job)
                    live.clear()
                self._publish()
        except Exception:
            self.failed.set()
            self._diagnostic(traceback.format_exc())
        finally:
            self.stopping.set()
            for job in live.values():
                job.abort(APIError("server is stopping", 503))
                self._release(job)
            while not self.pending.empty():
                job = self.pending.get_nowait()
                job.abort(APIError("server is stopping", 503))
                self._release(job)

            self._publish()

    def begin_drain(self, timeout=10.0):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("drain timeout must be finite and non-negative")
        with self.lock:
            if not self.draining.is_set():
                self.drain_deadline = time.monotonic() + timeout
                self.draining.set()

    def close(self, drain_timeout=None):
        if drain_timeout is None:
            with self.lock:
                self.stopping.set()
        else:
            self.begin_drain(drain_timeout)
        # In-flight device work is not preemptible; release at the next boundary.
        self.thread.join()


def usage(prompt_tokens, completion_tokens):
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "layers/0.1"
    timeout = 5
    disable_nagle_algorithm = True

    def log_message(self, format, *args):
        pass

    def _headers(self, status, content_type, length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Connection", "close")
        self.send_header("x-request-id", self.request_id)
        if length is not None:
            self.send_header("Content-Length", str(length))
        else:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True

    def _json(self, body, status=200):
        data = json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode()
        self._headers(status, "application/json", len(data))
        self.wfile.write(data)

    def _sse(self, body):
        data = body if isinstance(body, str) else json.dumps(body, ensure_ascii=True, separators=(",", ":"))
        self.wfile.write(("data: " + data + "\n\n").encode())
        self.wfile.flush()

    def _disconnect(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _check_peer(self, job):
        if job.cancelled.is_set():
            raise ConnectionAbortedError()
        readable, _, _ = select.select([self.connection], [], [], 0)
        if readable and not self.connection.recv(1, socket.MSG_PEEK):
            raise ConnectionAbortedError()

    def do_GET(self):
        self.request_id = "req-" + uuid.uuid4().hex
        try:
            if self.path in ("/healthz", "/readyz", "/metrics"):
                bridge = self.server.bridge
                body = bridge.snapshot()
                if self.path == "/metrics":
                    body["model"] = getattr(bridge.engine, "artifact_manifest", {})
                    body["engine"] = type(bridge.engine).__name__
                healthy = bridge.thread.is_alive() and not bridge.failed.is_set()
                status = 200 if self.path == "/metrics" or (
                    bridge.ready if self.path == "/readyz" else healthy) else 503
                self._json(body, status)
                return
            if self.path != "/v1/models":
                raise APIError("unknown endpoint", 404)
            self._json({"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "created": self.server.created,
                "owned_by": "layers", "description": DESCRIPTION,
            }]})
        except APIError as exc:
            self._json(exc.body(), exc.status)
        except OSError:
            pass

    def do_POST(self):
        self.request_id = "req-" + uuid.uuid4().hex
        job = None
        try:
            if self.path != "/v1/chat/completions":
                raise APIError("unknown endpoint", 404)
            if self.headers.get_content_type() != "application/json":
                raise APIError("Content-Type must be application/json", 415)
            if self.headers.get("Transfer-Encoding"):
                raise APIError("send a Content-Length body, not transfer encoding")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise APIError("invalid Content-Length") from exc
            if not 0 < length <= 1_048_576:
                raise APIError("body must contain 1..1048576 bytes", 413)
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ConnectionAbortedError()
                def invalid_constant(value):
                    raise ValueError(value)
                body = json.loads(raw, parse_constant=invalid_constant)
            except (ValueError, UnicodeError) as exc:
                raise APIError("invalid JSON body") from exc
            prompt, count, sampling, stream, include_usage = parse_request(body, self.server.bridge.engine)
            job = self.server.bridge.submit(prompt, count, sampling, self._disconnect,
                                             stops=validate_stop(body.get("stop")),
                                             request_id=self.request_id)
            while not job.ready.wait(0.01):
                self._check_peer(job)
            if job.error:
                raise job.error
            base = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": MODEL_ID}

            def chunk(delta, finish_reason=None, choices=None, token_usage=None):
                result = {**base, "object": "chat.completion.chunk", "choices": choices if choices is not None else [
                    {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
                ]}
                if include_usage:
                    result["usage"] = token_usage
                return result

            if stream:
                self._headers(200, "text/event-stream")
                self._sse(chunk({"role": "assistant", "content": ""}))
            parts = []
            peer_check_at = time.monotonic()
            while True:
                if job.cancelled.is_set():
                    raise ConnectionAbortedError()
                # Writes already detect a broken SSE socket. Poll EOF at most
                # every 10 ms (also in JSON mode), rather than releasing the
                # GIL for select on every generated token.
                now = time.monotonic()
                if now >= peer_check_at:
                    self._check_peer(job)
                    peer_check_at = now + 0.01
                try:
                    content, finished, generated = job.output.get(timeout=0.01)
                except queue.Empty:
                    continue
                if stream:
                    if content:
                        self._sse(chunk({"content": content}))
                else:
                    parts.append(content)
                if finished:
                    token_usage = usage(len(prompt), generated)
                    if stream:
                        self._sse(chunk({}, job.finish_reason))
                        if include_usage:
                            self._sse(chunk({}, choices=[], token_usage=token_usage))
                        self._sse("[DONE]")
                    else:
                        self._json({**base, "object": "chat.completion", "choices": [{
                            "index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                            "logprobs": None, "finish_reason": job.finish_reason,
                        }], "usage": token_usage})
                    break
        except APIError as exc:
            self._json(exc.body(), exc.status)
        except OSError:
            pass
        finally:
            if job is not None:
                job.cancelled.set()


class ChatServer(ThreadingHTTPServer):
    daemon_threads = False
    request_queue_size = 32

    def __init__(self, address, engine, **bridge_options):
        if engine.tokenizer is None:
            raise ValueError("HTTP serving requires a tokenizer")
        super().__init__(address, ChatHandler)
        self.created = int(time.time())
        self.drain_timeout = bridge_options.pop("drain_timeout", None)
        self.bridge = EngineBridge(engine, **bridge_options)

    def server_close(self):
        if hasattr(self, "bridge"):
            self.bridge.close(self.drain_timeout)
        super().server_close()


def make_server(host, port, engine, **bridge_options) -> ThreadingHTTPServer:
    return ChatServer((host, port), engine, **bridge_options)


def load_engine(checkpoint=CHECKPOINT, tokenizer_path=TOKENIZER, engine_name="l6",
                manifest_path=MODEL_MANIFEST):
    from layers import BatchedSamplingEngine

    if engine_name not in ("l6", "l7"):
        raise ValueError("engine must be l6 or l7")
    model, tokenizer, manifest = load_assets(checkpoint, tokenizer_path, manifest_path=manifest_path)
    cls = BatchedEngine if engine_name == "l6" else BatchedSamplingEngine
    engine = cls(model, tokenizer)
    engine.artifact_manifest = manifest
    return engine


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8611)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--engine", choices=("l6", "l7"), default="l6")
    parser.add_argument("--manifest", type=Path, default=MODEL_MANIFEST)
    parser.add_argument("--drain-timeout", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.drain_timeout) or args.drain_timeout < 0:
        parser.error("--drain-timeout must be finite and non-negative")
    server = make_server(args.host, args.port, load_engine(args.checkpoint, args.tokenizer, args.engine,
                                                         args.manifest),
                         drain_timeout=args.drain_timeout)

    def stop(signum, frame):
        server.bridge.begin_drain(args.drain_timeout)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Serving {MODEL_ID} at http://{args.host}:{server.server_port}; {DESCRIPTION}", flush=True)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
