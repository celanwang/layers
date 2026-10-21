"""OpenAI-compatible text chat subset, implemented without a web framework.

Run ``.venv/bin/python server/app.py --port 8611``. The L3 checkpoint is a
500-step base model, not instruction tuned. Messages use ``role: content``
lines, without an invented assistant suffix. Only the documented text subset
is accepted; unsupported generation options are rejected rather than ignored.
"""

from __future__ import annotations

import argparse
import codecs
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
        lines.append(f"{item['role']}: {item['content']}")
    return "\n".join(lines)


def parse_request(body, engine):
    if not isinstance(body, dict):
        raise APIError("request must be a JSON object")
    supported = {"model", "messages", "stream", "max_tokens", "temperature",
                 "top_p", "seed", "stream_options"}
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
    decoder: object = field(default_factory=lambda: codecs.getincrementaldecoder("utf-8")("replace"))

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

    def __init__(self, engine, *, pending_size=32, output_size=128):
        if pending_size <= 0 or output_size <= 0:
            raise ValueError("queue capacities must be positive")
        import mlx.core as mx

        # A lazy graph may refer to its creator's thread-local stream. Evaluate
        # weights before handing sole inference ownership to another thread.
        mx.eval(engine.model.parameters())
        self.engine, self.output_size = engine, output_size
        self.pending = queue.Queue(pending_size)
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="layers-engine", daemon=True)
        self.thread.start()

    def submit(self, prompt_ids, max_tokens, sampling, disconnect=None):
        job = Generation(prompt_ids, max_tokens, sampling, queue.Queue(self.output_size), disconnect)
        with self.lock:
            if self.stopping.is_set():
                raise APIError("server is stopping", 503)
            try:
                self.pending.put_nowait(job)
            except queue.Full as exc:
                raise APIError("request queue is full", 429, code="capacity_exceeded") from exc
        return job

    def _release(self, job):
        if job.seq_id is not None:
            self.engine.cancel_request(job.seq_id)
        job.done.set()

    def _admit(self, job, live):
        try:
            if not job.cancelled.is_set():
                job.seq_id = self.engine.add_request(
                    job.prompt_ids, Sampler(**job.sampling), job.max_tokens
                )
                live[job.seq_id] = job
        except RequestCapacityError as exc:
            job.error = APIError(str(exc), 429, code="capacity_exceeded")
        except Exception:
            traceback.print_exc()
            job.error = APIError("engine admission failed", 500)
        finally:
            job.ready.set()
            if job.seq_id is None:
                job.done.set()

    def _run(self):
        live = {}
        try:
            while not self.stopping.is_set():
                # Cancel before admitting, so reclaimed capacity is immediately reusable.
                for seq_id, job in list(live.items()):
                    if job.cancelled.is_set():
                        self._release(job)
                        del live[seq_id]
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
                        content = job.decoder.decode(
                            _token_bytes(self.engine.tokenizer, event.token_id),
                            final=event.finished,
                        )
                        try:
                            if not job.cancelled.is_set():
                                job.output.put_nowait((content, event.finished, job.count))
                        except queue.Full:
                            job.abort(APIError("client fell behind the output queue", 503))
                        if event.finished or job.cancelled.is_set():
                            self._release(job)
                            del live[event.seq_id]
                except Exception:
                    traceback.print_exc()
                    for job in live.values():
                        job.abort(APIError("engine generation failed", 500))
                        self._release(job)
                    live.clear()
        finally:
            for job in live.values():
                job.abort(APIError("server is stopping", 503))
                self._release(job)
            while not self.pending.empty():
                job = self.pending.get_nowait()
                job.abort(APIError("server is stopping", 503))
                self._release(job)

    def close(self):
        with self.lock:
            self.stopping.set()
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
            job = self.server.bridge.submit(prompt, count, sampling, self._disconnect)
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
            while True:
                self._check_peer(job)
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
                        self._sse(chunk({}, "length"))
                        if include_usage:
                            self._sse(chunk({}, choices=[], token_usage=token_usage))
                        self._sse("[DONE]")
                    else:
                        self._json({**base, "object": "chat.completion", "choices": [{
                            "index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                            "logprobs": None, "finish_reason": "length",
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
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, address, engine, **bridge_options):
        if engine.tokenizer is None:
            raise ValueError("HTTP serving requires a tokenizer")
        super().__init__(address, ChatHandler)
        self.created = int(time.time())
        self.bridge = EngineBridge(engine, **bridge_options)

    def server_close(self):
        if hasattr(self, "bridge"):
            self.bridge.close()
        super().server_close()


def make_server(host, port, engine, **bridge_options) -> ThreadingHTTPServer:
    return ChatServer((host, port), engine, **bridge_options)


def load_engine(checkpoint=CHECKPOINT, tokenizer_path=TOKENIZER):
    # MLX is the existing project runtime, not a serving dependency.
    import mlx.core as mx

    model = GPT(vocab_size=1028, n_layer=4, d_model=512, n_head=8, ctx=256)
    for name, value in mx.load(checkpoint).items():
        parts = name.split(".")
        current = model
        for part in parts[:-1]:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        setattr(current, parts[-1], value)
    mx.eval(model.parameters())
    return BatchedEngine(model, BPETokenizer.load(str(tokenizer_path)))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8611)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    args = parser.parse_args()
    server = make_server(args.host, args.port, load_engine(args.checkpoint, args.tokenizer))

    def stop(signum, frame):
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
