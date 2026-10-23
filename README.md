# layers

Build to understand — LLMs from scratch, layer by layer.

A local learning project spanning byte BPE, attention, a trained 13.1M GPT,
paged inference, quantization, batching and an HTTP/SSE service. The bundled
configuration is a 500-step Tiny Shakespeare **base model**, with 256-token
context and 1,028 token IDs. It is not instruction tuned; accepting a chat
schema does not give it assistant or tool-use capabilities.

## Local setup

Validated on Apple M4 Max, 48 GiB unified memory, macOS 15.8, Python 3.12.15.
The requirements record existing validated versions. No new runtime library
was introduced by the service completion work.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-runtime.txt
# For all Oracle tests and experiments:
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

## Model assets

The tokenizer is tracked at `results/l1/tokenizer-1024.json`. Large checkpoints
are local assets ignored by Git. Restore `results/l3/ckpt-final-500.npz` from
your existing experiment/backup before starting the server. The expected
checkpoint/tokenizer fingerprints are in `server/model-manifest.json`.
Startup verifies the exact parameter names, shapes, fp32 dtype, finite values,
vocabulary size and both SHA-256 hashes before installing weights.

In a **fresh checkout**, reproduce the training with:

```sh
.venv/bin/python scripts/train.py
.venv/bin/python -m server.runtime \
  --checkpoint results/l3/ckpt-final-500.npz \
  --tokenizer results/l1/tokenizer-1024.json \
  --training-metrics results/l3/train-metrics-500.json \
  --output results/l3/local-service-manifest.json
```

Training writes L3 experiment outputs; preserve existing experiment files before
running it in an existing checkout. GPU training may not reproduce the original
checkpoint bit for bit. Register a *new* manifest from your known training run;
registration checks its model/tokenizer metadata and refuses to overwrite an
existing manifest. This is a local provenance binding, not a cryptographic
signature proving the origin of training. Start with `--manifest
results/l3/local-service-manifest.json` when using that reproduction.

## Start and call the service

```sh
.venv/bin/python server/app.py --host 127.0.0.1 --port 8611 --engine l6
# Opt into L7's vectorized page materialization and batched sampling:
.venv/bin/python server/app.py --port 8611 --engine l7 --drain-timeout 10
```

L6 remains the default: L7 improves batch-8 throughput in archived experiments
but has a small singleton regression. Neither path implements a custom Metal
kernel reading compressed pages directly. Both use the same validated assets.
This release is intended for loopback/local use. Authentication, TLS, hot model
swapping, tools and distributed serving are outside its contract.

```sh
curl -s http://127.0.0.1:8611/readyz
curl -s http://127.0.0.1:8611/metrics
curl -sN http://127.0.0.1:8611/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"layers-gpt","messages":[{"role":"user","content":"ROMEO: "}],"temperature":0,"max_tokens":32,"stop":["\n\n"],"stream":true,"stream_options":{"include_usage":true}}'
```

Set `stream:false` or omit `stream` for one JSON response. Messages are joined
as `role: content` lines, without an invented instruction-training template.
Supported generation fields are `max_tokens`, `temperature`, `top_p`, `seed`,
`stop`, and stream usage. Unsupported fields are rejected. `/v1/models` lists
the local model. SSE uses incremental `delta.content` frames and `[DONE]`.

## Termination and resource ownership

EOS (`<|endoftext|>`) terminates automatically and is not emitted as text.
`stop` accepts one string or 1–4 strings, each 1–256 Unicode characters.
The first fully completed match terminates; if matches finish together, the
one starting earliest wins. Stop text and text following it are suppressed.
Potential stop prefixes and incomplete UTF-8 are held until disambiguated.
An unmatched prefix is flushed when the length limit is reached.

`finish_reason` is `stop` for EOS/string stops and `length` for the budget.
Usage counts sampled tokens, including the token triggering EOS/stop; output
character count and token usage therefore differ. The engine may already have
computed the next logits before the service notices a stop. It cancels the
request at that iteration boundary; other rows retain their original RNG order.
The library engine/generator defaults remain unchanged.

One owner thread controls model calls, scheduling and KV mutation. HTTP threads
communicate through bounded queues. Capacity rejection returns 429; stopped
admission during drain returns 503. Disconnects, slow-consumer overflow and
errors cancel the affected request and release its pages. There is no eviction
or resumable preemption.

SIGINT/SIGTERM first closes admission and readiness, drains accepted work up to
`--drain-timeout`, then cancels unfinished work. An in-flight device operation
cannot be interrupted, so reclamation occurs at its next boundary. HTTP worker
threads finish buffered responses; socket timeouts can add up to 5 seconds.
This supports drain-and-restart, not zero-downtime hot reload.

## Observability

- `/healthz`: 200 while the engine owner is alive without a fatal error; otherwise 503.
- `/readyz`: additionally requires open admission; draining returns 503.
- `/metrics`: JSON snapshot of submitted/completed/rejected/cancelled/error
  counters, sampled-token count, active/waiting requests and allocated pages.
  Queue/pool gauges are published by the owner at iteration boundaries.
- Latency windows retain at most 1,024 observations. TTFT measures submit to
  first nonempty text ready in the owner (queue, prefill and stop buffering
  included); ITL measures successive token-ready events per request. These
  exclude socket delivery and are not SSE chunk-interval measurements. Empty
  EOS-only responses have no text-TTFT sample.
- JSON request-end logs go to stderr with request ID, outcome, duration,
  completion tokens and finish reason. Prompt/completion content is not logged.
  Admission rejections are distinct from generation errors. Metrics are
  process-local and reset on restart; the endpoint is not Prometheus format.

## Validation and learning records

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/final_service_bench.py
```

The service benchmark compares direct and HTTP/SSE requests using the same
checkpoint and L6/L7 implementations, B=1/4/8, paired alternating order,
3 warmups and 10 observations. It records raw timings, median/p90, first-text
latencies, owner token intervals, asset/source hashes and reclamation checks.
The <5% API-overhead target applies to the singleton comparison. Concurrent
results include different admission/arrival patterns and are reported without
inventing a pure transport-only time share. Failed targets remain in results.

Research evidence lives in `tasks/` (specifications), `runs/` (principles,
implementation/comparison notes and retrospectives), and `results/` (raw data).
L0–L7 numerical/performance Oracles remain available. Service v1 documentation
is in `runs/service-finalize-*.md` and `results/service-finalize/`.
