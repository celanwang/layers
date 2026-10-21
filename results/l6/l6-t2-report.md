# L6-T2 API server execution report

L6-T2 is implemented on `l6-serving`. All three task-section-5 acceptance
commands pass. The final suite reports **119 passed**, API overhead is
**3.70% (<5%)**, and **182** direct/API output comparisons pass. Eight
concurrent streams complete without cross-talk; normal completion, early
client disconnect, overflow and shutdown reclaim their paged KV resources.
No push was performed.

## Implementation and scope

- `server/app.py`: stdlib `ThreadingHTTPServer`, explicit routing for
  `POST /v1/chat/completions` (JSON or SSE) and `GET /v1/models`; L3 checkpoint
  and L1 tokenizer loading, configurable checkpoint/tokenizer paths and port.
  API descriptions disclose that this is a 500-step base model without
  instruction tuning. Messages use newline-joined `role: content` lines.
- A dedicated engine thread owns all admission, sampling, steps and KV
  mutation. Bounded ingress (32 requests) and output (128 token events per
  request) queues bridge request threads. Overflow aborts only the affected
  stream and closes its socket; cancellation Events work even when ingress
  is full. Peer EOF polling is bounded to 10 ms, with write errors detected
  immediately. Cancelled work is released at iteration boundaries.
- Per-request Samplers and incremental UTF-8 decoders preserve seed/temperature
  isolation and buffer split characters. SSE uses content deltas and a terminal
  length chunk followed by `[DONE]`. Optional `stream_options.include_usage`
  adds a final usage chunk with empty choices; JSON reports usage directly.
- `layers/engine.py`: the **only** change is the additive, idempotent
  `cancel_request` method, which releases active pages and drops request history.
  All **12 existing methods are AST-identical**. All **10 existing test files
  are byte-identical**, covering the unchanged L1–L6-T1 tests (98 tests).
- `tests/test_api_server.py`: 21 cases covering wire format/schema, 8 distinct
  concurrent prompts with different seeds/temperatures/top_p, direct-engine
  equality, normal and early-disconnect cleanup in both SSE and JSON modes,
  bounded ingress/output overflow, capacity errors, shutdown, invalid requests,
  and multibyte UTF-8 split across token boundaries.
- `bench/l6_api_bench.py`: L0-protocol direct/API overhead, 1/4/8-way scaling,
  per-request latency, output oracle, raw timing records and prediction gates.
  Serving uses only stdlib plus existing layers/MLX runtime components. There
  are **zero new dependencies** and no dependency/version changes.

The blind implementation used only this repository, public MLX API docstrings,
and OpenAI's public protocol documentation. No inference serving framework
implementation source was read. Schema sources:
[chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events),
[models](https://developers.openai.com/api/reference/resources/models/methods/list).

## Actual changed file list

Scope baseline: `0280dbcbb27e`. All final repository changes are within the
whitelist. The pre-existing untracked
`results/l5/bf16-train-20261021T132935962922Z.json` is untouched and excluded.

```text
bench/l6_api_bench.py
layers/engine.py
results/l6/api-bench-20261021T143438682160Z.json
results/l6/api-bench-20261021T143755909668Z.json
results/l6/api-bench-20261021T144110591270Z.json
results/l6/api-bench-acceptance.txt
results/l6/api-bench-attempt1.txt
results/l6/api-bench-attempt2.txt
results/l6/api-bench-attempt3.txt
results/l6/api-cli-shutdown.txt
results/l6/api-client-process-probe.json
results/l6/api-curl-acceptance.txt
results/l6/api-curl-full.txt
results/l6/api-curl-headers.txt
results/l6/api-curl-job-control.txt
results/l6/api-curl-oracle.json
results/l6/api-overhead-profile.json
results/l6/api-scope-audit.json
results/l6/api-tests-acceptance.txt
results/l6/api-tests-attempt1.txt
results/l6/api-tests-polling.txt
results/l6/api-tests-stage1.txt
results/l6/bf16-train-20261021T143553890918Z.json
results/l6/bf16-train-20261021T143635985662Z.json
results/l6/bf16-train-20261021T144225563415Z.json
results/l6/l6-t2-report.md
runs/l6-implementation-notes.md
server/__init__.py
server/app.py
tests/test_api_server.py
```

## Acceptance commands and complete output

### 1. Full regression suite — exit 0

```bash
.venv/bin/python -m pytest tests/ -q
```

```text
........................................................................ [ 60%]
...............................................                          [100%]
=============================== warnings summary ===============================
tests/test_kv_quant.py::test_kv_roundtrip_bound_and_pytorch_public_api_oracle
  /Users/wangchen/Developer/layers/tests/test_kv_quant.py:38: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
    oracle = torch.quantize_per_channel(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
119 passed, 1 warning in 25.01s
```

### 2. API benchmark — exit 0

```bash
.venv/bin/python bench/l6_api_bench.py
```

```text
device: Apple M4 Max, 48 GiB, MLX 0.32.4
config_hash: 4474cab6ff21 (commit 0aa7dec7ec87)
overhead: direct 2.2592 ms/token, API 2.3429 ms/token, 3.70%
concurrency=1: 418.23 tok/s; request median 0.152961..0.152961s, p90 0.160074..0.160074s
concurrency=4: 991.17 tok/s; request median 0.257999..0.258120s, p90 0.266515..0.266754s
concurrency=8: 1298.17 tok/s; request median 0.393711..0.394206s, p90 0.397008..0.397820s
oracle: 182 request comparisons passed; all trial pools reclaimed
predictions: {"api_overhead_below_5_percent": true, "api_stream_equals_direct_engine": true, "eight_streams_isolated_completed_reclaimed": true}
written: results/l6/api-bench-20261021T144110591270Z.json
PASS: API overhead <5%, 1/4/8 streams isolated/completed/reclaimed, direct-engine text oracle
```

### 3. Actual curl SSE capture — exit 0

Executed in Bash with an additional `wait` after the specified command solely
to collect the background server's graceful exit. The command and 20-line SSE
capture are otherwise unchanged. No server is left listening on port 8611.

```bash
.venv/bin/python server/app.py --port 8611 & sleep 2 && curl -sN -X POST localhost:8611/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"layers-gpt","messages":[{"role":"user","content":"To be or not to be"}],"stream":true,"max_tokens":32,"temperature":0}' | head -20; kill %1
```

```text
Serving layers-gpt at http://127.0.0.1:8611; L3 500-step base model; not instruction tuned. Messages are joined as role: content lines. Text-only chat subset.
data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"\n"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Of"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"re"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ven"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ge"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ther"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":", "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-6f9b064e09834f8782391f795b770f7e","created":1792593928,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"the "},"logprobs":null,"finish_reason":null}]}
```

The required `head -20` truncates before termination. An additional **real,
untruncated curl** with the same request confirms `[DONE]` and text equality
against the direct engine. It receives 34 JSON chunks
(role + 32 content deltas + terminal), then `[DONE]`. Its HTTP headers and
complete SSE output follow; the test uses seed 42, the default for this request.

```text
HTTP/1.1 200 OK
Server: layers/0.1 Python/3.12.15
Date: Wed, 21 Oct 2026 14:43:41 GMT
Content-Type: text/event-stream
Connection: close
x-request-id: req-88ad51bf7c4b4bbfb02671355d395990
Cache-Control: no-cache
```

```text
data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"\n"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Of"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"re"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ven"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ge"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ther"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":", "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"the "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"w"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"a"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"li"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"tt"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"le "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"to "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"m"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"en"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"\n"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"the "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"s"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"or"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"row"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"our"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" s"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"u"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"b"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"le"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"d "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"to "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"my "},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"s"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-0e548c5676614fdd94a2601105332b73","created":1792593821,"model":"layers-gpt","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"length"}]}

data: [DONE]
```

`api-curl-oracle.json` records curl exit 0, `[DONE]`, exact concatenated-text
agreement, 32 completion tokens, and zero directly generated pool occupancy.
The standalone CLI process also exits 0 after one PID-directed SIGTERM in
0.073576 s (`api-cli-shutdown.txt`). The first noninteractive
Zsh job-control capture needed additional PID cleanup; the final Bash
job-control acceptance above completes by itself.

Additional checks: `.venv/bin/python -m compileall -q server layers/engine.py
 tests/test_api_server.py bench/l6_api_bench.py` exits 0 with empty output.
The complete source/documentation diff passes whitespace checking:

```bash
git diff --check 0280dbcbb27e HEAD -- . ':(exclude)results/l6/api-curl*.txt'
```

Raw curl text captures deliberately preserve HTTP CRLF and SSE's final blank
separator. An unrestricted Git whitespace check flags those protocol bytes as
trailing whitespace / blank-at-EOF; they are retained rather than edited.
The wire captures instead pass explicit CRLF/SSE framing and JSON parsing
checks, including the `[DONE]` terminator. There is no separate lint
configuration in the repo. `api-scope-audit.json` verifies unchanged existing
tests/methods, whitelist compliance, import roots, and exact accepted-benchmark
source hashes.

## Three benchmark groups

Hardware: **Apple M4 Max, 48 GiB unified memory**, GPU fp32,
macOS 15.8 arm64, Python 3.12.15, MLX 0.32.4. Checkpoint:
`results/l3/ckpt-final-500.npz`; tokenizer: `results/l1/tokenizer-1024.json`.
Configuration hash: **`4474cab6ff21`**. Three warmups and ten timed repeats,
median/p90 and all raw samples, paired alternating direct/API order, fresh
engines/pools/servers per trial. Each request generates 64 tokens with the
same seed and parameters. Both paths include template/tokenization, prefill,
sampling, decode and text output; API also includes fresh TCP setup, JSON,
queues and SSE consumption. Model/server startup is outside timing.

1. **API overhead:** direct **2.2592 ms/token**, API **2.3429 ms/token**;
   **+3.7034%**, passing the strict <5% gate. Direct request median/p90:
   **0.144589 / 0.159697 s**; API: **0.149944 / 0.157769 s**.
2. **Concurrency scaling:** aggregate throughput and the range across individual
   requests' latency percentiles are below. Every request's ten samples and
   median/p90 are preserved separately in JSON.

| Streams | Aggregate tok/s | Request median range (s) | Request p90 range (s) |
|---|---:|---:|---:|
| 1 | 418.23 | 0.152961–0.152961 | 0.160074–0.160074 |
| 4 | 991.17 | 0.257999–0.258120 | 0.266515–0.266754 |
| 8 | 1298.17 | 0.393711–0.394206 | 0.397008–0.397820 |

3. **End-to-end oracle:** all **182** request comparisons pass, including
   every 1/4/8-way stream. Text is decoded independently from the direct
   engine's generated IDs. The untruncated curl additionally agrees with direct
   engine text. All benchmark trial pools are reclaimed.

The accepted record's `git_commit` is the pre-optimization parent `0aa7dec`;
its per-source SHA-256 hashes **exactly match** delivered implementation
commit `ca8da0b1062c6f4ada279c6e9c9f5a9aed56e78f`. This is verified in the
scope audit; the source hashes, rather than an assumed clean working tree,
identify the measured code. The measurement protocol was unchanged.

## Principle-note section 5 prediction verdicts

| T2 prediction | Evidence | Verdict |
|---|---|---|
| API overhead <5% | 3.7034% with the unchanged L0 protocol | PASS |
| Eight concurrent streams, zero cross-talk, all complete | HTTP test plus all 1/4/8-way benchmark outputs equal isolated direct runs | PASS |
| Pages reclaimed, including early disconnect | Deterministic tests pause a live iteration, verify allocated pages, close the client, resume, and assert zero blocks/sequences/history; JSON and SSE both covered | PASS |
| Per-request sampler isolation | Eight simultaneous requests vary seed, temperature and top_p; every output equals isolated execution | PASS |

T1 performance predictions belong to the already accepted T1 task; T2 leaves
its numerical paths unchanged and reruns all its tests. This report does not
claim a new T1 throughput measurement.

## Failures, deviations and limits

- The first exploratory benchmark overlapped test debugging and is preserved
  as `api-bench-20261021T143438682160Z.json`; it is not the acceptance result.
- The first **serial** benchmark failed at **+9.37%** overhead. Its raw data and
  output remain in `api-bench-20261021T143755909668Z.json` and
  `api-bench-attempt2.txt`. The fix limits unnecessary per-token EOF syscalls
  to 10 ms intervals. The accepted run uses the same configuration/protocol,
  with no relaxed threshold or altered baseline. Profiling and the negative
  separate-client-process probe are retained. The client/server process
  placement in the accepted harness was not changed.
- Initial API tests found an MLX lazy-weight/thread-stream boundary; weights
  are now evaluated before worker ownership. A subsequent failed full-suite
  run was a new disconnect-test socket-ownership mistake; its complete output
  remains in `api-tests-attempt1.txt`. Existing tests were never modified.
- The unchanged bf16 regression generates result JSON under `results/l5/` as
  a side effect. Only the three new records from this task were relocated
  intact into allowed `results/l6/`. No existing L5 file was edited or removed.
- The optional L5 INT8 path is not selected; T2 serves the required L3 fp32
  checkpoint. This is the specified minimal text-only compatibility subset,
  with unsupported options rejected explicitly. There is no instruction tuning
  or EOS/stop-string extension in the existing engine; completion terminates
  at the requested token limit with `finish_reason: length`.
- No task-specification changes, dependency changes, existing-test changes,
  framework-source reads, pushes or other scope expansion occurred.

## Stage commits

- `0aa7dec` — `feat(l6): serve chat completions over HTTP and SSE`
- `ca8da0b` — `perf(l6): reduce per-token peer polling`
- This report, final scope audit and final job-control capture are committed
  as `docs(l6): archive API serving acceptance` after the passing checks.
