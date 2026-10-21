# L6 comparison reading: execution report (-o)

Task: tasks/l6-diff-analysis.md, section 6. Base revision: 7293ad3.
No -o destination was supplied in the request; this report uses the allowed
results/l6/l6-diff-report.md path. Main analysis: [runs/l6-diff-analysis.md](../../runs/l6-diff-analysis.md).

## Five conclusions, one sentence each

1. The 4.115× gain leaves 173.083 ms per eight-request decode window above the illustrative 8× target, with B-dependent sampling/KV/state work the strongest optimization candidate and no isolated kernel trace proving pure dispatch dominance (§1).
2. FastAPI/Uvicorn can improve connection scaling, but the measured 3.70–4.08% net API delta supports no fixed async speedup forecast and frontend workers should retain a single GPU model owner initially (§2).
3. An admissible 256-token prompt has a linear model-prefill estimate of 4.865 ms extra delay for each of eight streams, while the current ctx=256 API rejects that prompt before prefill (§3).
4. Our 32/128 bounded queues reject or abort requests and reclaim KV, whereas resumable preemption and persistent prefix-cache eviction require separate state and scheduling semantics (§4).
5. The inventory contains 11 evidenced work items with landing recommendations, including all eight required capabilities, while speculative decoding and multi-LoRA are not worth implementing for the current model (§5).

## Key quantitative accounting

All values reuse archived records on Apple M4 Max / 48 GiB, macOS 15.8,
Python 3.12.15, MLX 0.32.4, GPU fp32, the L3 checkpoint, 3 warmups/10 repeats;
the main analysis states each configuration hash, median/p90 and sample limits.

| Quantity | Result and location |
|---|---|
| T1 same-run B=8 comparison | 1436.41 vs 349.04 tok/s = **4.11534×**, configuration **2d2e2aeed61b** (batching-bench-20261021T133044874890Z.json /throughput_scaling/3). |
| Illustrative 8× target | 2792.31 tok/s; decode window **183.361 ms** versus actual **356.443 ms**; excess **173.083 ms**, requiring **48.56%** less current decode time (diff-accounting.json /gap). |
| Prefill versus decode | B=8 prefill median/p90 **20.369/24.965 ms**, decode **356.443/372.846 ms**, end-to-end median **377.921 ms**; independent upper-median per-trial shares **5.13% prefill / 95.49% decode** need not sum to 100% (even-sized upper medians), while individual trials sum exactly (main §1; /scaling). |
| B scaling | Equivalent step walls B=1/2/4/8 **2.143 / 2.803 / 3.778 / 5.658 ms**; B=8 window **2.640×** B=1, aggregate growth **3.030×** over the optimized singleton; not isolated GPU times (/scaling, /gap). |
| Gap identity | **173.083 = −48.369 + 221.452 ms**, singleton improvement plus B-dependent window increment; endpoint descriptive fit **1.641 + 0.5022 B ms**, not causal kernel shares (/gap). |
| Padding | **44,352 / 108,864 = 40.741%** wasted slots; equal real work **64,512 slots**, near-equal allocation **65,016 slots**; unequal decode **431.245 ms** versus **420.911 ms**, **2.396%** throughput penalty; scaling-case row padding is **zero** (/padding; main §1). |
| Padding uncertainty | Across all five archived T1 trials, penalties span **−0.354% to +3.306%**; changing length variance/max length is not a padding-only ablation (main §1). |
| API | Accepted **+3.703%**, independently rerun **+4.080%**; extra **0.08367–0.09057 ms/token**, not a measured async-framework delta (/api). |
| Blocking scenario | L4 128-token prefill **2.432541/2.653791 ms median/p90**, configuration **fa0b5fac54b5**; linear 256 estimate **4.865082 ms** plus B=8 equivalent step **5.657828 ms** gives **10.522910 ms**; eight-stream delay sum **38.921 stream-ms** is not extra wall time (/prefill_blocking_estimate). |

Derived values and input SHA-256 hashes: [diff-accounting.json](diff-accounting.json).
No new performance measurement or ablation script was needed.

## Gap inventory count and landing recommendations

**11 work items**: 8 mandated rows + 3 runtime carry-overs; each row of main §5
has project path/lines, a specific production reference and a landing decision.

| # | Missing or partial capability | Landing recommendation |
|---:|---|---|
| 1 | EOS / stop strings | L7 serving correctness. |
| 2 | Persistent cross-request KV prefix cache (active sharing already exists) | L7 cache专项, optional performance work. |
| 3 | Metrics / structured logs / tracing | L7 operations; basic metrics/logs first, sampled tracing later. |
| 4 | Authentication / tenant rate limits | L7 edge/access before shared exposure. |
| 5 | TLS | L7 deployment at the proxy; implementing TLS itself is not worth doing. |
| 6 | Speculative decoding | Not worth doing now; separate algorithm experiment only with a larger target and evidence. |
| 7 | Multiple LoRA | Not worth doing now without trained adapters/use case. |
| 8 | Graceful reload / readiness / drain | L7 lifecycle. |
| 9 | Async transport / persistence / connection limits | L7 transport. |
| 10 | Token/SLO scheduling / chunked prefill / optional resumable preemption | L7 scheduler; overload rejection remains a valid minimal policy. |
| 11 | Direct paged/compressed execution and L5 integration | L7 runtime performance, not a prerequisite for correctness. |

These are proposals for Kimi, not authorization to start L7 or a claim that
all optional capabilities are minimum production requirements.

## Actual changed files

- runs/l6-diff-analysis.md — five-point comparison, calculations, local/production evidence and recommendations.
- results/l6/diff-accounting.json — derived values only, source hashes, configuration/protocol metadata and explicit estimate limitations.
- results/l6/l6-diff-report.md — this section-6 -o report.
- results/l6/diff-acceptance.json — documentary validation and scope audit.

No implementation, tests, existing bench/script, task specification, existing
measurement record, dependency or reference-source file is changed.

## Validation and stage commits

- **712faaa** — docs(l6): explain serving performance gaps (main analysis + accounting).
- Final archive stage uses **docs(l6): archive comparison acceptance**, after passing the final checks; it includes the independently calculated decode-share correction.
- Git whitespace check for the complete task diff: **PASS**.
- JSON syntax, five numbered sections, 11 evidence/landing rows, all required gap topics, local links, source line ranges, input hashes, and independently recomputed numbers: **PASS** (details in diff-acceptance.json).
- Scope against 7293ad3: **PASS**, only the four allowed artifacts above; no deletions.
- Task §5 explicitly excludes the test suite; **no tests or performance benchmarks run**, no dependencies installed; no configured lint tool exists, so document whitespace/structure/reference checks were used.
- **No push** performed.

Transient validation-script failures are retained here as execution history:
the first inventory checker incorrectly included batch-size rows from §1 and
raised AssertionError; its shell batch did not stop on that failure, so the
first commit proceeded after the staged whitespace check instead of after a
fully passing structural audit.
A second attempted checker had a PosixPath subscription typo and raised
TypeError before running its checks.
The checker issues were corrected; the subsequent structural/hash/arithmetic
audit passed with no document defect from those failures.
The expanded final audit also needed a parenthesis typo corrected and a
format-sensitive assertion changed to match the complete bolded blocking phrase;
neither required an analysis or data change.
Final archive commands use fail-fast execution.
During final content review, the complementary-share shortcut for even-sized
upper medians was corrected by independently calculating decode fractions.

## External material locations

The following 23 references were opened online; exact sections are listed here
and used by ID beside claims in the main artifact; source bodies are not stored.

- **O1** — [Orca, OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf), §3 S1/S2, PDF pp.6–7 (printed 525–526): iteration scheduling and selective batching.
- **S1** — [Sarathi-Serve, OSDI 2024](https://www.usenix.org/system/files/osdi24-agrawal.pdf), §4.1–4.3, Algorithm 3, PDF pp.8–10 (printed 123–125): chunks, decode-first hybrid batches, SLO token budget.
- **V1** — [vLLM optimization guide](https://docs.vllm.ai/en/v0.10.2/configuration/optimization.html#preemption), “Preemption”, “Chunked Prefill”, “Performance Tuning with Chunked Prefill”.
- **V2** — [PagedAttention paper](https://arxiv.org/pdf/2309.06180), §4.5, §5, §7.3/Figure 19: resumable preemption, direct block access, swap/recompute comparison.
- **V3** — [Automatic Prefix Caching introduction](https://docs.vllm.ai/en/v0.10.2/features/automatic_prefix_caching.html#introduction): shared-prefix prefill reuse.
- **V4** — [APC design](https://docs.vllm.ai/en/v0.10.2/design/prefix_caching.html#operations), “Block Allocation”, “Free”, “Eviction (LRU)”: hashed blocks and zero-ref retention.
- **V5** — [SamplingParams API](https://docs.vllm.ai/en/v0.10.2/api/vllm/sampling_params.html#vllm.sampling_params.SamplingParams), documented stop, stop_token_ids and ignore_eos fields.
- **V6** — [Metrics design](https://docs.vllm.ai/en/v0.10.2/design/metrics.html#objectives), “Objectives”, “v0 Metrics”, “Metrics Collection”, “Metrics Publishing - Prometheus”: distinguish versioned metric names from desired instrumentation.
- **V7** — [OpenTelemetry POC](https://docs.vllm.ai/en/v0.10.2/examples/online_serving/opentelemetry.html), setup steps 3–6 and “Instrumentation of FastAPI”: OTLP and trace context.
- **V8** — [vllm serve CLI](https://docs.vllm.ai/en/v0.10.2/cli/serve.html), api-key, scheduling-policy, preemption-mode: distinguish V0 swap options from V1's documented recompute default.
- **V9** — [Speculative decoding](https://docs.vllm.ai/en/v0.10.2/features/spec_decode.html), opening warnings, draft model and lossless-guarantee sections.
- **V10** — [LoRA adapters](https://docs.vllm.ai/en/v0.10.2/features/lora.html#serving-lora-adapters), “Serving LoRA Adapters”.
- **P1** — [Python 3.12 http.server](https://docs.python.org/3.12/library/http.server.html#http.server.ThreadingHTTPServer), production limitation and ThreadingHTTPServer.
- **P2** — [Python 3.12 socketserver](https://docs.python.org/3.12/library/socketserver.html#socketserver.ThreadingMixIn), ThreadingMixIn and request_queue_size.
- **P3** — [Python 3.12 threading](https://docs.python.org/3.12/library/threading.html), opening CPython GIL implementation note.
- **F1** — [FastAPI async/await](https://fastapi.tiangolo.com/async/#path-operation-functions), “Asynchronous Code” and “Path operation functions”.
- **F2** — [FastAPI server workers](https://fastapi.tiangolo.com/deployment/server-workers/), process replication using Uvicorn.
- **U1** — [Uvicorn server behavior](https://uvicorn.dev/server-behavior/), headers, flow control, timeouts, resource limits, exceptions, graceful shutdown.
- **U2** — [Uvicorn settings](https://uvicorn.dev/settings/), “Production”, “Implementation”, “Logging”, “HTTPS”, “Timeouts”.
- **U3** — [Uvicorn deployment](https://uvicorn.dev/deployment/), “Built-in” process manager, SIGHUP and worker restarts.
- **U4** — [Uvicorn HTTP/2](https://uvicorn.dev/concepts/http2/), experimental status, enabling HTTP/2 and reverse proxies.
- **E1** — [Envoy v1.35.0 local rate limit](https://www.envoyproxy.io/docs/envoy/v1.35.0/configuration/http/http_filters/local_rate_limit_filter), token bucket/429 and “Using rate limit descriptors”.
- **E2** — [Envoy global rate limiting](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting), per-request checks and coordinated quota implementations.

## Deviations and evidence limits

- **No scope, implementation, dependency or measurement-protocol deviation.**
- The unspecified -o destination is resolved to this file inside results/l6/.
- Exact per-B GPU kernel timings/launch counts cannot be recovered from archived
  wall-clock benchmarks/cProfile; the report gives per-step wall scaling and
  source-level counts rather than fabricating the requested causal split.
- Literal 256-token arrival is inadmissible with ctx=256 and positive output;
  the requested numerical scenario assumes context >=257, at least nine active
  slots and sufficient pages, and marks the 4.865 ms as a linear contiguous-KV
  estimate excluding unmeasured L6 paging/tokenization/queue costs.
- The initial stage-commit validation sequencing deviation and final correction
  are disclosed above; final documentary checks pass.
