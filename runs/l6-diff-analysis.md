# L6 serving: comparison reading and quantitative accounting

Analysis against task [l6-diff-analysis](../tasks/l6-diff-analysis.md), based on
repository revision **7293ad3**, read on **2026-10-21**. Implementation and Oracle
alignment preceded this reading ([implementation notes](l6-implementation-notes.md),
lines 85–120 and 183–219). This task changes documentation and derived accounting
only. Public papers/documentation were read online; reference source bodies were
not copied into the repository. Follow-up choices below are recommendations for
Kimi's review, not approved implementation work.

## Evidence and measurement boundary

Local evidence IDs used below identify files **and JSON pointers**, so results
can be checked without treating a rounded report value as a new measurement.

| ID | Existing evidence and location |
|---|---|
| T1 | [Independent T1 rerun](../results/l6/batching-bench-20261021T133044874890Z.json): /throughput_scaling, /uneven_lengths, /protocol, /source_sha256; configuration **2d2e2aeed61b**, recorded commit **00f544d295bb**. This is the specification's 1436 tok/s / 4.1× case. |
| T1A | [Earlier accepted T1 run](../results/l6/batching-bench-20261021T132523026893Z.json): same pointers; configuration **d2f97623b7b7**. B=8 is 1400.44 vs 310.22 tok/s, **4.514×**. Keep it separate from T1's baseline. |
| T2 | [Accepted API run](../results/l6/api-bench-20261021T144110591270Z.json): /api_overhead and /concurrent_scaling; configuration **4474cab6ff21**. |
| T2R | [Independent API rerun](../results/l6/api-bench-20261021T145054448274Z.json): same pointers; configuration **846a3910f2ad**. |
| P | [Existing API profile](../results/l6/api-overhead-profile.json): entries selected by mode and rep, with elapsed_s and profile. Exploratory cProfile, not a Metal trace or L0-protocol performance trial. |
| L4 | [128-token prefill measurement](../results/l4/decode-bench-20261021T114111Z.json): /prefill_vs_decode/prefill and /protocol; configuration **fa0b5fac54b5**. |
| A | [Derived arithmetic](../results/l6/diff-accounting.json): input SHA-256 hashes, source configurations, formulas and full-precision derived values; **no new timing samples**. |

T1/T1A/T2/T2R/L4 record Apple **M4 Max / 48 GiB**, macOS **15.8 arm64**,
CPython **3.12.15**, MLX **0.32.4**, GPU **fp32**, the L3 500-step checkpoint,
**3 warmups / 10 repetitions**, median/p90. T1/T2 use paired alternating order,
fresh engines/pools, and retain raw samples; L4's older prefill summary retains
median/p90/min but not individual samples. T1 uses 32-token distinct prompts,
64 generated tokens/request, block size 16, prefix sharing off. Its 64 samples
per request require **63 decode forwards**, since prefill supplies the first
token's logits. All T1 timings include sampling, scheduler, KV gather/pad/scatter
and synchronization; they are **not isolated kernel timings**
(bench/l6_batching_bench.py:82–105, 108–147, 152–162).

## 1. Why 4.1× is below the illustrative 8× target

**Conclusion.** The same-run speedup is **4.115×**: an eight-request decode
window takes **356.443 ms**, versus a constant-time 8× target of **183.361 ms**.
The **173.083 ms** excess belongs to decode, not prefill; the strongest next
candidate is reducing per-request sampling/synchronization and KV construction,
rather than chasing the separate 40.7% padding case. These are wall-clock and
source-level inferences, not a measured allocation of time to GPU dispatch
[T1 /throughput_scaling/3; A /gap; local paths below].

### Prefill and decode have different denominators

| B | Prefill median / p90 ms | Decode median / p90 ms | End-to-end median ms | Median per-trial prefill share | Decode ms per forward-equivalent | Aggregate decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.259 / 4.544 | 134.992 / 138.451 | 137.243 | 1.70% | 2.143 | 474.10 |
| 2 | 4.814 / 6.599 | 176.566 / 185.483 | 181.428 | 2.54% | 2.803 | 724.94 |
| 4 | 8.514 / 15.426 | 238.017 / 249.220 | 246.634 | 3.47% | 3.778 | 1075.55 |
| 8 | 20.369 / 24.965 | 356.443 / 372.846 | 377.921 | 5.13% | 5.658 | 1436.41 |

Source: T1 /throughput_scaling/0–3/batched; A /scaling. Prefill fractions are
computed within each paired component sample, then summarized with the bench's
upper median (sorted[n//2]); decode shares are the complementary **98.30%,
97.46%, 96.53%, 94.87%**. Separately summarized component medians need not sum to
the end-to-end median. Dividing the decode window by 63 gives a useful average
forward-equivalent, including the 64th terminal sampling iteration; it does not
turn that window into a GPU timer.

For B=8, **512 / 0.356443 = 1436.41 tok/s**, compared with the unchanged L4
eight-request rate **512 / 1.466885 = 349.04 tok/s**. Idealized 8× reuse would
give **2792.31 tok/s**, or **1.466885 / 8 = 183.361 ms** per window. Reaching
that target requires **48.56% less current decode time**. Deleting prefill cannot
change this decode-only rate; even the impossible removal of all prefill would
recover only about the measured 5.13% prefill share of each end-to-end trial
[T1 /throughput_scaling/3; bench/l6_batching_bench.py:124–130; A /gap].

The comparison with the optimized singleton is stricter: **356.443 / 134.992 =
2.640×** longer windows, hence only **3.030×** aggregate throughput growth from
B=1 to B=8. Its constant-window target would be **3792.83 tok/s**. The task's 8×
is an illustrative weight-reuse assumption inherited from
[L4 comparison §1](l4-diff-analysis.md), not an algorithmic guarantee: activations,
attention, per-request state and sampling still grow with B. Do not mix this
singleton, T1A's slower L4 baseline, and T1's baseline to create a larger gain.

### What scales after the shared forward

The dense projection graph is shared: **24 block projections plus 2 half-width
head GEMMs** per batched forward, independent of B (layers/attention.py:214–230;
layers/transformer.py:47–50, 82–86; layers/model.py:88–105). But “kernel count no
longer grows with B” is only defensible for that shared arithmetic structure:

- Sampling still loops over B requests and each sampler executes an evaluation
  and scalar read: **8 sampler evaluations per iteration at B=8**, versus one
  at B=1 (layers/engine.py:319–342; layers/inference.py:174–181).
- Four layers each call pool append/gather for every row: **32 row appends and
  32 gathers per forward at B=8**, versus four each at B=1. Page reconstruction,
  history concatenation and stacking remain outside the compiled arithmetic
  helpers (layers/engine.py:278–307; layers/paged_kv.py:237–306, 308–336).
- These are operation/call counts. Page length, views, fusion and backend lowering
  determine actual Metal launches. No existing record measures a B-sweep of
  isolated GPU kernel time or launch counts. P's direct_thread rep=1 attributes
  89 ms of 131 ms wall time to step's own frame and 16 ms to sampling, but the
  frame includes evaluation/waiting; cProfile cannot label this pure CPU work
  or establish its B=8 share [P entries; layers/engine.py:341–342].

A transparent endpoint model is **T_equiv(B) = 1.641 + 0.5022 B ms**. It matches
B=1 and B=8 and underpredicts B=2/4 by **0.158/0.129 ms**. At B=8 the fitted
row-dependent term is **4.017 ms (71.0%)**, with a **1.641 ms** intercept.
This is a descriptive scaling model, not “71% proven dispatch.” Its row term
also contains useful matrix/attention work, memory copies and synchronization.
Removing the entire row term would imply about **4954 tok/s**, an unattainable
zero-row-work ceiling that must not be sold as an optimization prediction
[A /gap/endpoint_affine_model].

An exact accounting identity for the observed shortfall is:

~~~
356.443 - 183.361 = (134.992 - 183.361) + (356.443 - 134.992)
       173.083 ms =          -48.369 ms +          221.452 ms
~~~

The singleton batching changes already save 48.369 ms relative to the idealized
per-request L4 window, while growing the batch adds 221.452 ms. This locates the
scaling problem; it does not partition those 221.452 ms among kernels and host
work [A /gap].

### Padding wastes attention slots, not 40.7% of the whole engine

T1's unequal and near-equal experiments have the **same mean prompt length 96**.
Over 63 forwards, each head/layer processes **64,512 real key slots**.
Near-equal lengths allocate **65,016 slots** (504 wasted; **0.775%**);
unequal lengths allocate **108,864 slots** (**44,352 wasted; 40.741%**).
Thus attention-slot work is **1.6875× real work** in the unequal case. Eliminating
padding would remove 40.7% of padded slot work, not 40.7% of decode time; padding
is temporary attention storage, not persistent page waste
[T1 /uneven_lengths; bench/l6_batching_bench.py:135–147;
layers/engine.py:296–307].

Measured decode median/p90 is **420.911/428.170 ms** near-equal versus
**431.245/438.044 ms** unequal: **10.334 ms** longer and **2.396% lower throughput**.
This changes both padding and maximum attention length, so it is not an
isolated padding ablation. All five retained runs' throughput penalties span
**−0.354% to +3.306%**, including the initial failed batching gate; no stable
40.7% engine-time loss is supported
[results/l6/batching-bench-*.json, /uneven_lengths/throughput_penalty_percent;
T1A: +2.171%; implementation notes:95–104].

A scale check explains the small effect: at average real KV length **128**,
attention's two matmuls cost roughly **4 L d n = 1.049 MFLOP/output token**,
versus about **2P = 26.253 MFLOP** for dense weights (P=13,126,656, L=4, d=512).
Attention matmuls are approximately **3.84%** of this simplified FLOP count;
padding raises their work by 68.75%, about **2.64%** of the original combined
count. This excludes softmax, copies, materializations and dispatch and is
not a time prediction [L4 comparison:13; layers/model.py:28–70;
layers/attention.py:79–88; A /padding]. Crucially, T1's **1436 tok/s scaling
case uses equal 32-token prompts and has zero row padding**, so its 173 ms
shortfall cannot be assigned to the separate 40.7% experiment.

### Highest-value candidate and historical verdict

For the measured equal-length decode deficit, prioritize **batched sampling
with fewer scalar barriers and KV attention that consumes pages directly**,
then broader arithmetic fusion. That targets B-dependent work present even
without padding. PagedAttention's kernel reads block tables in place, whereas
our path reconstructs contiguous histories [V2 §5; local call counts above].
The evidence ranks this candidate above padding removal and API changes;
it cannot establish which of sampling, gather or GEMM utilization has the
largest causal share without a separately approved component measurement.

This agrees with the project's broader small-operation/host-cost explanation,
with an essential qualification. The “five corroborations” are not five traces
proving pure launch dominance: L0 reports **224 vs 451 GB/s** small/large
bandwidth; L1's **0.7–1.0 MB/s** is Python object/heap overhead; L2's ~2×
small-attention improvement has an IO budget of only **4.65 µs** versus a
~0.2 ms gap; L4 bounds a **2.94 ms** residual as utilization + dispatch +
synchronization; L5 finds **3.619–4.542 ms** in quantize-new, with scalar
validation and evaluation costs. L3's large training batch instead reaches
**35.7% MFU**, a useful boundary to this explanation
[runs/l0-hw-profile.md:18; runs/l1-retrospective.md:14;
runs/l2-retrospective.md:17–25; runs/l4-retrospective.md:18;
runs/l5-diff-analysis.md:79–91; runs/l5-retrospective.md:15;
runs/l3-diff-analysis.md:11–30]. L6 reinforces that qualified interpretation;
it supplies no new Metal trace.

## 2. ThreadingHTTPServer versus FastAPI/Uvicorn

**Conclusion.** Async serving is primarily a connection/concurrency improvement;
the existing **3.70–4.08%** single-request API delta does not justify predicting
a fixed async speedup. Keep one model owner on this single GPU, and treat
frontend workers and GPU replicas as different decisions
[T2/T2R /api_overhead; server/app.py:133–154, 398–431; F1/F2; U1/U2].

Our server handles accepted connections with threads (P1, ThreadingHTTPServer;
P2, ThreadingMixIn). With C live SSE connections, the design has approximately
**C handler threads + one engine owner + one serving thread**, excluding clients
and library threads. At eight streams that is about ten application threads,
a structural count, not an observed peak. A queue of 32 pending generations and
a listen backlog of 32 **do not cap accepted connections/handler threads**;
body parsing/tokenization precedes ingress admission (server/app.py:325–337,
398–407). The practical ceiling depends on file descriptors, stack/address-space
and memory budgets, context switching, Python CPU work and socket timeouts;
the repository measures eight streams, not a universal 100/1000-thread limit.

CPython 3.12 allows only one thread to run Python bytecode at once, while
I/O waits can overlap (P3, opening implementation note). SSE waiting is therefore
not continuously GIL-bound, but tokenization, JSON/event construction, locks
and the owner's Python dispatch can contend. The already documented per-token
peer check correction reduced API overhead **9.37% → 3.70%** under the same
protocol; it is evidence for avoiding unnecessary polling, not a measurement
of all GIL contention (runs/l6-implementation-notes.md:183–205;
server/app.py:356–368).

A Uvicorn worker uses an event loop to multiplex async I/O. FastAPI's async
routes must await nonblocking work; normal def routes use a thread pool.
Moving blocking engine.step or queue.get directly into async def would block
the event loop. A future adapter should retain the engine owner and deliver
completion events asynchronously [F1, “Path operation functions” and
“Asynchronous Code”; U2, “Implementation”; local bridge above].

T2 measures **2.25920 → 2.34287 ms/token**, a net **0.08367 ms/token** or
**5.355 ms/64-token request**. T2R measures **2.22004 → 2.31061**, net
**0.09057 ms/token**, **4.080%**. Removing all of this measured net delta while
holding engine work fixed would reduce API elapsed time by only **3.57–3.92%**
[A /api]. Async can reduce idle thread and wakeup costs at high C, but parsing,
SSE serialization and queues remain, and validation/middleware add work.
**No numeric async overhead forecast is supported**; similar or lower overhead
is a hypothesis, and regression at C=1 is possible. These runs include fresh TCP
setup and same-process clients, not keep-alive pools or remote TLS
[T2 /protocol; results/l6/l6-t2-report.md:273–305]. P contains five exploratory
samples per mode without the benchmark metadata/replication and cannot be
added up into this delta.

For a production transport follow-up:

| Decision | Recommendation and evidence |
|---|---|
| Worker count | Start evaluation with **one frontend worker and one model owner**; multiple frontend processes can scale CPU handling only after introducing a shared-engine IPC boundary. Loading a model/pool in every worker fragments batches and creates GPU contention. F2 documents worker processes; server/app.py:419–431 creates a new model/engine per load. Select counts from measured connection load, not CPU core count alone. |
| HTTP/2 | SSE works over HTTP/1.1; HTTP/2 offers multiplexed streams. The public Uvicorn docs read for this task describe **experimental**, opt-in HTTP/2 via zttp; do not claim all Uvicorn versions lack it, or that default Uvicorn automatically enables it. Prefer a validated TLS/HTTP/2 edge and HTTP/1.1 upstream initially; native H2 requires version/protocol qualification [U4, “Enabling HTTP/2”, “Using with Reverse Proxies”]. |
| Keep-alive and pools | Our response forces Connection: close (server/app.py:262–272), so pooling alone cannot reuse connections. A production adapter should permit keep-alive and configure client/upstream pools and idle limits; long HTTP/1.1 SSE streams each occupy a connection. U1 “HTTP Headers”/“Timeouts” and U2 “Timeouts” describe persistence/keep-alive control. Existing 3.7–4.1% does not quantify its benefit. |
| Resource and slow-client limits | Bound concurrent connections and request size before expensive parsing; retain a separate token/page admission limit. Uvicorn transport flow control and concurrency limits help but do not implement LLM token budgets [U1 “Flow Control”/“Resource Limits”; server/app.py:325, 141–164]. |

The stdlib choice made thread ownership and SSE framing inspectable without
dependencies. P1 explicitly describes http.server as unsuitable for production;
replacing it belongs to a new serving task, not this comparison.

## 3. Iteration-boundary prefill versus Sarathi-Serve

**Conclusion.** An admitted 256-token prompt would add approximately **4.87 ms
of model-only prefill delay to each of eight ongoing streams** under a linear
extrapolation, giving an illustrative **10.52 ms** next-token interval;
the unmodified ctx=256 API instead rejects that prompt before prefill.
This is an estimate with explicit admission assumptions, not a new latency
measurement [L4 /prefill_vs_decode; A /prefill_blocking_estimate;
server/app.py:99–103, 423; layers/engine.py:158–164].

Our owner drains arrivals at a step boundary and calls add_request for each,
which runs an entire model prefill and mx.eval before engine.step.
Up to 32 pending items can be drained in that loop; the bound prevents infinite
arrival draining but does not bound work in prompt tokens
(server/app.py:189–211; layers/engine.py:183–205). Joining only between
iterations does not make admission nonblocking for ongoing decodes.

Orca's **§3 S1** schedules one iteration at a time; **S2** batches selected
non-attention work while handling different-length attention separately.
Our shared linear projections follow the batching motivation, but our padded
attention views make a different implementation tradeoff [O1, PDF pp.6–7,
printed pp.525–526; layers/attention.py:204–230]. The 2024 Sarathi comparison
describes historical Orca/vLLM baselines; it should not be read as a claim that
later vLLM still lacks chunked prefill [S1 §3–4; V1 “Chunked Prefill”].

Sarathi-Serve **§4.2, Algorithm 3 lines 6–23** first reserves one token for each
ongoing decode, then schedules unfinished prefills and new prompts within the
remaining token budget, executing a hybrid batch. **§4.3** chooses that budget
against the TBT SLO and chunking overhead. A conceptual budget of 72 with eight
decodes leaves **64 prefill tokens**, so a 256-token prompt spans **four hybrid
iterations**. This illustrative budget is not tuned for M4. “Stall-free” means
ongoing decodes continue each iteration; it does not mean chunks have zero
latency cost. Smaller chunks repeat overhead and KV access [S1 §4.2–4.3,
Algorithm 3; V1 “Chunked Prefill”].

The quantitative estimate deliberately separates facts and assumptions:

| Quantity | Calculation / evidence |
|---|---|
| Measured 128-token contiguous-KV prefill | **2.432541 ms median / 2.653791 ms p90**, same M4/MLX/fp32/checkpoint, 3 warmups/10 repeats [L4; bench/l4_decode_bench.py:181–205]. |
| Estimated one 256-token model prefill | **2 × 2.432541 = 4.865082 ms**, assuming locally linear token scaling. Doubling the source p90 gives **5.307582 ms**, a scenario value, **not a measured 256-token p90**. Attention also has quadratic work; fixed dispatch makes exact linearity uncertain. |
| B=8 normal decode interval proxy | **356.443167 / 63 = 5.657828 ms**, includes sampling/state/sync, not measured per-token ITL [T1]. |
| Effect on eight streams | Each waits an extra **4.865 ms**, concurrently; next interval proxy **5.658 + 4.865 = 10.523 ms**, about **1.86×** normal. The sum of eight delays is **38.921 stream-ms**, not 38.921 ms of wall-clock blocking [A]. |
| Admission preconditions | At least **9 active slots**, adequate pages, and context **≥257** for a 256-token input plus output. A bench configured to max_concurrent=8 also rejects a ninth request. Actual loader ctx=256 rejects positive generation for this prompt, so **additional model-prefill blocking is 0 ms** there; parsing/admission itself still has nonzero cost [server/app.py:99–103, 423; layers/engine.py:149–164]. |

L4 uses contiguous KV, whereas L6 rebuilds paged histories; the **4.87 ms excludes
unmeasured paging, tokenization and queue delay**. The T1 B=8 prefill measurement
is eight separate 32-token admissions (**20.369 ms**), not one 256-token prefill:
equating them would pay eight fixed dispatch paths and silently change the
workload (bench/l6_batching_bench.py:87–94). No existing record directly measures
the requested staggered-arrival scenario. A useful follow-up would compare
admissible long prompts with chunked admission using TTFT, per-token tail ITL,
and throughput; this analysis does not claim a measured chunking speedup.

## 4. Backpressure, admission and eviction

**Conclusion.** The bounded queues and cancellation semantics protect existing
requests, but they are overload rejection and abort, not resumable eviction;
production needs coordinated connection/token/page admission, fairness and
measured TTFT/ITL objectives [server/app.py:141–245; layers/engine.py:166–180,
265–275; V1/V2; E1/E2].

| Event | Current behavior and evidence | Production comparison |
|---|---|---|
| Ingress full | 32 generation jobs; immediate **429 capacity_exceeded**. The bridge queue is not a wait-for-GPU-capacity scheduler (server/app.py:156–164, 172–187). | Reject before expensive work when possible; capacity admission and identity quotas are distinct. Envoy's local token bucket returns 429 when exhausted [E1, opening and descriptors]. |
| Active/page limit | add_request reserves the worst-case prompt + output pages; failure returns **429** rather than delaying/preempting a live request (layers/engine.py:149–180; server/app.py:179–180). | vLLM can reclaim live-request KV and later **recompute**. The V1 guide names recomputation as its default, distinct from historical swap support [V1 “Preemption”; V2 §4.5]. |
| Slow stream | 128 output items per request; put_nowait overflow aborts that client, shuts its socket and reclaims at a boundary. No global decode wait (server/app.py:125–138, 219–226, 284–288). | Transport flow control alone can suspend send; the engine still needs bounded per-request buffering and a cancellation/deadline policy [U1 “Write flow control”]. Item count is not a global byte or connection cap. |
| Disconnect/completion | Cancel pops request history and frees live KV; physical pages survive only while another active fork holds a reference (layers/engine.py:265–275; layers/paged_kv.py:178–195). | Cancellation releases ownership; zero-ref cached prefixes may remain reusable until cache eviction [V4 “Free”/“Eviction (LRU)”]. No guarantee of reconnect/resume follows from either behavior. |

The 2023 PagedAttention paper's **§4.5** preempts whole sequences, describes
CPU swapping or rebuilding KV from prompt plus already generated IDs, and keeps
request state so generation can resume. **§7.3/Figure 19** compares their costs.
That is different from permanently aborting a disconnected or slow request.
Its CPU/GPU transfer model is also not directly transferable to M4 unified
memory; copying elsewhere in the same physical memory does not create capacity.
Treat recomputation as a candidate here, not a mandate to port CUDA swap
[V2 §4.5, §7.3; layers/engine.py:265–275; T1 /device].

A scoped production policy proposal is to combine edge request/connection
limits with token-aware engine admission, page headroom and a finite queue
deadline; reserve decode budget before prefill; choose fairness/aging rules for
priorities; shed requests that cannot meet a declared service objective.
E2 documents coordination across proxy instances, while vLLM's CLI documents
FCFS or explicit priority ordering; neither makes priority automatically
SLO-aware [E2, global rate limiting implementations; V8, scheduling-policy].
Sarathi's SLO-derived token budget provides the scheduling reference [S1 §4.3].
Measure queue time, TTFT, ITL tails, rejections, KV occupancy and preemptions
[V6, “Objectives”, “v0 Metrics”, “Metrics Collection”]. This is a proposed
policy assembled from those mechanisms, not a claim that vLLM implements this
entire policy or that our closed-loop 1/4/8-concurrency benchmark validates it.

After SSE's 200 headers are sent, output overflow cannot replace the HTTP status
with 503; our code closes the stream, so clients observe an incomplete stream,
not necessarily a JSON error envelope (server/app.py:223, 352–382; U1 “Exceptions”).
A future API contract should distinguish pre-admission errors, streamed errors
and client cancellation, and avoid transparent retries of already emitted text.

## 5. Whole-project gaps and proposed landing points

**Conclusion.** The inventory has **11 work items**: the specification's eight
named capabilities plus three runtime gaps established above and carried from
L4/L5. They are not eleven prerequisites for a local research endpoint.
Termination, basic observability, access controls/TLS when exposed, and controlled
lifecycle/overload behavior form the proposed minimum service scope; speculative
decoding and multi-LoRA do not currently justify implementation.
Evidence and proposed scope are explicit in every row.

| # | Capability | Project gap and concrete production reference | Proposed landing point |
|---:|---|---|---|
| 1 | Generation termination: EOS / stop string | Engine only tests token/context length (layers/engine.py:323–329); API rejects stop and always emits finish_reason length (server/app.py:76–80, 376–387). V5 SamplingParams documents stop, stop_token_ids and ignore_eos. | **L7 serving-correctness专项.** Define EOS ID behavior and stop matching across token/UTF-8 boundaries, delay ambiguous text before emission, validate stop versus length reasons. Required for a useful generation contract. |
| 2 | Persistent cross-request KV prefix reuse | **Partial capability already exists**: active requests share prefixes/fork CoW (layers/engine.py:109–138, 183–198). Once their last owner is gone, pages are discarded (layers/paged_kv.py:178–195), so completed-request reuse/hash lookup/LRU is missing. V3 “Introduction” and V4 “Block Allocation”, “Free”, “Eviction” describe the reusable-block design. | **L7 cache专项**, after serving correctness; evaluate sequential repeated prompts and cache hit rate. Performance option, not a minimum correctness gate; preserve model/tokenizer/adapter identity in cache keys. |
| 3 | Observability: metrics / structured logs / tracing | HTTP logs are suppressed; errors use traceback; only x-request-id and startup text exist (server/app.py:181–182, 227–232, 259–266, 448). There is no metrics/readiness route in do_GET (297–305). V6 defines Prometheus gauges/histograms; V7 documents OTLP/FastAPI tracing; U2 “Logging” supports formatter configuration. | **L7 operations专项.** First expose readiness, active/waiting counts, page usage, TTFT/ITL and error/cancel counts plus request-ID JSON logs; then sampled traces. Basic metrics/logs are minimum scope; full distributed tracing is optional. |
| 4 | Authentication and rate limits | POST parsing contains no identity/header authentication; capacity 429 is global saturation, not a tenant quota (server/app.py:311–337, 156–164). V8 api-key checks supplied header keys; E1 token buckets/descriptors and E2 global coordination are separate controls. | **L7 edge/access专项** before shared exposure: authenticate, apply per-principal quotas and connection/token budgets. Do not call the current capacity rejection a rate limiter. |
| 5 | TLS | ChatServer uses plain ThreadingHTTPServer with no SSL wrapping; CLI configures only host/port/checkpoint/tokenizer (server/app.py:398–416, 434–441). U2 “HTTPS” documents certificates; U4 “h2: HTTP/2 over TLS” describes the encrypted endpoint. | **L7 deployment专项** for any untrusted network; terminate at a maintained proxy and verify SSE forwarding. Reimplementing TLS is **not worth doing**; loopback experiments need no new TLS layer. |
| 6 | Speculative decoding | One sampled token per active request; no draft/verifier/rollback path (layers/engine.py:319–350). V9 “Speculating with a draft model”, “Lossless guarantees” describe draft/verify mechanisms and warn performance depends on workload. | **Not worth doing now** for a 13M dispatch-sensitive teaching model. Revisit as a separate L7 algorithm专项 only with a realistic larger target and a measured acceptance/latency hypothesis; not a minimum-service requirement. |
| 7 | Multiple LoRA adapters | One fixed model ID and dense checkpoint loader; no adapter request routing or adapter parameters (server/app.py:35–41, 81–82, 419–431; layers/model.py:28–70). V10 “Serving LoRA Adapters” supports per-request model selection with configured max_loras/rank. | **Not worth doing now** without trained adapters or a multi-tenant use case. A future adapter-training/serving专项 must first define adapter quality and cache identity; not a minimum-service requirement. |
| 8 | Graceful reload / readiness / drain | SIGTERM stops HTTP serving, then bridge shutdown aborts live and queued jobs; no checkpoint reload/readiness/drain phase (server/app.py:233–245, 443–452). The 0.0736 s CLI exit proves cleanup, not zero-dropped-stream reload (results/l6/l6-t2-report.md:247–252). U3 “Built-in” documents one-at-a-time SIGHUP restarts; U1 “Graceful Process Shutdown” describes draining. | **L7 lifecycle专项.** Gate readiness, stop admission, drain with a deadline, warm/check the next version, switch routing, then release the old engine. Development autoreload is not this contract; zero-downtime swapping needs capacity planning. |
| 9 | Production HTTP concurrency / persistence | Thread per connection, forced close and no accepted-connection cap (server/app.py:262–272, 398–407); only eight-stream tests/measurements. P1/P2 and F1/U1/U2 provide the async/concurrency references in §2. | **L7 transport专项**, one measured async frontend to one owner first; keep-alive, slow-client/FD/thread budgets and version-qualified H2 at the edge. Multi-worker replication only when evidence requires it. |
| 10 | Token/SLO admission, chunked prefill, resumable preemption | Whole prefills drain before decode; worst-case reservation rejects rather than reschedules (server/app.py:189–211; layers/engine.py:166–180). S1 Algorithm 3, V1 chunking/preemption and V8 priority are the concrete counterparts (§3–4). | **L7 scheduler专项.** First finite queue deadlines/token budgets and long-prompt ITL tests; add chunking when tails justify it. Full recompute eviction only after a memory-pressure workload shows value; minimum service may safely reject instead. |
| 11 | Direct paged/compressed execution and L5 integration | Batched cache still gathers/pads fp32 tensors (layers/engine.py:278–307); API loads the fp32 model only (server/app.py:419–431). L5 reconstructs weights/KV rather than consuming compression in place (layers/quant.py:126–156, 179–201; runs/l5-diff-analysis.md §2–3). V2 §5 provides direct paged execution; the existing L5 comparison §2–3, references M7–M10/V1–V2, supplies the previously verified packed-matmul/compressed-KV production contrast. | **L7 runtime-performance专项** after serving correctness, with separate Oracle and latency/capacity gates. Compression alone is not a speedup. Optional for a small service; this integration was proposed for L6 but is not delivered by T1/T2. |

Rows 9–11 consolidate the transport, scheduling and precision carry-overs already
identified in [L4 comparison §5](l4-diff-analysis.md) and
[L5 comparison §5](l5-diff-analysis.md); they are not extra implementation scope
for this task. The serving model remains a **500-step base checkpoint, not
instruction tuned** (server/app.py:35–38). A usable research-model API therefore
does not establish chat-product quality. SFT/quality work would require a
separate algorithm task; this inventory does not silently redefine production
readiness as model capability.

## External reading locations

All links were opened for this task on 2026-10-21. vLLM is pinned to **v0.10.2**;
papers use their original proceedings/version. Python docs match the installed
3.12 series. FastAPI/Uvicorn and E2 are online documentation snapshots, **not
a benchmarked or installed dependency**; Uvicorn's current docs describe opt-in
experimental H2, which must be checked again when selecting an actual version.
References below name sections rather than navigation-page search results.

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

## Acceptance and limits

Five requested analysis points are present; §1 and §3 contain quantitative
accounts; every gap row has local evidence, a production reference and a landing
recommendation. Arithmetic derives only from archived measurements and retains
input hashes in A. No implementation/test/benchmark source or dependency changed,
no new performance experiment was run, and the test suite was not run as required
by task §5. Documentation checks and the section-6 execution report are archived
under results/l6/ by the final acceptance commit.

Two requested premises need qualification rather than invented evidence:
**shared dense graph ≠ measured constant total kernel count**, and a literal
**256-token prompt cannot be admitted by ctx=256 with positive output**.
The missing kernel trace is an evidence limit; the blocking figure is explicitly
an extrapolation under expanded admission/context assumptions. The task's
“-o file” has no supplied path, so the section-6 report uses the allowed
**results/l6/l6-diff-report.md** convention. No push is performed.
