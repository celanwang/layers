# L6-T1 implementation notes

## Design and blind implementation

The implementation uses only this project's L1–L5 components and the approved
L6 principle note. No inference engine or serving framework source was read.

- Add `BatchedEngine` without changing `Engine`'s request or decode behavior.
  This is a synchronous scheduler: `add_request` is the iteration-boundary
  admission/prefill operation, inherited from L4. Prompts prefill separately,
  including the existing prefix sharing and worst-case page reservation rules,
  then join the next `step` batch. Callers serialize admission and stepping.
- Sample in stable request order. Exclude terminal tokens from further model
  work, decode all continuing IDs as `[B, 1]`, distribute logits in that same
  order, and free completed sequences at the end of the iteration.
- Snapshot each sequence's real KV length before layer writes. Use it both as
  that row's RoPE position and as the inclusive attention key bound. Append
  real K/V to the unchanged page pool, gather, then right-pad to the batch max.
  Padding exists only in attention views and never consumes persistent pages.
- Mask scores with negative infinity before the explicit stable softmax.
  Inspect post-softmax probabilities directly in tests, including exact zeros.

## Assumptions and failure probes

Variable positions cannot use a single scalar RoPE offset. A separate batch
attention method makes the position broadcast explicit and preserves all old
operation paths. Tests compare absolute logit errors at `1e-5`, use both greedy
and seeded stochastic sampling, permute tokens together with their KV rows,
and exercise shared-page completion plus new arrivals.

Performance predictions remain hypotheses until the L0-protocol benchmark:
B=8 speedup at least 4x; B=1 latency regression below 10%. Negative results
will be retained alongside configuration and raw timing samples.

## Initial measurement and bounded optimization

The first uncompiled batch path passed all 97 tests. With 3 warmups and 10
paired repeats it measured B=8 at 1385.02 tok/s versus L4 at 346.94 tok/s:
3.992x, below the 4x gate. B=1 improved 15.42%. The full failed measurement
is retained in `results/l6/batching-bench-20261021T130956006308Z.json` and
`results/l6/bench-attempt1.txt`; the gate was not rounded up or weakened.

The batch path now compiles two pure arithmetic helpers (RoPE and explicit
masked attention) with `mx.compile`. Page operations remain inspectable Python;
the original single-request operations and all measurement settings are intact.
No `mx.fast` operation is used. We consulted only the installed public
`mx.compile` API docstring, without reading implementation sources.

The first compiled measurement passed at B=8 4.037x, with little margin (raw
record `batching-bench-20261021T131308951824Z.json`). A follow-up shape audit
made every dense projection explicitly rank two: attention Q/K/V/output,
MLP, and vocabulary projection operate on `[B, D]` rows. Attention still uses
`[B, H, 1, T]` and model inputs/outputs retain `[B, 1, ...]`. This directly
expresses shared-weight GEMM rather than relying on backend conversion of a
rank-three batch of one-row matrix multiplies. The L4 path is unchanged.

## Trained-checkpoint precision diagnosis

An additional check with the real width-512 checkpoint, eight variable prefixes,
and 33 sampled tokens found two absolute logit errors above `1e-5` out of
271,392 comparisons (max `1.1444091796875e-5`), although every token matched.
The failure is retained in `checkpoint-oracle-attempt1.json`. Disabling
compilation and changing dense layout produced the same two outliers
(`linear-layout-probes.json`), ruling out compilation as their source.

An fp64 CPU diagnostic decomposed each failing final logit into propagated
normalized-input drift and output-projection accumulation error. Input drift
was only `1.62e-6` / `-1.04e-6`; the batch vocabulary dot-product errors were
`8.39e-6` / `-1.02e-5` relative to fp64, versus reference GEMV errors below
`4.85e-7`. Thus the final long fp32 reduction dominated these failures.
`projection-precision-probe.json` preserves the decomposition and alternatives.

The batch vocabulary projection now sums two half-width GEMMs, shortening
the fp32 accumulation chains while continuing to share both weight halves
across all request rows. The diagnostic max absolute error fell to
`7.62939453125e-6`; no tolerance was relaxed. A trained-checkpoint regression
test covers the same eight prefixes and 32 actual decode forwards. CPU fp64
is diagnostic only; inference remains GPU fp32 throughout.

The unchanged bf16 training regression test automatically emits an L5 JSON.
Each newly generated artifact is relocated intact into `results/l6/` after the
suite; existing L5 files are neither edited nor removed.


## Final acceptance

- `.venv/bin/python -m pytest tests/ -q`: 98 passed (91 unchanged existing
  tests plus 7 new tests), one existing PyTorch quantization deprecation warning.
- `.venv/bin/python bench/l6_batching_bench.py`: exit 0; final record
  `results/l6/batching-bench-20261021T132523026893Z.json`, config hash
  `d2f97623b7b7`, implementation commit `e89de74f2ea8`.
- Apple M4 Max / 48 GiB / MLX 0.32.4, GPU fp32, L3 500-step checkpoint,
  3 warmups and 10 paired alternating repeats; raw samples, median/p90,
  checkpoint hash and source hashes are retained in each benchmark JSON.
- B=8: 1400.44 tok/s
  versus unchanged L4 310.22 tok/s,
  4.514x (>=4x gate passes).
- B=1: -24.18% latency change
  (<10% regression gate passes).
- Uneven B=8: 2.17% throughput penalty;
  padding occupies 40.7407% of key slots versus 0.7752% for near-equal lengths.
  Earlier trials showed 3.31% and -0.35%; this small timing effect varies, and
  the negative measurement is retained. The illustrative ~3k tok/s projection
  is not reached by this teaching implementation.
- Real-checkpoint oracle: 8 variable prefixes, 33 token events per request
  (32 decode forwards), max absolute logit error 7.62939453125e-6 with rtol=0;
  all seeded tokens match and all pages are reclaimed. See
  `results/l6/checkpoint-logit-oracle.json`.
- Scope audit: 9 pre-existing test files byte-identical, 25 existing functions
  and methods AST-identical, inference.py and paged_kv.py byte-identical.
  There are no new dependencies and no `mx.fast` calls. Syntax compilation
  and whitespace checks pass; no separate lint configuration is present.

## Scope interpretation and remaining limits

There is no change to the requested measurement protocol or acceptance gates.
Admission/prefill is inherited as a synchronous boundary operation between
steps, rather than adding a pending request queue. HTTP/SSE and framework
source comparisons are outside T1. No external inference/serving source was
read and no push was performed. Failed measurements and diagnostic probes
are preserved, not substituted for final acceptance.

## L6-T2: blind HTTP/SSE implementation

The serving layer is built directly from the approved principle note and
OpenAI's public API reference. No inference serving framework implementation
source was read. The schema oracle is the public
[chat completion reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[streaming event reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events),
and [model list reference](https://developers.openai.com/api/reference/resources/models/methods/list).

- `ThreadingHTTPServer` parses a deliberately explicit text-only chat subset.
  Messages become `role: content` lines. The model description discloses the
  500-step base checkpoint and absence of instruction tuning. The default
  runtime is fp32; the optional L5 INT8 extension is not selected for T2.
- One dedicated thread owns admission, sampling, stepping, cancellation and
  the paged pool. Bounded ingress and per-request output queues bridge HTTP
  threads to that owner. Cancellation uses an Event independent of ingress
  capacity. A full output queue closes only that client's socket and releases
  its request at the iteration boundary, without blocking other decodes.
- The only engine change is additive: idempotent `cancel_request`, which also
  forgets completed histories. No existing method changes. HTTP threads never
  touch mutable engine state; the tokenizer and context settings are read-only.
- Each request has its own Sampler and incremental UTF-8 decoder. SSE emits
  assistant role, content deltas, a length terminal chunk, and `[DONE]`.
  Optional `stream_options.include_usage` follows the public schema: null
  usage on ordinary chunks, then an empty-choices usage chunk before `[DONE]`.
  JSON mode concatenates the same decoded pieces and reports token usage.
- Invalid/unsupported parameters fail before admission with an OpenAI-style
  error envelope. Content-Length and socket timeouts bound body reads/writes.
  Connection resets, EOF, overflow and server shutdown all signal the owner
  to reclaim pages. Normal completion also drops the engine's request history.

### T2 failure probes and corrected assumptions

The first API test run exposed MLX lazy-weight ownership: transferring an
unevaluated model graph into another thread can fail with `There is no
Stream(gpu, 0) in current thread`. Evaluating model parameters before starting
the owner thread fixes the boundary without changing numerical operations.
The corrected tests include newly constructed lazy models, not just already
evaluated checkpoints.

The disconnect test holds the second engine iteration at a barrier, verifies
live allocated pages, receives an actual SSE event, then closes the client
with TCP reset. JSON mode closes before any response headers. This proves
early cancellation rather than merely waiting for a short request to finish.
An initial test revision tried accessing HTTPConnection.sock after response
headers transferred ownership to HTTPResponse; the harness now sets linger
before that transfer. The failed full-suite output is retained in
`results/l6/api-tests-attempt1.txt`.

The initial API benchmark completed successfully, but overlapped API test
debugging for part of its run. Its raw JSON and console output are retained
as exploratory evidence only; acceptance is rerun serially after tests pass.
The benchmark keeps L0's three warmups, ten repeats, median/p90, raw samples,
hardware and configuration/source hashes. Direct/API order alternates in
paired trials. Both paths include prompt tokenization, prefill, sampling,
decode and text output; HTTP also includes fresh TCP setup and SSE consumption.
Model/server startup is excluded. Concurrency 1/4/8 trials compare every output
with an independently generated direct-engine result and assert empty pools
after shutdown. No measurement threshold has been relaxed.

### T2 serial acceptance and bounded optimization

The first serial benchmark failed the overhead gate: direct 2.1797 ms/token,
HTTP/SSE 2.3840 ms/token, +9.37%. Its full record is
`results/l6/api-bench-20261021T143755909668Z.json` and console output is
`api-bench-attempt2.txt`. Correctness, completion and pool reclamation passed.
Profiling and a separate-client-process probe are retained in
`api-overhead-profile.json` and `api-client-process-probe.json`. Moving the
client into another process did not improve this probe, so the accepted
benchmark remains unchanged, including client/server process placement.

The implementation originally called select for peer EOF on every generated
token. This releases the GIL for a syscall even when the subsequent SSE write
already detects a broken socket. Peer checks are now limited to every 10 ms;
the queue consumer still checks cancellation on every item, SSE writes still
detect broken connections, and JSON-only disconnects are polled as well.
The deterministic live-page disconnect tests continue to pass.

With this one implementation change and the exact same benchmark protocol:

- API overhead: direct 2.2592 ms/token versus API 2.3429 ms/token, **+3.70%**,
  passing the strict <5% gate. Record: `api-bench-20261021T144110591270Z.json`,
  configuration hash `4474cab6ff21`.
- Streaming aggregate throughput at concurrency 1/4/8: **418.23 / 991.17 /
  1298.17 tok/s**. Per-request median/p90 and all raw timing samples are in JSON.
- All **182** direct/API text comparisons pass, including every concurrent
  request; all trial pools are empty after shutdown.
- Final `.venv/bin/python -m pytest tests/ -q`: **119 passed**, one existing
  PyTorch deprecation warning. All 10 pre-existing test files are byte-identical
  and all 12 existing engine methods are AST-identical. Syntax compilation and
  whitespace checks pass. No separate lint configuration is present.
- The exact curl/head acceptance request produced real SSE deltas. A second,
  untruncated curl capture contains 34 JSON chunks (role + 32 content chunks +
  terminal), then `[DONE]`; concatenated content equals direct engine output
  with seed 42 and temperature 0. A separate CLI SIGTERM check exits 0 in
  0.0736 s. See `api-curl-full.txt`, `api-curl-oracle.json` and
  `api-cli-shutdown.txt`.

The unchanged bf16 regression test's newly generated L5 JSON records are
relocated intact to `results/l6/`. The pre-existing untracked L5 artifact
`bf16-train-20261021T132935962922Z.json` is preserved in place. The complete
T2 report and acceptance outputs are archived under `results/l6/`; no push
or framework-source comparison is performed in this blind implementation task.
