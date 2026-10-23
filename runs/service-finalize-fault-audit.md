# Final local-service fault audit

## Goal and scope

Complete local service v1's terminal-state fault isolation: a broken or closed
stderr sink must not turn successful inference into failure, strand an admission
waiter, prevent the original engine error from being reported, or skip KV
reclamation. This is a bounded follow-up within the approved service completion
specification, not a new numerical/serving architecture iteration.

## Reproduction and implementation

The original `_report` wrote directly to stderr after marking a job reported but
before publishing done. An exception there could stop successful generation;
in admission rejection, the job was outside the live registry and could remain
without ready/done. Three direct traceback writers could themselves fail during
error cleanup. This was reproduced using independent injected BrokenPipeError
and ValueError writers: six tests failed before the repair. The captured output
is `results/service-finalize/fault-audit/reproduction.txt`.

One best-effort `_diagnostic` writer catches those I/O failures and increments
`log_write_failures` without recursively logging through the failing stream.
Request-end logs and exception tracebacks share it. Normal engine errors remain
fatal to readiness and still produce their original error outcomes; diagnostic
failure alone does not change generation success or admission status. The new
metric initializes at zero. No model, sampling, cache, generation schema,
dependency, existing test, previous numerical artifact or performance protocol
was modified.

The external failure contract is Python's broken-pipe/closed-stream I/O
behavior. The independent assertions inspect request ready/done flags, terminal
error status, next-request reuse, engine health and actual page-pool restoration.
This tests outcomes rather than mirroring the wrapper's implementation.

## Validation

- Before repair: 6 fault-injection tests failed.
- After repair: 6 passed (`targeted.txt`).
- Full suite: **189 passed**, one pre-existing PyTorch quantized-tensor API
  deprecation warning (`full-regression.txt`, 29.77 seconds).
- Full-suite CLI subprocess test also validated live-stream SIGTERM drain;
  its new output is retained under `fault-audit/test-generated/`.
- Syntax compilation and whitespace checks passed; no configured repository
  linter exists and no lint dependency was installed.
- Tracked test-generated records were captured and their original bytes
  restored; `artifact-preservation.json` lists these. Timestamped BF16 files
  created by pre-existing tests remain untracked. No existing artifact was
  deleted or replaced in the commits.

The request-end writer changes once per completed request and the three error
paths; token/model execution is unchanged. No performance rerun or new speed
claim is made. Prior source-hashed measurements remain evidence of their exact
archived versions, not an assertion that this revision was benchmarked.

## Closure and remaining work

The local inference pipeline and the approved service contract are complete for
this learning project. Final review/merge remains with Kimi/the user; no push or
merge was performed. Remaining Metal profiling, fused INT8/page consumption,
KV INT4/QAT, broader model quality and deployment security are independent
research/deployment scopes, not prerequisites invented for this local release.

Diagnostic I/O is still synchronous: a slow writer that does not raise can delay
the owner. The repair isolates thrown I/O errors; it does not claim a bounded
asynchronous log pipeline. Device operations also remain non-preemptible, as
already documented for draining.
