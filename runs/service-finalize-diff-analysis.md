# Service v1 comparison and evidence limits

## Completion semantics

L6 returned length-only completions. The service-owned filter adds EOS and
stop without modifying numerical engines. Its independent Oracle scans decoded
characters for the first completed match. Byte-partition tests include overlap,
CJK and emoji; eight concurrent scripted standard Samplers exercise L7's actual
batched-sampling path while rows finish at different iterations. A triggering
sample counts toward usage even when its spelling is suppressed.

After implementing the filter, inspected CPython 3.12.15's
`codecs.BufferedIncrementalDecoder.decode` in the installed `codecs.py`:
undecoded bytes are retained after `_buffer_decode`. Our wrapper adds Unicode
stop-prefix buffering above this byte buffer; it must never call final=True
until a length/EOS boundary. `inspect.getsource` initially failed for the frozen
codecs function; the installed source file was then read directly. No external
source was copied into this repository.

## Ownership, drain and HTTP workers

The L6 owner/queue architecture remains. The owner publishes metrics and frees
pages on termination; HTTP threads never mutate KV. Inspected the installed
`socketserver.ThreadingMixIn.process_request/server_close` after implementation:
worker daemon status follows `daemon_threads`, and server_close joins retained
workers. Using non-daemon workers enables completed drain output to be flushed.
The existing 5-second socket timeout bounds slow socket operations. Device
calls remain non-preemptible; deadline enforcement happens at owner boundaries.
This is drain-and-restart, with no hot model swap.

## Assets and engine selection

The L6 loader installed whatever names appeared in a checkpoint. The new loader
validates exact keys/shapes/fp32/finiteness and the 1028-ID tokenizer contract;
all manifest hashes are checked before parameter mutation. NumPy NPZ tests
supply malformed archives independently. A user-registered manifest binds
known local training outputs; it does not authenticate training history.
L7 stays explicitly selectable because the recorded B=1 regression precludes
an unmeasured dynamic/default path decision. Model/attention/cache numerics,
training code, dependency versions and previous tests/protocols are unchanged.

## Measurements

The new harness measures full-request paired direct/API latency and B=1/4/8,
3 warmups + 10 observations, alternating order. Startup is excluded, but a
fresh owner is the default trial unit. B>1 compares eager direct admission to
HTTP arrival, so their difference is not pure transport overhead. Metrics
TTFT/ITL are owner-ready times, not network delivery times; client first-content
latencies are recorded separately. JSON logs never retain request/completion
content.

The first archived fresh-owner measurement passed the singleton target for
L6 (3.41%) but failed L7 (12.25%). A separate persistent-owner diagnostic is
explicitly labeled, retains its own config and raw data, and cannot replace
that failure. Its initial run overlapped a short targeted test process near
startup; treat it as exploratory. Final isolated measurements and verdicts are
listed in the retrospective. No kernel trace or sole causal attribution of
thread/compile effects is claimed.

## Remaining boundaries

No authentication/TLS/public deployment, fused compressed-page Metal kernel,
full OpenAI API, instruction tuning or tool capability is introduced. L7 is
vectorized page materialization, not kernel-side zero-copy page attention.
The existing BF16 regression test writes a new timestamped L5 artifact when
running the full suite; preserve it but exclude it from this task's commits.
