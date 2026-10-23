# Local service completion: principle and interface

Generation has a numerical stage and a presentation/lifecycle stage. Preserve
the former. A service-owned completion filter consumes token bytes, buffers
incomplete UTF-8 and possible stop prefixes, and emits only text that cannot
be part of a future stop match. EOS and the earliest complete stop match
take precedence over a simultaneous length limit. An unmatched buffered prefix
is flushed on length termination. The full-text `str.find` result is the Oracle.

One engine-owner thread retains all mutable page/request ownership. Stop,
disconnect, deadline and failures converge on idempotent cancellation there.
Readiness is false while draining or after owner failure. Metrics publish owner
snapshots rather than reading the mutable page pool from HTTP threads.

FP32 weights and token IDs form one artifact contract. Check exact keys,
dimensions, finite fp32 data and embedding vocabulary before mutating a model.
Hash both files for provenance. L7 is opt-in because its singleton regression
is recorded; select the implementation explicitly rather than inventing an
unmeasured dynamic dispatch threshold.

Drain preserves accepted jobs and rejects new admissions. The deadline bounds
the drain wait; an in-flight MLX call cannot be preempted, so cleanup completes
at the next engine boundary. HTTP transmission failure still cancels the job.
