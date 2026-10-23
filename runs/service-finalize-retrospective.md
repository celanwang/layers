# Local inference service v1 retrospective

## Delivered behavior

The approved service scope is complete on branch `service-finalize`:
strict artifact loading/fingerprint binding, explicit L6/L7 selection,
EOS/string termination with Unicode-prefix buffering, process-local health and
bounded metrics, redacted structured request logs, deadline drain, joined HTTP
workers, pinned existing dependencies and reproducible startup/asset guidance.
Numerical model/engine code and all pre-existing tests remain unchanged.

## Functional evidence

- Full regression: **183 passed**, one pre-existing PyTorch quantization API
  warning; `results/service-finalize/tests-final.txt`.
- Final focused validation after strengthening the mixed early-stop test to use
  standard Samplers on L7's actual batched sampling path: **48 passed**;
  `results/service-finalize/tests-final-targeted.txt`.
- Actual subprocess: `--engine l7 --port 0`, receive first SSE content, SIGTERM,
  finish accepted 180-token request with terminal usage and `[DONE]`, exit 0;
  `results/service-finalize/cli-smoke.json`.
- Asset tests independently create malformed NPZs and reject missing/extra
  keys, shape/dtype/nonfinite values, wrong ID space and fingerprint mismatch.
  Registration checks training metadata and refuses manifest overwrite.
- Mixed concurrent EOS/string/length termination, random sampling parity,
  disconnect/overflow/capacity, owner failure, drain expiry and page recovery
  are covered. Metrics use owner snapshots and bounded 1024-value windows.
- Syntax compilation and `git diff --check` pass. No repository linter is
  configured; none was newly installed.

## Final isolated measurement

Raw record: `results/service-finalize/service-bench-20261023T054527680889Z.json`.
Apple M4 Max / 48 GiB, macOS 15.8, Python 3.12.15, MLX 0.32.4, fp32,
13,118,464 parameters, 64 output tokens per request, B=1/4/8, prefix sharing
Disabled, greedy, 3 warmups and 10 observations. Paired alternating direct/API
order; new owner/server per trial; startup excluded, full requests include
prefill. Source/artifact hashes, raw samples, client first-content latencies,
owner token-interval statistics, commit and working status are recorded.
No training or test process overlapped this final run.

| Backend | B | HTTP/SSE aggregate tok/s | API vs direct median time change |
|---|---:|---:|---:|
| L6 | 1 | 440.1 | +1.10% |
| L6 | 4 | 1020.1 | +5.02% |
| L6 | 8 | 1309.3 | +7.00% |
| L7 | 1 | 419.7 | +3.82% |
| L7 | 4 | 1186.2 | -1.91% |
| L7 | 8 | 1634.5 | +3.99% |

The existing singleton <5% target passes in the final run for both backends.
B>1 admission patterns differ (direct eager versus HTTP arrival); these ratios
are not isolated transport shares, and a negative ratio is not proof that HTTP
accelerates model computation. Full-request rates are not interchangeable with
L7's decode-only 1893–1918 tok/s archives. All outputs matched the direct Oracle
and all trial page pools were reclaimed. L6 remains default because singleton
behavior differs from batch-8 behavior.

## Retained failures and uncertainties

- First targeted run exposed terminal admission state being published after
  `ready`, widened by logging. Reordered terminal publication and preserved
  `tests-attempt-1.txt`. Existing test source was not changed.
- First fresh-owner benchmark: L6 singleton +3.41%, L7 +12.25% (L7 target failed).
  A persistent-owner diagnostic measured +5.78%/+10.77% and also failed. The
  diagnostic overlapped short targeted tests near startup and is exploratory.
  Both full raw records and request logs are retained.
- Final isolated run passes but does not explain the earlier variation. No
  target/protocol was relaxed and no benchmark was repeated until a preferred
  number appeared: the one predeclared final isolation check resolved the
  concurrent-workload concern. No stable universal overhead or compile/thread
  causal share is claimed. Persistent-owner and fresh-owner results remain
  separate protocols.
- A new scripted test initially had a generation budget exceeding its supplied
  token list; fixed the fixture. Existing tests were preserved.

## Limits and follow-up

The service is for loopback use with a 500-step Shakespeare base model; chat
schema support does not imply instruction/tool capability. No public exposure,
authentication/TLS, hot model swap, compressed-page kernel or speculative decode
was added. Drain waits cannot preempt an in-flight MLX call, and socket timeouts
can extend process close by up to five seconds. The service can compute one
unused next-logit step before noticing a textual stop. Usage counts that sampled
trigger token. Persistent counters reset on restart; timing windows are bounded.

The full regression's existing BF16 test generated timestamped files under
`results/l5/`. They were preserved and excluded from service commits along with
pre-existing untracked data. No push, merge, dependency upgrade, old artifact
rewrite or reference implementation vendoring was performed.
