# Local service v1 implementation notes

- Preserved all numerical model and engine code. `--engine l6|l7` explicitly
  selects the existing implementations; L6 remains default.
- Strict NPZ names, shapes, fp32 and finiteness checks precede parameter
  installation. A checked-in manifest binds the original checkpoint/tokenizer
  hashes. Local training reproductions register separate manifests against
  their training metadata; no checkpoint is newly committed.
- Service completion filtering uses incremental UTF-8 plus bounded stop-prefix
  retention. First completed match wins, earliest start resolves same-end ties.
  EOS and stop suppress control text; usage includes the sampled trigger token.
  The owner cancels a stopped row after the engine iteration (which may have
  computed its next logits). Numerical/RNG order of surviving rows is intact.
- Metrics retain 1024 latency observations, publish queue/page gauges only on
  owner boundaries, and expose owner failure/readiness/draining independently.
  TTFT is first visible text ready in owner, ITL uses per-request token events.
- Drain rejects new submits, processes existing work until a deadline and
  cancels at the next owner boundary. Device work is not preemptible. HTTP
  workers are joined to finish responses; sockets retain their 5-second timeout.
- First targeted run exposed a readiness/done publication race after adding
  structured outcome logging. Publish terminal admission state before ready;
  existing immutable test now passes. Failure output is retained.
- Original 156 tests plus new tests: first full run 176 passed (one existing
  PyTorch quantization API warning). Artifact-registration testing was added
  afterwards and its targeted tests pass; a final full run remains required.

## Final functional validation

- Full regression: 183 passed (one existing PyTorch API deprecation warning).
- Final focused suite: 48 passed, including actual L7 standard-Sampler early
  termination, malformed Unicode rejection and all old API wire tests.
- CLI process smoke: L7 accepted a 180-token SSE request, received SIGTERM after
  first content, delivered the terminal usage/[DONE] frames and exited 0.
- The stop fixture initially requested one token more than its scripted source;
  corrected that new-test fixture budget. Existing tests were never changed.
- Preserved first failing admission-state race output and all timing failures.
