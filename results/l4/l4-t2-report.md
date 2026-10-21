# L4-T2 Acceptance Report: Paged KV Pool and Continuous Scheduler

## Changed files

- `layers/paged_kv.py` — fixed-size block pool, sequence block tables, refcounts,
  fork, copy-on-write, and contiguous gather.
- `layers/engine.py` — eager prefill, longest common prompt-prefix fork,
  per-request decode steps, request completion and capacity handling.
- `layers/__init__.py` — public exports for the T2 pool and engine.
- `tests/test_engine.py` — seven new T2 invariant and behavior tests. No existing
  test file was modified.
- `bench/l4_engine_bench.py` — L0-style fragmentation, request-set throughput,
  and prefix-memory measurements.
- `runs/l4-implementation-notes.md` — T2 design, capacity policy, and results.
- `results/l4/engine-bench-20261021T114635Z.json` — initial pre-commit run.
- `results/l4/engine-bench-20261021T114814Z.json` — run after the first
  implementation commit and before longest-common-prefix support.
- `results/l4/engine-bench-20261021T114941Z.json` — final acceptance run after
  commit `c671927`.
- `results/l4/l4-t2-report.md` — this report.

## Acceptance commands and complete output

### Full test suite

Command:

```text
.venv/bin/python -m pytest tests/ -q
```

Output:

```text
............................................................             [100%]
60 passed in 5.04s
```

### L4-T2 benchmark

Command:

```text
.venv/bin/python bench/l4_engine_bench.py
```

Output:

```text
device: Apple M4 Max, mlx 0.32.4
config_hash: b4e856d94cfe (commit c671927eb61b)
fragmentation (block_size=16):
  N= 1: waste 65536 B / bound 131072 B (4 token slots)
  N= 4: waste 262144 B / bound 524288 B (16 token slots)
  N= 8: waste 802816 B / bound 1048576 B (49 token slots)
  N=16: waste 1671168 B / bound 2097152 B (102 token slots)
batching 8 requests / 128 generated tokens: continuous 0.329529s (388.43 tok/s), sequential 0.348616s (367.17 tok/s), speedup 1.058x
prefix sharing (8 x 32 tokens): 524288 B vs 4194304 B, saved 87.5%
written: results/l4/engine-bench-20261021T114941Z.json
```

## Key measurements

The final run used Apple M4 Max, MLX 0.32.4, float32, 3 warmups and 10 measured
repetitions. The raw result contains median, p90, minimum, device details, and
config hash.

| Measurement | Result |
|---|---:|
| Fragmentation, N=1 | 65,536 B / 131,072 B bound |
| Fragmentation, N=4 | 262,144 B / 524,288 B bound |
| Fragmentation, N=8 | 802,816 B / 1,048,576 B bound |
| Fragmentation, N=16 | 1,671,168 B / 2,097,152 B bound |
| Continuous, 8 requests / 128 tokens | 0.329529 s median, 0.353987 s p90, 388.43 tok/s |
| Sequential, same requests | 0.348616 s median, 0.383359 s p90, 367.17 tok/s |
| Throughput ratio | 1.058× (+5.79%) |
| Prefix pages, shared vs unshared | 524,288 B vs 4,194,304 B |
| Prefix memory saved | 3,670,016 B (87.5%) |

The throughput timing covers the complete request set, including prompt
prefill. All eight requests use the same four-token prompt, so the continuous
engine performs one prefill and forks the other seven; the sequential baseline
prefills once per request. The measured gain therefore includes shared-prefill
savings. Decode still invokes the model separately for each request, as required
by the shape constraint; this run does not measure kernel-level batching or
decode-time weight-read amortization.

For 16-token pages, an individual sequence can leave up to 15 unused slots. The
task's seeded aggregate check uses the N × 8-token bound; all four aggregate
measurements passed. This is not a deterministic per-sequence half-page
worst-case guarantee.

## Capacity behavior

The default pool has 512 physical blocks: 16 concurrent sequences × 512 context
tokens ÷ 16 tokens per block. The pool allocates page payloads lazily. Engine
admission is fail-fast with no hidden queue: it rejects requests beyond the
active-sequence limit and reserves the no-sharing worst-case page count before
admission. If that reservation exceeds pool capacity, `RequestCapacityError` is
raised; callers can retry after active requests complete and release pages.
Direct pool allocation raises `PoolExhaustedError` when fixed page capacity is
exhausted.

## Deviations and implementation commits

- No change was made outside the task whitelist. `layers/inference.py` and all
  pre-existing tests remain unchanged. No dependency was added, no external
  inference-engine implementation source was read, and no push was performed.
- The run at `114635Z` was captured before the implementation commit and is
  retained as an exploratory record. The final reported benchmark is the
  `114941Z` record whose config hash references the completed implementation
  commit.
- The continuous-vs-sequential result is an end-to-end request-set comparison
  with shared prompt prefill, as described above. It does not establish decode
  throughput scaling from batched model forwards, which the requested shape
  constraint excludes.
- `ruff` is not installed and the repository has no configured lint command;
  `git diff --check` passed. The full test suite and required benchmark passed.

Commits:

- `254a3fc` — `feat(l4): add paged KV engine`
- `c671927` — `feat(engine): share common prompt prefixes`
