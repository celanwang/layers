# L4-T1 Implementation Notes: Cached Decode

## Changed files

- `layers/inference.py` — append-only per-layer `KVCache`, `Sampler`, and streaming `generate`.
- `layers/attention.py` — optional cached key/value path with rotary position offsets and a past-aware causal mask.
- `layers/transformer.py` — passes the cache and layer index to attention when supplied.
- `layers/model.py` — accepts an optional cache; the no-cache branch retains its original call path.
- `layers/__init__.py` — exports `KVCache`, `Sampler`, and `generate`.
- `tests/test_inference.py` — cache oracle, sampling, UTF-8, and checkpoint generation checks.
- `bench/l4_decode_bench.py` — L0-style decode measurements.
- `results/l4/decode-bench-20261021T113854Z.json` — raw benchmark record.
- `runs/l4-implementation-notes.md` — this report.

No existing test file was changed.

## Implementation

- The cache stores rotary-encoded keys and values separately for each transformer layer. Each call appends the current chunk. Attention uses the current absolute RoPE offset and masks only keys that are in the future of each query.
- `generate` prefills the full prompt once, then forwards one sampled token per iteration. It uses Python's incremental UTF-8 decoder, so incomplete byte suffixes remain buffered between token IDs.
- `Sampler` applies temperature to logits, applies top-k before top-p, and renormalizes after nucleus truncation. Each draw seeds MLX with `seed + draw_index`, giving repeatable sequences without changing the no-cache model path.
- The L3 checkpoint test compares 32 consecutive cached decode logits against full-prefix logits, at `rtol=atol=1e-5`. The greedy smoke test generates 200 tokens from the 500-step L3 checkpoint and checks each generated-step logits array is finite.
- The 512-token KV curve is prefetched in chunks no larger than the model's `ctx=256`; cached decode then attends over the full 512-token cache.

## Acceptance commands

### Full test suite

Command:

```text
.venv/bin/python -m pytest tests/ -q
```

Output:

```text
.....................................................                    [100%]
53 passed in 4.90s
```

The 49 pre-existing tests passed unchanged; the four new inference tests also passed.

### Decode benchmark

Command:

```text
.venv/bin/python bench/l4_decode_bench.py
```

Output:

```text
device: Apple M4 Max, mlx 0.32.4
config_hash: 15cd4fbd762b (commit 987fe1533f35)
prefill 128 tokens: 52623.46 tok/s (median 0.002432s, p90 0.002562s)
decode 256 tokens: 314.96 tok/s (median 0.812790s, p90 0.852472s)
roofline: 8571.4 tok/s; measured 314.96 tok/s (3.675% of bound)
KV n= 64: 508.01 tok/s (median 0.001968s, p90 0.002651s)
KV n=128: 518.50 tok/s (median 0.001929s, p90 0.003478s)
KV n=256: 477.32 tok/s (median 0.002095s, p90 0.002242s)
KV n=512: 470.28 tok/s (median 0.002126s, p90 0.002309s)
cache ratio (32 new tokens): cached 0.093867s, uncached 0.103026s, uncached/cached 1.098x
written: results/l4/decode-bench-20261021T113854Z.json
```

Both stages use 3 warmups and 10 measured repetitions; median and p90 are recorded. The record includes the config hash, checkpoint, Python/MLX versions, OS, and Apple M4 Max device details.

## Measurement interpretation

The 256-token cached decode reached 314.96 tok/s against the 8,571.4 tok/s weight-bandwidth roofline estimate, or 3.675% of that bound. The measured rate is about 27.2× below the estimate, consistent with the principle note's prediction that per-step dispatch and small-operation overhead keep this workload far below the ideal weight-read ceiling.

The one-token decode curve moved from 508.01 tok/s at 64 cached tokens to 470.28 tok/s at 512 tokens, a 7.4% decline. The 64-to-128 medians are not monotonic, and the 128-token p90 is notably noisy; the measured trend is modest at these sequence lengths. For a 32-token generation, uncached execution took 1.098× the cached median time.

## Deviations and command history

- No remaining behavior or file-scope deviation from the task specification. Existing tests were not edited, no dependency was added, no reference inference-engine source was read, and no push was performed.
- The first direct benchmark invocation failed before measuring because Python could not import `layers` when launched from `bench/`. The script now adds the repository root to its import path; the exact acceptance command then completed successfully. The fix is in commit `987fe1533f35`.
- `ruff` is not installed in the environment and the repository has no lint configuration. Syntax compilation and `git diff --check` passed; no lint dependency was added.

## Implementation commits

- `b137e6f` — `feat(l4): add cached inference path`
- `987fe15` — `fix(l4): support direct benchmark invocation`
- `3f17a79` — `test(l4): assert finite decode logits`

## L4-T2 Implementation Notes: Paged KV and Continuous Scheduling

### Design

- `PagedKVPool` owns a fixed 512-page pool by default: 16 concurrent sequences ×
  512 context tokens ÷ 16 tokens per page. Page payloads are allocated lazily as
  fixed-size per-layer MLX K/V arrays. Each sequence owns a logical block table;
  `fork` increments page references and a write to a shared page copies that
  page before replacing the written slots.
- `PagedKVCache` adapts one sequence to the existing T1 cache protocol. It
  gathers the sequence's logical K/V prefix into contiguous tensors before
  attention consumes them. Attention remains request-local: the engine calls
  the model once per request and does not build padded multi-request batches or
  use a paged attention kernel.
- `Engine.add_request` prefills a new prompt eagerly. An active request with an
  identical tokenized prompt can be forked at that prompt length, sharing the
  prompt pages and saved prompt logits. Each `step` samples once for every
  active request and advances every non-final request by one token.
- Capacity is fixed and admission is fail-fast. The default pool covers 16
  concurrent requests with contexts up to 512 tokens. A request over the
  active-sequence limit, or one that needs pages beyond a custom pool's
  capacity, is rejected with `RequestCapacityError`; there is no hidden queue.
  Callers can retry after an active request finishes and frees its pages.

### Validation

- `tests/test_engine.py` adds a paged-vs-T1 per-token logits oracle with two
  interleaved sequences, parent and sibling CoW isolation, repeated allocation
  and release accounting, the average half-page fragmentation bound, scheduler
  token equivalence, and explicit pool-exhaustion behavior.
- `bench/l4_engine_bench.py` follows L0's 3 warmups / 10 measured repetitions
  and reports medians, p90, device, configuration hash, fragmentation, total
  request-set wall time and throughput, and shared-prefix memory savings.
- Acceptance command output and measured values are recorded in
  `results/l4/l4-t2-report.md` and `results/l4/engine-bench-*.json`.
