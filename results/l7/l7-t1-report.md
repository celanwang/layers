# L7-T1: direct page-table materialization

## Outcome

Correctness and both section 5 commands pass. All 119 existing tests remain
unchanged, and 14 added tests pass (133 total). The trained d512 checkpoint
matches L6 tokens and logits exactly in the recorded variable-prefix test
(max absolute error 0.0, required atol=1e-5 / rtol=0).

Final B=8 throughput is **1605.61 tok/s**, versus paired
L6 **1421.73 tok/s** (1.1293x), and
**1.1178x** the historical 1436.41412547426 tok/s L6 baseline.
The >=1700 prediction is **not supported**. The result is above the <1550
setback boundary, so the specified verdict is **intermediate**.
No measurement protocol or acceptance threshold was changed.

B=1 decode latency rises **1.32%** against the paired L6 path.
This is below L6's original 10% singleton regression bound, but strict zero
regression is **not established**. B=2 throughput is also slightly lower.
Performance acceptance must not be described as an unqualified speedup at
all batch sizes. The benchmark exits 0 for correctness/protocol checks and
records the separate performance prediction, as the task accepts either
prediction outcome as evidence.

## Delivered change and comparison with the L6 oracle

- `PagedKVPool.append_pages` retains the old append validation, lazy payload
  allocation, and page-level copy-on-write, but stops before prefix gathering.
  The old `append` method is unchanged.
- `materialize_batch` finds only referenced pages for one layer, assigns compact
  physical-bank indices, and submits one vectorized `mx.take` for both K/V and
  all block tables. A compiled pure helper combines bank concatenate,
  index, transpose, and reshape. Page lookup and writes remain Python.
- The physical page shape is `[1, H, block_size, D]`; logical tail lengths can
  be shorter. Short rows' dummy blocks can reference a real page, and forked
  tails can contain the parent's later tokens. A strict valid-length mask
  excludes both. Materialization never allocates persistent pages and handles
  missing/empty layer payloads explicitly.
- The added attention method keeps L6 Q/K/V/output projection order and the
  same per-row absolute RoPE positions. Only the KV source and its length mask
  change. Masked scores are negative infinity and probabilities exactly zero.
- `PagedAttentionEngine` is a new `BatchedEngine` subclass. Admission, prefill,
  prefix sharing, capacity reservation, cancellation and request history are
  inherited. Its step preserves sampler calls, event order, terminal-token
  handling and deferred page release, selecting only the new decode adapter.
  A local decode function repeats the unchanged block/head arithmetic because
  `model.py` and `transformer.py` are outside the whitelist. No monkey patching
  or global path replacement is used; the two engines coexist on one model.

L6 performs B appends that each gather a whole logical prefix, then per-row
pad and batch concatenation. L7 performs B write-only appends, one compact
bank construction and one table index per layer. This removes B Python gather
calls, but **does not remove O(B * padded_length) output materialization**.
The compact bank adds an intermediate copy, and page writes/metadata loops
remain. This is vectorized MLX materialization, not a fused device kernel or
online-softmax implementation. No inference framework attention/kernel source
was read; only project code, the supplied principle note and the public
installed `mx.compile` docstring were consulted. External source comparison
is excluded by the task; the comparison here is against the immutable L6 path.

## Exact measurement protocol and evidence

Hardware: Apple M4 Max, 48 GiB unified memory,
macOS-15.8-arm64-arm-64bit, Python 3.12.15, MLX 0.32.4; GPU fp32.
Checkpoint: `results/l3/ckpt-final-500.npz`,
SHA-256 `a6bc079ef679c90ea8a4fdff95cf5086d4d49b8c811ce0252b2c31c7f1531889`.

- Direct imports: `bench.l6_batching_bench.measure_pair` and `load_model`.
  The unmodified harness also supplies `run_once`, `summarize`, constants,
  synchronization boundaries, sampler configuration and timing definitions.
- 3 warmups + 10 measured repeats per path; paired alternating order; fresh
  engines/pools. Scaling B=1/2/4/8, distinct equal-length 32-token prompts,
  64 generated tokens/request, 63 decode forwards, block size 16,
  no shared prefixes, seed 20261021.
- Decode includes sampling, scheduling, KV work and synchronization; prefill
  is excluded and separately recorded alongside end-to-end timings. Median
  and p90 use L6's exact order-statistic convention (upper median for n=10).
- Final implementation commit: `e5b5755002e6f2427a5b1fd322b1ea3c98e6a606`.
  Config hash: `4e04eb84dc1c`. Record: `results/l7/paged-attn-bench-20261021T154404587918Z.json`.
  The JSON retains every raw decode/prefill/end-to-end sample for both paths,
  all source SHA-256 values, device metadata, and checkpoint identity.
- The protocol dictionary is byte-value-equivalent after JSON parsing to the
  historical L6 record. Its retained variance settings are legacy protocol
  metadata; this task executes the specified equal-prompt scaling groups,
  not L6's separate uneven-length ablation.
- `evidence-verification.json` checks all 240 timed samples (4 batches x 2
  paths x 3 metrics x 10 repeats), config hash, exact protocol equality and
  delivered-source hashes. All 195 paired request outputs / 12,480 token
  events, including warmups, match. Every trial asserts full page reclamation.

| B | Paired L6 tok/s | L7 tok/s | L7 / paired L6 | L6 median / p90 ms | L7 median / p90 ms | L7 / historical same-B L6 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 482.71 | 476.41 | 0.9869x | 132.585 / 136.638 | 134.339 / 136.937 | 1.0049x |
| 2 | 740.84 | 725.73 | 0.9796x | 172.776 / 179.414 | 176.374 / 200.060 | 1.0011x |
| 4 | 1109.55 | 1178.33 | 1.0620x | 230.724 / 235.734 | 217.257 / 220.533 | 1.0956x |
| 8 | 1421.73 | 1605.61 | 1.1293x | 360.125 / 380.118 | 318.882 / 337.236 | 1.1178x |

B=8 decode-window median falls from 360.125 to 318.882 ms,
a 11.45% time reduction. The 1700 tok/s prediction would
require <=301.176 ms for 512 token events.
B=1 per-generated-token latency: L6 2.0716 ms,
L7 2.0991 ms. These are loop averages,
not individual GPU kernel timings.

## Validation

Executed, exit 0:

```text
.venv/bin/python -m pytest tests/ -q
133 passed, 1 warning in 31.65s

.venv/bin/python bench/l7_paged_attn_bench.py
PASS: paired token oracle, pool reclamation, and unchanged L6 protocol; prediction recorded separately
```

Full final outputs: `tests-compiled-materialization.txt` and
`bench-compiled-materialization.txt`. The warning is the existing PyTorch
quantization API deprecation. Syntax compilation and `git diff --check` pass;
there is no repository lint configuration or installed lint runner.

New coverage includes greedy/stochastic sampling; singleton, equal and unequal
prompts; 33 token events with page-boundary crossings; dynamic arrival and
completion; full/partial shared-prefix CoW; row permutation; lazy/empty and
missing layers; exact-zero masked probabilities; context/capacity errors;
cancellation and freed-page reuse; and trained d512 checkpoint comparisons.
Tests also reject any legacy `append`/`gather` call during new-engine decode,
and directly instrument one vectorized `take` for combined K/V materialization.

`scope-audit.json`: 53 pre-existing functions/methods source-identical;
30 existing test/bench/model/transformer/sampler/server/script files
byte-identical. Existing implementation files have zero removed lines versus
`a89132b`, and every changed path is whitelisted. Sampling code, dependencies,
old tests and old benchmarks are unchanged.

## Retained first attempt, pitfalls and limits

The uncompiled bank path, implementation `fbbe38e62216e11d85dfc79be9ec5181477d8ec8`, config
`16e6ba63e56f`, is preserved at `results/l7/paged-attn-bench-20261021T154156026596Z.json` and
`bench-acceptance.txt`:

| B | Paired L6 tok/s | Uncompiled L7 tok/s | Paired ratio |
|---|---:|---:|---:|
| 1 | 491.92 | 452.35 | 0.9196x |
| 2 | 764.22 | 723.88 | 0.9472x |
| 4 | 1104.03 | 1182.44 | 1.0710x |
| 8 | 1434.17 | 1603.95 | 1.1184x |

Its B=1 latency rose 8.75% and B=8 remained below
1700. The only follow-up change compiles the new pure bank transform; all
correctness checks and the exact same benchmark protocol were rerun.
The final singleton regression is smaller, while B=8 stays near 1600 tok/s.
These are separate paired runs; cross-run drift prevents treating their
singleton difference as a precise isolated compile speedup.

The result supports a measurable cost for the old row-gather path, but does
not establish that KV reconstruction is the main row-dependent cost or that
host synchronization dominates the whole workload. Sampling remains untouched
and unmeasured as an isolated intervention (T2). No Metal trace or memory
traffic measurement was collected. Residual page writes, metadata handling,
bank copying, attention padding, and per-request synchronization remain
possible costs; their individual shares are unproven.

The unchanged bf16 regression test emits timestamped JSON under `results/l5/`.
Only newly emitted artifacts were relocated intact under `results/l7/` after
each suite; pre-existing artifacts were neither edited nor removed. This
known test side effect is the only transient write outside the whitelist.
The invocation supplied no literal `-o` path: its section 6 summary is saved as
`results/l7/l7-t1-output.md`, and this full report is
`results/l7/l7-t1-report.md`. No dependencies or protocol changes. No push.

## Actual changed files

- `bench/l7_paged_attn_bench.py`
- `layers/__init__.py`
- `layers/attention.py`
- `layers/engine.py`
- `layers/paged_kv.py`
- `results/l7/bench-acceptance.txt`
- `results/l7/bench-compiled-materialization.txt`
- `results/l7/bf16-train-20261021T153617876397Z.json`
- `results/l7/bf16-train-20261021T154102370179Z.json`
- `results/l7/bf16-train-20261021T154306713377Z.json`
- `results/l7/checkpoint-oracle.json`
- `results/l7/evidence-verification.json`
- `results/l7/l7-t1-output.md`
- `results/l7/l7-t1-report.md`
- `results/l7/paged-attn-bench-20261021T154156026596Z.json`
- `results/l7/paged-attn-bench-20261021T154404587918Z.json`
- `results/l7/scope-audit.json`
- `results/l7/tests-acceptance.txt`
- `results/l7/tests-compiled-materialization.txt`
- `runs/l7-implementation-notes.md`
- `tests/test_l7_paged_attention.py`

## Staged commits

- `ad87130`: `docs(l7): record paged attention implementation plan`
- `fbbe38e`: `feat(l7): add direct paged batch attention`
- `a266903`: `perf(l7): record initial paged attention measurement`
- `e5b5755`: `perf(l7): compile page bank materialization`
- Final report/evidence archive commit follows verification; no push.
