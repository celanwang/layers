# L7-T1 implementation notes

## Authorized design and blind implementation

Read `AGENTS.md`, `runs/l7-decode-perf-principle.md`, and
`tasks/l7-paged-attention.md`. Work stays on `l7-decode-perf`.
No external inference framework attention/kernel source is consulted.
The project’s existing `BatchedEngine`, attention methods, page methods,
sampler, tests, and L6 measurement harness remain unchanged.

- Add a write-only page append method preserving the existing validation,
  lazy allocation, and copy-on-write rules; it never calls `gather`.
- Build a compact bank of referenced physical pages for one layer, combine
  K/V in that bank, and use one vectorized page-table index for the batch.
  Reshape the selected pages to block-aligned attention tensors. Logical
  lengths mask stale tail slots and repeated dummy pages for short rows.
- Add a length-masked attention method, retaining the existing projection
  arithmetic and per-row absolute RoPE positions.
- Add a `BatchedEngine` subclass with the same sampling/event/release order.
  A local model decode function repeats the existing block/head arithmetic
  because `model.py` and `transformer.py` are outside the whitelist.
- Import L6 `measure_pair` and `load_model` directly. Keep all warmup, repeat,
  timing, alternating-order, checkpoint, prompt, and generation settings.
  Performance predictions are verdicts, never reasons to adjust the protocol.

## Initial verification

Before edits: `.venv/bin/python -m pytest tests/ -q` exits 0 with
119 passed and one existing PyTorch quantization deprecation warning.
The unchanged bf16 test emits a new timestamped artifact in `results/l5/`;
this newly generated file is relocated intact into `results/l7/`, following
L6 archival practice. Existing artifacts are untouched.

## Reporting destination

The invocation did not specify a literal `-o` path. The section 6 execution
summary will be saved as `results/l7/l7-t1-output.md`; the full report is
`results/l7/l7-t1-report.md`.

## Component and oracle verification

Added `PagedKVPool.append_pages` and `materialize_batch`; a compact bank uses
one `mx.take` for both K/V across all rows, with no per-row gather. Page payloads
remain lazy, fixed-size physical blocks with logical tail lengths. Unused
blocks may point at a real page: only the length mask grants visibility.
This is vectorized materialization, not a fused Metal PagedAttention kernel.

`PagedAttentionEngine` inherits L6 admission/cancel/prefill and duplicates only
its step orchestration to select the new decode path. Sampler invocations and
release order are unchanged. The local decode function retains L6 residual,
MLP, normalization, and two-half vocabulary projection operation order.

- New tests: 14 passed; full suite: 133 passed (119 pre-existing + 14 new),
  one pre-existing PyTorch warning. Full output: `results/l7/tests-acceptance.txt`.
- Trained d512 checkpoint: eight variable prefixes, 33 token events each,
  32 actual decode forwards; max absolute logit error versus L6 is 0.0.
  All seeded tokens match and both pools are fully reclaimed.
- Scope audit against `a89132b`: 53 existing methods source-identical;
  30 existing tests/bench/model/transformer/sampler/server/script files
  byte-identical. See `results/l7/scope-audit.json`.
- Syntax compilation and `git diff --check` pass. No project lint
  configuration or installed lint runner is declared in the repository.

## First paired measurement (retained)

Implementation `fbbe38e`, config `16e6ba63e56f`,
`paged-attn-bench-20261021T154156026596Z.json`: B=1/2/4/8 new throughput
452.35 / 723.88 / 1182.44 / 1603.95 tok/s. Paired L6 throughput:
491.92 / 764.22 / 1104.03 / 1434.17 tok/s.
B=8 improves 1.118x against the paired oracle, but misses the >=1700
prediction; it is above the <1550 setback boundary (intermediate verdict).
Singleton latency rises 8.75%, so zero regression is not established.

The new bank construction/index/transpose/reshape has multiple separate MLX
host calls. A bounded follow-up will compile only that pure array transform,
leaving page lookup/writes, sampling, and the exact measurement protocol intact.
This is an implementation change, not repetition of an unchanged benchmark
until a preferred timing appears. All first-run evidence remains archived.
