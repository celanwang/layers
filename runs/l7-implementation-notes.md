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
