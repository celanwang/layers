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

## Compiled materialization verification

Moved the pure bank concatenate/index/transpose/reshape into a compiled helper.
Consulted only the installed public `mx.compile` docstring (array-tree inputs
and shape specialization); no MLX implementation source was opened.
Page metadata and CoW remain ordinary Python. A single `mx.take` still selects
both K/V. The full suite passes again: 133 passed, one existing warning,
31.65 seconds; syntax and whitespace checks pass. This implementation change
is the sole reason for the second paired acceptance measurement.

## Final acceptance and interpretation

- Full suite: 133 passed / one existing warning; all 119 old tests unchanged.
- Final benchmark exits 0; config `4e04eb84dc1c`, implementation
  `e5b5755002e6f2427a5b1fd322b1ea3c98e6a606`, record
  `results/l7/paged-attn-bench-20261021T154404587918Z.json`.
- B=1/2/4/8 new throughput: 476.41 / 725.73 / 1178.33 / 1605.61 tok/s;
  paired L6: 482.71 / 740.84 / 1109.55 / 1421.73 tok/s.
- B=8: 1.1293x paired L6,
  1.1178x historical 1436.41 tok/s.
  >=1700 prediction not reached; <1550 setback not triggered (intermediate).
- Singleton decode latency +1.32%; strict zero regression is not proven.
  Do not report unqualified performance success. T2 sampling remains untouched.
- Final audit reconfirms 53 existing methods and 30 existing files unchanged;
  existing implementation changes are entirely additive and whitelisted.
- Protocol equality, config hash, all source hashes and 240 raw timing samples
  are verified in `results/l7/evidence-verification.json`.
- Full report: `results/l7/l7-t1-report.md`; section 6 / -o summary:
  `results/l7/l7-t1-output.md`. Both retained measurements are described there.
  No new dependencies, external framework implementation reading, or push.

## L7-T2 authorized implementation plan

Read `AGENTS.md`, `runs/l7-decode-perf-principle.md`, and
`tasks/l7-batched-sampling.md`. The user explicitly authorizes execution of T2.
Baseline is `0668953` on `l7-decode-perf`, with a clean worktree and the
accepted 133-test T1 result. Changes are confined to the T2 whitelist.

- Add `sample_batch(samplers, logits)` for `[B, V]` input. Reuse each existing
  `Sampler.probabilities` verbatim; retain per-row `seed(seed + draws)`, draw
  increments and scalar categorical shape, including greedy cases. Stack lazy
  token arrays, evaluate once, and transfer once with `tolist()`.
- Add `BatchedSamplingEngine(PagedAttentionEngine)` as the additive step
  implementation, preserving T1 cache/decode calls, events and deferred frees.
  Existing engines and `Sampler` stay source-identical. Arbitrary callable
  samplers continue through the inherited T1 path; the single-fence guarantee
  applies to batches of standard `Sampler` instances.
- Test token-exact sampling with mixed parameters, prior draws, request/row
  permutations, dynamic arrivals/completions, shared-prefix cancellation,
  context/capacity limits, and the trained d512 checkpoint. Instrument eval
  and host transfer to enforce one batch synchronization and no scalar reads.
- Import the unchanged L6 paired harness for each of L6/T1, T1/T2, L6/T2,
  retaining 3 warmups / 10 repeats, alternating order, raw timings, hashes and
  token oracles. Compute the T2 prediction from the directly paired T1/T2 arm.
- Run both section 5 commands, audit additive scope, commit verified stages,
  and archive all outcomes. No literal `-o` path was supplied: the summary is
  `results/l7/l7-t2-output.md`; full report `results/l7/l7-t2-report.md`.

Only existing project code and installed public MLX API docstrings have been
read. No external reference implementation source, dependency change or push.

## L7-T2 component and oracle verification

`sample_batch` builds the same per-row probability/seed/categorical graph as
`Sampler.__call__`; only the eval/host-transfer boundary is shared. It also
preserves `_draws` and the final global PRNG state. Greedy/top-k=1 still run
categorical and consume their original draw; no argmax shortcut or new RNG
key/shape is introduced. Validation occurs before any draw.

`BatchedSamplingEngine` overrides only step orchestration. Its continuing-row
cache adapter and decode call are the exact T1 calls; old `BatchedEngine`,
`PagedAttentionEngine`, `Sampler`, KV and attention implementations are intact.
Whole-batch fallback preserves custom callable and Sampler-subclass behavior.

- New tests: 23 passed, including fp32/fp16/bf16 token-exact sampling, mixed
  truncations, existing draw histories, batch membership/order changes, aliased
  sampler state, and the final global PRNG state. Instrumentation confirms
  exactly one eval of `[B]` and one `tolist`, rejecting item/NumPy/synchronize
  and scalar sampler calls. Engine tests confirm T1 direct-page decode calls.
- Section 5 full suite: 156 passed (133 previous + 23 new), one pre-existing
  PyTorch deprecation warning, 27.30s. Output `t2-tests-acceptance.txt`.
- Trained d512 oracle: 8 mixed-parameter requests x 33 tokens match L6/T1/T2;
  T2/T1 logits are bit-exact, maximum absolute error versus L6 is 0.0;
  complete pool reclamation. Evidence `t2-checkpoint-oracle.json`.
- Syntax compilation and `git diff --check` pass. No lint configuration or
  lint command is declared by the project.
- Scope audit: 14 pre-existing top-level definitions (including complete old
  classes) source-identical; 237 tracked files outside the whitelist byte-
  identical. Existing implementation edits contain only added lines.
- The unchanged bf16 test emits one timestamped `results/l5/` JSON; following
  T1 precedent, only that newly emitted artifact was relocated intact into
  `results/l7/`. This known test side effect is the sole transient write outside
  the whitelist. No pre-existing artifact was changed or removed.

Implementation and test verification are committed before the formal paired
performance run so the measurement records a reproducible implementation hash.

## L7-T2 formal measurement and final interpretation

Both section 5 commands exit 0. One formal performance run, implementation
`2f81b785f4bd` (full hash in JSON), config `08e7ca285efb`, record
`results/l7/sampling-bench-20261021T155558005176Z.json`:

- B=8 L6/T1 pair: 1449.17 / 1645.67 tok/s (+13.56%).
- B=8 T1/T2 pair: 1663.67 / 1918.45 tok/s (+15.31% T2 increment);
  decode median/p90 307.753/312.361 -> 266.882/269.831 ms,
  13.28% decode-time reduction. The >=5% throughput prediction is supported.
- B=8 L6/T2 pair: 1450.15 / 1891.05 tok/s (+30.40% combined);
  >=1800 tok/s combined prediction supported. Each ratio uses its own pair.
- T2/T1 gains for B=1/2/4: +1.79% / +3.09% / +13.48%. Combined T2/L6
  still has a B=1 latency regression of 4.03% and slightly lower B=2 throughput.
  No uniform small-batch performance improvement is claimed.
- The unchanged L6 harness supplies every timing/pairing operation. All source
  hashes match both the worktree and measurement commit; protocol equals L6's
  historical JSON exactly. Independently verified 720 raw timing values and
  aggregates. 312 engine runs / 37,440 paired token comparisons including
  warmups; all token/pool assertions pass. No discarded or repeated bench run.
- This controlled removal of synchronization boundaries supports their cost in
  the measured workload. Stochastic performance, other hardware/model shapes
  and the remaining cost distribution are unmeasured; no Metal trace.

Full report `results/l7/l7-t2-report.md`; section 6 / -o summary
`results/l7/l7-t2-output.md`; verification `t2-evidence-verification.json`.
Final scope audit and documentation checks accompany the archive commit.
No dependency, RNG, KV, threshold or protocol changes; no push.
