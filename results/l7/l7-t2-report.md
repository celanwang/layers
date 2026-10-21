# L7-T2: batched sampling with one host fence

## Outcome

Both section 5 commands pass. The full suite has **156 passing tests**
(133 previous + 23 new), with one existing PyTorch deprecation warning.
RNG token equality is exact: mixed temperature/top-k/top-p settings, prior draw
histories, changing row order and membership, and fp32/fp16/bf16 all match the
unchanged scalar `Sampler`. Trained d512 checkpoint tokens match all three
engines; T2/T1 logits are bit-exact and maximum absolute error versus L6 is 0.0.

In the direct **T1-only / T1+T2** B=8 pairing, throughput rises from
**1663.67 to 1918.45 tok/s**
(**+15.31%**). Decode median falls from
307.753 to 266.882 ms
(**13.28%** time reduction).
The section 5 >=5% throughput prediction is **supported**, exceeding the
original approximate 5–10% decode-time prediction.

The independent **L6 / T1+T2** B=8 pairing is
1450.15 / 1891.05 tok/s,
**+30.40%** throughput
(23.32% less decode time).
It exceeds the principle note's combined >=1800 tok/s and >=1.25x reference.
These controlled interventions support a measurable host operation/synchronization
cost in this local small-model workload. They do not quantify every remaining
row-dependent cost or replace a Metal trace.

## Three-stage paired results

Each row below is its own unchanged L6 paired experiment. The repeated T1 or
L6 values belong to different pairs; no cross-pair subtraction or chained
speedup is used to attribute T2. All B=8 stages are reported explicitly:

| Pair | L6 baseline tok/s | T1-only tok/s | T1+T2 tok/s | Within-pair gain |
|---|---:|---:|---:|---:|
| l6_vs_t1 | 1449.17 | 1645.67 | — | +13.56% |
| t1_vs_t2 | — | 1663.67 | 1918.45 | +15.31% |
| l6_vs_t2 | 1450.15 | — | 1891.05 | +30.40% |


### l6_vs_t1

| B | l6 tok/s | t1 tok/s | Throughput gain | l6 decode median / p90 ms | t1 decode median / p90 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 494.13 | 468.78 | -5.13% | 129.521 / 132.249 | 136.526 / 142.278 |
| 2 | 750.43 | 737.70 | -1.70% | 170.569 / 174.634 | 173.513 / 178.068 |
| 4 | 1104.05 | 1150.70 | +4.23% | 231.873 / 232.327 | 222.473 / 225.660 |
| 8 | 1449.17 | 1645.67 | +13.56% | 353.307 / 355.874 | 311.120 / 320.273 |


### t1_vs_t2

| B | t1 tok/s | t2 tok/s | Throughput gain | t1 decode median / p90 ms | t2 decode median / p90 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 468.91 | 477.29 | +1.79% | 136.486 / 137.373 | 134.091 / 137.066 |
| 2 | 741.67 | 764.61 | +3.09% | 172.583 / 175.190 | 167.406 / 169.441 |
| 4 | 1154.52 | 1310.13 | +13.48% | 221.737 / 230.170 | 195.401 / 205.235 |
| 8 | 1663.67 | 1918.45 | +15.31% | 307.753 / 312.361 | 266.882 / 269.831 |


### l6_vs_t2

| B | l6 tok/s | t2 tok/s | Throughput gain | l6 decode median / p90 ms | t2 decode median / p90 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 484.83 | 466.04 | -3.88% | 132.004 / 134.099 | 137.328 / 139.067 |
| 2 | 762.02 | 755.33 | -0.88% | 167.975 / 172.452 | 169.463 / 173.955 |
| 4 | 1101.28 | 1310.10 | +18.96% | 232.456 / 239.911 | 195.405 / 202.001 |
| 8 | 1450.15 | 1891.05 | +30.40% | 353.068 / 358.729 | 270.749 / 272.826 |


B=1 T2 versus T1 reduces decode time by
1.76%.
Against L6, the independent combined B=1 pairing has
**+4.03% decode latency**
and lower throughput. Combined B=2 throughput is also slightly lower than L6.
The optimization is therefore not a uniform improvement across batch sizes.
The performance gate applies to B=8; small-batch results remain in the record.

## Implementation and comparison with the immutable oracle

`sample_batch(samplers, logits)` accepts standard `Sampler` state/configuration
and stacked `[B, V]` logits, returning a Python list of B token IDs. It reuses
`Sampler.probabilities` without changing temperature, truncation or constructor
validation. For each row, the exact old operation order remains:

1. Compute that sampler's probabilities on the original scalar-row shape.
2. Call `mx.random.seed(sampler.seed + sampler._draws)` and increment `_draws`.
3. Build `mx.random.categorical(mx.log(probs))` with its original scalar shape.

The B lazy token arrays are stacked into `[B]`, passed to **one `mx.eval`**,
then copied with **one `tolist()`**. There is no per-row `.item()`, NumPy
conversion or synchronization. Keys are captured while constructing categorical;
subsequent seed resets do not overwrite earlier lazy draws. The tests verify
both emitted tokens and final global PRNG behavior against the scalar oracle.
A single categorical on `[B, V]` would change its random shape/stream and is
not substituted. Greedy and top-k=1 still consume their original categorical
draw, preserving history if parameters subsequently change. Reusing one sampler
object in multiple rows also preserves the original sequential draw order.

`BatchedSamplingEngine(PagedAttentionEngine)` provides additive step wiring,
following T1's subclass pattern. It collects active samplers/logits, samples the
batch, then uses the existing event, finish, continuing-row decode and deferred
free order. Continuing requests use the exact `_DirectPagedKVCache` and
`_decode_paged_batch` calls from T1. KV, attention, model and projection code
are untouched. The original `BatchedEngine.step`, `PagedAttentionEngine.step`,
and all existing call paths remain source-identical and independently callable.
The new class and `sample_batch` are exported from `layers`.

The existing admission API accepts arbitrary callable samplers. A batch
containing one, including a `Sampler` subclass with an overridden call,
follows the inherited T1 step for the entire iteration. This preserves old
call order and semantics. The one-fence guarantee applies to batches of
standard `Sampler` instances. The performance harness and all token-exact
fast-path tests use those standard instances; wrappers are not used to
accidentally bypass the optimized path.

This is the source comparison against the project's immutable L6/T1 oracle.
No external inference-framework source was read. Only the provided principle,
specification, project code and installed public MLX API docstrings were used.

## Measurement protocol and reproducibility

- Hardware: Apple M4 Max, 48 GiB unified memory;
  macOS-15.8-arm64-arm-64bit; Python 3.12.15; MLX 0.32.4; fp32 GPU.
- Implementation commit: `2f81b785f4bddc64e2ecf4abeb3438997d9ca5f7`.
  Config hash: **`08e7ca285efb`**.
- Checkpoint: `results/l3/ckpt-final-500.npz`;
  SHA-256 `a6bc079ef679c90ea8a4fdff95cf5086d4d49b8c811ce0252b2c31c7f1531889`.
- Raw evidence: [`sampling-bench-20261021T155558005176Z.json`](sampling-bench-20261021T155558005176Z.json);
  stdout and exit status: [`t2-bench-acceptance.txt`](t2-bench-acceptance.txt).
- Import the unchanged `bench.l6_batching_bench.measure_pair`, which itself uses
  unchanged `run_once`, `summarize`, model loader, constants and synchronization.
  The original T1 benchmark is also unmodified.
- Each of L6/T1, T1/T2, L6/T2 uses 3 warmups + 10 repeats per path,
  paired alternating order and a fresh engine/pool each trial. B=1/2/4/8;
  32-token distinct prompts; 64 generated tokens/request and 63 decode forwards;
  block size 16; no prefix sharing; seed 20261021; temperature=0.
- Decode includes sampling, scheduling, KV work and synchronization. Prefill
  remains separate, and end-to-end measurements are retained. Median is L6's
  upper order statistic for 10 samples; p90 is sorted index 8. Every raw sample
  and source hash is retained. The protocol dictionary exactly matches the
  historical L6 JSON, including unchanged legacy variance metadata; only the
  equal-prompt scaling workload specified for T1/T2 is run.
- The three comparison edges each keep L6's paired protocol. There is no new
  three-way timing harness. T2 attribution uses `t1_vs_t2` only; combined
  attribution uses `l6_vs_t2` only. This avoids treating cross-run drift as
  component performance.
- [`t2-evidence-verification.json`](t2-evidence-verification.json) independently
  verifies all **720** raw timing values and summary statistics, throughput
  arithmetic, pairing ratios, config hash, checkpoint/historical hashes, and
  source hashes against both the worktree and recorded implementation commit.
- Every paired trial checks exact output equality and full pool reclamation:
  156 paired trials / 312 engine runs including warmups; 585 paired request
  sequences / **37,440 token comparisons**. There was one formal benchmark run;
  no unfavorable measurement was discarded or rerun for a preferred outcome.

## Acceptance and coverage

Both required commands exited 0:

```text
.venv/bin/python -m pytest tests/ -q
156 passed, 1 warning in 27.30s

.venv/bin/python bench/l7_sampling_bench.py
PASS: paired token oracles, pool reclamation, unchanged L6 protocol; prediction recorded separately
```

Full test stdout: [`t2-tests-acceptance.txt`](t2-tests-acceptance.txt).
The warning comes from the existing PyTorch quantization oracle.
`py_compile` for all new/extended Python files and `git diff --check` pass;
no project lint configuration or lint command is declared.

The 23 added tests cover:

- Exact scalar-oracle tokens in fp32/fp16/bf16, mixed temperature and top-k/top-p,
  greedy, top-k=1, top-k above vocabulary size, tiny/full top-p, tied/uniform
  logits, 48 changing batches, prior draws, duplicate seeds, unrelated global
  RNG activity and subsequent parameter changes.
- Shared sampler objects, early shape/count/type validation and empty batches.
  Instrumentation enforces exactly one `[B]` eval and one host list transfer,
  rejecting scalar sampler calls, `.item()`, NumPy reads and synchronize calls.
- Singleton, equal/unequal prefixes and greedy/random/mixed parameters for
  33 tokens; dynamic arrival, completion, forked shared prefixes, cancellation,
  freed-page reuse, context/capacity limits, events and pool reclamation.
- Explicit batch-entry/direct-page-decode wiring; callable/subclass fallback;
  trained d512 checkpoint versus all three stages with 8 mixed requests x
  33 tokens. Evidence [`t2-checkpoint-oracle.json`](t2-checkpoint-oracle.json).
  T2/T1 logits are exact; max absolute error to L6 is 0.0 (limit 1e-5).

## Scope, pitfalls and deviations

Baseline: `0668953`. [`t2-scope-audit.json`](t2-scope-audit.json) verifies the
old inference/engine definitions source-identical, existing Python edits purely
additive, implementation notes append-only, and all 237 tracked files outside
the whitelist byte-identical. Old tests, benchmarks, dependencies, KV and
attention paths are unchanged. Only whitelisted files are delivered.

RNG preservation requires keeping the per-row seed and categorical construction;
T2 removes synchronization boundaries rather than random draws or row work.
The larger lazy graph could have lost performance; the measured B=8 outcome
supports the prediction here. The benchmark uses the original greedy workload;
stochastic token correctness is separately tested, but stochastic performance
and other hardware/model shapes are not measured. No Metal trace was collected.

The unchanged bf16 regression emits a timestamped JSON under `results/l5/`.
Following T1 precedent, only the new artifact was relocated intact to
`results/l7/bf16-train-20261021T155410864907Z.json`; no pre-existing file was
removed or edited. This inherited test side effect is the sole transient
write outside the whitelist. Test content and measurement protocol were not
modified to suppress it.

No literal `-o` filename was supplied. The section 6 summary is
[`l7-t2-output.md`](l7-t2-output.md), following T1's naming convention.
Additive subclass wiring and the three independently paired comparisons above
resolve the immutability/protocol constraints without changing old paths.
No dependency, RNG, KV, threshold or measurement-protocol deviation. No push.

## Actual changed files

- `bench/l7_sampling_bench.py`
- `layers/__init__.py`
- `layers/engine.py`
- `layers/inference.py`
- `results/l7/bf16-train-20261021T155410864907Z.json`
- `results/l7/l7-t2-output.md`
- `results/l7/l7-t2-report.md`
- `results/l7/sampling-bench-20261021T155558005176Z.json`
- `results/l7/t2-bench-acceptance.txt`
- `results/l7/t2-checkpoint-oracle.json`
- `results/l7/t2-evidence-verification.json`
- `results/l7/t2-scope-audit.json`
- `results/l7/t2-tests-acceptance.txt`
- `runs/l7-implementation-notes.md`
- `tests/test_l7_batched_sampling.py`

## Staged commits

- `63f5789`: `docs(l7): record batched sampling plan`
- `2f81b78`: `feat(l7): add single-fence batched sampling`
- The final archive commit contains this report, section 6 output, paired
  measurements, evidence verification and the updated scope audit. No push.
