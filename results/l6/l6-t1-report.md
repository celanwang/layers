# L6-T1 kernel batching report

L6-T1 is implemented and both section 5 acceptance commands pass. Final B=8
throughput is 4.514x the unchanged L4 per-request baseline.
All 91 existing tests remain byte-identical; 7 new tests bring the total to 98.
No push was performed.

## Implementation and scope

`BatchedEngine` inherits L4 admission, prefix sharing, and page reservations.
Synchronous `add_request` calls prefill prompts separately at iteration boundaries;
`step` samples in stable request order, forwards continuing IDs as `[B, 1]`,
assigns logits by that row order, and frees terminal requests at iteration end.
Per-row RoPE positions and negative-infinity masks isolate gathered, right-padded
KV views. Padding probabilities are checked after softmax with atol=1e-7 and an
additional exact-zero assertion. Persistent page payloads contain no padding.

All dense layers project `[B, D]` matrices. Two pure arithmetic helpers use
`mx.compile`; no `mx.fast` operation is used. The vocabulary projection uses two
half-width batched dot products to reduce fp32 accumulation error. The original
single-request methods are AST-identical, and inference.py/paged_kv.py are
byte-identical. No dependencies were added and no external inference engine or
serving framework implementation source was read. Only this repository and
installed public MLX API docstrings were consulted.

Implementation commits (each after passing tests and syntax/whitespace checks):

- `7e9bfdc` feat(l6): add batched paged decode
- `4c0db05` perf(l6): fuse batched attention math
- `665733f` perf(l6): batch dense projections as matrices
- `e89de74` fix(l6): bound batched logit accumulation error

The scope baseline is `4893dfe2ad52`, the parent of the first implementation
commit. The independently added T2 task specification was preserved.

## Actual changed file list

```text
bench/l6_batching_bench.py
layers/__init__.py
layers/attention.py
layers/engine.py
layers/model.py
layers/transformer.py
results/l6/batching-bench-20261021T130956006308Z.json
results/l6/batching-bench-20261021T131308951824Z.json
results/l6/batching-bench-20261021T131618165992Z.json
results/l6/batching-bench-20261021T132523026893Z.json
results/l6/bench-acceptance.txt
results/l6/bench-attempt1.txt
results/l6/bench-dense-acceptance.txt
results/l6/bench-final.txt
results/l6/bf16-train-20261021T130723177363Z.json
results/l6/bf16-train-20261021T131126158325Z.json
results/l6/bf16-train-20261021T131458011056Z.json
results/l6/bf16-train-20261021T132355686368Z.json
results/l6/checkpoint-logit-oracle.json
results/l6/checkpoint-oracle-attempt1.json
results/l6/l6-t1-report.md
results/l6/linear-layout-probes.json
results/l6/projection-precision-probe.json
results/l6/scope-checks.json
results/l6/tests-acceptance.txt
results/l6/tests-dense-acceptance.txt
results/l6/tests-final.txt
results/l6/tests-stage1.txt
runs/l6-implementation-notes.md
tests/test_batched_engine.py
```

## Acceptance commands and complete output

Command: `.venv/bin/python -m pytest tests/ -q` (exit 0).

```text
........................................................................ [ 73%]
..........................                                               [100%]
=============================== warnings summary ===============================
tests/test_kv_quant.py::test_kv_roundtrip_bound_and_pytorch_public_api_oracle
  /Users/wangchen/Developer/layers/tests/test_kv_quant.py:38: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
    oracle = torch.quantize_per_channel(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
98 passed, 1 warning in 21.88s
```

Command: `.venv/bin/python bench/l6_batching_bench.py` (exit 0).

```text
device: Apple M4 Max, 48 GiB, mlx 0.32.4
config_hash: d2f97623b7b7 (commit e89de74f2ea8)
B=1: L4 364.18 tok/s, batched 480.32 tok/s, speedup 1.319x; decode median/p90 0.133245/0.141046s
B=2: L4 339.03 tok/s, batched 718.82 tok/s, speedup 2.120x; decode median/p90 0.178069/0.186463s
B=4: L4 328.02 tok/s, batched 1073.15 tok/s, speedup 3.272x; decode median/p90 0.238549/0.242834s
B=8: L4 310.22 tok/s, batched 1400.44 tok/s, speedup 4.514x; decode median/p90 0.365600/0.381734s
B=1 latency: L4 2.7459 ms/token, batched 2.0820 ms/token; regression -24.18%
uneven B=8: small 1180.55 tok/s, large 1154.92 tok/s; penalty 2.17%
padding fractions: small 0.7752%, large 40.7407%
predictions: {"b1_regression_below_10_percent": true, "b8_speedup_at_least_4x": true}
written: results/l6/batching-bench-20261021T132523026893Z.json
PASS: L4 token oracle, B=8 throughput >=4x, B=1 latency regression <10%
```

Additional validation: `python -m compileall -q` for the seven changed Python
files and `git diff --check` both passed with empty output. There is no separate
repository lint configuration or installed lint runner. The scope audit verifies
9 existing test files and 25 existing functions/methods unchanged, and the final
benchmark's source hashes exactly match the delivered implementation.

## Three benchmark groups

Hardware: Apple M4 Max, 48 GiB unified memory,
GPU fp32; macOS-15.8-arm64-arm-64bit, Python 3.12.15, MLX 0.32.4.
L3 checkpoint `results/l3/ckpt-final-500.npz`; configuration hash
`d2f97623b7b7`, implementation commit `e89de74f2ea804f7b66025661059d5ffa1d6e27d`.
3 warmups and 10 timed repeats per path, paired alternating order, fresh engines
and pools, median/p90 plus raw samples. Each request emits 64 tokens using the
same seeded sampler configuration. Throughput timing includes sampling,
scheduling, KV operations and GPU synchronization; prefill is excluded from the
primary decode comparison and separately recorded with end-to-end timings.
The first token uses prompt logits, so 64 token events involve 63 decode forwards.

1. Throughput scaling, equal 32-token prompts:

| B | L4 tok/s | Batched tok/s | Speedup | Batched median / p90 seconds |
|---|---:|---:|---:|---:|
| 1 | 364.18 | 480.32 | 1.319x | 0.133245 / 0.141046 |
| 2 | 339.03 | 718.82 | 2.120x | 0.178069 / 0.186463 |
| 4 | 328.02 | 1073.15 | 3.272x | 0.238549 / 0.242834 |
| 8 | 310.22 | 1400.44 | 4.514x | 0.365600 / 0.381734 |

2. Single-request latency: L4 2.7459 ms/token,
batched 2.0820 ms/token;
-24.18% change. These are per-generated-token decode
loop averages, with both paths using identical generation counts.

3. Uneven-length B=8 comparison (same mean prompt length 96):

| Group | Prompt lengths | Length variance | tok/s | Median / p90 seconds | Padded key slots wasted |
|---|---|---:|---:|---:|---:|
| Small variance | [95, 95, 96, 96, 96, 96, 97, 97] | 0.50 | 1180.55 | 0.433696 / 0.457107 | 0.7752% |
| Large variance | [8, 24, 40, 72, 120, 152, 168, 184] | 4160.00 | 1154.92 | 0.443320 / 0.457886 | 40.7407% |

Large/small throughput ratio: 0.97829; measured
penalty 2.17%. Key-slot accounting is summed
across all 63 decode forwards per head/layer. Padding waste rises substantially,
while wall-time effects remain small; earlier trials ranged from -0.35% to 3.31%
and are retained. This run alone does not establish a stable padding cost curve.

## Section 5 verdicts and numerical evidence

- PASS: eight variable KV prefixes, at least 32 actual decode forwards, per-step
  logits within absolute atol=1e-5 with rtol=0. Both tiny-model greedy/stochastic
  tests and the trained checkpoint are covered. Trained-checkpoint maximum
  absolute error: `7.62939453125e-06`; all seeded token events match.
- PASS: row permutation, exact-zero post-softmax pad probabilities, dynamic
  insertion/completion with shared pages, capacity/context boundaries, and full
  resource reclamation.
- PASS: all 91 pre-existing L1–L5 tests unchanged, 98 tests total.
- PASS: B=8 throughput 4.514x >= 4x.
- PASS: B=1 latency change -24.18% < 10% regression.
- The principle note's illustrative ~3k tok/s projection is not reached:
  measured B=8 throughput is 1400.44 tok/s.
  The task's explicit relative-throughput gate passes.

## Deviations, failed experiments, and limits

No acceptance gate, existing test, measurement protocol, or dependency changed.
Admission/prefill uses inherited synchronous boundary calls; callers serialize
`add_request` and `step`. This is the iteration boundary model used by T1.
HTTP/SSE and external-source comparisons remain outside this task's scope.

The initial uncompiled run missed the 4x gate (3.992x); its complete output and
JSON are preserved. A supplementary trained-checkpoint probe then exposed two
absolute errors up to 1.1444091796875e-5. Compile toggles and matrix layout probes
left those outliers unchanged. An fp64 CPU diagnostic isolated vocabulary-dot
accumulation error; splitting that reduction into two batched halves reduced the
maximum error to 7.62939453125e-06. The tolerance was kept unchanged.
All failed records, intermediate measurements, and the final successful oracle
are retained under `results/l6/`. CPU fp64 was diagnostic only.

The unchanged bf16 regression test emits a timestamped JSON under `results/l5/`.
Only those newly generated files were moved intact into the whitelisted
`results/l6/`; no existing L5 artifact was edited or removed. No tracked changes
outside the whitelist remain. The only warning is the existing PyTorch
quantization API deprecation shown in the complete pytest output above.

This report is the section 6 output artifact, stored at
`results/l6/l6-t1-report.md`. Final archive commit follows validation; no push.
