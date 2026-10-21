# L5-T1 implementation notes

## Implementation and assumptions

- Quant/dequant are handwritten MLX core operations. Public matrices use
  `[output, input]`; project weights use `[input, output]`, with explicit
  transposition in `QuantizedLinear`. INT8 has one fp16 scale per output
  channel; INT4 has one fp16 scale per 64 input values and packs two signed
  two's-complement nibbles in each byte, low nibble first.
- The symmetric grid uses `qmax = 2**(bits-1)-1`, leaving the most-negative
  two's-complement code unused. Delta in the half-step error bound is the
  stored scale, rather than the affine range divided by `2**bits-1`.
- Scales are rounded upward to representable fp16 values before rounding
  integers. This prevents endpoint clipping from violating the half-step
  bound, including fp16 subnormal scales. Zero groups use scale 1; very tiny
  positive scales clamp to the smallest fp16 subnormal. Non-finite weights
  and ranges too large for finite fp16 scales raise explicit errors.
- Only block attention Q/K/V/O and MLP up/down projections are replaced.
  Quantized attention/MLP subclasses inherit the existing forward methods;
  properties supply freshly dequantized fp32 matrices. The embedding/tied
  head and layer norms keep their original fp32 array objects. No fp32
  projection copy is retained. Invalid configuration builds no partial
  conversion; quantizing an already quantized model is rejected.
- Initial interface probe: MLX array matmul raises `TypeError` before Python
  can dispatch a projection object's `__rmatmul__`. Dequantizing properties
  solve that constraint without changing attention, transformer, GPT, or
  inference code. No benchmark data was produced by that failed probe.
- No quantization reference implementation source was opened. Only public
  MLX API docstrings and calls were used. PyTorch per-channel quantization
  independently checks INT8 integers; MLX dequantization reconstructs our
  signed grid after an explicit unsigned packing/bias adapter. Native MLX
  affine quantize/dequantize is also measured, with its differing grid
  explicitly recorded rather than treated as identical symmetric semantics.

## Measurement definitions

- Load the unchanged L3 500-step checkpoint and reuse the L3 corpus split
  `len(all_ids)*9//10`. Quality evaluates **every** next-token pair inside
  the final 10%, with non-overlapping 256-token contexts, token weighting,
  and the final short context retained. It is a different estimator from
  L3's fixed sampled evaluation batch, using the same heldout partition.
- Storage measures actual uncompressed NPZ serialization in memory,
  recording SHA-256, array payload, container overhead, integer bytes,
  scale bytes and protected fp32 bytes. No large duplicate checkpoint files
  are needed. Both whole-model and projection-only compression are reported.
- Speed uses unchanged L4 `generate`/`Sampler`/`KVCache`, a 128-token heldout
  prompt and exactly 128 greedy generated tokens. The first generated token
  comes from prefill; timing between the first and last samples covers 127
  actual cached decode calls. The record also includes prefill/first-sample
  and full 128-token end-to-end timing, avoiding a 128/127 counting error.
- L0 protocol is unchanged: 3 warmups and 10 measured repetitions, its upper
  median and nearest-rank p90. Raw timing samples, hardware, package versions,
  source hashes, checkpoint/data/tokenizer hashes and full configuration
  accompany the JSON record. Benchmark outputs are timestamped, not replaced.

## Stage 1 validation

- Baseline before changes: `60 passed in 5.91s`.
- Full suite after implementation: `81 passed, 1 warning in 8.17s`.
- Full output: `results/l5/tests-stage1.txt`.
- The warning is PyTorch's public quantized-tensor API deprecation notice.
- `git diff --check` and Python syntax compilation passed. Ruff is not
  installed and this repository defines no lint configuration; no dependency
  was added.

## Final acceptance and measurements

- Both exact section-5 commands exited 0. The final test run was
  `81 passed, 1 warning in 7.43s`; existing tests remain byte-for-byte unchanged.
- Complete section-6 report (file manifest, full command output, measurements,
  prediction verdicts and deviations): `results/l5/l5-t1-report.md`.
- Raw record: `results/l5/quant-bench-20261021T121645745015Z.json`.
  Source commit: `d10f1f665a8a58a28abcce2ea723db40bc8a2843`;
  config hash: `f782bb66df96`; Apple M4 Max / 48 GiB / MLX 0.32.4;
  3 warmups, 10 repetitions; 44,467 heldout targets.

| Format | Serialized bytes | Whole-model compression | Projection compression | Loss | Relative loss increase | Decode tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 52,485,384 | 1.0000x | 1.0000x | 4.27175664 | 0.0000% | 394.87 | 1.0000x |
| INT8 | 14,780,712 | 3.5509x | 3.9883x | 4.27190090 | +0.0034% | 284.60 | 0.7208x |
| INT4 g64 | 8,845,608 | 5.9335x | 7.5294x | 4.29692684 | +0.5892% | 239.46 | 0.6064x |

### Predictions and open findings

- Memory: INT4's approximately 7.5x projection compression holds. The INT8
  approximately 4.1x premise is overturned as written: fp32-to-INT8 has a
  4x upper bound before scale overhead; measured projection compression is
  3.9883x. Protected fp32 arrays further lower whole-model compression.
- Quality: both predictions hold, with increases below 1% and 5%.
- Speed: the numeric INT8 interval [0.8x, 1.3x] is overturned by 0.7208x;
  the qualitative slowdown prediction holds. INT4 is slower still. Added
  unpack/dequantize work plausibly contributes, but kernel-level attribution
  is an open question; this measurement does not profile dispatch or memory.
- MLX reconstructs our adapted signed grid exactly in both formats. Native
  affine results differ because they use a bias and a different grid; full
  error/RMSE comparisons are retained in JSON and the report.

### Execution commits and scope

- `d10f1f6` — `feat(l5): add handwritten weight quantization`.
- `71bf8ee` — `test(l5): record quantization acceptance`.
- Report archival is a separate documentation commit. No push was performed.
- During execution, an independent `c6a354f` documentation commit added the
  T2 specification on the shared branch. This task did not change or stage
  that file. Its own commits remain entirely within the T1 whitelist.

## T2 implementation and assumptions

- Read `AGENTS.md`, the L5 principle note, and `tasks/l5-kv-bf16.md` before
  implementation. No quantization/mixed precision reference source was read.
  The existing T1 quantizer is reused; public PyTorch quantized-tensor API
  outputs independently check the new per-token layout. No dependencies added.
- `quantize_kv` flattens **all heads and features** of each batch/token into
  one vector and uses one fp16 scale. K and V have independent scales. The
  stored INT8 layout remains `[batch, heads, sequence, features]`; scales are
  `[batch, 1, sequence, 1]`. The half-step bound uses the stored scale.
- `QuantKVCache` preserves L4 append/get/seq_len semantics, including transient
  unequal layer lengths. It quantizes only new tokens, appends integers and
  scales in lockstep, and never retains a floating-point cache. Reads reconstruct
  fp32 arrays for the existing attention path. Validation precedes mutation.
- `generate` accepts an optional cache instance; its existing default cache
  construction and decode statements remain unchanged. No existing line is
  removed from `layers/inference.py` or `scripts/train.py`.
- `BF16TrainingEngine` keeps the original GPT as the fp32 master, and runs an
  independent GPT with bf16 parameter copies, including embedding and norms.
  Existing GPT/attention/MLP operations run with bf16 activations/backprop;
  logits are promoted before fp32 cross-entropy. Gradients are promoted to
  fp32 before clipping and the unchanged AdamW; compute copies refresh after
  every update. Master weights never round-trip through bf16. No loss scaling:
  bf16 retains fp32's exponent range; reduced mantissa precision is measured
  against the fp32 baseline instead of concealed by scaling.
- The default fp32 training function, engine, constants, threshold, artifact
  paths, and arithmetic order remain verbatim. `--precision bf16` runs the
  explicitly permitted **v1 200-step recipe**, with a 28% loss-reduction gate.
  All model/training/data/tokenizer settings must match the archived v1 baseline.
  Every new artifact goes under `results/l5/`, leaving L3 results intact.

## T2 measurement definitions and early validation

- Baseline existing suite before changes: `81 passed, 1 warning in 7.98s`.
  New tests: `10 passed, 1 warning in 13.06s`; full initial output is retained
  in `results/l5/t2-tests-initial.txt`. Initial standalone 200-step training
  passed; output and timestamped raw record are retained under `results/l5/`.
- KV benchmark uses contexts 128/512/2048 and 32 actual cached decode calls,
  with prefill excluded. Prefill is chunked into calls of at most the unchanged
  model ctx=256, preserving RoPE cache offsets. Beyond 256 is positional
  extrapolation, useful for storage/speed measurement, not evidence of learned
  long-context quality. L0: 3 warmups, 10 repetitions, upper median, nearest-rank
  p90, raw samples and hashes. fp32/INT8 order alternates between repetitions.
- KV storage is actual resident array payload including scales, recorded at
  exactly n tokens and after decode. Temporary fp32 dequantization is excluded
  from persistent KV bytes; total active allocation peaks are recorded separately.
  INT8 vs **fp32** has a theoretical compression near 4x, not the principle
  note's approximately 2x, which corresponds to a half-precision reference.
- Compare per-step logits on the **same fp32-greedy history**. Raw max-abs errors
  can decrease: softmax and projections do not imply monotone propagation.
  Record the raw curve and its monotone running-maximum envelope, plus the first
  aligned argmax disagreement. Independent free greedy runs also record their
  first token divergence, without mixing changed histories into aligned errors.
- bf16 benchmark applies L0's 3 warmup / 10 measured **independent 200-step
  runs**. Inside each run it preserves L3's estimator: exclude the first three
  steps, use the remaining 197 step rates and L3 median/p90. The outer aggregate
  uses L0 upper median/p90 over run summaries. All warmup and measured raw
  records are retained, with hardware/config/source/baseline hashes.
- MFU uses the archived v1 baseline's exact parameter-only numerator
  78.7e6 FLOP/token and 14.75e12 peak; it excludes attention FLOPs and should not
  be confused with L3's later corrected 500-step MFU. Peak memory uses the same
  active-Metal-allocation counter and reset boundary as L3. Historical speed
  comparison is not a contemporaneous paired fp32 remeasurement.
- Implementation-stage full suite: `91 passed, 1 warning in 20.68s`; complete
  output in `results/l5/t2-tests-stage1.txt`. All 81 pre-existing tests are
  unchanged. `git diff --check` and syntax compilation passed. Ruff remains
  unavailable and there is no lint configuration; no dependency was installed.
  Source-line subsequence checks confirm additive-only inference/train changes;
  AST comparisons confirm every original training function/class except CLI
  `main` is identical to its pre-task version.
