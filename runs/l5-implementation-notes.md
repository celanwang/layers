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

The final benchmark, acceptance output, prediction verdicts and file manifest
will be archived under `results/l5/` after measuring the committed code.
