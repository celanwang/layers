# L5-T1 weight quantization execution report

Both section-5 acceptance commands passed. The 60 existing tests passed unchanged; 21 new cases bring the total to 81. Only the task whitelist was modified by the execution commits. No quantization reference source was read, no dependency was added, and no push was performed.

## Files changed

- `layers/quant.py` — Handwritten symmetric quant/dequant, byte packing, QuantizedLinear and in-place model conversion.
- `layers/__init__.py` — Additive exports only.
- `tests/test_quant.py` — 21 new test cases; all six existing test files are unchanged.
- `bench/l5_quant_bench.py` — L0-protocol storage, complete heldout, cached greedy decode and public API Oracle measurements.
- `runs/l5-implementation-notes.md` — Implementation decisions, assumptions, edge cases and measurement definitions.
- `results/l5/tests-stage1.txt` — Complete implementation-stage test output.
- `results/l5/tests-acceptance.txt` — Complete final acceptance test output.
- `results/l5/bench-acceptance.txt` — Complete acceptance benchmark output.
- `results/l5/quant-bench-20261021T121645745015Z.json` — Raw samples, configuration, hashes, device, storage accounting and Oracle comparisons.
- `results/l5/l5-t1-report.md` — This section-6 execution report.

`layers/model.py`, `layers/attention.py`, `layers/transformer.py`, `layers/inference.py`, the L3 checkpoint and all existing tests were left unchanged.

## Acceptance command output (complete)

```text
.venv/bin/python -m pytest tests/ -q
........................................................................ [ 88%]
.........                                                                [100%]
=============================== warnings summary ===============================
tests/test_quant.py::test_int8_pytorch_public_api_oracle
  /Users/wangchen/Developer/layers/tests/test_quant.py:58: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
    oracle = torch.quantize_per_channel(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
81 passed, 1 warning in 7.43s
```

```text
.venv/bin/python bench/l5_quant_bench.py
device: Apple M4 Max, mlx 0.32.4
config_hash: f782bb66df96 (commit d10f1f665a8a)
heldout: split=400209, 44467 target tokens
fp32: NPZ=52485384 B, loss=4.27175664, decode=394.87 tok/s (median=0.321625s, p90=0.325494s)
int8: NPZ=14780712 B, loss=4.27190090, decode=284.60 tok/s (median=0.446236s, p90=0.458092s)
int4-g64: NPZ=8845608 B, loss=4.29692684, decode=239.46 tok/s (median=0.530368s, p90=0.547352s)
fp32: model compression=1.0000x, projection compression=1.0000x, loss change=+0.0000%, decode speedup=1.0000x
int8: model compression=3.5509x, projection compression=3.9883x, loss change=+0.0034%, decode speedup=0.7208x
int4-g64: model compression=5.9335x, projection compression=7.5294x, loss change=+0.5892%, decode speedup=0.6064x
MLX public API Oracle: {"available": true, "native_mode": "affine (scale plus bias), unlike our zero-point-zero symmetric grid", "int8_oracle_group": "128 (API supports 32/64/128); production task uses full input width", "comparisons": [{"bits": 8, "group_size": 128, "our_grid_vs_mlx_dequantize_max_abs": 0.0, "our_vs_native_affine_max_abs": 0.002490326762199402, "our_roundtrip_rmse": 0.0006323691923171282, "native_affine_roundtrip_rmse": 0.000572813383769244}, {"bits": 4, "group_size": 64, "our_grid_vs_mlx_dequantize_max_abs": 0.0, "our_vs_native_affine_max_abs": 0.041611939668655396, "our_roundtrip_rmse": 0.01060362346470356, "native_affine_roundtrip_rmse": 0.008869619108736515}]}
written: results/l5/quant-bench-20261021T121645745015Z.json
```

Both exited with status 0. The one warning is the installed PyTorch version deprecating its public quantized-tensor creation API; that API still passes the INT8 Oracle. Syntax compilation and `git diff --check` passed. Ruff is not installed and there is no repository lint configuration; no lint dependency was introduced.

## Three-axis measurements

Device: Apple M4 Max, 48 GiB unified memory, macOS-15.8-arm64-arm-64bit, Python 3.12.15, MLX 0.32.4. Code commit `d10f1f665a8a58a28abcce2ea723db40bc8a2843`, configuration hash `f782bb66df96`. Timing: 3 warmups / 10 measured runs; L0 upper median and nearest-rank p90; individual samples retained in JSON.

Quality uses the unchanged `results/l3/ckpt-final-500.npz`, L1 tokenizer and the final 10% of token IDs (`split=400209`), evaluating all 44,467 heldout next-token targets. Every format evaluates identical context windows and targets; the final short window is included.

| Format | Serialized NPZ bytes | Whole-model compression | Projection compression | Heldout loss (nats) | Loss increase | Decode tok/s | Decode speedup | Median / p90 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 52,485,384 | 1.0000x | 1.0000x | 4.27175664 | +0.0000% | 394.87 | 1.0000x | 0.321625 / 0.325494 |
| int8 | 14,780,712 | 3.5509x | 3.9883x | 4.27190090 | +0.0034% | 284.60 | 0.7208x | 0.446236 / 0.458092 |
| int4-g64 | 8,845,608 | 5.9335x | 7.5294x | 4.29692684 | +0.5892% | 239.46 | 0.6064x | 0.530368 / 0.547352 |

| Format | Integer bytes | fp16 scale bytes | Protected fp32 bytes | NPZ overhead bytes |
|---|---:|---:|---:|---:|
| fp32 | 0 | 0 | 2,142,208 | 11,528 |
| int8 | 12,582,912 | 36,864 | 2,142,208 | 18,728 |
| int4-g64 | 6,291,456 | 393,216 | 2,142,208 | 18,728 |

The fp32 projection payload is 50,331,648 bytes. INT8 projection payload is 12,619,776 bytes and INT4 is 6,684,672 bytes, including their scales. Storage is actual uncompressed NPZ serialization to an in-memory buffer, including headers; SHA-256 is recorded. These are serialized storage measurements, not peak inference memory or fused integer-kernel performance.

All runs call unchanged L4 `generate` with a 128-token heldout prompt, 128 generated tokens and greedy sampling, using its `KVCache`. The first generated token is sampled from prefill; the decode interval therefore times 127 actual single-token calls and divides by 127. Full generation end-to-end rates (all 128 output tokens, including prefill) are respectively fp32 394.20 tok/s, int8 282.21 tok/s, int4-g64 238.81 tok/s. Sampling and streaming costs remain in the decode interval. No existing measurement protocol was edited.

## Principle-note section-4 prediction verdicts

1. **Memory — overturned as written for INT8; INT4 holds.** Projection-only INT8 compression is 3.9883x, rather than the stated approximately 4.1x. fp32-to-INT8 cannot exceed 4x before scale overhead, so that numeric premise needs correction. INT4 g64 is 7.5294x, consistent with approximately 7.5x. With embedding/tied head and norms protected, whole-model serialized compression is lower: 3.5509x and 5.9335x. These two scopes must not be conflated.
2. **Quality — holds.** INT8 +0.0034% is below 1%; INT4 +0.5892% is below 5%, on the same complete heldout partition.
3. **Speed — numeric interval overturned; qualitative slowdown holds.** INT8 runs at 0.7208x fp32, outside the predicted [0.8x, 1.3x] interval; INT4 runs at 0.6064x. This educational dequantize-before-fp32-matmul path adds work and does not improve cached decode. Extra unpack/dequantize work is a plausible explanation, but this benchmark does not profile kernels and cannot independently quantify dispatch versus memory costs or generalize to fused production quantization kernels.

## Oracle and implementation evidence

- Hand-verified INT4 signed-nibble/group-boundary and INT8 per-channel/ties-to-even cases; per-group half-step error bounds against independent NumPy reconstruction, including zero groups, outliers and subnormal scales.
- PyTorch public per-channel quantization agrees with INT8 integer values and dequantization. MLX public dequantization agrees exactly (maximum absolute difference 0) for our adapted signed grid in both bit widths.
- Native MLX affine roundtrip differs from our symmetric roundtrip by maximum absolute 0.00249033 (INT8) and 0.04161194 (INT4). Native affine uses a bias and a different grid. Our/native RMSE is 0.000632369/0.000572813 for INT8 and 0.0106036/0.00886962 for INT4. Oracle INT8 uses group 128, which the public API supports, while the task model uses full-input-width per-channel scales.
- Scope tests prove original embedding and norm array identity, fp32 dtype and bitwise values; all six projections per block store q/scales only. Non-square linear shapes, finite logits, INT8 logit error below INT4, and cached-versus-full-prefix quantized logits pass.
- Quantized subclasses inherit existing attention/MLP forward methods. Properties dequantize afresh to fp32 at each projection access. The original fp32 GPT implementation and forward behavior are unchanged.

## Deviations and execution history

- No outstanding whitelist, dependency, existing-test, reference-source or push deviation. The model-conversion entry is exported from `layers/__init__.py`; editing `layers/model.py` was unnecessary.
- The complete heldout estimator extends coverage beyond L3's sampled evaluation batch, while keeping the identical final-10% partition. The 127-step decode numerator follows the unchanged generator's prefill semantics; full 128-token end-to-end timing is also included. Storage serialization is real NPZ serialization in memory, without saving duplicate binary checkpoints.
- An initial local interface probe failed because MLX does not dispatch a custom object's `__rmatmul__`; the final property approach resolves this without modifying non-whitelisted forward code. No acceptance command failed. The implementation notes retain that exploration and the fp16 rounding assumption.
- An independent documentation commit `c6a354f` adding `tasks/l5-kv-bf16.md` appeared on the shared branch during execution. This task did not author, alter or stage that file. Its own implementation and acceptance commits touch only whitelisted paths.

## Stage commits

- `d10f1f6` — `feat(l5): add handwritten weight quantization`
- `71bf8ee` — `test(l5): record quantization acceptance`
- The final report/notes are committed separately as `docs(l5): archive weight quantization report`.

All commits remain local.
