# L2 Attention Implementation Report

## File inventory

- `layers/embedding.py` — seeded MLX token embedding lookup.
- `layers/attention.py` — interleaved RoPE, causal score masking, stable softmax, and multi-head attention.
- `layers/__init__.py` — exports the L2 public components.
- `tests/test_attention.py` — hand calculation, causality, RoPE, PyTorch and MLX public API oracles, and tokenizer-to-attention smoke test.
- `bench/l2_attention_bench.py` — L0-style attention, KV memory, and softmax timing measurements.
- `runs/l2-implementation-notes.md` — assumptions, implementation notes, and differences from the principle note.
- `results/l2/attention-bench-20261021T093939Z.json` — full benchmark record with per-sample summaries, p90, device, protocol, and configuration.
- `results/l2/l2-attention-report.md` — this report.

## Acceptance commands

### `.venv/bin/python -m pytest tests/ -q`

```text
......................................                                   [100%]
38 passed in 3.81s
```

### `.venv/bin/python bench/l2_attention_bench.py`

```text
device: Apple M4 Max, mlx 0.32.4
config_hash: 02a1f2c5f29b
attention kernel medians (s):
  n= 128: naive=0.000360958 fast=0.000188292 naive/fast=1.92x
  n= 256: naive=0.000487625 fast=0.000236417 naive/fast=2.06x
  n= 512: naive=0.000475541 fast=0.000210667 naive/fast=2.26x
  n=1024: naive=0.00108487 fast=0.00033775 naive/fast=3.21x
  n=2048: naive=0.00436221 fast=0.000681333 naive/fast=6.40x
  n=4096: naive=0.0169531 fast=0.00186446 naive/fast=9.09x
KV allocation (measured/theoretical bytes):
  fp32 n= 128: 524296/524288 (1.000x)
  fp32 n= 256: 1048584/1048576 (1.000x)
  fp32 n= 512: 2097160/2097152 (1.000x)
  fp32 n=1024: 4194312/4194304 (1.000x)
  fp32 n=2048: 8388616/8388608 (1.000x)
  fp32 n=4096: 16777224/16777216 (1.000x)
  fp32 n=8192: 33554440/33554432 (1.000x)
  bf16 n= 128: 262148/262144 (1.000x)
  bf16 n= 256: 524292/524288 (1.000x)
  bf16 n= 512: 1048580/1048576 (1.000x)
  bf16 n=1024: 2097156/2097152 (1.000x)
  bf16 n=2048: 4194308/4194304 (1.000x)
  bf16 n=4096: 8388612/8388608 (1.000x)
  bf16 n=8192: 16777220/16777216 (1.000x)
softmax share: 0.00066225s / 0.00169608s = 39.05% (n=1024)
written: results/l2/attention-bench-20261021T093939Z.json
```

Both commands passed. The benchmark JSON records warmup 3, timed repetitions 10, median and p90, environment, and config hash. Environment: Apple M4 Max, 48 GiB, macOS 15.8, Python 3.12.15, MLX 0.32.4, PyTorch 2.14.1, NumPy 2.5.4.

## Infra predictions

1. **Naive vs fast kernel: rejected as a whole; large-length trend supported.** Fast was already 1.92× faster at n=128 and 3.21× at n=1024, so the predicted limited small-length gap did not appear under the measured relative-time ratio. The gap then widened to 6.40× at n=2048 and 9.09× at n=4096. There was no crossover: fast was faster at every tested length. The principle note did not define a numerical threshold for “limited”; the verdict uses these measured ratios and reports that ambiguity.
2. **KV growth: confirmed.** From n=128 to 8192, fp32 active allocation rose from 524,296 to 33,554,440 bytes while theory predicts 524,288 to 33,554,432; bf16 rose from 262,148 to 16,777,220 versus 262,144 to 16,777,216. The constant +8-byte fp32 and +4-byte bf16 offsets do not change the linear slope.
3. **Softmax share: confirmed at n=1024.** Stable softmax median was 0.00066225 s, compared with full-layer median 0.001696083 s, or 39.05%. This is a coarse isolated-component timing, as specified; p90 values are in the JSON record.

## Deviations and execution notes

- No remaining implementation or scope deviation from specification v2. Only the approved `torch` and `numpy` packages were installed in `.venv`; project dependency files were not changed.
- The first benchmark invocation exposed a direct-script import-path issue (`layers` was not found). The benchmark was fixed to add the repository root to `sys.path`; the acceptance command was then rerun successfully.
- `git diff --check` completed without output. The main implementation files contain no `mx.fast.*` or `mlx.nn.RoPE` calls; those APIs are used only by the test oracle and benchmark comparison.
- No push was performed.

## `runs/l2-implementation-notes.md` highlights

- Embedding uses an MLX gather over the 1028-ID L1 vocabulary.
- Attention applies interleaved-pair RoPE to Q/K, then scale, upper-triangular `-inf` masking, max-subtracted softmax, `P @ V`, and output projection.
- Public module calls return `(out, attn_weights)`; `forward_with_intermediates` exposes raw and rotated Q/K, V, scores, probabilities, and output for oracle alignment.
- Projection bias and initializer were unspecified in the principle note; implementation uses no bias, seeded normal weights, and `1/sqrt(d_model)` scaling. The RoPE oracle uses `traditional=True` to match adjacent-pair rotation.
