# L3-T2 Training Loop Report

## Changed Files

- `layers/optimizer.py` — manual AdamW, global gradient clipping, and cosine LR schedule.
- `scripts/train.py` — L1 tokenized corpus loading, strict train/heldout split, 200-step training, checkpoint, loss curve, and metrics.
- `tests/test_optimizer.py` — hand calculation, Torch AdamW Oracle, schedule/clipping boundaries, and fixed-batch overfit check.
- `bench/l3_train_bench.py` — full-config training-step benchmark.
- `runs/l3-implementation-notes.md` — appended L3-T2 implementation and result notes.
- `results/l3/ckpt-final.npz`, `loss-curve.json`, `train-metrics.json`, `train-bench.json` — final attempted run artifacts.
- `results/l3/*-attempt-{1,2,3,4}.json` — retained curves and metrics from the four recorded training attempts.

All edits and generated artifacts are within the task's §2 whitelist. No dependency was added, no T1 file was changed, the primary training path does not use `mlx.optimizers`, and no push was performed.

## Acceptance Commands and Complete Output

### 1. `/Users/wangchen/Developer/layers/.venv/bin/python -m pytest tests/ -q`

Exit status: `0`.

```text
................................................. [100%]
49 passed in 4.60s
```

### 2. `/Users/wangchen/Developer/layers/.venv/bin/python bench/l3_train_bench.py`

Exit status: `0`.

```text
mx.metal.reset_peak_memory is deprecated and will be removed in a future version. Use mx.reset_peak_memory instead.
mx.metal.get_peak_memory is deprecated and will be removed in a future version. Use mx.get_peak_memory instead.
config_hash: 7997ac760223
device: Apple M4 Max / mlx 0.32.4
protocol: warmup=3, measured=10, tokens/step=4096
throughput: median=62505.4, p90=62762.6 tokens/s
latency: median=0.0655, p90=0.0658 s
peak Metal memory: 2618.3 MiB
written: results/l3/train-bench.json
```

### 3. `/Users/wangchen/Developer/layers/.venv/bin/python scripts/train.py`

Exit status: `1`, due to the specified heldout-reduction assertion. The curve contains no NaN/Inf, but the measured reduction is below 30%.

```text
mx.metal.get_active_memory is deprecated and will be removed in a future version. Use mx.get_active_memory instead.
mx.metal.reset_peak_memory is deprecated and will be removed in a future version. Use mx.reset_peak_memory instead.
mx.metal.get_peak_memory is deprecated and will be removed in a future version. Use mx.get_peak_memory instead.
Traceback (most recent call last):
  File "/Users/wangchen/Developer/layers-wt-l3/scripts/train.py", line 407, in <module>
    main()
  File "/Users/wangchen/Developer/layers-wt-l3/scripts/train.py", line 403, in main
    run_training()
  File "/Users/wangchen/Developer/layers-wt-l3/scripts/train.py", line 371, in run_training
    raise AssertionError(
AssertionError: heldout loss reduction must exceed 30%: initial=7.0789, final=4.9758
```

## Infra Predictions

| Prediction | Measurement | Verdict |
|---|---:|---|
| Training memory about 4× parameters, around 210 MB, with activations under 1 GB | 13,118,464 parameters; 4×-state estimate 209,895,424 bytes; peak Metal allocation 2,745,532,676 bytes (2,618.34 MiB) | **Refuted** — measured peak exceeds the estimate plus the activation allowance. |
| MFU below 10% | Median 62,420.2 tokens/s × 78.7 MFLOP/token ÷ 14.75 TFLOP/s = 33.30% | **Refuted** under the task's specified formula and L0 reference peak. |
| Loss falls below random-init loss and continues downward | Heldout 7.0789 → 4.9758 nats; all recorded values finite and the heldout curve decreases at each 10-step checkpoint | **Supported for direction and finiteness**; the separate strict `>30%` acceptance gate is missed at 29.71%. |

## Model bpb and L1 Proxy Comparison

Heldout contains `44,468` tokens over `106,148` UTF-8 bytes (`0.418925` tokens/byte). Using `final loss (nats) / ln(2) × tokens/byte`, model bpb is **3.0073 bits/byte**. L1's uniform-ID proxy is **4.19** for layers and **4.69** for cl100k_base. The trained model score is lower because it captures learned next-token probabilities, while the L1 proxies assume uniform token-ID cost; the figures are useful context but are not directly equivalent measurements.

## Deviation and Preserved Negative Results

The specified architecture, fp32 dtype, batch 16, context 256, 200 updates, peak LR `3e-4`, 100-step warmup, `(0.9, 0.95)` betas, `1e-8` epsilon, and global clip norm `1.0` were retained. Since the task did not specify numeric weight decay, the final recorded run uses `0.1`. It also shuffles full-context windows within the training partition to avoid repeated random windows. The result is a clear loss decrease but misses the required `>30%` reduction by `0.29` percentage points. The trainer leaves this check enabled and exits with an assertion; the acceptance condition was not weakened. Attempt curves and metrics are retained as `*-attempt-{1,2,3,4}.json`.

| Attempt | Training-window sampling | Weight decay | Heldout reduction |
|---|---|---:|---:|
| 1 | Random windows with replacement; sampling seed `20261021` | `0.01` | `28.40%` |
| 2 | Random windows with replacement; sampling seed `42` | `0.01` | `28.56%` |
| 3 | Shuffled full-context windows; sampling seed `42` | `0.01` | `29.73%` |
| 4 (final artifacts) | Shuffled full-context windows; sampling seed `42` | `0.1` | `29.71%` |

The separate single-batch overfit check passed: `2.763561` → `0.000338` nats in 50 updates. The AdamW Torch Oracle matched for five updates within `1e-6`.
