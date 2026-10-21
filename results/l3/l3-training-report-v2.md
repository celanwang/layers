# L3-T2 Training Report — v2 (500 Steps)

## Change

- Updated `scripts/train.py` from 200 to 500 optimizer steps, keeping the `>30%` heldout-loss reduction gate unchanged.
- Made this run's outputs step-specific so the earlier 200-step results remain untouched: `ckpt-final-500.npz`, `loss-curve-500.json`, and `train-metrics-500.json`.

## Result

- Config hash: `2c846ded0b41`.
- Heldout loss: `7.0789208412` → `4.1752080917` nats.
- Relative reduction: **41.0191%**, passing the `>30%` assertion.
- Median / p90 throughput: `62,310.5` / `62,720.1` tokens/s.
- Peak Metal memory: `2,745,549,060` bytes (`2,618.36 MiB`); MFU using the task's L0 reference: `33.2463%`.
- Model bpb: `2.5234` bits/byte.
- The script's 50-step single-batch check also passed: `2.763561` → `0.000338` nats.

## Complete Command Output

Command:

```bash
/Users/wangchen/Developer/layers/.venv/bin/python scripts/train.py
```

Exit status: `0`.

```text
mx.metal.get_active_memory is deprecated and will be removed in a future version. Use mx.get_active_memory instead.
mx.metal.reset_peak_memory is deprecated and will be removed in a future version. Use mx.reset_peak_memory instead.
mx.metal.get_peak_memory is deprecated and will be removed in a future version. Use mx.get_peak_memory instead.
config_hash: 2c846ded0b41
device: Apple M4 Max / 51539607552 bytes
corpus: 444677 tokens; split=400209
model: 13,118,464 parameters; predicted 4x-state memory=200.2 MiB
heldout loss: 7.0789 -> 4.1752 nats (41.0% reduction)
throughput: median=62310.5, p90=62720.1 tokens/s; MFU=33.2463%
memory: peak=2618.4 MiB; model bpb=2.5234
single-batch overfit: 2.7636 -> 0.0003 nats in 50 steps
written: /Users/wangchen/Developer/layers-wt-l3/results/l3/ckpt-final-500.npz, /Users/wangchen/Developer/layers-wt-l3/results/l3/loss-curve-500.json, /Users/wangchen/Developer/layers-wt-l3/results/l3/train-metrics-500.json
```

Existing 200-step failure artifacts and reports were left unchanged. No push was performed.
