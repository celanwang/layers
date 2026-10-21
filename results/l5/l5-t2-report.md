# L5-T2 execution report

All three section-5 commands passed with exit status 0. The full suite has 91 passing tests; all 81 pre-existing tests remain byte-for-byte unchanged. Only whitelisted paths were changed. No quantization or mixed precision reference implementation source was read, no dependency was added, and no push was performed.

This is the section-6 output report, saved as `results/l5/l5-t2-report.md` because the request did not supply a separate `-o` pathname.

## Files changed

- `bench/l5_kv_bf16_bench.py`
- `layers/inference.py`
- `layers/quant.py`
- `results/l5/bf16-train-20261021T122519142791Z.json`
- `results/l5/bf16-train-20261021T122743465258Z.json`
- `results/l5/bf16-train-20261021T122923139217Z.json`
- `results/l5/bf16-train-20261021T122954995481Z.json`
- `results/l5/bf16-train-20261021T123036983632Z.json`
- `results/l5/bf16-train-20261021T123049114189Z.json`
- `results/l5/bf16-train-20261021T123101270154Z.json`
- `results/l5/bf16-train-20261021T123113854875Z.json`
- `results/l5/bf16-train-20261021T123125976950Z.json`
- `results/l5/bf16-train-20261021T123138103774Z.json`
- `results/l5/bf16-train-20261021T123150227177Z.json`
- `results/l5/bf16-train-20261021T123202690264Z.json`
- `results/l5/bf16-train-20261021T123215330097Z.json`
- `results/l5/bf16-train-20261021T123228243830Z.json`
- `results/l5/bf16-train-20261021T123241483376Z.json`
- `results/l5/bf16-train-20261021T123254460891Z.json`
- `results/l5/bf16-train-20261021T123306949467Z.json`
- `results/l5/bf16-train-20261021T123344743861Z.json`
- `results/l5/kv-bf16-bench-20261021T123306957293Z.json`
- `results/l5/l5-t2-report.md`
- `results/l5/t2-acceptance-status.json`
- `results/l5/t2-bench-acceptance.txt`
- `results/l5/t2-kv-test-logits.json`
- `results/l5/t2-scope-checks.txt`
- `results/l5/t2-tests-acceptance.txt`
- `results/l5/t2-tests-initial.txt`
- `results/l5/t2-tests-stage1.txt`
- `results/l5/t2-train-acceptance.txt`
- `results/l5/t2-train-initial.txt`
- `runs/l5-implementation-notes.md`
- `scripts/train.py`
- `tests/test_bf16_train.py`
- `tests/test_kv_quant.py`

`layers/quant.py` adds per-token KV helpers reusing T1's symmetric grid. `layers/inference.py` adds `QuantKVCache` and optional cache injection into `generate`. `scripts/train.py` adds a separate bf16 engine/200-step recipe and the precision CLI argument. The two new test files cover storage, append semantics, public API Oracle, aligned logit curves, compute/master/Adam/loss dtypes, gradient comparison and real training. The benchmark and implementation notes record the protocol and evidence. Results include every warmup/measured training record, full command output, scope checks and this report.

The existing lines of `layers/inference.py` and `scripts/train.py` are retained in their original order. Every original training function/class except CLI `main` is AST-identical to the pre-task version. Existing model/optimizer files, dependencies, tests, L3 checkpoints and L3 results are unchanged. See `results/l5/t2-scope-checks.txt` for hashes and checks.

## Complete acceptance command output

```text
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 79%]
...................                                                      [100%]
=============================== warnings summary ===============================
tests/test_kv_quant.py::test_kv_roundtrip_bound_and_pytorch_public_api_oracle
  /Users/wangchen/Developer/layers/tests/test_kv_quant.py:38: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
    oracle = torch.quantize_per_channel(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
91 passed, 1 warning in 21.28s
exit status: 0
```

```text
$ .venv/bin/python bench/l5_kv_bf16_bench.py
device: Apple M4 Max; config_hash: b7ce7d1d3a50
KV n=128: fp32=2097152 B, INT8=526336 B (3.984436x); decode fp32=461.49, INT8=141.74 tok/s (0.307127x); max logit diff=0.03315496; greedy fork=None
KV n=512: fp32=8388608 B, INT8=2105344 B (3.984436x); decode fp32=477.67, INT8=141.86 tok/s (0.296977x); max logit diff=0.04887322; greedy fork=None
KV n=2048: fp32=33554432 B, INT8=8421376 B (3.984436x); decode fp32=465.85, INT8=136.14 tok/s (0.292234x); max logit diff=0.03614235; greedy fork=None
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97565460 (29.7124% reduction)
throughput: median=81573.38 tok/s; p90=82347.94; MFU=43.5242%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.3070021587057246, "mfu_relative_change": 0.30700215870572456, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": -2.1753857308737246e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123036983632Z.json
bf16 repetition 1/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97574663 (29.7111% reduction)
throughput: median=81527.31 tok/s; p90=82162.17; MFU=43.4997%
memory: peak=1516241032 bytes (1446.00 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.3062639905436746, "mfu_relative_change": 0.3062639905436746, "peak_memory_relative_reduction": 0.4477457882322452, "final_loss_relative_difference": -3.25828699776487e-06, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123049114189Z.json
bf16 repetition 2/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97601271 (29.7073% reduction)
throughput: median=81583.47 tok/s; p90=82826.54; MFU=43.5296%
memory: peak=1516241032 bytes (1446.00 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.3071637694450402, "mfu_relative_change": 0.3071637694450402, "peak_memory_relative_reduction": 0.4477457882322452, "final_loss_relative_difference": 5.021595255394473e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123101270154Z.json
bf16 repetition 3/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97584248 (29.7097% reduction)
throughput: median=80100.97 tok/s; p90=80976.02; MFU=42.7386%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.283410642385744, "mfu_relative_change": 0.28341064238574387, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": 1.6003939077302576e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123113854875Z.json
bf16 repetition 4/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97578907 (29.7105% reduction)
throughput: median=81682.30 tok/s; p90=82202.02; MFU=43.5824%
memory: peak=1516241032 bytes (1446.00 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.3087473652708137, "mfu_relative_change": 0.3087473652708137, "peak_memory_relative_reduction": 0.4477457882322452, "final_loss_relative_difference": 5.270758378861373e-06, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123125976950Z.json
bf16 repetition 5/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97549915 (29.7146% reduction)
throughput: median=81590.98 tok/s; p90=82007.54; MFU=43.5336%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.3072841966309163, "mfu_relative_change": 0.30728419663091633, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": -5.2995079699136305e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123138103774Z.json
bf16 repetition 6/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97607422 (29.7065% reduction)
throughput: median=81668.33 tok/s; p90=82176.46; MFU=43.5749%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.3085233958077775, "mfu_relative_change": 0.30852339580777777, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": 6.257827675137939e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123150227177Z.json
bf16 repetition 7/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97571564 (29.7115% reduction)
throughput: median=78835.68 tok/s; p90=81420.42; MFU=42.0635%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.2631376560378667, "mfu_relative_change": 0.26313765603786665, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": -9.487365081772836e-06, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123202690264Z.json
bf16 repetition 8/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97601318 (29.7073% reduction)
throughput: median=76211.09 tok/s; p90=82072.93; MFU=40.6631%
memory: peak=1516241032 bytes (1446.00 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.2210853427937687, "mfu_relative_change": 0.22108534279376868, "peak_memory_relative_reduction": 0.4477457882322452, "final_loss_relative_difference": 5.0311784524526004e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123215330097Z.json
bf16 repetition 9/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97563457 (29.7127% reduction)
throughput: median=73624.10 tok/s; p90=81673.41; MFU=39.2828%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.1796354845409316, "mfu_relative_change": 0.1796354845409316, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": -2.5778800070597185e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123228243830Z.json
bf16 repetition 10/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97560978 (29.7130% reduction)
throughput: median=71450.06 tok/s; p90=81317.10; MFU=38.1228%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.1448022258655024, "mfu_relative_change": 0.14480222586550218, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": -3.076206253782576e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123241483376Z.json
bf16 repetition 11/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97575760 (29.7109% reduction)
throughput: median=75727.39 tok/s; p90=81006.98; MFU=40.4051%
memory: peak=1516241032 bytes (1446.00 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.2133353238628835, "mfu_relative_change": 0.21333532386288345, "peak_memory_relative_reduction": 0.4477457882322452, "final_loss_relative_difference": -1.0541516757278657e-06, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123254460891Z.json
bf16 repetition 12/13 complete
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97594929 (29.7082% reduction)
throughput: median=75653.84 tok/s; p90=84750.89; MFU=40.3658%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.212156911668894, "mfu_relative_change": 0.21215691166889417, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": 3.7470300474407026e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123306949467Z.json
bf16 repetition 13/13 complete
bf16 L0 aggregate: {"tokens_per_second_median": 78835.68470207993, "tokens_per_second_p90": 81668.32598497435, "peak_metal_memory_bytes_median": 1516224648, "peak_metal_memory_bytes_p90": 1516241032, "final_heldout_loss_median": 4.9757890701293945, "final_heldout_loss_p90": 4.97601318359375, "mfu_percent": 42.063514481719935}
bf16 aggregate vs L3: {"tokens_per_second_ratio": 1.2631376560378667, "mfu_relative_change": 0.26313765603786665, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": 5.270758378861373e-06}
bf16 predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/kv-bf16-bench-20261021T123306957293Z.json
exit status: 0
```

```text
$ /Users/wangchen/Developer/layers/.venv/bin/python scripts/train.py --precision bf16
config_hash: b3b52a21c348
device: Apple M4 Max / 51539607552 bytes
bf16: 200 steps; master fp32=True; Adam fp32=True; loss fp32
heldout loss: 7.07899284 -> 4.97540474 (29.7159% reduction)
throughput: median=78505.97 tok/s; p90=82213.78; MFU=41.8876%
memory: peak=1516224648 bytes (1445.98 MiB)
vs L3 fp32: {"baseline": "results/l3/train-metrics.json", "tokens_per_second_ratio": 1.2578548569590553, "mfu_relative_change": 0.2578548569590555, "peak_memory_relative_reduction": 0.4477517557089291, "final_loss_relative_difference": -7.196980986268198e-05, "mfu_definition": "same L3 parameter-only 78.7e6 FLOP/token / 14.75e12 peak; excludes attention FLOPs", "timing_protocol": "unchanged L3: first 3 training steps excluded; median/p90 of 197 steps"}
predictions: {"mfu_increase_le_10_percent": false, "memory_reduction_35_to_40_percent": false, "final_loss_difference_lt_2_percent": true}
written: results/l5/bf16-train-20261021T123344743861Z.json
exit status: 0
```

Syntax compilation and `git diff --check` passed. There is no repository lint configuration and Ruff is not installed; no dependency was installed for lint. The single warning is the installed PyTorch release deprecating its public quantized-tensor creation API. The Oracle still passes.

## Measurement provenance and definitions

Apple M4 Max, 48 GiB unified memory, macOS 15.8, Python 3.12.15, MLX 0.32.4. Implementation commit `fa4f4542acf5bd97eaa21f55895d9922857efe9f`, combined config hash `b7ce7d1d3a50`. Raw record: `results/l5/kv-bf16-bench-20261021T123306957293Z.json`. Each record contains hardware, package versions, source/checkpoint/corpus/baseline hashes and raw samples.

KV: each context uses 3 warmups plus 10 measured repetitions, alternating format order; L0 upper median and nearest-rank p90. Timed work is 32 single-token cached forwards plus greedy argmax, after prefill. Persistent KV memory is the actual integer/floating array payload **including scales**, recorded at exactly n prompt tokens; it excludes temporary fp32 reconstructions. Total active allocation peaks and post-decode payloads are retained separately in JSON. This storage ratio is not a total inference-memory reduction claim. Existing ctx=256 is preserved: longer prompts are prefetched in chunks <=256, with the normal RoPE cache offset. n=512/2048 are positional extrapolation tests, without a claim of trained long-context quality.

bf16: L0 3 warmup plus 10 measured independent **200-step runs**, each from seed 42. Within each run the unchanged L3 recipe/estimator excludes its first three steps and computes rates over 197 steps. The outer table reports L0 median/p90 of those run summaries. Every model/training/corpus/tokenizer setting is checked against `results/l3/train-metrics.json`. Loss uses the same fixed seed-43 heldout batch. Historical fp32 data is not a contemporaneous paired remeasurement.

MFU uses the archived v1 formula (78.7e6 parameter FLOP/token / 14.75e12 peak), excluding attention FLOPs. This is the exact baseline convention requested by the task, rather than the later corrected 500-step L3 MFU. Memory uses the same active-Metal-allocation peak/reset boundary as L3, rather than OS RSS or allocator reserved bytes.

## KV memory and speed

| n | fp32 KV bytes | INT8 KV bytes incl scales | Compression | fp32 decode tok/s | INT8 decode tok/s | Speedup | fp32 median / p90 seconds | INT8 median / p90 seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2,097,152 | 526,336 | 3.984436x | 461.49 | 141.74 | 0.307127x | 0.069341 / 0.071907 | 0.225772 / 0.228694 |
| 512 | 8,388,608 | 2,105,344 | 3.984436x | 477.67 | 141.86 | 0.296977x | 0.066992 / 0.068294 | 0.225579 / 0.226866 |
| 2048 | 33,554,432 | 8,421,376 | 3.984436x | 465.85 | 136.14 | 0.292234x | 0.068692 / 0.069535 | 0.235058 / 0.238015 |

One K scale and one V scale cover each token's full d_model=512 vector across all heads, independently per layer/batch item. Each token costs 16,384 bytes in the four-layer fp32 cache, versus 4,096 integer bytes + 16 fp16-scale bytes in INT8: 3.984436x. The principle note's approximately 2x prediction corresponds to a hypothetical half-precision reference (1.992218x); the specified measured reference is fp32. This is a corrected prediction comparison, without changing the algorithm or measurement protocol.

The neutral-speed prediction is overturned. This educational implementation reconstructs fp32 K/V for attention and uses the existing T1 quantizer's explicit validation/evaluation synchronization. These add work; there is no fused integer attention kernel. The benchmark does not profile individual kernel or dispatch costs, so it cannot assign causal percentages to those operations.

## KV correctness and divergence

| n | Observed max absolute aligned logit difference | First aligned greedy disagreement | First independent greedy divergence |
|---|---:|---|---|
| 128 | 0.03315496 | None | None |
| 512 | 0.04887322 | None | None |
| 2048 | 0.03614235 | None | None |

`None` means no divergence in the observed horizon: the prefill prediction followed by 32 cached decode predictions. Aligned comparisons feed identical fp32-greedy tokens into both caches at every step. Every raw max-abs-diff point is retained, together with a running-maximum envelope. Raw errors are finite and small in these observations, but are not mathematically required to grow monotonically: softmax/projection/attention mixtures can attenuate as well as amplify perturbations. The envelope is monotone by definition and is explicitly labeled. This does not establish an unmeasured general long-context error bound or a heldout perplexity claim.

Hand-computed ties-to-even/per-token examples and NumPy roundtrip half-step bounds pass. The independent public PyTorch per-channel API agrees exactly with the integers and reconstructed values after the per-token layout adapter. Append/get/seq_len, shape validation, immutable older tokens, actual INT8-only storage, independent K/V scales, batched/multihead/multilayer appends and generator injection pass. Original default generator behavior agrees with an injected fp32 cache.

## bf16 throughput, MFU, memory and loss

| Quantity | Archived L3 fp32 (200 steps) | bf16 median of 10 runs | bf16 p90 | Relative change |
|---|---:|---:|---:|---:|
| Training tokens/s | 62412.58 | 78835.68 | 81668.33 | +26.3138% |
| MFU (v1 convention) | 33.3008% | 42.0635% | — | +26.3138% |
| Peak active Metal bytes | 2,745,549,060 | 1,516,224,648 | 1,516,241,032 | -44.7752% |
| Final heldout loss (nats) | 4.97576284 | 4.97578907 | 4.97601318 | +0.000527% |

Peak memory is 2618.36 MiB fp32 versus 1445.98 MiB bf16. MFU rises by 8.7627 percentage points (26.3138% relative). Same-seed measured final losses range from 4.97549915 to 4.97607422; raw curves and every per-step training loss are retained. All 13 benchmark runs have finite losses, at least 28% heldout-loss reduction, fp32 master weights and fp32 Adam states.

The separate required CLI acceptance run has initial/final heldout loss 7.07899284 / 4.97540474, reduction 29.7159%, throughput 78505.97 tok/s, MFU 41.8876%, and peak 1,516,224,648 bytes. Its full output is above and its raw record is `results/l5/bf16-train-20261021T123344743861Z.json`.

The compute model has bf16 weights/activations and bf16 parameter gradients. Logits are cast to fp32 before cross-entropy, whose exponentials and loss reduction remain fp32. Gradients are cast to fp32 before clipping and the unchanged AdamW update. A separate fp32 master retains small updates; it is never overwritten from rounded compute weights. bf16 has fp32's exponent range, so the teaching implementation uses no loss scaling. Tests verify gradient/loss proximity to the existing fp32 path and directly assert every required dtype.

## Principle section-6 prediction verdicts

1. **MFU increase <=10%: overturned.** Measured relative increase is 26.3138%, using the identical archived-v1 MFU convention. This is a historical-baseline comparison, and does not alone prove the cause of the speedup or generalize to other batch/model sizes.
2. **Peak memory reduction 35–40%: overturned, with greater savings.** Measured reduction is 44.7752%. fp32 master/Adam storage is preserved; the compute path's activations and gradients are reduced precision. No tensor-category attribution is inferred without an ablation.
3. **Final loss difference <2%: holds.** Aggregate difference is +0.000527%. Every 200-step run is finite and exceeds the unchanged-for-bf16 28% acceptance threshold. The <2% comparison is recorded as a prediction verdict, deliberately not a failing assertion.

## Deviations, constraints and remaining limits

- No whitelist, existing-test, dependency, reference-source, deletion or push deviation. Source edits are additive; original fp32 training behavior remains unchanged. Complete scope verification is retained in `results/l5/t2-scope-checks.txt`.
- The task explicitly allows the v1 200-step recipe for bf16; the existing fp32 default remains the untouched L3 500-step function. The bf16 path writes only under `results/l5/`, so no L3 artifact is overwritten.
- The approximately 2x KV prediction has a reference-dtype mismatch; the actual specified fp32 comparison and byte accounting are reported without changing the principle note. n>256 uses existing cached RoPE extrapolation with chunked prefill. Finite observed logits and no observed greedy fork do not establish general long-context model quality.
- For strict L0 compliance, the benchmark performs 3+10 full training repetitions while retaining L3's inner 197-step timing estimator for baseline comparability. No existing protocol file was edited. Historical baseline and v1 parameter-only MFU limits are explicit above.
- No acceptance command failed. Initial implementation training/tests and every warmup run are retained, including the negative speed result and overturned predictions. No reference comparison reading or retrospective beyond this task was started.

## Stage commits

- `fa4f454` — `feat(l5): add INT8 KV cache and bf16 training` (implementation, new tests, first full-suite validation, initial evidence).
- Acceptance outputs, all final measurements, scope checks and this report are archived in the separate local commit `test(l5): record KV and bf16 acceptance`.

All commits remain local; no push was performed.
