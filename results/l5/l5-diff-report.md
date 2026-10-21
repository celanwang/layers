# L5 comparison reading: execution report

Section-6 output (`-o` artifact): `results/l5/l5-diff-report.md`. No separate output filename was supplied, so this report uses the task's allowed results directory, following the existing T2 report convention. Main analysis: [runs/l5-diff-analysis.md](../../runs/l5-diff-analysis.md).

## Five conclusions

1. **bf16:** the measured +26.31–34.21% training throughput and −44.77% peak memory are consistent with reduced activation/intermediate traffic, while L0's ~1.1× matmul ratio is explained by MLX's shared float MMA fragments/accumulation on this M4/OS path (analysis §1; E0/E3/K1/K2/A1, M1–M5).
2. **INT8 KV:** 3.9844× persistent compression coexists with 0.292–0.337× decode because the teaching interface reconstructs full fp32 history and runs costly new-token quantization/check/evaluation stages; full dequant alone does not explain the measured penalty (analysis §2; K1/K2/A1).
3. **Weight PTQ:** our separate reconstruction/matmul adds intermediate traffic, while MLX/MLX-LM packed qmm and MoE gather-qmm consume compressed weights inside compute; existing data establishes no repeatable INT8 speedup and shows INT4 slowdown in both batches (analysis §3; W1/W2, M7–M10).
4. **Methods/quality:** RTN uses only weight-group ranges, GPTQ compensates errors with inverse-Hessian information, and AWQ protects activation-important channels; our +0.5892% cross-entropy is +2.5490% PPL, numerically a modest few-percent change but not a matched comparison to paper g128 results (analysis §4; W1/W2, G1/A2).
5. **Follow-ups:** propose KV INT4 and layer sensitivity as precision专项, QAT as a separate training iteration, cached/uncached paired measurements at context n≥256 as the L4 carry-over专项, and compressed-cache consumption/kernel batching as L6 integration work (analysis §5; local source gaps and L4 §1/§3).

## Key quantitative evidence

- **L0 open question 1 is explicitly answered.** E0 fp32/fp16/bf16 4096³ = 13.14/14.60/14.75 TFLOP/s; arithmetic intensity 683/1365 FLOP/B for fp32/bf16 is above the measured 32.7 FLOP/B balance point. MLX v0.32.4 converts storage inputs into float MMA fragments; the NAX gate is closed on macOS 15.8 / `applegpu_g16s`. Narrow loads therefore do not imply a 2× GEMM multiplier. This identifies the implementation mechanism, not exact hardware cycle shares.
- **bf16 memory:** 2,618.36 → 1,446.00 MiB; reduction 1,172.36 MiB. Four fp32 parameter states total only 200.17 MiB, and the extra bf16 compute-weight copy is 25.02 MiB. Representative activations fall 800 → 400 MiB; the illustrative remainder-halving model predicts 1,434.29 MiB, 11.71 MiB below measured peak. It is not a liveness measurement.
- **bf16 timing:** A1 exact GELU = 0.7228/0.7320 ms fp32 vs 0.3949/0.3984 ms bf16 median/p90 (**1.830×**); rectangular matmul = 0.7870/0.7998 vs 0.7084/0.7118 ms (**1.111×**). K1/K2 training outer medians are 78,836/83,761 tok/s versus historical fp32 62,413 tok/s; this baseline is not paired.
- **KV bytes per current context token:** fp32 = 16,384 B; INT8 including scales = 4,112 B; dequant minimum read/write = 20,496 B; including attention's reconstructed-KV read = 36,880 B. At n=2048 these are 32, 8.031, 40.031 and 72.031 MiB. Separate q-cast intermediates can add another 64 MiB of logical read/write.
- **KV operations:** 24 dequant primitive slots (8 arrays × two casts + multiply), 16 concatenate slots, 16 host scalar reads and 12 explicit eval calls per model token; these are source estimates, not profiler kernel/stall counts.
- **KV timing:** K1/K2 add approximately 4.60–5.20 ms/token. A1 dequant-only medians are 0.194/0.210/0.320 ms at n₀=128/512/2048, while quantize-new-only is 4.542/3.769/3.619 ms. Supplying bit-identical prequantized new data removes 80–84% of isolated append time; this includes multiple costs and is not an end-to-end attribution percentage.
- **Quality correction:** loss 4.27175664 → 4.29692684; PPL 71.6474 → 73.4736. AWQ Table 4's Llama-2-7B INT4 g128 PPL is 5.47 FP16, 5.73 RTN, 5.69 GPTQ, 5.63 GPTQ-Reorder, 5.60 AWQ. Different corpus/model/group/protected tensors preclude a method ranking against our g64 result.

Hardware, warmups/repetitions, quantile definitions, all raw sample locations, configuration hashes and source locations appear in the analysis evidence table and the two JSON artifacts. [diff-accounting.json](diff-accounting.json) preserves the arithmetic with input SHA-256 checksums.

## Actual files changed

All six files are newly added, within the task whitelist:

- `bench/l5_precision_ablation.py` — isolated, reproducible cache-stage/GELU/rectangular-matmul and dtype probes.
- `results/l5/precision-ablation-20261021T124703434945Z.json` — complete 3+10 samples, device/OS/version, source hashes and config hash `8ce12717067e`.
- `runs/l5-diff-analysis.md` — all five required analyses, quantitative budgets, L0 answer, precise references and limitations.
- `results/l5/diff-accounting.json` — derived memory/traffic/timing/PPL arithmetic and input checksums.
- `results/l5/diff-acceptance.txt` — verification command outputs.
- `results/l5/l5-diff-report.md` — this section-6 output report.

No `layers/`, `scripts/`, `tests/`, existing benchmark, principle note, dependency file, or existing measurement was changed. All negative results and prior measurements remain intact.

## Self-check output

`.venv/bin/python -B bench/l5_precision_ablation.py` exited 0:

```text
written: results/l5/precision-ablation-20261021T124703434945Z.json
config_hash: 8ce12717067e
n=128  original/prequantized/quantize-new/dequant/fp32-concat median ms: 5.7303/0.8965/4.5424/0.1936/0.1659
n=512  original/prequantized/quantize-new/dequant/fp32-concat median ms: 4.4622/0.7380/3.7688/0.2099/0.1689
n=2048 original/prequantized/quantize-new/dequant/fp32-concat median ms: 4.4957/0.9021/3.6194/0.3200/0.2678
prequantized/original max absolute output difference: 0 at every context
```

Syntax, JSON/sample/hash integrity, numeric accounting, local Markdown links, five-section/L0 coverage, staged whitespace, and full task diff whitelist checks passed; exact output is in [diff-acceptance.txt](diff-acceptance.txt). Per specification §5, the test suite was **not rerun**. There is no configured repository linter and no new lint dependency was added.

## External reading locations

Precise line anchors/versions are listed under “External reading locations” in the main analysis. The inspected locations are:

- MLX v0.32.4: [`transforms.h` 56–59](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/steel/gemm/transforms.h#L56-L59), [`gemm.h` 19–69](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/steel/gemm/gemm.h#L19-L69), [`mma.h` 451–531](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/steel/gemm/mma.h#L451-L531), [`device.cpp` 978–995](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp#L978-L995), [`matmul.cpp` 964–974](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp#L964-L974), [`unary_ops.h` 163–180](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/unary_ops.h#L163-L180).
- MLX quantization: [`mx.quantize` docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html), [`QuantizedLinear` 250–285](https://github.com/ml-explore/mlx/blob/v0.32.4/python/mlx/nn/layers/quantized.py#L250-L285), [`quantized.h` 850–963](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/quantized.h#L850-L963), [`quantized.cpp` 465–543](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L465-L543), [`gather_qmm` docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html), [compilation/fusion guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
- MLX-LM commit `72be8c955d9a0d3099bd01865af288e0e947c3fa`: [`utils.py` 496–519](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/utils.py#L496-L519), [`switch_layers.py` 93–106](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/models/switch_layers.py#L93-L106).
- [GPTQ v2](https://arxiv.org/pdf/2210.17323v2): §3–§4, Algorithm 1; [AWQ v5](https://arxiv.org/html/2306.00978v5): §3.1–§3.2, §5.1, Table 4.
- [vLLM v0.10.2 attention kernel](https://github.com/vllm-project/vllm/blob/v0.10.2/csrc/attention/attention_kernels.cuh#L279-L288): 279–288 and 409–414; [current FP8 KV documentation](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/): overview, FA3 note, scale schemes.
- [PyTorch 2.14 AMP docs](https://docs.pytorch.org/docs/2.14/amp.html#autocasting): autocast/backward dtype discussion and CPU bf16 example.

## Deviations and limitations

- No scope, dependency, source-vendoring, measurement-protocol or push deviation. The optional ablation is explicitly allowed by task §§2/5 and uses only installed dependencies.
- Output-path convention: since no literal `-o` pathname was provided, the section-6 report is saved at the whitelisted path stated above; no permission question was necessary.
- The GitHub connector was not logged in; public source was read through web/raw HTTP. The AWQ PDF exceeded the web fetch size limit; the versioned HTML tables supplied the numerical evidence. No reference source body was saved in the repository.
- Kernel counts are static estimates, memory/traffic budgets are calculations, and causal timing shares are not presented as measured. A1's unpaired component medians cannot be summed into full decode latency. No exact Metal trace, interleaved fp32 training rerun, or GPTQ/AWQ model benchmark was performed; none is required by this task.

## Stage commits

- `44c45f0` — `bench(l5): measure precision and cache overhead` (new ablation and raw data; syntax and whitespace checks passed).
- The final analysis, derived arithmetic, validation log and this report are committed together as `docs(l5): explain quantization tradeoffs`.

No push was performed.
