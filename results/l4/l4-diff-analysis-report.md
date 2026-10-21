# L4 Diff Analysis Report

## Five conclusions

1. The latest 8-request result is 1.044× end-to-end, but shared 4-token prompt prefill is included and all requests finish together; no-bubble scheduling is not isolated.
2. A source-level forward has 59 core stage slots (75 named op groups including residuals and cache appends); the 3.675% nominal roofline result is 7.38% against L0's small-workload bandwidth, with exact Metal launch count unavailable because capture is unsupported.
3. Cache-vs-uncached measurements are 1.098× and 0.968×; the dense-prefix arithmetic-intensity model crosses L0's 32 FLOP/B balance around 64 tokens and predicts more visible gains around 128–256 tokens.
4. Our 16-token block matches vLLM's empirically selected default; sharing 8 × 32-token prefixes saves 87.5% of prompt KV, while T2's gather/concatenate path materializes one K/V gather per layer per step once the cache spans pages.
5. L5 should measure KV-cache quantization; L6 should add kernel batching, mixed prefill/decode with eviction, and then speculative decode.

## Key numbers

- **Continuous scheduling:** 8 requests × 16 outputs; latest medians 0.321320 s vs 0.335489 s, 398.36 vs 381.53 tok/s, 1.044×; other retained runs span 1.031–1.062×.
- **Batch-8 weight-read ledger:** 8 independent forwards ≈420.05 MB/iteration; one batched forward ≈52.51 MB, an ideal 367.54 MB / 87.5% reduction. At 224.1 GB/s, the weight-only batch-8 ceiling is ≈34.1k aggregate tok/s. Preserving the observed sequential efficiency would imply ≈3.05k tok/s, not 34.1k.
- **Decode:** 314.96 tok/s against 8,571 nominal tok/s = 3.675%; against 224.1 GB/s = 4,268 tok/s and 7.38%. Median time is 3.176 ms/token; one 52.5 MB read is ≈0.234 ms at 224.1 GB/s.
- **Cache:** prompt32/max_new32 cached/uncached medians are 0.093867/0.103026 s (1.098×) and 0.097695/0.094606 s (0.968×). The tested uncached loop processes 1,520 token rows versus 64 in the cached path, but both still perform a model forward and weight pass per decode call.
- **KV pages:** 16 KB/token FP32 across four layers; 8 requests × 32 shared prefix tokens use 524,288 B vs 4,194,304 B unshared, saving 3,670,016 B (87.5%).

## External source locations

- PagedAttention paper §§4.4, 5, 6.3–6.4, 7.2: https://arxiv.org/abs/2309.06180
- vLLM block-size default 16, v0.6.2 Engine Arguments: https://docs.vllm.ai/_/downloads/en/v0.6.2/pdf/
- vLLM scheduler and attention metadata: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py ; https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py
- llama.cpp server batching and batch API: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md#batching ; https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- mlx-lm generate and speculative paths: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- MLX lazy-evaluation behavior: https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst
- Public M4 Max MLX-LM decode data and protocol: https://github.com/john-rocky/apple-silicon-llm-bench/blob/main/results/hybrid/hybrid-apple-silicon-bench-2.md#results

## Deviations

- The consulted sources contain no same-size (~13M) public MLX-LM decode result. The closest public decode result is Granite-4.0-H 350M at 521.7 tok/s; a 10M MLX-LM model card has no decode tok/s. The 350M row is presented only as context, with precision, architecture, hardware-memory and protocol differences noted in the main analysis.
- The requested 1.04× no-bubble attribution cannot be recovered from the current equal-length, shared-prefill benchmark. The report gives the measured result and isolates the kernel-weight-read opportunity mathematically, but does not invent a scheduler/kernel split.
- A true Metal launch count could not be collected: MLX returned “Capturing is not supported”; the 75 figure is a source-level op-group count, not a kernel trace.
