# L7 comparison — execution output (`-o`)

## Five conclusions

1. T1 still writes a **4 MiB bank + 4 MiB selected KV output per layer** at B=8/equal L=128/fp32; T2 shares a sampling fence but retains row graphs, unlike page-reading attention and batch-oriented production processors.
2. The direct combined-pair endpoint slope falls **37.83–39.75%**, but the intercept rises **0.1550–0.2367 ms**, leaving **32.21–32.28%** below the historical 2792.31 tok/s projection.
3. Small-batch bank/graph costs are poorly amortized, and B=1/2 performance drifts across runs, so the data do not establish a safe dispatch threshold.
4. Controlled intervention supports host small-operation/synchronization overhead as a major scaling constraint in this workload, without proving a host-time majority or generalization beyond greedy/13M/unified-memory inference.
5. Recommend **L7 profiling first**, then an **L7 fused-SDPA ablation**, and a separate **L5×runtime INT8-page task** after conversion and synchronization costs are understood.

The main analysis provides evidence locations for each point:
[`runs/l7-diff-analysis.md`](../../runs/l7-diff-analysis.md), §§1–5.
All arithmetic and original source hashes:
[`diff-accounting.json`](diff-accounting.json), analysis hash **`24ed91a0c873`**.

## Old/new endpoint parameters

`T(B)=1000*decode_median_seconds(B)/63=a+bB`, using B=1/8 endpoints;
B=2/4 are residual checks. This is not OLS or a CPU/GPU cost decomposition.

| Series | a ms | b ms/row | Δa vs historical | Slope reduction |
|---|---:|---:|---:|---:|
| Historical L6 | 1.6406 | 0.5022 | — | — |
| Execution S1, direct L6/T2 edge, T2 arm | 1.8773 | 0.3025 | +0.2367 | 39.75% |
| Independent S2, direct L6/T2 edge, T2 arm | 1.7955 | 0.3122 | +0.1550 | 37.83% |
| S1, T1/T2 edge, T2 arm | 1.8273 | 0.3011 | +0.1867 | 40.04% |
| S2, T1/T2 edge, T2 arm | 1.8605 | 0.2993 | +0.2199 | 40.39% |

The direct combined B=8 results are **1891.05 / 1893.00 tok/s**;
**1918.45 / 1909.94** belongs to the independent T1/T2 edge.
Direct-pair B=2 residuals are +0.2075/+0.2085 ms, exposing the endpoint
model's imperfect small-batch fit. Main analysis §2 includes the L6/T1 control
fits, all B=1/2/4/8 medians/p90s and source pointers.

**Largest residual candidate:** per-row page writes, Python metadata/bank
construction and row-graph dispatch. B=8 still makes 32 page append calls per
forward across four layers. Their individual time shares are unmeasured.
`logits[row,0]` split/restack is the weakest cost attribution; source calls
alone do not prove a large copy or stall. Evidence: main analysis §2 table.

## Validation and actual changed files

- **2160 archived raw timings** independently verified against stored upper medians/p90s and throughput arithmetic; both T2 measurements and all three T1 measurements retained.
- Endpoint equations, byte budgets, source hashes, local evidence links and whitelist checks pass; `git diff --check` passes.
- No test suite or benchmark run: task §5 explicitly prescribes documentation acceptance without tests. No project lint configuration/command exists; whitespace checks are the applicable document check.
- Full validation record: [`diff-acceptance.json`](diff-acceptance.json).

Only these new files are delivered:

- `runs/l7-diff-analysis.md`
- `results/l7/diff-accounting.json`
- `results/l7/diff-acceptance.json`
- `results/l7/l7-diff-output.md`

Stage commit **`45edfc6`**: `docs(l7): account for decode scaling and page copies`.
The final archive commit contains the analysis, acceptance record and this
output. No implementation/test/benchmark/dependency/protocol changes, deletions,
external source copies, push or merge.

## External locations

vLLM source is pinned to v0.10.2 commit
`01efc7ef781391e744ed08c3292817a773d654e6`; accessed 2026-10-22.

- [PagedAttention paper, §5.1](https://arxiv.org/html/2309.06180v1#S5): fused writes, reads with attention, and CoW copies; §4.6/§6.1 for control/hardware boundaries.
- [FlashAttention paper, §3.1 / Algorithm 1](https://arxiv.org/html/2205.14135v2#S3.SS1): block normalization and kernel fusion; Appendix B.1 recurrence.
- [vLLM v0.10.2 design](https://docs.vllm.ai/en/v0.10.2/design/paged_attention.html): historical warning, Inputs/Key/Softmax/Value.
- [Tagged attention kernel, lines 198–270, 298–351, 389–398, 529–568, 638–660](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/csrc/attention/attention_kernels.cuh#L198-L270): page addressing and reductions.
- [Tagged decode dispatch, lines 20–21 and 123–195](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/attention/ops/paged_attn.py#L123-L195) and [versioned API `forward_decode`](https://docs.vllm.ai/en/v0.10.2/api/vllm/attention/ops/paged_attn.html#vllm.attention.ops.paged_attn.PagedAttention.forward_decode): batch/head/length heuristic.
- [Tagged processor interface, lines 59–97](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/sample/logits_processor/interface.py#L59-L97) and [versioned API](https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/sample/logits_processor/interface.html#vllm.v1.sample.logits_processor.interface.LogitsProcessor): batch logits and state updates.
- [Tagged sampler, lines 85–117 and 128–192](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/sample/sampler.py#L85-L192): processor, temperature and greedy/random paths.
- [MLX SDPA API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html), matching installed MLX 0.32.4 public docstring: mask/shapes/`force_fused`.
- [MLX Metal Debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html), plus installed public capture docstrings: capture prerequisites and replay.

## Deviations and clarified approximations

No scope or protocol deviations. Existing evidence sufficed; no ablation was
added. The invocation supplies no literal `-o` filename, so this file follows
the existing T1/T2 output convention. The connected GitHub tool was unavailable;
public raw GitHub reads supplied the tagged source, kept out of the repository.

The specification's approximate B=8 1893–1918 range mixes independent edges;
the direct combined edge is 1891/1893. Its approximate small-batch regression
range also omits S2's +3.63% combined B=2 result. These are reported as source
corrections, not filtered or repeated experiments. Mean L=128 byte accounting
states its equal-length/no-sharing assumptions. Proposed time/gain estimates
are conditional sensitivities and work estimates, not new measurements or
approved follow-up work.
