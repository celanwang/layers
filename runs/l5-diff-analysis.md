# L5 comparison: quantization and mixed precision

This is step 4 of the comparison-led implementation cycle, executed against [the task specification](../tasks/l5-diff-analysis.md). The implementations and Oracle checks already exist; this task reads production references and adds one isolated ablation, without changing them. Conclusions below distinguish measurements, byte/operation estimates, and causal hypotheses.

## Evidence and measurement scope

All local timing uses Apple M4 Max / 48 GiB unified memory, macOS 15.8, Python 3.12.15, MLX 0.32.4. L0 and L5 microbenchmarks use 3 warmups + 10 observations, upper median `sorted[n//2]` and nearest-rank p90 `sorted[int(0.9*n)-1]`. Training has an additional outer 3 warmup + 10 independent 200-step runs; each run excludes its first 3 steps and summarizes 197 timed steps. The inner training estimator is the unchanged L3 `sorted[max(0,int(n*q)-1)]` (indices 97/176 at q=0.5/0.9), rather than the microbenchmark median convention (`scripts/train.py:231–234,512–527`). p90 throughput is an upper throughput percentile, not tail latency. Relevant configurations:

| ID | Record and exact location | Configuration hash |
|---|---|---|
| E0 | [L0 hardware profile](../results/l0/hw-profile-20261021T065826Z.json), `matmul`, `bandwidth` | `ccb4c0a70605` |
| E3 | [L3 200-step baseline](../results/l3/train-metrics.json), `training`; [memory decomposition](l3-diff-analysis.md), §2 | `7997ac760223` |
| W1 | [Weight measurement](../results/l5/quant-bench-20261021T121645745015Z.json), `measurements` | `f782bb66df96` |
| W2 | [Independent weight rerun](../results/l5/quant-bench-20261021T122153246044Z.json), `measurements` | `fcbe6c7f8bdb` |
| K1 | [KV/training measurement](../results/l5/kv-bf16-bench-20261021T123306957293Z.json), `kv`, `bf16.summary`, `bf16.comparison` | `b7ce7d1d3a50` |
| K2 | [Independent KV/training rerun](../results/l5/kv-bf16-bench-20261021T124040042080Z.json), same fields | `718fd1a5fea3` |
| A1 | [New component ablation](../results/l5/precision-ablation-20261021T124703434945Z.json), `kv`, `gelu`, `rectangular_matmul`, `dtype_probe` | `8ce12717067e` |

K1's bf16 inner configuration is `b3b52a21c348`; K2's is `f7602b2398b1`. The two batch summaries are +26.31% and +34.21% throughput against E3, respectively; “+30%” is shorthand for these observations, not a more precise aggregate. E3 is a historical baseline, not an interleaved fp32 control. K1/K2 retain all warmup paths, measured training records, and raw samples. Weight/KV inference uses the same L3 500-step checkpoint (SHA-256 `a6bc079ef679c90ea8a4fdff95cf5086d4d49b8c811ce0252b2c31c7f1531889`).

Reproduce A1 with `.venv/bin/python -B bench/l5_precision_ablation.py`. Its immutable inputs are prepared/evaluated before timing; each cache trial starts at the same length. Only a new timestamped L5 JSON is written. The script and imported component hashes are in A1. It contains component timing, not a replacement training/decode protocol. Derived arithmetic and input checksums are recorded in [the accounting artifact](../results/l5/diff-accounting.json).

## 1. bf16 training: memory traffic explains more than the GEMM peak ratio

**Conclusion.** The measured 1.263–1.342× end-to-end gain is consistent with halving activation/intermediate traffic alongside a roughly 1.11–1.12× GEMM gain; parameter-state savings cannot explain the 1,172 MiB reduction. L0's low half-precision ratio has a specific implementation explanation: this machine's MLX Steel GEMM uses float operand fragments and float accumulation for all three storage dtypes, rather than a separate half-precision multiply path [M1–M4, E0, A1].

### Memory budget and traffic

E3 peaks at **2,618.36 MiB**; K2's median is **1,446.00 MiB**, a **44.77% / 1,172.36 MiB** reduction (K1: 1,445.98 MiB). For `P=13,118,464`, one fp32 parameter array is `4P=50.04 MiB`; weights + gradients + Adam m/v total **200.17 MiB**. The bf16 engine retains fp32 masters/m/v, makes a **25.02 MiB** bf16 compute-weight copy, produces bf16 gradients, then promotes them to fp32 before clipping/Adam (`scripts/train.py:404–448`). Therefore it does **not** halve all four optimizer states. A gradient tensor is 25.02 rather than 50.04 MiB before promotion; update-time fp32 gradients and possible overlapping buffers remain.

| Representative forward arrays, B16/T256/H8/d512/L4 | fp32 MiB | bf16 MiB | Reduction MiB |
|---|---:|---:|---:|
| Three score/mask/probability-sized tensors per layer | 384 | 192 | 192 |
| MLP preactivation and GELU output per layer | 256 | 128 | 128 |
| Q/K/V and rotated Q/K per layer | 160 | 80 | 80 |
| Representative inventory | **800** | **400** | **400** |

These are the shape counts from [L3 §2](l3-diff-analysis.md), supported by `layers/attention.py:25–35,51–61,105–125` and `layers/transformer.py:24–31,47–50`; they are not simultaneously live tensor measurements. For one full read and write of that inventory, logical traffic falls from **1,600 to 800 MiB**; backward reuse adds more opportunities. This is an array traffic budget, not measured DRAM transactions. A1 confirms bf16 output/storage for GELU, LayerNorm, exp, mean, sum, and the probed input gradient. RoPE's fp32 trigonometric preparation is explicitly cast back to the input dtype; loss logits are explicitly promoted to fp32 (`scripts/train.py:424–425`). Internal reduction arithmetic can differ from output dtype.

An illustrative high-water model keeps the four-state **200.17 MiB** allowance, halves the **2,418.19 MiB** remainder, and adds the compute-weight copy: `200.17 + 2418.19/2 + 25.02 = 1,434.29 MiB`, **11.71 MiB** below K2. This numerical agreement supports a mostly dtype-sensitive remainder; it does not prove its exact liveness or allocator composition. L3's cache-limit-zero ablation left the peak unchanged, so cached memory remaining after training is not itself the cause of the peak. Further saved activations, operator workspaces, backward temporaries, and promotion buffers are not individually measured [L3 §2, E3, K2].

### Elementwise work and end-to-end time

MLX's Metal `Erf` converts its scalar input to float, evaluates erf, and casts the result back to `T` [M5]. Thus bf16 GELU need not have twice the scalar arithmetic throughput to benefit from narrower arrays. The uncompiled expression in our MLP stores/reads multiple full-size intermediates (`layers/transformer.py:49`); compilation can fuse such work, but neither training path calls `mx.compile` [M6; `scripts/train.py:143–185,404–448`]. A1 measures:

| Same operation and shape | fp32 median / p90 ms | bf16 median / p90 ms | Median speedup |
|---|---:|---:|---:|
| Exact GELU, `[16,256,2048]` | 0.7228 / 0.7320 | 0.3949 / 0.3984 | **1.830×** |
| `[16,256,512] @ [512,2048]` | 0.7870 / 0.7998 | 0.7084 / 0.7118 | **1.111×** |

K2's outer throughput median / p90 is **83,761 / 83,940 tok/s**, versus E3's **62,413 / 62,767 tok/s** step distribution; K1's outer median / p90 is **78,836 / 81,668 tok/s**. These distributions have different replication levels and are not paired samples. A two-component illustration with GEMM speed `14.75/13.14=1.1225` and traffic-sensitive work speed 2 gives `1/R=(1-f)/1.1225+f/2`: the observed `R=1.263–1.342` corresponds to **f=25.4–37.3%** of baseline *time*. This is a consistency model, not a fitted kernel profile. L3's 92.6% projection/MLP *FLOP share* does not constrain elementwise/backward work to 7.4% of time [L3 §1, E0, K1/K2, A1].

### Explicit answer to L0 open question 1

L0's 4096³ fp32/fp16/bf16 medians are **13.14 / 14.60 / 14.75 TFLOP/s** (latency median/p90 **10.4625/10.4775**, **9.4135/9.5516**, **9.3205/9.3316 ms**). Their ideal minimum array arithmetic intensities are `2N³/(3N²·bytes)=N/(1.5·bytes)`: **683 FLOP/B fp32**, **1,365 FLOP/B bf16**. Both far exceed the L0 balance point **14.75 TFLOP/s / 450.6 GB/s = 32.7 FLOP/B**; cutting global bytes alone cannot force a 2× speedup for this large GEMM [E0].

The concrete path explains the weak dtype dependence: `AccumHelper<T>` selects **float**; `GEMMKernel` passes it to `BlockMMA`; A/B/C tiles use that same type; loads cast storage elements into float fragments before `simdgroup_multiply_accumulate` [M1–M3]. The alternative NAX route requires macOS ≥26.2 and architecture generation ≥17 (≥18 for suffix p); our recorded system is macOS 15.8 / `applegpu_g16s`, so that gate is closed [M4, A1.config.device]. **L0's ~1.1× is therefore consistent with a shared float MMA path plus narrower loads/tile choices, not evidence that bf16 failed to take effect. NVIDIA half-precision ratios are not a valid target for this path.** The rectangular A1 result independently reproduces that scale. No assembly trace establishes cycle-level shares, and this task does not claim the path is globally optimal or replace the proposed MPS/shape survey. The original “why only ~1.1×?” is answered at the implementation level; detailed tile tuning remains optional.

PyTorch AMP provides a useful contrast: autocast selects dtype per operation, and backward follows corresponding forward types [P1]. Our engine instead explicitly casts a compute model and promotes loss/optimizer work. Its 200-step heldout difference is approximately **−0.00084%** for K2's median (K1: +0.00053%), which validates this recipe's observed numerical behavior, not general AMP equivalence or all-bf16 numerical safety [E3, K1/K2].

## 2. INT8 KV: capacity savings coexist with conversion and synchronization costs

**Conclusion.** Persistent KV storage compresses **3.9844×**, but decode is **0.292–0.337×** because our interface materializes fp32 history and repeatedly quantizes/checks/synchronizes new tokens. The new component ablation shows the quantize-new path is much larger than dequant-only; attributing the entire slowdown to full-history dequantization would be unsupported [K1/K2, A1, `layers/inference.py:85–121`, `layers/quant.py:29–68,179–201`].

K1 times 32 single-token forwards plus greedy argmax, excluding prefill, at initial contexts 128/512/2048; fp32 and INT8 medians are **2.167/7.055**, **2.093/7.049**, **2.147/7.346 ms/token**, an extra **4.888–5.199 ms/token**. K2 adds **4.597–4.998 ms/token**. All use 3 warmups + 10 repetitions. K1's 32-token-window median/p90 seconds are fp32 **0.069341/0.071907, 0.066992/0.068294, 0.068692/0.069535**; INT8 **0.225772/0.228694, 0.225579/0.226866, 0.235058/0.238015**. K2's corresponding raw distributions are retained under `kv[*].formats[*].decode`. This near-constant multims penalty over 16× context growth is evidence against a purely bandwidth-proportional explanation.

### Per-step byte accounting

For current cache length n, B1/L4/d512, `N=2Ld·n=4096n` K/V elements. fp32 persistence is **16,384n B**; INT8 plus one fp16 scale per token/K-or-V/layer is **4,096n+16n=4,112n B**. Hence `16384/4112=3.984436×`, including scales. Each append quantizes only new tokens, then `get()` dequantizes *all* n tokens; old tokens are not requantized (`layers/inference.py:111–121`).

| Current n | INT8 + scales MiB | Reconstructed fp32 MiB | Minimum dequant read + write MiB | Including next attention read MiB | Extra q-cast intermediate read/write if separate MiB |
|---|---:|---:|---:|---:|---:|
| 128 | 0.502 | 2 | 2.502 | 4.502 | 4 |
| 512 | 2.008 | 8 | 10.008 | 18.008 | 16 |
| 2048 | 8.031 | 32 | 40.031 | 72.031 | 64 |

Minimum dequant traffic is **20,496n B** (q/scales read + fp32 output write); including attention consuming that output is **36,880n B**. An unfused q→fp32 cast adds **32,768n B** of intermediate write/read; scale casts add smaller traffic. These are logical traffic scenarios, not cache-miss measurements. INT8 concatenation copies q and scales, approximately **8,224n B read+write**; the fp32 baseline also concatenates K/V, approximately **32,768n B**. Thus storage ratio alone cannot determine net traffic, and the baseline's copying must not be omitted. During the 32-step windows n grows from n₀+1 to n₀+32, averaging n₀+16.5; the table is indexed by *current* n rather than falsely assuming fixed history.

### Estimated kernel work and measured stage costs

Each of 8 K/V arrays executes q-cast, scale-cast, broadcast multiply: **24 primitive operation slots** for dequantization, or a hypothetical **8 fused dequant kernels**. There are **16 concatenate slots** for qK/qV/sK/sV across four layers, versus 8 for fp32 K/V. Each of 8 new-token quantizations also invokes two host scalar checks (`.item()` at `layers/quant.py:44,50`) and one `mx.eval` (line 67); append adds four further eval calls: **16 scalar reads + 12 explicit eval calls per model token**. These are source counts, not profiler launch counts or 28 disjoint GPU stalls. Reduction implementations, copies, fusion, and already-completed buffers change actual launches; no Metal trace was obtained. A safe count is 24 dequant primitive slots + 16 concat slots, with new-token quantization/reductions in addition, rather than an invented exact kernel total.

A1 isolates the original four-layer append/get path and a variant supplied with the **identical precomputed new q/scales** (bitwise dequant output error **0** at all contexts). It also times the individual work groups; values are **median / p90 ms**:

| Initial n₀ | Original append + dequant | Prequantized append + dequant | Quantize new only | Dequant full only | fp32 concat only |
|---|---:|---:|---:|---:|---:|
| 128 | 5.730 / 5.913 | 0.897 / 0.908 | 4.542 / 4.786 | 0.194 / 0.207 | 0.166 / 0.173 |
| 512 | 4.462 / 4.591 | 0.738 / 0.751 | 3.769 / 3.821 | 0.210 / 0.233 | 0.169 / 0.172 |
| 2048 | 4.496 / 4.537 | 0.902 / 0.915 | 3.619 / 3.670 | 0.320 / 0.333 | 0.268 / 0.272 |

Supplying prequantized data removes **80–84%** of isolated original-append median time; the removed path includes quantization arithmetic, validation, synchronizations, and dispatch. It does not isolate `.item()` alone. Nor may these component medians be added to reconstruct K1/K2 end-to-end latency: barriers, evaluation overlap, and load conditions differ. The 128-token A1 original path is slower than the longer-context cases; all samples are retained and no monotonicity is asserted. Nevertheless, dequant-only at **0.19–0.32 ms** cannot on this evidence explain a **4.6–5.2 ms** total penalty by itself.

### Production comparison

vLLM v0.10.2's paged attention loads compressed K/V and applies `fp8::scaled_convert` inside the attention kernel before dot products; it avoids a separately materialized full-history fp32 buffer and its write/read [V1]. It still reads history: fusion removes intermediate traffic and launches, not attention's O(n) dependency. Current docs additionally describe FP8-domain attention with FlashAttention 3, with quantized queries, and per-tensor/per-head scale choices [V2]. FP8 vs typical bf16 saves about **2×** payload, whereas our reported **~4×** compares INT8 to **fp32**. Neither FP8 support nor a compressed cache alone guarantees a single-sequence latency gain; backend/hardware and capacity-driven batching matter. These CUDA/ROCm references are mechanisms, not a runnable M4 benchmark [V1/V2].

## 3. Weight PTQ needs the matmul kernel to consume packed weights

**Conclusion.** Our quantization reduces checkpoint bytes but rematerializes fp32 weights per projection; production MLX/MLX-LM connects packed weights directly to quantized matrix multiplication. This is the required coupling between quantization format and execution kernel, not a guarantee of speed from PTQ alone [`layers/quant.py:71–111,126–156`; M7–M10].

W1/W2 have identical quality/storage but different timing: **INT8 0.7208× / 1.0071×**, **INT4 g64 0.6064× / 0.6920×**. W2's decode median/p90 seconds are fp32 **0.327319/0.332780**, INT8 **0.325023/0.331335**, INT4 **0.473002/0.490421** (127 actual cached calls per observation); W1's are **0.321625/0.325494**, **0.446236/0.458092**, **0.530368/0.547352**. Thus no repeatable INT8 speedup is established; saying INT8 always slows by 28% would discard the independent rerun. INT4 slows in both. Neither batch profiles the cause of their timing variation [W1/W2].

All **24 block projections** access freshly reconstructed weights, with no fp32 cache. At an unfused primitive level, INT8 requires q-cast + scale-cast + multiply + matmul: **4 slots/projection, 96 total**, versus **24 quantized-matmul slots** in a fused idealization; transpose/reshape views are excluded. INT4 additionally uses AND, shift, integer casts, stack/reshape, sign reconstruction, and fp32 cast (`layers/quant.py:87–92`), so it adds more unpack work. Actual Metal launch counts depend on lowering/split-K/copy handling and are not measured.

The projection payload is **50,331,648 B fp32**, **12,619,776 B INT8**, **6,684,672 B INT4 g64**, including scales. Even an optimistic fused-dequant-then-separate-matmul INT4 path reads **6.685 MB**, writes **50.332 MB**, and rereads **50.332 MB**: **107.348 MB/token** of logical projection-weight traffic, versus **50.332 MB** baseline, before unfused intermediate costs. A fused packed matmul can avoid the full-weight write/read round trip. Actual DRAM traffic can be reduced by caches [W1.storage; `layers/quant.py:106–108`].

MLX's `QuantizedLinear.__call__` calls `mx.quantized_matmul`; native affine `mx.quantize` packs unsigned values with scales and biases, unlike our symmetric zero-point-zero signed grid [M7/M8]. MLX-LM's loader passes stored per-layer quantization to `nn.quantize`; its switched-expert layer uses `mx.gather_qmm` to select packed experts directly [M9]. **gather-qmm is the MoE path, not the dense GPT path used here.** MLX's Metal qmv/qmm kernels decode packed chunks within their compute loop rather than calling our full-array dequantizer [M10]. Such W4A16-style kernels can unpack to floating-point local values before multiply; “production quantization” does not necessarily mean integer MAC instructions. These source paths explain how to realize traffic savings without claiming an unmeasured native speed ratio.

The pedagogical choice exposes q/scales, signed packing, roundtrip bounds, and separate conversion cost; production chooses fused consumption and backend-specific dispatch. The native public Oracle already reports zero max error for our adapted grid and different native-affine RMSE; this establishes representation alignment, not identical native algorithms or quality [W1/W2.`mlx_public_api_oracle`].

## 4. RTN, GPTQ, and AWQ: compare objectives and use the same metric

**Conclusion.** Our calibration-free INT4 g64 RTN result is a modest **+2.549% perplexity** change on this checkpoint/heldout set; it is not a +0.59% perplexity result, and it does not rank RTN above GPTQ/AWQ across different models and group sizes [W1/W2; G1; A2].

| Method | What determines the quantized weights | Concrete evidence |
|---|---|---|
| Ours: symmetric RTN | Group max/7, stored fp16 scale, nearest-even rounding; no calibration activations or inter-column compensation | `layers/quant.py:29–68`; INT4 default g64 at lines 17–26 |
| GPTQ | Minimize layer output reconstruction error using calibration X; `H=2XXᵀ`, damped inverse/Cholesky information compensates rounding errors in remaining columns | [G1] §3–§4, equations (1)–(3), Algorithm 1 |
| AWQ | Activation statistics select important input channels; channel scaling protects them while preserving the equivalent linear map, avoiding mixed-bit exception weights | [A2] §3.1–§3.2, equations (2)–(5) |

AWQ's Table 4 supplies a same-dataset **WikiText-2 INT4 g128** comparison (values below are PPL; percentages are calculated relative to each FP16 row):

| Model | FP16 | RTN | GPTQ | GPTQ-Reorder | AWQ |
|---|---:|---:|---:|---:|---:|
| Llama-2-7B | 5.47 | 5.73 (+4.75%) | 5.69 (+4.02%) | 5.63 (+2.93%) | 5.60 (+2.38%) |
| LLaMA-7B | 5.68 | 5.96 (+4.93%) | 6.22 (+9.51%) | 5.83 (+2.64%) | 5.78 (+1.76%) |

The table also illustrates why specifying GPTQ ordering matters; it does not establish universal method superiority [A2, Table 4]. GPTQ's original paper documents its own second-order objective/implementation; the g128 numbers here are explicitly the AWQ paper's comparison, not misattributed to GPTQ's original table [G1; A2].

Our loss **4.27175664 → 4.29692684 nats** is **+0.589224% cross-entropy**; `PPL=exp(loss)` gives **71.6474 → 73.4736**, or `exp(0.02517019)-1 = +2.548964%`. Quality covers all **44,467 heldout target tokens**, context 256, the L1 tokenizer and a 13.1M model, with embeddings/tied head/norms protected [W1/W2.`config.quality`, `measurements`]. Numerically this is in the same few-percent PPL band as the paper examples, but the corpus/tokenizer, training maturity, model size, protected tensors, symmetric vs affine format, and **g64 vs g128** differ. g64 spends **4.25 bits/projection parameter** with fp16 scales; our equivalent g128 bookkeeping would be **4.125 bits**, before any quality effect. We therefore pass the specified <5% loss gate, while a same-checkpoint g128 + calibrated-method experiment is still needed to compare methods. No GPTQ/AWQ model was run in this task.

## 5. Missing pieces and proposed landing points

These are scoped follow-up recommendations for planning review, not approved implementation work.

| Missing piece | Evidence for the gap | Proposed layer and measurable question |
|---|---|---|
| **KV INT4** | `QuantKVCache` stores INT8 only; its scale crosses all heads/features (`layers/quant.py:179–201`). INT8 already adds 4.6–5.2 ms/token, and >256-token quality probes extrapolate beyond training context [K1/K2]. | **Precision专项 after L6's fused-cache baseline**: compare per-token/per-head/grouped INT4, scale overhead and packed-kernel latency; measure aligned-logit error and long-context task quality. First remove the conversion bottleneck, then ask whether another ~2× payload reduction pays off. |
| **QAT** | RTN INT4 adds 0.02517 nats; no fake-quant/STE path exists in `layers/quant.py:29–68`, and the bf16 engine does not train with quantization noise (`scripts/train.py:404–448`). | **Separate algorithm/training iteration after L5**, not a serving prerequisite: same checkpoint, bit/group format and heldout targets, equal additional training budget, Oracle-check fake-quant/gradient semantics; compare recovered loss and added train cost. |
| **Mixed precision/granularity for sensitive layers** | Our exclusions are categorical, not measured per-layer sensitivity (`layers/quant.py:159–175`); native MLX supports path-dependent predicates and MLX-LM consumes per-layer settings [M7/M9]. | **Precision专项**: one-layer-at-a-time quantization at fixed checkpoints; rank Δloss and choose g64/g128 or 4/8/bf16 under a fixed byte budget. Feed the chosen format into **L6** only after its kernel supports each format. |
| **KV cached/uncached crossover, context n≥256** | [L4 §3](l4-diff-analysis.md) predicts a context-length crossover and leaves paired measurement open; K1/K2 compare fp32-KV to INT8-KV, not cached to uncached. | **Independent L4 carry-over专项, before interpreting L6 speedups**: compare paired cached/uncached latencies at context n=256 and explicitly configured larger contexts, with identical teacher-forced history/output positions and alternating/randomized order. Predeclare the warmup/repetition count under a separate approved protocol; report median/p90, paired confidence intervals, prefill, hashes/device/precision, and native ≤256 vs RoPE-extrapolated >256 behavior. Here n is context length; the number of sample pairs is a separate protocol parameter. |

The **L6 runtime** candidate is true kernel batching plus attention that consumes compressed/page-table KV directly; this follows the source differences in §2–§3 and the unbatched per-request loop identified in [L4 §1](l4-diff-analysis.md). Precision research and runtime integration have distinct acceptance questions; neither the proposed work nor a production-kernel speedup is claimed as completed here.

## External reading locations

Read on 2026-10-21. Reference source bodies were not copied into the repository. MLX is pinned to installed version tag v0.32.4; MLX-LM is pinned to the inspected commit, not a moving main link.

- **M1** — [MLX `AccumHelper`, `transforms.h:56–59`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/steel/gemm/transforms.h#L56-L59): float accumulation type.
- **M2** — [MLX `GEMMKernel`, `gemm.h:19–69`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/steel/gemm/gemm.h#L19-L69): storage/accumulator template connection.
- **M3** — [MLX `mma.h:57–65,198–205,451–483,512–531`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/steel/gemm/mma.h#L451-L531): operand fragments, typed loads, MMA.
- **M4** — [MLX Metal device NAX gate, `device.cpp:978–995`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp#L978-L995) and [matmul routing, `matmul.cpp:964–974,1088–1116`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp#L964-L974).
- **M5** — [MLX `unary_ops.h:163–180`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/unary_ops.h#L163-L180): scalar Erf and Exp implementations; [native bfloat alias](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/bf16.h#L9-L15).
- **M6** — [MLX Compilation guide, “Why Compile?” and fusion discussion](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
- **M7** — [MLX `QuantizedLinear.__call__`, `quantized.py:250–285`](https://github.com/ml-explore/mlx/blob/v0.32.4/python/mlx/nn/layers/quantized.py#L250-L285); [`nn.quantize` path predicate, lines 24–93](https://github.com/ml-explore/mlx/blob/v0.32.4/python/mlx/nn/layers/quantized.py#L24-L93).
- **M8** — [MLX `mx.quantize`, affine representation and return values](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html); [`gather_qmm`, indexed packed multiplication](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html).
- **M9** — MLX-LM commit `72be8c955d9a0d3099bd01865af288e0e947c3fa`: [loader `utils.py:496–519`](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/utils.py#L496-L519), [switched quantized linear `switch_layers.py:93–106`](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/models/switch_layers.py#L93-L106).
- **M10** — [MLX Metal packed dequantization helpers and qmv compute, `quantized.h:558–662,850–963`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/quantized.h#L850-L963); [dispatch `quantized.cpp:465–543`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L465-L543).
- **G1** — [GPTQ, arXiv:2210.17323v2](https://arxiv.org/pdf/2210.17323v2), §3–§4, Algorithm 1 (paper pp.3–5); layer reconstruction and compensated quantization.
- **A2** — [AWQ, arXiv:2306.00978v5](https://arxiv.org/html/2306.00978v5), §3.1–§3.2, §5.1 and **Table 4 (INT4 g128 WikiText-2)**. Tables are present in HTML; the large PDF fetch failed, so numeric rows were read from HTML tables.
- **V1** — [vLLM v0.10.2 `attention_kernels.cuh:279–288,409–414`](https://github.com/vllm-project/vllm/blob/v0.10.2/csrc/attention/attention_kernels.cuh#L279-L288): in-attention FP8 conversion of K and V.
- **V2** — [vLLM quantized KV cache docs, “FP8 KV Cache Overview” and scale schemes](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/). The current FA3 note and the version-pinned V1 kernel are distinct backends/versions.
- **P1** — [PyTorch 2.14 AMP docs, autocast and CPU bf16 example](https://docs.pytorch.org/docs/2.14/amp.html#autocasting): per-operation selection and backward types, not evidence about Metal hardware speed.

## Validation and limits

The new ablation completed with exit 0; all prequantized/original cache outputs agree exactly. Syntax compilation and staged whitespace checks passed. Per task §5, the test suite was not rerun and no existing benchmark was edited or rerun. [Acceptance log](../results/l5/diff-acceptance.txt) records checks; [section-6 output report](../results/l5/l5-diff-report.md) records files, five conclusions, key numbers, source locations and deviations. No dependencies were added, no source reference bodies were vendored, and no push was performed.

The justified scope is an implementation-level explanation plus component evidence. End-to-end paired precision timing, exact live-memory attribution, Metal launch/cycle counts, and same-model GPTQ/AWQ quality remain unmeasured. These limits do not turn byte budgets or the Amdahl illustration into measured kernel shares.
