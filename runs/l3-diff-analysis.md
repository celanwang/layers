# L3 Diff Analysis — Training Stack

> Scope: compare the L3 self-built MLX trainer with nanoGPT, nanochat, and MLX-LM's public trainer. Evidence uses the recorded L3/L0/L2 artifacts and a small, process-isolated MLX memory ablation. No implementation, training, or test files were changed.

## Measurement basis

- Main training reference: 500 steps, config hash `2c846ded0b41`, Apple M4 Max / 48 GiB, MLX 0.32.4, fp32, 4 layers, `d=512`, `h=8`, `T=256`, batch 16 (`4096` tokens/update). Throughput median `62,023.6` tokens/s; the recorded MFU uses `78.7 MFLOP/token` and `14.75 TFLOP/s` ([metrics](../results/l3/train-metrics-500.json), [training report](../results/l3/l3-training-report-v2.md)).
- Memory ablation: Apple M4 Max / 48 GiB, macOS 15.8, Python 3.12.15, MLX 0.32.4; separate process per case; one pre-reset loss evaluation, then peak reset and three updates/forwards. One run per case, so these are allocation high-water marks, not timing distributions or independent-run estimates. The per-case config hashes and raw API readings are in [memory-ablation.json](../results/l3/memory-ablation.json); the allowed harness is [l3_memory_ablation.py](../bench/l3_memory_ablation.py).
- FLOP convention counts multiply-add as two FLOPs and approximates forward+backward matmul work as 3× forward. It omits lower-order elementwise ops, normalization, clipping, and optimizer arithmetic. The corrected `6N + attention` convention matches nanoGPT's own `estimate_mfu` implementation [G1].

## 1. Why the measured MFU is about 33%, not below 10%

The 500-step run's reported `33.09%` is the result of applying `6ND` alone. With dense causal attention included, the comparable MFU is about **35.7%** of the L0 fp32/bf16 large-matmul reference peak; the <10% prediction in the principle note §6 missed because each update processes 4096 tokens through dense batched projections, rather than doing mostly tiny scalar work ([principle note](l3-training-principle.md)).

### FLOPs per token and per update

For `N=13,118,464`, `L=4`, `d=512`, `h=8`, `d_head=64`, `T=256`, `B=16`:

| Work | Calculation | FLOPs/token | FLOPs/update |
|---|---:|---:|---:|
| Attention/MLP/output linear projections | explicit matmuls: `72Ld² + 6dV`, `V=1028` | 78,655,488 | 322.173 GF |
| 6N proxy for those linear operations | `6 × 13,118,464` | 78,710,784 | 322.399 GF |
| Dense attention score and value products | `12L T d` per input token; equivalently `12BLT²d` per update | 6,291,456 | 25.770 GF |
| Corrected nanoGPT-style estimate | `6N + 12L T d` | **85,002,240** | **348.169 GF** |

The explicit linear matmuls are within `0.07%` of `6N`; that approximation is adequate for their accounting. The missing sequence-dependent term is `12 × 4 × 256 × 512 = 6.291 MFLOP/token` (about `8.0%` of the corrected total). Attention is only `7.40%` of this model's total training FLOPs, so the missing term cannot explain why the observed MFU was high; it instead makes the old MFU modestly too low. The formula is also the one implemented by nanoGPT at `model.py:289–303` [G1].

Using the recorded 500-step median `62,023.6 tokens/s`:

- 6ND-only numerator: `62,023.6 × 78.711 MFLOP = 4.882 TFLOP/s`; divided by `14.75 TFLOP/s` = **33.10%** (recorded: `33.093%`).
- Explicit projection plus attention numerator: `62,023.6 × 84.947 MFLOP = 5.268 TFLOP/s`; divided by `14.75 TFLOP/s` = **35.72%**.
- Using the canonical `6N + attention` estimate instead gives `5.272 TFLOP/s`, **35.74%**. The `0.02` percentage-point difference is the small 6N-vs-explicit-linear approximation above.

The 200-step run tells the same story: `33.30%` recorded and `35.94–35.97%` corrected, depending on explicit versus 6N linear accounting ([metrics](../results/l3/train-metrics.json)).

### Why this small model reached that range

The L0 matmul medians rise from `6.23 TFLOP/s` at `1024³` to `11.39 TFLOP/s` at `2048³` and `13.14 TFLOP/s` at `4096³`; the cited `14.75 TFLOP/s` is the L0 bf16 `4096³` result, not a measured ceiling for every shape ([L0 profile](../results/l0/hw-profile-20261021T065826Z.json)). L3 processes `B×T=4096` token rows per update, but the source keeps the tensor batched: Q/K/V/O are batched `16 × (256×512) @ (512×512)`, and the MLP uses `16 × (256×512) @ (512×2048)` then `16 × (256×2048) @ (2048×512)` (`layers/attention.py:105–125`, `layers/transformer.py:47–50`). Those shapes are larger than pointwise operations and the projection/MLP path accounts for about `92.6%` of the estimated FLOPs; the exact batched shapes were not benchmarked in L0, so this is a mechanism consistent with the data, not a measured per-kernel explanation. The measured full-step throughput is roughly `5.27 TFLOP/s` under the corrected operation count, or `35.7%` of the L0 peak reference.

L2 found its fast attention path was `2.07×` faster than the explicit path at `n=256`, and isolated softmax took `40.6%` of full attention time at `n=1024` ([L2 report](../results/l2/l2-attention-report.md), [benchmark JSON](../results/l2/attention-bench-20261021T094129Z.json)). Those are meaningful attention-kernel costs, but attention contributes only `7.40%` of L3's estimated training FLOPs. The MLP and projections dominate total arithmetic.

### What “MFU” means here

`6ND` is useful for quick scaling estimates, but it omits the `O(T²d)` attention products; for this configuration it understates the corrected utilization by about `2.62` absolute percentage points. Conversely, the corrected count remains a model-based numerator: it assumes conventional dense attention and 3× forward matmul work, while the denominator is L0's best large bf16 matmul, not a sustained peak measured on L3's fp32 rectangular kernels. Thus **35.7% is a corrected, peak-normalized estimate**, not a hardware-counter measurement or a claim that every kernel reaches 35.7% of its own attainable maximum.

## 2. Where the 2618 MiB peak comes from

The four nominal fp32 parameter states are `13,118,464 × 4 bytes × 4 = 209,895,424 bytes = 200.17 MiB` (parameters, gradients, Adam first moment, Adam second moment). The principle note predicted these states plus activations below 1 GiB (§6); the 500-step run instead reports `2,618.36 MiB`, leaving a **2,418.19 MiB** process peak-allocation remainder. That remainder is not a direct measurement of simultaneously live activations ([principle note](l3-training-principle.md), [metrics](../results/l3/train-metrics-500.json)).

### Shape-based activation budget

At batch 16, sequence 256, eight heads, fp32:

- One `[B,H,T,T]` attention tensor is `16×8×256²×4 = 33,554,432 bytes = 32 MiB` per layer; one such tensor across four layers is `128 MiB`. Three representative full-size arrays along the score/mask/probability path account for `384 MiB`; the scaled-score and stable-softmax shift/exponential temporaries can add further arrays. The operations are explicit in `layers/attention.py:51–61, 119–125`.
- One MLP preactivation `[B,T,4d]` is `16×256×2048×4 = 32 MiB` per layer. Keeping both its preactivation and GELU output across four layers accounts for `256 MiB`, before the additional elementwise GELU intermediates (`layers/transformer.py:47–50`).
- Q, K, V, and rotated Q/K are five `[B,T,d]` tensors, `8 MiB` each: up to `160 MiB` across four layers (`layers/attention.py:105–120`). Residual streams, logits, normalization, and backward temporaries are additional.

These visible tensors give a useful lower-order inventory (`384 + 256 + 160 = 800 MiB` for representative score, MLP, and Q/K/V storage). It is not a liveness proof: MLX's lazy graph is evaluated on demand, while reverse-mode differentiation also needs values for backward; whether an intermediate is retained, reused, or released affects the high-water mark [M1].

This is also a concrete attention-path difference from nanoGPT: L3 explicitly computes `QKᵀ`, applies a mask, runs stable softmax, and multiplies by V (`layers/attention.py:51–61, 119–125`); nanoGPT calls PyTorch scaled-dot-product attention when available, with its explicit score/mask/softmax implementation as fallback (`model.py:61–72`) [G1]. The L3 path favors inspectable values and Oracle tests, while the fused reference can avoid storing the full score/probability matrices. This explains why L3's memory profile is not predicted by parameter states alone, although the ablation above shows those matrices alone do not explain the entire peak.

### Measured group ablations

The isolated full-training case reached **2,668.3 MiB** after three updates; the original 200/500-step runs report **2,618.3–2,618.4 MiB**. The ablation is `49.9 MiB` higher, so it supports scale attribution but does not reproduce the main-run peak exactly. It uses random token IDs, one process per case, and only one run per setting.

| Case | Peak MiB | Change vs full training | End active MiB | End cached MiB |
|---|---:|---:|---:|---:|
| Train, B=16, L=4 | 2,668.3 | — | 250.3 | 2,631.8 |
| Forward only, B=16, L=4 | 1,492.6 | −1,175.7 | 66.1 | 1,426.4 |
| Train, B=8, L=4 | 2,018.8 | −649.5 | 250.3 | 1,852.5 |
| Train, B=16, L=2 | 1,963.3 | −705.0 | 130.2 | 2,002.2 |
| Train, B=16, L=4, cache limit 0 | 2,668.3 | 0.0 | 250.3 | 0.0 |

The forward-only run still allocates the score and MLP workspaces; adding autograd, gradients, optimizer state, and backward temporaries raises its peak by about `1.10 GiB` in the main run and `1.15 GiB` in the three-update ablation. Halving batch lowers the measured peak by `24.3%`; halving layer count lowers it by `26.4%`. Those controlled shape changes support activations and reverse-mode intermediates as the main excess beyond the four parameter states.

MLX documents that `get_active_memory()` excludes cached buffers, `get_cache_memory()` measures unused buffers not returned to the system allocator, and `get_peak_memory()` records the maximum used since reset [M2]. In the ablation, setting the cache limit to zero changed the end cache from `2,631.8 MiB` to zero but left the peak at `2,668.3 MiB`. Thus cache retention explains why memory remains reserved after a step; it does not, by itself, explain the measured peak. The remaining gap between the explicit `~800 MiB` tensor inventory and the `2.4 GiB` non-parameter high-water includes further saved activations, backward/operator temporaries, and allocator high-water effects; this experiment does not isolate those into exact tensor-by-tensor shares.

## 3. Hyperparameters versus small-model references

L3 keeps a recognizable GPT AdamW recipe (`wd=0.1`, `β₂=0.95`, cosine floor 10%) but chooses a lower peak LR and a much larger warmup fraction than the closest nanoGPT miniature recipe; the short 500-step horizon makes absolute `warmup=100` equal to 20% of all updates.

| Setting | L3 (500-step run) | nanoGPT small Shakespeare config | nanoGPT main GPT-2 defaults | Current nanochat defaults |
|---|---:|---:|---:|---:|
| Shape / tokens per update | 4×512, 4096 tokens | 6×384, 16,384 chars | 12×768, 491,520 tokens (`12×1024×40`) | dynamic depth/sequence; auto total batch, d12 reference `524,288` tokens |
| Peak LR | `3e-4` | `1e-3` | `6e-4` | separate embedding/unembedding/matrix/scalar rates; not one AdamW LR |
| Weight decay | `0.1` | inherited default `0.1` | `0.1` | `0.28` Muon default, scaled; AdamW groups `0.001–0.01` |
| β₂ | `0.95` | `0.99` (comment says longer averaging for small tokens/update) | `0.95` | per-group: AdamW `0.95/0.96/0.995`, Muon `0.9` |
| Warmup fraction | `100/500 = 20%` (`100/200 = 50%` in the earlier run) | `100/5000 = 2%` | `2000/600000 = 0.33%` | 40 steps; warmdown spans `65%` of training |
| End LR | `10%` of peak, cosine | `10%` of peak, cosine | `10%` of peak, cosine | `5%` of initial, linear warmdown |

The nanoGPT Shakespeare model is near L3 in parameter scale: with tied output weights and no linear/norm bias, `65×384 + 256×384 + 6×(12×384² + 2×384) + 384 = 10.745M` parameters, versus L3's `13.12M`; both use context 256. The batch is four times larger in the Shakespeare recipe, while the number of update steps is ten times larger. It is therefore a useful same-order scale comparison, not an apples-to-apples data or tokenizer comparison. Its comments directly tie `β₂=.99` to small tokens/update and its warmup to 100 of 5000 steps [G2].

The `3e-4` L3 rate is half nanoGPT's 124M GPT-2 default and below the mini-model's `1e-3`; this is a conservative choice for a short from-scratch run, not an empirical optimum established by these results. `wd=.1` and `β₂=.95` match nanoGPT's main GPT-2 defaults. The 20% warmup is much longer proportionally than either nanoGPT horizon; relative to the earlier 200-step attempt it consumes half the run. The cosine endpoint itself matches nanoGPT's 10% minimum. Current nanochat is a different production-style recipe: Muon for matrices with Adam for other parameter groups, `40` warmup steps, linear warmdown over the last `65%`, and a `5%` terminal rate [G1, G3].

L3 intentionally stayed fp32 for cross-framework/oracle transparency. Both nanoGPT and nanochat have lower-precision execution paths; nanoGPT wraps model calls in autocast, and nanochat explicitly uses GradScaler only for fp16 (bf16/fp32 need none) [G1, G3]. This is a teaching/debugging tradeoff, not evidence that fp32 is the preferred throughput mode.

## 4. Training-loop gaps and a provisional L4/L5 placement

L3 already implements a cosine schedule to 10% and global norm clipping, but it is a single-process, fixed-batch, fixed-horizon run; its end checkpoint stores model weights only. The following placement is a dependency-based suggestion for this training-systems work and does not settle the project's separate iteration order.

| Capability | L3 status | Reference evidence | Suggested layer |
|---|---|---|---|
| Gradient accumulation | Missing; one batch/update | nanoGPT scales microbatch loss and updates after its accumulation loop; nanochat computes `grad_accum_steps` from total/per-device batch; MLX-LM carries `grad_accumulation_steps` in `TrainingArgs` [G1, G3, G4] | **L4** — preserves effective batch while reducing per-device activation memory |
| bf16 autocast / lower precision | Missing; metrics report `float32` | nanoGPT uses `torch.amp.autocast`; nanochat has compute dtype and fp16-only GradScaler [G1, G3] | **L4** — first add a small-batch Oracle/parity check, then benchmark |
| Checkpoint resume | Partial save only: final model NPZ, no optimizer, step, sampler, or RNG state | nanoGPT saves model, optimizer, iteration and best val loss; nanochat saves model, optimizer, loop and dataloader state; MLX-LM supports saving adapter weights [G1, G3, G4] | **L4** — single-host restart determinism before scaling |
| LR decay / early stop | Cosine decay to 10% exists; no patience-based early stop or best-validation checkpoint selection | nanoGPT has cosine/warmup and best-val checkpoint; nanochat has warmup/linear warmdown; MLX-LM validates at configured intervals [G1, G3, G4] | **L4** — keep the current scheduler, add validation-driven stopping/checkpoint selection only if needed |
| Distributed training | Missing | nanoGPT supports DDP/NCCL; nanochat initializes distributed ranks and balances total batch; MLX-LM averages gradients across its distributed group [G1, G3, G4] | **L5** — after the single-device batch/dtype/checkpoint path is stable |

The MLX-LM trainer is the closest systems reference for the project's framework: it compiles the train step, supports gradient accumulation, periodically evaluates, saves adapter weights, and can clear cache above a threshold (`mlx_lm/tuner/trainer.py:22–24, 43–84, 221–260, 285–332, 367–380`) [G4]. Its saved adapters do not include a full optimizer/data-loader resume state. L3 does less so that its handwritten AdamW, clipping, and LR code stay inspectable; L3's `TrainingEngine.step` makes that choice explicit (`scripts/train.py:157–180`).

## 5. Put L1 proxy and L3 model bpb on a common footing

The `4.19`/`4.69` L1 values are uniform-ID tokenizer proxies, while L3's `2.5234 bpb` is learned heldout next-token loss after 500 updates: `4.1752 nats/token × 0.418925 tokens/byte ÷ ln(2)`. They use different probability assumptions, and the lower L3 number cannot be read as a direct tokenizer win ([L3 metrics](../results/l3/train-metrics-500.json), [L1 comparison](../runs/l1-diff-analysis.md)).

An equal-parameter follow-up is feasible at this scale but carries a substantial architecture caveat:

1. Fix the validation split once on raw UTF-8 bytes, then encode that exact heldout span with the existing 1028-ID tokenizer and `cl100k_base`; report each token count, tokens/byte, and heldout byte count. Do not split after tokenization, since token boundaries differ.
2. Compare two tied-embedding GPTs under the same approximately 13.1M parameter budget, four layers, byte-identical training/heldout text, optimizer-step budget, and evaluation procedure. Keep L3 at `d=512,h=8`; for the 100,277-ID cl100k vocabulary, `d=124,h=2` gives about `13.175M` parameters (`100277×124 + 4×(12×124²+4×124)+2×124`), within `0.43%` of L3. The smaller width is required because the cl100k embedding/output table alone would be `51.3M` parameters at `d=512`.
3. Calculate bpb as total heldout next-token NLL in nats divided by `ln(2) × raw heldout bytes`; also report total estimated training FLOPs from the per-step token counts and model dimensions. This keeps the outcome byte-normalized while making the unequal compute cost visible.

This test is possible without large-scale training, but the equal-parameter cl100k model spends about `94%` of its parameters on the vocabulary table and has a much narrower Transformer body. The result would compare tokenizer-plus-model systems under a fixed parameter budget, not isolate tokenizer quality. A second equal-backbone comparison would instead use `d=512` for both, but the cl100k model would be about `63.9M` parameters; its bpb and training cost must be labeled separately. I did not train either comparison model or regenerate L1's proxy.

## External source locations

- **[G1] nanoGPT**, pinned `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`: [`model.py` attention, MLP, pre-LN block](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L29-L106); [`estimate_mfu` with the `6N + 12LHTQ` formula](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L289-L303); [`train.py` defaults, dtype, checkpoint, accumulation, autocast, cosine and DDP](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L48-L74) and [training loop](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L158-L202) / [LR and update loop](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L230-L336); [same-scale Shakespeare recipe](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/config/train_shakespeare_char.py#L1-L37).
- **[G2] nanoGPT mini Shakespeare config**, same pinned revision: [size, batch, LR, beta₂, decay, and warmup at `config/train_shakespeare_char.py:1–37`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/config/train_shakespeare_char.py#L1-L37).
- **[G3] nanochat**, pinned `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`: [`scripts/base_train.py` arguments/defaults](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L55-L77), [distributed init](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L83-L90), [resume state and dtype handling](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L153-L162) / [GradScaler](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L318-L326), [training horizon and LR warmdown](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L338-L370), [batch reference and scaling](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L270-L315), [per-parameter-group AdamW/Muon betas](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L419-L454), [gradient accumulation](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L406-L413) / [checkpoint and update](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L476-L540).
- **[G4] MLX-LM trainer**, pinned `72be8c955d9a0d3099bd01865af288e0e947c3fa`: [`trainer.py` cache handling and gradient checkpointing](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/tuner/trainer.py#L22-L40), [training options](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/tuner/trainer.py#L43-L84), [compiled train step, distributed gradient averaging and accumulation](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/tuner/trainer.py#L221-L260), [evaluation and cache clearing](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/mlx_lm/tuner/trainer.py#L285-L332).
- **[M1] MLX lazy evaluation**: [MLX 0.32.4 documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html#lazy-evaluation), especially graph recording/evaluation and memory discussion.
- **[M2] MLX memory APIs**: [`get_active_memory`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html), [`get_cache_memory`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html), [`get_peak_memory`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html), and [`set_cache_limit`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html).
- **[T1] cl100k vocabulary configuration**: [OpenAI tiktoken `openai_public.py`, pinned L1 reference](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py#L75-L92).

## Execution report (`-o`, specification §6)

### Five conclusions

1. Correcting 6ND with the attention sequence term raises the 500-step MFU from `33.09%` to about `35.7%`; the 4096-token update supplies a substantial aggregate workload to dense batched projections (`16×256` rows per update).
2. The 2618.36 MiB run contains `200.17 MiB` of nominal parameter/gradient/Adam states plus a `2418.19 MiB` non-parameter high-water; forward-only and batch/layer ablations show substantial saved activations and backward/operator temporaries, while cache clearing does not reduce the peak counter.
3. `wd=.1`, `β₂=.95`, and a 10% cosine floor match nanoGPT's main recipe, but L3's 20% warmup is much longer than the close-scale nanoGPT recipe's 2%; current nanochat uses a different optimizer and linear warmdown.
4. Gradient accumulation, lower precision, full-state resume, and validation-driven stopping are absent or partial; the cosine decay exists, while distributed training belongs after single-device memory and restart behavior are stable.
5. L1 uniform-ID `4.19/4.69` and L3 learned-model `2.5234 bpb` are not the same probability metric; a fixed raw-byte heldout plus near-equal-parameter models is feasible, but cl100k's vocabulary would force a narrow body.

### Key numbers

- Corrected 500-step FLOPs: `84.947 MFLOP/token` explicit linear+attention, or `85.002 MFLOP/token` with the nanoGPT-style 6N approximation; corrected MFU `35.72–35.74%`.
- Main peak: `2618.36 MiB`; four fp32 states: `200.17 MiB`; remainder: `2418.19 MiB` high-water. The separate 3-update full ablation reached `2668.3 MiB`, forward-only `1492.6 MiB`, B=8 `2018.8 MiB`, L=2 `1963.3 MiB`.
- L3 warmup ratio: `20%` at 500 steps (`50%` in the original 200-step record); L3 heldout bpb at 500 steps: `2.5234`.

### External references

Pinned source revisions and exact code/doc locations are listed above in [External source locations](#external-source-locations). Local numeric evidence is in `results/l0/hw-profile-20261021T065826Z.json`, `results/l2/attention-bench-20261021T094129Z.json`, `results/l3/train-metrics.json`, `results/l3/train-metrics-500.json`, and `results/l3/memory-ablation.json`.

### Deviations from the task specification

- Added the permitted `bench/l3_memory_ablation.py` and `results/l3/memory-ablation.json`; no files under `layers/`, `scripts/`, or `tests/` were modified, no dependency was added, and no large tokenizer-comparison training was run.
- The memory ablation runs three updates per setting in one isolated process each and has no independent repetitions/quantiles. Its full-training peak is `49.9 MiB` above the committed full-run peak; it is used for directional group attribution, not as an exact decomposition of the 200/500-step run.
- No test suite was run, as required by the acceptance section. The ablation command itself completed successfully. No push was performed.
