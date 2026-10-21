# L2 对照精读：自研注意力与参照实现

## 范围与测量口径

本报告按 `tasks/l2-diff-analysis.md` 的五个分析点对照自研注意力、nanochat、minGPT、FlashAttention v1 与 MLX 0.32.4。只阅读实现和资料，没有修改 `layers/`、`tests/` 或测量协议；本次只新增本报告。

L2 attention benchmark 输入为 `B=1, H=8, d_head=64`，attention dtype 为 fp32，序列长度为 128–4096；每次记录预热 3 次、计时 10 次，保留 median、p90、min。两份记录的配置哈希分别是 `02a1f2c5f29b` 和 `5a1c15a53469`，硬件均为 Apple M4 Max / 48 GiB，macOS 15.8、Python 3.12.15、MLX 0.32.4。哈希不同是因为记录还包含不同的 Git commit；形状、数据类型、预热和重复次数相同。原始数据见 [`attention-bench-20261021T093939Z.json`](../results/l2/attention-bench-20261021T093939Z.json) 与 [`attention-bench-20261021T094129Z.json`](../results/l2/attention-bench-20261021T094129Z.json)。

L0 使用同机 fp32 `add` 在最大工作集 `2^28` 上测得 450.6 GB/s median（预热 3 次、计时 10 次；config hash `ccb4c0a70605`）。该操作的带宽口径是两个输入读取加一个输出写入，见 [`L0 JSON`](../results/l0/hw-profile-20261021T065826Z.json) 和 [`L0 运行记录`](l0-hw-profile.md)。下面用它估算“若两个完整 fp32 的 n×n 矩阵跨该带宽写回并再次读取，需要多少时间”；这是可比用的流量预算，不是 L2 实测 DRAM 计数，也不是对 M4 缓存带宽的硬件下界。

## 1. naive 与 fast 的差距机制

**结论：** dense naive 路径至少产生二次规模的 score/probability 中间量读写，L0 带宽估算从 n=128 的 4.65 μs 增至 n=4096 的 4.77 ms；测量中 fast 两次运行在每个长度都更快（1.92×–9.16×），小长度的领先主要不能由大矩阵带宽解释。

### 字节账

设 score 矩阵 `S=QKᵀ`、softmax 概率矩阵 `P=softmax(S)`，每个长度有 `B×H×n² = 8n²` 个 fp32 元素。若 `S` 和 `P` 都完整物化，单元素的乐观读写账是：写 `S` 4 B + 读 `S` 4 B + 写 `P` 4 B + 读 `P` 4 B = **16 B/score cell**。这只计算两张矩阵各一次写和一次读；不含 Q/K/V、输出、mask、稳定 softmax 的额外扫描、框架临时量或投影。

| n | score cells（8n²） | S 或 P 单矩阵 | S/P 读写估算（16 B/cell） | L0 450.6 GB/s 时间 | naive median（记录 1 / 2） | naive/fast median（记录 1 / 2） | IO 估算 / naive |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 131,072 | 0.5 MiB | 2 MiB | 0.00465 ms | 0.361 / 0.392 ms | 1.92× / 2.14× | 1.29% / 1.19% |
| 256 | 524,288 | 2 MiB | 8 MiB | 0.01862 ms | 0.488 / 0.503 ms | 2.06× / 2.07× | 3.82% / 3.70% |
| 512 | 2,097,152 | 8 MiB | 32 MiB | 0.07447 ms | 0.476 / 0.990 ms | 2.26× / 3.82× | 15.66% / 7.52% |
| 1024 | 8,388,608 | 32 MiB | 128 MiB | 0.29786 ms | 1.085 / 1.530 ms | 3.21× / 3.56× | 27.46% / 19.47% |
| 2048 | 33,554,432 | 128 MiB | 512 MiB | 1.19146 ms | 4.362 / 4.394 ms | 6.40× / 6.61× | 27.31% / 27.12% |
| 4096 | 134,217,728 | 512 MiB | 2,048 MiB | 4.76583 ms | 16.953 / 17.034 ms | 9.09× / 9.16× | 28.11% / 27.98% |

例如 n=4096 时，每张矩阵是 512 MiB，`S` 与 `P` 一写一读合计 2 GiB；以 L0 的 450.6 GB/s 折算为约 4.77 ms。naive attention 核心实测约 16.95–17.03 ms，而 fast 约 1.86 ms。估算的矩阵流量足以解释 naive 延迟中约 28% 的量级，不能单独解释完整 9.1× 差距；剩余差异还包含 softmax 扫描、mask、算子/临时量与内核路径差异。L0 与 L2 的数据规模和算子不同，因此这里不把剩余时间反推成某个具体硬件瓶颈。

在 n=128 时，完整 `S/P` 读写账只有 2 MiB，对应约 4.65 μs，低于两次记录的 naive/fast 差值（约 0.173–0.209 ms）近两个数量级。这个尺寸下 fast 已达 1.92×–2.14×，因此不能将领先归因于“吞吐完 2 MiB 的内存流量”；更一致的机制是 fast 路径把多个操作组合到更少的 kernel/中间量步骤，减少小任务的调度与框架开销。L0 本身也显示小工作集带宽为 224.1 GB/s，而最大工作集为 450.6 GB/s，并将小尺寸塌缩归因于启动/调度开销；但 L2 没有 kernel dispatch trace，故对 n=128 的具体开销归因属于机制解释，不是单独测出的耗时拆分。自研代码的 `QKᵀ → mask → stable softmax → P@V` 顺序见 [`attention.py`](../layers/attention.py#L42-L61)，对照 benchmark 路径见 [`l2_attention_bench.py`](../bench/l2_attention_bench.py#L114-L145)。

naive attention 仅按 QKᵀ 和 PV 计约 `4n²Hd_head` FLOPs；除以上述最低矩阵读写量 `16n²H` bytes，算术强度为 `d_head/4 = 16 FLOP/byte`，低于 L0 约 `14.75 TFLOP/s ÷ 450.6 GB/s ≈ 32.7 FLOP/byte` 的平衡点。这个简化 roofline 预测支持中长序列下内存流量重要，但不能预测 dispatch 成本和缓存命中。

两份记录均显示六个长度点 median 上 fast 全胜；p90 也全胜：记录 1 的 naive/fast p90 比值为 2.12×–9.01×，记录 2 为 2.41×–8.79×。512/1024 的 naive 中位数在重复记录间有明显波动（512 从 0.476 ms 到 0.990 ms，1024 从 1.085 ms 到 1.530 ms），故以“全长度都更快”和总体趋势为结论，不把单个中间点的精确比例解释成稳定的硬件常数。L2 结果与原理笔记“n≤1024 差距有限”的预言不符；可解释的修正是：相对加速比还会受多阶段算子固定成本影响，小尺寸也可能因融合与调度减少而明显领先。L0 的 450.6 GB/s 字节账解释了随 n 增大的物化代价，但不是 n=128 已经 2× 的原因。

证据位置：L0 带宽和协议见 [`results/l0/hw-profile-20261021T065826Z.json`](../results/l0/hw-profile-20261021T065826Z.json) 的 `bandwidth[2]`（以及 `bandwidth[0]`）；L2 median/p90、shape、dtype、协议及 config hash 见上述两份 JSON 的 `attention_kernels`、`config`、`device` 字段；naive 算子次序见 [`layers/attention.py`](../layers/attention.py#L42-L61)。

## 2. FlashAttention：tiling、online softmax 与 ~40% softmax

**结论：** FlashAttention 将 Q/K/V 分块，把 score tile 留在片上并以运行中的行最大值、归一化和输出累积完成精确注意力；softmax 的指数与归一化仍要计算，但独立的 n² 概率矩阵及其全局读写和独立阶段被消除或压缩进融合 kernel。

FlashAttention v1 §3.1 Algorithm 1 将 `Q` 分为 query blocks、`K/V` 分为 key blocks，在片上计算局部 `S_ij=Q_i K_jᵀ`。对每一行保留运行最大值 `m`、归一化和 `ℓ` 与未归一化输出 `O`；新 tile 到来时先更新 `m_new=max(m_old,rowmax(S_ij))`，再按指数尺度重标旧状态，并把当前 tile 的 `exp(S_ij−m_new)V_j` 累加到输出，最后除以更新后的 `ℓ_new`。这与全量 softmax 数学等价，只有浮点累加顺序不同。论文指出完整 attention `S/P` 不写回 HBM，前向保留 `O,m,ℓ`，反向按块重算；同时把矩阵乘、softmax、mask/dropout（若启用）和第二次矩阵乘融合在 kernel 内（§3.1 与 Algorithm 1）。

MLX 的公开 API 是 `mx.fast.scaled_dot_product_attention`，要求传 `scale`、可用 `mask="causal"`，支持 GQA/MQA；0.32.4 文档说明可以在 heuristic 之外用 `force_fused=True` 强制 fused kernel。0.32.4 Metal 源码包含 full-attention Steel 路径，并使用 `bq/bk` 分块配置和按 query block dispatch。L2 benchmark 没有记录具体 kernel dispatch；因此此处将测得的 `mx.fast` 视为 MLX 优化 attention 路径，并将 tile/fusion 作为其公开接口和对应版本源码所示的实现机制，不声称它就是 FlashAttention v1 的同一 CUDA 算法或逐行照搬。

L2 的 n=1024 粗粒度测量中，单独 `_stable_softmax` median 为 0.662 ms / 0.676 ms，整层前向为 1.696 ms / 1.666 ms，比例是 39.05% / 40.60%。这说明 naive 端将 softmax 单列计时，确实占据了约四成整层时间；对照点 `score_and_mask` 为 0.424 / 0.432 ms，`P@V` 为 0.266 / 0.250 ms。这里 softmax 时间是独立组件 median 除以 full-forward median 的粗测，且 full forward 还包含投影、RoPE 等操作，因此这些组件不可相加成严格分解，也不能推出融合恰好带来 40% 提速。因果关系在算法层面是：融合后 softmax 的 `max/exp/sum/divide` 在 tile 内或随 tile 状态执行，不再将完整 `P` 物化并供另一独立 PV kernel 读取；softmax 算术仍在，只省去中间矩阵交通和阶段边界。MLX fast API 只返回 `O`，与不保留完整注意力权重相一致。

证据位置：FlashAttention v1 论文 §2.2、§3.1 Algorithm 1 与前向/反向讨论；[`L2 benchmark`](../results/l2/attention-bench-20261021T093939Z.json) 的 `softmax_fraction` 和第二次复跑对应字段；[`bench/l2_attention_bench.py`](../bench/l2_attention_bench.py#L180-L209) 明确记录了 `coarse isolated component timing`；MLX API 文档的 `force_fused`、scale、mask 与输出参数，以及 0.32.4 Metal SDPA 源码位置列在文末资料表。

## 3. RoPE 配对约定与权重兼容

**结论：** 交错配对与前后半配对是对同一组二维旋转平面的不同通道排列；一致置换 Q/K 投影输出后，两者数学上等价，但权重文件不能在未做 Q/K 通道重排时直接互换。

自研实现按 `(x₀,x₁),(x₂,x₃),…` 相邻维配对，对应 [`RotaryEmbedding`](../layers/attention.py#L21-L39) 的 even/odd 切片与 `mx.stack`，Oracle 使用 `mlx.nn.RoPE(traditional=True)` 复现该约定。MLX 0.32.4 文档将 `traditional=True` 定义为旋转连续相邻维。nanochat 当前源码则令 `d=head_dim/2`，将最后一维拆成 `x[..., :d]` 和 `x[..., d:]` 再旋转，属于前后半配对；其注释还指出旋转方向为教材约定的转置，并为 checkpoint compatibility 保留该方向。MLX LLM 模型配置里也可见 `rope_traditional=False`，所以生态里确实有两种配对布局，不应只靠“RoPE”名称假设一致。

形式化地，令 `P` 将交错表示 `[x₀,x₁,x₂,x₃,…]` 重排为 `[x₀,x₂,…,x₁,x₃,…]`。相同角频率/方向下，半分布局的旋转矩阵满足 `R_half(m)=P R_interleaved(m) Pᵀ`。若同时令 `q_half=Pq_interleaved`、`k_half=Pk_interleaved`，则 `q_halfᵀR_half(m)ᵀR_half(n)k_half = q_interleavedᵀR_interleaved(m)ᵀR_interleaved(n)k_interleaved`；内积和注意力 logits 不变，正是 RoFormer §3.2 对位置旋转以及相对位置内积性质的换基表达。

实际加载 checkpoint 时，若从交错实现转换为半分实现，应逐 head 重排 Q 与 K projection 的输出行（以及对应 bias 或其他按 Q/K 通道存储的参数）；若 QKV 合并打包，只重排 Q、K 区段。基本注意力结构里的 V 与输出投影不因这次 Q/K 基底置换而变。若不重排，Q/K 通道和 RoPE 旋转平面错位，logits 改变，权重虽能加载但计算行为不兼容。还需同时核对 base、频率缩放、offset、旋转方向、head_dim；配对置换本身不能修复这些差异。

证据位置：自研实现 [`layers/attention.py`](../layers/attention.py#L21-L39)；nanochat [`gpt.py` 的 RoPE 和 Q/K 应用](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L1380-L1394)；MLX 0.32.4 [`fast.rope` 文档](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html) 中 `traditional` 参数；RoFormer 论文 §3.2.1–3.2.2 Eq. 12–14 与 §3.3。

## 4. 多头布局：可读性、内存与缓存

**结论：** 自研与 minGPT 都把 attention 内核输入整理为 `[B,H,T,D]`，而 nanochat 为 FA3 和预分配 KV cache 保持 `[B,T,H,D]`；布局取舍影响转置、连续性和缓存写入方式，本轮没有测量其性能优劣。

自研先用三个独立投影生成 `[B,T,H,D]`，再 `transpose(0,2,1,3)` 成 `[B,H,T,D]`，执行 attention 后转回 `[B,T,H,D]` 并 reshape 拼头（[`layers/attention.py`](../layers/attention.py#L105-L125)）。这与 MLX fast SDPA 的公开形状 `[B,N_q,T_q,D]` 对齐，也便于 score 矩阵按 head 分组。自研保留 Q/K/V、scores、probabilities 等中间值以便 oracle 对齐，教学上易检查；没有像 minGPT 那样将 QKV 合成一次 projection。

minGPT 用单个 `c_attn` 线性层投出 QKV 后 split，reshape/transpose 为 `[B,nh,T,hs]`，计算完再 `transpose(1,2).contiguous().view(B,T,C)` 拼头。nanochat 则分别投影 Q/K/V，直接 reshape 为 `[B,T,H,D]`，源码注释称这是 FA3 native layout，因此 attention 前无需 head/sequence transpose；它的预分配 cache 也是 `[B,T_max,H_kv,D]` 并沿 token 轴写入。MLX 官方 Llama inference 示例采用 `[B,H,T,D]`，并在该轴布局下通过 cache 拼接 K/V（`axis=2`），说明生产系统可以按调用算子和缓存策略选择布局。

从数据连续性看，token-major `[B,T,H,D]` 让单 token 的不同 head 相邻，适合投影输出和按 token 追加的静态 cache；head-major `[B,H,T,D]` 让每个 head 的 token 序列连续，直接符合 MLX SDPA，适合分 head 做 attention。MLX 的 transpose 表达本身不能据此断言一定复制；后续 kernel 对 stride 的支持、reshape 是否物化、cache 是静态分配还是拼接，才决定实际数据搬运。L2 attention benchmark 的 Q/K/V 已经是 `[1,8,n,64]`，并未测投影、transpose、拼接或 cache 更新，所以本轮只能列出设计差异，不能宣称其中一种更 cache-friendly 或更快。

证据位置：自研 reshape/transpose [`attention.py`](../layers/attention.py#L105-L125)；minGPT [`model.py`](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L863-L913)；nanochat [`gpt.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L1426-L1496) 与 [`flash_attention.py` 的预分配 cache 形状](https://github.com/karpathy/nanochat/blob/master/nanochat/flash_attention.py#L790-L860)；MLX [`LLM inference attention 示例`](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html#attention-layer)。

## 5. 缺失件与后续归属

**结论：** 当前实现足以验证全量 causal MHA 的数值路径，但没有训练正则化接口、可配置缩放或增量推理接口；将 dropout/scale 扩展留给 L3，将 KV cache 与 GQA/MQA 留给 L4。

| 缺失件 | 自研现状与参照证据 | 建议归属 |
|---|---|---|
| Attention / residual dropout | 当前 attention 无 dropout；minGPT 在 softmax 后使用 `attn_dropout`、输出 projection 后用 `resid_dropout`（`model.py` L871-L875、L904-L913）。nanochat 的当前 `flash_attn_func` 调用也没有传 dropout 参数（`gpt.py` L1463-L1466）。 | L3：若训练协议采用 dropout，再纳入训练模式与可复现随机数语义。 |
| 可选 attention scale | 自研 [`attention_scores`](../layers/attention.py#L42-L54) 固定 `1/sqrt(d_head)`；MLX fast API 将 `scale` 作为显式参数，minGPT 同样在源码中写出缩放公式（`model.py` L898-L900）。 | L3：训练/model 配置需要自定义 logits 标度时扩展；当前 Vaswani 默认保持简单。 |
| KV cache / offset 接口 | 自研 `forward_with_intermediates(x)` 每次要求整段等长 Q/K，无 cache 参数；nanochat 的 `forward(..., kv_cache)` 会取 layer cache 并调用 cache-aware attention；MLX Llama 示例也实现 cache offset/拼接。 | L4：与 decode、缓存布局和生命周期管理一起实现。 |
| GQA / MQA 扩展点 | 自研令 `d_model=n_head*d_head`，Q/K/V 都是相同的 `d_model` 宽度（`attention.py` L67-L96）；nanochat 明确区分 `n_head`、`n_kv_head` 并约束整除关系；MLX fast SDPA 文档支持 query heads 多于 KV heads。 | L4：与 KV cache 的显存/内存带宽收益共同验证。 |

这份清单只标注后续层级，不启动 L3/L4 改动。nanochat 当前层同时集成 GQA、KV cache、RoPE、QK norm、滑窗与 FlashAttention/SDPA 选择；本次不将这些扩展移植进 L2，因为研究问题限定为可读的 naive 全量 MHA 与对照机制。

## 任务汇报（规格第 6 节 / `-o` 输出）

### 五点结论（每点一句）

1. dense naive 的 score/probability 两矩阵乐观物化账为 16 B/cell，L0 带宽折算从 n=128 的 4.65 μs 增至 n=4096 的 4.77 ms，而 fast 在两个记录的所有长度点都更快。
2. FlashAttention 的 tiling 与 online softmax 保留每行运行统计和输出，去掉完整 n² 概率矩阵流量；softmax arithmetic 仍执行，L2 单独计时占 full-forward median 的 39.05% / 40.60%。
3. 交错配对与半分配对可由通道置换互相换基，但跨布局加载权重必须重排每个 head 的 Q/K projection 输出通道。
4. 自研/minGPT attention 采用 `[B,H,T,D]`，nanochat 的 FA3 与 cache 路径采用 `[B,T,H,D]`，本轮不含布局/cache 测量，不能据此排序性能。
5. 当前缺少 dropout 与可选 scale（归到 L3），也缺少 KV cache 与 GQA/MQA（归到 L4）。

### 第一点字节账关键数字

以 `B=1,H=8,fp32`、S/P 每个 score cell 各写读一次估算：S 或 P 单矩阵大小由 n=128 的 0.5 MiB 增至 n=4096 的 512 MiB；两矩阵物化流量由 2 MiB 增至 2 GiB；按 L0 450.6 GB/s 对应约 0.00465–4.76583 ms。n=4096 naive median 16.953 / 17.034 ms，fast median 1.864 / 1.860 ms，分别为 9.09× / 9.16×；矩阵流量估算约占 naive 28%。

### 外部资料位置

- FlashAttention v1：论文 §2.2、§3.1 Algorithm 1、前向/反向的 `O,m,ℓ` 保存与重算说明：[arXiv HTML](https://arxiv.org/html/2205.14135)。
- RoFormer：§3.2.1–3.2.2 Eq. 12–14、§3.3 频率定义：[arXiv HTML](https://arxiv.org/html/2104.09864)。
- nanochat：RoPE、`(B,T,H,D)`、GQA、KV cache 调用：[gpt.py L1380-L1496](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L1380-L1496)；cache buffer shape 与 fallback：[flash_attention.py L790-L860](https://github.com/karpathy/nanochat/blob/master/nanochat/flash_attention.py#L790-L860)。
- minGPT：融合 QKV、显式 scale/mask/softmax/dropout 与 head 布局：[model.py L862-L913](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L862-L913)。
- MLX 0.32.4：SDPA 参数与形状、`force_fused`、GQA/MQA：[public API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)；RoPE `traditional` 参数：[public API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html)；Metal backend 的 tiled attention path：[scaled_dot_product_attention.cpp](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp#L3037-L3270)。
- MLX cache/layout 示例：[Llama inference attention](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html#attention-layer)。

### 偏差

- 无白名单偏差：只新增 `runs/l2-diff-analysis.md`；没有修改 `layers/`、`tests/`、既有测量协议或结果 JSON。
- 未新增对照实验数据：已有两份 L2 benchmark 与 L0 profile 足以完成规格要求的字节账和性能对照。
- 未运行测试套件，符合本任务“无代码改动，不跑测试套件”；文档变更将执行 `git diff --check` 并按阶段性纪律提交，不 push。
