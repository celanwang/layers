# L7 原理笔记：decode host 路径直读化（性能专项）

> 目的：检验全项目贯穿性假说"本机小负载性能由 host 端小操作与同步成本主导"。这是对照精读（`runs/l6-diff-analysis.md` §1）指出的唯一未证伪的关键假说——五次佐证均为有界解释，无 Metal trace 证据。本专项用**消除已知 host 开销后的吞吐增量**做假说的判决性实验。

## 1. 问题陈述

L6-T1 拼批后，dense 投影图已共享（24 个 block 投影 + 2 个半宽 head GEMM，不随 B 增长），但 B=8 仅达 4.115×（1436 vs 349 tok/s）。端点模型 T(B) = 1.641 + 0.5022B ms：B=8 时行相关项 4.017 ms 占 71%。行相关项里已定位的三类 host 开销：

1. **KV 重建**（`_BatchedPagedKVCache.append`，layers/engine.py:290-307）：每个 forward 每层对 B 行逐行 `pool.append`，而 `pool.append` 内部 `gather` 把该行**全前缀**从 block 拼回连续张量（Python 级 concatenate 循环），再逐行 pad+concat。B=8、4 层 ⇒ 每 forward 32 次 append + 32 次隐式全前缀重建，每次 O(L) 拷贝。
2. **采样标量栅栏**（`Sampler.__call__`，layers/inference.py:174-181）：每请求每步 `mx.eval(token)` + `.item()` 一次主机同步。B=8 ⇒ 每步 8 次串行同步栅栏。
3. **padding 浪费**：已被对照精读证伪为次要因素（attention matmul 仅占 FLOP 3.84%），不是本专项目标。

## 2. 原理：为什么 gather 在数学上不必要

Softmax 对 key 集合是可分解的：把 K 按 block 分组，各 block 独立算 scores，用 running max/sum 重缩放即可在线归并（online softmax，与 FlashAttention/PagedAttention 同一数学）。PagedAttention 论文 §5 的 kernel 直接按 block table 读页，永不重建连续前缀。**但本机约束不同**：MLX 无自定义 Metal kernel，逐 block Python 循环反而增加 host 操作——正是我们要消除的东西。

因此教学形态的正确问题是：**如何用向量化物化拿到同样的渐近效果**——把"逐行 Python gather（B 次、每次 O(L) 拷贝、跨多次调用）"换成"按 block table 一次扁平索引（单 kernel）"。掩码在 block 粒度上做（补齐到 block 倍数而非行最大长度只是常数因子差异，关键是不再有 B 次 Python 级前缀重建）。

采样同理：RNG 语义必须逐请求隔离且与既有实现位精确一致（`mx.random.seed(seed+draws)` 后 `categorical`），但 8 次 `eval+.item()` 可以合并为**一次构图、一次 eval、一次批量主机传输**——消除的是同步栅栏数，不是计算。

## 3. 接口设计（草图，实现细节属盲做范围）

- `PagedKVPool` 加法式扩展：按层的扁平物化路径，输入 block table 集合，输出 `(k_all, v_all)`（block 对齐 padded）+ 有效长度；单次向量化索引，无逐行 Python gather。
- `CausalSelfAttention` 加法式扩展：`decode_batch` 的直读变体，消费上述物化结果，掩码用有效长度构造。
- 采样：批式入口，逐请求 RNG 流与参数（seed/temperature/top_k/top_p）隔离保持不变，整批一次同步。
- `BatchedEngine` 数值路径经由新组件时，对外语义（事件、序、容量错误）不变。

## 4. T1/T2 拆分（归因需要）

- **T1 直读 page attention**：只换 KV 物化与 attention 消费路径，采样不动。
- **T2 批式采样**：在 T1 之上，只换采样同步结构，KV 路径不动。
- 两者独立可测（各自 bench 消融），归因到组件。

## 5. 预言

以 L6 同协议（3 预热/10 重复、配对交替、config hash、原始样本保留）测量，判决标准：

1. **T1**：B=8 decode 窗口从 356.4 ms 降至 ≤ ~300 ms（聚合吞吐 1436 → ≥ 1700 tok/s）。若 <1550，说明 KV 重建不是行相关项主体。
2. **T2**：在 T1 之上再降 ~5–10%。若 <3%，说明标量栅栏不是瓶颈。
3. **合并判决**：B=8 聚合 ≥ 1800 tok/s（≥1.25×）→ "host 小操作/同步主导"假说获得直接支持；< 1600（<1.12×）→ 假说被削弱，需回到 Metal trace 重新归因。
4. **数值不变式**：与 L6 `BatchedEngine` 逐 token 一致、logits atol 1e-5（黄金不变式的批次版延续）。

## 6. Oracle 方案

- 正确性：新路径 vs 既有 `BatchedEngine`（不变），同 seed 同参数逐 token 相等 + logits 1e-5；既有 119 项测试原样通过。
- 性能：L6 bench 同协议消融（T1-only / T1+T2），与 `results/l6/batching-bench-20261021T133044874890Z.json` 直接可比。
- 盲做纪律：禁止阅读 vLLM/FlashAttention kernel 源码；PagedAttention 论文 §5（算法层面）可读。生产对照留给对照精读步骤。

## 参考资料（纸面）

- PagedAttention 论文 §5（block table 直读、online softmax 归并思想）
- `runs/l6-diff-analysis.md` §1（行相关项定位、端点模型、证据边界）
- `results/l6/diff-accounting.json`（派生账与输入哈希）
