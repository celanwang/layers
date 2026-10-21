# L2 实现记录：嵌入 + RoPE + 多头因果注意力

## 实现要点

- `TokenEmbedding` 将 `[vocab_size, d_model]` 的 MLX 权重表按 token ID gather；L1 词表的总 ID 空间为 1028。
- `CausalSelfAttention` 采用无 bias 的输入维在前权重矩阵，按 head 拆分，先对 Q/K 应用 RoPE，再手写缩放点积、上三角 `-inf` 掩码、稳定 softmax、`P @ V` 与输出投影。
- `forward_with_intermediates` 暴露原始 Q/K/V、旋转后的 Q/K、scaled scores、概率和输出，便于逐中间量验算；普通调用仍返回 `(out, attn_weights)`。
- 所有权重在初始化时以 `mx.random.seed(42)` 固定种子；矩阵权重按 `1/sqrt(d_model)` 缩放，Embedding 权重按 0.02 缩放。

## 卡壳点与假设

- MLX 的 RoPE 公共 Oracle 默认使用跨半维配对；本实现按原理笔记使用相邻维交错配对，因此对照使用 `mlx.nn.RoPE(traditional=True)`。
- RoPE 的 90° 特例通过 `offset=π/2` 验证旋转几何；常规 token offset 仍为整数。
- 原理笔记未指定投影 bias、权重分布或输入 batch 接口。本实现采用无 bias、seeded normal 权重、支持 `[n,d_model]` 与 `[batch,n,d_model]`；`d_model` 必须等于 `n_head*d_head`。
- 因果注意力按 self-attention 设计，要求 Q/K 序列长度相同；未来位置由 `mx.where` 写为 `-inf`，对角线保留。

## 与原理笔记的关系

- 保持“先缩放，再加因果约束，按行减最大值后 softmax”的顺序，且 RoPE 作用在每个 head 的 Q/K 上。
- benchmark 把 naive 与 fast 的 Q/K/V 核心算子作同输入对照；KV 曲线同时记录 MLX 活跃内存增量与理论字节数；softmax 比例是独立组件时间除以完整层前向时间的粗粒度估计。
- 结果和环境配置哈希保存在 `results/l2/`，包含本机信息、预热次数、重复次数和 median/p90。
