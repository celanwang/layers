# L2 复盘：嵌入 + RoPE + 多头因果注意力（对照式自研第 5 步）

> 证据链：`tasks/l2-attention.md`（规格 v2）、`runs/l2-attention-principle.md`、`runs/l2-implementation-notes.md`、`runs/l2-diff-analysis.md`、`results/l2/`。

## 结论

五步闭环完成。38 项测试（含 PyTorch 逐中间量 1e-5 对齐、模型级因果性探针、RoPE 相对位置性质）全绿；naive/fast、KV、softmax 三组测量两份记录一致。

## 原理预言 vs 实测

| 预言 | 实测 | 判定 |
|---|---|---|
| naive vs fast：n≤1024 差距有限，n≥2048 拉开 | fast **全尺度胜出**（1.9×@128 → 9.2×@4096），无交叉点 | **推翻**（大 n 趋势方向正确） |
| KV 随 n 线性增长（2·n·d·dtype 字节） | fp32/bf16 全尺度实测/理论 = 1.000× | **成立**（最精确的一次预言） |
| softmax 两次全行读写，小 n 下占比可观 | n=1024 占整层前向 ~40% | **成立** |

**认知修正**（本轮最重要）：注意力 naive 路径算术强度 ≈ 16 FLOP/byte，低于 L0 平衡点 32 → roofline 正确预言了"内存受限"，但字节账只能解释大 n 下 naive 延迟的 ~28%；n=128 时 fast 快 2× 与带宽无关（IO 账仅 4.65 μs，差值 0.2 ms），机制是**融合减少 kernel/调度固定成本**。教训：roofline 判趋势，不预测小尺寸的 dispatch 开销；预言要写数值阈值（"有限"未量化是笔记缺陷，执行层指出得对）。

**关联闭环**：softmax 占 ~40% 与 FlashAttention 的关系在对照精读中打通——融合消除的不是 softmax 算术，而是 n² 中间矩阵的物化与阶段边界；MLX fast API 只返回输出不返回注意力权重，正是该设计的接口投影。

**工程发现**：RoPE 交错（GPT-J 风格）与前后半（GPT-NeoX 风格）配对经通道置换数学等价，但 checkpoint 不能在不重排 Q/K 通道时互换——自研选交错，与生产模型权重互换时需置换，已记入分析文档 §3。

## 负结果（保留）

- 原理笔记"小 n 差距有限"被推翻，机制归因（dispatch 开销）目前无 kernel trace 证据，如实标注为"机制解释而非实测拆分"。

## 开放问题（带入后续）

1. n=128 fast 2× 的 dispatch 归因需 kernel trace/instrumentation → **L4 推理**迭代。
2. 对照 nanochat 的缺失件清单：dropout、KV cache 接口、GQA/MQA 扩展点 → **L3 已承接组装**、**L4 承接 KV cache**。

## 对 L3 的输入

L3 已在 `l3-training` 分支（worktree `layers-wt-l3`）并行推进：T1 组装（PyTorch 梯度级对齐）、T2 训练循环（手写 AdamW + MFU/内存/模型 bpb 测量）。
