# L4 复盘：推理与 mini 推理引擎（对照式自研第 5 步）

> 证据链：`tasks/l4-decode.md`、`tasks/l4-paged-engine.md`、`tasks/l4-diff-analysis.md`、`runs/l4-inference-principle.md`、`runs/l4-implementation-notes.md`、`runs/l4-diff-analysis.md`、`results/l4/`。

## 结论

五步闭环完成。推理路径（KV cache + 采样 + 流式 UTF-8）与 mini 引擎（分页 KV 池 + fork CoW + 连续批处理调度）落地，60 项测试全绿（含零串话、调度等价性、refcount 复原），mini-vLLM 开放问题闭环。

## 原理预言 vs 实测

| 预言 | 实测 | 判定 |
|---|---|---|
| decode 纯内存受限，roofline ~8.5k tok/s，实测远低于上限 | 326 tok/s = 名义上限 **3.7%**（小任务带宽口径 7.4%） | 方向**成立**；但"launch 开销主导"无法实测归因（Metal capture 不支持），降级为机制解释 |
| 批处理吞吐随并发提升 | 连续批处理仅 **1.04×**，且含共享 prefill 混淆 | **不成立**于 mini 形态；收益分解为调度无气泡 + kernel batching，后者缺失是主缺口 |
| 分页内部碎片 ≤ block_size/2 | N=1/4/8/16 全部低于上界 | **成立** |
| （未预言）KV cache 收益 | ~1.0×（1.098/0.968 两次横跨） | 意外：cache 在小模型短上下文**无收益**——权重读取是每步固定成本，免重算省不出时间；转正条件 n≈128–256 已推导，待验证 |

**认知修正**：decode 的真实瓶颈既不只是带宽，也无法在当前环境证明是 launch——75 个源码操作组的静态核算 + 0.23ms 权重读取 vs 3.18ms 实测的差距（2.94ms）只能归因到"小算子低利用率 + dispatch + 同步"的组合。**诚实的性能分析要知道哪些结论超出了测量能力**——本轮分析层两次推翻规格/直觉的过度归因（"1.04× 来自无气泡调度"、"launch 主导"），是本迭代最值钱的部分。

**mini-vLLM 差距定量界定**：kernel batching 的权重摊薄理论上限 8×（420MB→52.5MB/迭代，带宽上限 34.1k tok/s），现实投影 ~3k tok/s；我们的 gather+逐请求形态与 vLLM kernel 直读页表的差距已文档化。

## 负结果（保留）

- KV cache 零收益对照（两次原始记录 1.098×/0.968× 全保留）。
- Metal kernel trace 不可得的测量限制说明。

## 开放问题（带入后续）

1. KV cache 量化（INT8/4bit KV）→ **L5**。
2. kernel batching、prefill/decode 混排、eviction 策略 → **L6 服务化**。
3. cache 转正条件的配对测量（n≥256 同长度 cached/uncached）→ 专项小实验，可挂 L5。
4. Metal kernel trace：环境限制（`mx.metal` capture 不支持、无 xctrace），挂起待环境条件变化。

## L5 选题建议

**量化 / 混合精度**（原候选序列确认）：权重量化（INT8/4bit）+ KV cache 量化 + bf16 训练对照（接 L3 缺失件），测量精度损失 vs 吞吐/内存收益的 trade-off 曲线——量化同时接住了训练（L3）与推理（L4）两条线的遗留件。
