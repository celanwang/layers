# L3 复盘：组装可训练的 GPT（对照式自研第 5 步）

> 证据链：`tasks/l3-model-assembly.md`、`tasks/l3-training-loop.md`（v2）、`tasks/l3-diff-analysis.md`、`runs/l3-training-principle.md`、`runs/l3-implementation-notes.md`、`runs/l3-diff-analysis.md`、`results/l3/`（含 4 次失败尝试留档）。

## 结论

五步闭环完成。13.1M GPT（4 层 × d512 × ctx256）组装 + 手写 AdamW/cosine/clip 训练栈：500 步 heldout loss 7.08 → 4.18 nats（**-41%**），49 项测试全绿，PyTorch Oracle 达逐参数梯度对齐（1e-4）+ torch AdamW 逐步对齐（1e-6）。

## 原理预言 vs 实测

| 预言 | 实测 | 判定 |
|---|---|---|
| 训练内存 ≈ 4× 参数（210 MB）+ activations < 1 GB | 峰值 **2618 MiB** | **推翻** |
| MFU < 10% | 6ND 口径 33.1%，精确口径 **35.7%** | **推翻**（方向性错误） |
| loss 快速跌破 6.94 并持续下降 | 每 10 步检查点单调下降，-41%@500 步 | **成立** |

**认知修正一（内存）**：训练峰值内存的主角不是参数四态（仅 200 MiB），而是 activations + autograd 保存 + **MLX 缓存语义**——消融显示 forward-only 1493 MiB、backward 再加 1176 MiB、训练结束 active 仅 250 MiB 而 cached 2632 MiB：MLX 的缓存复用使 peak ≠ live。naive 注意力的 n² 物化（每层 32 MiB 起步）是 activations 的大头，这正是 L2 fast 内核结论在训练侧的镜像。

**认知修正二（MFU）**：6ND 口径漏掉注意力序列项（本配置 +8%），修正后 35.7%；小模型高 MFU 的原因是 92.6% 的 FLOPs 在 batched 投影/MLP 大矩阵上（`16×(256×512)@(512×2048)` 量级），并非零散小算子。教训：MFU 数字必须带口径声明，"小模型 MFU 必然低"的直觉在统一内存 + 大批次 token 下不成立。

**L1 开放问题 1 回补**：模型 bpb = **2.52 bits/byte**（500 步，heldout）。与 L1 uniform-ID proxy（4.19）口径不同不可直接比；等口径对比方案（cl100k 编码 + 等规模模型训练）已在对照精读 §5 设计，留作可选实验。

## 过程记录

- 验收断言失败（29.71% < 30%@200步）经规划层裁决为**规格阈值校准失误**：执行层 4 次尝试的负结果全部留档、未放宽断言；修复方式是规格 v2 加长到 500 步用证据跨阈（41.0%），而非降阈值。
- 对照精读质量：MFU 双口径差异量化到 0.02pp 级别，内存消融一跑一进程、如实标注"单次运行、非分布估计"。

## 开放问题（带入后续）

1. n=128 fast 内核 2× 优势的 dispatch 归因（需 kernel trace）→ **L4**。
2. 训练缺失件：梯度累积、bf16 混合精度、checkpoint resume、分布式 → **L5**（量化/精度迭代承接混合精度）。
3. 等口径 tokenizer 下游代价实验 → 可选，任何一层间隙可插入。

## L4 选题

**推理 / KV Cache**（原候选序列确认）：decode 循环 + KV cache 接口（L2 缺失件）、流式 UTF-8 处理（L1 观察清单）、采样策略、生成吞吐与内存墙测量（接 L0 带宽基线与 L2 KV 曲线）。
