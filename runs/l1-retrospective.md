# L1 复盘：Tokenizer（对照式自研第 5 步）

> 规划层收官文档。证据链：`tasks/l1-tokenizer.md`（规格 v2）、`runs/l1-bpe-principle.md`（原理）、`runs/l1-implementation-notes.md`（盲做笔记）、`runs/l1-diff-analysis.md`（对照精读）、`results/l1/`（全部测量）。

## 结论

五步闭环完成。自研 byte-level BPE（vocab 1024，无预分词）通过 29 项测试与三重 Oracle 对齐，训练 1.0 MB/s / 编码 0.7 MB/s（median，纯 Python CPU）。

## 原理预言 vs 实测

| 原理笔记的预言 | 实测 | 判定 |
|---|---|---|
| 不做预分词 → 压缩率低于 regex 预分词方案 | vocab 1024 下 HF 有正则 **49,420** > 无正则 **46,683** tokens，预分词反而**多**切 5.9% | **推翻** |
| encode/train 吞吐应按内存受限任务对待（算术强度 ≪ 32 FLOP/byte） | 0.7–1.0 MB/s，距带宽上限 3 个数量级，Python 对象/堆开销主导 | **未证实**，瓶颈在解释器而非内存 |
| （未预言）盲做实现的正确性 | 自研与 HF `use_regex=False` 在同语料同词表下 token 数**完全一致**（46,683） | 意外收获：独立实现精确互证，强于"测试通过" |

**认知修正**：预分词首先是"防止跨词边界合并"的语义/工程决策，其压缩率收益依赖词表规模——小词表下限制合并自由度的代价为负。与 cl100k 的 15,214 token 差距分解：词表大小 **+15,976**（8192 词表实验证实是主因）、预分词 **−2,737**、残余 +1,975（训练分布/正则/rank 差异，不可再分）。

## 负结果（保留）

- HF 训练器 `use_regex=False` 整语料单序列卡死（>150s @ 99% CPU），终止后改用常规配置——见实现笔记。
- uniform-ID proxy 指标会掩盖真实压缩差距（小 id 空间占便宜），执行层主动披露而非邀功；tokenizer 对比应以 tokens/byte 为主指标。

## 开放问题（带入后续迭代）

1. tokenizer 对下游 loss 的因果影响，需模型侧 bits-per-byte（交叉熵）回答 → **L3 训练循环**迭代具备测量条件后回补。
2. encode 吞吐 0.7 MB/s 若在推理路径成为瓶颈（Rust/C 扩展 vs 维持纯 Python 教学性）→ **L4 推理**迭代再议。
3. 流式 decode 的不完整 UTF-8 片段处理 → **L4 推理**观察清单。

## L2 选题建议

**注意力（scaled dot-product + causal mask + multi-head）**。理由：token id 序列已就绪，需要可训练的前向路径来消耗；注意力是 transformer 的核心算子，Infra 测量面直接接 L0 基线（算术强度判据、KV 随序列增长曲线）。备选：嵌入层 + 位置编码（RoPE）并入 L2 作为前置小节。
