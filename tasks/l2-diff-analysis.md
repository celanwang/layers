# 任务规格：l2-diff-analysis（对照精读）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。分析任务，**不修改任何实现/测试代码**。

## 1. 研究问题

自研 naive 注意力与参照实现（nanochat / minGPT / FlashAttention 论文 / MLX `mx.fast`）的设计分歧与取舍；用 L0 带宽基线定量解释"为何 fast 内核全尺度胜出"（推翻原理笔记的交叉点预言），以及 softmax 占前向 ~40% 与内核融合收益的因果关系。

## 2. 改动白名单

- `runs/l2-diff-analysis.md`（主产物）
- `results/l2/`（补充对照实验数据）
- `bench/l2_*_ablation.py`（如需消融，可新增）

其余一律禁止，特别是 `layers/`、`tests/`。

## 3. 禁止事项

- `git push`、删文件、改实现/测试代码、改既有测量协议、新依赖
- 泛泛文献综述——每论点必须有证据（数字或具体源码/论文位置）

## 4. 实现要求

允许并需要阅读参照：nanochat / minGPT 注意力实现、FlashAttention 论文（v1 即可）、MLX `mx.fast.scaled_dot_product_attention` 公开文档/头文件注释、RoPE 原论文。源码不入库。

分析点（每点给证据）：

1. **naive vs fast 差距机制**：用 L0 实测带宽（~450 GB/s）与 n² 中间矩阵物化的字节账，定量估算各 n 下 naive 实现的 IO 下界，与实测 2.1×→9.2× 的差距对照；解释为何 n=128 时 fast 已快 ~2×（推翻原理笔记"小 n 差距有限"预言的机制层面原因）。
2. **FlashAttention 的核心思想**：tiling + online softmax 如何消除 n² 物化；这与 L2 实测"softmax 占 ~40%"的关系——融合后这部分开销去了哪。
3. **RoPE 配对约定**：交错（GPT-J 风格）vs 前后半（GPT-NeoX 风格，`traditional=False`）的数学等价性与生态分歧；自研选交错是否影响与生产模型的权重互换。
4. **多头实现形态**：自研的 reshape/transpose 布局 vs nanochat/minGPT 的布局选择；对 cache 友好性与 MLX 的影响。
5. **缺失件清单**：对照 nanochat 层设计，列出自研注意力还缺什么（dropout、scale 可选化、KV cache 接口、GQA/MQA 扩展点），标注哪些留给 L3/L4。

## 5. 验收命令

无代码改动，不跑测试套件。验收标准：`runs/l2-diff-analysis.md` 存在，5 个分析点齐全，第 1 点有定量字节账与实测对照，每论点附证据位置。

## 6. 汇报要求

`-o` 文件写入：五点结论各一句话、第 1 点的字节账关键数字、外部资料位置清单、与规格的偏差。完成后提交（如 `docs(l2): 对照精读分析`），**不要 push**。
