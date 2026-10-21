# 任务规格：l5-diff-analysis（对照精读）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。分析任务，**不修改任何实现/测试代码**。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l5-quantization`；Python 用 `.venv/bin/python`。

## 1. 研究问题

量化与混合精度的教学实现和生产实现的差距；定量解释两个实测反差点：bf16 单算子仅 1.1× 但训练端到端 +30%；INT8 KV 存储省 4× 但 decode 慢至 0.3×。

## 2. 改动白名单

- `runs/l5-diff-analysis.md`（主产物）
- `results/l5/`（补充对照实验数据）
- `bench/l5_*_ablation.py`（如需消融，可新增）

其余一律禁止（含 `layers/`、`scripts/`、`tests/`、既有 bench）。

## 3. 禁止事项

- `git push`、删文件、改实现/测试/脚本、新依赖
- 泛泛文献综述——每论点必须有证据（数字或具体源码/文档/论文位置）

## 4. 实现要求

允许并需要阅读参照：MLX `mx.quantize`/QuantizedLinear 与 mlx-lm 量化路径公开源码/文档、GPTQ 与 AWQ 论文、vLLM FP8 KV cache 公开资料、PyTorch AMP 文档。源码不入库。

分析点（每点给证据）：

1. **bf16 +30% 归因（顺带闭环 L0 遗留问题）**：L0 单算子 bf16 matmul 仅 1.1×，训练为何 +30%？从内存流量账入手（activations/梯度减半对 L3 已拆解的 2618 MiB 峰值的影响）+ MLX 对 bf16 元素级算子的处理；给出"单算子峰值比"与"端到端负载"两口径的统一解释，并明确回答 L0 开放问题 1（半精度加速比偏低的原因）。
2. **INT8 KV decode 0.3× 开销分解**：教学形态每步对增长中的 KV 全量 dequant；生产形态（vLLM FP8 KV、融合 dequant+attention）如何避免；估算我们每步 dequant 的字节账与 kernel 数。
3. **权重量化无提速的机制**：mlx-lm 的 QuantizedLinear / gather-qmm 融合路径如何做到真提速（公开资料），与我们 dequant→matmul 的 kernel 数对比；说明"PTQ 提速需要 kernel 配合"这一层耦合。
4. **方法论对照**：朴素 round-to-nearest（我们）vs GPTQ（Hessian 误差补偿）vs AWQ（激活感知缩放）；文献中 INT4 g128 的困惑度恶化典型数字 vs 我们的 INT4 g64 +0.59%，我们处于什么水平。
5. **缺失件清单 → L6/专项**：KV INT4、QAT、混合粒度（敏感层高位）、KV cache 转正条件专项实验（L4 挂来的 n≥256 配对测量）——每项给落层建议。

## 5. 验收命令

无代码改动，不跑测试套件（消融脚本可直接运行）。验收标准：`runs/l5-diff-analysis.md` 存在，5 点齐全，第 1、2 点有定量账，每论点附证据位置，L0 开放问题 1 有明确回答。

## 6. 汇报要求

`-o` 文件写入：五点结论各一句话、bf16 归因与 KV 开销的关键数字、外部资料位置清单、与规格的偏差。阶段性 commit（简洁约定式信息），**不要 push**。
