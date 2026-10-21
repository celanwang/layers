# 任务规格：l3-diff-analysis（对照精读）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。分析任务，**不修改任何实现/测试代码**。
> **工作环境**：git worktree `/Users/wangchen/Developer/layers-wt-l3`（分支 `l3-training`）；Python 用 `/Users/wangchen/Developer/layers/.venv/bin/python`。

## 1. 研究问题

自研训练栈与参照（nanoGPT / nanochat / MLX examples）的设计分歧；定量解释两条被推翻的 Infra 预言：MFU 33% 为何远超"<10%"预期、峰值内存 2618 MiB 的超额来自哪里。

## 2. 改动白名单

- `runs/l3-diff-analysis.md`（主产物）
- `results/l3/`（补充对照实验数据）
- `bench/l3_*_ablation.py`（如需消融，可新增）

其余一律禁止（含 `layers/`、`scripts/train.py`、`tests/`）。

## 3. 禁止事项

- `git push`、删文件、改实现/测试/训练脚本、新依赖
- 泛泛文献综述——每论点必须有证据（数字或具体源码/论文位置）

## 4. 实现要求

允许并需要阅读参照：nanoGPT（`model.py`/`train.py`）、nanochat 训练脚本、MLX 文档（惰性求值与内存语义）。源码不入库。

分析点（每点给证据）：

1. **MFU 33% 归因**：6ND 近似忽略了注意力的序列相关 FLOPs——先补齐精确 FLOPs 账（含 4 层注意力的 12·n²·d_head·h 项与投影），重算真实 MFU；再结合 L0/L2 发现（matmul 效率、batch 4096 tokens/step 的矩阵规模）解释为何小模型也能到 ~30%。给出"MFU 定义口径"讨论（6ND vs 精确账）各自的误导性。
2. **峰值内存拆解**：2618 MiB = 参数四态（200 MiB）+ ？。定量估算 activations（n² 物化注意力 ×4 层 × autograd 保存的全部中间量，fp32）、MLX 惰性求值/cache 行为的影响；如可行，用 `mx` 内存 API 做分组测量（消融：关 autograd 只前向 / 减 batch / 减层数）支撑归因。
3. **超参与生产惯例对照**：lr=3e-4、wd=0.1、β₂=0.95、warmup 100/总步数比例、cosine 到 10%——与 nanoGPT 同规模配置、当前主流小模型训练惯例的异同及理由。
4. **训练循环缺失件清单**：对照 nanoGPT/nanochat，列出梯度累积、混合精度（bf16 autocast）、checkpoint resume、LR decay 与 early stop、分布式；标注各自适合落在 L4/L5 哪一层。
5. **模型 bpb 口径统一**：L1 uniform-ID proxy（4.19/4.69）与本迭代模型 bpb（2.52@500 步）口径不同；设计一个等口径对比方案（如同一 heldout 上 cl100k 编码 + 等规模模型训练的估计成本），评估其可行性但不实施大规模训练。

## 5. 验收命令

无代码改动，不跑测试套件（消融实验脚本可直接运行）。验收标准：`runs/l3-diff-analysis.md` 存在，5 点齐全，第 1、2 点有定量账，每论点附证据位置。

## 6. 汇报要求

`-o` 文件写入：五点结论各一句话、真实 MFU 与内存拆解的关键数字、外部资料位置清单、与规格的偏差。阶段性 commit（简洁约定式信息），**不要 push**。
