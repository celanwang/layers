# 任务规格：l4-diff-analysis（对照精读）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。分析任务，**不修改任何实现/测试代码**。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l4-inference`；Python 用 `.venv/bin/python`。

## 1. 研究问题

mini 推理引擎与生产参照（vLLM / llama.cpp / mlx-lm）的架构差距定量分析；解释两个实测现象：decode 仅为 roofline 的 3.7%、KV cache 在本模型无收益（~1.0×）；连续批处理只有 1.04× 的缺口归因。

## 2. 改动白名单

- `runs/l4-diff-analysis.md`（主产物）
- `results/l4/`（补充对照实验数据）
- `bench/l4_*_ablation.py`（如需消融，可新增）

其余一律禁止（含 `layers/`、`tests/`、既有 bench）。

## 3. 禁止事项

- `git push`、删文件、改实现/测试、新依赖
- 泛泛文献综述——每论点必须有证据（数字或具体源码/文档位置）

## 4. 实现要求

允许并需要阅读参照：vLLM（PagedAttention 论文 + 公开文档/源码）、llama.cpp、mlx-lm 的 generate 路径、Ollama 公开资料。源码不入库。

分析点（每点给证据）：

1. **连续批处理的收益分解**：调度无气泡 vs kernel batching 两部分；我们的 mini 形态（gather + 逐请求前向）只吃到前者（实测 1.04×）；估算 vLLM 形态在本模型/本机上的理论收益上限（8 请求 decode 的权重读取摊薄账），说明缺口在哪。
2. **decode 3.7% roofline 归因**：数一遍单 token decode 的 kernel 调用数（embedding + 4 层 × (QKV/RoPE/scores/softmax/PV/O + MLP + LN) + LM head），按 L0 小任务带宽塌缩（224 GB/s）与 launch 开销两口径估算，定位主导项；与 mlx-lm 跑同规模模型的公开数字对比（注明口径差异）。
3. **KV cache 零收益的条件分析**：cache 收益 = 省下的重算 FLOPs vs KV 读取与维护开销，推导出收益转正的模型规模/序列长度条件；用 L2/L3/L4 已有数据标定，预测在什么规模下 cache 加速比显著 >1。
4. **分页设计对照**：block_size=16 vs vLLM 默认 16 的巧合/依据；CoW 与前缀共享（87.5% 节省）对应的生产场景（system prompt、多采样、beam search）；我们 gather 到连续视图 vs vLLM kernel 直读页表的代价差。
5. **缺失件清单 → L5/L6 挂载**：KV cache 量化（→ L5）、投机解码、prefill/decode 混排与 eviction、kernel batching；每项给出落层建议。

## 5. 验收命令

无代码改动，不跑测试套件（消融脚本可直接运行）。验收标准：`runs/l4-diff-analysis.md` 存在，5 点齐全，第 1、2 点有定量账，每论点附证据位置。

## 6. 汇报要求

`-o` 文件写入：五点结论各一句话、收益分解与 kernel 账的关键数字、外部资料位置清单、与规格的偏差。阶段性 commit（简洁约定式信息），**不要 push**。
