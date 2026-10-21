# 任务规格：l3-model-assembly（L3-T1）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：本任务在 git worktree `/Users/wangchen/Developer/layers-wt-l3`（分支 `l3-training`）执行；该目录无 `.venv`，所有 Python 命令用主仓库解释器绝对路径 `/Users/wangchen/Developer/layers/.venv/bin/python`，cwd 保持在 worktree 内（bench/测试脚本已能把自身所在仓库根加入 sys.path）。

## 1. 研究问题

把 L2 组件组装成完整 GPT（pre-LN block + weight tying + 最终 LN），数值上与 PyTorch 同构模型做**前向 logits + 逐参数梯度**双对齐。失效形式：残差漏接、LN 位置错、tying 转置维度错、logits 未过最终 LN、堆叠后因果性破损、梯度路径断裂。

## 2. 改动白名单（worktree 内）

- `layers/transformer.py`、`layers/model.py`、`layers/__init__.py`
- `tests/test_model.py`
- `runs/l3-implementation-notes.md`
- `results/l3/`

其余一律禁止（含 tokenizer/attention/embedding 既有实现与测试、`bench/`、AGENTS.md）。

## 3. 禁止事项

- 新依赖（`.venv` 已有 mlx/torch/numpy/pytest/tiktoken/tokenizers）；`git push`；删文件；改既有测量协议
- 阅读任何 GPT 参照实现源码（nanoGPT / nanochat / minGPT / HF transformers / MLX examples）；torch API 仅作 Oracle
- 主路径使用 `mx.fast.*` / `mlx.nn.RoPE`（仅对照）；优化器/调度/裁剪不在本任务范围（属 L3-T2）
- 为通过测试硬编码期望输出

## 4. 实现要求

- 结构与接口严格按 `runs/l3-training-principle.md` §2/§5/§7：pre-LN、GELU MLP（4×）、weight tying、最终 LN 后出 logits；`n_layer=4, d_model=512, n_head=8, ctx=256`，词表读 `results/l1/tokenizer-1024.json`（1028）。
- 复用 L2 的 `TokenEmbedding`、`CausalSelfAttention`（已在 `layers/` 内），不得修改其代码；注意力返回 `(out, weights)` 时模型只取 out。
- 权重初始化固定 `mx.random.seed(42)`，规模与 L2 约定一致（投影 1/√d_model，embedding 0.02）。
- `cross_entropy` 用 log-sum-exp 稳定实现（本任务实现 loss；optimizer 属 T2）。
- PyTorch Oracle：同构模型（torch.nn 手写同结构，不用现成 GPT），numpy 桥接逐张量拷权重，前向 logits 与**全部命名参数梯度**对齐，fp32，atol/rtol=1e-4。
- 模型级因果性探针：改动位置 t 之后的 token，位置 < t 的 logits 逐位不变。

## 5. 验收命令（规划层将独立重跑）

```bash
/Users/wangchen/Developer/layers/.venv/bin/python -m pytest tests/ -q
```

测试必须覆盖：
1. 手算微模型用例（n_layer=1, d_model=2）LN/残差/CE 精确对齐
2. PyTorch Oracle：logits + 逐参数梯度对齐（1e-4）
3. 模型级因果性探针
4. 参数量核算：实测 == 理论（含 tying 只计一次），误差 0
5. 端到端烟测：真实文本 → L1 tokenizer → GPT logits，shape 与有限性
6. 单 batch 过拟合预测试：固定 batch 手推 50 步朴素 SGD（lr=0.01，仅为本测试临时内联，不进 `layers/`），loss 显著下降即可——验证梯度通路完整

## 6. 汇报要求

`-o` 文件写入：文件清单、验收命令完整输出、参数量实测 vs 理论、与规格的偏差及原因、实现笔记要点。阶段性 commit（简洁约定式信息），**不要 push**。
