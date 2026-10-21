# 任务规格：l3-training-loop（L3-T2）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：git worktree `/Users/wangchen/Developer/layers-wt-l3`（分支 `l3-training`）；无 `.venv`，Python 一律用 `/Users/wangchen/Developer/layers/.venv/bin/python`，cwd 保持在 worktree 内。
> **前置**：L3-T1 已验收（`layers/model.py` GPT 可用）。

## 1. 研究问题

手写 AdamW + cosine 调度 + 梯度裁剪，驱动 13.1M GPT 在 tiny-shakespeare 上真实训练，验证 L3 原理笔记三条 Infra 预言（训练内存 ≈ 4× 参数、MFU < 10%、loss 曲线与模型侧 bpb），并用模型 bpb 回补 L1 开放问题 1（tokenizer 的下游代价）。失效形式：偏差修正遗漏、weight decay 耦合进自适应分母、grad/param 错位、train/heldout 泄漏、loss 假下降（评估集混入训练）。

## 2. 改动白名单（worktree 内）

- `layers/optimizer.py`、`scripts/train.py`（新增 `scripts/`）
- `tests/test_optimizer.py`
- `bench/l3_train_bench.py`
- `runs/l3-implementation-notes.md`（追加 T2 段）
- `results/l3/`

其余一律禁止（含 T1 的 `layers/model.py`、`layers/transformer.py` 及其测试）。

## 3. 禁止事项

- 新依赖；`git push`；删文件；改既有测量协议；改 T1 代码
- 阅读任何训练框架/参照实现源码（nanoGPT train loop、HF Trainer、mlx-lm trainer 等）；torch 仅作 Oracle
- 手写优化器之外使用 `mlx.optimizers` 作主路径（仅可对照）
- 为通过测试硬编码期望输出

## 4. 实现要求

- 严格按 `runs/l3-training-principle.md` §4/§7：`AdamW`（偏差修正、decoupled weight decay、β=(0.9,0.95)、ε=1e-8）、`cosine_lr`（warmup 100 步、峰值 3e-4、衰减至 10%）、`clip_grad_norm`（global norm ≤ 1.0）。
- 训练配置：ctx=256、batch=16、步数 200（短跑验收版）；数据用 L1 tokenizer（`results/l1/tokenizer-1024.json`）编码整个 `data/tiny-shakespeare.txt`，train/heldout 沿用 `len*9//10` 切分；从 heldout 采样评估 batch，评估不前向训练梯度。
- 检查点：训练结束保存权重到 `results/l3/ckpt-final.npz`（或 MLX 原生格式），loss 曲线（train/heldout 每 10 步）存 `results/l3/loss-curve.json`。
- 测量（沿用 L0 协议，JSON 入 `results/l3/`）：
  1. 训练步吞吐 tokens/s（median/p90，含 config hash）
  2. 峰值内存（`mx.metal` 内存 API），对照预言 210 MB + activations
  3. MFU：实测 tokens/s × 78.7 MFLOP/token ÷ L0 实测 14.75 TFLOP/s
  4. **模型 bpb**：heldout 上 final loss（nats）÷ ln2 × （tokens/bytes）换算 bits/byte，与 L1 的 uniform-ID proxy（4.19 vs 4.69）并列讨论
- AdamW Oracle：`torch.optim.AdamW` 同参同梯度，对齐前 5 步参数更新（atol 1e-6）。

## 5. 验收命令（规划层将独立重跑）

```bash
/Users/wangchen/Developer/layers/.venv/bin/python -m pytest tests/ -q
/Users/wangchen/Developer/layers/.venv/bin/python bench/l3_train_bench.py
/Users/wangchen/Developer/layers/.venv/bin/python scripts/train.py
```

测试必须覆盖：
1. AdamW 手算微用例（标量参数 3 步，手算可验证，含偏差修正）
2. `torch.optim.AdamW` Oracle 5 步对齐（1e-6）
3. cosine 调度边界（warmup 端点、峰值点、终值 = 10% 峰值）
4. clip：构造 norm=2.0 的梯度，裁剪后 global norm == 1.0
5. `scripts/train.py` 验收判据：heldout loss 相对初始（≈6.94 nats）降幅 > 30%，且曲线无 NaN/Inf；单 batch 过拟合测试（50 步压到 < 0.5 nats）

## 6. 汇报要求

`-o` 文件写入：文件清单、三条命令完整输出、三条 Infra 预言的实测判定（成立/推翻 + 数字）、模型 bpb 与 L1 proxy 的对照解读、偏差及原因。阶段性 commit（简洁约定式信息），**不要 push**。
