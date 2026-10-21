# 任务规格：l2-attention

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明，不要自由发挥。

## 1. 研究问题

从零实现嵌入 + RoPE + 多头因果注意力（MLX 主栈）：数值上与 PyTorch 逐中间量对齐，Infra 上验证 L2 原理笔记的三条预言（naive vs fast 内核的交叉点、KV 增长曲线、softmax 内存开销占比）。可能的失效形式：softmax 数值溢出、因果掩码方向/对角错误（未来信息泄漏）、head 切分拼接错误、RoPE 配对/角度约定错误、scale 因子遗漏。

## 2. 改动白名单

- `layers/embedding.py`、`layers/attention.py`（实现），`layers/__init__.py`（导出更新）
- `tests/test_attention.py`
- `bench/l2_attention_bench.py`
- `runs/l2-implementation-notes.md`（卡壳点、假设、与原理笔记的偏差）
- `results/l2/`（测量数据）

其余一律禁止（含 `layers/tokenizer.py`、`tests/test_tokenizer.py`、既有 `bench/`、`runs/`、`results/l0|l1/`、`AGENTS.md`）。

## 3. 禁止事项

- 新依赖仅批准 `torch`（Oracle 用，`uv pip install --python .venv/bin/python torch`）；不得改既有依赖版本
- `git push`、删文件、改既有测量协议
- 阅读任何参照实现源码（nanochat / minGPT / HF transformers / MLX 官方示例的注意力实现）；只允许调用 `mx.fast.scaled_dot_product_attention`、`mlx.nn.RoPE`、`torch` 的公开 API 作对照
- 主路径使用 `mx.fast.*` 或 `mlx.nn.RoPE`（仅 Oracle 对照可用）
- 为通过测试硬编码期望输出

## 4. 实现要求

- 接口与约束严格按 `runs/l2-attention-principle.md`（接口设计节与算子约束节）。
- 超参数：`d_model=512, n_head=8, d_head=64, rope_base=10000`，词表复用 L1 产物 `results/l1/tokenizer-1024.json`（id 空间 1028）。
- softmax 手写数值稳定版（减行最大值）；因果掩码含对角（位置 i 可见 ≤ i）。
- 权重初始化固定 seed（`mx.random.seed(42)`），保证 Oracle 对齐可复现。
- PyTorch 对照：同构注意力，权重经 numpy 桥接逐张量拷贝，逐中间量（Q/K/V、scaled scores、probs、out）对齐，fp32，atol=rtol=1e-5。
- bench 要求（沿用 L0 协议：预热 3、计时 10、median/p90、config hash、JSON 入 `results/l2/`）：
  1. naive 主实现 vs `mx.fast.scaled_dot_product_attention`，n ∈ {128, 256, 512, 1024, 2048, 4096}，验证交叉点预言
  2. KV 内存增长曲线：d_model=512 fp32/bf16，n 至 8192，实测 vs 理论（2·n·d_model·dtype_size）
  3. softmax 占整层前向的时间比例（粗粒度计时分解即可）

## 5. 验收命令（规划层将独立重跑）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l2_attention_bench.py
```

测试必须覆盖：
1. 手算微型用例（d_head=2, n=3）全链路精确对齐
2. 因果性探针：改动位置 t 之后的输入，位置 ≤ t 的输出必须逐位不变
3. RoPE 90° 特例 + 相对位置性质（`R_mᵀ R_n = R_{n-m}` 数值验证）
4. PyTorch Oracle 逐中间量对齐（atol/rtol 1e-5）
5. `mx.fast` / `mlx.nn.RoPE` 对照主实现
6. 与 L1 tokenizer 的端到端烟测：真实文本 → encode → embedding → attention 前向，shape 与有限性检查

## 6. 汇报要求

`-o` 文件写入：实际文件清单、第 5 节命令完整输出、三条 Infra 预言的实测判定（成立/推翻 + 数字）、与规格的偏差及原因、`runs/l2-implementation-notes.md` 要点摘录。阶段性 commit（简洁约定式信息），**不要 push**。
