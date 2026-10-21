# 任务规格：l5-weight-quant（L5-T1）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l5-quantization`；Python 用 `.venv/bin/python`。

## 1. 研究问题

实现对称 per-group 权重量化（INT8 per-channel / INT4 group=64，PTQ）并接入推理路径，测量三轴 trade-off：内存压缩比、heldout loss 恶化、decode 速度变化（验证原理笔记 §4 预言——提速有限甚至变慢）。失效形式：scale 分组错位、嵌入/LN 被误量化、dequant 后 matmul 维度错、roundtrip 误差超界、质量测量用了训练集而非 heldout。

## 2. 改动白名单

- `layers/quant.py`
- `layers/model.py`、`layers/__init__.py`（**仅加法式**：量化模型构建入口，不改既有前向行为）
- `tests/test_quant.py`
- `bench/l5_quant_bench.py`
- `runs/l5-implementation-notes.md`
- `results/l5/`

其余一律禁止（含全部既有测试——原样通过；`layers/inference.py` 复用不改）。

## 3. 禁止事项

- 新依赖；`git push`；删文件；改既有测量协议
- 阅读任何量化库/参照实现源码（GPTQ / AWQ / bitsandbytes / mlx-lm quant / llama.cpp quant）；MLX `mx.quantize`/`mx.dequantize` 公开 API 仅作 Oracle 对照
- 主路径使用 `mx.nn.QuantizedLinear` 或等价现成件（必须手写 quant/dequant）
- 为通过测试硬编码期望输出；修改既有测试

## 4. 实现要求

- 严格按 `runs/l5-quantization-principle.md` §2/§3/§7：对称量化；INT8 用 per-channel（每输出通道一个 scale），INT4 用 per-group（group_size=64，scale 存 fp16）；只量化 block 内投影（QKV/O/MLP），嵌入（含 tied head）与 LN 保持 fp32。
- 前向教学形态：dequant → matmul（fp32），不做融合 kernel。
- 复用 L4 推理路径（`layers/inference.py` 的 generate/KVCache）做 decode 测量；模型加载 L3 的 `results/l3/ckpt-final-500.npz`。
- bench（L0 协议，JSON 入 `results/l5/`）：
  1. 内存：量化产物序列化字节数 vs fp32（压缩比，含 scale 开销核算）
  2. 质量：同一 heldout（沿用 L1/L3 的 `len*9//10` 切分后 10%）loss：fp32 / INT8 / INT4(g64) 三档
  3. 速度：decode tokens/s（prompt 128、生成 128，greedy）三档对照
- 如 MLX 提供 `mx.quantize/dequantize`，做输出对照并记录差异（不强制存在）。

## 5. 验收命令（规划层将独立重跑）

```bash.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l5_quant_bench.py
```

测试必须覆盖：
1. quant/dequant 手算微用例（已知输入，期望值手算可验证，含 group 边界）
2. roundtrip 误差断言：max |w − ŵ| ≤ Δ/2 + ε（per-group）
3. 量化作用域：嵌入/LN 参数逐位未动；block 投影全部被量化
4. 量化模型前向有限性 + 与 fp32 模型 logits 偏差的量级检查（INT8 < INT4）
5. 质量门槛：INT8 heldout loss 恶化 <1%、INT4(g64) <5%（相对 fp32 基线）
6. L1–L4 全部既有测试原样通过（60 项）

## 6. 汇报要求

`-o` 文件写入：文件清单、验收命令完整输出、三轴测量数字表、原理笔记 §4 三条预言的实测判定（成立/推翻）、与规格的偏差及原因。阶段性 commit（简洁约定式信息），**不要 push**。
