# 任务规格：l5-kv-bf16（L5-T2）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l5-quantization`；Python 用 `.venv/bin/python`。
> **前置**：L5-T1 已验收（`layers/quant.py` 可用）。

## 1. 研究问题

T2a：per-token 对称 INT8 KV cache，验证"INT8 KV 近无损"与长上下文内存减半的预言。T2b：bf16 混合精度训练对照（接 L3 缺失件），验证原理笔记 §6 三条预言（MFU 提升 ≤10%、内存降 35–40%、loss 曲线重合 <2%）。失效形式：KV 量化误差经 softmax 放大、quant 状态与 cache 追加错位、bf16 训练梯度下溢/loss 发散、master 权重被 bf16 覆盖。

## 2. 改动白名单

- `layers/quant.py`（可追加 KV 量化函数）
- `layers/inference.py`（**仅加法式**：新增 QuantKVCache 路径，既有 KVCache 行为不动）
- `scripts/train.py`（**仅加法式**：`--precision {fp32,bf16}` 入口，默认行为与 L3 逐位一致）
- `tests/test_kv_quant.py`、`tests/test_bf16_train.py`
- `bench/l5_kv_bf16_bench.py`
- `runs/l5-implementation-notes.md`（追加 T2 段）
- `results/l5/`

其余一律禁止（含全部既有测试——原样通过；`layers/model.py`、`layers/optimizer.py` 不改）。

## 3. 禁止事项

- 新依赖；`git push`；删文件；改既有测量协议
- 阅读任何量化/混合精度参照实现源码；MLX/torch 公开 API 仅作 Oracle
- 修改既有测试；为通过测试硬编码期望输出

## 4. 实现要求

**T2a（KV 量化）**：
- per-token 对称 INT8 量化 K 与 V（每个 token 的 d_model 维向量一个 scale，K 与 V 各自独立）；`QuantKVCache` 接口与 L4 `KVCache` 对齐，可互换注入 `generate`。
- 正确性不变式：同 seed 同 prompt，INT8 KV 与 fp32 KV 的逐步 logits 偏差有界且单调可解释（记录 max abs diff 曲线，不设零阈值，但 decode 生成结果在 greedy 下允许分叉——分叉点需记录）。

**T2b（bf16 训练）**：
- 混合精度教学形态：前向/反向 bf16，master 权重、Adam 状态、loss 归约保持 fp32；不用 loss scaling（bf16 不需要，实现笔记说明理由）。
- `scripts/train.py --precision bf16`：同 seed 同 200 步配置（沿用 v1 步数即可，断言阈值 28% 保持不变以防误报），与 fp32 基线对照。

**bench（L0 协议，JSON 入 `results/l5/`）**：
1. KV 量化：n ∈ {128, 512, 2048} decode 的 KV 内存（量化 vs fp32，应 ~2×）与 decode tok/s 变化
2. bf16 训练：200 步的 tokens/s、峰值内存、终值 loss vs L3 fp32 基线（`results/l3/train-metrics.json`）

## 5. 验收命令（规划层将独立重跑）

```bash.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l5_kv_bf16_bench.py
/Users/wangchen/Developer/layers/.venv/bin/python scripts/train.py --precision bf16
```

测试必须覆盖：
1. KV 量化手算微用例 + roundtrip 误差 ≤ Δ/2
2. QuantKVCache 与 KVCache 接口一致性（append/读取语义对齐）
3. INT8 KV 与 fp32 KV 的逐步 logits max-abs-diff 曲线有限且被记录
4. bf16 训练：master 权重 dtype 为 fp32 断言；200 步 loss 有限且降幅 ≥28%；与 fp32 终值差 <2% 的记录（超差不断言失败，但必须写入结果）
5. L1–L5-T1 全部既有测试原样通过

## 6. 汇报要求

`-o` 文件写入：文件清单、验收命令完整输出、KV 量化两轴（内存/速度）与 bf16 三轴（MFU/内存/loss）数字、§6 三条预言的实测判定、与规格的偏差及原因。阶段性 commit（简洁约定式信息），**不要 push**。
