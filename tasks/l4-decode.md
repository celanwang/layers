# 任务规格：l4-decode（L4-T1）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l4-inference`；Python 用 `.venv/bin/python`。

## 1. 研究问题

给 L3 训练好的 GPT 装上推理路径：KV cache 免重算 decode、采样器（temperature/top-k/top-p）、流式 UTF-8 增量解码，并测量 decode 负载的内存受限特性（roofline vs 实测）。失效形式：cache 与无 cache 逐步 logits 不一致（黄金不变式破坏）、采样截断未重归一、流式输出出现 ``、cache 追加错位（off-by-one）、改造破坏 L2/L3 既有行为。

## 2. 改动白名单

- `layers/inference.py`（KVCache / Sampler / generate）
- `layers/attention.py`、`layers/transformer.py`、`layers/model.py`（**仅加法式**：新增可选 cache 参数，不传时行为与现状逐位一致）
- `layers/__init__.py`（导出更新）
- `tests/test_inference.py`
- `bench/l4_decode_bench.py`
- `runs/l4-implementation-notes.md`
- `results/l4/`

其余一律禁止（含 `layers/tokenizer.py`、`scripts/train.py`、既有测试文件——**不允许修改任何既有测试**，它们必须原样通过）。

## 3. 禁止事项

- 新依赖；`git push`；删文件；改既有测量协议
- 阅读任何推理引擎/参照实现源码（vLLM / SGLang / llama.cpp / nanochat generate / mlx-lm）；只允许用既有 `layers/` 代码与已训练 ckpt
- 主路径使用 `mx.fast.*`（对照可用）
- 为通过测试硬编码期望输出；修改既有测试来迁就新代码

## 4. 实现要求

- 接口与约束严格按 `runs/l4-inference-principle.md` §2/§6：`KVCache`、`Sampler(temperature, top_k, top_p, seed)`、`generate(...) -> Iterator[str]` 流式。
- cache 设计：每层追加式；`generate` 首步 prefill 整个 prompt，后续每步只前向新 token。
- 模型改造加法式：`forward(ids, cache=None)` 形式，不传 cache 时与 L3 行为逐位一致（用 L3 的 logits 对齐测试证明）。
- 采样：temperature 作用在 logits；top-k/top-p 截断后重归一；`mx.random` seed 固定可复现。
- 流式 UTF-8：增量缓冲不完整字节后缀，输出拼接必须等于一次性 decode 且无 ``。
- 加载 `results/l3/ckpt-final-500.npz` 做生成烟测（greedy，200 token）。
- bench（L0 协议，JSON 入 `results/l4/`）：
  1. prefill vs decode 分阶段 tokens/s（prompt 128，生成 256）
  2. decode 实测 vs roofline（450 GB/s ÷ 52.5 MB 权重 ≈ 8.5k tok/s 上限），并测 n ∈ {64,128,256,512} 时 decode 速率随 KV 增长的衰减曲线
  3. cache vs 无 cache 的端到端生成耗时比（同 max_new）

## 5. 验收命令（规划层将独立重跑）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l4_decode_bench.py
```

测试必须覆盖：
1. **黄金不变式**：同 seed 同 prompt，cache 与无 cache 逐步 logits 逐位一致（1e-5），≥ 32 步
2. 采样语义：temperature=0 == greedy；top_k=1 == greedy；top-p 截断后概率和 1；固定 seed 复现
3. 流式 UTF-8：跨 token 的 CJK/emoji 序列增量拼接 == 一次性 decode，无 ``
4. 加法改造回归：不传 cache 的前向与 L3 基线 logits 对齐（用 L3 ckpt 前向对比）
5. L2/L3 全部既有测试原样通过（49 项）
6. 生成烟测：greedy 200 token 有限、可 decode、非重复单字符退化

## 6. 汇报要求

`-o` 文件写入：文件清单、验收命令完整输出、roofline vs 实测的对照数字与一句话解读、与规格的偏差及原因、实现笔记要点。阶段性 commit（简洁约定式信息），**不要 push**。
