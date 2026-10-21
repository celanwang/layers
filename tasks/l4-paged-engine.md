# 任务规格：l4-paged-engine（L4-T2）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l4-inference`；Python 用 `.venv/bin/python`。
> **前置**：L4-T1 已验收（`layers/inference.py` 的 KVCache/Sampler/generate 可用）。

## 1. 研究问题

实现 mini 推理引擎：PagedAttention 思想的分页 KV 池（块表、fork/前缀共享、写时复制）+ 连续批处理调度。验证：内部碎片 ≤ block_size/2、序列间零串话、连续批处理相对顺序执行的吞吐提升。失效形式：块表错位导致序列间 KV 污染（最严重）、fork 后写污染（CoW 缺失）、块 refcount 泄漏、调度饥饿。

**形态约束（防止自由发挥）**：分页只落在**调度与内存管理层**——注意力计算前把逻辑序列的 K/V 从页池 gather 成连续视图喂给 T1 前向；不做 kernel 级 paged attention，不做多请求 padded batch 前向（逐请求前向 + 调度级连续批处理）。与 vLLM 的架构差距（kernel batching 缺失对吞吐扩展性的影响）留作对照精读分析点。

## 2. 改动白名单

- `layers/paged_kv.py`、`layers/engine.py`
- `layers/__init__.py`（导出更新）
- `tests/test_engine.py`
- `bench/l4_engine_bench.py`
- `runs/l4-implementation-notes.md`（追加 T2 段）
- `results/l4/`

其余一律禁止（含 T1 的 `layers/inference.py` 与全部既有测试——**不允许修改既有测试**）。

## 3. 禁止事项

- 新依赖；`git push`；删文件；改既有测量协议
- 阅读任何推理引擎实现源码（vLLM / SGLang / llama.cpp / TensorRT-LLM）；PagedAttention **论文**允许阅读（思想已在原理笔记 §5 概述）
- 修改 T1 代码来迁就 T2；为通过测试硬编码期望输出

## 4. 实现要求

- `PagedKVPool`：固定块数页池，block_size=16；每序列块表；`alloc/append/free/fork`；fork 共享前缀块（refcount），追加写时若块 refcount>1 先复制（CoW）；提供 `gather(seq_id) -> 连续 (k, v)` 供前向。
- `Engine`：`add_request(prompt, sampler) -> seq_id`、`step()` 每个 iteration 对所有活跃请求各 decode 一步（逐请求前向）、完成的请求释放块；支持前缀相同请求的 fork（共享 prompt 块）。
- 调度语义：同 seed 同请求集合下，连续批处理的每请求生成结果与单独执行**逐 token 一致**（调度不得改变生成内容）。
- bench（L0 协议，JSON 入 `results/l4/`）：
  1. 内部碎片：N ∈ {1, 4, 8, 16} 条随机长度序列（32–256 token），实测浪费字节 ≤ N × block_size/2 × 每块字节数
  2. 连续批处理 vs 顺序执行：同一批 8 条请求的总墙钟时间与吞吐（tokens/s）
  3. fork 前缀共享的内存节省：8 条同 prompt 请求，共享 vs 不共享的池占用对比
- 页池容量：按 ctx ≤ 512、≤ 16 并发设计，容量不足时的拒绝/排队行为需在实现笔记中说明。

## 5. 验收命令（规划层将独立重跑）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l4_engine_bench.py
```

测试必须覆盖：
1. **零串话（黄金不变式推广）**：两条序列交错 decode，paged 路径与 T1 单请求 cache 路径逐步 logits 一致（1e-5）
2. fork CoW：fork 后子序列继续生成，父序列 KV 逐位不变；两 fork 分支互不污染
3. refcount 无泄漏：反复 alloc/append/free 后池完全复原（free 块数 == 初始）
4. 调度等价性：同 seed 同请求集，连续批处理与单独执行逐 token 一致
5. T1/L2/L3 全部既有测试原样通过
6. 碎片上界：测试 1 的浪费字节断言 ≤ 理论上界

## 6. 汇报要求

`-o` 文件写入：文件清单、验收命令完整输出、三组 bench 关键数字、容量不足行为的说明、与规格的偏差及原因。阶段性 commit（简洁约定式信息），**不要 push**。
