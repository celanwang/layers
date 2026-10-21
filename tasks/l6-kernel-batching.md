# 任务规格：l6-kernel-batching（L6-T1）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l6-serving`；Python 用 `.venv/bin/python`。

## 1. 研究问题

把 L4 的逐请求 decode 循环升级为拼批前向：B 个活跃请求各 1 token 拼成一批，共享权重读取，变长 KV 用 pad + per-request 掩码处理。验证黄金不变式的批次版（拼批 == 逐请求 logits）与吞吐摊薄预言（B=8 ≥ 4×）。失效形式：padding 污染（掩码漏掉 pad 位）、批内行错位（请求 A 拿到请求 B 的 logits）、批大小变化时行为漂移、新请求插入/完成退出时块表与批索引脱节。

## 2. 改动白名单

- `layers/engine.py`、`layers/attention.py`、`layers/transformer.py`、`layers/model.py`（**仅加法式**：拼批前向路径，既有单请求行为不动）
- `layers/__init__.py`
- `tests/test_batched_engine.py`
- `bench/l6_batching_bench.py`
- `runs/l6-implementation-notes.md`
- `results/l6/`

其余一律禁止（含全部既有测试——原样通过；`layers/inference.py`、`layers/paged_kv.py` 复用不改）。

## 3. 禁止事项

- 新依赖；`git push`；删文件；改既有测量协议
- 阅读任何推理引擎/服务框架实现源码（vLLM / SGLang / TGI / llama.cpp server / mlx-lm server）
- 主路径使用 `mx.fast.*`（对照可用）；修改既有测试；硬编码期望输出

## 4. 实现要求

- 严格按 `runs/l6-serving-principle.md` §3/§4/§6：`BatchedEngine` 加法式扩展 L4 `Engine`；拼批 decode（ids [B,1] 一批前向），KV 从 paged pool gather 后 pad 到批内最长 + per-request 掩码；prefill 在 iteration 边界逐个处理后加入 decode batch。
- 掩码必须保证 pad 位注意力权重恰为 0（含 softmax 后的数值验证，不许只靠"应该没问题"）。
- 完成请求在 iteration 末退出并释放块；批大小随到达/完成动态变化。
- bench（L0 协议，JSON 入 `results/l6/`）：
  1. 吞吐扩展：B ∈ {1, 2, 4, 8}，每请求生成 64 token，聚合 tokens/s vs 逐请求基线（L4 形态），计算摊薄比
  2. 单请求延迟：B=1 拼批路径 vs L4 逐请求路径（退化须 <10%）
  3. 批内长度不齐惩罚：8 请求 KV 长度方差大 vs 小的吞吐对比（padding 浪费的量化）
- 正确性基线：同 seed 同请求集，BatchedEngine 每请求生成结果与 L4 Engine 逐 token 一致。

## 5. 验收命令（规划层将独立重跑）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l6_batching_bench.py
```

测试必须覆盖：
1. **批次版黄金不变式**：8 请求变长 KV 拼批，每请求逐步 logits 与逐请求路径一致（1e-5），≥ 32 步
2. 行隔离：交换批内两行输入，输出对应交换（证明无跨请求泄漏到错误行）
3. pad 掩码数值验证：pad 位注意力概率 == 0（1e-7 断言）
4. 动态批：请求中途完成/插入，剩余请求 logits 不受影响
5. L1–L5 全部既有测试原样通过（91 项）
6. 吞吐门槛：B=8 聚合吞吐 ≥ 逐请求 4×

## 6. 汇报要求

`-o` 文件写入：文件清单、验收命令完整输出、三组 bench 数字、§5 预言判定、与规格的偏差及原因。阶段性 commit（简洁约定式信息），**不要 push**。
