# 任务规格：l7-batched-sampling（批式采样单栅栏）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。前置：L7-T1 已验收并合入本分支；原理笔记 `runs/l7-decode-perf-principle.md` 必读。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l7-decode-perf`；Python 用 `.venv/bin/python`。

## 1. 研究问题

把拼批 decode 的采样从"每请求 `mx.eval` + `.item()`"（B=8 ⇒ 每步 8 次串行主机同步栅栏）换成"整批一次构图、一次 eval、一次批量主机传输"，验证标量栅栏在行相关项中的占比。预言：在 T1 之上再降 5–10% decode 时间；<3% 则说明栅栏不是瓶颈。

可能的失效形式：RNG 位精确约束迫使保留逐请求 seed 操作，构图变大反而更慢；lazy 求值下 8 次小 eval 的实际成本低于预期。

## 2. 改动白名单

- `layers/inference.py`、`layers/engine.py`（**仅加法式**；既有 `Sampler` 与既有调用路径行为一字不动）
- `layers/__init__.py`（导出新符号，如需要）
- `tests/test_l7_batched_sampling.py`（新增）
- `bench/l7_sampling_bench.py`（新增；可扩展 T1 bench，但既有 bench 文件不删不改）
- `runs/l7-implementation-notes.md`（追加）
- `results/l7/`（测量数据与汇报）

其余一律禁止。

## 3. 禁止事项

- `git push`、删文件、改动既有实现/测试/基准、新依赖
- 不得改变任何请求的 RNG 语义：第 i 个请求第 d 次采样的随机性必须等于 `mx.random.seed(seed_i + d)` 后 `mx.random.categorical` 的既有结果（位精确）
- 不得改动温度/top_k/top_p 语义与边界校验
- 不得改动测量协议口径

## 4. 实现要求

1. **批式采样入口**：输入 B 个请求的 sampler 配置/状态与 stacked logits `[B, V]`，输出 B 个 token id；每请求 RNG 流独立且与既有 `Sampler` 位精确一致（同 seed 同参数同历史 ⇒ 同 token 序列）。核心约束：**整批至多一次主机同步**（一次 eval + 一次批量传输），禁止逐行 `.item()`。
2. **greedy 与随机路径都覆盖**；temperature=0、top_k=1 等退化情形语义不变。
3. **引擎接线**：`BatchedEngine.step` 的采样循环走批式路径（加法式）；事件顺序、完成判定、池回收语义与 L6/L7-T1 完全一致。
4. **数值不变式**：同 seed 同参数下新路径与既有逐请求路径**逐 token 相等**（这是硬门槛，不是 1e-5 容差——token 序列必须完全一致）。测试参照 `tests/test_batched_engine.py` 模式，等长/不等长/动态到达/多温度混合场景都覆盖。
5. bench 协议与 L6/L7-T1 一致（3 预热/10 重复、配对交替、原始样本、config hash、源码哈希），在 T1 交付的 bench 基础上报三项配对：L6 基线 vs T1-only vs T1+T2。

## 5. 验收命令

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l7_sampling_bench.py
```

门槛：正确性全绿（token 位精确）；B=8 聚合吞吐相对 T1-only 再提升 **≥5%** 记为预言支持，**<3%** 记为受挫——如实记录。

## 6. 汇报要求

`-o` 文件写入：实际改动文件清单、三级配对数字（L6 基线 / T1-only / T1+T2）、T2 单项增量、预言判定、坑与偏差。完整报告写 `results/l7/l7-t2-report.md`。阶段性 commit（简洁约定式信息），**不要 push**。
