# 任务规格：l7-paged-attention（直读 page 的拼批 attention）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。前置：L6 已收官，`BatchedEngine` 为不可变 Oracle；原理笔记 `runs/l7-decode-perf-principle.md` 必读。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l7-decode-perf`；Python 用 `.venv/bin/python`。

## 1. 研究问题

把拼批 decode 的 KV 物化从"逐行 Python gather 全前缀"（B=8、4 层 ⇒ 每 forward 32 次 append + 32 次全前缀重建）换成"按 block table 单次向量化物化"，验证 KV 重建是否为行相关项主体。预言：B=8 聚合吞吐 1436 → ≥1700 tok/s。

可能的失效形式：扁平索引本身引入新的大拷贝（收益被抵消）；block 对齐 padding 在 maxlen 大时反而更费；MLX lazy 求值把单 kernel 假设打破。

## 2. 改动白名单

- `layers/paged_kv.py`、`layers/attention.py`、`layers/engine.py`（**仅加法式**：新增类/方法/函数；既有方法与既有调用路径的行为一字不动）
- `layers/__init__.py`（导出新符号，如需要）
- `tests/test_l7_paged_attention.py`（新增）
- `bench/l7_paged_attn_bench.py`（新增）
- `runs/l7-implementation-notes.md`（新增）
- `results/l7/`（测量数据与汇报）

其余一律禁止（含既有测试、既有 bench、`server/`、`scripts/`）。

## 3. 禁止事项

- `git push`、删文件、改动既有实现/测试/基准、新依赖
- **禁止阅读 vLLM、FlashAttention、mlx-lm 或任何推理框架的 attention/kernel 实现源码**；PagedAttention 论文 §5 与 MLX 公开 API 文档/docstring 可读
- 不得改动测量协议口径（必须与 L6 bench 同协议，见第 5 节）

## 4. 实现要求

1. **直读物化路径**：`PagedKVPool` 新增按层扁平物化接口——输入一组 seq_id（或其 block table），单次向量化索引产出 block 对齐的 `(k_all, v_all)` 与每行有效长度；禁止逐行调用既有 `gather`。页有效载荷仍是惰性分配的，物化路径要处理"同一层内各页形状已知、尾部页不满"的实际情况。
2. **attention 直读变体**：`CausalSelfAttention` 新增方法（不改既有 `decode_batch*`），消费上述物化结果，用有效长度构造掩码（允许 -inf 掩码或等效手段），RoPE 位置语义与既有路径一致。
3. **引擎接线**：让 `BatchedEngine` 的 decode 走新路径（可用新 cache adapter 或子类，但对外语义——事件顺序、容量错误、cancel/fork、池回收——与 L6 完全一致；`_BatchedPagedKVCache` 既有类保留不删）。
4. **数值不变式**：新路径与既有路径逐 token 相等、logits atol 1e-5。测试模式参照 `tests/test_batched_engine.py`：等长/不等长 prompt、动态到达与完成、共享页 fork、训练 checkpoint（d512）场景都要覆盖新旧路径对照；singleton（B=1）路径也要验证不劣化。
5. **采样路径本任务不动**（T2 范围）。
6.  bench 协议与 L6 完全一致：3 预热/10 重复、median/p90、配对交替新旧路径、B=1/2/4/8 scaling、等长 32-token prompts、64 生成 tokens、保留全部原始样本、config hash、源码 SHA-256、负结果保留。直接复用/导入 `bench/l6_batching_bench.py` 的测量 harness 与 `load_model`（checkpoint `results/l3/ckpt-final-500.npz`），保证与 `results/l6/batching-bench-20261021T133044874890Z.json` 可逐字段比较。

## 5. 验收命令

```bash
.venv/bin/python -m pytest tests/ -q                 # 既有 119 项原样通过 + 新测试
.venv/bin/python bench/l7_paged_attn_bench.py        # 同协议消融，PASS 门槛见下
```

门槛：正确性全绿；B=8 聚合吞吐相对 L6 基线（1436.41 tok/s，同机同协议）**≥1700 tok/s** 记为预言支持，**<1550** 记为预言受挫——两者都要如实记录，不许调协议凑数。

## 6. 汇报要求

`-o` 文件写入：实际改动文件清单、B=1/2/4/8 新旧配对数字、相对 L6 基线的倍数、预言判定、遇到的坑与偏差。完整报告写 `results/l7/l7-t1-report.md`。阶段性 commit（简洁约定式信息），**不要 push**。
