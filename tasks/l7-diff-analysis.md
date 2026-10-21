# 任务规格：l7-diff-analysis（对照精读）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。分析任务，**不修改任何实现/测试代码**。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l7-decode-perf`；Python 用 `.venv/bin/python`。

## 1. 研究问题

L7 两项干预（直读 page attention、单栅栏采样）合并 +30% 后：与生产实现的差距还剩什么、剩余缺口如何归因、"host 成本主导"假说的表述边界在哪里。

## 2. 改动白名单

- `runs/l7-diff-analysis.md`（主产物）
- `results/l7/`（补充对照实验数据）
- `bench/l7_*_ablation.py`（如需消融，可新增）

其余一律禁止（含 `layers/`、`server/`、`scripts/`、`tests/`、既有 bench）。

## 3. 禁止事项

- `git push`、删文件、改实现/测试/脚本、新依赖
- 泛泛文献综述——每论点必须有证据（数字或具体源码/文档/论文位置）

## 4. 实现要求

允许并需要阅读参照：PagedAttention 论文 §5、vLLM 公开文档/源码中 PagedAttention kernel 与采样路径描述（v0.10.2 文档、GitHub 公开源码可引用位置但不入库）、FlashAttention 论文（online softmax）、MLX `mx.fast` 公开 docstring。

已有证据优先引用（不必重测）：`results/l7/paged-attn-bench-*.json`、`results/l7/sampling-bench-*.json`（含规划层复跑两份）、`results/l7/l7-t1-report.md`、`results/l7/l7-t2-report.md`、`results/l6/diff-accounting.json`（端点模型）、`runs/l7-decode-perf-principle.md`。

分析点（每点给证据）：

1. **与生产实现的结构差距**：我们的"向量化物化"（compact bank 一次 `mx.take`）vs PagedAttention kernel 的零拷贝直读；我们的"lazy 图合并 + 单 eval"vs 生产采样的 batched logits processor。定量：物化输出仍是 O(B×padded_length) 拷贝，估算 B=8、平均 KV 长 128 时每次 forward 每层的物化字节数与 bank 中间拷贝字节数；说明 kernel 融合消除的是哪一项。
2. **剩余缺口归因**：L6→T2 合并 +30%（B=8 1893–1918 tok/s）后，相对 L4 投影的 8× 目标（2792 tok/s）还剩 ~32% 缺口。用 T2 后的 B=1/2/4/8 数据重新拟合端点模型 T(B)，对比 L6 的 1.641+0.5022B：行相关项斜率降了多少、截距变了多少；逐项讨论剩余行相关成本候选（padded attention 算术、page 写入/元数据 Python 循环、调度开销、每步 logits 行拆分 `logits[row,0]`），指出哪一项证据最弱。
3. **B=1/B=2 回归的机制解释**：新路径在小批量回归 -0.9%~-5.1%（噪声大），T1 B=1 回归在 1.3%~8.8% 间漂移。从物化固定成本（bank 构建、编译图常数项）vs gather 在小 L 时的实际成本入手解释；讨论生产系统"按批量/长度选路径"的取舍（vLLM 是否有类似分派，给文档证据）。
4. **假说判决的表述边界**：controlled intervention 支持"host 小操作/同步成本主导"，但明确列出不能声称的内容——无 Metal trace、greedy（temperature=0）单一工作负载、单一 13M 模型、统一内存单设备；说明什么证据能把判决再收紧一档（如 Metal trace、随机采样负载复测、更大模型）。
5. **后续专项建议**：评估三件事的性价比并给落层建议——(a) Metal trace/系统级 profiling（ Instruments 或 mx metal 工具链）补最后一环证据；(b) `mx.fast.scaled_dot_product_attention` 接入拼批 decode 的可行性（变长掩码支持度，docstring 证据）；(c) L5 INT8 KV 与 T1 物化路径整合（压缩页直读 vs 先 dequant 再物化）。每项给出预期收益的量级估算与所需工作量。

## 5. 验收命令

无代码改动，不跑测试套件（消融脚本可直接运行）。验收标准：`runs/l7-diff-analysis.md` 存在，5 点齐全，第 1、2 点有定量账，每论点附证据位置，第 5 点每项有明确落层建议。

## 6. 汇报要求

`-o` 文件写入：五点结论各一句话、端点模型新旧参数对比、剩余缺口最大候选、外部资料位置清单、与规格的偏差。阶段性 commit（简洁约定式信息），**不要 push**。
