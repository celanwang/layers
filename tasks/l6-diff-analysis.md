# 任务规格：l6-diff-analysis（对照精读）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。分析任务，**不修改任何实现/测试代码**。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l6-serving`；Python 用 `.venv/bin/python`。

## 1. 研究问题

服务化层教学实现与生产实现的差距；定量解释 T1 遗留的 4.1× vs 理论 8× 吞吐缺口；给出全项目（L0–L6）距离"最小可用生产服务"的缺失件总清单。

## 2. 改动白名单

- `runs/l6-diff-analysis.md`（主产物）
- `results/l6/`（补充对照实验数据）
- `bench/l6_*_ablation.py`（如需消融，可新增）

其余一律禁止（含 `layers/`、`server/`、`scripts/`、`tests/`、既有 bench）。

## 3. 禁止事项

- `git push`、删文件、改实现/测试/脚本、新依赖
- 泛泛文献综述——每论点必须有证据（数字或具体源码/文档/论文位置）

## 4. 实现要求

允许并需要阅读参照：vLLM 公开文档/论文（continuous batching、Sarathi-Serve chunked prefill）、FastAPI/uvicorn 与 Python stdlib `http.server` 公开文档、Orca 论文、生产 serving 观测/限流公开实践。源码不入库。

已有证据可直接引用（不必重测）：`results/l6/batching-bench-*.json`（T1 拼批基准）、`results/l6/api-bench-*.json`（T2 API 基准）、`results/l6/api-overhead-profile.json`、`runs/l6-implementation-notes.md`、`results/l6/l6-t2-report.md`。

分析点（每点给证据）：

1. **4.1× vs 理论 8× 缺口归因**：T1 实测 B=8 聚合 1436 tok/s = 单路 4.1×。拼批后 kernel 数已不随 B 增长，项目既有结论是"dispatch 开销主导"。定量拆解剩余缺口：从 bench 原始数据拆 prefill/decode 占比、每步 kernel 时间随 B 的 scaling、padding 40.7% 的浪费占比；说明"再优化哪一项收益最大"，并判断该结论与 L0–L5 五次佐证的 dispatch 主导结论是否一致。
2. **stdlib vs FastAPI/uvicorn 生产选型**：`ThreadingHTTPServer` 每连接一线程模型的扩展上限（线程数、GIL 对 SSE 长连接的影响）vs uvicorn 单进程 async 事件循环；我们 3.7–4% 的 API 开销在 async 框架下预期如何变化；生产还需什么（worker 数、HTTP/2、keep-alive 池）。
3. **iteration 边界 prefill vs chunked prefill（Sarathi-Serve）**：我们整段 prefill 在迭代边界执行，长 prompt 阻塞同批 decode；Sarathi-Serve 的 token 预算混合调度如何做；用我们的实测数据估算一个 256-token prompt 到达时同批 8 路 decode 被阻塞的毫秒数。
4. **背压与 eviction 生产方案**：我们有界队列（ingress 32 / output 128）+ 拒绝 vs vLLM 的 preemption（recompute/swap）；admission control、优先级、SLO 驱动调度在生产中的做法；我们的 eviction 语义（断连即释放 KV）与生产的差异。
5. **全项目缺失件总清单**：L0–L6 全部结束后，距离"最小可用生产服务"还差什么。逐项列：生成终止（EOS/stop string）、跨请求 KV 前缀共享、观测（metrics/结构化日志/tracing）、鉴权与限流、TLS、推测解码、多 LoRA、graceful reload。每项给证据位置（本项目缺口 + 生产参照）与落层建议（L7 专项 or 不值得做）。

## 5. 验收命令

无代码改动，不跑测试套件（消融脚本可直接运行）。验收标准：`runs/l6-diff-analysis.md` 存在，5 点齐全，第 1、3 点有定量账，每论点附证据位置，第 5 点每项有明确落层建议。

## 6. 汇报要求

`-o` 文件写入：五点结论各一句话、4.1× 缺口拆解的关键数字、缺失件清单总数、外部资料位置清单、与规格的偏差。阶段性 commit（简洁约定式信息），**不要 push**。
