# L6 复盘：服务化（对照式自研第 5 步）

> 证据链：`tasks/l6-kernel-batching.md`、`tasks/l6-api-server.md`、`tasks/l6-diff-analysis.md`、`runs/l6-serving-principle.md`、`runs/l6-implementation-notes.md`、`runs/l6-diff-analysis.md`、`results/l6/`（含 T1/T2 基准、对照精读派生账与全部原始数据）。

## 结论

五步闭环完成，L0–L6 全管线收官。119 项测试全绿。T1 kernel batching（BatchedEngine，拼批 paged decode）与 T2 stdlib HTTP/SSE API 服务（OpenAI 兼容子集）全部落地，引擎专用线程 + 有界队列背压、断连即释放 KV、零新依赖。

## 原理预言 vs 实测

| 预言 | 实测 | 判定 |
|---|---|---|
| T1：B=8 拼批吞吐 ≥ 逐请求 4× | **4.115×**（1436.41 vs 349.04 tok/s，规划层复跑） | **成立**（压线过） |
| T1：B=1 延迟退化 <10% | 反而**改善 24.6%**（singleton 走拼批路径摊薄 dispatch） | **推翻**（利好方向） |
| T2：API 开销 <5% | **3.70%**（执行层）/ **4.08%**（规划层复跑），净 0.084–0.091 ms/token | **成立** |
| T2：8 路流式零串话、全完成、资源全回收 | 182 项直接引擎对照全等；断连/溢出/关停测试池占用归零 | **成立** |
| （隐含）padding 40.7% 会显著拖累吞吐 | uneven 用例吞吐惩罚仅 **2.4%** | **推翻**（见认知修正 2） |

## 认知修正

1. **8× 是说明性目标，不是算法保证**。dense 投影图共享后 kernel 数不随 B 增长，但 sampling 逐请求标量同步（B=8 每步 8 次 sampler 求值）、KV append/gather（每 forward 32+32 次行操作）仍随 B 线性增长。端点模型 T(B)=1.641+0.5022B ms：B=8 行相关项占 71%。缺口恒等式 173.083 = −48.369（singleton 已省）+ 221.452（批增长成本）。最高价值改造点是**批式采样 + 直读 page 的 attention**（PagedAttention 式），而非 padding 消除。
2. **padding 浪费的是 attention slot，不是全局时间**。T1 scaling 用例等长 prompt 零 padding，173ms 缺口与 40.7% 无关；uneven 用例 attention matmul 仅占简化 FLOP 3.84%，padding 把它放大 68.75% 也只值 ~2.6%。
3. **"dispatch 主导"需限定语义**。五次佐证是 host 端小操作/同步成本的有界解释，不是纯 launch 开销的 Metal trace 证据；L3 大 batch 35.7% MFU 是该解释的边界。L6 未提供新 trace，这是明确的证据上限。
4. **stdlib 选型达成教学目的**（线程所有权与 SSE 帧可审查、零依赖），async 框架是连接/并发改进而非现成提速：3.7–4% 开销不支持预测固定 async 收益。生产化是传输层换代（单 async 前端 + 单引擎 owner 起步），不是渐进补丁。
5. **我们的背压是拒绝+中止，不是可恢复 eviction**。vLLM 的 preemption（recompute 优先于 swap）保留请求状态可恢复；统一内存下"swap 到 CPU"不产生新容量，可借鉴的是 recompute 而非 CUDA swap。

## 负结果（保留）

- T2 首版串行基准 API 开销 **9.37% 超标**（每 token 一次 EOF 轮询系统调用），修到 3.70%；失败数据留存 `results/l6/api-bench-20261021T143755909668Z.json`。
- T1 早期一次 batching gate 失败记录（吞吐惩罚 +3.306%）保留在五份基准原始数据中。

## 开放问题（带入后续）

对照精读给出 11 项缺失件清单（`runs/l6-diff-analysis.md` §5），按落层建议归并：

1. **最小服务范围（L7 候选）**：EOS/stop 终止、基础观测（readiness/TTFT/ITL/池占用）、鉴权限流与 TLS（对外暴露时）、优雅重载与排水。
2. **性能专项（证据驱动再做）**：跨请求 KV 前缀缓存、token 预算调度 + chunked prefill、压缩 KV 直读 + L5 整合。
3. **明确不值得做**：推测解码、多 LoRA（13M dispatch 敏感教学模型，无真实负载假设）。
4. **历史挂账专项**：KV cache 转正条件配对测量（n≥256，L4 挂来）、KV INT4/层敏感度、QAT。

## 全管线收官

L0 硬件画像 → L1 tokenizer → L2 attention → L3 模型与训练 → L4 推理与分页 KV → L5 量化与混合精度 → L6 服务化，每层均按"原理先行→盲做→Oracle 对齐→对照精读→测量复盘"闭环，测试累计 119 项。贯穿性结论最终表述：本机小负载性能由 **host 端小操作与同步成本主导**（有界解释，边界为大 batch 训练负载）；从 tokenizer 到 OpenAI 兼容 API 的每一段都有可复核的测量证据。
