# L6 原理笔记：服务化——kernel batching 与 OpenAI 兼容 API

> 对照式自研第 1 步产物（原理先行）。写作时未阅读任何参照实现源码。L6 拆两个任务：T1 kernel batching（§3–§4）、T2 HTTP/SSE 服务层（§1–§2、§5）。

## 1. 服务化的本质

把有状态的引擎（会话、KV、调度）通过无状态协议暴露：请求生命周期 = 接收 → 参数解析 → prefill → 流式 decode → 资源释放。服务层的正确性不只在协议格式，更在**并发隔离**（请求间零串话，L4 黄金不变式的服务化延伸）与**资源回收**（连接断开必须释放 KV 块，否则泄漏）。

## 2. OpenAI 兼容协议要点

- `POST /v1/chat/completions`：`{model, messages, stream, temperature, top_p, max_tokens}`；`GET /v1/models`。
- 流式：`text/event-stream`，每个 delta 一个 `data: {json}` 行块，以 `data: [DONE]` 终止；增量语义（`choices[0].delta.content`）而非全量重发。
- chat template：我们的 500 步 base 模型没有指令微调，messages 拼接用最小模板（role 标注换行拼接）——如实说明这是 base model 行为，不假装 chat 能力。
- 依赖决策：**stdlib `http.server` 手写 HTTP/SSE**（零新依赖，教学暴露协议细节；生产会选 FastAPI/uvicorn，留对照精读）。

## 3. Kernel batching（L4 缺口的正面对决）

B 个请求各 1 token 的 decode 拼成一批前向：权重读取从 B×52.5MB 摊薄到 52.5MB/步——L4 分析的理论上限 8×、带宽 ceiling 34.1k tok/s、现实投影 ~3k tok/s。变长 KV 的教学方案：**pad 到批内最长 + per-request 因果掩码**（paged pool gather 出连续视图后 pad；不追求 vLLM 的 kernel 直读页表）。

正确性不变式：**拼批 decode 与逐请求 decode 的逐步 logits 逐位一致（1e-5）**——padding 不得污染任何请求的输出（掩码错误的唯一允许值是零影响）。

## 4. 调度与混排

服务化下请求随机到达，prefill（长任务）会阻塞 decode 流水。教学形态：iteration 级循环——新到请求在 iteration 边界先做 prefill 再加入 decode batch；chunked prefill（把长 prefill 切片混入 decode 步）作为对照精读分析点，不实现。背压：客户端消费慢于产出时 SSE 队列应有界（防内存膨胀），超限断连并释放资源。

## 5. T1/T2 预言

1. **T1**：B=8 拼批吞吐 ≥ 逐请求 4×（L4 投影 ~3k tok/s）；单请求延迟（B=1）相对 L4 逐请求路径退化 <10%。
2. **T2**：API 层开销（JSON 序列化 + HTTP + SSE）在每 token 3ms 的 decode 面前应 <5%；并发 8 路流式请求零串话、全完成、资源全回收（连接断开后池占用归零）。

## 6. 接口设计

```python
# T1：layers/engine.py 加法式扩展
class BatchedEngine(Engine):
    def step(self):  # 批内所有活跃请求拼批前向，每请求取自己的 logits 行
# T2：server/app.py（新目录）
def make_server(host, port, engine) -> http.server.ThreadingHTTPServer
# POST /v1/chat/completions（stream=true SSE / false JSON）、GET /v1/models
```

## 7. Oracle 方案

- 拼批 == 逐请求逐步 logits（1e-5，黄金不变式的批次版）
- API 响应 schema 与 OpenAI 公开文档字段对照（手写校验，不引入 pydantic）
- 端到端：同 seed 同参数，`curl` 流式结果 == 直接调用 engine 结果

## 参考资料（纸面）

- L4 连续批处理收益分解与权重摊薄账（`runs/l4-diff-analysis.md` §1）
- OpenAI API 公开文档（SSE 与 delta 语义）
- chunked prefill / Sarathi-Serve 论文（思想层面，对照精读展开）
