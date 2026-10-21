# 任务规格：l6-api-server（L6-T2）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含；遇规格缺陷停下并在汇报中说明。
> **工作环境**：主仓库 `/Users/wangchen/Developer/layers`，分支 `l6-serving`；Python 用 `.venv/bin/python`。
> **前置**：L6-T1 已验收（`layers/engine.py` 的 BatchedEngine 可用）。

## 1. 研究问题

用 stdlib 手写 OpenAI 兼容 HTTP/SSE 服务层，把 BatchedEngine 暴露为 `POST /v1/chat/completions` 与 `GET /v1/models`。验证并发隔离（8 路流式零串话）、资源回收（连接断开池占用归零）、API 开销 <5%。失效形式：SSE 格式错误（缺 `data: ` 前缀/`[DONE]`）、delta 全量重发、连接断开未释放 KV 块、并发请求间 sampler seed 互相污染、线程安全问题（引擎单线程 step 与多连接线程的竞争）。

## 2. 改动白名单

- `server/app.py`、`server/__init__.py`（新目录）
- `layers/engine.py`（**仅加法式**：线程安全封装如需要）
- `tests/test_api_server.py`
- `bench/l6_api_bench.py`
- `runs/l6-implementation-notes.md`（追加 T2 段）
- `results/l6/`

其余一律禁止（含全部既有测试——原样通过；不引入任何 Web 框架）。

## 3. 禁止事项

- 新依赖（**stdlib only**：`http.server`/`json`/`threading`/`queue` 等）；`git push`；删文件；改既有测量协议
- 阅读任何推理服务框架实现源码（vLLM entrypoints / TGI / llama.cpp server / mlx-lm server / FastAPI 教程代码照搬）；OpenAI API **公开文档**允许阅读
- 修改既有测试；硬编码期望输出

## 4. 实现要求

- 严格按 `runs/l6-serving-principle.md` §1/§2/§5：`ThreadingHTTPServer` 手写路由；`POST /v1/chat/completions` 支持 `stream=true`（SSE，`data: {chunk}` + `data: [DONE]`，`choices[0].delta.content` 增量）与 `stream=false`（一次性 JSON）；`GET /v1/models`；加载 L3 ckpt（可选用 L5 INT8 量化路径，配置项）。
- chat template：最小模板（`role: content` 换行拼接），响应头/错误体按 OpenAI schema；base model 行为在 API 描述字段中如实标注。
- 并发桥接：引擎 step 循环在专用线程，请求线程通过有界 queue 递交/消费（背压：队列满 → 断连 + 释放该请求 KV 块）。
- 资源回收：连接断开（含客户端中途关闭）必须释放该请求的 paged KV 块——测试用池占用归零断言。
- bench（L0 协议，JSON 入 `results/l6/`）：
  1. API 开销：同参数生成，API 路径 vs 直接 engine 调用的每 token 耗时差（占比须 <5%）
  2. 并发扩展：1/4/8 路并发流式请求的聚合吞吐与单请求延迟
  3. 端到端：同 seed 同参数，API 流式结果 == 直接 engine 调用结果

## 5. 验收命令（规划层将独立重跑）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python bench/l6_api_bench.py
.venv/bin/python server/app.py --port 8611 & sleep 2 && curl -sN -X POST localhost:8611/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"layers-gpt","messages":[{"role":"user","content":"To be or not to be"}],"stream":true,"max_tokens":32,"temperature":0}' | head -20; kill %1
```

测试必须覆盖：
1. SSE 格式：`data: ` 前缀、JSON chunk schema、`[DONE]` 终止、delta 增量语义（拼接 == 完整文本）
2. schema 对照：响应字段与 OpenAI 公开文档（id/object/created/model/choices/usage）
3. 并发隔离：8 路不同 prompt 流式请求，结果各自与单独执行一致（零串话）
4. 资源回收：正常完成与客户端中途断开两种路径，池占用均归零
5. sampler 隔离：并发请求 seed/temperature 互不污染
6. L1–L6-T1 全部既有测试原样通过

## 6. 汇报要求

`-o` 文件写入：文件清单、三条验收命令完整输出（含 curl 实抓 SSE 流）、三组 bench 数字、§5 预言判定、与规格的偏差及原因。阶段性 commit（简洁约定式信息），**不要 push**。
