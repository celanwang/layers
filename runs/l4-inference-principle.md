# L4 原理笔记：推理——decode 循环、KV Cache 与 mini 推理引擎

> 对照式自研第 1 步产物（原理先行）。写作时未阅读任何参照实现源码。L4 拆两个任务：T1 decode + KV cache + 采样 + 流式 UTF-8（§1–§4）、T2 分页 KV 池 + 连续批处理（§5）。

## 1. Prefill 与 Decode：两种完全不同的负载

- **Prefill**：prompt 全序列一次前向，矩阵大、并行度高，是**计算受限**负载（同训练前向）。
- **Decode**：每步只处理 1 个新 token，但要用全部权重。算术强度 ≈ 2N FLOP ÷ (4N 字节权重 + KV 读取) ≈ **0.5 FLOP/byte**，比 L0 平衡点（32）低两个量级 → **纯内存受限**。
- **Roofline 预言**：本机 450 GB/s 带宽下，13.1M fp32 模型（52.5 MB 权重）理论上限 ≈ 450/0.0525 ≈ **8.5k tokens/s**；KV 读取随 n 增长会使上限缓降。预言实测**远低于**上限（每步大量小 kernel 的 launch/dispatch 开销主导——L0 带宽塌缩、L2 fast@n=128、L3 MFU 三次教训的同族现象），且这就是真实世界 decode 用批处理摊薄权重读取的动机（→ T2）。

## 2. KV Cache：用内存换计算

无 cache 时第 n 个 token 需重算全部 n 个位置的 K/V（每 token O(n) 重算，整体 O(n²)）；cache 把每步新 K/V 追加存储，注意力只对 1 个 q 与全部缓存 K/V 计算（每 token O(n) 读取）。代价：内存随 n 线性增长（L2 已精确验证曲线：2·n·d·dtype/层）。**正确性不变式**：cache 与无 cache 的逐步 logits 必须逐位一致——这是 T1 最强 Oracle。

## 3. 采样策略

`logits/T → softmax → 采样`。T<1 尖锐化（更保守），T>1 平坦化；top-k 截断后**必须重归一**；top-p（nucleus）按累积概率截断，动态候选数；greedy = argmax（T→0 极限）。失效形式：截断未重归一、温度作用在概率而非 logits、种子不可复现。

## 4. 流式 UTF-8（L1 观察清单兑现）

token 边界可能切开多字节 UTF-8 字符；逐 token `decode(errors="replace")` 会产生 ``。正确做法：增量解码器缓冲不完整的字节后缀，只有确认完整才吐出文本，replace 仅在流终止时兜底。

## 5. T2：分页 KV 池与连续批处理（mini 推理引擎）

- **PagedAttention 思想**（OS 虚拟内存类比）：逻辑 KV 序列 → 固定大小物理块（block）→ 块表按序列索引。免连续预分配；浪费只有每序列最后一块（内部碎片 ≤ block_size/2）；支持前缀共享与 fork。
- **连续批处理**（iteration-level 调度）：请求级批处理有气泡（最长序列决定步数）；连续批处理每个 iteration 独立调度——完成的退出、新请求插入，prefill 与 decode 可混排。
- **T2 预言**：分页内部碎片 ≤ block_size/2 per seq；批处理吞吐随并发近线性增长至拐点（权重读取被摊薄，roofline 上限不变但利用率上升）；单请求延迟不退化超过可测阈值。

## 6. 接口设计

```python
class KVCache:            # 每层 (k, v) 追加；seq_len 属性
class Sampler:            # temperature / top_k / top_p / seed；__call__(logits) -> id
def generate(model, tokenizer, prompt, max_new, sampler) -> Iterator[str]   # 流式吐文本
# T2:
class PagedKVPool:        # block_size 固定页池；alloc/free/fork；块表 per sequence
class Engine:             # add_request(prompt) / step() -> 活跃请求的新 token 事件
```

模型改造约束：`layers/attention.py`/`model.py`/`transformer.py` 只做**加法式**修改（新增可选 cache 参数），L2/L3 既有测试不改且必须保持全绿。

## 7. Oracle 方案（T1）

- **cache vs 无 cache**：同 seed 同 prompt，逐步 logits 逐位一致（1e-5）——黄金不变式。
- **采样语义**：temperature=0 与 top_k=1 均退化为 greedy；top-p 截断后概率和为 1；固定 seed 采样序列可复现。
- **流式 UTF-8**：构造跨 token 的 CJK/emoji 序列，增量输出拼接 == 一次性 decode，且无 ``。
- **生成烟测**：加载 L3 的 500 步 ckpt，greedy 生成 200 token，有限性 + 可 decode + 与训练分布同域（Shakespeare 风格宏观判断）。

## 参考资料（纸面）

- L0 带宽基线与平衡点；L2 KV 增长曲线；L3 MFU 口径讨论
- PagedAttention 论文（vLLM, Kwon et al. 2023，思想层面）
- 采样综述（Holtzman et al. 2020, top-p）
