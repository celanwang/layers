# L2 原理笔记：嵌入 + RoPE + 多头因果注意力

> 对照式自研第 1 步产物（原理先行）。写作时未阅读任何参照实现源码。

## 问题

把离散的 token id 序列变成上下文相关的连续表示：每个位置既要"看到自己之前的所有位置"（因果约束），又要让信息交换的权重由内容决定（注意力），还要让模型感知相对位置（RoPE）。

## 推导要点

**嵌入（Embedding）**：查表 `E[id]`, `E ∈ R^{V×d}`。本质是 one-hot 与 E 的矩阵乘，实现用 gather。可学习，随训练获得语义几何。

**缩放点积注意力**：`Attn = softmax(QKᵀ/√d_k)V`
- 点积 `q·k = Σ q_i k_i`，若分量近似独立零均值，方差随 d_k 线性增长 → logits 过大使 softmax 饱和（一端概率趋 1，梯度趋 0）。除以 √d_k 把方差归一，保持梯度流动。
- softmax 必须数值稳定：先减行最大值再 exp（数学上等价，防止上溢）。

**因果掩码**：对 `QKᵀ` 的上三角（j > i）置 -inf 再 softmax，使位置 i 只 attend 到 ≤ i 的位置。这是训练并行化的关键：全序列一次前向，但信息不泄漏到未来。失效形式：掩码方向写反（attend 未来）、off-by-one（漏掉自己）。

**多头**：`d_model = h × d_head`，Q/K/V 各自线性投影后按 head 切分，独立注意力，拼接后经 W_o 投影回 d_model。意义：不同 head 在不同表示子空间学不同关系（局部/远程、语法/语义）；单头等价于把多种关系压进一个 softmax 被迫竞争。

**RoPE（旋转位置编码）**：把 d_head 维两两配对，第 i 对按角度 `m·θ_i`（m = 位置，θ_i = base^(-2i/d_head)，base 常取 10000）做二维旋转。核心性质：`R_mᵀ R_n = R_{n-m}`——注意力 logit 只依赖相对距离 n-m，绝对位置被消去。失效形式：配对约定不一致（交错 vs 前后半）、θ 底数/指数写错、旋转方向反。

## 算术强度与 Infra 预言（接 L0 判据，平衡点 ≈ 32 FLOP/byte）

- QKᵀ 与 PV 两个 batched matmul：FLOPs ≈ 4·n²·d_head·h，naive 实现显存读写含 n×n 中间矩阵 → n 小时算术强度高（计算受限），n 大时 n² 中间矩阵的读写主导（滑向内存受限）。预言：本机小 n（≤1024）下 MLX naive 实现与 `mx.fast` 内核差距有限，n≥2048 拉开。
- KV 增长：每 token 每层 KV 占用 = 2·d_model·dtype 字节。fp32 d_model=512 时 4 KB/token/层——n 到数万时就是 GB 级，这是 L4 KV Cache 管理的动机，本次只测曲线。
- softmax 的两次全行读写（减 max、归一）是纯内存操作，小 n 下占比可观。

## 接口设计

```python
class TokenEmbedding:            # __call__(ids) -> [n, d_model]
class RotaryEmbedding:           # __call__(x, offset=0)，x: [..., n, d_head]
def attention_scores(q, k): ...  # 内部，含 scale + mask
class CausalSelfAttention:       # wq/wk/wv/wo；__call__(x) -> [n, d_model]
    # 返回 (out, attn_weights) 便于 Oracle 逐中间量对齐
```

约束：主实现只用 `mlx.core` 基础算子（matmul/softmax 手写数值稳定版）；`mx.fast.scaled_dot_product_attention`、`mlx.nn.RoPE` 只作 Oracle 对照，不进主路径。模块容器可用 `mlx.nn.Module`。

## Oracle 方案

- **PyTorch（本迭代起安装，作 Oracle 栈）**：同构实现 + 相同权重（numpy 桥接），逐中间量对齐（Q/K/V、scores、probs、out），fp32，atol/rtol 1e-5。
- **mx.fast.scaled_dot_product_attention / mlx.nn.RoPE**：fast path 对照主实现。
- **手算用例**：d_head=2、n=3 的微型注意力全程手算可验证；RoPE 用 90° 旋转特例验证。

## 参考资料（纸面）

- Attention Is All You Need（ Vaswani 2017，§3.2）
- RoFormer（Su et al. 2021，RoPE 推导）
- L0 机器平衡点 32 FLOP/byte 作受限判据基线
