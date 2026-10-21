# L5 原理笔记：量化与混合精度

> 对照式自研第 1 步产物（原理先行）。写作时未阅读任何参照实现源码。L5 拆两个任务：T1 权重量化推理（§1–§4）、T2 KV cache 量化 + bf16 训练对照（§5–§6）。

## 1. 量化的本质

用仿射映射 `w ≈ s·(q − z)`（整数 q、scale s、零点 z）把 fp32 压到 b 位整数。均匀量化误差 ≤ Δ/2（Δ = 该组动态范围 ÷ (2^b−1)）——误差是**舍入噪声**，其大小由分组粒度与离群值决定，与"模型大小"无关。核心 trade-off：每参数 bit 数 × 质量损失的曲线。

## 2. 设计空间

- **对称（z=0）vs 非对称**：权重分布近似零中心 → 对称足够且省零点；激活（ReLU/GELU 后偏正）才需要零点。**选对称**。
- **粒度**：per-tensor（一个 scale，误差被离群通道放大）→ per-channel（每输出通道一个 scale，INT8 甜点）→ per-group（每 64/128 元素一个 scale，INT4 必需；group=64 时 INT4 实际 4.25 bit/param——scale 开销不可忽略）。
- **PTQ vs QAT**：训练后量化直接压权重，无需重训；QAT 把量化噪声纳入训练。**L5 只做 PTQ**，QAT 留作对照精读话题。
- **敏感件保护**：嵌入/LN 参数/输出头对量化最敏感——教学实现只对 block 内投影矩阵量化，嵌入（含 tied head）与 LN 保持 fp32。这是生产混合精度策略的最小版。

## 3. 量化前向的教学形态

存储 `q(int) + scales(fp16)`，前向时 dequant 回 fp32/bf16 再 matmul。**这不是生产形态**（生产用融合 kernel 直接对 int 做乘加），但如实暴露量化语义与开销分解；dequant 开销 vs 权重读取节省的净效应正是测量点。与 MLX `mx.quantize`（如存在）做 Oracle 对照。

## 4. T1 预言

1. 内存：INT8 ≈ 4.1×、INT4(g64) ≈ 7.5× 压缩（含 scale 开销）。
2. 质量：INT8 heldout loss 恶化 <1%；INT4(g64) 恶化 <5%（group 内离群值主导）。
3. 速度：**decode 提速有限甚至变慢**——L4 已证明 decode 的 92%+ 时间在 dispatch/小算子而非权重读取，dequant 反而增加 kernel。预言 INT8 decode 变化 ∈ [0.8×, 1.3×]；若推翻则修正对 decode 瓶颈的归因。

## 5. KV cache 量化（T2a）

KV 是推理内存的**增长项**（L2：2·n·d·dtype/层线性增长）。per-token 对称 INT8 量化 K/V：误差经注意力 softmax 传播，但注意力对单点扰动平滑（权重 ≤1）→ INT8 KV 通常近无损。预言：长上下文（n≥512）decode 内存降 ~2×，logits 偏差可测但小；对速度影响中性（KV 读取仅占权重载荷 4%@n=128，L4 已测）。

## 6. bf16 训练对照（T2b，接 L3 缺失件）

bf16：8 指数位（范围同 fp32）+ 7 尾数位——牺牲精度保范围，训练通常**无需 loss scaling**（fp16 才需要）。混合精度形态：前向/反向 bf16 + master 权重 fp32 + 优化器状态 fp32。预言（接 L0 实测）：M4 上 bf16 matmul 仅快 ~1.1×（L0：14.75 vs 13.14 TFLOP/s），**MFU 提升 ≤10%**；峰值内存降 ~35–40%（activations bf16 + 部分四态）；loss 曲线与 fp32 基本重合（<2% 终值差）。bf16 无感的根源（L0 半精度加速比异常）正是本预言最可能的翻车点。

## 7. 接口设计

```python
def quantize(w, bits, group_size) -> (q, scales)      # 对称 per-group
def dequantize(q, scales, bits, group_size) -> w_hat
class QuantizedLinear:     # 存 q/scales，forward = dequant→matmul
def quantize_model_(model, bits, group_size)  # block 内投影，嵌入/LN 跳过
# T2a
class QuantKVCache:        # per-token INT8，接口与 KVCache 对齐
# T2b：scripts/train.py 加 --precision {fp32,bf16}（既有文件仅加法式修改）
```

## 8. Oracle 方案

- quant/dequant 手算用例 + roundtrip 误差 ≤ Δ/2 断言
- MLX `mx.quantize/dequantize`（公开 API）输出对照（如可用）
- 质量轴：同一 ckpt 的 heldout loss（fp32 vs INT8 vs INT4），测量即对照
- bf16 训练：同 seed 同 200 步，loss 曲线与 fp32 基线对照（L3 产物）

## 参考资料（纸面）

- L0 半精度 matmul 数据；L3 内存四态与 MFU 口径；L4 decode 瓶颈归因
- GPTQ/AWQ 论文（PTQ 思想层面，T2 后对照精读展开）
