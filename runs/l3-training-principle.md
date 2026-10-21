# L3 原理笔记：组装可训练的 GPT（模块 + 训练循环）

> 对照式自研第 1 步产物（原理先行）。写作时未阅读任何参照实现源码。L3 拆两个任务：T1 模块组装（本笔记 §1–§3）、T2 训练循环（§4–§6）。

## 1. 问题

把 L1/L2 的组件组装成一个前向可算 loss、反向可更新参数的完整 GPT，并让 loss 在真实语料上可信下降。

## 2. Transformer Block

**残差流（residual stream）**：`x = x + sublayer(LN(x))`。残差让梯度有直通路径（反向时 `∂L/∂x` 含恒等项），是深堆叠可训练的前提；每个子层只是向流中"写入"增量。

**Pre-LN vs Post-LN**：原始 Transformer 用 post-LN（`LN(x + sublayer(x))`），深层时 warmup 外仍易不稳；pre-LN 把归一化移进分支、主干恒等，梯度范数更可控，现代实现（GPT-2 起）默认 pre-LN。**选 pre-LN**，差异留作对照精读分析点。

**LayerNorm**：在特征维做归一（`μ,σ` 沿 d_model），不跨样本——BatchNorm 的 batch 依赖在序列建模中引入样本间耦合，且推理时统计口径麻烦。`LN(x) = γ⊙(x-μ)/√(σ²+ε) + β`。

**MLP**：`GELU(xW₁)W₂`，hidden = 4·d_model。约 2/3 参数在 MLP（每层 attn 4d² vs MLP 8d²）。GELU 相对 ReLU 在 0 附近平滑，实证更好（机制解释留对照精读）。

**Weight tying**：输出投影 = 嵌入矩阵转置（`logits = x @ Eᵀ`）。省 0.5M 参数，且嵌入同时接受"输入相似性"与"输出预测"两路梯度信号。

## 3. 组装的失效形式

残差漏接、LN 位置错（pre/post 混淆）、weight tying 转置维度错、logits 忘记用最终 LN 后的表示、causality 在堆叠后破损（L2 探针上移到模型级重测）。

## 4. 训练：loss、优化器、调度

**Cross-entropy**：`L = -log p_target`，实现用 log-sum-exp 保持数值稳定，对 target 做 gather。与 L1 的衔接：**bpb = L_nats / ln 2 / (bytes/token)**——模型侧 bpb 可回补 L1 开放问题 1（tokenizer 对下游 loss 的影响）。随机起点 loss ≈ ln(1028) ≈ 6.94 nats。

**AdamW**：`m=β₁m+(1-β₁)g`, `v=β₂v+(1-β₂)g²`，偏差修正 `m̂=m/(1-β₁ᵗ)`，更新 `θ -= lr·(m̂/(√v̂+ε) + λθ)`。相对 SGD+momentum：按参数自适应步长，对嵌入（稀疏梯度）与注意力（尺度差异）混合的模型关键。weight decay 解耦（AdamW 之 W）：不进入自适应分母，否则被 v 缩放扭曲。

**LR 调度**：线性 warmup（早期 v 估计不稳，大步长会发散）→ cosine 衰减至峰值 10%。**梯度裁剪**：global norm ≤ 1.0，防 loss spike。

## 5. 超参数与规模

`n_layer=4, d_model=512, n_head=8, ctx=256`，词表 1028（复用 L1）→ 参数 ≈ 12·4·512²/1e6 + 0.53 ≈ **13.1M**。batch=16、lr=3e-4、warmup=100 步、cosine 到 10%、clip 1.0、bf16 暂缓（fp32 先对齐，混合精度留 L5）。

## 6. Infra 预言（接 L0 判据）

1. **训练内存 ≈ 4× 参数**：fp32 权重+梯度+Adam m/v ≈ 13.1M×16B ≈ **210 MB**（activations 另计，ctx=256 batch=16 下应 < 1 GB）。实测验证。
2. **MFU**：训练 FLOPs ≈ 6·N·D（每 token 6 倍参数量）。13.1M 参数 → 78.7 MFLOP/token；L0 实测 matmul 峰值 ~14 TFLOP/s → 理论上限 ~178k tokens/s。预言本机小模型实测 MFU < 10%（小矩阵 + launch 开销，呼应 L0 带宽塌缩与 L2 softmax 占比）。
3. **loss 曲线**：应快速跌破 6.94 并持续下降；tiny-shakespeare 上 13M 模型几百步内可见明显下降。模型 bpb 与 L1 tokenizer 的 tokens/byte 结合后应能定量回答"tokenizer 差距的下游代价"。

## 7. 接口设计

```python
class LayerNorm:        # γ, β, eps=1e-5
class MLP:              # W1: d→4d, W2: 4d→d, GELU
class TransformerBlock: # pre-LN: x + attn(LN(x)); x + mlp(LN(x))
class GPT:              # embed + blocks + final LN + tied logits; __call__(ids) -> [.., vocab]
class AdamW:            # step(params, grads); 状态 m/v 与参数同形
def cross_entropy(logits, targets)  # log-sum-exp 稳定版
def clip_grad_norm(grads, max_norm)
def cosine_lr(step, total, peak, warmup)
```

约束：前向/反向用 MLX autograd（框架机制，非本层学习对象）；**优化器、调度、裁剪手写**（本层学习对象）。

## 8. Oracle 方案

- **PyTorch 同构模型**：numpy 桥接权重，逐层中间量 + logits + **逐参数梯度**对齐（fp32，atol/rtol 1e-4；梯度对齐是最强的组装正确性证据）。
- **手算用例**：单层 d_model=2 微模型的 LN/残差/CE 手算。
- 训练动态合理性：loss 曲线单调性宏观检查 + 过拟合单 batch 能力测试（单 batch 死循环应能压到接近 0）。

## 参考资料（纸面）

- Vaswani 2017（block 结构）；GPT-2 论文（pre-LN、tying 决策）
- Adam（Kingma & Ba 2015）/ Decoupled Weight Decay（Loshchilov & Hutter 2019）
- Chinchilla 风格的 6ND 训练算量近似（MFU 判据）
