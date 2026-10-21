# L5 复盘：量化与混合精度（对照式自研第 5 步）

> 证据链：`tasks/l5-weight-quant.md`、`tasks/l5-kv-bf16.md`、`tasks/l5-diff-analysis.md`、`runs/l5-quantization-principle.md`、`runs/l5-implementation-notes.md`、`runs/l5-diff-analysis.md`、`results/l5/`（含全部消融原始数据与 checksum）。

## 结论

五步闭环完成。91 项测试全绿。权重量化（INT8 per-channel / INT4 g64，PTQ 手写）、INT8 KV cache、bf16 混合精度训练全部落地，三轴 trade-off（内存/质量/速度）均有双跑证据。

## 原理预言 vs 实测

| 预言 | 实测 | 判定 |
|---|---|---|
| INT8 压缩 4.1×、INT4 7.5× | 3.99× / 7.53×（投影口径） | INT8 数值**推翻**（预言 4.1× 超过 4× 理论上限，规划层算术错误） |
| INT8 质量恶化 <1%、INT4 <5% | **+0.003%** / +0.59% | **成立**（INT8 近无损） |
| decode 提速有限甚至变慢 | INT8 无提速（0.72–1.01× 波动）、INT4 稳定 0.6× | **成立**（dispatch 主导第 5 次佐证） |
| INT8 KV 内存 ~2× 降、速度中性 | 存储 3.98×；decode **0.3×** | 内存**成立**、速度**推翻**（开销在 quantize-new 4.5ms，非 dequant） |
| bf16 MFU ≤+10% | **+29~34%**（MFU 43%） | **推翻**（利好方向） |
| bf16 内存 -35~40% | **-44.8%** | 略超预期 |
| bf16 loss 重合 <2% | **-0.006%** | **成立**（几乎完全重合） |

## 认知修正

1. **L0 开放问题 1 闭环**：MLX 在本机路径把窄精度输入转换为 float MMA fragment（macOS 15.8 上 NAX gate 关闭），单算子 bf16 无 2× 峰值；bf16 的真实收益在**流量减半**（GELU 实测 1.83×、activations 800→400 MiB）。单算子峰值比与端到端负载两口径就此统一——这也是 bf16 训练 +30% 与内存 -45% 的共同来源。
2. **量化提速是 kernel 耦合问题**：PTQ 的内存收益立即兑现，但提速需要融合 kernel（MLX packed qmm / gather-qmm 在计算内直接消费压缩权重）；教学形态 dequant→matmul 反贴 kernel。省内存 ≠ 提速，中间隔着 kernel 工程。
3. **KV 量化开销在"新 token 量化"而非"历史 dequant"**：消融显示 prequantized 输入可消除 80–84% 的 append 耗时——这指明了教学实现到生产实现的精确改造点。

## 负结果（保留）

- INT8 decode 高方差两次记录（0.72× 与 1.01×）全部保留；INT4 稳定变慢。
- 规划层 INT8 压缩比预言算术错误如实记录。

## 开放问题（带入后续）

1. KV INT4、层敏感度混合粒度 → 专项精度实验。
2. QAT → 独立训练迭代（如需）。
3. KV cache 转正条件配对测量（n≥256 cached/uncached，L4 挂来）→ 专项。
4. 压缩 KV 直读 + kernel batching + prefill/decode 混排 → **L6 服务化**。

## L6 选题建议

**服务化**（原候选序列确认，全管线收官层）：OpenAI 兼容 API、流式响应、多请求 kernel batching（L4 缺口）、压缩推理路径整合（L5 产物），把 L1–L5 的全部组件收敛为一个可对外服务的产品形态。
