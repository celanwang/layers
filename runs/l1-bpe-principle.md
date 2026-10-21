# L1 原理笔记：Byte-level BPE Tokenizer

> 对照式自研第 1 步产物（原理先行）。写作时未阅读任何参照实现源码。

## 问题

把任意文本映射为离散 token 序列，使序列既可逆还原原文，又比逐字节编码短得多（压缩），且词表大小可控。BPE（Byte Pair Encoding）用数据驱动的方式学出"哪些字节串值得成为独立 token"。

## 算法骨架

**训练**：
1. 初始词表 = 256 个字节（byte-level，GPT-2 风格的关键决策：对字节而非 Unicode 字符操作，天然无 OOV/unk，任何 UTF-8 文本可编码）。
2. 在语料上统计相邻 token 对频率，取最高频对合并为新 token，记录该合并规则（rank = 合并顺序）。
3. 重复直到词表达目标大小。

**编码**：把文本转为字节 id 序列，按 rank 顺序反复应用合并规则（每次找当前序列中 rank 最小的相邻对合并），直到无可合并。**顺序是正确性的核心**——乱序合并会得到与训练分布不同的切分。

**解码**：id → 字节串查表拼接，整体按 UTF-8 解码；序列中间可能出现不完整的 UTF-8 片段（流式场景的已知坑，记入 L4 推理迭代的观察清单），单次完整解码用 `errors="replace"` 兜底。

## 关键设计决策与失效形式

| 决策 | 选择 | 理由 / 风险 |
|---|---|---|
| 合并平局处理 | 固定 tie-break（频率相同取 pair 字典序小者） | 不固定则训练不可复现，config hash 失去意义 |
| 预分词（pre-tokenization） | **v1 不做**（全语料直接 BPE） | 已知代价：合并会跨越词边界，压缩率低于 GPT-2/4 的 regex 预分词方案；这个差距留到第 4 步对照精读（nanochat rustbpe / tiktoken）时量化分析，是本次迭代的预期学习点而非缺陷 |
| 特殊 token | 保留 id 段（≥ vocab_size），由 encode API 层面处理，BPE 本身永不生成 | 训练/推理边界清晰 |
| 压缩率指标 | bits-per-byte（nanochat 同款指标） | 可与 tiktoken 横向对比，量化 v1 不做预分词的代价 |

## 接口设计

```python
class BPETokenizer:
    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer"
    def encode(self, text: str) -> list[int]
    def decode(self, ids: list[int]) -> str
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> "BPETokenizer"
    @property
    def vocab_size(self) -> int
```

## Oracle 方案

- **手算玩具用例**：小微语料上的合并序列人工可验证（精确对齐）。
- **HF `tokenizers`**：同语料同词表大小训练 BPE，对比 roundtrip 无损性、分词统计分布（平局/预分词差异不要求逐 token 一致，要求数量级与趋势可解释）。
- **tiktoken cl100k_base**：held-out 文本上比 bits-per-byte，预期 v1 更差，差多少是测量结论。

## 参考资料（纸面）

- Sennrich et al. 2016, *Neural Machine Translation of Rare Words with Subword Units*
- GPT-2 论文（byte-level BPE 决策）+ Karpathy《Let's build the GPT Tokenizer》视频（仅概念）
- 测量判据沿用 L0：encode/train 吞吐按内存受限任务对待（算术强度 ≪ 32 FLOP/byte）
