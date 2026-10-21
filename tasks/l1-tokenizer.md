# 任务规格：l1-tokenizer

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含，严格按六要素执行；遇规格缺陷停下并在汇报中说明，不要自由发挥。

## 1. 研究问题

从零实现 byte-level BPE tokenizer：在 1MB 级语料上训练词表，编码/解码无损可逆，训练与编码吞吐可测量，并与外部 Oracle 对齐。可能的失效形式：合并顺序错误导致切分漂移、平局处理不确定导致训练不可复现、UTF-8 边界解码错误、特殊 token 泄漏进 BPE 词表。

## 2. 改动白名单

允许新增/修改：

- `layers/__init__.py`、`layers/tokenizer.py`（实现）
- `tests/test_tokenizer.py`
- `bench/l1_tokenizer_throughput.py`
- `data/`（语料，见第 4 节）
- `runs/l1-implementation-notes.md`（实现笔记：卡壳点、关键假设、与原理笔记 `runs/l1-bpe-principle.md` 的任何偏差）
- `results/l1/`（测量数据）

其余路径一律禁止改动（包括 `AGENTS.md`、`bench/l0_hw_profile.py`、既有 `runs/`、`results/l0/`）。

## 3. 禁止事项

- 引入规格未批准的新依赖（批准清单见第 4 节）；改既有依赖版本
- `git push`、删除任何文件、改动 `bench/l0_hw_profile.py` 的测量协议
- 阅读/参照任何 BPE 参照实现源码（tiktoken、HF tokenizers、nanochat 等）的**实现代码**——只允许调用其公开 API 做对照（本迭代为盲做实现，对照精读是后续独立步骤）
- 为通过测试而硬编码测试用例的期望输出

## 4. 实现要求

- 接口严格按 `runs/l1-bpe-principle.md` 第"接口设计"节：`BPETokenizer.train/encode/decode/save/load/vocab_size`。
- 算法约束：初始词表 256 字节；平局时取 pair 字典序小者（训练必须可复现：同语料同参数两次训练产物完全一致）；encode 按 merge rank 顺序合并；decode 用 `errors="replace"` 兜底；不做预分词（v1 有意为之）。
- 特殊 token：定义 4 个保留 id（`<|beginoftext|>`、`<|endoftext|>`、`<|padding|>`、`<|unknown|>`，id 从 vocab_size 起分配），不进入 BPE 合并空间。
- 语料：`data/tiny-shakespeare.txt`，从 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tiny-shakespeare/input.txt 下载（约 1.1MB）。
- 训练参数：vocab_size = 1024（256 字节 + 768 合并 + 4 特殊 token）。
- 批准的依赖（用 `uv pip install --python .venv/bin/python <pkg>` 装入 `.venv`）：`pytest`、`tiktoken`、`tokenizers`。
- save/load 格式自定，但必须 roundtrip 无损。

## 5. 验收命令（规划层将独立重跑）

```bash
.venv/bin/python -m pytest tests/test_tokenizer.py -q
.venv/bin/python bench/l1_tokenizer_throughput.py
```

测试必须覆盖：
1. 玩具语料手算合并序列精确对齐（自构造 ≤5 步合并的微语料，期望值人工可验证）
2. roundtrip 无损：ASCII、中文、emoji、混合空白、空串、超长文本
3. 训练可复现性：同语料两次训练，词表与合并规则完全一致
4. Oracle 对照：HF `tokenizers` 同语料同 vocab_size 训练 BPE，两者在 10 段采样文本上的 token 数之比落在 [0.5, 2.0] 且各自 roundtrip 无损；与 tiktoken cl100k_base 对比 bits-per-byte（数值记录即可，不设阈值）
5. save/load roundtrip 后 encode 结果与原实例一致

bench 要求：报告训练吞吐（MB/s）与 encode 吞吐（MB/s），沿用 L0 协议（预热 + 重复 + median/p90 + config hash），JSON 写入 `results/l1/`。

## 6. 汇报要求

汇报写入 `-o` 指定文件，包含：
1. 实际改动/新增文件清单（逐项）
2. 第 5 节全部命令的完整输出
3. bits-per-byte 对照数字（自研 vs tiktoken）及一句话解读
4. 与规格的任何偏差及原因；发现的规格缺陷
5. `runs/l1-implementation-notes.md` 的要点摘录

完成后按阶段提交（信息简洁、见名知意，如 `feat(tokenizer): byte-level BPE 实现与测试`），**不要 push**。
