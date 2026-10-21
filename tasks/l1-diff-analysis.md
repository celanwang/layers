# 任务规格：l1-diff-analysis（对照精读）

> 执行侧：Codex（gpt-6.1-sol）。本规格自包含。本任务为分析任务，**不修改任何实现代码**。

## 1. 研究问题

自研盲做 BPE（`layers/tokenizer.py`，vocab 1024，无预分词）与参照实现在设计上分歧在哪、各自为什么；教学性与生产性的取舍差异。必须回答：48% 的 token 数差距（held-out 46683 vs cl100k 31469）中，"无预分词"和"词表大小"各占多少。

## 2. 改动白名单

- `runs/l1-diff-analysis.md`（主产物）
- `results/l1/`（补充对照实验数据，可新增 JSON/脚本输出）
- `bench/l1_pretoken_ablation.py`（如需消融实验，可新增）

其余一律禁止，特别是 `layers/tokenizer.py`、`tests/`（分析阶段不做任何"向参照对齐"的代码改动）。

## 3. 禁止事项

- `git push`、删文件、改测量协议、改实现/测试代码
- 新依赖（只用 `.venv` 已有的：tiktoken、tokenizers、pytest、mlx）
- 泛泛而谈的文献综述——每个论点必须有证据（数字或具体源码位置）

## 4. 实现要求

允许并需要阅读参照实现：nanochat（`karpathy/nanochat` 的 `nanochat/tokenizer.py` 与 rustbpe 用法）、`karpathy/minbpe`、tiktoken 公开文档。可用 `uv tool run`/`pip download` 或 FetchURL 获取源码，但不得把源码复制进本仓库。

分析点（每点给出证据）：

1. **预分词消融（核心）**：用已装的 HF `tokenizers` 在同一训练语料、同一 BPE 词表大小（1024）下，分别训 `use_regex=True/False` 两版，在同一 held-out 上比 token 数——分离"无预分词"的贡献；剩余差距归因于词表大小并说明推理链。
2. **训练算法复杂度**：自研的链表+懒堆方案 vs minbpe 的朴素全量重扫 vs nanochat rustbpe（Rust）的设计取舍；用 bench 数字（自研 1MB 语料 ~1s）与文献/rustbpe 公开基准对比外推。
3. **encode 策略**：rank 顺序合并 vs longest-match，两者在什么输入下发散（构造一个最小发散用例并验证）。
4. **特殊 token 处理**：自研保留 id 段方案 vs tiktoken/nanochat 的方案差异。
5. **序列化**：自研确定性 JSON vs tiktoken 的 `.tiktoken` 排名文件 vs nanochat 的 pickle，教学性/可移植性取舍。

## 5. 验收命令

无代码改动，不跑测试套件。验收标准：`runs/l1-diff-analysis.md` 存在，5 个分析点齐全，预分词消融有具体数字（含实验脚本与 `results/l1/` 数据），每个论点附证据。

## 6. 汇报要求

`-o` 文件写入：五个分析点的结论各一句话、消融实验关键数字、引用的外部源码位置清单、与规格的任何偏差。完成后提交（如 `docs(l1): 对照精读分析`），**不要 push**。
