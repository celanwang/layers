# L1 BPE reference implementation analysis

Scope: step 4 of the L1 learning cycle, after the blind implementation and
Oracle checks. This analysis changes no tokenizer implementation, tests,
dependencies, or existing measurement protocol. The comparison follows
`tasks/l1-diff-analysis.md`; all external source references below use pinned
commits. [Source manifest](../results/l1/diff-reference-sources.json) records
their URLs, SHA-256 hashes, and line counts; downloaded source stays outside
this repository.

## 1. Pre-tokenization and vocabulary ablation

### Controlled experiment

Run from the repository root:

```sh
.venv/bin/python bench/l1_pretoken_ablation.py --output results/l1/pretoken-ablation-rerun.json
```

The script refuses to overwrite existing results. The archived run is
[pretoken-ablation.json](../results/l1/pretoken-ablation.json). This is a
token-count experiment, not a latency benchmark: each case trains once and
evaluates the entire held-out suffix once. No timing conclusions are drawn
from these runs. Source/model hashes, installed versions, configuration hash,
requested/actual vocabulary sizes, and decode assertions are in the JSON.
The archived configuration hash is `dfa84ff23211`.

The input is the existing Tiny Shakespeare corpus: 1,115,394 UTF-8 bytes,
SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The split remains `len(text) * 9 // 10`: 1,003,854 training bytes, SHA-256
`a9e24e23a1ec77744dad26844bfd5a09b6e041954e1eef0000e7f24cba6db735`, and
111,540 evaluation bytes, SHA-256
`c54f3753a4e6e3c3d1759212815a7caf826e68a33021b25312984400bed40a1f`.
The script checks these against
[the original Oracle record](../results/l1/oracle-comparison.json) and
reproduces the 46,683 / 31,469 baseline before training.

The two core HF cases use tokenizers 0.23.3, `BPE()`, 1,024 ordinary tokens,
all 256 byte symbols, `min_frequency=1`, no special tokens or normalizer,
`ByteLevel(add_prefix_space=False)`, and the ByteLevel decoder. The **only
configuration difference is `use_regex`**; both train on `[training]`, one
complete iterator item. Chunking the false case would impose new boundaries
and invalidate this intervention. See the script's `cases`, `config`, and
training loop; [H1] lines 119–145 shows the regex switch followed by the same
byte mapping. This measures the effect of changing training **and** encoding
boundaries together, with retraining, as specified.

The true case uses HF's GPT-2 pattern, not cl100k's pattern: compare [H1]
lines 41–45 with [T3] lines 75–92. Nanochat additionally caps numeric chunks
at two digits ([N1] lines 23–26); cl100k uses three ([T3] line 89). Thus a
regex-on HF tokenizer is a controlled ablation arm, not a reproduction of
either reference tokenizer.

| Tokenizer | Regex | Requested / actual ordinary vocabulary | Held-out tokens |
|---|---|---:|---:|
| Own saved model | none | 1,024 / 1,024 | 46,683 |
| HF core treatment | none | 1,024 / 1,024 | 46,683 |
| HF core control | GPT-2 | 1,024 / 1,024 | 49,420 |
| HF vocabulary control | GPT-2 | 8,192 / 8,192 | 35,005 |
| HF vocabulary saturation control | GPT-2 | 100,256 / 20,319 | 33,444 |
| cl100k_base, tiktoken 0.14.0 | cl100k | pretrained / 100,256 | 31,469 |

Evidence: `baseline` and `hf_cases` in the ablation JSON. All listed encodings
roundtrip the complete evaluation suffix. The large HF case **does not reach
100,256**: on this corpus and these boundaries the trainer exhausts available
merges at 20,319 tokens; the requested size is not evidence of actual capacity.

The same-regex vocabulary intervention reduces tokens by 14,415 from 1,024 to
8,192, then by another 1,561 to saturation: 15,976 tokens in total. Even the
saturated model retains 1,975 more evaluation tokens than cl100k. These
subtractions establish a vocabulary-capacity effect **within this training
setup**; they do not identify the effect of replacing cl100k's training data,
pattern, or learned ranks. Evidence: the three HF true rows and cl100k row.

### What explains the original 48.35% gap?

The original excess is `46683 - 31469 = 15214`, or **48.345991%** of cl100k's
count. On the controlled 1,024-token HF comparison, the signed effect of
**omitting** pre-tokenization is `46683 - 49420 = -2737` tokens: **-17.990009%**
of the original excess, or **-8.697448 percentage points** relative to cl100k.
For this corpus and capacity, omitting regex boundaries improves the token
count; the earlier predicted penalty is not supported. This is a count
result, not a downstream model-quality conclusion. Evidence: `decomposition`
and the two 1,024-token `hf_cases` in the JSON.

A mechanism probe supports why boundaries can cost tokens here: **335 of
768 learned own pieces** split into multiple HF regex chunks when each piece
is pre-tokenized on its own. Held-out encoding uses these flagged pieces
15,119 times, including `", "` (511), `"the "` (244), and `"and "` (230).
See `cross_regex_vocab_probe` in the API probe JSON. Own can spend merges
across boundaries that HF enforces ([H1] lines 126–129 versus
`layers/tokenizer.py` lines 90–94). This standalone-piece probe illustrates
that mechanism; it neither counts context-dependent regex boundaries in the
full text nor attributes all 2,737 saved tokens to specific pieces, since
the ablation retrains both vocabularies.

After removing this negative contribution, the residual is
`49420 - 31469 = 17951` tokens: **117.990009%** of the original excess, or
**57.043440 percentage points** relative to cl100k. An exact accounting is:

```text
own - cl100k
 = (own - HF_no_regex_1024)                 implementation bridge
 + (HF_no_regex_1024 - HF_regex_1024)       regex intervention
 + (HF_regex_1024 - HF_regex_saturated)     vocabulary intervention
 + (HF_regex_saturated - cl100k)            remaining reference mismatch
 = 0 - 2737 + 15976 + 1975
 = 15214 tokens.
```

The vocabulary intervention accounts for **105.008545%** of the original
excess; the remaining mismatch is **12.981464%**. The zero bridge is observed
token-count agreement, not proof that own and HF have identical token IDs,
rules, or behavior on all inputs. Since effects depend on intervention order
and vocabulary size, this path decomposition is not a universal allocation.

Under the specification's two-bucket convention one could label the residual
**117.99% “vocabulary and reference-training setup”**, against **-17.99%
“no pre-tokenization”**. Calling all 117.99% a *pure vocabulary-size effect*
would not follow from this experiment: the additional vocabulary control
identifies 15,976 tokens on this path, while training distribution, regex and
ranks still differ from cl100k. The honest answer is therefore signed regex
effect **-2,737**, controlled vocabulary effect **15,976**, and unisolated
remainder **1,975**, rather than a forced causal split into exactly two
positive shares.

## 2. Training algorithm: rescan, position index, and unique chunks

Let `N` be initial byte length and `K = V - 256` the number of learned rules.
The following complexity statements are deductions from the cited loops,
assuming expected constant-time hash-table operations; they are not measured
speedup claims.

| Implementation | What each selected pair updates | Consequence and evidence |
|---|---|---|
| minbpe BasicTokenizer | Recounts every current edge, selects the maximum, builds a replacement list | Work is proportional to the sum of live sequence lengths over `K` rounds, bounded by `O(KN)`, with a fresh sequence allocation each round. [M1] lines 31–42; [M2] lines 13–41. |
| Own tokenizer | Occurrence sets over a linked sequence; changes at most three old and two new edges per actual replacement | Avoids a full corpus scan per rule; pays for occurrence indexing, sorted left-to-right replacement, and stale heap entries. `layers/tokenizer.py` lines 23–38 and 90–128. |
| rustbpe used by nanochat | Counts distinct regex chunks with multiplicities; indexes affected chunk IDs; rescans only those chunks and applies weighted count deltas | Repeated chunks share training work; a chunk merge still scans/rebuilds that chunk. Initial counting and ingestion use Rayon, while the merge loop over affected chunks is serial in this revision. [R1] lines 52–99, 134–165, 226–253, 360–413. |

For the own implementation, every successful replacement removes one live
position, so there are at most `N-1` such replacements. The initial edge index
is linear in `N`; subsequent edge updates and heap pushes are bounded by a
constant per replacement. Sorting occurrence snapshots and heap operations
give an expected `O(N log N)` bound for this **sequence/index/scheduler work**,
with `O(N)` position/index/heap storage. This excludes vocabulary byte-string
construction and storage (`layers/tokenizer.py` lines 61–72), whose cost also
depends on the sum of token byte lengths. It is not a blanket complexity
bound for every allocation made by `train`.

Both own and rustbpe resolve equal counts by ascending token-ID pairs: own
heap key `(-count, pair)` (`layers/tokenizer.py` lines 95–106), rustbpe's
comparator ([R1] lines 122–130). minbpe's `max(stats, key=stats.get)` instead
retains the first encountered maximum ([M1] lines 33–35; insertion order
comes from [M2] lines 19–21). Even equal vocab sizes therefore do not require
equal learned rules. HF sorts its byte-mapped Unicode alphabet ([H2] lines
345–358), whereas own starts with numeric byte IDs 0–255
(`layers/tokenizer.py` lines 61–63); tied pair priorities can differ.

Nanochat explicitly trains with rustbpe and constructs tiktoken for inference
([N1] lines 42–61). This separates training data structures from the deployed
encoder. For this L1 exercise, retaining readable Python edge updates exposes
overlap and stale-entry handling; the production reference instead adds
native data structures, chunk deduplication and parallel ingestion. These
are interpretations of the mechanisms in the table, not measured estimates
of how much each mechanism contributes.

### Measured anchor and public scale comparison

The existing committed [benchmark JSON](../results/l1/tokenizer-throughput-20261021T074112897335Z.json)
has configuration hash `62265f0f2948`: Apple M4 Max, 48 GiB unified memory,
macOS 15.8, CPython 3.12.15, synchronous CPU execution. It reuses L0 unchanged:
3 warmups, 10 measured runs, median = sorted sample 6, p90 = sorted sample 9.
For 1,003,854 training bytes and 768 merges, median is **1.013166 s**, p90
**1.058788 s**, median throughput **0.990809 decimal MB/s**. All samples and
source/configuration hashes remain in that artifact. No new throughput
measurement was substituted for it.

Karpathy's [nanochat launch report, “Train the tokenizer”](https://github.com/karpathy/nanochat/discussions/1)
reports roughly **one minute for 2 billion characters and a 65,536-token total
vocabulary**, in an 8×H100 cloud-node context. It does not specify CPU model,
CPU threads, warmups, repetitions, quantiles, or a configuration hash. The
training code is CPU Rust; the node's GPU count is context, not evidence that
BPE training used those GPUs. The manifest records the discussion author,
update timestamp and body hash. Current nanochat defaults instead specify
2B characters and 32,768 total tokens ([N2] lines 16–19), so the historical
claim must not be presented as a benchmark of the pinned current revision.

For an explicitly hypothetical ASCII input of 2 GB, maintaining the local
measured rate would take `2000 / 0.9908086 = 2018.55 s`, about **33.64 minutes**.
The public character rate is about 33.3 million characters/s. Their roughly
33.6-fold arithmetic contrast is **not an isolated Rust speedup or a forecast**:
hardware, regex boundaries, repeated-chunk distribution, vocabulary size,
input units and measurement methodology differ. The evidence supports why
the indexed/deduplicated native path is relevant to scale, while leaving its
same-machine speedup unmeasured. No rustbpe dependency was installed.

The previous HF no-regex attempt was terminated after more than 150 seconds
([implementation notes](l1-implementation-notes.md), “Retained unsuccessful
experiment”). A source-level explanation worth testing is that HF's training
word merge inserts/removes vector elements ([H3] lines 107–157, especially
143–145); without regex, one “word” is the million-byte prefix. Repeated
vector shifts can make that case expensive. This is a mechanism-based
hypothesis, not a profile or a general claim that native code is slower.

## 3. Encoding: merge rank differs from longest match

Own encoding selects the smallest available rank, then leftmost position,
invalidates stale edges, and only queues the two new neighboring edges
(`layers/tokenizer.py` lines 130–154). minbpe repeatedly rescans for the
smallest available merge index ([M1] lines 57–74): same rank semantics,
different scheduling cost. The distinction from greedy longest-match can be
demonstrated without changing any implementation.

| Input and rules in rank order | Own rank encoding | Greedy longest-match | tiktoken with the same byte ranks |
|---|---|---|---|
| `abc`; `bc→256`, `ab→257` | `[97,256]` = `a|bc` | `[257,99]` = `ab|c` | `[97,256]` |
| `abcd`; add `cd→258` | `[97,256,100]` = `a|bc|d` (3 tokens) | `[257,258]` = `ab|cd` (2 tokens) | `[97,256,100]` |

Evidence: the script's `counterexamples()` and JSON `encode_counterexamples`.
Both segmentations decode to the input, and own equals tiktoken in both
cases. `abc` is minimal for competing overlapping two-byte rules: fewer than
three input bytes cannot contain two distinct overlapping pair positions;
one rule provides no competing choice. The first example isolates different
segmentation, while the second also changes token count. Thus replacing rank
merging with a vocabulary longest-match search would change the model's
tokenization contract, not merely optimize this encoder.

## 4. Special tokens: ID allocation and text policy are separate

| Implementation | ID allocation | How literal marker text is handled |
|---|---|---|
| Own | Four IDs above the actual ordinary vocabulary; with `V=1024`, IDs 1024–1027 | Automatically recognizes all four spellings at the text boundary; ordinary training never inserts reserved IDs. `layers/tokenizer.py` lines 14–20, 52–56, 74–77, 90, 156–165. |
| cl100k / tiktoken | 100,256 mergeable entries and five explicitly assigned special IDs; maximum ID 100,276 | `encode` defaults to rejecting special spellings; callers can allow them or encode them as ordinary text. [T1] lines 82–127; [T3] lines 75–92. |
| Nanochat | Appends nine markers after ordinary ranks, subtracting nine from the requested total vocabulary | Ordinary text uses `encode_ordinary`; control IDs enter through explicit marker lookup/prepend/append and conversation rendering. [N1] lines 9–21, 42–61, 89–117, 168–191. |

Own and nanochat therefore both reserve a contiguous tail, but the size
parameter has different meaning: own `1024` means **1028 total IDs**;
nanochat `1024` requests **1015 ordinary + 9 special IDs**. cl100k's `n_vocab`
is **100277**, not its entry count: 100256 ordinary + 5 special = 100261
entries, leaving 16 unused IDs below the maximum. Evidence: own property at
`layers/tokenizer.py` lines 205–208, [N1] lines 46–54, [T1] constructor lines
44–52, and [API probe output](../results/l1/tokenizer-api-probes.json).

The same probe evaluates literal `<|endoftext|>`: own gives `[1025]`, tiktoken
default raises `ValueError`, explicit allow gives `[100257]`, and
`encode_ordinary` / `disallowed_special=()` give the seven ordinary IDs
`[27,91,8862,728,428,91,29]`. A minimal read-only reproduction is:

```sh
.venv/bin/python - <<'PY'
import tiktoken
from layers import BPETokenizer
own = BPETokenizer.load('results/l1/tokenizer-1024.json')
enc = tiktoken.get_encoding('cl100k_base')
s = '<|endoftext|>'
print('own', own.encode(s))
try:
    enc.encode(s)
except ValueError:
    print('default', 'ValueError')
print('allowed', enc.encode(s, allowed_special={s}))
print('ordinary', enc.encode_ordinary(s))
print('disallowed_empty', enc.encode(s, disallowed_special=()))
PY
```

The teaching implementation makes marker handling visible in one small API;
the references expose a caller-controlled distinction between content and
structure. The functional consequence is literal text becoming a control ID
under own `encode`, as the probe demonstrates. `<|unknown|>` is reserved but
is not an OOV fallback: all 256 bytes are available and `encode` only emits
that reserved ID for its literal spelling (`layers/tokenizer.py` lines
61–63, 156–165). This analysis proposes no API change.

## 5. Serialization: explicit schema versus rank table versus Python object

| Format | What is stored and validated | Teaching / portability consequence |
|---|---|---|
| Own deterministic JSON | Version, ordinary size, hex byte pieces, ordered ID pairs, and special-ID mapping; loader reconstructs and checks exact payload equality. `layers/tokenizer.py` lines 179–203. | Redundant fields make the learned construction inspectable and detect inconsistencies; hex and JSON can be read without Python object reconstruction. The saved 1,024-token model is 40,090 bytes; no cross-format size comparison is claimed. |
| tiktoken `.tiktoken` | Rank-sorted lines containing base64 bytes and an integer rank; loader reads that mapping. [T2] lines 147–171. | Language-neutral ordinary-token table, but regex and special IDs are supplied separately by the encoding constructor ([T3] lines 75–92). The rank file alone is not a complete tokenizer configuration. |
| Nanochat `tokenizer.pkl` | Pickles its constructed tiktoken encoding and loads it back. [N1] lines 63–68, 132–138. | Convenient for the Python/tiktoken runtime, but cross-language readers need Python pickle/object semantics. Custom tiktoken encoding state contains name, pattern, ranks, and special mapping; registered encodings may instead serialize by name ([T1] lines 409–428). |

The [API probe](../results/l1/tokenizer-api-probes.json) confirms that canonical
`json.dumps(payload, indent=2, sort_keys=True) + '\n'` equals the existing
40,090-byte model file and that reconstructing from its merge tuple produces
the same serialized bytes; its SHA-256 is recorded there. Own persistence is
therefore deterministically inspectable for this artifact, not a claim that
arbitrary JSON writers or reference pickle files are byte-identical.

## Acceptance and limits

The whitelist is limited to this document, `bench/l1_pretoken_ablation.py`,
and new records under `results/l1/`. Existing tokenizer/test/model/benchmark
files are read only. Per task section 5, the test suite is not rerun; validation
uses syntax parsing, the script's actual Oracle/roundtrip assertions, source
hash checks, JSON arithmetic and `git diff --check`. No new dependencies,
push, merge, or measurement-protocol edits are part of this task.

Completed checks: four HF training/evaluation cases and two tiktoken
counterexamples passed; all 19 pinned external source hashes matched freshly
fetched content; configuration/source/model hashes and the decomposition
identity passed. Every external source URL/line range and relative artifact
link in this document was checked. There is no configured repository lint
command; Python syntax parsing and diff whitespace checks are used here.

The required causal wording “attribute all remaining difference to vocabulary
size” needs qualification: the controlled vocabulary intervention measures
one contribution, while the cl100k comparison also changes pattern and
learned training distribution. The saturated HF result still differs by
1,975 tokens. Those confounders remain open; assigning them to vocabulary
size would exceed the evidence. The final report records this limitation
instead of presenting a forced two-cause partition.

## External source locations

- **[N1]** Nanochat `nanochat/tokenizer.py`, commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`: [pattern and trainer, lines 9–61](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py#L9-L61), [text API, lines 89–117](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py#L89-L117), [pickle save, lines 132–138](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py#L132-L138), [conversation markers, lines 168–191](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py#L168-L191).
- **[N2]** Same nanochat commit, [scripts/tok_train.py, defaults at lines 16–19 and timed scope at 47–52](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/tok_train.py#L16-L52).
- **[M1]** minbpe `minbpe/basic.py`, commit `1acefe89412b20245db5a22d2a02001e547dc602`: [training, lines 17–49](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py#L17-L49), [encoding, lines 57–74](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py#L57-L74).
- **[M2]** Same minbpe commit, [minbpe/base.py, count and merge helpers, lines 13–41](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/base.py#L13-L41).
- **[R1]** rustbpe `src/lib.rs`, commit `ddf848f6961a0655dc8693742fc338e5682c0d3b`: [chunk merge, lines 52–99](https://github.com/karpathy/rustbpe/blob/ddf848f6961a0655dc8693742fc338e5682c0d3b/src/lib.rs#L52-L99), [heap/counting, lines 122–165](https://github.com/karpathy/rustbpe/blob/ddf848f6961a0655dc8693742fc338e5682c0d3b/src/lib.rs#L122-L165), [incremental loop, lines 173–253](https://github.com/karpathy/rustbpe/blob/ddf848f6961a0655dc8693742fc338e5682c0d3b/src/lib.rs#L173-L253), [parallel chunk ingestion, lines 360–413](https://github.com/karpathy/rustbpe/blob/ddf848f6961a0655dc8693742fc338e5682c0d3b/src/lib.rs#L360-L413).
- **[T1]** tiktoken `tiktoken/core.py`, commit `4e71bbe0c078468e00fefbf94b39849389f346e5`: [constructor, lines 44–57](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/core.py#L44-L57), [special-token API, lines 82–127](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/core.py#L82-L127), [pickle state, lines 409–428](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/core.py#L409-L428).
- **[T2]** Same tiktoken commit, [tiktoken/load.py, rank-file dump/load, lines 147–171](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/load.py#L147-L171).
- **[T3]** Same tiktoken commit, [tiktoken_ext/openai_public.py, cl100k configuration, lines 75–92](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py#L75-L92).
- **[H1]** HF tokenizers v0.23.3, commit `301277072263fb3830eb66eec7e47adadf76f1dc`: [tokenizers/src/pre_tokenizers/byte_level.rs, pattern at 41–45, switch and mapping at 119–145](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/pre_tokenizers/byte_level.rs#L119-L145).
- **[H2]** Same HF commit, [tokenizers/src/models/bpe/trainer.rs, alphabet ordering at lines 345–358](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/models/bpe/trainer.rs#L345-L358).
- **[H3]** Same HF commit, [tokenizers/src/models/bpe/word.rs, training vector merge, lines 107–157](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/tokenizers/src/models/bpe/word.rs#L107-L157).
