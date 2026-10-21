# L1 implementation notes

## Scope and assumptions

- Implemented from `runs/l1-bpe-principle.md` and task specification v2. No
  tiktoken, Hugging Face, nanochat, or other BPE implementation source was read.
  Oracle interaction uses public APIs and their API documentation only.
- `vocab_size=1024` means 256 byte tokens plus 768 ordered merges. Special IDs
  1024–1027 represent begin/end/padding/unknown in that order; they are excluded
  from BPE training and merging. Literal spellings are recognized by `encode`
  and restored by `decode`. Training sees their UTF-8 spelling as ordinary
  bytes, never as reserved IDs. There is no linguistic pre-tokenization.
- When a tiny corpus exhausts all pairs, training stops early and reports its
  actual BPE vocabulary size. Reserved IDs begin at that actual size.
- The original download URL and vocabulary arithmetic were corrected in v2
  before implementation. The requested report is `/tmp/l1-tokenizer-report.md`.
- The corpus is split at `len(text) * 9 // 10`: the first 90% (about 1 MB)
  trains both implementations; the final 10% is held out for compression
  comparison. Ten deterministic 2048-character samples come from that suffix.
- The downloaded corpus is preserved byte-for-byte (SHA-256
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`).
  A whole-diff whitespace check flags its existing trailing spaces on lines
  2478–2479. Source/document whitespace checks exclude this immutable input;
  the corpus is not normalized to satisfy a code-style check.

## Implementation decisions and difficulties

- A full recount and replacement for each of 768 merges would repeatedly scan
  the million-byte corpus. The implementation instead keeps pair occurrence
  sets over a linked list of byte positions. Only the edges touching each
  replacement change. A lazy heap orders pairs by `(-frequency, pair)`.
- Overlapping occurrences are the main correctness hazard: `aaaaa` initially
  has four `aa` edges but only two disjoint replacements. Sorting each chosen
  pair's positions gives deterministic left-to-right replacement; stale
  overlapping positions are skipped. Heap entries are checked against the
  current count before selection.
- Encode uses a separate heap ordered by merge rank and then position. It
  validates that both endpoints still form the recorded edge before merging,
  avoiding stale candidates. It does not choose the longest vocabulary piece.
- UTF-8 is decoded after concatenating all token byte strings. Splitting inside
  a code point is safe for complete sequences; incomplete input uses the
  required replacement fallback. Invalid IDs raise `ValueError`.
- Persistence stores ordered pairs, hex byte pieces, vocabulary size, special
  IDs, and a format version in deterministic JSON. Loading reconstructs the
  vocabulary from the pairs and checks that all stored fields agree.

## Verification

- Hand-calculated examples cover ties and overlapping pairs with three merges.
  An independently written direct-definition algorithm checks training and
  rank-based encoding on 120 deterministic random micro-corpora.
- Full-size training is repeated; vocabulary, merges, and serialized bytes
  must agree. Roundtrips include multilingual text, whitespace, empty input,
  long input, special names, persistence, and partial UTF-8 decoding.
- HF uses its public ByteLevel mapping with `add_prefix_space=False` and
  `use_regex=True`, a complete byte alphabet, and the same corpus and BPE size.
  IDs need not match because the byte alphabets and tie-breaking may differ.
- Bits-per-byte is explicitly a tokenizer-only uniform-ID proxy:
  `token_count * log2(total_id_space) / UTF8_bytes`. It includes reserved IDs
  for both tokenizers and is not a measured language-model cross-entropy or an
  actual entropy-coded file size. No language model is part of this task.
- The completed test suite passes all 29 checks. Across ten held-out samples,
  own/HF token-count ratios are 0.921053–0.965594, inside [0.5, 2.0]. On the
  complete 111,540-byte held-out suffix, own BPE emits 46,683 tokens and
  cl100k_base emits 31,469. The proxy is respectively 4.187669 and 4.687237
  bits/byte: the smaller ID space offsets the larger token count under this
  formula, rather than establishing better language-model compression.

## Retained unsuccessful experiment

The first pytest run reached 28 passing checks, then the public HF trainer was
still using approximately 99% of one CPU after more than 150 seconds with
`ByteLevel(use_regex=False)` and the entire training prefix as one sequence.
The run was terminated rather than treated as a pass. The Oracle configuration
was changed to its conventional regex boundaries; the specification requires
the same training corpus and BPE size, but only the self implementation must
omit pre-tokenization. This change does not relax the token-count ratio or
roundtrip assertions. No HF source was consulted to diagnose the slowdown.

A benchmark import smoke check via `runpy.run_path` initially failed because
the script-directory-only import of L0 helpers was not available in that
invocation. Importing `bench.l0_hw_profile` through the repository root fixes
both direct-script execution and the smoke check without changing L0.

## Relation to the principle note

The training, rank order, byte reversibility, tie rule, and reserved-ID boundary
follow the note. The note predicts a compression penalty from omitting regex
pre-tokenization; this comparison also changes vocabulary size and training
data, so it cannot isolate that cause. Measured results and open questions will
be recorded after acceptance. Source-level comparison remains a later task,
as required by this iteration's explicit prohibition.
