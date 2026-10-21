"""L1 correctness and public-API Oracle checks (no reference source inspection)."""

import hashlib
import json
import math
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pytest
import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from layers import BPETokenizer
from layers.tokenizer import SPECIAL_TOKENS

ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 1024


@pytest.fixture(scope="session")
def corpus_split():
    text = (ROOT / "data/tiny-shakespeare.txt").read_text(encoding="utf-8")
    boundary = len(text) * 9 // 10
    return text[:boundary], text[boundary:]


@pytest.fixture(scope="session")
def tokenizer(corpus_split):
    return BPETokenizer.train(corpus_split[0], VOCAB_SIZE)


@pytest.mark.parametrize(
    ("text", "merges", "encoded"),
    [
        # ab has frequency 3; then (ab, ab) has frequency 2, replaced leftmost.
        ("ababab", [(97, 98), (256, 256), (257, 256)], [258]),
        # aa overlaps: aa aa a. At the next tie, (256, 97) < (256, 256).
        ("aaaaa", [(97, 97), (256, 97), (256, 257)], [258]),
        # All initial counts are 1: (a, a) beats (a, b) and (b, a).
        ("baab", [(97, 97), (98, 256), (257, 98)], [258]),
    ],
)
def test_hand_computed_merges(text, merges, encoded, tmp_path):
    model = BPETokenizer.train(text, 256 + len(merges))
    path = tmp_path / "toy.json"
    model.save(str(path))
    payload = json.loads(path.read_text())
    assert payload["merges"] == [list(pair) for pair in merges]
    assert model.encode(text) == encoded
    assert model.decode(encoded) == text


def test_encoding_respects_rank_not_longest_piece():
    # Training: ab(3) -> bc(2). On abc, ab must win although bc also exists.
    model = BPETokenizer.train("ababab cbcbc", 258)
    assert model._merges == ((97, 98), (98, 99))
    assert model.encode("abc") == [256, 99]


@pytest.mark.parametrize(
    "text",
    [
        "Hello, ASCII!\x00\x7f",
        "中文分词与字节边界，世界你好。",
        "😀🧑🏽‍💻🚀🏳️‍🌈",
        " \t\n\r\n\v\f multiple   spaces\u00a0\u3000",
        "",
        ("long text 中文🙂\n\t" * 70000),
        "e\u0301 ≠ é; مرحبا; Привет; नमस्ते",
    ],
)
def test_roundtrip(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in ids)


def test_all_bytes_and_utf8_replacement():
    model = BPETokenizer.train("", 256)
    assert model._vocab == tuple(bytes([value]) for value in range(256))
    assert model.decode(list(range(256))) == bytes(range(256)).decode("utf-8", "replace")
    assert model.decode([0xE4, 0xB8, 0xAD]) == "中"
    assert model.decode([0xE4, 0xB8]) == "�"


def test_special_ids_are_outside_bpe(tokenizer):
    assert tokenizer.vocab_size == VOCAB_SIZE
    assert len(tokenizer._merges) == 768
    for rank, pair in enumerate(tokenizer._merges, start=256):
        assert all(0 <= value < rank for value in pair)
    text = "中文" + "".join(SPECIAL_TOKENS) + "🙂"
    assert tokenizer.encode("".join(SPECIAL_TOKENS)) == list(range(1024, 1028))
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(list(range(1024, 1028))) == "".join(SPECIAL_TOKENS)
    # Even a training corpus containing the spellings cannot create reserved IDs.
    special_model = BPETokenizer.train("".join(SPECIAL_TOKENS) * 5, 300)
    assert special_model.encode(SPECIAL_TOKENS[0]) == [special_model.vocab_size]
    assert all(value < special_model.vocab_size for pair in special_model._merges for value in pair)


@pytest.mark.parametrize("vocab_size", [255, 0, -1, 256.0, True])
def test_invalid_vocab_size(vocab_size):
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size)


@pytest.mark.parametrize("text", ["", "a", "🙂"])
def test_training_stops_when_no_pairs_remain(text):
    model = BPETokenizer.train(text, 1024)
    assert model.decode(model.encode(text)) == text
    assert 256 <= model.vocab_size < 1024


@pytest.mark.parametrize("token_id", [-1, 1028, 1.0, True])
def test_invalid_token_ids(tokenizer, token_id):
    with pytest.raises(ValueError):
        tokenizer.decode([token_id])


def test_training_reproducibility_and_serialization(tokenizer, corpus_split, tmp_path):
    second = BPETokenizer.train(corpus_split[0], VOCAB_SIZE)
    assert second._vocab == tokenizer._vocab
    assert second._merges == tokenizer._merges
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    tokenizer.save(str(first_path))
    second.save(str(second_path))
    assert first_path.read_bytes() == second_path.read_bytes()
    loaded = BPETokenizer.load(str(first_path))
    for text in (corpus_split[1], "中文🙂", "", "".join(SPECIAL_TOKENS)):
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == text
    loaded.save(str(second_path))
    assert first_path.read_bytes() == second_path.read_bytes()


def test_load_rejects_inconsistent_format(tokenizer, tmp_path):
    path = tmp_path / "model.json"
    tokenizer.save(str(path))
    payload = json.loads(path.read_text())
    payload["vocab"][256] = "00"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent"):
        BPETokenizer.load(str(path))
    payload["version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        BPETokenizer.load(str(path))


def test_incremental_algorithm_against_direct_definition():
    """An independently written, slow mathematical baseline for small inputs."""
    rng = random.Random(20261021)
    for _ in range(120):
        text = "".join(rng.choices("aaabbc \n中🙂", k=rng.randrange(1, 90)))
        size = rng.randrange(256, 281)
        ids = list(text.encode("utf-8"))
        expected = []
        for new_id in range(256, size):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            expected.append(pair)
            replaced, position = [], 0
            while position < len(ids):
                if position + 1 < len(ids) and tuple(ids[position:position + 2]) == pair:
                    replaced.append(new_id)
                    position += 2
                else:
                    replaced.append(ids[position])
                    position += 1
            ids = replaced
        model = BPETokenizer.train(text, size)
        assert model._merges == tuple(expected)
        # Apply the definition by rescanning all current pairs at every step.
        query = "".join(rng.choices("aabbc \n中🙂", k=100))
        query_ids = list(query.encode("utf-8"))
        ranks = {pair: rank for rank, pair in enumerate(expected, start=256)}
        while True:
            candidates = [
                (ranks[pair], index)
                for index, pair in enumerate(zip(query_ids, query_ids[1:]))
                if pair in ranks
            ]
            if not candidates:
                break
            rank, index = min(candidates)
            query_ids[index:index + 2] = [rank]
        assert model.encode(query) == query_ids
        assert model.decode(query_ids) == query


def test_public_api_oracles(tokenizer, corpus_split):
    training, heldout = corpus_split
    oracle = Tokenizer(models.BPE())
    # Compare to the production-style public byte mapping and regex boundaries.
    oracle.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    oracle.decoder = decoders.ByteLevel()
    oracle.train_from_iterator(
        [training],
        trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=1,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    assert oracle.get_vocab_size() == tokenizer.vocab_size == VOCAB_SIZE
    rows = []
    for index in range(10):
        start = index * (len(heldout) - 2048) // 9
        sample = heldout[start:start + 2048]
        own_ids, hf_ids = tokenizer.encode(sample), oracle.encode(sample).ids
        assert tokenizer.decode(own_ids) == sample
        assert oracle.decode(hf_ids) == sample
        ratio = len(own_ids) / len(hf_ids)
        assert 0.5 <= ratio <= 2.0
        rows.append({"start": start, "characters": len(sample), "own_tokens": len(own_ids),
                     "hf_tokens": len(hf_ids), "ratio": ratio})
    for sample in ("中文🙂\t\n", "", "  mixed\r\n spaces\x00"):
        assert oracle.decode(oracle.encode(sample).ids) == sample
    reference = tiktoken.get_encoding("cl100k_base")
    own_ids, reference_ids = tokenizer.encode(heldout), reference.encode(heldout)
    assert tokenizer.decode(own_ids) == heldout
    assert reference.decode(reference_ids) == heldout
    byte_count = len(heldout.encode("utf-8"))
    metrics = {
        "formula": "token_count * log2(total_id_space) / utf8_bytes (uniform-ID proxy)",
        "evaluation": "last 10% of corpus, excluded from training; no language-model loss",
        "training_bytes": len(training.encode("utf-8")),
        "training_sha256": hashlib.sha256(training.encode("utf-8")).hexdigest(),
        "evaluation_bytes": byte_count,
        "evaluation_sha256": hashlib.sha256(heldout.encode("utf-8")).hexdigest(),
        "bpe_vocab_size": tokenizer.vocab_size,
        "total_id_space": tokenizer.vocab_size + 4,
        "own_tokens": len(own_ids),
        "own_bits_per_byte": len(own_ids) * math.log2(tokenizer.vocab_size + 4) / byte_count,
        "tiktoken_encoding": reference.name,
        "tiktoken_id_space": reference.n_vocab,
        "tiktoken_tokens": len(reference_ids),
        "tiktoken_bits_per_byte": len(reference_ids) * math.log2(reference.n_vocab) / byte_count,
        "hf_samples": rows,
        "hf_bytelevel_use_regex": True,
        "versions": {name: version(name) for name in ("pytest", "tokenizers", "tiktoken")},
    }
    output = ROOT / "results/l1/oracle-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
