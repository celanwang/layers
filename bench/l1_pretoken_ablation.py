"""Controlled L1 token-count ablation; this is not a throughput benchmark.

Run: .venv/bin/python bench/l1_pretoken_ablation.py
The full training prefix is ONE iterator item, even with use_regex=False.
No implementation, tests, existing models, or measurement protocol are changed.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tiktoken  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from layers import BPETokenizer  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def longest_match(model: BPETokenizer, text: str) -> list[int]:
    """A deliberately different encoder, only for the counterexample."""
    data = text.encode("utf-8")
    pieces = sorted(enumerate(model._vocab), key=lambda item: (-len(item[1]), item[0]))
    ids = []
    offset = 0
    while offset < len(data):
        token_id, piece = next(item for item in pieces if data.startswith(item[1], offset))
        ids.append(token_id)
        offset += len(piece)
    return ids


def counterexamples() -> list[dict]:
    results = []
    for text, merges in (
        ("abc", ((98, 99), (97, 98))),
        ("abcd", ((98, 99), (97, 98), (99, 100))),
    ):
        own = BPETokenizer(merges)
        ranked = own.encode(text)
        greedy = longest_match(own, text)
        reference = tiktoken.Encoding(
            name="l1-rank-counterexample",
            pat_str="(?s).+",
            mergeable_ranks={piece: i for i, piece in enumerate(own._vocab)},
            special_tokens={},
        )
        oracle = reference.encode(text)
        assert ranked == oracle and ranked != greedy
        assert own.decode(ranked) == own.decode(greedy) == text
        results.append({
            "text": text,
            "merges": [list(pair) for pair in merges],
            "rank_ids": ranked,
            "rank_pieces": [own._vocab[i].decode("ascii") for i in ranked],
            "longest_ids": greedy,
            "longest_pieces": [own._vocab[i].decode("ascii") for i in greedy],
            "tiktoken_ids": oracle,
            "roundtrip": True,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/l1/pretoken-ablation.json")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.parent != (ROOT / "results/l1").resolve():
        parser.error("output must be directly within results/l1/")
    if output.exists():
        parser.error("refusing to overwrite an existing result; choose a fresh --output")

    raw = (ROOT / "data/tiny-shakespeare.txt").read_bytes()
    text = raw.decode("utf-8")
    boundary = len(text) * 9 // 10
    training, heldout = text[:boundary], text[boundary:]
    baseline_path = ROOT / "results/l1/oracle-comparison.json"
    baseline = json.loads(baseline_path.read_text())
    assert sha256(training.encode()) == baseline["training_sha256"]
    assert sha256(heldout.encode()) == baseline["evaluation_sha256"]
    model_path = ROOT / "results/l1/tokenizer-1024.json"
    own = BPETokenizer.load(str(model_path))
    reference = tiktoken.get_encoding("cl100k_base")
    own_ids, reference_ids = own.encode(heldout), reference.encode(heldout)
    assert len(own_ids) == baseline["own_tokens"] == 46683
    assert len(reference_ids) == baseline["tiktoken_tokens"] == 31469
    assert own.decode(own_ids) == reference.decode(reference_ids) == heldout

    cases = [(True, 1024), (True, 8192), (True, 100256), (False, 1024)]
    config = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("tokenizers", "tiktoken")},
        "source_sha256": {
            p: sha256((ROOT / p).read_bytes())
            for p in ("bench/l1_pretoken_ablation.py", "layers/tokenizer.py")
        },
        "corpus": {
            "path": "data/tiny-shakespeare.txt", "sha256": sha256(raw),
            "bytes": len(raw), "training_bytes": len(training.encode()),
            "training_sha256": sha256(training.encode()),
            "heldout_bytes": len(heldout.encode()),
            "heldout_sha256": sha256(heldout.encode()),
            "split": "first 90% of Unicode characters train; last 10% held out",
            "iterator_items": 1,
        },
        "hf": {
            "cases": [{"use_regex": regex, "requested_vocab": size} for regex, size in cases],
            "model": "BPE()", "add_prefix_space": False,
            "decoder": "ByteLevel()", "initial_alphabet": "all 256 bytes",
            "min_frequency": 1, "special_tokens": [], "normalizer": None,
            "encode_add_special_tokens": False,
        },
        "baseline_sha256": sha256(baseline_path.read_bytes()),
        "own_model_sha256": sha256(model_path.read_bytes()),
        "measurement": "single deterministic token-count evaluation; no latency claims",
    }
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_hash": sha256(json.dumps(config, sort_keys=True).encode())[:12],
        "complete": False,
        "baseline": {"own_tokens": len(own_ids), "cl100k_tokens": len(reference_ids),
                     "cl100k_mergeable_vocab": len(reference._mergeable_ranks),
                     "cl100k_n_vocab": reference.n_vocab},
        "encode_counterexamples": counterexamples(),
        "hf_cases": [],
    }

    def write_result() -> None:
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    write_result()
    for regex, size in cases:
        print(f"training use_regex={regex}, vocab_size={size}, one full prefix", flush=True)
        hf = Tokenizer(models.BPE())
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=regex)
        hf.decoder = decoders.ByteLevel()
        hf.train_from_iterator([training], trainers.BpeTrainer(
            vocab_size=size, min_frequency=1, special_tokens=[],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
        ))
        ids = hf.encode(heldout, add_special_tokens=False).ids
        assert hf.decode(ids) == heldout
        actual = hf.get_vocab_size()
        if size == 1024:
            assert actual == size
        row = {
            "use_regex": regex, "requested_vocab": size, "actual_vocab": actual,
            "heldout_tokens": len(ids), "roundtrip": True,
            "model_sha256": sha256(hf.to_str().encode()),
        }
        result["hf_cases"].append(row)
        write_result()
        print(json.dumps(row), flush=True)

    counts = {(r["use_regex"], r["requested_vocab"]): r["heldout_tokens"]
              for r in result["hf_cases"]}
    delta = len(own_ids) - len(reference_ids)
    no_regex, regex = counts[False, 1024], counts[True, 1024]
    large = counts[True, 100256]
    result["decomposition"] = {
        "original_gap_tokens": delta,
        "original_gap_percent_of_cl100k": 100 * delta / len(reference_ids),
        "own_minus_hf_no_regex": len(own_ids) - no_regex,
        "no_regex_effect_tokens": no_regex - regex,
        "no_regex_share_of_original_gap_percent": 100 * (no_regex - regex) / delta,
        "residual_hf_regex_minus_cl100k": regex - len(reference_ids),
        "residual_share_of_original_gap_percent": 100 * (regex - len(reference_ids)) / delta,
        "controlled_vocab_effect_tokens": regex - large,
        "large_hf_minus_cl100k": large - len(reference_ids),
        "identity": "own-cl100k = (own-HF_false_1024) + (HF_false_1024-HF_true_1024) + (HF_true_1024-HF_true_large) + (HF_true_large-cl100k)",
        "caveat": "Residual is NOT a pure vocabulary-size causal effect: training data, regex, tie rules and learned ranks also differ from cl100k. Large HF vocabulary may saturate below the requested size.",
    }
    result["complete"] = True
    write_result()
    print(json.dumps(result["decomposition"], indent=2), flush=True)
    print(f"written: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
