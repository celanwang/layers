"""L1 CPU tokenizer throughput with the L0 warmup/repetition protocol.

Run from the repository root with .venv/bin/python bench/l1_tokenizer_throughput.py.
All timing is synchronous Python work; Metal evaluation is not applicable.
"""

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tiktoken  # noqa: E402

from layers import BPETokenizer  # noqa: E402
from bench.l0_hw_profile import REPS, WARMUP, device_info, git_commit, summarize  # noqa: E402

VOCAB_SIZE = 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timed(fn) -> list[float]:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPS):
        started = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - started)
        del result  # Release output outside the timing interval.
    return sorted(samples)


def throughput(samples: list[float], byte_count: int) -> dict:
    stats = summarize(samples)  # Exactly the existing L0 quantile convention.
    megabytes = byte_count / 1e6
    return {
        **stats,
        "input_bytes": byte_count,
        "samples_s": samples,
        "throughput_mbs_median": megabytes / stats["median_s"],
        "throughput_mbs_at_p90_latency": megabytes / stats["p90_s"],
        "throughput_mbs_best": megabytes / stats["min_s"],
    }


def main() -> None:
    corpus_path = ROOT / "data/tiny-shakespeare.txt"
    raw = corpus_path.read_bytes()
    text = raw.decode("utf-8")
    boundary = len(text) * 9 // 10
    training, heldout = text[:boundary], text[boundary:]
    training_bytes, heldout_bytes = training.encode("utf-8"), heldout.encode("utf-8")
    device = device_info()
    config = {
        "git_commit": git_commit(),
        "git_status_porcelain": subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines(),
        "device": device,
        "python_full_version": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("pytest", "tiktoken", "tokenizers", "mlx")},
        "source_sha256": {
            path: sha256((ROOT / path).read_bytes())
            for path in ("layers/__init__.py", "layers/tokenizer.py",
                         "bench/l1_tokenizer_throughput.py", "bench/l0_hw_profile.py")
        },
        "corpus": {
            "path": str(corpus_path.relative_to(ROOT)),
            "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "sha256": sha256(raw),
            "bytes": len(raw),
            "split": "first 90% of Unicode characters train; last 10% held out",
            "training_bytes": len(training_bytes),
            "training_sha256": sha256(training_bytes),
            "heldout_bytes": len(heldout_bytes),
            "heldout_sha256": sha256(heldout_bytes),
        },
        "bpe_vocab_size": VOCAB_SIZE,
        "special_tokens": 4,
        "pretokenization": False,
        "warmup": WARMUP,
        "reps": REPS,
        "clock": "time.perf_counter",
        "execution": "synchronous CPython on CPU; single process",
        "units": "decimal MB = 1e6 UTF-8 bytes",
        "quantiles": "L0: sorted seconds[n//2], seconds[int(n*0.9)-1]",
        "timed_scope": "UTF-8 conversion and operation; excludes file I/O and output destruction",
        "encode_input": "entire corpus, including train and heldout",
    }
    config_hash = sha256(json.dumps(config, sort_keys=True).encode("utf-8"))[:12]
    print(f"device: {device['chip']}, {device['memory_bytes'] / 2**30:.0f} GiB, CPU", flush=True)
    print(f"config_hash: {config_hash}", flush=True)
    print(f"protocol: warmup={WARMUP}, reps={REPS}, BPE vocab={VOCAB_SIZE}, total IDs=1028", flush=True)
    print(f"corpus: {len(raw)} bytes; training={len(training_bytes)}; heldout={len(heldout_bytes)}", flush=True)

    train_stats = throughput(timed(lambda: BPETokenizer.train(training, VOCAB_SIZE)), len(training_bytes))
    model = BPETokenizer.train(training, VOCAB_SIZE)
    if model.vocab_size != VOCAB_SIZE:
        raise RuntimeError("training failed to reach the configured BPE size")
    print(
        f"train: {train_stats['throughput_mbs_median']:.6f} MB/s (median); "
        f"median={train_stats['median_s']:.6f}s; p90={train_stats['p90_s']:.6f}s", flush=True,
    )
    encode_stats = throughput(timed(lambda: model.encode(text)), len(raw))
    if model.decode(model.encode(text)) != text:
        raise RuntimeError("corpus roundtrip failed")
    print(
        f"encode: {encode_stats['throughput_mbs_median']:.6f} MB/s (median); "
        f"median={encode_stats['median_s']:.6f}s; p90={encode_stats['p90_s']:.6f}s", flush=True,
    )

    reference = tiktoken.get_encoding("cl100k_base")
    own_ids, reference_ids = model.encode(heldout), reference.encode(heldout)
    if model.decode(own_ids) != heldout or reference.decode(reference_ids) != heldout:
        raise RuntimeError("held-out roundtrip failed")
    comparison = {
        "formula": "token_count * log2(total_id_space) / utf8_bytes (uniform-ID proxy)",
        "evaluation_bytes": len(heldout_bytes),
        "own_tokens": len(own_ids),
        "own_id_space": model.vocab_size + 4,
        "own_bits_per_byte": len(own_ids) * math.log2(model.vocab_size + 4) / len(heldout_bytes),
        "tiktoken_encoding": reference.name,
        "tiktoken_tokens": len(reference_ids),
        "tiktoken_id_space": reference.n_vocab,
        "tiktoken_bits_per_byte": len(reference_ids) * math.log2(reference.n_vocab) / len(heldout_bytes),
    }
    output_dir = ROOT / "results/l1"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tokenizer-1024.json"
    model.save(str(model_path))
    if BPETokenizer.load(str(model_path)).encode(heldout) != own_ids:
        raise RuntimeError("saved model roundtrip failed")
    record = {
        "iteration": "L1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "config": config,
        "device": device,
        "training": train_stats,
        "encode": encode_stats,
        "comparison": comparison,
        "model": {"path": str(model_path.relative_to(ROOT)), "sha256": sha256(model_path.read_bytes())},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / f"tokenizer-throughput-{stamp}.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"heldout bits/byte (uniform-ID proxy): own={comparison['own_bits_per_byte']:.6f}, "
        f"cl100k_base={comparison['tiktoken_bits_per_byte']:.6f}", flush=True,
    )
    print(f"written: {output.relative_to(ROOT)}", flush=True)
    print(f"model: {model_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
