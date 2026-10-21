"""L2 attention benchmarks using the L0 warmup/repetition/reporting protocol."""

import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from layers import CausalSelfAttention
from layers.attention import _stable_softmax, attention_scores


WARMUP = 3
REPS = 10
LENGTHS = (128, 256, 512, 1024, 2048, 4096)
KV_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)
D_MODEL = 512
N_HEAD = 8
D_HEAD = D_MODEL // N_HEAD


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def build_config() -> dict:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "mlx": mx.__version__,
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "git_commit": git_commit(),
        "warmup": WARMUP,
        "reps": REPS,
        "lengths": list(LENGTHS),
        "kv_lengths": list(KV_LENGTHS),
        "d_model": D_MODEL,
        "n_head": N_HEAD,
        "d_head": D_HEAD,
        "dtype": "fp32 for attention; fp32 and bf16 for KV storage",
    }


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def device_info() -> dict:
    def sysctl(key: str) -> str:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True
        ).stdout.strip()

    memory = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_bytes": int(memory) if memory.isdigit() else None,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
    }


def timed(fn) -> list[float]:
    for _ in range(WARMUP):
        result = fn()
        mx.eval(result)
        del result
    samples = []
    for _ in range(REPS):
        start = time.perf_counter()
        result = fn()
        mx.eval(result)
        samples.append(time.perf_counter() - start)
        del result
    return sorted(samples)


def summarize(samples: list[float]) -> dict:
    n = len(samples)
    return {
        "median_s": samples[n // 2],
        "p90_s": samples[int(n * 0.9) - 1],
        "min_s": samples[0],
        "reps": n,
        "warmup": WARMUP,
    }


def attention_kernel_bench() -> list[dict]:
    rows = []
    for length in LENGTHS:
        mx.random.seed(42)
        q = mx.random.normal((1, N_HEAD, length, D_HEAD))
        k = mx.random.normal((1, N_HEAD, length, D_HEAD))
        v = mx.random.normal((1, N_HEAD, length, D_HEAD))
        mx.eval(q, k, v)

        def naive():
            scores = attention_scores(q, k)
            return _stable_softmax(scores) @ v

        def fast():
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=D_HEAD**-0.5, mask="causal"
            )

        naive_stats = summarize(timed(naive))
        fast_stats = summarize(timed(fast))
        rows.append(
            {
                "sequence_length": length,
                "shape_qkv": [1, N_HEAD, length, D_HEAD],
                "naive": naive_stats,
                "fast": fast_stats,
                "naive_over_fast_median": naive_stats["median_s"] / fast_stats["median_s"],
            }
        )
        del q, k, v
        gc.collect()
    return rows


def kv_memory_curve() -> list[dict]:
    rows = []
    for dtype_name, dtype, dtype_size in (
        ("fp32", mx.float32, 4),
        ("bf16", mx.bfloat16, 2),
    ):
        for length in KV_LENGTHS:
            gc.collect()
            before_bytes = mx.get_active_memory()
            keys = mx.zeros((length, D_MODEL), dtype=dtype)
            values = mx.zeros((length, D_MODEL), dtype=dtype)
            mx.eval(keys, values)
            after_bytes = mx.get_active_memory()
            measured_bytes = max(0, after_bytes - before_bytes)
            theoretical_bytes = 2 * length * D_MODEL * dtype_size
            rows.append(
                {
                    "dtype": dtype_name,
                    "sequence_length": length,
                    "d_model": D_MODEL,
                    "measured_active_bytes": measured_bytes,
                    "theoretical_bytes": theoretical_bytes,
                    "measured_over_theoretical": (
                        measured_bytes / theoretical_bytes if theoretical_bytes else None
                    ),
                }
            )
            del keys, values
            gc.collect()
    return rows


def softmax_fraction_bench() -> dict:
    length = 1024
    mx.random.seed(42)
    model = CausalSelfAttention(d_model=D_MODEL, n_head=N_HEAD, d_head=D_HEAD, seed=42)
    x = mx.random.normal((1, length, D_MODEL))
    full_out = model.forward_with_intermediates(x)[0]
    mx.eval(full_out)

    _, _, intermediates = model.forward_with_intermediates(x)
    q, k, v = intermediates["rotary_q"], intermediates["rotary_k"], intermediates["v"]
    scores = attention_scores(q, k)
    mx.eval(scores, v)
    probs = _stable_softmax(scores)
    mx.eval(probs)

    full_stats = summarize(timed(lambda: model.forward_with_intermediates(x)[0]))
    score_stats = summarize(timed(lambda: attention_scores(q, k)))
    softmax_stats = summarize(timed(lambda: _stable_softmax(scores)))
    value_stats = summarize(timed(lambda: probs @ v))
    return {
        "sequence_length": length,
        "full_layer": full_stats,
        "score_and_mask": score_stats,
        "softmax_only": softmax_stats,
        "probability_times_value": value_stats,
        "softmax_over_full_layer_percent": (
            100.0 * softmax_stats["median_s"] / full_stats["median_s"]
        ),
        "measurement_note": "coarse isolated component timing divided by full forward median",
    }


def main() -> None:
    config = build_config()
    record = {
        "iteration": "L2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "config": config,
        "device": device_info(),
        "attention_kernels": attention_kernel_bench(),
        "kv_memory": kv_memory_curve(),
        "softmax_fraction": softmax_fraction_bench(),
    }
    out_dir = Path("results/l2")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"attention-bench-{stamp}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"device: {record['device']['chip']}, mlx {record['device']['mlx']}")
    print(f"config_hash: {record['config_hash']}")
    print("attention kernel medians (s):")
    for row in record["attention_kernels"]:
        print(
            f"  n={row['sequence_length']:>4}: naive={row['naive']['median_s']:.6g} "
            f"fast={row['fast']['median_s']:.6g} "
            f"naive/fast={row['naive_over_fast_median']:.2f}x"
        )
    print("KV allocation (measured/theoretical bytes):")
    for row in record["kv_memory"]:
        print(
            f"  {row['dtype']:>4} n={row['sequence_length']:>4}: "
            f"{row['measured_active_bytes']}/{row['theoretical_bytes']} "
            f"({row['measured_over_theoretical']:.3f}x)"
        )
    share = record["softmax_fraction"]
    print(
        "softmax share: "
        f"{share['softmax_only']['median_s']:.6g}s / "
        f"{share['full_layer']['median_s']:.6g}s = "
        f"{share['softmax_over_full_layer_percent']:.2f}% "
        f"(n={share['sequence_length']})"
    )
    print(f"written: {out}")


if __name__ == "__main__":
    main()
