"""L4 decode benchmark using the L0 warmup, repetition, and percentile protocol."""

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from layers import GPT, KVCache


WARMUP = 3
REPS = 10
PREFILL_TOKENS = 128
GENERATED_TOKENS = 256
CURVE_LENGTHS = (64, 128, 256, 512)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def device_info() -> dict:
    def sysctl(key: str) -> str:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True
        ).stdout.strip()

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(sysctl("hw.memsize")),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
    }


def config_hash() -> str:
    payload = {
        "platform": platform.platform(),
        "python": sys.version,
        "mlx": mx.__version__,
        "git_commit": git_commit(),
        "warmup": WARMUP,
        "reps": REPS,
        "prefill_tokens": PREFILL_TOKENS,
        "generated_tokens": GENERATED_TOKENS,
        "curve_lengths": CURVE_LENGTHS,
    }
    encoded = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_s": ordered[len(ordered) // 2],
        "p90_s": ordered[int(len(ordered) * 0.9) - 1],
        "min_s": ordered[0],
        "reps": len(ordered),
        "warmup": WARMUP,
    }


def timed_samples(fn) -> list[float]:
    for _ in range(WARMUP):
        mx.eval(fn())
    samples = []
    for _ in range(REPS):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - start)
    return samples


def load_model() -> GPT:
    model = GPT(vocab_size=1028, n_layer=4, d_model=512, n_head=8, ctx=256)
    checkpoint_path = Path("results/l3/ckpt-final-500.npz")
    with np.load(checkpoint_path) as checkpoint:
        for name in checkpoint.files:
            parts = name.split(".")
            current = model
            for part in parts[:-1]:
                current = current[int(part)] if part.isdigit() else getattr(current, part)
            setattr(current, parts[-1], mx.array(checkpoint[name]))
    mx.eval(model.parameters())
    return model


def fill_cache(model: GPT, token_ids: mx.array) -> tuple[KVCache, mx.array]:
    cache = KVCache(model.n_layer)
    logits = None
    for start in range(0, token_ids.shape[-1], model.ctx):
        logits = model(token_ids[start : start + model.ctx], cache=cache)
        mx.eval(logits)
    return cache, logits


def decode_loop(model: GPT, cache: KVCache, first_token: mx.array, count: int):
    current = first_token
    logits = None
    for _ in range(count):
        logits = model(current, cache=cache)
        mx.eval(logits)
        current = mx.argmax(logits[-1]).reshape((1,)).astype(mx.int32)
        mx.eval(current)
    return logits


def benchmark_curve(model: GPT) -> list[dict]:
    rows = []
    for length in CURVE_LENGTHS:
        context = mx.array(np.arange(length, dtype=np.int32) % model.vocab_size)
        samples = []
        for repetition in range(WARMUP + REPS):
            cache, logits = fill_cache(model, context)
            first_token = mx.argmax(logits[-1]).reshape((1,)).astype(mx.int32)
            mx.eval(first_token)
            start = time.perf_counter()
            result = model(first_token, cache=cache)
            mx.eval(result)
            elapsed = time.perf_counter() - start
            if repetition >= WARMUP:
                samples.append(elapsed)
        stats = summarize(samples)
        rows.append(
            {
                "kv_tokens_before_decode": length,
                "decode_tokens": 1,
                "decode_tokens_per_second_median": 1.0 / stats["median_s"],
                **stats,
            }
        )
    return rows


def benchmark_cache_ratio(model: GPT) -> dict:
    prompt = mx.array(np.arange(32, dtype=np.int32) % model.vocab_size)
    max_new = 32

    def cached_run():
        cache, logits = fill_cache(model, prompt)
        first = mx.argmax(logits[-1]).reshape((1,)).astype(mx.int32)
        return decode_loop(model, cache, first, max_new)

    def uncached_run():
        prefix = prompt
        logits = None
        for _ in range(max_new):
            logits = model(prefix)
            mx.eval(logits)
            token = mx.argmax(logits[-1]).reshape((1,)).astype(mx.int32)
            mx.eval(token)
            prefix = mx.concatenate((prefix, token), axis=0)
        return logits

    cached = summarize(timed_samples(cached_run))
    uncached = summarize(timed_samples(uncached_run))
    return {
        "prompt_tokens": int(prompt.shape[-1]),
        "max_new": max_new,
        "cached": cached,
        "uncached": uncached,
        "uncached_over_cached_median_ratio": uncached["median_s"] / cached["median_s"],
    }


def main() -> None:
    model = load_model()
    prompt = mx.array(np.arange(PREFILL_TOKENS, dtype=np.int32) % model.vocab_size)

    prefill_stats = summarize(timed_samples(lambda: model(prompt, cache=KVCache(model.n_layer))))

    decode_samples = []
    for _ in range(WARMUP + REPS):
        cache, logits = fill_cache(model, prompt)
        first = mx.argmax(logits[-1]).reshape((1,)).astype(mx.int32)
        mx.eval(first)
        start = time.perf_counter()
        result = decode_loop(model, cache, first, GENERATED_TOKENS)
        mx.eval(result)
        elapsed = time.perf_counter() - start
        if len(decode_samples) < REPS and _ >= WARMUP:
            decode_samples.append(elapsed)
    decode_stats = summarize(decode_samples)
    prefill_decode = {
        "prompt_tokens": PREFILL_TOKENS,
        "generated_tokens": GENERATED_TOKENS,
        "prefill": {
            **prefill_stats,
            "tokens_per_second_median": PREFILL_TOKENS / prefill_stats["median_s"],
        },
        "decode": {
            **decode_stats,
            "tokens_per_second_median": GENERATED_TOKENS / decode_stats["median_s"],
        },
    }

    roofline_tps = 450.0 / 0.0525
    measured_tps = prefill_decode["decode"]["tokens_per_second_median"]
    record = {
        "iteration": "L4-T1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(),
        "git_commit": git_commit(),
        "device": device_info(),
        "protocol": {
            "warmup": WARMUP,
            "reps": REPS,
            "percentiles": ["median", "p90"],
            "dtype": "float32",
            "checkpoint": "results/l3/ckpt-final-500.npz",
        },
        "prefill_vs_decode": prefill_decode,
        "roofline": {
            "assumed_bandwidth_gb_per_second": 450.0,
            "weight_size_mb": 52.5,
            "theoretical_tokens_per_second": roofline_tps,
            "measured_decode_tokens_per_second": measured_tps,
            "measured_fraction_of_roofline": measured_tps / roofline_tps,
        },
        "decode_rate_by_kv_length": benchmark_curve(model),
        "cached_vs_uncached": benchmark_cache_ratio(model),
    }

    out_dir = Path("results/l4")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"decode-bench-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"device: {record['device']['chip']}, mlx {record['device']['mlx']}")
    print(f"config_hash: {record['config_hash']} (commit {record['git_commit'][:12]})")
    print(
        f"prefill {PREFILL_TOKENS} tokens: "
        f"{prefill_decode['prefill']['tokens_per_second_median']:.2f} tok/s "
        f"(median {prefill_stats['median_s']:.6f}s, p90 {prefill_stats['p90_s']:.6f}s)"
    )
    print(
        f"decode {GENERATED_TOKENS} tokens: "
        f"{measured_tps:.2f} tok/s "
        f"(median {decode_stats['median_s']:.6f}s, p90 {decode_stats['p90_s']:.6f}s)"
    )
    print(
        f"roofline: {roofline_tps:.1f} tok/s; measured {measured_tps:.2f} tok/s "
        f"({100 * measured_tps / roofline_tps:.3f}% of bound)"
    )
    for row in record["decode_rate_by_kv_length"]:
        print(
            f"KV n={row['kv_tokens_before_decode']:>3}: "
            f"{row['decode_tokens_per_second_median']:.2f} tok/s "
            f"(median {row['median_s']:.6f}s, p90 {row['p90_s']:.6f}s)"
        )
    ratio = record["cached_vs_uncached"]
    print(
        f"cache ratio ({ratio['max_new']} new tokens): "
        f"cached {ratio['cached']['median_s']:.6f}s, "
        f"uncached {ratio['uncached']['median_s']:.6f}s, "
        f"uncached/cached {ratio['uncached_over_cached_median_ratio']:.3f}x"
    )
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
