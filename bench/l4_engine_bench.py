"""L4-T2 page-pool and continuous-scheduler measurements using the L0 protocol."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers import BPETokenizer, Engine, GPT, Sampler
from layers.paged_kv import PagedKVPool


WARMUP = 3
REPS = 10
FRAGMENTATION_COUNTS = (1, 4, 8, 16)
REQUESTS = 8
PROMPT_TEXT = "ROMEO: "
GENERATED_TOKENS = 16
BLOCK_SIZE = 16
RANDOM_SEED = 20261021


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
        "fragmentation_counts": FRAGMENTATION_COUNTS,
        "requests": REQUESTS,
        "prompt": PROMPT_TEXT,
        "generated_tokens": GENERATED_TOKENS,
        "block_size": BLOCK_SIZE,
        "random_seed": RANDOM_SEED,
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


def load_model() -> GPT:
    model = GPT(vocab_size=1028, n_layer=4, d_model=512, n_head=8, ctx=256)
    with np.load("results/l3/ckpt-final-500.npz") as checkpoint:
        for name in checkpoint.files:
            parts = name.split(".")
            current = model
            for part in parts[:-1]:
                current = current[int(part)] if part.isdigit() else getattr(current, part)
            setattr(current, parts[-1], mx.array(checkpoint[name]))
    mx.eval(model.parameters())
    return model


def fragmentation_measurements(model: GPT) -> list[dict]:
    lengths = np.random.default_rng(RANDOM_SEED).integers(32, 257, size=16).tolist()
    rows = []
    for count in FRAGMENTATION_COUNTS:
        pool = PagedKVPool(
            num_layers=model.n_layer,
            block_size=BLOCK_SIZE,
            max_context=512,
            max_sequences=16,
        )
        selected = lengths[:count]
        for length in selected:
            seq_id = pool.alloc()
            for layer_idx in range(model.n_layer):
                keys = mx.zeros((1, model.n_head, length, model.d_model // model.n_head))
                values = mx.ones((1, model.n_head, length, model.d_model // model.n_head))
                pool.append(seq_id, keys, values, layer_idx=layer_idx)
        mx.eval(*[pool.gather(seq_id, layer)[0] for seq_id in range(count) for layer in range(model.n_layer)])
        upper_bound = count * (BLOCK_SIZE // 2) * pool.bytes_per_token
        if pool.wasted_bytes > upper_bound:
            raise AssertionError(
                f"fragmentation exceeded bound for N={count}: "
                f"{pool.wasted_bytes} > {upper_bound} bytes"
            )
        rows.append(
            {
                "sequences": count,
                "lengths": selected,
                "block_size_tokens": BLOCK_SIZE,
                "bytes_per_token": pool.bytes_per_token,
                "bytes_per_block": pool.block_bytes,
                "physical_blocks_used": pool.allocated_blocks,
                "wasted_tokens": pool.wasted_tokens,
                "wasted_bytes": pool.wasted_bytes,
                "theoretical_upper_bound_bytes": upper_bound,
                "bound_passed": True,
            }
        )
    return rows


def _run_continuous(model: GPT, prompt_ids: list[int]) -> int:
    engine = Engine(
        model,
        max_context=model.ctx,
        max_concurrent=REQUESTS,
        max_new_tokens=GENERATED_TOKENS,
        block_size=BLOCK_SIZE,
        share_prefixes=True,
    )
    for index in range(REQUESTS):
        engine.add_request(
            prompt_ids,
            Sampler(temperature=0, seed=RANDOM_SEED + index),
        )
    emitted = 0
    while engine.active_requests:
        emitted += len(engine.step())
    return emitted


def _run_sequential(model: GPT, prompt_ids: list[int]) -> int:
    engine = Engine(
        model,
        max_context=model.ctx,
        max_concurrent=1,
        max_new_tokens=GENERATED_TOKENS,
        block_size=BLOCK_SIZE,
        share_prefixes=False,
    )
    emitted = 0
    for index in range(REQUESTS):
        engine.add_request(
            prompt_ids,
            Sampler(temperature=0, seed=RANDOM_SEED + index),
        )
        emitted += len(engine.run())
    return emitted


def timed_samples(fn) -> tuple[list[float], int]:
    count = 0
    for _ in range(WARMUP):
        count = fn()
    samples = []
    for _ in range(REPS):
        start = time.perf_counter()
        count = fn()
        samples.append(time.perf_counter() - start)
    return samples, count


def batch_measurement(model: GPT, prompt_ids: list[int]) -> dict:
    continuous_samples, continuous_tokens = timed_samples(
        lambda: _run_continuous(model, prompt_ids)
    )
    sequential_samples, sequential_tokens = timed_samples(
        lambda: _run_sequential(model, prompt_ids)
    )
    continuous = summarize(continuous_samples)
    sequential = summarize(sequential_samples)
    continuous_tps = continuous_tokens / continuous["median_s"]
    sequential_tps = sequential_tokens / sequential["median_s"]
    return {
        "requests": REQUESTS,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens_per_request": GENERATED_TOKENS,
        "total_generated_tokens": continuous_tokens,
        "continuous_batching": {
            **continuous,
            "tokens_per_second": continuous_tps,
        },
        "sequential": {
            **sequential,
            "tokens_per_second": sequential_tps,
        },
        "throughput_speedup": continuous_tps / sequential_tps,
        "throughput_improvement_percent": 100 * (continuous_tps / sequential_tps - 1),
    }


def prefix_memory_measurement(model: GPT, prompt_length: int = 32) -> dict:
    shape = (1, model.n_head, prompt_length, model.d_model // model.n_head)
    keys = mx.zeros(shape)
    values = mx.ones(shape)

    shared = PagedKVPool(
        num_layers=model.n_layer,
        block_size=BLOCK_SIZE,
        max_context=512,
        max_sequences=REQUESTS,
    )
    shared_parent = shared.alloc()
    for layer_idx in range(model.n_layer):
        shared.append(shared_parent, keys, values, layer_idx=layer_idx)
    shared_ids = [shared_parent]
    shared_ids.extend(shared.fork(shared_parent, prefix_len=prompt_length) for _ in range(REQUESTS - 1))

    unshared = PagedKVPool(
        num_layers=model.n_layer,
        block_size=BLOCK_SIZE,
        max_context=512,
        max_sequences=REQUESTS,
    )
    for _ in range(REQUESTS):
        seq_id = unshared.alloc()
        for layer_idx in range(model.n_layer):
            unshared.append(seq_id, keys, values, layer_idx=layer_idx)

    shared_bytes = shared.allocated_bytes
    unshared_bytes = unshared.allocated_bytes
    return {
        "requests": REQUESTS,
        "shared_prompt_tokens": prompt_length,
        "shared": {
            "physical_blocks_used": shared.allocated_blocks,
            "allocated_bytes": shared_bytes,
        },
        "unshared": {
            "physical_blocks_used": unshared.allocated_blocks,
            "allocated_bytes": unshared_bytes,
        },
        "bytes_saved": unshared_bytes - shared_bytes,
        "savings_percent": 100 * (unshared_bytes - shared_bytes) / unshared_bytes,
    }


def main() -> None:
    model = load_model()
    tokenizer = BPETokenizer.load("results/l1/tokenizer-1024.json")
    prompt_ids = tokenizer.encode(PROMPT_TEXT)
    if len(prompt_ids) + GENERATED_TOKENS > model.ctx:
        raise ValueError("benchmark prompt plus generation exceeds model context")

    fragmentation = fragmentation_measurements(model)
    batching = batch_measurement(model, prompt_ids)
    prefix_memory = prefix_memory_measurement(model)
    record = {
        "iteration": "L4-T2",
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
        "fragmentation": fragmentation,
        "continuous_batching_vs_sequential": batching,
        "prefix_sharing_memory": prefix_memory,
    }

    out_dir = Path("results/l4")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"engine-bench-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"device: {record['device']['chip']}, mlx {record['device']['mlx']}")
    print(f"config_hash: {record['config_hash']} (commit {record['git_commit'][:12]})")
    print(f"fragmentation (block_size={BLOCK_SIZE}):")
    for row in fragmentation:
        print(
            f"  N={row['sequences']:>2}: waste {row['wasted_bytes']} B / "
            f"bound {row['theoretical_upper_bound_bytes']} B "
            f"({row['wasted_tokens']} token slots)"
        )
    print(
        f"batching {REQUESTS} requests / {batching['total_generated_tokens']} generated tokens: "
        f"continuous {batching['continuous_batching']['median_s']:.6f}s "
        f"({batching['continuous_batching']['tokens_per_second']:.2f} tok/s), "
        f"sequential {batching['sequential']['median_s']:.6f}s "
        f"({batching['sequential']['tokens_per_second']:.2f} tok/s), "
        f"speedup {batching['throughput_speedup']:.3f}x"
    )
    print(
        f"prefix sharing ({REQUESTS} x {prefix_memory['shared_prompt_tokens']} tokens): "
        f"{prefix_memory['shared']['allocated_bytes']} B vs "
        f"{prefix_memory['unshared']['allocated_bytes']} B, "
        f"saved {prefix_memory['savings_percent']:.1f}%"
    )
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
