"""Full-config training-step throughput benchmark using the L0 timing protocol."""

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from scripts.train import (
    BATCH_SIZE,
    CONTEXT,
    PEAK_LR,
    STEPS,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    AdamW,
    TrainingEngine,
    WindowBatchSampler,
    build_config,
    config_hash,
    device_info,
    estimate_parameter_count,
    make_corpus,
    new_model,
)
from layers.optimizer import cosine_lr


WARMUP = 3
REPS = 10


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return float(ordered[index])


def main() -> None:
    _tokenizer, train_ids, _heldout_ids, corpus = make_corpus()
    config = build_config(corpus["data_sha256"], corpus["tokenizer_sha256"])
    run_hash = config_hash(config)
    model = new_model(seed=42)
    mx.eval(model.parameters())
    parameter_count = estimate_parameter_count(model)
    engine = TrainingEngine(model, AdamW(learning_rate=PEAK_LR, weight_decay=WEIGHT_DECAY))
    sampler = WindowBatchSampler(train_ids, BATCH_SIZE, CONTEXT, np.random.default_rng(42))
    mx.metal.reset_peak_memory()
    samples_seconds = []
    for index in range(WARMUP + REPS):
        inputs, targets = sampler.next_batch()
        step_number = min(index + 1, STEPS)
        lr = cosine_lr(step_number, STEPS, PEAK_LR, WARMUP_STEPS)
        started = time.perf_counter()
        engine.step(inputs, targets, lr)
        elapsed = time.perf_counter() - started
        if index >= WARMUP:
            samples_seconds.append(elapsed)

    tokens_per_step = BATCH_SIZE * CONTEXT
    throughput = [tokens_per_step / elapsed for elapsed in samples_seconds]
    latencies = {
        "median_s": percentile(samples_seconds, 0.5),
        "p90_s": percentile(samples_seconds, 0.9),
        "reps": REPS,
        "warmup": WARMUP,
    }
    throughput_stats = {
        "median_tokens_per_second": percentile(throughput, 0.5),
        "p90_tokens_per_second": percentile(throughput, 0.9),
        "tokens_per_step": tokens_per_step,
    }
    peak_memory = int(mx.metal.get_peak_memory())
    record = {
        "iteration": "L3-T2",
        "measurement": "full-config training step throughput",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": run_hash,
        "config": config,
        "device": device_info(),
        "parameter_count": parameter_count,
        "warmup_repetitions": WARMUP,
        "measured_repetitions": REPS,
        "latency": latencies,
        "throughput": throughput_stats,
        "peak_metal_memory_bytes": peak_memory,
        "peak_metal_memory_mib": peak_memory / (1024**2),
    }
    out_dir = ROOT / "results/l3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train-bench.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"config_hash: {run_hash}")
    print(f"device: {record['device']['chip']} / mlx {record['device']['mlx']}")
    print(f"protocol: warmup={WARMUP}, measured={REPS}, tokens/step={tokens_per_step}")
    print(
        f"throughput: median={throughput_stats['median_tokens_per_second']:.1f}, "
        f"p90={throughput_stats['p90_tokens_per_second']:.1f} tokens/s"
    )
    print(f"latency: median={latencies['median_s']:.4f}, p90={latencies['p90_s']:.4f} s")
    print(f"peak Metal memory: {peak_memory / (1024**2):.1f} MiB")
    print(f"written: {out_path.relative_to(ROOT)}")
    del engine, model
    gc.collect()


if __name__ == "__main__":
    main()
