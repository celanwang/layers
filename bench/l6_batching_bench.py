"""L6 shared decode forwards versus unchanged L4 Engine, under the L0 protocol."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import BatchedEngine, Engine, GPT, Sampler


WARMUP = 3
REPS = 10
GENERATED_TOKENS = 64
BATCH_SIZES = (1, 2, 4, 8)
BLOCK_SIZE = 16
SEED = 20261021
CHECKPOINT = ROOT / "results/l3/ckpt-final-500.npz"
SMALL_VARIANCE = (95, 95, 96, 96, 96, 96, 97, 97)
LARGE_VARIANCE = (8, 24, 40, 72, 120, 152, 168, 184)


def git_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def device_info():
    def sysctl(key):
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(sysctl("hw.memsize")),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
    }


def load_model():
    model = GPT(vocab_size=1028, n_layer=4, d_model=512, n_head=8, ctx=256)
    with np.load(CHECKPOINT) as checkpoint:
        for name in checkpoint.files:
            parts = name.split(".")
            current = model
            for part in parts[:-1]:
                current = current[int(part)] if part.isdigit() else getattr(current, part)
            setattr(current, parts[-1], mx.array(checkpoint[name]))
    mx.eval(model.parameters())
    return model


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_s": ordered[len(ordered) // 2],
        "p90_s": ordered[int(len(ordered) * 0.9) - 1],
        "min_s": ordered[0],
        "warmup": WARMUP,
        "reps": len(samples),
        "samples_s": samples,
    }


def run_once(model, engine_type, lengths):
    engine = engine_type(
        model, block_size=BLOCK_SIZE, max_concurrent=len(lengths),
        max_new_tokens=GENERATED_TOKENS, share_prefixes=False,
    )
    start = time.perf_counter()
    seq_ids = []
    for row, length in enumerate(lengths):
        # Distinct prompts with identical lengths: no prefix reuse confound.
        prompt = ((np.arange(length) + row * 29) % model.vocab_size).tolist()
        seq_ids.append(engine.add_request(prompt, Sampler(temperature=0, seed=SEED + row)))
    mx.synchronize()
    decode_start = time.perf_counter()
    events = engine.run()
    mx.synchronize()
    finish = time.perf_counter()
    assert len(events) == len(lengths) * GENERATED_TOKENS
    assert engine.pool.allocated_blocks == engine.pool.active_sequences == 0
    return {
        "decode_s": finish - decode_start,
        "prefill_s": decode_start - start,
        "end_to_end_s": finish - start,
        "outputs": [engine.generated_tokens(seq_id) for seq_id in seq_ids],
    }


def measure_pair(model, left_type, right_type, left_lengths, right_lengths):
    samples = [{key: [] for key in ("decode_s", "prefill_s", "end_to_end_s")} for _ in range(2)]
    for repetition in range(WARMUP + REPS):
        results = [None, None]
        # Alternate order to reduce drift bias; every trial has a fresh pool.
        for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            results[index] = run_once(
                model, (left_type, right_type)[index], (left_lengths, right_lengths)[index]
            )
        if left_lengths == right_lengths:
            assert results[0]["outputs"] == results[1]["outputs"], "L4 token oracle mismatch"
        if repetition >= WARMUP:
            for index in range(2):
                for key in samples[index]:
                    samples[index][key].append(results[index][key])

    summaries = []
    for index, lengths in enumerate((left_lengths, right_lengths)):
        summary = {key.removesuffix("_s"): summarize(values) for key, values in samples[index].items()}
        summary["tokens_per_second"] = len(lengths) * GENERATED_TOKENS / summary["decode"]["median_s"]
        summary["end_to_end_tokens_per_second"] = (
            len(lengths) * GENERATED_TOKENS / summary["end_to_end"]["median_s"]
        )
        summaries.append(summary)
    return summaries


def padding_accounting(lengths):
    # Engine samples the first token from prompt logits, then does 63 forwards.
    real = sum(sum(length + step for length in lengths) for step in range(1, GENERATED_TOKENS))
    padded = sum(len(lengths) * (max(lengths) + step) for step in range(1, GENERATED_TOKENS))
    return {
        "prompt_lengths": lengths,
        "mean_prompt_length": float(np.mean(lengths)),
        "variance_prompt_length": float(np.var(lengths)),
        "real_attention_key_slots_per_head_layer": real,
        "padded_attention_key_slots_per_head_layer": padded,
        "wasted_key_slots": padded - real,
        "padding_fraction": (padded - real) / padded,
    }


def main():
    device = device_info()
    protocol = {
        "warmup": WARMUP, "reps": REPS, "percentiles": ["median", "p90"],
        "dtype": "float32", "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT), "seed": SEED,
        "batch_sizes": BATCH_SIZES, "prompt_tokens_scaling": 32,
        "generated_tokens_per_request": GENERATED_TOKENS,
        "block_size": BLOCK_SIZE, "share_prefixes": False,
        "small_variance_lengths": SMALL_VARIANCE, "large_variance_lengths": LARGE_VARIANCE,
        "timing": "decode includes sampling, scheduling, KV gather/pad/scatter, and synchronization; prefill separate",
        "decode_forwards_per_request": GENERATED_TOKENS - 1,
        "trial_order": "paired, alternating; fresh engine and pool per trial",
    }
    sources = {
        path: sha256(ROOT / path) for path in (
            "layers/engine.py", "layers/attention.py", "layers/transformer.py",
            "layers/model.py", "layers/inference.py", "layers/paged_kv.py",
            "bench/l6_batching_bench.py",
        )
    }
    config = {"device": device, "git_commit": git_commit(), "protocol": protocol, "source_sha256": sources}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    model = load_model()
    print(f"device: {device['chip']}, {device['memory_bytes'] / 2**30:.0f} GiB, mlx {device['mlx']}", flush=True)
    print(f"config_hash: {config_hash} (commit {config['git_commit'][:12]})", flush=True)
    scaling = []
    for batch_size in BATCH_SIZES:
        lengths = (32,) * batch_size
        baseline, batched = measure_pair(model, Engine, BatchedEngine, lengths, lengths)
        row = {
            "batch_size": batch_size, "l4_per_request": baseline, "batched": batched,
            "speedup": batched["tokens_per_second"] / baseline["tokens_per_second"],
            "token_oracle_passed": True,
        }
        scaling.append(row)
        print(
            f"B={batch_size}: L4 {baseline['tokens_per_second']:.2f} tok/s, "
            f"batched {batched['tokens_per_second']:.2f} tok/s, speedup {row['speedup']:.3f}x; "
            f"decode median/p90 {batched['decode']['median_s']:.6f}/{batched['decode']['p90_s']:.6f}s",
            flush=True,
        )

    small, large = measure_pair(model, BatchedEngine, BatchedEngine, SMALL_VARIANCE, LARGE_VARIANCE)
    padding = {
        "small_variance": {**padding_accounting(SMALL_VARIANCE), **small},
        "large_variance": {**padding_accounting(LARGE_VARIANCE), **large},
        "large_over_small_throughput": large["tokens_per_second"] / small["tokens_per_second"],
        "throughput_penalty_percent": 100 * (1 - large["tokens_per_second"] / small["tokens_per_second"]),
    }
    singleton = scaling[0]
    regression = singleton["batched"]["decode"]["median_s"] / singleton["l4_per_request"]["decode"]["median_s"] - 1
    latency = {
        "l4_ms_per_generated_token": singleton["l4_per_request"]["decode"]["median_s"] * 1000 / GENERATED_TOKENS,
        "batched_ms_per_generated_token": singleton["batched"]["decode"]["median_s"] * 1000 / GENERATED_TOKENS,
        "regression_percent": 100 * regression,
        "passed": regression < 0.10,
    }
    predictions = {"b8_speedup_at_least_4x": scaling[-1]["speedup"] >= 4, "b1_regression_below_10_percent": latency["passed"]}
    record = {
        "iteration": "L6-T1", "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash, **config, "throughput_scaling": scaling,
        "single_request_latency": latency, "uneven_lengths": padding, "predictions": predictions,
    }
    out_dir = ROOT / "results/l6"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path = out_dir / f"batching-bench-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"B=1 latency: L4 {latency['l4_ms_per_generated_token']:.4f} ms/token, batched {latency['batched_ms_per_generated_token']:.4f} ms/token; regression {latency['regression_percent']:.2f}%")
    print(f"uneven B=8: small {small['tokens_per_second']:.2f} tok/s, large {large['tokens_per_second']:.2f} tok/s; penalty {padding['throughput_penalty_percent']:.2f}%")
    print(f"padding fractions: small {padding['small_variance']['padding_fraction']:.4%}, large {padding['large_variance']['padding_fraction']:.4%}")
    print(f"predictions: {json.dumps(predictions, sort_keys=True)}")
    print(f"written: {out_path.relative_to(ROOT)}", flush=True)
    if not all(predictions.values()):
        raise SystemExit("FAIL: performance prediction(s) did not meet the specified thresholds; raw evidence retained")
    print("PASS: L4 token oracle, B=8 throughput >=4x, B=1 latency regression <10%")


if __name__ == "__main__":
    main()
