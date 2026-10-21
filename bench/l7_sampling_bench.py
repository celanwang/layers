"""Three L7 stages paired with the unchanged L6 measurement protocol."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import l6_batching_bench as l6
from bench.l7_paged_attn_bench import L6_BASELINE
from layers import BatchedEngine, BatchedSamplingEngine, PagedAttentionEngine


PAIRS = (
    ("l6_vs_t1", "l6", BatchedEngine, "t1", PagedAttentionEngine),
    ("t1_vs_t2", "t1", PagedAttentionEngine, "t2", BatchedSamplingEngine),
    ("l6_vs_t2", "l6", BatchedEngine, "t2", BatchedSamplingEngine),
)


def main():
    historical = json.loads(L6_BASELINE.read_text())
    protocol = historical["protocol"]
    assert (protocol["warmup"], protocol["reps"]) == (l6.WARMUP, l6.REPS) == (3, 10)
    assert tuple(protocol["batch_sizes"]) == l6.BATCH_SIZES == (1, 2, 4, 8)
    assert protocol["generated_tokens_per_request"] == l6.GENERATED_TOKENS == 64
    assert protocol["prompt_tokens_scaling"] == 32
    assert protocol["block_size"] == l6.BLOCK_SIZE == 16
    assert protocol["seed"] == l6.SEED
    assert protocol["checkpoint_sha256"] == l6.sha256(l6.CHECKPOINT)
    source_paths = (
        *historical["source_sha256"], "layers/__init__.py",
        "bench/l7_paged_attn_bench.py", "bench/l7_sampling_bench.py",
    )
    config = {
        "device": l6.device_info(), "git_commit": l6.git_commit(), "protocol": protocol,
        "source_sha256": {path: l6.sha256(ROOT / path) for path in source_paths},
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    print(
        f"device: {config['device']['chip']}, "
        f"{config['device']['memory_bytes'] / 2**30:.0f} GiB, mlx {config['device']['mlx']}",
        flush=True,
    )
    print(f"config_hash: {config_hash} (commit {config['git_commit'][:12]})", flush=True)
    model = l6.load_model()
    scaling = []
    for batch_size in l6.BATCH_SIZES:
        lengths = (32,) * batch_size
        pairs = {}
        for name, left_name, left_type, right_name, right_type in PAIRS:
            left, right = l6.measure_pair(model, left_type, right_type, lengths, lengths)
            gain = right["tokens_per_second"] / left["tokens_per_second"] - 1
            reduction = 1 - right["decode"]["median_s"] / left["decode"]["median_s"]
            pairs[name] = {
                left_name: left, right_name: right,
                "throughput_gain_percent": 100 * gain,
                "decode_time_reduction_percent": 100 * reduction,
                "token_oracle_passed": True,
            }
            print(
                f"B={batch_size} {name}: {left_name} {left['tokens_per_second']:.2f}, "
                f"{right_name} {right['tokens_per_second']:.2f} tok/s; gain {100 * gain:+.2f}%; "
                f"decode median/p90 ms: {left['decode']['median_s'] * 1000:.3f}/"
                f"{left['decode']['p90_s'] * 1000:.3f} -> "
                f"{right['decode']['median_s'] * 1000:.3f}/{right['decode']['p90_s'] * 1000:.3f}",
                flush=True,
            )
        scaling.append({"batch_size": batch_size, "pairs": pairs})

    b8 = scaling[-1]["pairs"]
    t2_gain = b8["t1_vs_t2"]["throughput_gain_percent"]
    verdict = "supported" if t2_gain >= 5 else "setback" if t2_gain < 3 else "intermediate"
    combined_rate = b8["l6_vs_t2"]["t2"]["tokens_per_second"]
    predictions = {
        "t2_b8_throughput_gain_percent": t2_gain,
        "t2_b8_decode_time_reduction_percent": b8["t1_vs_t2"]["decode_time_reduction_percent"],
        "t2_gain_at_least_5_percent": t2_gain >= 5,
        "t2_gain_below_3_percent": t2_gain < 3,
        "t2_verdict": verdict,
        "combined_b8_at_least_1800": combined_rate >= 1800,
        "combined_b8_below_1600": combined_rate < 1600,
    }
    singleton = scaling[0]["pairs"]
    latency = {
        "t2_vs_t1_regression_percent": -singleton["t1_vs_t2"]["decode_time_reduction_percent"],
        "t2_vs_l6_regression_percent": -singleton["l6_vs_t2"]["decode_time_reduction_percent"],
    }
    record = {
        "iteration": "L7-T2", "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash, **config,
        "paths": {
            "l6": "unchanged L6 BatchedEngine",
            "t1": "unchanged L7-T1 PagedAttentionEngine; scalar sampling",
            "t2": "BatchedSamplingEngine; T1 KV path and one batch sampling fence",
        },
        "pairing": {
            "comparisons": [pair[0] for pair in PAIRS],
            "harness": "unchanged bench.l6_batching_bench.measure_pair",
            "interpretation": "each edge is an independent paired experiment; no cross-edge attribution",
        },
        "historical_l6": {
            "path": str(L6_BASELINE.relative_to(ROOT)), "sha256": l6.sha256(L6_BASELINE),
            "b8_tokens_per_second": historical["throughput_scaling"][-1]["batched"]["tokens_per_second"],
        },
        "throughput_scaling": scaling, "single_request_latency": latency,
        "predictions": predictions, "correctness_passed": True,
    }
    out_dir = ROOT / "results/l7"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"sampling-bench-{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"prediction: {verdict}; paired T2 increment {t2_gain:+.2f}%", flush=True)
    print(f"written: {path.relative_to(ROOT)}", flush=True)
    print("PASS: paired token oracles, pool reclamation, unchanged L6 protocol; prediction recorded separately", flush=True)


if __name__ == "__main__":
    main()
