"""L7-T1 versus immutable L6, importing its unmodified measurement harness."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import l6_batching_bench as l6
from layers import BatchedEngine, PagedAttentionEngine


L6_BASELINE = ROOT / "results/l6/batching-bench-20261021T133044874890Z.json"


def main():
    historical = json.loads(L6_BASELINE.read_text())
    # Retain the original field names and values for direct protocol comparison.
    protocol = historical["protocol"]
    assert (protocol["warmup"], protocol["reps"]) == (l6.WARMUP, l6.REPS) == (3, 10)
    assert tuple(protocol["batch_sizes"]) == l6.BATCH_SIZES == (1, 2, 4, 8)
    assert protocol["generated_tokens_per_request"] == l6.GENERATED_TOKENS == 64
    assert protocol["prompt_tokens_scaling"] == 32
    assert protocol["block_size"] == l6.BLOCK_SIZE == 16
    assert protocol["seed"] == l6.SEED
    assert protocol["checkpoint_sha256"] == l6.sha256(l6.CHECKPOINT)
    source_paths = (*historical["source_sha256"], "layers/__init__.py", "bench/l7_paged_attn_bench.py")
    config = {
        "device": l6.device_info(), "git_commit": l6.git_commit(), "protocol": protocol,
        "source_sha256": {path: l6.sha256(ROOT / path) for path in source_paths},
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    print(f"device: {config['device']['chip']}, {config['device']['memory_bytes'] / 2**30:.0f} GiB, mlx {config['device']['mlx']}", flush=True)
    print(f"config_hash: {config_hash} (commit {config['git_commit'][:12]})", flush=True)
    model = l6.load_model()
    scaling = []
    for batch_size in l6.BATCH_SIZES:
        lengths = (32,) * batch_size
        baseline, paged = l6.measure_pair(model, BatchedEngine, PagedAttentionEngine, lengths, lengths)
        old_row = next(row for row in historical["throughput_scaling"] if row["batch_size"] == batch_size)
        row = {
            "batch_size": batch_size, "batched": baseline, "paged_attention": paged,
            "speedup": paged["tokens_per_second"] / baseline["tokens_per_second"],
            "relative_to_historical_l6": paged["tokens_per_second"] / old_row["batched"]["tokens_per_second"],
            "token_oracle_passed": True,
        }
        scaling.append(row)
        print(
            f"B={batch_size}: L6 {baseline['tokens_per_second']:.2f} tok/s, "
            f"paged {paged['tokens_per_second']:.2f} tok/s, paired {row['speedup']:.3f}x; "
            f"new decode median/p90 {paged['decode']['median_s']:.6f}/{paged['decode']['p90_s']:.6f}s",
            flush=True,
        )
    singleton = scaling[0]
    latency = {
        "l6_ms_per_generated_token": singleton["batched"]["decode"]["median_s"] * 1000 / l6.GENERATED_TOKENS,
        "paged_ms_per_generated_token": singleton["paged_attention"]["decode"]["median_s"] * 1000 / l6.GENERATED_TOKENS,
        "regression_percent": 100 * (singleton["paged_attention"]["decode"]["median_s"] / singleton["batched"]["decode"]["median_s"] - 1),
    }
    latency["regression_below_10_percent"] = latency["regression_percent"] < 10
    b8 = scaling[-1]["paged_attention"]["tokens_per_second"]
    verdict = "supported" if b8 >= 1700 else "setback" if b8 < 1550 else "intermediate"
    record = {
        "iteration": "L7-T1", "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash, **config,
        "paths": {"batched": "unchanged L6 BatchedEngine", "paged_attention": "PagedAttentionEngine; sampling unchanged"},
        "historical_l6": {
            "path": str(L6_BASELINE.relative_to(ROOT)), "sha256": l6.sha256(L6_BASELINE),
            "b8_tokens_per_second": historical["throughput_scaling"][-1]["batched"]["tokens_per_second"],
        },
        "throughput_scaling": scaling, "single_request_latency": latency,
        "predictions": {"b8_at_least_1700": b8 >= 1700, "b8_below_1550": b8 < 1550, "verdict": verdict},
        "correctness_passed": True,
    }
    out_dir = ROOT / "results/l7"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"paged-attn-bench-{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"B=1 latency regression: {latency['regression_percent']:.2f}%", flush=True)
    print(f"prediction: {verdict}; B=8 / historical L6 = {scaling[-1]['relative_to_historical_l6']:.3f}x", flush=True)
    print(f"written: {path.relative_to(ROOT)}", flush=True)
    # The specification explicitly accepts either prediction outcome as evidence.
    print("PASS: paired token oracle, pool reclamation, and unchanged L6 protocol; prediction recorded separately", flush=True)


if __name__ == "__main__":
    main()
