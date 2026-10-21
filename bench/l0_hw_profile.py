"""L0: hardware profile of the local experiment machine (Apple M4 Max, 48GB).

Measures the two quantities every later iteration reasons about:
  - memory bandwidth (GB/s) via a triad-like elementwise op  c = a + b
  - matmul throughput (TFLOP/s) across dtypes and sizes

Protocol (per AGENTS.md measurement discipline): warmup runs, repeated timed
runs, median/p90 reported, and a config hash (git commit + package versions)
attached to every record. Results are appended as JSON to results/l0/.
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

WARMUP = 3
REPS = 10


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def config_hash() -> str:
    payload = f"{platform.platform()}|{sys.version}|mlx-{mx.__version__}|{git_commit()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


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


def timed(fn, *args) -> list[float]:
    for _ in range(WARMUP):
        mx.eval(fn(*args))
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples


def summarize(samples: list[float]) -> dict:
    n = len(samples)
    return {
        "median_s": samples[n // 2],
        "p90_s": samples[int(n * 0.9) - 1],
        "min_s": samples[0],
        "reps": n,
        "warmup": WARMUP,
    }


def bench_bandwidth() -> list[dict]:
    rows = []
    for n in [2**24, 2**26, 2**28]:  # 16M / 64M / 256M elements
        a = mx.random.normal((n,), dtype=mx.float32)
        b = mx.random.normal((n,), dtype=mx.float32)
        mx.eval(a, b)
        stats = summarize(timed(lambda x, y: x + y, a, b))
        gb_moved = 3 * n * 4 / 1e9  # read a, read b, write c (fp32)
        rows.append(
            {
                "op": "add(fp32)",
                "elements": n,
                "bandwidth_gbs_median": round(gb_moved / stats["median_s"], 1),
                "bandwidth_gbs_best": round(gb_moved / stats["min_s"], 1),
                **stats,
            }
        )
    return rows


def bench_matmul() -> list[dict]:
    rows = []
    for dtype in [mx.float32, mx.float16, mx.bfloat16]:
        for n in [1024, 2048, 4096]:
            a = mx.random.normal((n, n), dtype=dtype)
            b = mx.random.normal((n, n), dtype=dtype)
            mx.eval(a, b)
            stats = summarize(timed(lambda x, y: x @ y, a, b))
            tflops = 2 * n**3 / 1e12
            rows.append(
                {
                    "op": f"matmul({dtype})",
                    "shape": [n, n, n],
                    "tflops_median": round(tflops / stats["median_s"], 2),
                    "tflops_best": round(tflops / stats["min_s"], 2),
                    **stats,
                }
            )
    return rows


def main() -> None:
    record = {
        "iteration": "L0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(),
        "device": device_info(),
        "bandwidth": bench_bandwidth(),
        "matmul": bench_matmul(),
    }
    out_dir = Path("results/l0")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"hw-profile-{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"device: {record['device']['chip']}, "
          f"{record['device']['memory_bytes'] / 2**30:.0f} GiB, mlx {record['device']['mlx']}")
    print(f"config_hash: {record['config_hash']}")
    for r in record["bandwidth"]:
        print(f"  bandwidth {r['op']:>14} n=2^{r['elements'].bit_length()-1}: "
              f"{r['bandwidth_gbs_median']:>6} GB/s (median)")
    for r in record["matmul"]:
        print(f"  matmul    {r['op']:>18} {r['shape'][0]:>5}: "
              f"{r['tflops_median']:>7} TFLOP/s (median)")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
