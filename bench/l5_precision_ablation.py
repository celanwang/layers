"""L5 read-only component ablations; never changes production paths/protocols."""

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.inference import QuantKVCache
from layers.quant import dequantize_kv, quantize_kv
from layers.transformer import LayerNorm


def measure(operation):
    samples = []
    for index in range(13):
        started = time.perf_counter()
        result = operation()
        mx.eval(result)
        elapsed = time.perf_counter() - started
        if index >= 3:
            samples.append(elapsed)
    ordered = sorted(samples)
    return {"median_s": ordered[5], "p90_s": ordered[8], "samples_s": samples}


def main():
    config = {
        "device": mx.device_info(), "python": platform.python_version(),
        "platform": platform.platform(), "mlx": mx.__version__, "seed": 42,
        "warmup": 3, "reps": 10, "contexts": [128, 512, 2048],
        "percentiles": "L0 upper median sorted[5], nearest-rank p90 sorted[8]",
        "scope": "isolated four-layer cache stages; no attention/model/sampling",
        "cache": {"batch": 1, "heads": 8, "features": 64, "layers": 4},
        "timer": "host perf_counter through mx.eval; setup inputs already evaluated",
        "sources_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "bench/l5_precision_ablation.py", "layers/quant.py",
                "layers/inference.py", "layers/transformer.py",
            )
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    mx.random.seed(config["seed"])
    cache_results = []
    for length in config["contexts"]:
        old = [mx.random.normal((1, 8, length, 64)) for _ in range(8)]
        new = [mx.random.normal((1, 8, 1, 64)) for _ in range(8)]
        mx.eval(old, new)
        packed_old = [quantize_kv(array) for array in old]
        packed_new = [quantize_kv(array) for array in new]
        packed_joined = [
            (mx.concatenate((q, nq), axis=-2), mx.concatenate((s, ns), axis=-2))
            for (q, s), (nq, ns) in zip(packed_old, packed_new)
        ]
        mx.eval(packed_joined)

        def original_append():
            cache = QuantKVCache(4)
            # Fresh wrapper around immutable evaluated inputs, so every trial
            # appends one token to the same length, rather than growing trials.
            for layer in range(4):
                cache._keys[layer], cache._key_scales[layer] = packed_old[2 * layer]
                cache._values[layer], cache._value_scales[layer] = packed_old[2 * layer + 1]
            outputs = []
            for layer in range(4):
                outputs.extend(cache.append(layer, new[2 * layer], new[2 * layer + 1]))
            return outputs

        def prequantized_append():
            outputs = []
            for layer in range(4):
                joined = []
                for index in (2 * layer, 2 * layer + 1):
                    q, s = packed_old[index]
                    nq, ns = packed_new[index]
                    joined.append((mx.concatenate((q, nq), axis=-2),
                                   mx.concatenate((s, ns), axis=-2)))
                mx.eval(joined)
                outputs.extend(dequantize_kv(q, s) for q, s in joined)
            return outputs

        original = original_append()
        prequantized = prequantized_append()
        mx.eval(original, prequantized)
        max_error = max(float(mx.max(mx.abs(a - b)).item())
                        for a, b in zip(original, prequantized))
        assert max_error == 0.0
        stages = {
            "original_append_dequant": measure(original_append),
            "prequantized_append_dequant": measure(prequantized_append),
            "quantize_new_only": measure(lambda: [quantize_kv(a) for a in new]),
            "dequant_full_only": measure(lambda: [dequantize_kv(q, s) for q, s in packed_joined]),
            "fp32_concat_only": measure(lambda: [mx.concatenate((a, b), axis=-2)
                                                   for a, b in zip(old, new)]),
        }
        cache_results.append({"initial_context": length, "read_context": length + 1,
                              "prequantized_max_abs_difference": max_error, "stages": stages})

    elementwise = []
    matmul = []
    dtypes = []
    for dtype in (mx.float32, mx.bfloat16):
        x = mx.random.normal((16, 256, 2048)).astype(dtype)
        a = mx.random.normal((16, 256, 512)).astype(dtype)
        w = mx.random.normal((512, 2048)).astype(dtype)
        norm = LayerNorm(512)
        norm.gamma = norm.gamma.astype(dtype)
        norm.beta = norm.beta.astype(dtype)
        mx.eval(x, a, w, norm.parameters())
        gelu = lambda: 0.5 * x * (1.0 + mx.erf(x / (2.0 ** 0.5)))
        elementwise.append({"dtype": str(dtype), "shape": list(x.shape),
                            "operation": "uncompiled exact GELU from layers/transformer.py:49",
                            **measure(gelu)})
        matmul.append({"dtype": str(dtype), "shapes": [list(a.shape), list(w.shape)],
                       **measure(lambda: a @ w)})
        dtypes.append({"input": str(dtype), "gelu": str(gelu().dtype),
                       "layernorm": str(norm(a).dtype), "mean": str(mx.mean(a).dtype),
                       "exp": str(mx.exp(a).dtype), "sum": str(mx.sum(a).dtype),
                       "gradient": str(mx.grad(lambda z: mx.sum(gelu_for(z)))(a).dtype)})

    output = {"iteration": "L5-diff-ablation", "config": config,
              "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
              "kv": cache_results, "gelu": elementwise, "rectangular_matmul": matmul,
              "dtype_probe": dtypes,
              "limits": "Component times are not additive; synchronization/overlap differ. No kernel trace or full-step causal attribution."}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / f"results/l5/precision-ablation-{stamp}.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"written: {path.relative_to(ROOT)}; config_hash: {output['config_hash']}")
    for row in cache_results:
        print(row["initial_context"], {k: round(v["median_s"] * 1000, 4)
                                       for k, v in row["stages"].items()})
    print("GELU", [(r["dtype"], r["median_s"]) for r in elementwise])
    print("matmul", [(r["dtype"], r["median_s"]) for r in matmul])
    print("dtype", dtypes)


def gelu_for(x):
    return 0.5 * x * (1.0 + mx.erf(x / (2.0 ** 0.5)))


if __name__ == "__main__":
    main()
