"""L5-T1: serialized storage, heldout quality, and cached greedy decode.

Keeps L0's three warmups, ten repetitions, upper median and nearest-rank p90.
Only the optional Oracle calls MLX's public quantization APIs.
"""

import hashlib
import io
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import GPT, Sampler, cross_entropy, dequantize, generate, quantize, quantize_model_
from scripts.train import device_info, make_corpus, sha256_file


WARMUP = 3
REPS = 10
PROMPT_TOKENS = 128
GENERATED_TOKENS = 128
CONTEXT = 256
BATCH_SIZE = 16
CHECKPOINT = ROOT / "results/l3/ckpt-final-500.npz"
FORMATS = (("fp32", None, None), ("int8", 8, None), ("int4-g64", 4, 64))


def load_model() -> GPT:
    model = GPT()
    with np.load(CHECKPOINT) as checkpoint:
        for name in checkpoint.files:
            parts = name.split(".")
            current = model
            for part in parts[:-1]:
                current = current[int(part)] if part.isdigit() else getattr(current, part)
            setattr(current, parts[-1], mx.array(checkpoint[name]))
    mx.eval(model.parameters())
    return model


def named_arrays(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from named_arrays(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, (tuple, list)):
        for index, value in enumerate(node):
            yield from named_arrays(value, f"{prefix}.{index}")
    else:
        yield prefix, node


def storage_stats(model, quantized: bool) -> dict:
    """Serialize actual stored arrays to uncompressed NPZ, including headers."""
    arrays = {name: np.asarray(value) for name, value in named_arrays(model.parameters())}
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload_bytes = sum(value.nbytes for value in arrays.values())
    if quantized:
        integer_bytes = sum(value.nbytes for name, value in arrays.items() if name.endswith(".q"))
        scale_bytes = sum(value.nbytes for name, value in arrays.items() if name.endswith(".scales"))
        projection_bytes = integer_bytes + scale_bytes
    else:
        integer_bytes = scale_bytes = 0
        projection_bytes = sum(
            value.nbytes for name, value in arrays.items()
            if ".attn.w" in name or ".mlp.w" in name
        )
    return {
        "array_payload_bytes": payload_bytes,
        "projection_payload_bytes": projection_bytes,
        "integer_bytes": integer_bytes,
        "fp16_scale_bytes": scale_bytes,
        "protected_fp32_bytes": payload_bytes - projection_bytes,
        "serialized_npz_bytes": buffer.tell(),
        "container_overhead_bytes": buffer.tell() - payload_bytes,
        "serialized_npz_sha256": hashlib.sha256(buffer.getbuffer()).hexdigest(),
        "serialization": "numpy.savez, uncompressed, in memory; q/scales only for projections",
    }


def heldout_loss(model, heldout: np.ndarray) -> float:
    """Token-weighted loss over every next-token pair in the heldout partition.

    Non-overlapping context windows reset attention at each 256-token boundary.
    The final short window is retained, and no target crosses the train split.
    """
    tokens = len(heldout) - 1
    if tokens < 1:
        raise ValueError("heldout must contain at least two tokens")
    full_windows, tail = divmod(tokens, CONTEXT)
    weighted_loss = 0.0
    offsets = np.arange(CONTEXT)
    for start in range(0, full_windows, BATCH_SIZE):
        starts = np.arange(start, min(start + BATCH_SIZE, full_windows)) * CONTEXT
        inputs = mx.array(heldout[starts[:, None] + offsets])
        targets = mx.array(heldout[starts[:, None] + offsets + 1])
        loss = cross_entropy(model(inputs), targets)
        mx.eval(loss)
        weighted_loss += float(loss.item()) * len(starts) * CONTEXT
    if tail:
        start = full_windows * CONTEXT
        loss = cross_entropy(
            model(mx.array(heldout[start:-1])), mx.array(heldout[start + 1:])
        )
        mx.eval(loss)
        weighted_loss += float(loss.item()) * tail
    return weighted_loss / tokens


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_s": ordered[len(ordered) // 2],
        "p90_s": ordered[int(len(ordered) * 0.9) - 1],
        "min_s": ordered[0],
        "warmup": WARMUP,
        "reps": len(ordered),
        "samples_s": samples,
    }


class _TimedGreedy(Sampler):
    """Timestamp after sampling to separate prefill from steady cached decode."""

    def __init__(self):
        super().__init__(temperature=0)
        self.first_sample_at = None
        self.last_sample_at = None

    def __call__(self, logits):
        token = super().__call__(logits)
        now = time.perf_counter()
        if self.first_sample_at is None:
            self.first_sample_at = now
        self.last_sample_at = now
        return token


def decode_stats(model, tokenizer, prompt: list[int]) -> dict:
    decode_samples, total_samples, prefill_samples = [], [], []
    for repetition in range(WARMUP + REPS):
        sampler = _TimedGreedy()
        started = time.perf_counter()
        # generate creates a fresh L4 KVCache, prefills once, then decodes.
        text = "".join(generate(model, tokenizer, prompt, GENERATED_TOKENS, sampler))
        finished = time.perf_counter()
        if not text or sampler._draws != GENERATED_TOKENS:
            raise AssertionError("generation did not emit the requested token count")
        if repetition >= WARMUP:
            total_samples.append(finished - started)
            prefill_samples.append(sampler.first_sample_at - started)
            decode_samples.append(sampler.last_sample_at - sampler.first_sample_at)
    # First generated token comes from prefill, leaving 127 single-token calls.
    decode = summarize(decode_samples)
    total = summarize(total_samples)
    return {
        "prompt_tokens": len(prompt),
        "generated_tokens": GENERATED_TOKENS,
        "timed_cached_decode_tokens": GENERATED_TOKENS - 1,
        "decode_timing": "after first sampled token through final sampled token; sampling/streaming included",
        "decode": {
            **decode,
            "tokens_per_second_median": (GENERATED_TOKENS - 1) / decode["median_s"],
        },
        "prefill_and_first_sample": summarize(prefill_samples),
        "end_to_end": {
            **total,
            "tokens_per_second_median": GENERATED_TOKENS / total["median_s"],
        },
    }


def mlx_api_oracle() -> dict:
    """Compare public API outputs, without reading any quantization source.

    Native affine quantize is a different algorithm, so equality is not expected.
    Separately, encode our signed integers as unsigned (q+qmax), pass bias=-s*qmax
    to native dequantize, and check independent reconstruction of *our* grid.
    """
    if not all(hasattr(mx, name) for name in ("quantize", "dequantize")):
        return {"available": False}
    w = mx.array(np.random.default_rng(5).normal(0, 0.1, (16, 128)).astype(np.float32))
    rows = []
    for bits, group in ((8, 128), (4, 64)):
        q, scales = quantize(w, bits, group)
        ours = dequantize(q, scales, bits, group)
        raw = np.asarray(q).astype(np.int32)
        if bits == 4:
            nibbles = np.stack((raw & 15, raw >> 4), axis=-1).reshape(w.shape)
            raw = np.where(nibbles >= 8, nibbles - 16, nibbles)
        qmax = 2 ** (bits - 1) - 1
        unsigned = (raw + qmax).astype(np.uint32)
        per_word = 32 // bits
        words = np.bitwise_or.reduce(
            unsigned.reshape(w.shape[0], -1, per_word)
            << (np.arange(per_word, dtype=np.uint32) * bits), axis=-1
        )
        s32 = scales.astype(mx.float32)
        native_our_grid = mx.dequantize(
            mx.array(words), s32, -qmax * s32, group_size=group, bits=bits
        )
        native_q, native_s, native_b = mx.quantize(w, group_size=group, bits=bits)
        native_roundtrip = mx.dequantize(
            native_q, native_s, native_b, group_size=group, bits=bits
        )
        mx.eval(ours, native_our_grid, native_roundtrip)
        grid_error = float(mx.max(mx.abs(ours - native_our_grid)).item())
        if grid_error > 1e-6:
            raise AssertionError(f"MLX dequantize Oracle disagrees: {grid_error}")
        rows.append({
            "bits": bits,
            "group_size": group,
            "our_grid_vs_mlx_dequantize_max_abs": grid_error,
            "our_vs_native_affine_max_abs": float(mx.max(mx.abs(ours - native_roundtrip)).item()),
            "our_roundtrip_rmse": float(mx.sqrt(mx.mean((w - ours) ** 2)).item()),
            "native_affine_roundtrip_rmse": float(mx.sqrt(mx.mean((w - native_roundtrip) ** 2)).item()),
        })
    return {
        "available": True,
        "native_mode": "affine (scale plus bias), unlike our zero-point-zero symmetric grid",
        "int8_oracle_group": "128 (API supports 32/64/128); production task uses full input width",
        "comparisons": rows,
    }


def main() -> None:
    out_dir = ROOT / "results/l5"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, _train, heldout, corpus = make_corpus()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    source_paths = (
        "layers/quant.py", "layers/__init__.py", "layers/model.py", "layers/attention.py",
        "layers/transformer.py", "layers/inference.py", "bench/l5_quant_bench.py", "scripts/train.py",
    )
    config = {
        "git_commit": commit,
        "sources_sha256": {path: sha256_file(ROOT / path) for path in source_paths},
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "device": device_info(),
        "corpus": corpus,
        "warmup": WARMUP, "reps": REPS,
        "percentiles": "L0 upper median (sorted[n//2]), p90 (sorted[int(n*0.9)-1])",
        "formats": FORMATS,
        "scale_dtype": "float16", "matmul_dtype": "float32",
        "int4_packing": "two signed nibbles per uint8, low nibble first",
        "quality": {"context": CONTEXT, "batch_size": BATCH_SIZE, "evaluated_target_tokens": len(heldout) - 1,
                    "scope": "entire last 10% partition, nonoverlapping windows, token weighted, tail retained"},
        "generation": {"prompt_tokens": PROMPT_TOKENS, "max_new": GENERATED_TOKENS,
                       "sampler": "L4 Sampler(temperature=0)", "prompt": "first 128 heldout IDs"},
    }
    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    print(f"device: {config['device']['chip']}, mlx {mx.__version__}", flush=True)
    print(f"config_hash: {run_hash} (commit {commit[:12]})", flush=True)
    print(f"heldout: split={corpus['split_token_index']}, {len(heldout) - 1} target tokens", flush=True)
    rows = []
    for name, bits, group in FORMATS:
        model = load_model()
        if bits is not None:
            quantize_model_(model, bits, group)
        memory = storage_stats(model, bits is not None)
        loss = heldout_loss(model, heldout)
        speed = decode_stats(model, tokenizer, heldout[:PROMPT_TOKENS].tolist())
        row = {"format": name, "storage": memory, "heldout_loss_nats": loss, "generation": speed}
        rows.append(row)
        print(f"{name}: NPZ={memory['serialized_npz_bytes']} B, loss={loss:.8f}, "
              f"decode={speed['decode']['tokens_per_second_median']:.2f} tok/s "
              f"(median={speed['decode']['median_s']:.6f}s, p90={speed['decode']['p90_s']:.6f}s)", flush=True)
        del model
    baseline = rows[0]
    for row in rows:
        memory = row["storage"]
        memory["whole_model_serialized_compression"] = baseline["storage"]["serialized_npz_bytes"] / memory["serialized_npz_bytes"]
        memory["projection_payload_compression"] = baseline["storage"]["projection_payload_bytes"] / memory["projection_payload_bytes"]
        row["heldout_loss_relative_increase"] = row["heldout_loss_nats"] / baseline["heldout_loss_nats"] - 1.0
        row["decode_speedup"] = row["generation"]["decode"]["tokens_per_second_median"] / baseline["generation"]["decode"]["tokens_per_second_median"]
    record = {
        "iteration": "L5-T1", "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": run_hash, "config": config, "measurements": rows,
        "mlx_public_api_oracle": mlx_api_oracle(),
        "quality_gates": {"int8_relative_increase_lt_0.01": rows[1]["heldout_loss_relative_increase"] < 0.01,
                          "int4_relative_increase_lt_0.05": rows[2]["heldout_loss_relative_increase"] < 0.05},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"quant-bench-{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['format']}: model compression={row['storage']['whole_model_serialized_compression']:.4f}x, "
              f"projection compression={row['storage']['projection_payload_compression']:.4f}x, "
              f"loss change={row['heldout_loss_relative_increase']:+.4%}, decode speedup={row['decode_speedup']:.4f}x")
    print("MLX public API Oracle: " + json.dumps(record["mlx_public_api_oracle"]))
    print(f"written: {path.relative_to(ROOT)}", flush=True)
    if not all(math.isfinite(row["heldout_loss_nats"]) for row in rows) or not all(record["quality_gates"].values()):
        raise AssertionError("heldout quality gates failed; raw measurements were retained")


if __name__ == "__main__":
    main()
