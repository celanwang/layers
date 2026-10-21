"""L5-T2: KV storage/decode (L0 3+10) and L3-recipe bf16 training."""

import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.l5_quant_bench import CHECKPOINT, load_model
from layers.inference import KVCache, QuantKVCache
from scripts.train import device_info, make_corpus, run_bf16_training, sha256_file

WARMUP = 3
REPS = 10
CONTEXTS = (128, 512, 2048)
DECODE_STEPS = 32


def prefill(model, prompt, cache):
    """Keep each call within trained ctx; RoPE offsets follow the full cache.

    Contexts beyond 256 are positional extrapolation of the L3 checkpoint,
    not a claim about quality of a model trained on 2048-token examples.
    """
    for start in range(0, len(prompt), model.ctx):
        logits = model(mx.array(prompt[start:start + model.ctx], dtype=mx.int32), cache=cache)
        mx.eval(logits)
    return logits[-1]


def cache_bytes(cache):
    if isinstance(cache, QuantKVCache):
        return cache.storage_nbytes
    return sum(x.nbytes for arrays in (cache._keys, cache._values)
               for x in arrays if x is not None)


def summarize(samples):
    ordered = sorted(samples)
    return {"median_s": ordered[len(ordered) // 2],
            "p90_s": ordered[int(len(ordered) * 0.9) - 1],
            "min_s": ordered[0], "warmup": WARMUP, "reps": len(samples),
            "samples_s": samples}


def logit_comparison(model, prompt, steps=DECODE_STEPS):
    """Compare logits on an identical fp32-greedy history; retain raw curve.

    Individual errors need not increase with sequence length. Their running
    maximum is a monotone envelope; independent greedy divergence is measured
    separately so changed input tokens cannot masquerade as quantization error.
    """
    fp, quant = KVCache(model.n_layer), QuantKVCache(model.n_layer)
    a, b = prefill(model, prompt, fp), prefill(model, prompt, quant)
    rows = []
    running_max = 0.0
    first_aligned_disagreement = None
    for step in range(steps + 1):
        mx.eval(a, b)
        error = float(mx.max(mx.abs(a - b)).item())
        if not math.isfinite(error):
            raise AssertionError("INT8 KV logits diverged to non-finite values")
        running_max = max(running_max, error)
        fp_token, quant_token = int(mx.argmax(a).item()), int(mx.argmax(b).item())
        if fp_token != quant_token and first_aligned_disagreement is None:
            first_aligned_disagreement = step
        rows.append({"step": step, "cache_tokens": fp.seq_len,
                     "max_abs_diff": error, "running_max_abs_diff": running_max,
                     "fp32_greedy_token": fp_token, "int8_greedy_token": quant_token})
        if step < steps:
            ids = mx.array([fp_token], dtype=mx.int32)
            a, b = model(ids, cache=fp)[-1], model(ids, cache=quant)[-1]
    return {"history": "identical fp32 greedy tokens fed to both caches",
            "curve": rows, "max_abs_diff": running_max,
            "first_aligned_greedy_disagreement_step": first_aligned_disagreement,
            "monotonicity": "raw errors can decrease; running maximum is a monotone envelope"}


def decode_trial(model, prompt, cache_type):
    cache = cache_type(model.n_layer)
    logits = prefill(model, prompt, cache)
    initial_bytes = cache_bytes(cache)
    token = int(mx.argmax(logits).item())
    tokens = [token]
    mx.reset_peak_memory()
    started = time.perf_counter()
    for _ in range(DECODE_STEPS):
        logits = model(mx.array([token], dtype=mx.int32), cache=cache)[-1]
        mx.eval(logits)
        token = int(mx.argmax(logits).item())
        tokens.append(token)
    elapsed = time.perf_counter() - started
    return {"seconds": elapsed, "initial_kv_bytes": initial_bytes,
            "final_kv_bytes": cache_bytes(cache), "final_cache_tokens": cache.seq_len,
            "peak_total_active_bytes": int(mx.get_peak_memory()), "greedy_tokens": tokens}


def kv_measurements(model, heldout):
    rows = []
    for context in CONTEXTS:
        prompt = heldout[:context].tolist()
        trials = {"fp32": [], "int8": []}
        formats = (("fp32", KVCache), ("int8", QuantKVCache))
        for repetition in range(WARMUP + REPS):
            # Alternate order to reduce systematic warm/thermal ordering bias.
            for name, cache_type in formats[::1 if repetition % 2 == 0 else -1]:
                trial = decode_trial(model, prompt, cache_type)
                if repetition >= WARMUP:
                    trials[name].append(trial)
        formats_record = {}
        for name, samples in trials.items():
            timing = summarize([sample["seconds"] for sample in samples])
            formats_record[name] = {
                "decode": {**timing, "tokens_per_second_median": DECODE_STEPS / timing["median_s"]},
                "initial_kv_bytes": samples[0]["initial_kv_bytes"],
                "final_kv_bytes": samples[0]["final_kv_bytes"],
                "final_cache_tokens": samples[0]["final_cache_tokens"],
                "peak_total_active_bytes_samples": [s["peak_total_active_bytes"] for s in samples],
                "greedy_tokens": samples[0]["greedy_tokens"],
            }
        fp, quant = formats_record["fp32"], formats_record["int8"]
        divergence = next((i for i, (a, b) in enumerate(zip(fp["greedy_tokens"], quant["greedy_tokens"]))
                           if a != b), None)
        row = {
            "context_tokens": context, "decode_steps": DECODE_STEPS, "formats": formats_record,
            "kv_compression_vs_fp32": fp["initial_kv_bytes"] / quant["initial_kv_bytes"],
            "decode_speedup": quant["decode"]["tokens_per_second_median"] / fp["decode"]["tokens_per_second_median"],
            "first_free_greedy_divergence_step": divergence,
            "logit_comparison": logit_comparison(model, prompt),
        }
        rows.append(row)
        print(f"KV n={context}: fp32={fp['initial_kv_bytes']} B, INT8={quant['initial_kv_bytes']} B "
              f"({row['kv_compression_vs_fp32']:.6f}x); decode fp32={fp['decode']['tokens_per_second_median']:.2f}, "
              f"INT8={quant['decode']['tokens_per_second_median']:.2f} tok/s ({row['decode_speedup']:.6f}x); "
              f"max logit diff={row['logit_comparison']['max_abs_diff']:.8f}; greedy fork={divergence}", flush=True)
    return rows


def bf16_measurements():
    """L0 outer repetitions; each repetition retains L3's inner estimator."""
    measured, warmup_paths = [], []
    for repetition in range(WARMUP + REPS):
        record = run_bf16_training()
        if repetition < WARMUP:
            warmup_paths.append(record["result_path"])
        else:
            measured.append(record)
        print(f"bf16 repetition {repetition + 1}/{WARMUP + REPS} complete", flush=True)
        mx.clear_cache()
    baseline = json.loads((ROOT / "results/l3/train-metrics.json").read_text())["training"]
    rates = sorted(r["training"]["step_tokens_per_second_median"] for r in measured)
    peaks = sorted(r["training"]["peak_metal_memory_bytes"] for r in measured)
    losses = sorted(r["training"]["final_heldout_loss_nats"] for r in measured)
    rate, peak, loss = rates[REPS // 2], peaks[REPS // 2], losses[REPS // 2]
    mfu = rate * measured[0]["training"]["mfu_parameter_flops_per_token"] / measured[0]["training"]["mfu_oracle_peak_flops"]
    comparison = {
        "tokens_per_second_ratio": rate / baseline["step_tokens_per_second_median"],
        "mfu_relative_change": mfu / baseline["mfu_fraction"] - 1,
        "peak_memory_relative_reduction": 1 - peak / baseline["peak_metal_memory_bytes"],
        "final_loss_relative_difference": loss / baseline["final_heldout_loss_nats"] - 1,
    }
    result = {
        "warmup": WARMUP, "reps": REPS, "warmup_result_paths": warmup_paths,
        "measured_runs": measured,
        "summary": {"tokens_per_second_median": rate, "tokens_per_second_p90": rates[int(REPS * .9) - 1],
                    "peak_metal_memory_bytes_median": peak, "peak_metal_memory_bytes_p90": peaks[int(REPS * .9) - 1],
                    "final_heldout_loss_median": loss, "final_heldout_loss_p90": losses[int(REPS * .9) - 1],
                    "mfu_percent": 100 * mfu},
        "comparison": comparison,
        "prediction_verdicts": {
            "mfu_increase_le_10_percent": comparison["mfu_relative_change"] <= .10,
            "memory_reduction_35_to_40_percent": .35 <= comparison["peak_memory_relative_reduction"] <= .40,
            "final_loss_difference_lt_2_percent": abs(comparison["final_loss_relative_difference"]) < .02,
        },
    }
    print("bf16 L0 aggregate: " + json.dumps(result["summary"]), flush=True)
    print("bf16 aggregate vs L3: " + json.dumps(comparison), flush=True)
    print("bf16 predictions: " + json.dumps(result["prediction_verdicts"]), flush=True)
    return result


def main():
    out_dir = ROOT / "results/l5"
    out_dir.mkdir(parents=True, exist_ok=True)
    _tokenizer, _train, heldout, corpus = make_corpus()
    source_paths = (
        "bench/l5_kv_bf16_bench.py", "bench/l5_quant_bench.py", "layers/quant.py",
        "layers/inference.py", "scripts/train.py", "layers/model.py", "layers/attention.py",
        "layers/optimizer.py", "layers/transformer.py", "layers/embedding.py",
    )
    config = {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip(),
        "sources_sha256": {name: sha256_file(ROOT / name) for name in source_paths},
        "device": device_info(), "corpus": corpus,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "warmup": WARMUP, "reps": REPS, "contexts": CONTEXTS, "decode_steps": DECODE_STEPS,
        "percentiles": "L0 upper median sorted[n//2], p90 sorted[int(n*0.9)-1]",
        "kv": {"integer_dtype": "int8", "scale_dtype": "float16", "read_dtype": "float32",
               "scale_granularity": "one per batch/token across all heads/features; independent K/V",
               "memory_scope": "persistent array payload incl scales; excludes ephemeral fp32 reads",
               "context_scope": "chunked <=256-token prefill; >256 is RoPE extrapolation",
               "timing_scope": "32 single-token forwards plus greedy argmax; prefill excluded"},
        "training_protocol": "L0 3 warmup + 10 measured independent 200-step runs; inside each, unchanged L3 first 3 steps excluded and 197 observations; same recipe/hash check",
    }
    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    print(f"device: {config['device']['chip']}; config_hash: {run_hash}", flush=True)
    model = load_model()
    kv = kv_measurements(model, heldout)
    # Release inference arrays before the training peak-memory measurement.
    del model
    mx.clear_cache()
    training = bf16_measurements()
    record = {"iteration": "L5-T2", "timestamp": datetime.now(timezone.utc).isoformat(),
              "config_hash": run_hash, "config": config, "kv": kv, "bf16": training}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"kv-bf16-bench-{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"written: {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
