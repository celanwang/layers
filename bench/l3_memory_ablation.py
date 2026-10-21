"""Measure MLX peak allocation for L3 training memory ablations.

Run the parent once to execute each configuration in a fresh process and write
``results/l3/memory-ablation.json``.  The child modes isolate allocator peaks
so the cache from a larger case cannot contaminate a smaller one.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import GPT, cross_entropy
from layers.optimizer import AdamW, clip_grad_norm, cosine_lr

CASES = {
    "train-b16-l4": {"batch_size": 16, "n_layer": 4, "autograd": True},
    "train-b16-l4-cache0": {"batch_size": 16, "n_layer": 4, "autograd": True, "cache_limit": 0},
    "forward-b16-l4": {"batch_size": 16, "n_layer": 4, "autograd": False},
    "train-b8-l4": {"batch_size": 8, "n_layer": 4, "autograd": True},
    "train-b16-l2": {"batch_size": 16, "n_layer": 2, "autograd": True},
}
CONTEXT = 256
D_MODEL = 512
N_HEAD = 8
VOCAB = 1028
REPEATS = 3


def run_case(name: str) -> dict:
    case = CASES[name]
    if "cache_limit" in case:
        mx.set_cache_limit(case["cache_limit"])
    model = GPT(
        vocab_size=VOCAB,
        n_layer=case["n_layer"],
        d_model=D_MODEL,
        n_head=N_HEAD,
        ctx=CONTEXT,
        seed=42,
    )
    mx.eval(model.parameters())
    batch_size = case["batch_size"]
    inputs = mx.random.randint(0, VOCAB, (batch_size, CONTEXT))
    targets = mx.random.randint(0, VOCAB, (batch_size, CONTEXT))

    def loss_fn(current_model, x, y):
        return cross_entropy(current_model(x), y)

    if case["autograd"]:
        value_and_grad = nn.value_and_grad(model, loss_fn)
        optimizer = AdamW(learning_rate=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    initial_loss = loss_fn(model, inputs, targets)
    mx.eval(initial_loss)
    initial_loss_value = float(initial_loss.item())
    del initial_loss
    gc.collect()
    mx.synchronize()
    baseline_active = int(mx.get_active_memory())
    baseline_cache = int(mx.get_cache_memory())
    mx.reset_peak_memory()

    observed_losses = []
    for step in range(REPEATS):
        if case["autograd"]:
            loss, grads = value_and_grad(model, inputs, targets)
            mx.eval(loss, grads)
            clipped, _ = clip_grad_norm(grads, 1.0)
            updated = optimizer.step(
                model.trainable_parameters(),
                clipped,
                cosine_lr(step + 1, REPEATS, 3e-4, 1),
            )
            model.update(updated)
            mx.eval(model.parameters(), optimizer.first_moment, optimizer.second_moment)
        else:
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            mx.eval(loss)
        observed_losses.append(float(loss.item()))

    mx.synchronize()
    config = {
        **case,
        "context": CONTEXT,
        "d_model": D_MODEL,
        "n_head": N_HEAD,
        "vocab_size": VOCAB,
        "dtype": "float32",
        "seed": 42,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1, "clip_norm": 1.0},
        "measured_updates": REPEATS,
        "pre_reset_eval": True,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    result = {
        "case": name,
        "config_hash": config_hash,
        "config": config,
        "baseline_active_bytes": baseline_active,
        "baseline_cache_bytes": baseline_cache,
        "peak_bytes": int(mx.get_peak_memory()),
        "active_after_bytes": int(mx.get_active_memory()),
        "cache_after_bytes": int(mx.get_cache_memory()),
        "losses": observed_losses,
        "initial_loss": initial_loss_value,
        "platform": platform.platform(),
        "mlx": mx.__version__,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--output", type=Path, default=ROOT / "results/l3/memory-ablation.json")
    args = parser.parse_args()
    if args.case:
        print(json.dumps(run_case(args.case), sort_keys=True))
        return

    records = []
    for name in CASES:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", name],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        records.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        memory_bytes = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        chip, memory_bytes = None, None
    artifact = {
        "iteration": "L3-T2 memory ablation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip(),
        "protocol": {
            "process_isolation": True,
            "measured_updates_per_case": REPEATS,
            "one_initial_eval_before_peak_reset": True,
            "metric": "MLX get_peak_memory high-water since reset_peak_memory; active/cache recorded separately",
            "device": {"chip": chip, "memory_bytes": memory_bytes, "mlx_device": mx.default_device().__repr__()},
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "mlx": mx.__version__,
            "no_independent_repetitions_or_quantiles": True,
        },
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"written: {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
