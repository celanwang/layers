"""Train the assembled L3 GPT on the L1-tokenized tiny Shakespeare corpus."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import BPETokenizer, GPT, cross_entropy
from layers.optimizer import AdamW, clip_grad_norm, cosine_lr


DATA_PATH = ROOT / "data/tiny-shakespeare.txt"
TOKENIZER_PATH = ROOT / "results/l1/tokenizer-1024.json"
RESULTS_DIR = ROOT / "results/l3"
CONTEXT = 256
BATCH_SIZE = 16
STEPS = 500
WARMUP_STEPS = 100
PEAK_LR = 3e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0
SEED = 42
PARAMETER_FLOPS_PER_TOKEN = 78.7e6
ORACLE_PEAK_FLOPS = 14.75e12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(data_hash: str, tokenizer_hash: str) -> dict:
    return {
        "model": {"vocab_size": 1028, "n_layer": 4, "d_model": 512, "n_head": 8, "ctx": CONTEXT},
        "training": {
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "warmup_steps": WARMUP_STEPS,
            "peak_lr": PEAK_LR,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "clip_norm": CLIP_NORM,
            "seed": SEED,
            "sampling": "shuffled-non-overlapping-context-windows",
        },
        "dtype": "float32",
        "data_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
        "platform": platform.platform(),
    }


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def make_corpus() -> tuple[BPETokenizer, np.ndarray, np.ndarray, dict]:
    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))
    text = DATA_PATH.read_text(encoding="utf-8")
    all_ids = np.asarray(tokenizer.encode(text), dtype=np.int32)
    split = len(all_ids) * 9 // 10
    train_ids = all_ids[:split]
    heldout_ids = all_ids[split:]
    if len(train_ids) <= CONTEXT or len(heldout_ids) <= CONTEXT:
        raise ValueError("tokenized train and heldout partitions must each exceed ctx")
    heldout_text = tokenizer.decode(heldout_ids.tolist())
    heldout_bytes = len(heldout_text.encode("utf-8"))
    details = {
        "total_tokens": int(len(all_ids)),
        "train_tokens": int(len(train_ids)),
        "heldout_tokens": int(len(heldout_ids)),
        "split_token_index": int(split),
        "heldout_utf8_bytes": heldout_bytes,
        "heldout_tokens_per_byte": float(len(heldout_ids) / heldout_bytes),
        "data_sha256": sha256_file(DATA_PATH),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
    }
    return tokenizer, train_ids, heldout_ids, details


def sample_batch(ids: np.ndarray, batch_size: int, context: int, rng: np.random.Generator):
    starts = rng.integers(0, len(ids) - context, size=batch_size)
    offsets = np.arange(context, dtype=np.int64)
    inputs = ids[starts[:, None] + offsets]
    targets = ids[starts[:, None] + offsets + 1]
    return mx.array(inputs), mx.array(targets)


class WindowBatchSampler:
    """Shuffle full-context windows and cycle through each train epoch once."""

    def __init__(self, ids: np.ndarray, batch_size: int, context: int, rng: np.random.Generator):
        self.ids = ids
        self.batch_size = batch_size
        self.context = context
        self.rng = rng
        self.starts = np.arange(0, len(ids) - context, context, dtype=np.int64)
        if len(self.starts) < batch_size:
            raise ValueError("training partition must contain at least one full batch of windows")
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = self.rng.permutation(self.starts)
        self.cursor = 0

    def next_batch(self):
        batch_starts = []
        while len(batch_starts) < self.batch_size:
            available = len(self.order) - self.cursor
            needed = self.batch_size - len(batch_starts)
            take = min(available, needed)
            batch_starts.extend(self.order[self.cursor : self.cursor + take].tolist())
            self.cursor += take
            if self.cursor == len(self.order):
                self._reshuffle()
        offsets = np.arange(self.context, dtype=np.int64)
        starts = np.asarray(batch_starts, dtype=np.int64)
        inputs = self.ids[starts[:, None] + offsets]
        targets = self.ids[starts[:, None] + offsets + 1]
        return mx.array(inputs), mx.array(targets)


def flatten_arrays(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from flatten_arrays(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from flatten_arrays(value)
    else:
        yield tree


class TrainingEngine:
    """One differentiable forward/backward/update path using manual AdamW."""

    def __init__(self, model: GPT, optimizer: AdamW, clip_norm: float = CLIP_NORM):
        self.model = model
        self.optimizer = optimizer
        self.clip_norm = clip_norm

        def loss_fn(current_model, inputs, targets):
            return cross_entropy(current_model(inputs), targets)

        self._loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(self, inputs, targets, learning_rate: float) -> tuple[float, float]:
        loss, grads = self._loss_and_grad(self.model, inputs, targets)
        mx.eval(loss, grads)
        clipped_grads, original_norm = clip_grad_norm(grads, self.clip_norm)
        current_params = self.model.trainable_parameters()
        updated_params = self.optimizer.step(current_params, clipped_grads, learning_rate)
        self.model.update(updated_params)
        mx.eval(self.model.parameters(), self.optimizer.first_moment, self.optimizer.second_moment)
        return float(loss.item()), float(original_norm.item())

    def evaluate(self, inputs, targets) -> float:
        loss = cross_entropy(self.model(inputs), targets)
        mx.eval(loss)
        return float(loss.item())


def new_model(seed: int = 42) -> GPT:
    return GPT(vocab_size=1028, n_layer=4, d_model=512, n_head=8, ctx=CONTEXT, seed=seed)


def run_single_batch_overfit_check() -> dict:
    """Exercise the full model/update path on one fixed, compact batch."""
    model = GPT(vocab_size=16, n_layer=1, d_model=16, n_head=2, ctx=8, seed=17)
    optimizer = AdamW(learning_rate=0.03, betas=BETAS, eps=EPS, weight_decay=0.0)
    engine = TrainingEngine(model, optimizer, clip_norm=1.0)
    pattern = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    inputs = mx.array(np.tile(pattern[:-1], (8, 1)))
    targets = mx.array(np.tile(pattern[1:], (8, 1)))
    initial_loss = engine.evaluate(inputs, targets)
    for _ in range(50):
        engine.step(inputs, targets, learning_rate=0.03)
    final_loss = engine.evaluate(inputs, targets)
    result = {"steps": 50, "initial_loss_nats": initial_loss, "final_loss_nats": final_loss}
    if not math.isfinite(final_loss) or final_loss >= 0.5:
        raise AssertionError(f"single-batch overfit check failed: {result}")
    return result


def estimate_parameter_count(model: GPT) -> int:
    return sum(int(array.size) for array in flatten_arrays(model.trainable_parameters()))


def save_checkpoint(model: GPT, path: Path) -> None:
    arrays = {}

    def collect(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                collect(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                collect(value, f"{prefix}.{index}")
        else:
            arrays[prefix] = np.asarray(node)

    collect(model.trainable_parameters())
    np.savez(path, **arrays)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return float(ordered[index])


def device_info() -> dict:
    def sysctl(key: str) -> str | None:
        try:
            value = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=False
            ).stdout.strip()
            return value or None
        except OSError:
            return None

    memory = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_bytes": int(memory) if memory and memory.isdigit() else None,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
    }


def run_training() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _tokenizer, train_ids, heldout_ids, corpus = make_corpus()
    config = build_config(corpus["data_sha256"], corpus["tokenizer_sha256"])
    run_hash = config_hash(config)
    rng = np.random.default_rng(SEED + 1)
    heldout_inputs, heldout_targets = sample_batch(heldout_ids, BATCH_SIZE, CONTEXT, rng)
    train_sampler = WindowBatchSampler(
        train_ids, BATCH_SIZE, CONTEXT, np.random.default_rng(SEED)
    )

    model = new_model(seed=42)
    mx.eval(model.parameters())
    parameter_count = estimate_parameter_count(model)
    parameter_bytes = parameter_count * 4
    predicted_training_bytes = parameter_count * 16
    optimizer = AdamW(
        learning_rate=PEAK_LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )
    engine = TrainingEngine(model, optimizer)
    initial_heldout_loss = engine.evaluate(heldout_inputs, heldout_targets)
    loss_curve = [{"step": 0, "train_loss_nats": None, "heldout_loss_nats": initial_heldout_loss}]
    step_seconds = []
    last_train_losses = []
    peak_memory_before = int(mx.metal.get_active_memory())
    mx.metal.reset_peak_memory()

    for step in range(1, STEPS + 1):
        train_inputs, train_targets = train_sampler.next_batch()
        learning_rate = cosine_lr(step, STEPS, PEAK_LR, WARMUP_STEPS)
        started = time.perf_counter()
        train_loss, _grad_norm = engine.step(train_inputs, train_targets, learning_rate)
        step_seconds.append(time.perf_counter() - started)
        last_train_losses.append(train_loss)

        if step % 10 == 0:
            heldout_loss = engine.evaluate(heldout_inputs, heldout_targets)
            loss_curve.append(
                {
                    "step": step,
                    "train_loss_nats": float(np.mean(last_train_losses[-10:])),
                    "heldout_loss_nats": heldout_loss,
                }
            )

    final_heldout_loss = float(loss_curve[-1]["heldout_loss_nats"])
    peak_memory_bytes = int(mx.metal.get_peak_memory())
    tokens_per_step = BATCH_SIZE * CONTEXT
    timed_rates = [tokens_per_step / seconds for seconds in step_seconds[3:]]
    median_tokens_per_second = percentile(timed_rates, 0.5)
    p90_tokens_per_second = percentile(timed_rates, 0.9)
    mfu = median_tokens_per_second * PARAMETER_FLOPS_PER_TOKEN / ORACLE_PEAK_FLOPS
    tokens_per_byte = corpus["heldout_tokens_per_byte"]
    model_bpb = final_heldout_loss / math.log(2.0) * tokens_per_byte
    overfit = run_single_batch_overfit_check()

    checkpoint_path = RESULTS_DIR / f"ckpt-final-{STEPS}.npz"
    save_checkpoint(model, checkpoint_path)
    curve_path = RESULTS_DIR / f"loss-curve-{STEPS}.json"
    curve_path.write_text(json.dumps(loss_curve, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "iteration": "L3-T2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": run_hash,
        "config": config,
        "device": device_info(),
        "corpus": corpus,
        "model": {
            **config["model"],
            "parameter_count": parameter_count,
            "parameter_bytes_fp32": parameter_bytes,
            "predicted_4x_parameter_bytes": predicted_training_bytes,
        },
        "training": {
            "steps": STEPS,
            "tokens_per_step": tokens_per_step,
            "warmup_excluded_steps": 3,
            "timed_steps": len(timed_rates),
            "step_tokens_per_second_median": median_tokens_per_second,
            "step_tokens_per_second_p90": p90_tokens_per_second,
            "step_seconds_median": percentile(step_seconds[3:], 0.5),
            "step_seconds_p90": percentile(step_seconds[3:], 0.9),
            "final_train_loss_nats_10_step_mean": loss_curve[-1]["train_loss_nats"],
            "initial_heldout_loss_nats": initial_heldout_loss,
            "final_heldout_loss_nats": final_heldout_loss,
            "heldout_loss_relative_reduction": 1.0 - final_heldout_loss / initial_heldout_loss,
            "heldout_model_bits_per_byte": model_bpb,
            "heldout_tokens_per_byte": tokens_per_byte,
            "mfu_fraction": mfu,
            "mfu_percent": mfu * 100.0,
            "peak_metal_memory_bytes": peak_memory_bytes,
            "peak_metal_memory_mib": peak_memory_bytes / (1024**2),
            "active_memory_before_training_bytes": peak_memory_before,
            "adamw_oracle": "five-step alignment is checked by tests/test_optimizer.py",
            "single_batch_overfit_check": overfit,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "loss_curve": str(curve_path.relative_to(ROOT)),
        },
        "l1_uniform_id_proxy_bits_per_byte": {
            "layers_tokenizer": 4.19,
            "cl100k_base": 4.69,
        },
    }
    metrics_path = RESULTS_DIR / f"train-metrics-{STEPS}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if not all(
        math.isfinite(float(row["heldout_loss_nats"]))
        and (row["train_loss_nats"] is None or math.isfinite(float(row["train_loss_nats"])))
        for row in loss_curve
    ):
        raise AssertionError("loss curve contains NaN or infinity")
    if final_heldout_loss >= initial_heldout_loss * 0.70:
        raise AssertionError(
            "heldout loss reduction must exceed 30%: "
            f"initial={initial_heldout_loss:.4f}, final={final_heldout_loss:.4f}"
        )
    print(f"config_hash: {run_hash}")
    print(f"device: {metrics['device']['chip']} / {metrics['device']['memory_bytes']} bytes")
    print(f"corpus: {corpus['total_tokens']} tokens; split={corpus['split_token_index']}")
    print(
        f"model: {parameter_count:,} parameters; predicted 4x-state memory="
        f"{predicted_training_bytes / (1024**2):.1f} MiB"
    )
    print(
        f"heldout loss: {initial_heldout_loss:.4f} -> {final_heldout_loss:.4f} nats "
        f"({metrics['training']['heldout_loss_relative_reduction']:.1%} reduction)"
    )
    print(
        f"throughput: median={median_tokens_per_second:.1f}, "
        f"p90={p90_tokens_per_second:.1f} tokens/s; MFU={mfu * 100.0:.4f}%"
    )
    print(
        f"memory: peak={peak_memory_bytes / (1024**2):.1f} MiB; "
        f"model bpb={model_bpb:.4f}"
    )
    print(
        f"single-batch overfit: {overfit['initial_loss_nats']:.4f} -> "
        f"{overfit['final_loss_nats']:.4f} nats in {overfit['steps']} steps"
    )
    print(f"written: {checkpoint_path}, {curve_path}, {metrics_path}")
    return metrics


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
