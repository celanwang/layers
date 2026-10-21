"""Small, explicit training utilities for the L3 training-loop exercise."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mlx.core as mx


def _flatten(tree: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], mx.array]:
    if isinstance(tree, Mapping):
        flattened = {}
        for key, value in tree.items():
            flattened.update(_flatten(value, path + (key,)))
        return flattened
    if isinstance(tree, (list, tuple)):
        flattened = {}
        for index, value in enumerate(tree):
            flattened.update(_flatten(value, path + (index,)))
        return flattened
    if not path:
        return {(): mx.array(tree)}
    return {path: mx.array(tree)}


def _rebuild(template: Any, leaves: Mapping[tuple[Any, ...], mx.array], path=()):
    if isinstance(template, Mapping):
        return type(template)(
            (key, _rebuild(value, leaves, path + (key,)))
            for key, value in template.items()
        )
    if isinstance(template, list):
        return [_rebuild(value, leaves, path + (index,)) for index, value in enumerate(template)]
    if isinstance(template, tuple):
        values = [_rebuild(value, leaves, path + (index,)) for index, value in enumerate(template)]
        return type(template)(*values) if hasattr(template, "_fields") else tuple(values)
    return leaves[path]


def global_grad_norm(grads: Any) -> mx.array:
    """Return the L2 norm over every gradient leaf in a parameter tree."""
    leaves = list(_flatten(grads).values())
    if not leaves:
        raise ValueError("grads must contain at least one array")
    squared = [mx.sum(mx.square(gradient.astype(mx.float32))) for gradient in leaves]
    return mx.sqrt(sum(squared[1:], squared[0]))


def clip_grad_norm(grads: Any, max_norm: float) -> tuple[Any, mx.array]:
    """Scale a gradient tree so its global L2 norm is at most ``max_norm``.

    Returns the clipped tree and the pre-clipping norm. The scale is computed
    once across all leaves, matching global-norm clipping rather than per-array
    clipping.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be a finite positive number")
    leaves = _flatten(grads)
    if not leaves:
        raise ValueError("grads must contain at least one array")
    norm = global_grad_norm(grads)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-12))
    clipped = {path: gradient * scale for path, gradient in leaves.items()}
    return _rebuild(grads, clipped), norm


class AdamW:
    """Adam with bias correction and decoupled weight decay, in plain MLX ops."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if learning_rate < 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("learning_rate and weight_decay must be non-negative; eps positive")
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.first_moment: dict[tuple[Any, ...], mx.array] = {}
        self.second_moment: dict[tuple[Any, ...], mx.array] = {}

    def step(self, params: Any, grads: Any, learning_rate: float | None = None) -> Any:
        """Return an updated parameter tree; ``params`` and ``grads`` must align."""
        parameter_leaves = _flatten(params)
        gradient_leaves = _flatten(grads)
        if parameter_leaves.keys() != gradient_leaves.keys():
            raise ValueError("parameter and gradient trees must have identical structure")
        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("learning_rate must be finite and non-negative")

        self.step_count += 1
        time_step = self.step_count
        updated = {}
        for path, parameter in parameter_leaves.items():
            gradient = gradient_leaves[path]
            if parameter.shape != gradient.shape:
                raise ValueError(f"gradient shape mismatch at {path}: {gradient.shape} != {parameter.shape}")
            if path not in self.first_moment:
                self.first_moment[path] = mx.zeros_like(parameter)
                self.second_moment[path] = mx.zeros_like(parameter)
            first = self.beta1 * self.first_moment[path] + (1.0 - self.beta1) * gradient
            second = self.beta2 * self.second_moment[path] + (1.0 - self.beta2) * mx.square(gradient)
            self.first_moment[path] = first
            self.second_moment[path] = second
            first_hat = first / (1.0 - self.beta1**time_step)
            second_hat = second / (1.0 - self.beta2**time_step)
            adaptive_update = first_hat / (mx.sqrt(second_hat) + self.eps)
            # The decay term is applied directly to parameters and never enters v.
            updated[path] = parameter * (1.0 - lr * self.weight_decay) - lr * adaptive_update
        return _rebuild(params, updated)


def cosine_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int = 100) -> float:
    """Linear warmup followed by cosine decay to 10% of peak.

    ``step`` is one-based: step 1 is the first update and ``total_steps`` is
    the final update. Thus warmup ends exactly at ``peak_lr`` and the final
    step is exactly 10% of peak (when there is a decay phase).
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not math.isfinite(peak_lr) or peak_lr < 0:
        raise ValueError("peak_lr must be finite and non-negative")
    if not 1 <= step <= total_steps:
        raise ValueError("step must be within [1, total_steps]")
    if warmup_steps and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return peak_lr
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    return peak_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
