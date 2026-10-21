"""Hand checks and a PyTorch Oracle for the manually implemented L3 optimizer."""

import math

import mlx.core as mx
import numpy as np
import pytest
import torch

from layers.optimizer import AdamW, clip_grad_norm, cosine_lr, global_grad_norm
from scripts.train import run_single_batch_overfit_check


def test_adamw_three_step_scalar_matches_hand_calculation_with_bias_correction():
    optimizer = AdamW(
        learning_rate=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    parameter = mx.array(1.0, dtype=mx.float32)
    gradient = mx.array(2.0, dtype=mx.float32)
    expected = 1.0
    first, second = 0.0, 0.0
    for step in range(1, 4):
        first = 0.9 * first + 0.1 * 2.0
        second = 0.95 * second + 0.05 * (2.0**2)
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.95**step)
        expected = expected * (1.0 - 0.1 * 0.1) - 0.1 * (
            first_hat / (math.sqrt(second_hat) + 1e-8)
        )
        parameter = optimizer.step({"p": parameter}, {"p": gradient}, learning_rate=0.1)["p"]
        mx.eval(parameter)
        assert float(parameter.item()) == pytest.approx(expected, abs=2e-7)


def test_adamw_matches_torch_adamw_for_first_five_updates():
    initial = {
        "a": np.asarray([0.25, -0.5, 1.25], dtype=np.float32),
        "nested": {"b": np.asarray([[2.0, -1.0], [0.125, 0.75]], dtype=np.float32)},
    }
    ours = {"a": mx.array(initial["a"]), "nested": {"b": mx.array(initial["nested"]["b"])}}
    torch_params = [
        torch.nn.Parameter(torch.from_numpy(initial["a"].copy())),
        torch.nn.Parameter(torch.from_numpy(initial["nested"]["b"].copy())),
    ]
    torch_optimizer = torch.optim.AdamW(
        torch_params,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    optimizer = AdamW(learning_rate=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    gradients = [
        np.asarray([0.5, -0.25, 1.0], dtype=np.float32),
        np.asarray([[0.125, -0.75], [1.25, -0.5]], dtype=np.float32),
    ]
    for _ in range(5):
        torch_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(torch_params, gradients):
            parameter.grad = torch.from_numpy(gradient.copy())
        torch_optimizer.step()
        ours = optimizer.step(
            ours,
            {"a": mx.array(gradients[0]), "nested": {"b": mx.array(gradients[1])}},
        )
        mx.eval(ours["a"], ours["nested"]["b"])
        np.testing.assert_allclose(np.asarray(ours["a"]), torch_params[0].detach().numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(ours["nested"]["b"]), torch_params[1].detach().numpy(), rtol=0, atol=1e-6
        )


def test_cosine_lr_warmup_peak_and_terminal_boundaries():
    peak, total, warmup = 3e-4, 200, 100
    assert cosine_lr(1, total, peak, warmup) == pytest.approx(peak / warmup)
    assert cosine_lr(warmup, total, peak, warmup) == pytest.approx(peak)
    assert cosine_lr(warmup + 1, total, peak, warmup) < peak
    assert cosine_lr(total, total, peak, warmup) == pytest.approx(0.1 * peak)


def test_clip_grad_norm_scales_all_leaves_to_unit_global_norm():
    grads = {"first": mx.array([1.2, 1.6]), "nested": [mx.array([0.0])]}
    clipped, original_norm = clip_grad_norm(grads, max_norm=1.0)
    mx.eval(original_norm, clipped)
    assert float(original_norm.item()) == pytest.approx(2.0, abs=1e-6)
    assert float(global_grad_norm(clipped).item()) == pytest.approx(1.0, abs=1e-6)


def test_full_optimizer_path_can_overfit_one_fixed_batch_in_fifty_steps():
    result = run_single_batch_overfit_check()
    assert result["steps"] == 50
    assert result["final_loss_nats"] < 0.5
