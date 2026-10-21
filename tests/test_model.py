"""GPT assembly checks against hand calculations and a PyTorch Oracle."""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import pytest
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f

from layers import BPETokenizer, GPT, LayerNorm, cross_entropy


ROOT = Path(__file__).resolve().parents[1]


def as_numpy(value):
    return np.asarray(value, dtype=np.float32)


def mlx_named_parameters(model):
    """Flatten the module tree using the same dotted paths as Torch."""
    tree = model.trainable_parameters()
    flattened = {}

    def visit(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                visit(value, f"{prefix}.{index}")
        else:
            flattened[prefix] = node

    visit(tree)
    return flattened


def all_model_arrays(model):
    """List each stored parameter once; the tied output has no second array."""
    arrays = [model.token_embedding.weight]
    for block in model.blocks:
        arrays.extend((block.ln_attn.gamma, block.ln_attn.beta))
        arrays.extend((block.attn.wq, block.attn.wk, block.attn.wv, block.attn.wo))
        arrays.extend((block.ln_mlp.gamma, block.ln_mlp.beta))
        arrays.extend((block.mlp.w1, block.mlp.w2))
    arrays.extend((model.final_norm.gamma, model.final_norm.beta))
    return arrays


class TorchLayerNorm(torch_nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = torch_nn.Parameter(torch.ones(d_model))
        self.beta = torch_nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return torch_f.layer_norm(x, (x.shape[-1],), self.gamma, self.beta, eps=1e-5)


class TorchMLP(torch_nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = torch_nn.Parameter(torch.empty(d_model, 4 * d_model))
        self.w2 = torch_nn.Parameter(torch.empty(4 * d_model, d_model))

    def forward(self, x):
        return torch_f.gelu(x @ self.w1, approximate="none") @ self.w2


class TorchAttention(torch_nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        for name in ("wq", "wk", "wv", "wo"):
            setattr(self, name, torch_nn.Parameter(torch.empty(d_model, d_model)))

    def forward(self, x):
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        batch, length, _ = x.shape
        q, k, v = [
            (x @ weight).reshape(batch, length, self.n_head, self.d_head).transpose(1, 2)
            for weight in (self.wq, self.wk, self.wv)
        ]
        positions = torch.arange(length, dtype=x.dtype, device=x.device)
        pair_indices = torch.arange(0, self.d_head, 2, dtype=x.dtype, device=x.device)
        inv_freq = torch.exp(-pair_indices * (math.log(10000.0) / self.d_head))
        angles = (positions[:, None] * inv_freq[None, :]).reshape(
            1, 1, length, self.d_head // 2
        )

        def rotate(value):
            even, odd = value[..., 0::2], value[..., 1::2]
            return torch.stack(
                (even * torch.cos(angles) - odd * torch.sin(angles),
                 even * torch.sin(angles) + odd * torch.cos(angles)),
                dim=-1,
            ).reshape(value.shape)

        q, k = rotate(q), rotate(k)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        future = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        weights = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)
        context = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, self.d_model)
        out = context @ self.wo
        return out[0] if unbatched else out


class TorchBlock(torch_nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln_attn = TorchLayerNorm(d_model)
        self.attn = TorchAttention(d_model, n_head)
        self.ln_mlp = TorchLayerNorm(d_model)
        self.mlp = TorchMLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_attn(x))
        return x + self.mlp(self.ln_mlp(x))


class TorchGPT(torch_nn.Module):
    def __init__(self, vocab_size, n_layer, d_model, n_head):
        super().__init__()
        self.token_embedding = torch_nn.Embedding(vocab_size, d_model)
        self.blocks = torch_nn.ModuleList(
            [TorchBlock(d_model, n_head) for _ in range(n_layer)]
        )
        self.final_norm = TorchLayerNorm(d_model)

    def forward(self, ids):
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x @ self.token_embedding.weight.T


def torch_named_parameters(model):
    """Keep the Torch Oracle's names aligned with the MLX module tree."""
    params = {"token_embedding.weight": model.token_embedding.weight}
    for index, block in enumerate(model.blocks):
        prefix = f"blocks.{index}"
        for module_name in ("ln_attn", "attn", "ln_mlp", "mlp"):
            module = getattr(block, module_name)
            for name, parameter in module.named_parameters(recurse=False):
                params[f"{prefix}.{module_name}.{name}"] = parameter
    for name, parameter in model.final_norm.named_parameters(recurse=False):
        params[f"final_norm.{name}"] = parameter
    return params


def test_hand_computed_micro_gpt_layernorm_residual_tying_and_cross_entropy():
    norm = LayerNorm(d_model=2, eps=1e-5)
    norm.gamma = mx.array([1.5, 0.5])
    norm.beta = mx.array([-0.25, 0.75])
    input_values = np.array([[1.0, 3.0], [-2.0, 2.0]], dtype=np.float32)
    actual_norm = as_numpy(norm(mx.array(input_values)))
    means = input_values.mean(axis=-1, keepdims=True)
    centered = input_values - means
    expected_norm = centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)
    expected_norm = expected_norm * np.array([1.5, 0.5]) + np.array([-0.25, 0.75])
    np.testing.assert_allclose(actual_norm, expected_norm, rtol=0, atol=1e-6)

    model = GPT(vocab_size=5, n_layer=1, d_model=2, n_head=1, ctx=3)
    model.token_embedding.weight = mx.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.25, -0.5]],
        dtype=mx.float32,
    )
    block = model.blocks[0]
    for name in ("wq", "wk", "wv", "wo"):
        setattr(block.attn, name, mx.zeros((2, 2)))
    block.mlp.w1 = mx.zeros((2, 8))
    block.mlp.w2 = mx.zeros((8, 2))

    ids = np.array([0, 1], dtype=np.int32)
    targets = np.array([0, 1], dtype=np.int32)
    logits = as_numpy(model(mx.array(ids)))
    embedded = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    centered = embedded - embedded.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)
    expected_logits = normalized @ as_numpy(model.token_embedding.weight).T
    np.testing.assert_allclose(logits, expected_logits, rtol=0, atol=1e-6)

    shifted = expected_logits - expected_logits.max(axis=-1, keepdims=True)
    expected_loss = np.mean(
        np.log(np.exp(shifted).sum(axis=-1)) - shifted[np.arange(2), targets]
    )
    actual_loss = float(cross_entropy(mx.array(logits), mx.array(targets)))
    assert actual_loss == pytest.approx(expected_loss, rel=0, abs=1e-6)


def test_pytorch_oracle_matches_logits_and_every_named_parameter_gradient():
    vocab_size, n_layer, d_model, n_head = 17, 2, 8, 2
    model = GPT(
        vocab_size=vocab_size,
        n_layer=n_layer,
        d_model=d_model,
        n_head=n_head,
        ctx=8,
        seed=42,
    )
    torch_model = TorchGPT(vocab_size, n_layer, d_model, n_head)
    mlx_params = mlx_named_parameters(model)
    torch_params = torch_named_parameters(torch_model)
    assert mlx_params.keys() == torch_params.keys()
    with torch.no_grad():
        for name, parameter in torch_params.items():
            parameter.copy_(torch.from_numpy(as_numpy(mlx_params[name]).copy()))

    ids_np = np.array([[1, 4, 2, 9], [3, 2, 5, 7]], dtype=np.int32)
    targets_np = np.array([[4, 2, 9, 1], [2, 5, 7, 3]], dtype=np.int32)

    def mlx_loss_fn(current_model, input_ids, target_ids):
        return cross_entropy(current_model(input_ids), target_ids)

    loss, grads = mlx_nn.value_and_grad(model, mlx_loss_fn)(
        model, mx.array(ids_np), mx.array(targets_np)
    )
    mlx_logits = model(mx.array(ids_np))
    torch_logits = torch_model(torch.from_numpy(ids_np.astype(np.int64)))
    torch_loss = torch_f.cross_entropy(
        torch_logits.reshape(-1, vocab_size),
        torch.from_numpy(targets_np.astype(np.int64)).reshape(-1),
    )
    torch_loss.backward()
    mx.eval(loss, grads, mlx_logits)

    np.testing.assert_allclose(
        as_numpy(mlx_logits), torch_logits.detach().numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(float(loss), float(torch_loss.detach()), rtol=1e-4, atol=1e-4)

    mlx_grads = {}

    def visit(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                visit(value, f"{prefix}.{index}")
        else:
            mlx_grads[prefix] = node

    visit(grads)
    assert mlx_grads.keys() == torch_params.keys()
    for name, torch_parameter in torch_params.items():
        np.testing.assert_allclose(
            as_numpy(mlx_grads[name]),
            torch_parameter.grad.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"gradient mismatch for {name}",
        )


def test_model_level_causality_is_preserved_across_blocks():
    model = GPT(vocab_size=23, n_layer=2, d_model=8, n_head=2, ctx=8, seed=42)
    ids = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)
    changed = mx.array([1, 2, 3, 4, 19, 6], dtype=mx.int32)
    original_logits = model(ids)
    changed_logits = model(changed)
    mx.eval(original_logits, changed_logits)
    np.testing.assert_array_equal(as_numpy(original_logits[:4]), as_numpy(changed_logits[:4]))


def test_default_model_parameter_count_matches_tied_theory():
    model = GPT()
    measured = sum(int(np.prod(array.shape)) for array in all_model_arrays(model))
    expected = (
        model.vocab_size * model.d_model
        + model.n_layer * (12 * model.d_model**2 + 4 * model.d_model)
        + 2 * model.d_model
    )
    assert model.vocab_size == BPETokenizer.load(
        str(ROOT / "results/l1/tokenizer-1024.json")
    ).vocab_size + 4
    assert measured == expected == 13_118_464


def test_real_l1_tokenizer_text_flows_through_gpt():
    tokenizer = BPETokenizer.load(str(ROOT / "results/l1/tokenizer-1024.json"))
    token_ids = tokenizer.encode("A small model learns to predict the next token.")
    model = GPT(
        vocab_size=tokenizer.vocab_size + 4,
        n_layer=1,
        d_model=16,
        n_head=4,
        ctx=64,
    )
    logits = model(mx.array(token_ids, dtype=mx.int32))
    mx.eval(logits)
    assert logits.shape == (len(token_ids), tokenizer.vocab_size + 4)
    assert np.isfinite(as_numpy(logits)).all()


def test_fixed_batch_gradient_path_can_overfit_with_inline_sgd():
    model = GPT(vocab_size=9, n_layer=1, d_model=8, n_head=2, ctx=4, seed=42)
    ids = mx.array([0, 1, 2, 0], dtype=mx.int32)
    targets = mx.array([1, 2, 0, 1], dtype=mx.int32)

    def loss_fn(current_model, input_ids, target_ids):
        return cross_entropy(current_model(input_ids), target_ids)

    loss_and_grad = mlx_nn.value_and_grad(model, loss_fn)
    initial_loss = float(cross_entropy(model(ids), targets))

    def tree_map(function, left, right):
        if isinstance(left, dict):
            return {key: tree_map(function, left[key], right[key]) for key in left}
        if isinstance(left, (list, tuple)):
            return [tree_map(function, a, b) for a, b in zip(left, right)]
        return function(left, right)

    for _ in range(50):
        loss, grads = loss_and_grad(model, ids, targets)
        updates = tree_map(
            lambda parameter, grad: parameter - 0.01 * grad,
            model.trainable_parameters(),
            grads,
        )
        model.update(updates)
        mx.eval(loss, model.parameters())
    final_loss = float(cross_entropy(model(ids), targets))
    assert final_loss < initial_loss * 0.8, (initial_loss, final_loss)
