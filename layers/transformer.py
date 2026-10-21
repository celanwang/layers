"""Pre-LN transformer blocks assembled from the L2 attention component."""

import math

import mlx.core as mx
import mlx.nn as nn

from .attention import CausalSelfAttention


class LayerNorm(nn.Module):
    """Layer normalization over the final feature dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.gamma = mx.ones((d_model,))
        self.beta = mx.zeros((d_model,))
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.gamma.shape[0]:
            raise ValueError("the last input dimension must equal d_model")
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        variance = mx.mean(centered * centered, axis=-1, keepdims=True)
        normalized = centered * mx.rsqrt(variance + self.eps)
        return normalized * self.gamma + self.beta


class MLP(nn.Module):
    """Two bias-free projections with a 4x hidden width and exact GELU."""

    def __init__(self, d_model: int, seed: int = 42):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        mx.random.seed(seed)
        scale = 1.0 / math.sqrt(d_model)
        self.w1 = mx.random.normal((d_model, 4 * d_model)) * scale
        self.w2 = mx.random.normal((4 * d_model, d_model)) * scale

    def __call__(self, x: mx.array) -> mx.array:
        hidden = x @ self.w1
        gelu = 0.5 * hidden * (1.0 + mx.erf(hidden / math.sqrt(2.0)))
        return gelu @ self.w2


class TransformerBlock(nn.Module):
    """A pre-LN attention and MLP block with two residual connections."""

    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        seed: int = 42,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.ln_attn = LayerNorm(d_model, eps=eps)
        self.attn = CausalSelfAttention(d_model=d_model, n_head=n_head, seed=seed)
        self.ln_mlp = LayerNorm(d_model, eps=eps)
        self.mlp = MLP(d_model=d_model, seed=seed)

    def __call__(
        self, x: mx.array, cache=None, layer_idx: int | None = None
    ) -> mx.array:
        if cache is None:
            attn_out, _ = self.attn(self.ln_attn(x))
        else:
            attn_out, _ = self.attn(
                self.ln_attn(x), cache=cache, layer_idx=layer_idx
            )
        x = x + attn_out
        return x + self.mlp(self.ln_mlp(x))
