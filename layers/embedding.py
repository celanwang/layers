"""A small learned token lookup table implemented with MLX core arrays."""

import mlx.core as mx
import mlx.nn as nn


class TokenEmbedding(nn.Module):
    """Map integer token IDs to learned ``d_model``-dimensional vectors."""

    def __init__(self, vocab_size: int = 1028, d_model: int = 512, seed: int = 42):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        mx.random.seed(seed)
        self.weight = mx.random.normal((vocab_size, d_model)) * 0.02

    def __call__(self, ids: mx.array) -> mx.array:
        """Gather embeddings for an ID array of any shape."""
        return mx.take(self.weight, mx.array(ids), axis=0)
