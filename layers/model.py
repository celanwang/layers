"""A small GPT assembled from the project's learned components."""

import math

import mlx.core as mx
import mlx.nn as nn

from .embedding import TokenEmbedding
from .transformer import LayerNorm, TransformerBlock


def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    """Mean token cross-entropy computed with a stable log-sum-exp."""
    if logits.ndim < 2 or targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits in every dimension except vocabulary")
    if logits.shape[-1] <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")

    targets = mx.array(targets).astype(mx.int32)
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    log_partition = mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True))
    log_partition = log_partition + max_logits
    target_logits = mx.take_along_axis(logits, targets[..., None], axis=-1)
    nll = log_partition - target_logits
    return mx.mean(nll)


class GPT(nn.Module):
    """Pre-LN GPT with learned token embeddings and tied output weights."""

    def __init__(
        self,
        vocab_size: int = 1028,
        n_layer: int = 4,
        d_model: int = 512,
        n_head: int = 8,
        ctx: int = 256,
        seed: int = 42,
        eps: float = 1e-5,
    ):
        super().__init__()
        if vocab_size <= 0 or n_layer <= 0 or d_model <= 0 or n_head <= 0 or ctx <= 0:
            raise ValueError("vocab_size, n_layer, d_model, n_head, and ctx must be positive")
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.ctx = ctx
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, seed=seed)
        self.blocks = [
            TransformerBlock(d_model=d_model, n_head=n_head, seed=seed, eps=eps)
            for _ in range(n_layer)
        ]
        self.final_norm = LayerNorm(d_model, eps=eps)

        # L2 modules seed themselves on construction. Reinitialize this assembled
        # model from one seeded stream so different layers do not share copies.
        mx.random.seed(seed)
        self.token_embedding.weight = mx.random.normal((vocab_size, d_model)) * 0.02
        scale = 1.0 / math.sqrt(d_model)
        for block in self.blocks:
            for name in ("wq", "wk", "wv", "wo"):
                setattr(
                    block.attn,
                    name,
                    mx.random.normal((d_model, d_model)) * scale,
                )
            block.mlp.w1 = mx.random.normal((d_model, 4 * d_model)) * scale
            block.mlp.w2 = mx.random.normal((4 * d_model, d_model)) * scale

    def __call__(self, ids: mx.array, cache=None) -> mx.array:
        if ids.ndim not in (1, 2):
            raise ValueError("ids must have shape [sequence] or [batch, sequence]")
        if ids.shape[-1] > self.ctx:
            raise ValueError("sequence length exceeds ctx")
        x = self.token_embedding(ids)
        if cache is None:
            for block in self.blocks:
                x = block(x)
        else:
            for layer_idx, block in enumerate(self.blocks):
                x = block(x, cache=cache, layer_idx=layer_idx)
        x = self.final_norm(x)
        return x @ mx.transpose(self.token_embedding.weight)

    def decode_batch(self, ids: mx.array, cache) -> mx.array:
        """Return ``[batch, 1, vocab]`` logits using a paged batch cache."""
        if ids.ndim != 2 or ids.shape != (len(cache.seq_ids), 1) or not ids.shape[0]:
            raise ValueError("ids must have shape [non-empty cache batch, 1]")
        if any(length >= self.ctx for length in cache.lengths):
            raise ValueError("batched decode would exceed ctx")
        x = self.token_embedding(ids)
        for layer_idx, block in enumerate(self.blocks):
            x = block.decode_batch(x, cache, layer_idx)
        logits = self.final_norm(x[:, 0, :]) @ mx.transpose(self.token_embedding.weight)
        return logits[:, None, :]
