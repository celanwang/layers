"""Inspectable causal self-attention using MLX core operations only."""

import math

import mlx.core as mx
import mlx.nn as nn


class RotaryEmbedding(nn.Module):
    """Apply interleaved-pair rotary position encoding to ``[..., n, d]``."""

    def __init__(self, dims: int, base: float = 10000.0):
        super().__init__()
        if dims <= 0 or dims % 2:
            raise ValueError("dims must be a positive even integer")
        if base <= 0:
            raise ValueError("base must be positive")
        self.dims = dims
        self.base = float(base)

    def __call__(self, x: mx.array, offset: float = 0) -> mx.array:
        if x.ndim < 2 or x.shape[-1] < self.dims:
            raise ValueError("input must have a sequence axis and at least dims features")
        sequence_length = x.shape[-2]
        positions = mx.arange(sequence_length, dtype=mx.float32) + offset
        pair_indices = mx.arange(0, self.dims, 2, dtype=mx.float32)
        inverse_frequencies = mx.exp(-pair_indices * (math.log(self.base) / self.dims))
        angles = positions[:, None] * inverse_frequencies[None, :]
        broadcast_shape = (1,) * (x.ndim - 2) + (sequence_length, self.dims // 2)
        cos = mx.cos(angles).reshape(broadcast_shape).astype(x.dtype)
        sin = mx.sin(angles).reshape(broadcast_shape).astype(x.dtype)

        even = x[..., : self.dims : 2]
        odd = x[..., 1 : self.dims : 2]
        rotated = mx.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1)
        rotated = rotated.reshape(x.shape[:-1] + (self.dims,))
        if self.dims == x.shape[-1]:
            return rotated
        return mx.concatenate((rotated, x[..., self.dims :]), axis=-1)


def attention_scores(q: mx.array, k: mx.array) -> mx.array:
    """Return scaled, causally masked QKᵀ scores for ``[..., heads, n, d]``."""
    if q.ndim < 3 or k.ndim != q.ndim:
        raise ValueError("q and k must have matching [..., heads, sequence, features] ranks")
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k batch/head dimensions and feature sizes must match")
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("causal self-attention requires equal query and key lengths")
    length = q.shape[-2]
    scores = (q @ mx.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    positions = mx.arange(length)
    future = positions[None, :] > positions[:, None]
    return mx.where(future, -float("inf"), scores)


def _stable_softmax(scores: mx.array) -> mx.array:
    """Compute a row-wise softmax after subtracting each row maximum."""
    shifted = scores - mx.max(scores, axis=-1, keepdims=True)
    exponentials = mx.exp(shifted)
    return exponentials / mx.sum(exponentials, axis=-1, keepdims=True)


@mx.compile
def _decode_rope(q, k, positions, frequencies):
    """Fuse only pure batch-path arithmetic; page mutation stays in Python."""
    angles = positions[:, None, None, None] * frequencies
    cos, sin = mx.cos(angles).astype(q.dtype), mx.sin(angles).astype(q.dtype)

    def rotate(tensor):
        even, odd = tensor[..., ::2], tensor[..., 1::2]
        return mx.stack(
            (even * cos - odd * sin, even * sin + odd * cos), axis=-1
        ).reshape(tensor.shape)

    return rotate(q), rotate(k)


@mx.compile
def _decode_attention(q, k, v, positions):
    """Explicit masked attention, fused without any mx.fast operation."""
    scores = (q @ mx.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    valid = mx.arange(k.shape[-2])[None, :] <= positions[:, None]
    scores = mx.where(valid[:, None, None, :], scores, -float("inf"))
    probs = _stable_softmax(scores)
    return probs @ v, probs, scores, valid


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with inspectable intermediate arrays."""

    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        d_head: int | None = None,
        rope_base: float = 10000.0,
        seed: int = 42,
    ):
        super().__init__()
        if d_model <= 0 or n_head <= 0:
            raise ValueError("d_model and n_head must be positive")
        if d_head is None:
            if d_model % n_head:
                raise ValueError("d_model must be divisible by n_head when d_head is omitted")
            d_head = d_model // n_head
        if d_head <= 0 or d_model != n_head * d_head:
            raise ValueError("d_model must equal n_head * d_head")
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.rope_base = float(rope_base)

        mx.random.seed(seed)
        scale = 1.0 / math.sqrt(d_model)
        self.wq = mx.random.normal((d_model, d_model)) * scale
        self.wk = mx.random.normal((d_model, d_model)) * scale
        self.wv = mx.random.normal((d_model, d_model)) * scale
        self.wo = mx.random.normal((d_model, d_model)) * scale
        self.rope = RotaryEmbedding(d_head, base=rope_base)

    def forward_with_intermediates(
        self, x: mx.array, cache=None, layer_idx: int | None = None
    ):
        """Return output, probabilities, and arrays used to form each result."""
        if x.ndim not in (2, 3) or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [n, d_model] or [batch, n, d_model]")
        if cache is not None and layer_idx is None:
            raise ValueError("layer_idx is required when a KV cache is supplied")
        unbatched = x.ndim == 2
        batched = mx.expand_dims(x, axis=0) if unbatched else x
        batch_size, sequence_length, _ = batched.shape

        q = (batched @ self.wq).reshape(
            (batch_size, sequence_length, self.n_head, self.d_head)
        )
        k = (batched @ self.wk).reshape(
            (batch_size, sequence_length, self.n_head, self.d_head)
        )
        v = (batched @ self.wv).reshape(
            (batch_size, sequence_length, self.n_head, self.d_head)
        )
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        if cache is None:
            # Keep the original no-cache path and operation order unchanged.
            rotary_q = self.rope(q)
            rotary_k = self.rope(k)
            scaled_scores = attention_scores(rotary_q, rotary_k)
            attention_values = v
        else:
            cached_k, cached_v = cache.get(layer_idx)
            offset = 0 if cached_k is None else cached_k.shape[-2]
            rotary_q = self.rope(q, offset=offset)
            rotary_k = self.rope(k, offset=offset)
            all_k, attention_values = cache.append(layer_idx, rotary_k, v)
            scaled_scores = (
                rotary_q @ mx.swapaxes(all_k, -1, -2)
            ) * (1.0 / math.sqrt(self.d_head))
            query_positions = mx.arange(sequence_length) + offset
            key_positions = mx.arange(all_k.shape[-2])
            future = key_positions[None, :] > query_positions[:, None]
            scaled_scores = mx.where(future, -float("inf"), scaled_scores)
        probs = _stable_softmax(scaled_scores)
        context = probs @ attention_values
        context = mx.transpose(context, (0, 2, 1, 3)).reshape(
            (batch_size, sequence_length, self.d_model)
        )
        out = context @ self.wo

        if unbatched:
            out = out[0]
            probs = probs[0]
            intermediates = {
                "q": q[0],
                "k": k[0],
                "v": v[0],
                "rotary_q": rotary_q[0],
                "rotary_k": rotary_k[0],
                "scaled_scores": scaled_scores[0],
                "probs": probs,
                "out": out,
            }
        else:
            intermediates = {
                "q": q,
                "k": k,
                "v": v,
                "rotary_q": rotary_q,
                "rotary_k": rotary_k,
                "scaled_scores": scaled_scores,
                "probs": probs,
                "out": out,
            }
        return out, probs, intermediates

    def __call__(self, x: mx.array, cache=None, layer_idx: int | None = None):
        """Return ``(output, attention_weights)`` for a rank-2 or rank-3 input."""
        out, weights, _ = self.forward_with_intermediates(x, cache, layer_idx)
        return out, weights

    def decode_batch_with_intermediates(self, x: mx.array, cache, layer_idx: int):
        """Decode one token per row with independent RoPE positions and KV masks.

        ``cache`` appends each row to its own paged sequence, then gathers and
        right-pads the prefixes. No padded slot is ever stored in the pool.
        """
        if x.ndim != 3 or x.shape[1:] != (1, self.d_model):
            raise ValueError("decode input must have shape [batch, 1, d_model]")
        batch_size = x.shape[0]
        shape = (batch_size, 1, self.n_head, self.d_head)
        # Use one [B, D] matrix multiply, rather than B [1, D] matvecs.
        rows = x[:, 0, :]
        q = mx.transpose((rows @ self.wq).reshape(shape), (0, 2, 1, 3))
        k = mx.transpose((rows @ self.wk).reshape(shape), (0, 2, 1, 3))
        v = mx.transpose((rows @ self.wv).reshape(shape), (0, 2, 1, 3))

        # The one-token query's absolute position differs between requests.
        pairs = mx.arange(0, self.d_head, 2, dtype=mx.float32)
        frequencies = mx.exp(-pairs * (math.log(self.rope_base) / self.d_head))
        rotary_q, rotary_k = _decode_rope(q, k, cache.positions, frequencies)
        all_k, all_v = cache.append(layer_idx, rotary_k, v)
        # positions are the old lengths; valid keys include the appended token.
        context, probs, scores, valid = _decode_attention(
            rotary_q, all_k, all_v, cache.positions
        )
        context = mx.transpose(context, (0, 2, 1, 3)).reshape(x.shape)
        out = (context[:, 0, :] @ self.wo)[:, None, :]
        return out, probs, {
            "rotary_q": rotary_q,
            "rotary_k": rotary_k,
            "scaled_scores": scores,
            "probs": probs,
            "valid_keys": valid,
            "out": out,
        }

    def decode_batch(self, x: mx.array, cache, layer_idx: int):
        """Return one-token batched attention output and inspectable weights."""
        out, probs, _ = self.decode_batch_with_intermediates(x, cache, layer_idx)
        return out, probs
