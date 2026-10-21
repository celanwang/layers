"""Small, inspectable inference helpers for cached GPT decoding."""

import codecs
from collections.abc import Iterator

import mlx.core as mx

from .quant import dequantize_kv, quantize_kv


class KVCache:
    """Per-layer append-only storage for rotary keys and values."""

    def __init__(self, n_layers: int):
        if type(n_layers) is not int or n_layers <= 0:
            raise ValueError("n_layers must be a positive integer")
        self.n_layers = n_layers
        self._keys: list[mx.array | None] = [None] * n_layers
        self._values: list[mx.array | None] = [None] * n_layers

    @property
    def seq_len(self) -> int:
        """Number of tokens currently stored in each populated layer."""
        lengths = {
            key.shape[-2] for key in self._keys if key is not None
        }
        if len(lengths) > 1:
            raise RuntimeError("KV cache layers have different sequence lengths")
        return next(iter(lengths), 0)

    def get(self, layer_idx: int) -> tuple[mx.array | None, mx.array | None]:
        self._validate_layer(layer_idx)
        return self._keys[layer_idx], self._values[layer_idx]

    def append(
        self, layer_idx: int, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append one layer's ``[batch, heads, sequence, features]`` K/V arrays."""
        self._validate_layer(layer_idx)
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have rank four")
        if keys.shape[:-1] != values.shape[:-1]:
            raise ValueError("keys and values must agree except in feature width")
        previous_k, previous_v = self.get(layer_idx)
        if previous_k is None:
            all_k, all_v = keys, values
        else:
            if previous_k.shape[:2] != keys.shape[:2]:
                raise ValueError("cache batch and head dimensions cannot change")
            if previous_k.shape[-1] != keys.shape[-1]:
                raise ValueError("key feature width cannot change")
            if previous_v.shape[-1] != values.shape[-1]:
                raise ValueError("value feature width cannot change")
            all_k = mx.concatenate((previous_k, keys), axis=-2)
            all_v = mx.concatenate((previous_v, values), axis=-2)
        self._keys[layer_idx] = all_k
        self._values[layer_idx] = all_v
        return all_k, all_v

    def _validate_layer(self, layer_idx: int) -> None:
        if type(layer_idx) is not int or not 0 <= layer_idx < self.n_layers:
            raise ValueError("layer_idx is outside this cache")


class QuantKVCache(KVCache):
    """Append-only INT8 storage, exposing the same fp32 reads as KVCache.

    Each token is quantized once across its complete d_model vector. Reading
    or appending never requantizes older tokens and retains no fp32 copy.
    """

    def __init__(self, n_layers: int):
        super().__init__(n_layers)
        self._key_scales: list[mx.array | None] = [None] * n_layers
        self._value_scales: list[mx.array | None] = [None] * n_layers

    @property
    def storage_nbytes(self) -> int:
        return sum(
            array.nbytes
            for arrays in (self._keys, self._values, self._key_scales, self._value_scales)
            for array in arrays if array is not None
        )

    def get(self, layer_idx: int) -> tuple[mx.array | None, mx.array | None]:
        self._validate_layer(layer_idx)
        if self._keys[layer_idx] is None:
            return None, None
        return (
            dequantize_kv(self._keys[layer_idx], self._key_scales[layer_idx]),
            dequantize_kv(self._values[layer_idx], self._value_scales[layer_idx]),
        )

    def append(
        self, layer_idx: int, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self._validate_layer(layer_idx)
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have rank four")
        if keys.shape[:-1] != values.shape[:-1]:
            raise ValueError("keys and values must agree except in feature width")
        previous_k, previous_v = self._keys[layer_idx], self._values[layer_idx]
        if previous_k is not None:
            if previous_k.shape[:2] != keys.shape[:2]:
                raise ValueError("cache batch and head dimensions cannot change")
            if previous_k.shape[-1] != keys.shape[-1]:
                raise ValueError("key feature width cannot change")
            if previous_v.shape[-1] != values.shape[-1]:
                raise ValueError("value feature width cannot change")
        # Quantize both before mutation; invalid V cannot leave a partial K.
        qk, sk = quantize_kv(keys)
        qv, sv = quantize_kv(values)
        if previous_k is not None:
            qk = mx.concatenate((previous_k, qk), axis=-2)
            qv = mx.concatenate((previous_v, qv), axis=-2)
            sk = mx.concatenate((self._key_scales[layer_idx], sk), axis=-2)
            sv = mx.concatenate((self._value_scales[layer_idx], sv), axis=-2)
        mx.eval(qk, qv, sk, sv)
        self._keys[layer_idx], self._values[layer_idx] = qk, qv
        self._key_scales[layer_idx], self._value_scales[layer_idx] = sk, sv
        return self.get(layer_idx)


class Sampler:
    """Temperature, top-k, and nucleus sampling with a repeatable seed."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int = 42,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.temperature = float(temperature)
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self._draws = 0

    def probabilities(self, logits: mx.array) -> mx.array:
        """Return the normalized distribution after configured truncation."""
        logits = mx.array(logits)
        if logits.ndim != 1 or logits.shape[0] == 0:
            raise ValueError("logits must be a non-empty vocabulary vector")
        if self.temperature == 0 or self.top_k == 1:
            greedy = mx.argmax(logits)
            return (mx.arange(logits.shape[0]) == greedy).astype(mx.float32)

        scaled = logits.astype(mx.float32) / self.temperature
        order = mx.argsort(scaled)[::-1]
        rank = mx.argsort(order)
        if self.top_k is not None and self.top_k < logits.shape[0]:
            keep = rank < self.top_k
            scaled = mx.where(keep, scaled, -float("inf"))
        probs = mx.softmax(scaled, axis=-1)

        if self.top_p is not None and self.top_p < 1:
            sorted_probs = mx.take(probs, order, axis=0)
            cumulative = mx.cumsum(sorted_probs, axis=-1)
            keep_sorted = cumulative - sorted_probs < self.top_p
            keep_original = mx.take(keep_sorted, rank, axis=0)
            probs = mx.where(keep_original, probs, 0.0)
            probs = probs / mx.sum(probs)
        return probs

    def __call__(self, logits: mx.array) -> int:
        """Draw one token ID from a vocabulary logits vector."""
        probs = self.probabilities(logits)
        mx.random.seed(self.seed + self._draws)
        self._draws += 1
        token = mx.random.categorical(mx.log(probs))
        mx.eval(token)
        return int(token.item())


def _token_bytes(tokenizer, token_id: int) -> bytes:
    if 0 <= token_id < tokenizer.vocab_size:
        return tokenizer._vocab[token_id]
    return tokenizer.decode([token_id]).encode("utf-8")


def generate(
    model,
    tokenizer,
    prompt: str | list[int],
    max_new: int,
    sampler: Sampler,
    cache: KVCache | None = None,
) -> Iterator[str]:
    """Yield generated text incrementally while caching each layer's K/V."""
    if type(max_new) is not int or max_new < 0:
        raise ValueError("max_new must be a non-negative integer")
    prompt_ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if any(type(token_id) is not int for token_id in prompt_ids):
        raise ValueError("prompt token IDs must be integers")

    injected_cache = cache
    cache = KVCache(model.n_layer)
    if injected_cache is not None:
        if injected_cache.n_layers != model.n_layer:
            raise ValueError("cache layer count must match model")
        cache = injected_cache
    current_ids = mx.array(prompt_ids, dtype=mx.int32)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    for _ in range(max_new):
        logits = model(current_ids, cache=cache)
        mx.eval(logits)
        token_id = sampler(logits[-1])
        current_ids = mx.array([token_id], dtype=mx.int32)
        piece = decoder.decode(_token_bytes(tokenizer, token_id), final=False)
        if piece:
            yield piece
    final_piece = decoder.decode(b"", final=True)
    if final_piece:
        yield final_piece
