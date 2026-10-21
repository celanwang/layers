"""Fixed-size, copy-on-write KV pages for the L4 scheduling layer.

Pages hold rotary-encoded keys and values for every transformer layer. Attention
never indexes pages directly: :class:`PagedKVCache` gathers the logical prefix
into the contiguous tensors expected by the L4-T1 model interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx


class PoolExhaustedError(RuntimeError):
    """Raised when a request needs more physical pages than the pool owns."""


@dataclass
class _Page:
    refcount: int = 1
    layers: dict[int, tuple[mx.array, mx.array]] = field(
        default_factory=dict
    )


@dataclass
class _Sequence:
    block_table: list[int]
    layer_lengths: list[int]

    @property
    def seq_len(self) -> int:
        return max(self.layer_lengths, default=0)


class PagedKVPool:
    """A fixed-capacity physical page pool with per-sequence block tables.

    The default capacity covers 16 sequences of 512 tokens at 16 tokens per
    block. Page payloads are allocated lazily as K/V values arrive.
    """

    def __init__(
        self,
        num_layers: int,
        block_size: int = 16,
        max_context: int = 512,
        max_sequences: int = 16,
        num_blocks: int | None = None,
    ):
        if num_layers <= 0 or block_size <= 0:
            raise ValueError("num_layers and block_size must be positive")
        if max_context <= 0 or max_sequences <= 0:
            raise ValueError("max_context and max_sequences must be positive")
        if num_blocks is None:
            num_blocks = math.ceil(max_context / block_size) * max_sequences
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.num_layers = int(num_layers)
        self.block_size = int(block_size)
        self.max_context = int(max_context)
        self.max_sequences = int(max_sequences)
        self.capacity = int(num_blocks)
        self._pages: list[_Page | None] = [None] * self.capacity
        self._free: list[int] = list(range(self.capacity - 1, -1, -1))
        self._sequences: dict[int, _Sequence] = {}
        self._next_seq_id = 0
        self._layer_bytes_per_token: list[int | None] = [None] * self.num_layers

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def free_blocks(self) -> int:
        """Number of reusable physical blocks (convenience alias)."""
        return self.num_free_blocks

    @property
    def allocated_blocks(self) -> int:
        return self.capacity - self.num_free_blocks

    @property
    def block_bytes(self) -> int:
        """Bytes in one full physical block, once layer shapes are known."""
        return self.block_size * self.bytes_per_token

    @property
    def bytes_per_token(self) -> int:
        return sum(value or 0 for value in self._layer_bytes_per_token)

    @property
    def allocated_bytes(self) -> int:
        return self.allocated_blocks * self.block_bytes

    @property
    def wasted_tokens(self) -> int:
        """Unused token slots at the ends of all live logical sequences."""
        return sum(
            len(state.block_table) * self.block_size - state.seq_len
            for state in self._sequences.values()
        )

    @property
    def wasted_bytes(self) -> int:
        return self.wasted_tokens * self.bytes_per_token

    @property
    def active_sequences(self) -> int:
        return len(self._sequences)

    def _state(self, seq_id: int) -> _Sequence:
        try:
            return self._sequences[seq_id]
        except KeyError as exc:
            raise KeyError(f"unknown sequence id: {seq_id}") from exc

    def alloc(self) -> int:
        """Create an empty sequence and return its unique ID."""
        if len(self._sequences) >= self.max_sequences:
            raise PoolExhaustedError(
                f"sequence limit reached ({self.max_sequences} active sequences)"
            )
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self._sequences[seq_id] = _Sequence(
            block_table=[], layer_lengths=[0] * self.num_layers
        )
        return seq_id

    def _new_page(self) -> int:
        if not self._free:
            raise PoolExhaustedError(
                f"KV page pool exhausted ({self.capacity} blocks total)"
            )
        block_id = self._free.pop()
        self._pages[block_id] = _Page()
        return block_id

    def _ensure_page(self, state: _Sequence, block_index: int) -> int:
        if block_index > len(state.block_table):
            raise RuntimeError("non-contiguous KV append would leave a block-table gap")
        if block_index == len(state.block_table):
            state.block_table.append(self._new_page())
        return state.block_table[block_index]

    def _copy_page(self, block_id: int) -> int:
        old = self._pages[block_id]
        if old is None:
            raise RuntimeError("block table references a free page")
        copied = _Page()
        for layer_idx, (keys, values) in old.layers.items():
            # Arrays are immutable in MLX; materializing a new expression gives
            # the child page its own payload before a slot is replaced.
            copied.layers[layer_idx] = (
                keys + mx.zeros_like(keys),
                values + mx.zeros_like(values),
            )
        new_id = self._new_page()
        self._pages[new_id] = copied
        return new_id

    def _writable_page(self, state: _Sequence, block_index: int) -> int:
        block_id = self._ensure_page(state, block_index)
        page = self._pages[block_id]
        if page is None:
            raise RuntimeError("block table references a free page")
        if page.refcount > 1:
            new_id = self._copy_page(block_id)
            page.refcount -= 1
            state.block_table[block_index] = new_id
            return new_id
        return block_id

    def _release_page(self, block_id: int) -> None:
        page = self._pages[block_id]
        if page is None:
            raise RuntimeError("attempted to release a free page")
        page.refcount -= 1
        if page.refcount == 0:
            self._pages[block_id] = None
            self._free.append(block_id)

    def free(self, seq_id: int) -> None:
        """Release a sequence and return pages whose reference count reaches 0."""
        state = self._sequences.pop(seq_id, None)
        if state is None:
            raise KeyError(f"unknown sequence id: {seq_id}")
        for block_id in state.block_table:
            self._release_page(block_id)

    def fork(self, seq_id: int, prefix_len: int | None = None) -> int:
        """Create a sequence sharing the source prefix until its first write."""
        source = self._state(seq_id)
        if len(self._sequences) >= self.max_sequences:
            raise PoolExhaustedError(
                f"sequence limit reached ({self.max_sequences} active sequences)"
            )
        if prefix_len is None:
            prefix_len = source.seq_len
        if prefix_len < 0 or prefix_len > source.seq_len:
            raise ValueError("prefix_len must be within the source sequence")
        if any(length < prefix_len for length in source.layer_lengths):
            raise ValueError("every layer must contain the requested prefix")

        blocks_needed = math.ceil(prefix_len / self.block_size)
        table = source.block_table[:blocks_needed]
        if len(table) != blocks_needed:
            raise RuntimeError("source block table is shorter than its logical prefix")
        for block_id in table:
            page = self._pages[block_id]
            if page is None:
                raise RuntimeError("block table references a free page")
            page.refcount += 1

        child_id = self._next_seq_id
        self._next_seq_id += 1
        self._sequences[child_id] = _Sequence(
            block_table=list(table), layer_lengths=[prefix_len] * self.num_layers
        )
        return child_id

    @staticmethod
    def _dtype_size(dtype) -> int:
        name = str(dtype).lower()
        if "float64" in name or "int64" in name or "uint64" in name:
            return 8
        if any(value in name for value in ("float16", "bfloat16", "int16", "uint16")):
            return 2
        if "bool" in name or "int8" in name or "uint8" in name:
            return 1
        return 4

    def append(self, seq_id: int, k: mx.array, v: mx.array, layer_idx: int = 0):
        """Append one layer's K/V chunk, applying page-level copy-on-write."""
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers})")
        if k.ndim == 3:
            k = mx.expand_dims(k, axis=0)
            v = mx.expand_dims(v, axis=0)
        if k.ndim != 4 or v.ndim != 4 or k.shape[0] != 1 or v.shape[0] != 1:
            raise ValueError("K and V must have shape [1, heads, tokens, head_dim]")
        if k.shape != v.shape or k.shape[-2] <= 0:
            raise ValueError("K and V must have matching shapes with at least one token")

        bytes_per_token = (
            2 * k.shape[0] * k.shape[1] * k.shape[3] * self._dtype_size(k.dtype)
        )
        known_bytes = self._layer_bytes_per_token[layer_idx]
        if known_bytes is not None and known_bytes != bytes_per_token:
            raise ValueError("K/V shape or dtype changed within a transformer layer")
        self._layer_bytes_per_token[layer_idx] = bytes_per_token

        state = self._state(seq_id)
        keys_in, values_in = k.shape[-2], v.shape[-2]
        if keys_in != values_in:
            raise ValueError("K and V token counts differ")
        consumed = 0
        while consumed < keys_in:
            position = state.layer_lengths[layer_idx]
            block_index, slot = divmod(position, self.block_size)
            block_id = self._writable_page(state, block_index)
            page = self._pages[block_id]
            if page is None:
                raise RuntimeError("writable block unexpectedly became free")
            count = min(keys_in - consumed, self.block_size - slot)
            old = page.layers.get(layer_idx)
            prefix_k = (
                old[0][..., :slot, :]
                if old is not None
                else mx.zeros((1, k.shape[1], slot, k.shape[3]), dtype=k.dtype)
            )
            prefix_v = (
                old[1][..., :slot, :]
                if old is not None
                else mx.zeros((1, v.shape[1], slot, v.shape[3]), dtype=v.dtype)
            )
            suffix_start = slot + count
            suffix_k = (
                old[0][..., suffix_start:, :]
                if old is not None
                else mx.zeros(
                    (1, k.shape[1], self.block_size - suffix_start, k.shape[3]),
                    dtype=k.dtype,
                )
            )
            suffix_v = (
                old[1][..., suffix_start:, :]
                if old is not None
                else mx.zeros(
                    (1, v.shape[1], self.block_size - suffix_start, v.shape[3]),
                    dtype=v.dtype,
                )
            )
            key_parts = [part for part in (prefix_k, k[..., consumed : consumed + count, :], suffix_k) if part.shape[-2]]
            value_parts = [part for part in (prefix_v, v[..., consumed : consumed + count, :], suffix_v) if part.shape[-2]]
            page.layers[layer_idx] = (
                key_parts[0] if len(key_parts) == 1 else mx.concatenate(key_parts, axis=-2),
                value_parts[0] if len(value_parts) == 1 else mx.concatenate(value_parts, axis=-2),
            )
            state.layer_lengths[layer_idx] += count
            consumed += count
        return self.gather(seq_id, layer_idx)

    def append_pages(self, seq_id: int, k: mx.array, v: mx.array, layer_idx: int = 0):
        """Write a K/V chunk without reconstructing its logical prefix.

        Validation, lazy page payloads, and CoW match :meth:`append`. This
        additive entry point intentionally stops before the legacy gather.
        """
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers})")
        if k.ndim == 3:
            k = mx.expand_dims(k, axis=0)
            v = mx.expand_dims(v, axis=0)
        if k.ndim != 4 or v.ndim != 4 or k.shape[0] != 1 or v.shape[0] != 1:
            raise ValueError("K and V must have shape [1, heads, tokens, head_dim]")
        if k.shape != v.shape or k.shape[-2] <= 0:
            raise ValueError("K and V must have matching shapes with at least one token")

        bytes_per_token = (
            2 * k.shape[0] * k.shape[1] * k.shape[3] * self._dtype_size(k.dtype)
        )
        known_bytes = self._layer_bytes_per_token[layer_idx]
        if known_bytes is not None and known_bytes != bytes_per_token:
            raise ValueError("K/V shape or dtype changed within a transformer layer")
        self._layer_bytes_per_token[layer_idx] = bytes_per_token

        state = self._state(seq_id)
        keys_in, values_in = k.shape[-2], v.shape[-2]
        if keys_in != values_in:
            raise ValueError("K and V token counts differ")
        consumed = 0
        while consumed < keys_in:
            position = state.layer_lengths[layer_idx]
            block_index, slot = divmod(position, self.block_size)
            block_id = self._writable_page(state, block_index)
            page = self._pages[block_id]
            if page is None:
                raise RuntimeError("writable block unexpectedly became free")
            count = min(keys_in - consumed, self.block_size - slot)
            old = page.layers.get(layer_idx)
            prefix_k = (
                old[0][..., :slot, :]
                if old is not None
                else mx.zeros((1, k.shape[1], slot, k.shape[3]), dtype=k.dtype)
            )
            prefix_v = (
                old[1][..., :slot, :]
                if old is not None
                else mx.zeros((1, v.shape[1], slot, v.shape[3]), dtype=v.dtype)
            )
            suffix_start = slot + count
            suffix_k = (
                old[0][..., suffix_start:, :]
                if old is not None
                else mx.zeros(
                    (1, k.shape[1], self.block_size - suffix_start, k.shape[3]),
                    dtype=k.dtype,
                )
            )
            suffix_v = (
                old[1][..., suffix_start:, :]
                if old is not None
                else mx.zeros(
                    (1, v.shape[1], self.block_size - suffix_start, v.shape[3]),
                    dtype=v.dtype,
                )
            )
            key_parts = [part for part in (prefix_k, k[..., consumed : consumed + count, :], suffix_k) if part.shape[-2]]
            value_parts = [part for part in (prefix_v, v[..., consumed : consumed + count, :], suffix_v) if part.shape[-2]]
            page.layers[layer_idx] = (
                key_parts[0] if len(key_parts) == 1 else mx.concatenate(key_parts, axis=-2),
                value_parts[0] if len(value_parts) == 1 else mx.concatenate(value_parts, axis=-2),
            )
            state.layer_lengths[layer_idx] += count
            consumed += count

    def materialize_batch(self, seq_ids, layer_idx: int = 0):
        """Index one layer's physical pages once for a logical batch.

        Return ``(keys, values, lengths)`` with tensors ``[B, H, N*S, D]``.
        The compact bank contains each referenced physical page once; one
        ``take`` selects both K/V and every row's block table. Tail slots and
        dummy blocks are unspecified and must be masked with ``lengths``.
        Empty layers return ``(None, None, lengths)`` without allocating pages.
        """
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers})")
        seq_ids = tuple(seq_ids)
        if not seq_ids or len(set(seq_ids)) != len(seq_ids):
            raise ValueError("a page batch requires non-empty, distinct sequence IDs")
        states = [self._state(seq_id) for seq_id in seq_ids]
        lengths = tuple(state.layer_lengths[layer_idx] for state in states)
        blocks = math.ceil(max(lengths) / self.block_size)
        if not blocks:
            return None, None, lengths

        page_indices, payloads, tables = {}, [], []
        for state, length in zip(states, lengths):
            needed = math.ceil(length / self.block_size)
            if len(state.block_table) < needed:
                raise RuntimeError("block table ended before the logical KV sequence")
            table = []
            for block_id in state.block_table[:needed]:
                if block_id not in page_indices:
                    page = self._pages[block_id]
                    if page is None:
                        raise RuntimeError("block table references a free page")
                    if layer_idx not in page.layers:
                        raise RuntimeError("a KV layer is missing from a referenced page")
                    page_indices[block_id] = len(payloads)
                    payloads.append(page.layers[layer_idx])
                table.append(page_indices[block_id])
            # Index a real page for unused blocks; the length mask hides it.
            tables.append(table + [0] * (blocks - needed))

        keys, values = zip(*payloads)
        bank = mx.concatenate((*keys, *values), axis=0)
        table = mx.array(tables, dtype=mx.int32)
        indices = mx.stack((table, table + len(payloads)), axis=0)
        selected = mx.take(bank, indices, axis=0)  # [2, B, N, H, S, D]
        selected = mx.transpose(selected, (0, 1, 3, 2, 4, 5))
        selected = selected.reshape(
            (2, len(seq_ids), bank.shape[1], blocks * self.block_size, bank.shape[-1])
        )
        return selected[0], selected[1], lengths

    def gather(self, seq_id: int, layer_idx: int = 0):
        """Gather a logical layer prefix into contiguous ``(k, v)`` tensors."""
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers})")
        state = self._state(seq_id)
        length = state.layer_lengths[layer_idx]
        if length == 0:
            return None, None
        keys_out = []
        values_out = []
        remaining = length
        for block_id in state.block_table:
            if remaining <= 0:
                break
            page = self._pages[block_id]
            if page is None:
                raise RuntimeError("block table references a free page")
            layer_data = page.layers.get(layer_idx)
            if layer_data is None:
                raise RuntimeError("a KV layer is missing from a referenced page")
            count = min(remaining, self.block_size)
            keys_out.append(layer_data[0][..., :count, :])
            values_out.append(layer_data[1][..., :count, :])
            remaining -= count
        if remaining:
            raise RuntimeError("block table ended before the logical KV sequence")
        keys = keys_out[0] if len(keys_out) == 1 else mx.concatenate(keys_out, axis=-2)
        values = values_out[0] if len(values_out) == 1 else mx.concatenate(values_out, axis=-2)
        return keys, values

    def seq_len(self, seq_id: int) -> int:
        return self._state(seq_id).seq_len

    def block_table(self, seq_id: int) -> tuple[int, ...]:
        return tuple(self._state(seq_id).block_table)

    def refcount(self, block_id: int) -> int:
        if not 0 <= block_id < self.capacity:
            raise IndexError("block id is outside the pool")
        page = self._pages[block_id]
        return 0 if page is None else page.refcount


class PagedKVCache:
    """Adapter from the T1 cache protocol to one sequence in a paged pool."""

    def __init__(self, pool: PagedKVPool, seq_id: int):
        self.pool = pool
        self.seq_id = seq_id

    def get(self, layer_idx: int):
        return self.pool.gather(self.seq_id, layer_idx)

    def append(self, layer_idx: int, k: mx.array, v: mx.array):
        return self.pool.append(self.seq_id, k, v, layer_idx=layer_idx)
