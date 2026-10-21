"""Iteration-level scheduling over single-request cached model forwards."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import numpy as np

from .paged_kv import PagedKVCache, PagedKVPool, PoolExhaustedError


class RequestCapacityError(RuntimeError):
    """Raised when an engine cannot admit another request immediately."""


@dataclass(frozen=True)
class TokenEvent:
    """One sampled token produced during an engine iteration."""

    seq_id: int
    token_id: int
    finished: bool


@dataclass
class _Request:
    seq_id: int
    prompt_ids: tuple[int, ...]
    sampler: object
    max_new_tokens: int
    prompt_logits: mx.array
    next_logits: mx.array
    generated: list[int]
    active: bool = True


class Engine:
    """A small continuous-batching scheduler with one model forward per request.

    Requests are admitted eagerly. Each :meth:`step` samples one token for
    every active request, advances its cache independently, and releases pages
    as soon as the request reaches its configured generation limit.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        pool: PagedKVPool | None = None,
        block_size: int = 16,
        max_context: int | None = None,
        max_concurrent: int = 16,
        max_new_tokens: int | None = None,
        share_prefixes: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        model_context = int(getattr(model, "ctx", 512))
        if max_context is None:
            max_context = min(model_context, 512)
        if max_context <= 0 or max_context > model_context:
            raise ValueError("max_context must be positive and no larger than model.ctx")
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive when specified")

        if pool is None:
            pool = PagedKVPool(
                num_layers=int(model.n_layer),
                block_size=block_size,
                max_context=max_context,
                max_sequences=max_concurrent,
            )
        elif pool.num_layers != int(model.n_layer):
            raise ValueError("pool layer count must match model.n_layer")
        self.pool = pool
        self.max_context = min(int(max_context), pool.max_context)
        self.max_concurrent = min(int(max_concurrent), pool.max_sequences)
        self.max_new_tokens = max_new_tokens
        self.share_prefixes = bool(share_prefixes)
        self._requests: dict[int, _Request] = {}

    @property
    def active_seq_ids(self) -> tuple[int, ...]:
        return tuple(req.seq_id for req in self._requests.values() if req.active)

    @property
    def active_requests(self) -> int:
        return len(self.active_seq_ids)

    def _prompt_ids(self, prompt) -> tuple[int, ...]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("a tokenizer is required for string prompts")
            values = self.tokenizer.encode(prompt)
        else:
            values = np.asarray(prompt, dtype=np.int64).reshape(-1).tolist()
        token_ids = tuple(int(value) for value in values)
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        if any(value < 0 for value in token_ids):
            raise ValueError("token IDs must be non-negative")
        return token_ids

    def _prefix_source(
        self, prompt_ids: tuple[int, ...]
    ) -> tuple[_Request, int] | None:
        if not self.share_prefixes:
            return None
        best_source = None
        best_length = 0
        best_exact = False
        for request in self._requests.values():
            if not request.active:
                continue
            shared_length = 0
            for left, right in zip(prompt_ids, request.prompt_ids):
                if left != right:
                    break
                shared_length += 1
            exact = request.prompt_ids == prompt_ids
            if shared_length > best_length or (shared_length == best_length and exact):
                best_source = request
                best_length = shared_length
                best_exact = exact
        if best_source is None or best_length == 0:
            return None

        # A saved prompt logit is reusable only when the complete prompt is
        # identical. For a shorter prompt that prefixes an active request,
        # re-run its final token so the logits correspond to the shorter prefix.
        if best_length == len(prompt_ids) and not best_exact:
            best_length -= 1
        return best_source, best_length

    def add_request(self, prompt, sampler, max_new_tokens: int | None = None) -> int:
        """Admit a prompt and return its sequence ID.

        Capacity is fail-fast: callers receive :class:`RequestCapacityError`
        and may retry after an active request completes. No hidden queue is
        maintained.
        """
        if not callable(sampler):
            raise TypeError("sampler must be callable")
        if self.active_requests >= self.max_concurrent:
            raise RequestCapacityError(
                f"engine is at its active request limit ({self.max_concurrent})"
            )
        prompt_ids = self._prompt_ids(prompt)
        if len(prompt_ids) > self.max_context:
            raise ValueError(
                f"prompt length {len(prompt_ids)} exceeds context {self.max_context}"
            )
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self.max_context - len(prompt_ids)
        if max_new_tokens <= 0:
            raise ValueError("prompt leaves no room for generated tokens")
        max_new_tokens = min(int(max_new_tokens), self.max_context - len(prompt_ids))

        # Reserve the no-sharing worst case so an admitted request cannot run
        # out of pages half way through decode. Prefix sharing may lower actual
        # occupancy, but capacity admission stays deterministic and safe.
        reserved_blocks = sum(
            math.ceil((len(request.prompt_ids) + request.max_new_tokens) / self.pool.block_size)
            for request in self._requests.values()
            if request.active
        )
        reserved_blocks += math.ceil(
            (len(prompt_ids) + max_new_tokens) / self.pool.block_size
        )
        if reserved_blocks > self.pool.capacity:
            raise RequestCapacityError(
                "request set exceeds the pool's worst-case page reservation "
                f"({reserved_blocks} required, {self.pool.capacity} available)"
            )

        prefix = self._prefix_source(prompt_ids)
        seq_id = None
        try:
            if prefix is not None:
                source, shared_length = prefix
                seq_id = self.pool.fork(source.seq_id, prefix_len=shared_length)
                suffix = prompt_ids[shared_length:]
                if suffix:
                    logits = self.model(
                        mx.array(suffix, dtype=mx.int32),
                        cache=PagedKVCache(self.pool, seq_id),
                    )
                    mx.eval(logits)
                    prompt_logits = logits[-1]
                else:
                    prompt_logits = source.prompt_logits
            else:
                seq_id = self.pool.alloc()
                cache = PagedKVCache(self.pool, seq_id)
                prompt_array = mx.array(prompt_ids, dtype=mx.int32)
                logits = self.model(prompt_array, cache=cache)
                mx.eval(logits)
                prompt_logits = logits[-1]
        except PoolExhaustedError as exc:
            if seq_id is not None:
                self.pool.free(seq_id)
            raise RequestCapacityError(str(exc)) from exc
        except Exception:
            if seq_id is not None:
                self.pool.free(seq_id)
            raise

        self._requests[seq_id] = _Request(
            seq_id=seq_id,
            prompt_ids=prompt_ids,
            sampler=sampler,
            max_new_tokens=max_new_tokens,
            prompt_logits=prompt_logits,
            next_logits=prompt_logits,
            generated=[],
        )
        return seq_id

    def step(self) -> list[TokenEvent]:
        """Produce one token event for every request active at iteration start."""
        events = []
        current = [request for request in self._requests.values() if request.active]
        for request in current:
            token_id = int(request.sampler(request.next_logits))
            request.generated.append(token_id)
            finished = (
                len(request.generated) >= request.max_new_tokens
                or len(request.prompt_ids) + len(request.generated) >= self.max_context
            )
            events.append(TokenEvent(request.seq_id, token_id, finished))
            if finished:
                request.active = False
                self.pool.free(request.seq_id)
                continue

            token_array = mx.array([token_id], dtype=mx.int32)
            logits = self.model(
                token_array, cache=PagedKVCache(self.pool, request.seq_id)
            )
            mx.eval(logits)
            request.next_logits = logits[-1]
        return events

    def run(self) -> list[TokenEvent]:
        """Run until all currently admitted requests have completed."""
        events = []
        while self.active_requests:
            events.extend(self.step())
        return events

    def generated_tokens(self, seq_id: int) -> tuple[int, ...]:
        """Return tokens emitted so far, including those of completed requests."""
        try:
            return tuple(self._requests[seq_id].generated)
        except KeyError as exc:
            raise KeyError(f"unknown request id: {seq_id}") from exc

    def cancel_request(self, seq_id: int) -> None:
        """Release a request and forget its history, including after completion.

        Idempotent, but like admission and stepping this must be called by the
        scheduler's owner thread, between iterations. HTTP threads signal that
        owner instead of mutating the pool themselves.
        """
        request = self._requests.pop(seq_id, None)
        if request is not None and request.active:
            self.pool.free(seq_id)
            request.active = False


class _BatchedPagedKVCache:
    """Snapshot row order/positions and adapt the unchanged L4 page pool."""

    def __init__(self, pool: PagedKVPool, seq_ids):
        self.pool = pool
        self.seq_ids = tuple(seq_ids)
        if not self.seq_ids or len(set(self.seq_ids)) != len(self.seq_ids):
            raise ValueError("a decode batch requires non-empty, distinct sequence IDs")
        self.lengths = tuple(pool.seq_len(seq_id) for seq_id in self.seq_ids)
        self.positions = mx.array(self.lengths, dtype=mx.float32)
        self.padded_length = max(self.lengths) + 1

    def append(self, layer_idx: int, k: mx.array, v: mx.array):
        keys, values = [], []
        for row, seq_id in enumerate(self.seq_ids):
            row_k, row_v = self.pool.append(
                seq_id, k[row : row + 1], v[row : row + 1], layer_idx=layer_idx
            )
            pad = self.padded_length - row_k.shape[-2]
            if pad:
                zeros = mx.zeros(
                    (1, row_k.shape[1], pad, row_k.shape[-1]), dtype=row_k.dtype
                )
                row_k = mx.concatenate((row_k, zeros), axis=-2)
                row_v = mx.concatenate((row_v, zeros), axis=-2)
            keys.append(row_k)
            values.append(row_v)
        if len(keys) == 1:
            return keys[0], values[0]
        return mx.concatenate(keys, axis=0), mx.concatenate(values, axis=0)


class BatchedEngine(Engine):
    """L4 admission/prefill with one shared decode forward per iteration.

    Like Engine, this synchronous scheduler admits and prefills requests through
    ``add_request`` between iterations. Each new prompt is processed separately
    before joining the next decode batch; prefix sharing and reservation rules
    are inherited unchanged. Callers serialize add_request/step at boundaries.
    """

    def step(self) -> list[TokenEvent]:
        current = [request for request in self._requests.values() if request.active]
        events, continuing, tokens = [], [], []
        finished_requests = []
        for request in current:
            token_id = int(request.sampler(request.next_logits))
            request.generated.append(token_id)
            finished = (
                len(request.generated) >= request.max_new_tokens
                or len(request.prompt_ids) + len(request.generated) >= self.max_context
            )
            events.append(TokenEvent(request.seq_id, token_id, finished))
            if finished:
                finished_requests.append(request)
            else:
                continuing.append(request)
                tokens.append([token_id])

        if continuing:
            cache = _BatchedPagedKVCache(
                self.pool, [request.seq_id for request in continuing]
            )
            logits = self.model.decode_batch(mx.array(tokens, dtype=mx.int32), cache)
            mx.eval(logits)
            for row, request in enumerate(continuing):
                request.next_logits = logits[row, 0]

        # Release only after the iteration, including all surviving-row writes.
        for request in finished_requests:
            request.active = False
            self.pool.free(request.seq_id)
        return events
