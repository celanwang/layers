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

    def _prefix_source(self, prompt_ids: tuple[int, ...]) -> _Request | None:
        if not self.share_prefixes:
            return None
        for request in self._requests.values():
            if request.active and request.prompt_ids == prompt_ids:
                return request
        return None

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

        source = self._prefix_source(prompt_ids)
        seq_id = None
        try:
            if source is not None:
                seq_id = self.pool.fork(source.seq_id, prefix_len=len(prompt_ids))
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
