"""Byte-level BPE, implemented from the L1 principle note without source references.

Ordinary token IDs are bytes (0..255), then merges in rank order. The four
special IDs live immediately above that vocabulary and are handled only at
the text API boundary. There is no linguistic pre-tokenization.
"""

import heapq
import json
import re
from collections import defaultdict
from pathlib import Path

SPECIAL_TOKENS = (
    "<|beginoftext|>",
    "<|endoftext|>",
    "<|padding|>",
    "<|unknown|>",
)
_SPECIAL_PATTERN = re.compile("|".join(map(re.escape, SPECIAL_TOKENS)))


class _Sequence:
    """A linked list over fixed byte positions; merging keeps the left position."""

    def __init__(self, data: bytes):
        self.ids = list(data)
        self.prev = list(range(-1, len(data) - 1))
        self.next = list(range(1, len(data))) + ([-1] if data else [])

    def merge(self, left: int, token_id: int) -> None:
        right = self.next[left]
        after = self.next[right]
        self.ids[left] = token_id
        self.ids[right] = -1  # An invalidated position, never a vocabulary ID.
        self.next[left] = after
        if after != -1:
            self.prev[after] = left

    def tokens(self) -> list[int]:
        result = []
        position = 0 if self.ids else -1
        while position != -1:
            result.append(self.ids[position])
            position = self.next[position]
        return result


class BPETokenizer:
    """A deterministic, reversible tokenizer with an ordinary byte vocabulary.

    ``vocab_size`` excludes the four reserved IDs. Literal special-token names
    in encode input map to those IDs; decode restores their literal spelling.
    Training always sees ordinary UTF-8 bytes, never reserved IDs. If training
    runs out of adjacent pairs, the actual vocabulary can be smaller than the
    requested size (including for empty input).
    """

    def __init__(self, merges: tuple[tuple[int, int], ...] = ()):
        self._merges = tuple(tuple(pair) for pair in merges)
        vocab = [bytes([value]) for value in range(256)]
        self._ranks = {}
        for pair in self._merges:
            if len(pair) != 2 or any(
                type(value) is not int or not 0 <= value < len(vocab)
                for value in pair
            ):
                raise ValueError("merge operands must reference earlier ordinary IDs")
            if pair in self._ranks:
                raise ValueError("duplicate merge pair")
            self._ranks[pair] = len(vocab)
            vocab.append(vocab[pair[0]] + vocab[pair[1]])
        self._vocab = tuple(vocab)
        self._special_ids = {
            token: self.vocab_size + offset
            for offset, token in enumerate(SPECIAL_TOKENS)
        }

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        """Repeatedly merge the most frequent pair, breaking ties by ID tuple.

        Pair occurrence sets index live edges in a linked list. Only edges
        touching a merge can change. A lazy heap selects (-count, pair), and
        a sorted snapshot of occurrences enforces left-to-right replacement
        for overlapping pairs such as ``aaa``.
        """
        if type(vocab_size) is not int or vocab_size < 256:
            raise ValueError("vocab_size must be an integer >= 256")
        sequence = _Sequence(text.encode("utf-8"))
        ids, previous, following = sequence.ids, sequence.prev, sequence.next
        positions = defaultdict(set)
        for left in range(len(ids) - 1):
            positions[(ids[left], ids[left + 1])].add(left)
        heap = [(-len(found), pair) for pair, found in positions.items()]
        heapq.heapify(heap)
        merges = []

        while len(merges) + 256 < vocab_size and heap:
            negative_count, pair = heapq.heappop(heap)
            if -negative_count != len(positions[pair]) or not positions[pair]:
                continue  # This heap entry predates a neighboring merge.
            token_id = 256 + len(merges)
            merges.append(pair)
            changed = set()
            for left in sorted(positions[pair]):
                right = following[left]
                if ids[left] != pair[0] or right == -1 or ids[right] != pair[1]:
                    continue  # An earlier occurrence in this pass overlapped.
                before = previous[left]
                # Remove the three old edges before changing their endpoints.
                for start in (before, left, right):
                    if start != -1 and following[start] != -1:
                        old_pair = (ids[start], ids[following[start]])
                        positions[old_pair].remove(start)
                        changed.add(old_pair)
                sequence.merge(left, token_id)
                # Restore the (at most two) new edges.
                for start in (before, left):
                    if start != -1 and following[start] != -1:
                        new_pair = (ids[start], ids[following[start]])
                        positions[new_pair].add(start)
                        changed.add(new_pair)
            for changed_pair in changed:
                count = len(positions[changed_pair])
                if count:
                    heapq.heappush(heap, (-count, changed_pair))
        return cls(tuple(merges))

    def _encode_bytes(self, data: bytes) -> list[int]:
        sequence = _Sequence(data)
        ids, previous, following = sequence.ids, sequence.prev, sequence.next
        heap = []
        for left in range(len(ids) - 1):
            rank = self._ranks.get((ids[left], ids[left + 1]))
            if rank is not None:
                heap.append((rank, left, left + 1))
        heapq.heapify(heap)
        while heap:
            rank, left, right = heapq.heappop(heap)
            if (
                ids[left] == -1
                or following[left] != right
                or self._ranks.get((ids[left], ids[right])) != rank
            ):
                continue
            sequence.merge(left, rank)
            for start in (previous[left], left):
                if start != -1 and following[start] != -1:
                    end = following[start]
                    new_rank = self._ranks.get((ids[start], ids[end]))
                    if new_rank is not None:
                        heapq.heappush(heap, (new_rank, start, end))
        return sequence.tokens()

    def encode(self, text: str) -> list[int]:
        """Encode UTF-8 bytes by merge rank, recognizing reserved spellings."""
        result = []
        start = 0
        for match in _SPECIAL_PATTERN.finditer(text):
            result.extend(self._encode_bytes(text[start:match.start()].encode("utf-8")))
            result.append(self._special_ids[match.group()])
            start = match.end()
        result.extend(self._encode_bytes(text[start:].encode("utf-8")))
        return result

    def decode(self, ids: list[int]) -> str:
        """Join bytes before UTF-8 decoding; incomplete sequences use replacement."""
        chunks = []
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < self.vocab_size + 4:
                raise ValueError(f"invalid token ID: {token_id!r}")
            if token_id < self.vocab_size:
                chunks.append(self._vocab[token_id])
            else:
                chunks.append(SPECIAL_TOKENS[token_id - self.vocab_size].encode("utf-8"))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def _payload(self) -> dict:
        return {
            "version": 1,
            "vocab_size": self.vocab_size,
            "vocab": [piece.hex() for piece in self._vocab],
            "merges": [list(pair) for pair in self._merges],
            "special_tokens": self._special_ids,
        }

    def save(self, path: str) -> None:
        """Write deterministic JSON with byte pieces represented as hex strings."""
        Path(path).write_text(
            json.dumps(self._payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported tokenizer format version")
        tokenizer = cls(tuple(tuple(pair) for pair in payload["merges"]))
        if payload != tokenizer._payload():
            raise ValueError("inconsistent vocabulary or reserved IDs")
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Number of ordinary byte/BPE tokens, excluding the four reserved IDs."""
        return len(self._vocab)
