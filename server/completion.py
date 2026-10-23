"""Incremental text termination with no leaked partial stop markers."""

import codecs


def validate_stop(value):
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if (not isinstance(values, list) or not 1 <= len(values) <= 4
            or any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in values)):
        raise ValueError("stop must be a nonempty string or 1..4 strings of 1..256 characters")
    for value in values:
        value.encode("utf-8")
    return tuple(values)


class CompletionFilter:
    """Hold only the longest suffix that could begin a stop string."""

    def __init__(self, stops=()):
        self.stops = tuple(stops)
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending = ""
        self.stopped = False
        self.closed = False

    def feed(self, data, final=False):
        if self.closed or self.stopped:
            return "", self.stopped
        self.pending += self.decoder.decode(data, final=final)
        hits = [(self.pending.find(s) + len(s), self.pending.find(s))
                for s in self.stops if self.pending.find(s) >= 0]
        if hits:
            text = self.pending[:min(hits)[1]]
            self.pending = ""
            self.stopped = self.closed = True
            return text, True
        hold = 0
        if not final:
            for stop in self.stops:
                for length in range(1, min(len(stop), len(self.pending) + 1)):
                    if self.pending.endswith(stop[:length]):
                        hold = max(hold, length)
        boundary = len(self.pending) - hold
        text, self.pending = self.pending[:boundary], self.pending[boundary:]
        self.closed = final
        return text, False
