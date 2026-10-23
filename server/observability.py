"""Bounded in-process counters and latency samples; never retains user text."""

from collections import Counter, deque
import threading


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = Counter({name: 0 for name in (
            "received", "completed", "rejected", "cancelled", "errors", "generated_tokens",
            "log_write_failures")})
        self.samples = {name: deque(maxlen=1024) for name in ("ttft_seconds", "itl_seconds")}
        self.gauges = {"active_requests": 0, "waiting_requests": 0, "allocated_pages": 0}

    def count(self, name, amount=1):
        with self.lock:
            self.counters[name] += amount

    def observe(self, name, value):
        with self.lock:
            self.samples[name].append(value)

    def publish(self, **values):
        with self.lock:
            self.gauges.update(values)

    def snapshot(self):
        with self.lock:
            samples = {}
            for name, values in self.samples.items():
                ordered = sorted(values)
                samples[name] = {
                    "window_count": len(ordered), "window_limit": 1024,
                    "p50": ordered[len(ordered) // 2] if ordered else None,
                    "p90": ordered[max(0, int(len(ordered) * .9) - 1)] if ordered else None,
                }
            return {"counters": dict(self.counters), "gauges": dict(self.gauges),
                    "latencies": samples}
