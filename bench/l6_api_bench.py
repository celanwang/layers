"""L0-protocol HTTP/SSE overhead, concurrent scaling, and text oracle.

Fresh engine/pool per trial; paired alternating direct/API order, 3 warmups,
10 repeats, median/p90 and all raw samples. Timings include the complete
request (template/tokenization, admission/prefill, decode, UTF-8 output;
HTTP additionally includes TCP setup, parsing, queues, and SSE consumption).
Server/model startup and shutdown are outside the timed request.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import BatchedEngine, Sampler
from server.app import CHECKPOINT, TOKENIZER, chat_template, load_engine, make_server

WARMUP = 3
REPS = 10
TOKENS = 64
SEED = 20261021
BATCH_SIZES = (1, 4, 8)
PROMPTS = [f"To be or not to be, that is the question. Scene {row}." for row in range(8)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(samples):
    ordered = sorted(samples)
    return {"median_s": ordered[len(ordered) // 2],
            "p90_s": ordered[int(len(ordered) * 0.9) - 1],
            "min_s": ordered[0], "warmup": WARMUP, "reps": len(samples),
            "samples_s": samples}


def body(row):
    return {"model": "layers-gpt", "messages": [{"role": "user", "content": PROMPTS[row]}],
            "stream": True, "max_tokens": TOKENS, "temperature": 0,
            "top_p": 1.0, "seed": SEED + row}


def new_engine(model, tokenizer):
    return BatchedEngine(model, tokenizer, max_concurrent=8, share_prefixes=False)


@contextmanager
def serving(model, tokenizer):
    server = make_server("127.0.0.1", 0, new_engine(model, tokenizer))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        assert server.bridge.engine.pool.allocated_blocks == 0
        assert server.bridge.engine.pool.active_sequences == 0
        assert not server.bridge.engine._requests


def direct(model, tokenizer, row=0):
    engine = new_engine(model, tokenizer)
    request = body(row)
    start = time.perf_counter()
    ids = tokenizer.encode(chat_template(request["messages"]))
    seq_id = engine.add_request(ids, Sampler(temperature=0, top_p=1.0, seed=SEED + row), TOKENS)
    events = engine.run()
    text = tokenizer.decode(list(engine.generated_tokens(seq_id)))
    elapsed = time.perf_counter() - start
    assert len(events) == TOKENS and engine.pool.allocated_blocks == engine.pool.active_sequences == 0
    return {"elapsed_s": elapsed, "text": text}


def api(server, row=0, barrier=None):
    if barrier is not None:
        barrier.wait(10)
    start = time.perf_counter()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body(row)),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200, response.read()
        pieces = []
        first_content_s = None
        done = False
        finish_reason = None
        for line in response:
            if line == b"\n":
                continue
            assert line.startswith(b"data: "), line
            if line == b"data: [DONE]\n":
                done = True
                break
            chunk = json.loads(line[6:])
            choice = chunk["choices"][0]
            piece = choice["delta"].get("content", "")
            if piece:
                if first_content_s is None:
                    first_content_s = time.perf_counter() - start
                pieces.append(piece)
            finish_reason = choice["finish_reason"] or finish_reason
        elapsed = time.perf_counter() - start
        assert done and finish_reason == "length"
        return {"elapsed_s": elapsed, "text": "".join(pieces), "first_content_s": first_content_s}
    finally:
        connection.close()


def metadata():
    import mlx.core as mx

    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    device = {"chip": sysctl("machdep.cpu.brand_string"), "memory_bytes": int(sysctl("hw.memsize")),
              "os": platform.platform(), "python": sys.version.split()[0], "mlx": mx.__version__}
    protocol = {
        "warmup": WARMUP, "reps": REPS, "percentiles": ["median", "p90"],
        "generated_tokens_per_request": TOKENS, "temperature": 0, "top_p": 1,
        "seed": SEED, "batch_sizes": BATCH_SIZES, "prompts": PROMPTS,
        "dtype": "float32", "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT), "tokenizer_sha256": sha256(TOKENIZER),
        "share_prefixes": False, "block_size": 16, "max_concurrent": 8,
        "pending_queue_size": 32, "output_queue_size": 128,
        "timing": "end-to-end request including prefill; API includes fresh TCP connection and SSE consumption",
        "trial_order": "paired alternating direct/API; fresh engine/pool/server per trial; startup excluded",
        "overhead": "100 * (API median seconds/token / direct median seconds/token - 1)",
        "latency": "per-request wall time, and time to first nonempty content (not necessarily first token for UTF-8)",
    }
    sources = {p: sha256(ROOT / p) for p in (
        "server/app.py", "layers/engine.py", "layers/model.py", "layers/attention.py",
        "layers/inference.py", "layers/paged_kv.py", "layers/tokenizer.py", "bench/l6_api_bench.py",
    )}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    config = {"device": device, "protocol": protocol, "source_sha256": sources, "git_commit": commit}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    return config, config_hash


def main():
    config, config_hash = metadata()
    loaded = load_engine()
    model, tokenizer = loaded.model, loaded.tokenizer
    print(f"device: {config['device']['chip']}, {config['device']['memory_bytes'] / 2**30:.0f} GiB, MLX {config['device']['mlx']}", flush=True)
    print(f"config_hash: {config_hash} (commit {config['git_commit'][:12]})", flush=True)
    direct_samples, api_samples = [], []
    oracle_trials = 0
    oracle_example = None
    for repetition in range(WARMUP + REPS):
        results = [None, None]
        for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            if index == 0:
                results[0] = direct(model, tokenizer)
            else:
                with serving(model, tokenizer) as server:
                    results[1] = api(server)
        assert results[0]["text"] == results[1]["text"], "API/direct text oracle failed"
        oracle_trials += 1
        oracle_example = results[0]["text"]
        if repetition >= WARMUP:
            direct_samples.append(results[0]["elapsed_s"])
            api_samples.append(results[1]["elapsed_s"])
    left, right = summarize(direct_samples), summarize(api_samples)
    overhead_percent = 100 * (right["median_s"] / left["median_s"] - 1)
    overhead = {"direct": left, "api_stream": right,
                "direct_ms_per_token": left["median_s"] * 1000 / TOKENS,
                "api_ms_per_token": right["median_s"] * 1000 / TOKENS,
                "overhead_percent": overhead_percent}
    print(f"overhead: direct {overhead['direct_ms_per_token']:.4f} ms/token, API {overhead['api_ms_per_token']:.4f} ms/token, {overhead_percent:.2f}%", flush=True)

    # Independently generated expected texts for every distinct concurrent prompt.
    expected = [direct(model, tokenizer, row)["text"] for row in range(8)]
    scaling = []
    for count in BATCH_SIZES:
        aggregate, latency, ttfc = [], [[] for _ in range(count)], [[] for _ in range(count)]
        for repetition in range(WARMUP + REPS):
            with serving(model, tokenizer) as server:
                with ThreadPoolExecutor(max_workers=count) as pool:
                    barrier = threading.Barrier(count + 1)
                    futures = [pool.submit(api, server, row, barrier) for row in range(count)]
                    start = time.perf_counter()
                    barrier.wait(10)
                    results = [f.result() for f in futures]
                    elapsed = time.perf_counter() - start
            assert [r["text"] for r in results] == expected[:count], "concurrent text oracle failed"
            oracle_trials += count
            if repetition >= WARMUP:
                aggregate.append(elapsed)
                for row, result in enumerate(results):
                    latency[row].append(result["elapsed_s"])
                    ttfc[row].append(result["first_content_s"])
        wall = summarize(aggregate)
        entry = {"concurrency": count, "aggregate_wall": wall,
                 "aggregate_tokens_per_second": count * TOKENS / wall["median_s"],
                 "per_request_latency": [summarize(samples) for samples in latency],
                 "per_request_first_content": [summarize(samples) for samples in ttfc],
                 "token_oracle_passed": True, "all_completed": True, "pool_reclaimed": True}
        scaling.append(entry)
        medians = [r["median_s"] for r in entry["per_request_latency"]]
        p90s = [r["p90_s"] for r in entry["per_request_latency"]]
        print(f"concurrency={count}: {entry['aggregate_tokens_per_second']:.2f} tok/s; request median {min(medians):.6f}..{max(medians):.6f}s, p90 {min(p90s):.6f}..{max(p90s):.6f}s", flush=True)
    predictions = {"api_overhead_below_5_percent": overhead_percent < 5,
                   "eight_streams_isolated_completed_reclaimed": True,
                   "api_stream_equals_direct_engine": True}
    record = {"iteration": "L6-T2", "timestamp": datetime.now(timezone.utc).isoformat(),
              "config_hash": config_hash, **config, "api_overhead": overhead,
              "concurrent_scaling": scaling,
              "end_to_end_oracle": {"passed": True, "request_comparisons": oracle_trials,
                                    "example_text": oracle_example,
                                    "example_sha256": hashlib.sha256(oracle_example.encode()).hexdigest()},
              "predictions": predictions}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / f"results/l6/api-bench-{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"oracle: {oracle_trials} request comparisons passed; all trial pools reclaimed")
    print(f"predictions: {json.dumps(predictions, sort_keys=True)}")
    print(f"written: {path.relative_to(ROOT)}", flush=True)
    if not all(predictions.values()):
        raise SystemExit("FAIL: API overhead >=5%; raw evidence retained")
    print("PASS: API overhead <5%, 1/4/8 streams isolated/completed/reclaimed, direct-engine text oracle")


if __name__ == "__main__":
    main()
