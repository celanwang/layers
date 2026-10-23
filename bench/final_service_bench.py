"""Local service v1: paired direct/HTTP, real L3 assets and L6/L7 paths."""
from concurrent.futures import ThreadPoolExecutor
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

from layers import BatchedEngine, BatchedSamplingEngine, Sampler
from layers.inference import _token_bytes
from server.app import CHECKPOINT, TOKENIZER, chat_template, load_engine, make_server
from server.completion import CompletionFilter
from server.runtime import file_hash

WARMUP, REPS, TOKENS = 3, 10, 64
COUNTS = (1, 4, 8)


def summary(values):
    ordered = sorted(values)
    return {'median_s': ordered[len(ordered)//2],
            'p90_s': ordered[int(len(ordered)*.9)-1], 'samples_s': values}


def body(row):
    return {'model':'layers-gpt', 'messages':[{'role':'user', 'content':
        f'To be or not to be, that is the question. Scene {row}.'}],
        'max_tokens':TOKENS, 'temperature':0, 'seed':20261021+row,
        'stream':True, 'stream_options':{'include_usage':True}}


def new_engine(model, tokenizer, backend):
    cls = BatchedEngine if backend == 'l6' else BatchedSamplingEngine
    return cls(model, tokenizer, max_concurrent=8, share_prefixes=False)


def direct_trial(model, tokenizer, backend, count):
    engine = new_engine(model, tokenizer, backend)
    start = time.perf_counter()
    filters, text, counts, reasons = {}, {}, {}, {}
    for row in range(count):
        request = body(row)
        seq = engine.add_request(tokenizer.encode(chat_template(request['messages'])),
            Sampler(temperature=0, seed=request['seed']), TOKENS)
        filters[seq], text[seq], counts[seq], reasons[seq] = CompletionFilter(), '', 0, 'length'
    eos = tokenizer._special_ids['<|endoftext|>']
    while engine.active_requests:
        for event in engine.step():
            seq = event.seq_id
            counts[seq] += 1
            is_eos = event.token_id == eos
            part, matched = filters[seq].feed(b'' if is_eos else _token_bytes(tokenizer,event.token_id),
                                             final=event.finished or is_eos)
            text[seq] += part
            if is_eos or matched:
                reasons[seq] = 'stop'
                engine.cancel_request(seq)
    elapsed = time.perf_counter()-start
    for seq in text: engine.cancel_request(seq)
    assert engine.pool.allocated_blocks == engine.pool.active_sequences == 0
    return {'elapsed_s':elapsed, 'outputs':[(text[s],counts[s],reasons[s]) for s in text]}


def client(server, row, barrier):
    barrier.wait(30)
    start = time.perf_counter()
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=30)
    try:
        conn.request('POST','/v1/chat/completions',json.dumps(body(row)),
                     {'Content-Type':'application/json'})
        response = conn.getresponse()
        assert response.status == 200, response.read()
        parts, first, reason, usage, done = [], None, None, None, False
        for line in response:
            if line == b'\n': continue
            assert line.startswith(b'data: ')
            if line == b'data: [DONE]\n':
                done = True
                break
            chunk = json.loads(line[6:])
            if chunk.get('usage'): usage = chunk['usage']
            for choice in chunk['choices']:
                part = choice['delta'].get('content','')
                if part and first is None: first = time.perf_counter()-start
                parts.append(part)
                reason = choice['finish_reason'] or reason
        assert done and usage is not None
        return {'output':(''.join(parts),usage['completion_tokens'],reason),
                'elapsed_s':time.perf_counter()-start, 'ttft_s':first}
    finally:
        conn.close()


def api_trial(model, tokenizer, backend, count):
    server = make_server('127.0.0.1',0,new_engine(model,tokenizer,backend))
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.01})
    thread.start()
    try:
        with ThreadPoolExecutor(count) as pool:
            barrier = threading.Barrier(count+1)
            futures = [pool.submit(client,server,row,barrier) for row in range(count)]
            start = time.perf_counter()
            barrier.wait(30)
            results = [f.result() for f in futures]
            elapsed = time.perf_counter()-start
        deadline = time.monotonic()+5
        while server.bridge.snapshot()['gauges']['allocated_pages']:
            assert time.monotonic() < deadline
            time.sleep(.001)
        state = server.bridge.snapshot()
        assert state['gauges']['active_requests'] == state['gauges']['allocated_pages'] == 0
        return {'elapsed_s':elapsed,'outputs':[r['output'] for r in results],
                'clients':results, 'state':state}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        assert server.bridge.engine.pool.allocated_blocks == 0


def main():
    loaded = load_engine()
    config = {
        'protocol':{'warmup':WARMUP,'reps':REPS,'counts':COUNTS,'max_new':TOKENS,
                    'order':'paired alternating direct/API; fresh engine/pool per trial',
                    'share_prefixes':False,'generation':'greedy; EOS enabled',
                    'scope':'full request incl prefill; API adds connection, queues and SSE',
                    'concurrency_caveat':'direct eager admission vs HTTP staggered arrival',
                    'ttft':'client first nonempty content; ITL from owner token-ready events'},
        'device':{'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                  'memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
                  'platform':platform.platform(),'python':sys.version,
                  'mlx':__import__('mlx.core',fromlist=['']).__version__},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'assets':loaded.artifact_manifest,
        'source_sha256':{p:file_hash(ROOT/p) for p in ('server/app.py','server/runtime.py',
             'server/completion.py','server/observability.py','layers/engine.py',
             'layers/inference.py','layers/paged_kv.py','bench/final_service_bench.py')},
    }
    records = []
    for backend in ('l6','l7'):
        for count in COUNTS:
            direct, api, client_rows, snapshots = [],[],[],[]
            for rep in range(WARMUP+REPS):
                result = {}
                for mode in (('direct','api') if rep%2==0 else ('api','direct')):
                    result[mode] = (direct_trial if mode=='direct' else api_trial)(
                        loaded.model,loaded.tokenizer,backend,count)
                assert result['direct']['outputs'] == result['api']['outputs'], 'output Oracle mismatch'
                if rep>=WARMUP:
                    direct.append(result['direct']['elapsed_s'])
                    api.append(result['api']['elapsed_s'])
                    client_rows.append(result['api']['clients'])
                    snapshots.append(result['api']['state'])
            left,right = summary(direct),summary(api)
            tokens = sum(r[1] for r in result['api']['outputs'])
            row = {'engine':backend,'concurrency':count,'direct':left,'api':right,
                   'output_tokens':tokens,'api_tokens_per_second':tokens/right['median_s'],
                   'direct_tokens_per_second':tokens/left['median_s'],
                   'overhead_percent':100*(right['median_s']/left['median_s']-1),
                   'per_request_latency':[summary([x[r]['elapsed_s'] for x in client_rows]) for r in range(count)],
                   'per_request_ttft':[summary([x[r]['ttft_s'] for x in client_rows if x[r]['ttft_s'] is not None])
                                       for r in range(count)],
                   'owner_metric_snapshots':snapshots,'oracle_passed':True,'reclaimed':True}
            records.append(row)
            print(f"{backend} B={count}: API={row['api_tokens_per_second']:.1f} tok/s, overhead={row['overhead_percent']:.2f}%",flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = ROOT/'results/service-finalize'/f'service-bench-{stamp}.json'
    verdict = all(r['overhead_percent']<5 for r in records if r['concurrency']==1)
    record = {'timestamp':stamp,'config':config,'config_hash':hashlib.sha256(
        json.dumps(config,sort_keys=True).encode()).hexdigest()[:12],
        'measurements':records,'singleton_api_overhead_below_5_percent':verdict}
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(f'written: {path}; singleton <5% target: {verdict}',flush=True)
    return 0 if verdict else 1


if __name__=='__main__':
    raise SystemExit(main())
