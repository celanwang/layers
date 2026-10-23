"""Real process accepts SSE work, drains on SIGTERM and flushes its final frame."""
import http.client
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import tempfile


def test_l7_cli_sigterm_finishes_live_stream_and_exits_cleanly():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryFile(mode='w+t') as errors:
        proc = subprocess.Popen([sys.executable,'server/app.py','--port','0','--engine','l7',
                                 '--drain-timeout','5'], cwd=root, stdout=subprocess.PIPE,
                                stderr=errors,text=True)
        conn = None
        try:
            assert select.select([proc.stdout],[],[],20)[0], 'server startup timed out'
            startup = proc.stdout.readline()
            port = int(re.search(r'127\.0\.0\.1:(\d+)',startup).group(1))
            conn = http.client.HTTPConnection('127.0.0.1',port,timeout=15)
            body = {'model':'layers-gpt','messages':[{'role':'user','content':'ROMEO: '}],
                    'stream':True,'stream_options':{'include_usage':True},
                    'temperature':0,'max_tokens':180}
            conn.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
            response = conn.getresponse()
            assert response.status == 200
            frames, signalled, done = [],False,False
            for line in response:
                if line == b'\n': continue
                if line == b'data: [DONE]\n':
                    done = True
                    break
                frame = json.loads(line[6:])
                frames.append(frame)
                if not signalled and any(c['delta'].get('content') for c in frame['choices']):
                    proc.send_signal(signal.SIGTERM)
                    signalled = True
            assert done and signalled
            assert frames[-1]['usage']['completion_tokens'] > 0
            assert frames[-2]['choices'][0]['finish_reason'] in ('stop','length')
            assert proc.wait(15) == 0
            errors.seek(0)
            logs = [json.loads(line) for line in errors.read().splitlines() if line.startswith('{')]
            assert logs[-1]['outcome'] == 'completed'
            result = {'startup':startup.strip(),'done_frame_received':True,'exit_code':proc.returncode,
                      'usage':frames[-1]['usage'],'request_log':logs[-1]}
            (root/'results/service-finalize/cli-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
        finally:
            if conn is not None: conn.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
