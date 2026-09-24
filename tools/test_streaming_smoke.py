"""Real HTTP smoke test. Starts/stops only its own local release server.
Usage: python3 tools/test_streaming_smoke.py MODEL_DIR OUTPUT_JSON
"""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def main():
    model, output = sys.argv[1:]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, MOE_MODEL_DIR=model, MOE_BIND=f'127.0.0.1:{port}', MOE_CONTEXT='512', MOE_CACHE_BYTES_PER_LAYER='134217728')
    with tempfile.TemporaryFile() as log:
        server = subprocess.Popen(['target/release/moe-tier-engine'], env=env, stdout=log, stderr=log)
        def request(body=None, path='/v1/chat/completions'):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
            conn.request('GET' if body is None else 'POST', path, body=None if body is None else json.dumps(body), headers={'Content-Type':'application/json'})
            return conn, conn.getresponse()
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read().decode())
                try:
                    conn, resp = request(path='/health')
                    assert resp.status == 200
                    resp.read(); conn.close(); break
                except OSError:
                    if time.monotonic() > deadline: raise
                    time.sleep(0.1)
            # Receive role, verify busy/health, disconnect during prefill.
            body = {'model':'olmoe','messages':[{'role':'user','content':'What is 2 + 2?'}],'max_tokens':32,'stream':True}
            conn, resp = request(body)
            assert resp.status == 200
            while True:
                line = resp.readline()
                assert line, 'stream closed before first event'
                if line.startswith(b'data: '): break
            c, r = request(dict(body, stream=False)); assert r.status == 429; r.read(); c.close()
            c, r = request(path='/health'); assert r.status == 200; r.read(); c.close()
            started = time.monotonic()
            resp.close(); conn.close()
            # Invalid request becomes 400 only after the previous permit is released.
            while True:
                c, r = request(dict(body, max_tokens=0)); status = r.status; r.read(); c.close()
                if status == 400: break
                assert status == 429
                if time.monotonic()-started > 10: raise AssertionError('cancellation timed out')
                time.sleep(0.01)
            cancellation_ms = (time.monotonic()-started)*1000
            reports = []
            for prompt in ['What is 2 + 2?', 'Hãy viết đúng hai từ: Việt Nam']:
                body['messages'][0]['content'] = prompt
                started = time.monotonic()
                c, r = request(body)
                assert r.status == 200 and r.getheader('Content-Type').startswith('text/event-stream')
                chunks, final, first_content_ms, done = [], None, None, False
                while True:
                    line = r.readline()
                    if not line: break
                    if not line.startswith(b'data: '): continue
                    data = line[6:].strip()
                    if data == b'[DONE]': done = True; break
                    event = json.loads(data)
                    assert 'error' not in event, event
                    choice = event['choices'][0]
                    delta = choice['delta'].get('content', '')
                    if delta and first_content_ms is None:
                        first_content_ms = (time.monotonic()-started)*1000
                        # Health must still be available while reading tokens.
                        hc, hr = request(path='/health'); assert hr.status == 200; hr.read(); hc.close()
                    chunks.append(delta)
                    if choice['finish_reason'] is not None: final = event
                total_ms = (time.monotonic()-started)*1000
                r.close(); c.close()
                assert done and final is not None
                c, r = request(dict(body, stream=False)); assert r.status == 200
                regular = json.loads(r.read()); c.close()
                text = ''.join(chunks)
                assert text == regular['choices'][0]['message']['content'], (text, regular)
                assert final['usage'] == regular['usage']
                assert final['choices'][0]['finish_reason'] == regular['choices'][0]['finish_reason']
                reports.append({'prompt':prompt,'text':text,'content_chunks':sum(bool(x) for x in chunks),'first_content_ms':first_content_ms,'stream_total_ms':total_ms,'usage':final['usage'],'finish_reason':final['choices'][0]['finish_reason']})
            Path(output).write_text(json.dumps({'model_directory':model,'cancellation_observed_ms':cancellation_ms,'runs':reports,'checks':['HTTP 429 while busy','health during prefill and streaming','disconnect during prefill releases permit','invalid request returns 400 before SSE','stream text/usage/finish match non-stream','terminal DONE received'],'notes':['single smoke run, not a throughput benchmark','cancellation observation includes HTTP round trips; not a guaranteed latency bound']}, ensure_ascii=False, indent=2))
        finally:
            server.terminate()
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired: server.kill(); server.wait()


if __name__ == '__main__':
    main()
