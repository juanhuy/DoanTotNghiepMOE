import datetime
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_http_session as http
import run_benchmark as common

run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-stage1-smoke-' + uuid.uuid4().hex
out = ROOT / 'benchmarks/runs/i5' / run_id
out.mkdir(parents=True)
shutil.copyfile(__file__, out / 'smoke_runner.py')
env = dict(os.environ, RAYON_NUM_THREADS='8', MOE_CACHE_BYTES_PER_LAYER=str(128 * 1024**2), MOE_CONTEXT='512', MOE_DENSE_BYTES=str(3 * 1024**3), MOE_EXPERT_CACHE_POLICY='victim', MOE_MODEL_DIR=str((ROOT / 'models/olmoe-1b-7b-int8').resolve()))
record = {'status': 'failed', 'kind': 'model_http_smoke', 'testbed': 'i5', 'run_id': run_id,
          'git_commit': common.checked_text(['git', 'rev-parse', 'HEAD'], ROOT, env), 'git_dirty': True,
          'config': {key: env[key] for key in ('RAYON_NUM_THREADS', 'MOE_CACHE_BYTES_PER_LAYER', 'MOE_CONTEXT', 'MOE_DENSE_BYTES', 'MOE_EXPERT_CACHE_POLICY')},
          'notes': ['Exploratory protocol/quality smoke; not clean-revision baseline or causal performance comparison', 'Checkpoint hashing warms OS cache', 'one loaded server, no concurrent build/tests'], 'cases': []}
record['source_sha256'] = {}
with zipfile.ZipFile(out / 'source_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for folder, suffixes in [('src', {'.rs'}), ('tools', {'.py', '.cjs'}), ('web', {'.html', '.js'})]:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and path.suffix in suffixes:
                name = str(path.relative_to(ROOT))
                record['source_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.write(path, name)
    for name in ['Cargo.toml', 'Cargo.lock']:
        archive.write(ROOT / name, name)
        record['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
record['checkpoint'] = common.checkpoint_manifest(Path(env['MOE_MODEL_DIR']))
record['hardware_observed'] = common.hardware_observed(ROOT, env)
record['memory_start_kib'] = http.memory_snapshot()
binary = ROOT / 'target/release/moe-tier-engine'
record['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
env['MOE_BIND'] = f'127.0.0.1:{port}'
print('Artifact: ' + str(out), flush=True)
with (out / 'server.log').open('x') as log:
    server = subprocess.Popen([str(binary)], env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if server.poll() is not None: raise RuntimeError('server exited')
            try:
                http.request(port, '/health', timeout=1); break
            except OSError: time.sleep(.1)
        else: raise TimeoutError('readiness')
        record['status_start'] = json.loads(http.request(port, '/experts'))
        assert record['status_start']['gpu']['enabled'], 'GPU must be enabled'
        assert record['status_start']['expert_cache_policy'] == 'victim'
        asset = http.request(port, '/chat_stream.js').decode()
        assert 'consumeChatStream' in asset
        record['web_asset_served'] = True
        def reset():
            deadline = time.monotonic() + 60
            while True:
                try:
                    http.request(port, '/api/cache/reset', {}); return
                except RuntimeError as error:
                    if 'HTTP 429' not in str(error) or time.monotonic() > deadline: raise
                    time.sleep(.1)
        def chat(case_id, messages, session_id, limit):
            before = http.process_snapshot(server.pid)
            response = http.stream_chat(port, messages, session_id, limit)
            sample = {'id': case_id, 'messages': messages, 'max_tokens': limit, 'session_id': session_id,
                      'response': response, 'process_before': before, 'process_after': http.process_snapshot(server.pid)}
            common.write_exclusive_json(out / f'{case_id}.json', sample)
            record['cases'].append(sample)
            print(f"{case_id}: PASS protocol; wall={response['metrics']['total_ms']/1000:.2f}s", flush=True)
            return response
        reset()
        messages = [{'role': 'user', 'content': 'Where is Mars'}]
        mars = chat('mars', messages, 'stage1-conversation', 32)
        messages += [{'role': 'assistant', 'content': mars['text']}, {'role': 'user', 'content': 'What is love ?'}]
        chat('love', messages, 'stage1-conversation', 32)
        # Close a real TCP stream immediately after the first content event.
        conn = HTTPConnection('127.0.0.1', port, timeout=120)
        conn.request('POST', '/v1/chat/completions', json.dumps({'model':'olmoe', 'messages':[{'role':'user', 'content':'Explain the planets in detail.'}], 'session_id':'stage1-abort', 'max_tokens':128, 'stream':True}), {'Content-Type':'application/json'})
        response = conn.getresponse()
        assert response.status == 200
        saw_content = False
        for line in response:
            if line.startswith(b'data:'):
                event = json.loads(line[5:])
                if event.get('error'): raise RuntimeError(event['error'])
                if event.get('choices', [{}])[0].get('delta', {}).get('content'):
                    saw_content = True; break
        response.close(); conn.close()
        assert saw_content
        reset()  # 429 retries prove the worker eventually releases the model.
        record['abort_then_reset'] = 'pass'
        chat('after-abort-reset', [{'role':'user','content':'Where is Mars'}], 'stage1-after-abort', 16)
        reset()
        long_prompt = http.scenario_spec('long-context')['prompts'][0]
        chat('long-context', [{'role':'user','content':long_prompt}], 'stage1-long', 16)
        # Heuristic quality suite: record every answer/failure, never claim standardized quality.
        suite = ROOT / 'benchmarks/suites/quality-v1.json'
        record['quality_suite_sha256'] = hashlib.sha256(suite.read_bytes()).hexdigest()
        quality = []
        for case in json.loads(suite.read_text())['cases']:
            result = chat('quality-' + case['id'], case['messages'], 'quality-' + case['id'], case['max_tokens'])
            text = result['text'].strip()
            passed = (text == case['exact']) if 'exact' in case else all(word.casefold() in text.casefold() for word in case.get('contains_all', []))
            quality.append({'id':case['id'], 'heuristic_pass':passed})
            record['quality'] = quality
        record['status_end'] = json.loads(http.request(port, '/experts'))
        for layer in record['status_end']['layers']:
            assert layer['cache_bytes'] + layer['recycled_bytes'] <= 128 * 1024**2
        record['status'] = 'success'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        print(record['error'], flush=True)
    finally:
        server.terminate()
        try: server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
record['memory_end_kib'] = http.memory_snapshot()
common.write_exclusive_json(out / 'result.json', record)
print(record['status'] + ': ' + str(out / 'result.json'), flush=True)
sys.exit(0 if record['status'] == 'success' else 1)
