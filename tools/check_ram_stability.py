#!/usr/bin/env python3
"""Supervised HTTP memory test, not a speed benchmark. Requires a built release server."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import benchmark_http_session as http
import run_benchmark as common

ROOT = common.WORKSPACE_DIR


def pressure_reason(sample, args):
    return http.memory_pressure_reason(sample, args)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--testbed', choices=('i5', 'i7'), required=True)
    cli.add_argument('--gpu', choices=('cpu', 'auto'), required=True)
    cli.add_argument('--cache', type=common.nonnegative, required=True, help='MiB per layer')
    cli.add_argument('--model-dir', type=Path, default=Path('models/olmoe-1b-7b-int8'))
    cli.add_argument('--threads', type=common.positive, default=8)
    cli.add_argument('--context', type=common.positive, default=512)
    cli.add_argument('--cycles', type=common.positive, default=12, help='number of requests')
    cli.add_argument('--max-tokens', type=common.positive, default=32)
    cli.add_argument('--keep-cache', action='store_true', help='reset only before the first request')
    cli.add_argument('--ram-limit-mib', type=common.positive, default=5120)
    cli.add_argument('--headroom-mib', type=common.positive, default=1024)
    cli.add_argument('--stop-rss-mib', type=common.positive, default=5632)
    cli.add_argument('--min-available-mib', type=common.positive, default=1536)
    cli.add_argument('--stop-swap-mib', type=common.nonnegative, default=128)
    cli.add_argument('--output-root', type=Path, default=Path('benchmarks/runs'))
    cli.add_argument('--allow-dirty', action='store_true')
    cli.add_argument('--dry-run', action='store_true')
    return cli


def execute(args):
    if args.stop_rss_mib <= args.ram_limit_mib:
        raise ValueError('watchdog RSS threshold must exceed engine RSS limit')
    env = os.environ.copy()
    dirty = bool(common.checked_text(['git', 'status', '--porcelain'], ROOT, env))
    if dirty and not args.allow_dirty:
        raise ValueError('dirty tree; --allow-dirty is exploratory only')
    if args.dry_run:
        print(json.dumps({'kind': 'ram_stability', 'config': vars(args),
                          'notes': 'No build/load. Stop other inference before execution.'}, default=str, indent=2))
        return 0
    if not Path('/proc/self/status').exists():
        raise RuntimeError('memory supervision requires Linux /proc')
    binary = ROOT / 'target/release/moe-tier-engine'
    if not binary.exists():
        raise RuntimeError('build release server before running; no build during supervision')
    model = common.absolute(args.model_dir, ROOT)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-ram-' + uuid.uuid4().hex
    directory = common.absolute(args.output_root, ROOT) / args.testbed / run_id
    directory.mkdir(parents=True)
    record = {'kind': 'ram_stability', 'status': 'failed', 'run_id': run_id,
              'git_commit': common.checked_text(['git', 'rev-parse', 'HEAD'], ROOT, env),
              'git_dirty': dirty, 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'notes': ['bounded stability observation, not causal performance comparison or proof of no leaks',
                        'checkpoint hashing warms OS cache', 'watchdog terminates only its own child server'], 'turns': []}
    record['checkpoint'] = common.checkpoint_manifest(model)
    record['hardware_observed'] = common.hardware_observed(ROOT, env)
    record['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    with zipfile.ZipFile(directory / 'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [p for folder in ('src', 'tools', 'web') for p in (ROOT / folder).rglob('*')
                 if p.is_file() and p.suffix in ('.rs', '.py', '.js', '.cjs', '.html')]
        paths += [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
        record['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for p in paths:
            archive.write(p, str(p.relative_to(ROOT)))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    env.update(MOE_BIND=f'127.0.0.1:{port}', MOE_MODEL_DIR=str(model), RAYON_NUM_THREADS=str(args.threads),
               MOE_CACHE_BYTES_PER_LAYER=str(args.cache * 1024**2), MOE_CONTEXT=str(args.context),
               MOE_EXPERT_CACHE_POLICY='victim', MOE_RAM_LIMIT_BYTES=str(args.ram_limit_mib * 1024**2),
               MOE_RAM_HEADROOM_BYTES=str(args.headroom_mib * 1024**2))
    stop = threading.Event()
    violations = []
    print('Artifact: ' + str(directory), flush=True)
    with (directory / 'server.log').open('x') as log:
        server = subprocess.Popen([str(binary)], cwd=ROOT, env=env, stdout=log, stderr=log)

        def supervise():
            with (directory / 'memory.jsonl').open('x') as samples:
                while not stop.is_set():
                    if server.poll() is not None:
                        return
                    sample = {'elapsed_s': time.monotonic() - started, 'process': http.process_snapshot(server.pid),
                              'system': http.memory_snapshot()}
                    samples.write(json.dumps(sample) + '\n'); samples.flush()
                    reason = pressure_reason(sample, args)
                    if reason:
                        violations.append({'reason': reason, 'sample': sample})
                        server.terminate()
                        return
                    stop.wait(.25)
        started = time.monotonic()
        watcher = threading.Thread(target=supervise, daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('server exited during load; see server.log')
                try:
                    http.request(port, '/health', timeout=1); break
                except OSError:
                    time.sleep(.1)
            else:
                raise TimeoutError('server load timeout')
            status = json.loads(http.request(port, '/experts'))
            if args.gpu == 'cpu' and status['gpu']['enabled']:
                http.request(port, '/api/gpu/toggle', {})
                status = json.loads(http.request(port, '/experts'))
            if args.gpu == 'auto' and not status['gpu']['enabled']:
                raise RuntimeError('GPU requested but unavailable')
            if status['cache_bytes_per_layer'] != args.cache * 1024**2:
                raise RuntimeError('cache mismatch')
            if status.get('ram_rss_limit_bytes') != args.ram_limit_mib * 1024**2:
                raise RuntimeError('binary missing expected RAM guard; rebuild first')
            record['status_start'] = status
            history, session = [], ''
            prompts = ('Where is Mars', 'What planet did I just ask about?', 'Explain a Python dictionary.', 'What is love ?')
            for turn in range(args.cycles):
                phase = turn % 4
                if turn == 0 or (not args.keep_cache and phase in (0, 3)):
                    http.request(port, '/api/cache/reset', {})
                if phase != 1:
                    history, session = [], 'ram-' + uuid.uuid4().hex
                history.append({'role': 'user', 'content': prompts[phase]})
                response = http.stream_chat(port, history, session, args.max_tokens)
                status = json.loads(http.request(port, '/experts'))
                sample = {'turn': turn, 'messages': [dict(m) for m in history], 'session_id': session,
                          'response': response, 'status': status, 'process': http.process_snapshot(server.pid)}
                common.write_exclusive_json(directory / f'turn-{turn}.json', sample)
                record['turns'].append(sample)
                history.append({'role': 'assistant', 'content': response['text']})
                print(f"turn={turn} RSS={sample['process'].get('VmRSS')}KiB swap={sample['process'].get('VmSwap')}KiB", flush=True)
            record['status'] = 'completed'
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        finally:
            stop.set(); watcher.join(timeout=2)
            server.terminate()
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        record['watchdog_violations'] = violations
        if violations: record['status'] = 'watchdog_stopped'
    common.write_exclusive_json(directory / 'result.json', record)
    print(record['status'] + ': ' + str(directory / 'result.json'), flush=True)
    return 0 if record['status'] == 'completed' else 1


if __name__ == '__main__':
    try:
        sys.exit(execute(parser().parse_args()))
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
