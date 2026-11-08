"""Paired decode lookahead protocol; invoked only through run_benchmark.py."""
import copy
import hashlib
import math
import statistics

import run_benchmark as common


def validate_protocol(args):
    if args.http_scenario != 'topics' or args.compare_cache_policies or args.suite or args.case:
        raise ValueError('--compare-lookahead requires topics, no policy comparison or suite')
    if args.gpu != 'cpu':
        raise ValueError('lookahead-v2 requires CPU mode; GPU needs a fixed VRAM budget protocol')
    if args.expert_cache_policy != 'victim':
        raise ValueError('lookahead-v2 fixes victim policy')


def block_plan(pairs):
    return [{'pair': pair, 'mode': mode} for pair in range(pairs)
            for mode in (('off', 'on') if pair % 2 == 0 else ('on', 'off'))]


def source_fingerprint(root):
    # All tracked/untracked inference/tooling inputs, plus build manifests and fixtures.
    # Avoid including benchmark output and documentation edited by a concurrent IDE.
    paths = [root / name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs')]
    for name in ('src', 'tools', 'tests/fixtures'):
        paths.extend(p for p in (root / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def validate_gate_counters(counts, enabled):
    fields = ('evaluated_layers', 'no_next_layer', 'blocked_by_misses', 'eligible_layers',
              'disabled_eligible_layers', 'prediction_calls', 'no_positive_score_calls',
              'all_candidates_cached_calls', 'positive_candidates', 'candidates_in_ram',
              'candidates_in_vram', 'candidates_available')
    if any(type(counts.get(k)) is not int or counts[k] < 0 for k in fields):
        raise ValueError('missing/invalid lookahead gate counters')
    histogram = counts.get('miss_histogram')
    if not isinstance(histogram, list) or any(type(v) is not int or v < 0 for v in histogram):
        raise ValueError('invalid lookahead miss histogram')
    if (counts['evaluated_layers'] != counts['no_next_layer'] + counts['blocked_by_misses'] + counts['eligible_layers']
            or sum(histogram) != counts['blocked_by_misses'] + counts['eligible_layers']
            or (histogram[0] if histogram else 0) != counts['eligible_layers']):
        raise ValueError('lookahead gate accounting mismatch')
    if counts['disabled_eligible_layers'] != (0 if enabled else counts['eligible_layers']):
        raise ValueError('lookahead gate mode mismatch')
    if counts['prediction_calls'] != counts['eligible_layers'] - counts['disabled_eligible_layers']:
        raise ValueError('lookahead prediction-call accounting mismatch')
    if counts['positive_candidates'] != sum(counts[k] for k in ('candidates_in_ram', 'candidates_in_vram', 'candidates_available')):
        raise ValueError('lookahead candidate accounting mismatch')
    if counts['no_positive_score_calls'] + counts['all_candidates_cached_calls'] > counts['prediction_calls']:
        raise ValueError('lookahead no-candidate reason mismatch')
    if counts['prediction_calls'] == 0 and any(counts[k] for k in fields[6:]):
        raise ValueError('candidate evaluation reported without predictor call')


def validate_sample(sample, enabled, status, cache_mib):
    m = sample['response']['metrics']
    if m.get('lookahead_enabled') is not enabled:
        raise ValueError('request lookahead mode mismatch')
    counts = m.get('lookahead', {})
    validate_gate_counters(counts, enabled)
    fields = ('eligible_layers', 'proposed', 'loaded', 'load_errors', 'used', 'unused',
              'abandoned', 'pending', 'file_bytes', 'buffer_peak_bytes')
    if any(type(counts.get(k)) is not int or counts[k] < 0 for k in fields):
        raise ValueError('missing/invalid lookahead counters')
    if counts['proposed'] != counts['loaded'] + counts['load_errors']:
        raise ValueError('lookahead proposal accounting mismatch')
    if counts['loaded'] != sum(counts[k] for k in ('used', 'unused', 'abandoned', 'pending')):
        raise ValueError('lookahead lifecycle accounting mismatch')
    if counts['pending'] or counts['abandoned'] or counts['load_errors']:
        raise ValueError('unfinished/failed speculative work in completed request')
    if not enabled and any(counts[k] for k in fields if k != 'eligible_layers'):
        raise ValueError('disabled mode performed speculative work')
    for key in ('decode_ms', 'total_ms', 'time_to_first_token_ms', 'decode_tokens_per_second'):
        value = m.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid timing: ' + key)
    if m.get('cached_prompt_tokens') != 0:
        raise ValueError('independent workload unexpectedly reused prefix KV')
    active_limit = status['expert_request_payload_limit_bytes']
    host_limit = status['num_layers'] * cache_mib * 1024**2 + active_limit
    for key, limit in (('decode_expert_active_peak_bytes', active_limit),
                       ('decode_expert_host_peak_bytes', host_limit)):
        value = m.get(key)
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError('expert capacity reserve exceeded: ' + key)
    if counts['buffer_peak_bytes'] > m['decode_expert_active_peak_bytes']:
        raise ValueError('speculative buffers missing from active capacity')
    return counts['loaded']


def compare_pair(off, on):
    if len(off) != len(on) or not off:
        raise ValueError('incomplete paired block')
    rows = []
    for a, b in zip(off, on):
        ra, rb = a['response'], b['response']
        if (a['messages'] != b['messages'] or ra['usage'] != rb['usage'] or
                ra['text'] != rb['text'] or ra['finish_reason'] != rb['finish_reason']):
            raise ValueError('paired workload/output mismatch; inspect preserved block files')
        deltas = {}
        for key in ('decode_ms', 'total_ms', 'time_to_first_token_ms', 'decode_tokens_per_second'):
            av, bv = ra['metrics'][key], rb['metrics'][key]
            deltas[key] = {'off': av, 'on': bv, 'on_minus_off': bv - av,
                           'change_pct': (bv / av - 1) * 100 if av > 0 else None}
        rows.append({'prompt': a['prompt'], 'turn': a['turn'], 'metrics': deltas})
    return rows


def measure_paired(binary, model, args, env, directory, root, measure):
    validate_protocol(args)
    from benchmark_http_session import scenario_spec
    expected_prompts = scenario_spec("topics")["prompts"]
    fingerprint = source_fingerprint(root)
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    blocks, pending, pairs = [], {}, []
    loaded = 0
    def checkpoint_state():
        return {str(p.relative_to(model)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in model.rglob('*') if p.is_file()}
    checkpoint_before = checkpoint_state()
    for index, entry in enumerate(block_plan(args.runs)):
        if checkpoint_state() != checkpoint_before:
            raise ValueError('checkpoint metadata drift during paired experiment')
        if source_fingerprint(root) != fingerprint or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash:
            raise ValueError('source/binary drift during paired experiment')
        block = directory / f"block-{index:02d}-{entry['mode']}"
        block.mkdir()
        block_args = copy.copy(args)
        block_args.runs = 1
        block_args.compare_lookahead = False
        block_args.lookahead = entry['mode']
        result = measure(binary, model, block_args, env.copy(), block)
        common.write_exclusive_json(block / 'result.json', {'protocol': 'lookahead-v2', **entry, 'result': result})
        if [r['prompt'] for r in result['runs']] != expected_prompts or result['warmups']:
            raise ValueError('incomplete or unexpected topics block')
        status = result['status_start']
        for state in (status, result['status_end']):
            if state.get('gpu_mode_at_load') != 'cpu':
                raise ValueError('CPU was not selected before load')
            if state['cache_bytes_per_layer'] != args.cache * 1024**2:
                raise ValueError('cache budget changed')
            if state['gpu']['enabled'] != (args.gpu == 'auto'):
                raise ValueError('backend changed during block')
            if state['lookahead_enabled'] is not (entry['mode'] == 'on'):
                raise ValueError('lookahead changed during block')
            if any(layer['cache_bytes'] + layer['recycled_bytes'] > args.cache * 1024**2 for layer in state['layers']):
                raise ValueError('retained cache/pool budget exceeded')
        for sample in result['runs']:
            loaded += validate_sample(sample, entry['mode'] == 'on', status, args.cache)
        blocks.append({**entry, 'path': str(block.relative_to(directory))})
        pending[entry['mode']] = result['runs']
        if index % 2 == 1:
            rows = compare_pair(pending['off'], pending['on'])
            pairs.append({'pair': entry['pair'], 'rows': rows})
            common.write_exclusive_json(directory / f"pair-{entry['pair']}.json", pairs[-1])
            pending.clear()
    if checkpoint_state() != checkpoint_before:
        raise ValueError('checkpoint metadata drift at completion')
    if source_fingerprint(root) != fingerprint or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash:
        raise ValueError('source/binary drift at completion')
    summary = []
    for turn in range(len(pairs[0]['rows'])):
        metrics = {}
        for key in pairs[0]['rows'][turn]['metrics']:
            changes = [p['rows'][turn]['metrics'][key]['change_pct'] for p in pairs]
            changes = [v for v in changes if v is not None]
            metrics[key] = {'paired_change_pct_median': statistics.median(changes) if changes else None,
                            'min': min(changes) if changes else None, 'max': max(changes) if changes else None}
        summary.append({'prompt': pairs[0]['rows'][turn]['prompt'], 'metrics': metrics})
    return {'protocol': 'lookahead-v2', 'blocks': blocks, 'pairs': pairs, 'summary': summary,
            'activation': 'observed' if loaded else 'no_speculative_loads',
            'limits': ['HTTP text/usage parity, not full-model logits parity',
                       'sampled decode expert capacity, not hard RSS cap',
                       'CPU selected before CUDA initialization; OS cache uncontrolled']}
