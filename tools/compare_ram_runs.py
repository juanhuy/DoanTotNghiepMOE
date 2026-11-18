#!/usr/bin/env python3
"""ABBA comparison of check_ram_stability runs (qwen3-cache-ab-v1).
Blocks: A0 B1 B2 A3. Pair turn t of A0 with B1 and of A3 with B2 -> 2 x turns comparisons.
Rule (pre-registered): B/A total_ms and decode_ms ratio <= 1/1.05 in every comparison => B faster;
>= 1.05 in every comparison => B slower; otherwise no consistent difference. Text must match.
Usage: compare_ram_runs.py A0 B1 B2 A3 [--out summary.json]
"""
import argparse
import json
import statistics
from pathlib import Path


def load(path):
    record = json.loads((Path(path) / 'result.json').read_text())
    if record['status'] != 'completed':
        raise SystemExit(f'{path}: status {record["status"]}')
    return record


def turn_metrics(record):
    rows = []
    for turn in record['turns']:
        m = turn['response']['metrics']
        rows.append({'text': turn['response']['text'], 'total_ms': m['total_ms'], 'decode_ms': m['decode_ms'],
                     'ttft_ms': m['time_to_first_token_ms'], 'decode_tps': m['decode_tokens_per_second'],
                     'hits': m['expert_cache_hits'], 'misses': m['expert_cache_misses'],
                     'file_gb': m['expert_file_bytes_read'] / 1e9, 'io_ms': m['expert_io_ms']})
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('blocks', nargs=4)
    cli.add_argument('--out', type=Path)
    args = cli.parse_args()
    records = [load(p) for p in args.blocks]
    rows = [turn_metrics(r) for r in records]
    if len({len(r) for r in rows}) != 1:
        raise SystemExit('blocks have different turn counts')
    comparisons = []
    for a, b in ((0, 1), (3, 2)):
        for t, (ra, rb) in enumerate(zip(rows[a], rows[b])):
            comparisons.append({'pair': f'A{a}-B{b}', 'turn': t,
                                'total_ratio': rb['total_ms'] / ra['total_ms'],
                                'decode_ratio': rb['decode_ms'] / ra['decode_ms'],
                                'text_match': ra['text'] == rb['text']})
    faster = all(c['total_ratio'] <= 1 / 1.05 and c['decode_ratio'] <= 1 / 1.05 for c in comparisons)
    slower = all(c['total_ratio'] >= 1.05 and c['decode_ratio'] >= 1.05 for c in comparisons)

    def mode(indices, key, fn=statistics.median):
        return fn([row[key] for i in indices for row in rows[i]])
    summary = {
        'protocol': 'qwen3-cache-ab-v1',
        'blocks': [str(p) for p in args.blocks],
        'cache_mib': [r['config']['cache'] for r in records],
        'b_faster': faster, 'b_slower': slower,
        'text_match_all': all(c['text_match'] for c in comparisons),
        'total_ratio_range': [min(c['total_ratio'] for c in comparisons), max(c['total_ratio'] for c in comparisons)],
        'decode_ratio_range': [min(c['decode_ratio'] for c in comparisons), max(c['decode_ratio'] for c in comparisons)],
        'median': {name: {k: mode(ix, k) for k in ('decode_tps', 'ttft_ms', 'total_ms', 'file_gb', 'io_ms')}
                   for name, ix in (('A', (0, 3)), ('B', (1, 2)))},
        'hit_rate': {name: [rr['hits'] / (rr['hits'] + rr['misses']) for i in ix for rr in rows[i]]
                     for name, ix in (('A', (0, 3)), ('B', (1, 2)))},
        'memory_limit': [{k: r.get('memory_limit', {}).get(k) for k in
                          ('memory_max_mib', 'peak_current_mib', 'peak_anon_mib', 'peak_file_mib', 'oom_kill_events',
                           'checkpoint_resident_bytes_after_evict')} for r in records],
        'comparisons': comparisons,
    }
    text = json.dumps(summary, indent=1)
    if args.out:
        args.out.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
