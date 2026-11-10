#!/usr/bin/env python3
"""Pre-measurement memory preflight: 3 samples 5s apart, no other engine/build, swap-out stable.

Writes the samples and eligibility to a JSON file; exit code 0 only when eligible.
Not a benchmark; it only decides whether a protocol may start.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ACTIVE_MARKERS = ('moe-tier-engine', 'cargo ', 'rustc ', 'run_benchmark', 'check_ram_stability', 'stress_http')


def sample():
    meminfo = {line.split(':')[0]: int(line.split()[1]) for line in open('/proc/meminfo')}
    vmstat = {line.split()[0]: int(line.split()[1]) for line in open('/proc/vmstat')}
    try:
        psi = open('/proc/pressure/memory').read().split('\n')[:2]
    except OSError:
        psi = None
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'available_kib': meminfo['MemAvailable'],
            'swap_used_kib': meminfo['SwapTotal'] - meminfo['SwapFree'],
            'pswpout': vmstat.get('pswpout'), 'psi_memory': psi}


def active_processes(ps_lines, own=('preflight_memory.py',)):
    return [line.strip() for line in ps_lines
            if any(marker in line for marker in ACTIVE_MARKERS) and not any(o in line for o in own)]


def eligible(samples, active, min_available_mib):
    swap_stable = samples[0]['pswpout'] is not None and samples[0]['pswpout'] == samples[-1]['pswpout']
    min_available = min(s['available_kib'] for s in samples)
    return (not active) and swap_stable and min_available >= min_available_mib * 1024


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--min-available-mib', type=int, default=5120)
    cli.add_argument('--samples', type=int, default=3)
    cli.add_argument('--interval-s', type=float, default=5.0)
    args = cli.parse_args(argv)
    samples = []
    for index in range(args.samples):
        samples.append(sample())
        if index < args.samples - 1:
            time.sleep(args.interval_s)
    ps = subprocess.run(['ps', '-eo', 'pid,comm,args'], capture_output=True, text=True).stdout.splitlines()
    active = active_processes(ps)
    ok = eligible(samples, active, args.min_available_mib)
    record = {'samples': samples, 'active_engine_build_processes': active,
              'min_available_required_mib': args.min_available_mib, 'eligible': ok}
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'min_available_mib': min(s['available_kib'] for s in samples) / 1024,
                      'pswpout': [s['pswpout'] for s in samples], 'active': active, 'eligible': ok}))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
