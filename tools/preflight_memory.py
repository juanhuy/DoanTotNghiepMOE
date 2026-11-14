#!/usr/bin/env python3
"""Pre-measurement preflight: 3 samples 5s apart, no other engine/build, swap-out stable, on AC, GPU not throttled.

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

POWER_SUPPLY_ROOT = Path('/sys/class/power_supply')
# Clock-event reasons that mean the GPU is being slowed by heat/power policy (not idle/app clocks).
GPU_THROTTLE_FIELDS = ('clocks_event_reasons.hw_slowdown', 'clocks_event_reasons.hw_thermal_slowdown',
                       'clocks_event_reasons.sw_thermal_slowdown', 'clocks_event_reasons.hw_power_brake_slowdown',
                       'clocks_event_reasons.sw_power_cap')
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


def read_power(root=POWER_SUPPLY_ROOT):
    """AC/battery state from sysfs. A machine without a Mains supply (desktop) counts as on AC."""
    supplies = {}
    for entry in sorted(root.glob('*')) if root.exists() else []:
        def text(name):
            try:
                return (entry / name).read_text().strip()
            except OSError:
                return None
        supplies[entry.name] = {'type': text('type'), 'online': text('online'), 'status': text('status'),
                                'capacity': text('capacity')}
    mains = [v for v in supplies.values() if v['type'] == 'Mains']
    return {'supplies': supplies, 'on_ac': (not mains) or any(v['online'] == '1' for v in mains)}


def read_gpu_throttle():
    """Throttle reasons from nvidia-smi; absent GPU/tool is recorded, not treated as throttled."""
    query = ['nvidia-smi', '--query-gpu=pstate,clocks.gr,temperature.gpu,power.draw,' + ','.join(GPU_THROTTLE_FIELDS),
             '--format=csv,noheader']
    try:
        out = subprocess.run(query, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'absent', 'detail': str(exc), 'throttled': False}
    if out.returncode != 0 or not out.stdout.strip():
        return {'status': 'absent', 'detail': out.stderr.strip(), 'throttled': False}
    return parse_gpu_throttle(out.stdout.strip().splitlines()[0])


def parse_gpu_throttle(line):
    values = [v.strip() for v in line.split(',')]
    reasons = dict(zip(GPU_THROTTLE_FIELDS, values[4:]))
    active = [k for k, v in reasons.items() if v == 'Active']
    return {'status': 'observed', 'pstate': values[0], 'clock_gr': values[1], 'temperature': values[2],
            'power_draw': values[3], 'reasons': reasons, 'active_throttle': active, 'throttled': bool(active)}


def active_processes(ps_lines, own=('preflight_memory.py',)):
    return [line.strip() for line in ps_lines
            if any(marker in line for marker in ACTIVE_MARKERS) and not any(o in line for o in own)]


def eligible(samples, active, min_available_mib, power=None, gpu=None):
    if power is not None and not power['on_ac']:
        return False
    if gpu is not None and gpu['throttled']:
        return False
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
    power, gpu = read_power(), read_gpu_throttle()
    ok = eligible(samples, active, args.min_available_mib, power, gpu)
    record = {'samples': samples, 'active_engine_build_processes': active, 'power': power, 'gpu': gpu,
              'min_available_required_mib': args.min_available_mib, 'eligible': ok}
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'min_available_mib': min(s['available_kib'] for s in samples) / 1024,
                      'pswpout': [s['pswpout'] for s in samples], 'active': active, 'on_ac': power['on_ac'],
                      'gpu_throttle': gpu.get('active_throttle', gpu['status']), 'eligible': ok}))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
