"""Read-only Linux reclaim telemetry. No process names, paths or exception text in output."""
import math
import os
from pathlib import Path, PurePosixPath
import re


STAT_KEYS = {'anon', 'file', 'shmem', 'unevictable', 'pgfault', 'pgmajfault',
             'pgscan', 'pgsteal', 'pgscan_kswapd', 'pgscan_direct',
             'pgsteal_kswapd', 'pgsteal_direct', 'workingset_refault_anon',
             'workingset_refault_file'}
CGROUP_FILES = ('memory.current', 'memory.max', 'memory.high',
                'memory.swap.current', 'memory.swap.max', 'memory.events',
                'memory.events.local', 'memory.stat', 'memory.pressure')


def unavailable(reason):
    return {'status': 'unavailable', 'reason': reason}


def read_probe(path, parse):
    try:
        return {'status': 'ok', 'values': parse(path.read_text())}
    except FileNotFoundError:
        return unavailable('not_found')
    except PermissionError:
        return unavailable('permission_denied')
    except OSError:
        return unavailable('read_error')
    except (ValueError, OverflowError):
        return unavailable('invalid_data')


def counters(raw):
    result = {}
    for line in raw.splitlines():
        key, value = line.split()
        number = int(value)
        if number < 0 or key in result:
            raise ValueError('invalid counter')
        result[key] = number
    if not result:
        raise ValueError('empty counters')
    return result


def pressure(raw):
    result = {}
    for line in raw.splitlines():
        kind, *fields = line.split()
        if kind not in ('some', 'full') or kind in result:
            raise ValueError('invalid PSI kind')
        parsed = {}
        for field in fields:
            key, value = field.split('=')
            if key not in ('avg10', 'avg60', 'avg300', 'total') or key in parsed:
                raise ValueError('invalid PSI field')
            number = int(value) if key == 'total' else float(value)
            if number < 0 or (key != 'total' and (not math.isfinite(number) or number > 100)):
                raise ValueError('invalid PSI value')
            parsed[key] = number
        if set(parsed) != {'avg10', 'avg60', 'avg300', 'total'}:
            raise ValueError('missing PSI field')
        result[kind] = parsed
    if 'some' not in result:
        raise ValueError('missing PSI some')
    return result


def scalar(raw, allow_max=False):
    value = raw.strip()
    if allow_max and value == 'max':
        return value
    number = int(value)
    if number < 0:
        raise ValueError('negative memory value')
    return number


def absolute_path(raw):
    path = PurePosixPath(raw)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('path outside visible hierarchy')
    return path


def membership(raw):
    matches = [line[3:] for line in raw.splitlines() if line.startswith('0::')]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('cgroup v2 membership unavailable')
    return absolute_path(matches[0])


def unescape_mount(raw):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), raw)


def cgroup_location(member, mountinfo):
    candidates = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(' - ')
        if not separator or not right.split() or right.split()[0] != 'cgroup2':
            continue
        fields = left.split()
        if len(fields) < 6:
            raise ValueError('invalid mountinfo')
        root = absolute_path(unescape_mount(fields[3]))
        mount = Path(absolute_path(unescape_mount(fields[4])))
        if member.is_relative_to(root):
            candidates.append((len(root.parts), mount / member.relative_to(root), mount))
    if not candidates:
        raise ValueError('no matching visible cgroup2 mount')
    _, node, mount = max(candidates, key=lambda item: item[0])
    return node, mount


def cgroup_snapshot(pid, proc_root=Path('/proc')):
    member_file = proc_root / str(pid) / 'cgroup'
    # /proc/PID/cgroup and this mountinfo are both interpreted in the observer's
    # namespace. Do not combine membership with a different process's mount tree.
    location = read_probe(member_file, membership)
    if location['status'] != 'ok':
        return location
    member = location['values']
    if member is None:
        return unavailable('cgroup_v2_not_visible')
    mountinfo = read_probe(proc_root / 'self/mountinfo', lambda raw: cgroup_location(member, raw))
    if mountinfo['status'] != 'ok':
        return mountinfo
    node, mount = mountinfo['values']
    ancestors = []
    while True:
        try:
            identity = node.stat()
        except OSError:
            return unavailable('cgroup_directory_unavailable')
        files = {}
        for name in CGROUP_FILES:
            parse = pressure if name == 'memory.pressure' else counters if name in (
                'memory.events', 'memory.events.local', 'memory.stat') else (
                    lambda raw: scalar(raw, name in ('memory.max', 'memory.high', 'memory.swap.max')))
            probe = read_probe(node / name, parse)
            if name == 'memory.stat' and probe['status'] == 'ok':
                probe['values'] = {k: v for k, v in probe['values'].items()
                                   if k in STAT_KEYS or k.startswith(
                                       ('pgscan', 'pgsteal', 'workingset_refault_'))}
                if not probe['values']:
                    probe = unavailable('selected_counters_unavailable')
            files[name] = probe
        ancestors.append({'ancestor_level': len(ancestors),
                          'identity': {'device': identity.st_dev, 'inode': identity.st_ino},
                          'files': files})
        if node == mount:
            break
        node = node.parent
    end = read_probe(member_file, membership)
    if end['status'] != 'ok' or end['values'] != member:
        return unavailable('membership_changed_or_exited')
    statuses = [probe['status'] for ancestor in ancestors for probe in ancestor['files'].values()]
    status = 'ok' if all(s == 'ok' for s in statuses) else (
        'partial' if 'ok' in statuses else 'unavailable')
    return {'status': status, 'scope': 'child_and_mounted_ancestors_only', 'ancestors': ancestors}


def reclaim_snapshot(pid, proc_root=Path('/proc')):
    vmstat = read_probe(proc_root / 'vmstat', counters)
    if vmstat['status'] == 'ok':
        values = vmstat['values']
        vmstat['values'] = {k: v for k, v in values.items()
                            if k in ('pswpin', 'pswpout') or k.startswith(
                                ('pgscan_', 'pgsteal_', 'allocstall_'))}
        if not all(k in values for k in ('pswpin', 'pswpout')):
            vmstat['status'] = 'partial'
            vmstat['reason'] = 'swap_counters_missing'
    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, OSError, ValueError):
        page_size = None
    return {'schema_version': 1, 'page_size_bytes': page_size,
            'vmstat': vmstat,
            'psi_memory': read_probe(proc_root / 'pressure/memory', pressure),
            'cgroup_v2': cgroup_snapshot(pid, proc_root)}
