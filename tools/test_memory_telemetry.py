#!/usr/bin/env python3
"""Synthetic proc/cgroup fixtures; no model, GPU, sockets or OS configuration changes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import benchmark_http_session as http
import memory_telemetry as telemetry


PSI = 'some avg10=1.25 avg60=0.50 avg300=0.10 total=12345\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'


class ReclaimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / 'proc'
        self.mount = self.root / 'cgroup space'
        self.member = self.mount / 'private-team/child'
        self.member.mkdir(parents=True)
        for directory in ('self', '123', 'pressure'):
            (self.proc / directory).mkdir(parents=True)
        (self.proc / '123/cgroup').write_text('0::/private-team/child\n')
        (self.proc / 'self/cgroup').write_text('0::/observer-only\n')
        self.mountinfo()
        (self.proc / 'vmstat').write_text('pswpin 0\npswpout 17\npgscan_direct 3\npgsteal_kswapd 2\npgfault 999\n')
        (self.proc / 'pressure/memory').write_text(PSI)
        for node in (self.member, self.member.parent, self.mount):
            self.populate(node)

    def mountinfo(self, hierarchy_root='/'):
        escaped = str(self.mount).replace(' ', r'\040')
        (self.proc / 'self/mountinfo').write_text(
            f'10 1 0:1 {hierarchy_root} {escaped} rw - cgroup2 cgroup rw\n')

    def populate(self, node):
        for name in ('memory.current', 'memory.swap.current'):
            (node / name).write_text('0\n')
        for name in ('memory.max', 'memory.high', 'memory.swap.max'):
            (node / name).write_text('max\n')
        for name in ('memory.events', 'memory.events.local'):
            (node / name).write_text('high 0\noom 0\noom_kill 0\n')
        (node / 'memory.stat').write_text('anon 4096\npgscan 9\npgsteal 7\npgscan_proactive 1\nfile 0\nunused_counter 42\n')
        (node / 'memory.pressure').write_text(PSI)

    def sample(self):
        return telemetry.reclaim_snapshot(123, self.proc)

    def test_child_membership_escaped_mount_ancestors_units_and_privacy(self):
        sample = self.sample()
        self.assertEqual(sample['vmstat']['values'], {
            'pswpin': 0, 'pswpout': 17, 'pgscan_direct': 3, 'pgsteal_kswapd': 2})
        self.assertGreater(sample['page_size_bytes'], 0)
        self.assertEqual(sample['psi_memory']['values']['some']['total'], 12345)
        self.assertEqual(sample['psi_memory']['values']['some']['avg10'], 1.25)
        cg = sample['cgroup_v2']
        self.assertEqual(cg['status'], 'ok')
        self.assertEqual([x['ancestor_level'] for x in cg['ancestors']], [0, 1, 2])
        self.assertEqual(cg['ancestors'][0]['identity']['inode'], self.member.stat().st_ino)
        self.assertEqual(cg['ancestors'][0]['files']['memory.current']['values'], 0)
        self.assertEqual(cg['ancestors'][0]['files']['memory.max']['values'], 'max')
        self.assertNotIn('unused_counter', cg['ancestors'][0]['files']['memory.stat']['values'])
        self.assertEqual(cg['ancestors'][0]['files']['memory.stat']['values']['pgscan_proactive'], 1)
        serialized = json.dumps(sample)
        for private in (str(self.root), 'private-team', 'observer-only', '/proc/'):
            self.assertNotIn(private, serialized)

    def test_subtree_mount_stops_at_visible_root_and_records_finite_limits(self):
        self.mountinfo('/hidden')
        (self.proc / '123/cgroup').write_text('0::/hidden/private-team/child\n')
        (self.member.parent / 'memory.high').write_text('1048576\n')
        cg = self.sample()['cgroup_v2']
        self.assertEqual(len(cg['ancestors']), 3)
        self.assertEqual(cg['ancestors'][1]['files']['memory.high']['values'], 1048576)
        self.assertEqual(cg['scope'], 'child_and_mounted_ancestors_only')

    def test_missing_controller_or_file_is_not_zero(self):
        (self.member / 'memory.swap.current').unlink()
        probe = self.sample()['cgroup_v2']
        self.assertEqual(probe['status'], 'partial')
        missing = probe['ancestors'][0]['files']['memory.swap.current']
        self.assertEqual(missing, {'status': 'unavailable', 'reason': 'not_found'})
        self.assertNotIn('values', missing)
        for node in (self.member, self.member.parent, self.mount):
            for path in node.glob('memory.*'):
                path.unlink()
        self.assertEqual(self.sample()['cgroup_v2']['status'], 'unavailable')

    def test_denied_read_is_not_zero_and_does_not_expose_path(self):
        read = Path.read_text

        def denied(path, *args, **kwargs):
            if path == self.member / 'memory.events':
                raise PermissionError('private secret path')
            return read(path, *args, **kwargs)

        with patch.object(Path, 'read_text', denied):
            probe = self.sample()['cgroup_v2']
        self.assertEqual(probe['status'], 'partial')
        self.assertEqual(probe['ancestors'][0]['files']['memory.events'], {
            'status': 'unavailable', 'reason': 'permission_denied'})
        self.assertNotIn('private secret', json.dumps(probe))

    def test_missing_and_partial_system_sources_are_not_zero(self):
        (self.proc / 'vmstat').write_text('pswpin 0\npgscan_direct 2\n')
        (self.proc / 'pressure/memory').unlink()
        sample = self.sample()
        self.assertEqual(sample['vmstat']['status'], 'partial')
        self.assertNotIn('pswpout', sample['vmstat']['values'])
        self.assertEqual(sample['psi_memory']['reason'], 'not_found')
        (self.proc / 'vmstat').unlink()
        self.assertEqual(self.sample()['vmstat']['status'], 'unavailable')

    def test_invalid_counters_limits_and_nonfinite_psi_are_marked(self):
        (self.member / 'memory.current').write_text('-1\n')
        (self.proc / 'vmstat').write_text('pswpin not-a-number\n')
        for value in ('nan', 'inf', '-0.1', '101'):
            (self.proc / 'pressure/memory').write_text(
                f'some avg10={value} avg60=0 avg300=0 total=1\n')
            sample = self.sample()
            self.assertEqual(sample['psi_memory']['reason'], 'invalid_data')
        self.assertEqual(sample['vmstat']['reason'], 'invalid_data')
        self.assertEqual(sample['cgroup_v2']['ancestors'][0]['files']['memory.current']['reason'], 'invalid_data')
        (self.member / 'memory.current').write_text('max\n')
        self.assertEqual(self.sample()['cgroup_v2']['ancestors'][0]['files']['memory.current']['reason'], 'invalid_data')

    def test_missing_child_and_v1_only_are_unavailable(self):
        (self.proc / '123/cgroup').unlink()
        self.assertEqual(self.sample()['cgroup_v2']['reason'], 'not_found')
        (self.proc / '123/cgroup').write_text('5:memory:/private-team/child\n')
        self.assertEqual(self.sample()['cgroup_v2']['reason'], 'cgroup_v2_not_visible')

    def test_page_size_unavailable_does_not_fabricate_zero_or_abort(self):
        with patch.object(telemetry.os, 'sysconf', side_effect=ValueError('not available')):
            sample = self.sample()
        self.assertIsNone(sample['page_size_bytes'])
        self.assertEqual(sample['vmstat']['status'], 'ok')

    def test_migration_during_sample_discards_wrong_attribution(self):
        read = Path.read_text
        reads = 0

        def migrate(path, *args, **kwargs):
            nonlocal reads
            if path == self.proc / '123/cgroup':
                reads += 1
                return '0::/private-team/child\n' if reads == 1 else '0::/other\n'
            return read(path, *args, **kwargs)

        with patch.object(Path, 'read_text', migrate):
            self.assertEqual(self.sample()['cgroup_v2'], {
                'status': 'unavailable', 'reason': 'membership_changed_or_exited'})

    def test_unmapped_or_traversing_membership_is_rejected(self):
        for member in ('0::/../escape\n', '0::relative\n'):
            (self.proc / '123/cgroup').write_text(member)
            self.assertEqual(self.sample()['cgroup_v2']['status'], 'unavailable')
        (self.proc / '123/cgroup').write_text('0::/elsewhere\n')
        self.mountinfo('/hidden')
        self.assertEqual(self.sample()['cgroup_v2']['status'], 'unavailable')

    def test_mount_selection_prefers_longest_matching_root(self):
        mount = self.root / 'subtree'
        escaped = str(self.mount).replace(' ', r'\040')
        raw = (f'10 1 0:1 / {escaped} rw - cgroup2 cgroup rw\n'
               f'11 1 0:1 /private-team {mount} rw - cgroup2 cgroup rw\n')
        self.assertEqual(telemetry.cgroup_location(telemetry.membership('0::/private-team/child'), raw),
                         (mount / 'child', mount))

    def test_next_sample_rereads_membership_and_cumulative_counters(self):
        before = self.sample()
        other = self.mount / 'other'
        other.mkdir()
        self.populate(other)
        (self.proc / '123/cgroup').write_text('0::/other\n')
        (self.proc / 'vmstat').write_text('pswpin 2\npswpout 23\n')
        after = self.sample()
        self.assertEqual(after['vmstat']['values']['pswpout'] - before['vmstat']['values']['pswpout'], 6)
        self.assertNotEqual(before['cgroup_v2']['ancestors'][0]['identity'],
                            after['cgroup_v2']['ancestors'][0]['identity'])
        self.assertEqual(after['cgroup_v2']['ancestors'][0]['identity']['inode'], other.stat().st_ino)

    def test_shared_sample_preserves_guard_and_watchdog_uses_reclaim(self):
        process = {'VmRSS': 6000 * 1024, 'VmSwap': 0}
        system = {'MemAvailable': 4000 * 1024}
        reclaim = self.sample()
        with tempfile.TemporaryDirectory() as name:
            child = Mock(pid=123)
            child.poll.return_value = None
            watcher = http.MemoryWatchdog(child, Path(name))
            with patch.object(http, 'process_snapshot', return_value=process), \
                    patch.object(http, 'memory_snapshot', return_value=system), \
                    patch.object(http, 'reclaim_snapshot', return_value=reclaim) as probe:
                watcher.thread.start()
                watcher.thread.join(timeout=2)
                watcher.finish()
            child.terminate.assert_called_once_with()
            probe.assert_called_once_with(123)
            saved = json.loads((Path(name) / 'memory.jsonl').read_text())
            self.assertEqual(saved['process'], process)
            self.assertEqual(saved['system'], system)
            self.assertEqual(saved['reclaim'], reclaim)
            self.assertEqual(watcher.violations[0]['reason'], 'RSS watchdog threshold')


if __name__ == '__main__':
    unittest.main()
