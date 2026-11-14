"""Preflight eligibility rules; no sleeping or /proc access needed."""
import unittest

import preflight_memory as pre


def s(avail_mib, pswpout):
    return {'available_kib': avail_mib * 1024, 'pswpout': pswpout}


class PreflightTests(unittest.TestCase):
    def test_eligible_only_when_quiet_stable_and_enough_memory(self):
        quiet = [s(9000, 5), s(8800, 5), s(8900, 5)]
        self.assertTrue(pre.eligible(quiet, [], 5120))
        self.assertFalse(pre.eligible([s(9000, 5), s(9000, 6), s(9000, 7)], [], 5120))
        self.assertFalse(pre.eligible([s(9000, 5), s(5000, 5), s(9000, 5)], [], 5120))
        self.assertFalse(pre.eligible(quiet, ['123 cargo cargo build'], 5120))
        self.assertFalse(pre.eligible([s(9000, None)] * 3, [], 5120))

    def test_power_and_gpu_throttle_block_eligibility(self):
        quiet = [s(9000, 5)] * 3
        ac, battery = {'on_ac': True}, {'on_ac': False}
        idle = pre.parse_gpu_throttle('P0, 990 MHz, 48, 16.20 W, Not Active, Not Active, Not Active, Not Active, Not Active')
        capped = pre.parse_gpu_throttle('P8, 300 MHz, 54, 3.55 W, Not Active, Not Active, Active, Not Active, Active')
        self.assertFalse(idle['throttled'])
        self.assertEqual(capped['active_throttle'], ['clocks_event_reasons.sw_thermal_slowdown',
                                                     'clocks_event_reasons.sw_power_cap'])
        self.assertTrue(pre.eligible(quiet, [], 5120, ac, idle))
        self.assertFalse(pre.eligible(quiet, [], 5120, battery, idle))
        self.assertFalse(pre.eligible(quiet, [], 5120, ac, capped))
        self.assertTrue(pre.eligible(quiet, [], 5120, ac, {'status': 'absent', 'throttled': False}))

    def test_read_power_from_sysfs_layout(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, values in (('ADP1', {'type': 'Mains', 'online': '0'}),
                                 ('BAT1', {'type': 'Battery', 'status': 'Discharging', 'capacity': '20'})):
                (root / name).mkdir()
                for key, value in values.items():
                    (root / name / key).write_text(value + '\n')
            self.assertFalse(pre.read_power(root)['on_ac'])
            (root / 'ADP1' / 'online').write_text('1\n')
            self.assertTrue(pre.read_power(root)['on_ac'])
            self.assertTrue(pre.read_power(root / 'missing')['on_ac'])

    def test_active_process_filter_ignores_itself(self):
        lines = ['1 python3 python3 tools/preflight_memory.py --output x',
                 '2 moe-tier-engine target/release/moe-tier-engine', '3 bash bash']
        self.assertEqual(pre.active_processes(lines), ['2 moe-tier-engine target/release/moe-tier-engine'])


if __name__ == '__main__':
    unittest.main()
