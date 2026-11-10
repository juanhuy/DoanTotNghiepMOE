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

    def test_active_process_filter_ignores_itself(self):
        lines = ['1 python3 python3 tools/preflight_memory.py --output x',
                 '2 moe-tier-engine target/release/moe-tier-engine', '3 bash bash']
        self.assertEqual(pre.active_processes(lines), ['2 moe-tier-engine target/release/moe-tier-engine'])


if __name__ == '__main__':
    unittest.main()
