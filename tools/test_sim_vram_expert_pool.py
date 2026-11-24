#!/usr/bin/env python3
"""Tests for the offline VRAM expert pool replay."""

import tempfile
import unittest
from pathlib import Path

import sim_vram_expert_pool as sim

MiB = 1024 * 1024


def write_trace(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return Path(handle.name)


class ParseTraceTest(unittest.TestCase):
    def test_splits_layer_major_prefill_from_decode_steps(self):
        # Two layers, prefill of 2 tokens, then one decode step; call 1 is prefill only.
        path = write_trace(
            [
                "0 0 0 1:0.5 2:0.5",
                "0 1 0 1:0.5 3:0.5",
                "0 0 1 4:0.5 5:0.5",
                "0 1 1 4:0.5 6:0.5",
                "0 2 0 2:0.5 3:0.5",
                "0 2 1 5:0.5 6:0.5",
                "1 0 0 7:0.5 1:0.5",
                "1 0 1 4:0.5 5:0.5",
            ]
        )
        calls, layers = sim.parse_trace(path)
        self.assertEqual(layers, 2)
        call, prefill, steps = calls[0]
        self.assertEqual(prefill, [[[1, 2], [1, 3]], [[4, 5], [4, 6]]])
        self.assertEqual(steps, [[[2, 3], [5, 6]]])
        self.assertEqual(calls[1][1], [[[7, 1]], [[4, 5]]])
        self.assertEqual(calls[1][2], [])


class ReplayTest(unittest.TestCase):
    def test_ram_cache_counts_hits_misses_and_evictions(self):
        # Budget holds two experts; prefill is empty so only decode touches the cache.
        calls = [(0, [[]], [[[1, 2]], [[1, 3]], [[1, 2]]])]
        per_call, _ = sim.run(calls, 1, 8, MiB, 2 * MiB, "none", 0)
        # step1: miss 1,2; step2: hit 1, miss 3 (evicts 2); step3: hit 1, miss 2 (evicts 3).
        self.assertEqual(per_call[0]["misses"], 4)
        self.assertEqual(per_call[0]["hits"], 2)
        self.assertEqual(per_call[0]["evictions"], 2)

    def test_a1_pool_is_fill_only_and_serves_experts_absent_from_ram(self):
        # RAM holds one expert; pool holds one slot and keeps the first expert read.
        calls = [(0, [[]], [[[1]], [[2]], [[1]], [[3]], [[1]]])]
        per_call, model = sim.run(calls, 1, 8, MiB, MiB, "a1_first", 1)
        self.assertEqual(model.pool, {(0, 1)})
        self.assertEqual(per_call[0]["uploads"], 1)
        # Expert 1 is read once from SSD, then served from the pool twice.
        self.assertEqual(per_call[0]["misses"], 3)
        self.assertEqual(per_call[0]["gpu_routes"], 2)

    def test_quota_spreads_slots_across_layers(self):
        # Layer-major prefill reads all of layer 0 first, so first-come fills layer 0 only.
        calls = [(0, [[[1, 2]], [[3, 4]]], [])]
        _, first = sim.run(calls, 2, 8, MiB, 0, "a1_first", 2)
        _, quota = sim.run(calls, 2, 8, MiB, 0, "a1_quota", 2)
        self.assertEqual(first.pool, {(0, 1), (0, 2)})
        self.assertEqual(quota.pool, {(0, 1), (1, 3)})

    def test_static_placement_never_uploads(self):
        calls = [(0, [[]], [[[1]], [[2]], [[1]]])]
        per_call, model = sim.run(calls, 1, 8, MiB, 0, "oracle_static", 1, [(0, 1)])
        self.assertEqual(per_call[0]["uploads"], 0)
        self.assertEqual(per_call[0]["gpu_routes"], 2)
        self.assertEqual(per_call[0]["misses"], 1)


if __name__ == "__main__":
    unittest.main()
