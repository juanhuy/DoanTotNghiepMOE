#!/usr/bin/env python3
"""Unit tests for analyze_route_trace.py tool."""
import tempfile
import unittest
from pathlib import Path

from analyze_route_trace import (
    compute_gini_coefficient,
    parse_usage_file,
    analyze_usage,
    parse_route_trace_file,
)


class TestAnalyzeRouteTrace(unittest.TestCase):
    def test_gini_coefficient(self):
        # Đồng đều tuyệt đối: Gini = 0
        uniform = [10.0] * 64
        self.assertAlmostEqual(compute_gini_coefficient(uniform), 0.0, places=4)

        # Lệch cực đại: chỉ 1 phần tử chiếm toàn bộ
        skewed = [0.0] * 63 + [100.0]
        gini_skewed = compute_gini_coefficient(skewed)
        self.assertGreater(gini_skewed, 0.95)

    def test_parse_usage_file_and_analysis(self):
        sample_usage = (
            "-1 2 4\n"
            "-2 1 3192021591\n"
            "0 1 100\n"
            "0 3 50\n"
            "1 0 20\n"
            "1 2 80\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(sample_usage)
            f.flush()
            tmp_path = Path(f.name)

        try:
            data = parse_usage_file(tmp_path)
            self.assertEqual(data["num_layers"], 2)
            self.assertEqual(data["num_experts"], 4)
            self.assertEqual(data["counts"][0][1], 100)
            self.assertEqual(data["counts"][1][2], 80)

            analysis = analyze_usage(data, top_k=2)
            self.assertEqual(analysis["total_expert_activations"], 250)
            l0 = analysis["layer_statistics"][0]
            self.assertEqual(l0["active_experts"], 2)
            self.assertEqual(l0["top_hot_experts"][0]["expert_id"], 1)
            self.assertEqual(l0["top_hot_experts"][0]["count"], 100)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_parse_route_trace_file(self):
        sample_trace = (
            "0 0 0 5:0.6500 12:0.3500\n"
            "0 1 0 5:0.8000 21:0.2000\n"
            "1 0 1 2:0.5000 3:0.5000\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(sample_trace)
            f.flush()
            tmp_path = Path(f.name)

        try:
            records = parse_route_trace_file(tmp_path)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["call_id"], 0)
            self.assertEqual(records[0]["row"], 0)
            self.assertEqual(records[0]["layer"], 0)
            self.assertEqual(len(records[0]["routes"]), 2)
            self.assertEqual(records[0]["routes"][0]["expert_id"], 5)
            self.assertEqual(records[0]["routes"][0]["gate"], 0.65)
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
