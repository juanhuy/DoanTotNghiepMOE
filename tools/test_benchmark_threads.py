"""A3 thread sweep checks with fake records only; no inference."""
import contextlib
import copy
import io
import json
import unittest

import benchmark_threads as sweep
import run_benchmark as common
import test_benchmark_gate_up_model as fixtures


def chat_block(threads, scale=1.0):
    r = fixtures.GateUpModelTests().chat_record("off")
    r["config"]["threads"] = threads
    r["config"]["output_root"] = f"t{threads}"
    for turn in r["turns"]:
        m = turn["response"]["metrics"]
        for key in sweep.RATIO_KEYS:
            m[key] = m.get(key, 10.0) * scale
        m["decode_tokens_per_second"] = 6.0 / scale
    return r


class ThreadSweepTests(unittest.TestCase):
    def test_order_is_palindrome_per_testbed(self):
        self.assertEqual(sweep.order(sweep.THREAD_SETS["i5"]), (4, 6, 8, 12, 12, 8, 6, 4))
        self.assertEqual(sweep.order(sweep.THREAD_SETS["i7"]), (4, 8, 16, 24, 24, 16, 8, 4))

    def test_compare_ignores_threads_but_not_output_or_work(self):
        a, b = chat_block(4), chat_block(8)
        sweep.compare_blocks(a, b)
        for change in ("text", "work", "config"):
            c = chat_block(8)
            if change == "text":
                c["turns"][6]["response"]["text"] = "other"
            elif change == "work":
                c["turns"][6]["response"]["metrics"]["expert_file_loads"] += 1
            else:
                c["config"]["cache"] = 128
            with self.assertRaises(ValueError):
                sweep.compare_blocks(a, c)

    def test_summary_rule_and_recommendation(self):
        sequence = list(sweep.order(sweep.THREAD_SETS["i5"]))
        # 6 threads 20% faster everywhere, 8 threads 2% faster (inconclusive), 12 threads 10% slower.
        scale = {4: 1.0, 6: 1 / 1.2, 8: 1 / 1.02, 12: 1.1}
        records = [chat_block(n, scale[n]) for n in sequence]
        per, comparisons, recommended = sweep.summarize(records, sequence, (4, 5, 6, 7))
        self.assertTrue(comparisons["6"]["faster_than_4"])
        self.assertFalse(comparisons["8"]["faster_than_4"] or comparisons["8"]["slower_than_4"])
        self.assertTrue(comparisons["12"]["slower_than_4"])
        self.assertEqual(len(comparisons["6"]["pairs"]), 8)
        self.assertEqual(comparisons["6"]["pairs"][0]["blocks"], [0, 1])
        self.assertEqual(comparisons["6"]["pairs"][4]["blocks"], [7, 6])
        self.assertEqual(recommended, 6)
        self.assertAlmostEqual(per["6"]["median_decode_tokens_per_second"], 7.2)
        records = [chat_block(n, 1.0) for n in sequence]
        self.assertEqual(sweep.summarize(records, sequence, (4, 5, 6, 7))[2], 4)

    def test_dry_run_protocol_and_scope(self):
        base = ["--thread-sweep-model", "--gpu", "auto", "--expert-batch-workload", "mixed", "--cache", "64",
                "--context", "512", "--max-tokens", "64", "--lookahead", "off", "--allow-dirty", "--dry-run"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(common.execute(common.parser().parse_args(["--testbed", "i7", *base])), 0)
        protocol = json.loads(out.getvalue())
        self.assertEqual((protocol["protocol"], protocol["total_requests"], protocol["gpu_expert_tier"]),
                         ("threads-chat-v1", 64, "off"))
        self.assertEqual(protocol["thread_order"], [4, 8, 16, 24, 24, 16, 8, 4])
        for bad in (["--gpu", "cpu"], ["--max-tokens", "1"], ["--expert-batch-workload", "long-prefill"]):
            argv = copy.copy(base)
            flag = bad[0]
            argv[argv.index(flag) + 1] = bad[1]
            with self.assertRaises(ValueError):
                common.execute(common.parser().parse_args(["--testbed", "i5", *argv]))
        with self.assertRaises(ValueError):
            common.execute(common.parser().parse_args(["--thread-sweep-model", "--gpu-tier-model-ab", "--testbed", "i5",
                                                       *base[1:]]))


if __name__ == "__main__":
    unittest.main()
