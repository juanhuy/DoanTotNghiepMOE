"""Gate-Up protocol regression tests; no real timing, model or GPU."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import benchmark_gate_up as gate
import run_benchmark as runner


def report():
    return {
        "schema_version": 1, "kind": "gate_up_kernel", "protocol": "gate-up-int8-v1",
        "testbed": "i5", "hidden": 2048, "intermediate": 1024, "threads": 1,
        "seeds": list(gate.SEEDS), "pairs_per_seed_scope": 9, "iterations": 100,
        "warmup_per_variant_scope": 10, "avx2_fma": True, "scopes": list(gate.SCOPES),
        "parity": [{"seed": seed, "scope": scope, "max_abs_error": 1e-7}
                   for seed in gate.SEEDS for scope in gate.SCOPES],
        "samples": [{"seed": seed, "scope": scope, "pair": pair, "position": pos,
                     "variant": "paired_simd" if (pair + pos) % 2 else "separate_simd",
                     "iterations": 100, "elapsed_ns": 100 if (pair + pos) % 2 else 120}
                    for seed in gate.SEEDS for scope in gate.SCOPES
                    for pair in range(9) for pos in range(2)]
    }


class GateUpProtocolTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parser().parse_args(["--testbed", "i5", "--gpu", "cpu",
                                          "--gate-up-kernel", *extra])

    def test_valid_pairs_and_predeclared_decision(self):
        data = report()
        gate.validate(data, "i5")
        summary = gate.summarize(data)
        self.assertTrue(summary["promising_for_model_ab"])
        self.assertAlmostEqual(summary["full_expert"]["median_paired_speed_ratio"], 1.2)
        self.assertTrue(all(s["wins"] == 9 for s in summary["full_expert"]["seeds"]))

    def test_missing_duplicate_or_wrong_order_is_rejected(self):
        for mutation in ("missing", "duplicate", "order", "nan"):
            data = report()
            if mutation == "missing":
                data["samples"].pop()
            elif mutation == "duplicate":
                data["samples"][-1] = copy.deepcopy(data["samples"][0])
            elif mutation == "order":
                data["samples"][0]["variant"] = "paired_simd"
            else:
                data["samples"][0]["elapsed_ns"] = float("nan")
            with self.assertRaises(ValueError):
                gate.validate(data, "i5")

    def test_bad_parity_or_cpu_metadata_is_rejected(self):
        for field, value in [("max_abs_error", 2e-5), ("max_abs_error", float("nan"))]:
            data = report()
            data["parity"][0][field] = value
            with self.assertRaises(ValueError):
                gate.validate(data, "i5")
        data = report()
        data["avx2_fma"] = False
        with self.assertRaises(ValueError):
            gate.validate(data, "i5")

    def test_gate_only_gain_does_not_promote_full_expert(self):
        data = report()
        for row in data["samples"]:
            if row["scope"] == "full_expert" and row["variant"] == "paired_simd":
                row["elapsed_ns"] = 130
        self.assertFalse(gate.summarize(data)["promising_for_model_ab"])

    def test_dry_run_does_not_build_or_hash_checkpoint(self):
        with patch.object(gate.common, "checked_text", side_effect=["a"*40, ""]),                 patch.object(gate.common, "run_command") as command,                 patch.object(gate, "source_files") as sources,                 contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(gate.execute(self.args("--dry-run"), Path("/unused")), 0)
        command.assert_not_called()
        sources.assert_not_called()
        preview = json.loads(output.getvalue())
        self.assertEqual(preview["measured_batches"], 108)
        self.assertEqual(preview["threads"], 1)

    def test_incompatible_protocol_is_rejected(self):
        for extra in [("--threads", "4"), ("--http-session",), ("--compare-lookahead",)]:
            with self.assertRaises(ValueError):
                gate.execute(self.args(*extra), Path("/unused"))

    def test_failed_build_preserves_failed_record_and_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("src", "tools"):
                (root / name).mkdir()
            for name in ("Cargo.toml", "Cargo.lock"):
                (root / name).write_text("fixture")
            args = self.args("--output-root", str(root / "out"))
            with patch.object(gate.common, "checked_text", side_effect=["b"*40, "", "rustc fixture"]),                     patch.object(gate.common, "hardware_observed", return_value={}),                     patch.object(gate.common, "run_command", return_value=subprocess.CompletedProcess([], 1, "build", "failed")),                     contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gate.execute(args, root), 1)
            result = next((root / "out").rglob("result.json"))
            data = json.loads(result.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertIn("build", data["error"])
            self.assertTrue((result.parent / "build.log").exists())
            self.assertTrue((result.parent / "source.zip").exists())
            self.assertFalse((result.parent / "kernel.json").exists())


if __name__ == "__main__":
    unittest.main()
