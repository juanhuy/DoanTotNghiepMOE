"""Validate the bounded resident protocol without loading a checkpoint or timing."""
import contextlib
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import benchmark_gate_up_resident as gate
import run_benchmark as runner


def report():
    return {
        "schema_version":1,"kind":"gate_up_resident","protocol":"gate-up-resident-v1",
        "testbed":"i5","hidden":2048,"intermediate":1024,"threads":4,"avx2_fma":True,
        "layers":list(gate.LAYERS),"expert_ids":list(range(8)),"seeds":list(gate.SEEDS),
        "scopes":list(gate.SCOPES),"pairs_per_case":9,"iterations":40,
        "warmup_per_variant_scope":10,"payload_budget_bytes":256*1024**2,
        "payload_bytes":24*(3*2048*1024+4096*4),
        "process_before":"VmHWM: 160000 kB\nVmSwap: 0 kB",
        "process_after":"VmHWM: 160000 kB\nVmSwap: 0 kB",
        "io_before":"read_bytes: 99","io_after":"read_bytes: 99",
        "parity":[{"layer":l,"seed":s,"scope":scope,"max_abs_error":0.0}
                  for l in gate.LAYERS for s in gate.SEEDS for scope in gate.SCOPES],
        "samples":[{"layer":l,"seed":s,"scope":scope,"pair":pair,"position":pos,
                    "variant":"paired_simd" if (pair+pos)%2 else "separate_simd",
                    "iterations":40,"elapsed_ns":100 if (pair+pos)%2 else 120}
                   for l in gate.LAYERS for s in gate.SEEDS for scope in gate.SCOPES
                   for pair in range(9) for pos in range(2)]
    }


def batch_report():
    r=report()
    r.update(kind="expert_batch_resident",protocol="expert-batch-int8-v1",
             expert_ids=[0],scopes=list(gate.BATCH_SCOPES),
             payload_bytes=3*(3*2048*1024+4096*4),baseline_gate_up="paired",
             baseline_dispatch="serial_tokens_row_parallel",
             scratch_budget_bytes=256*1024,scratch_peak_capacity_bytes=176*1024)
    for row in r["parity"]+r["samples"]:
        row["scope"]="batch2" if row["scope"]=="expert_tasks8" else "batch4"
        if "variant" in row:
            row["variant"]="shared_weights" if row["variant"]=="paired_simd" else "separate_tokens"
    return r


class ResidentProtocolTests(unittest.TestCase):
    def test_complete_report_and_rule(self):
        r=report()
        gate.validate(r,"i5")
        s=gate.summarize(r)
        self.assertTrue(s["passes_compute_rule"])
        self.assertEqual(s["expert_tasks8"]["wins"],81)
        self.assertAlmostEqual(s["expert_tasks8"]["median_paired_ratio"],1.2)

    def test_refuse_missing_duplicate_order_and_parity(self):
        mutations=[
            lambda r:r["samples"].pop(),
            lambda r:r["samples"].append(r["samples"][0]),
            lambda r:r["samples"][0].update(variant="paired_simd"),
            lambda r:r["parity"][0].update(max_abs_error=float("nan")),
            lambda r:r["parity"].pop(),
            lambda r:r.update(threads=1),
        ]
        for mutate in mutations:
            r=report();mutate(r)
            with self.assertRaises(ValueError):gate.validate(r,"i5")

    def test_refuse_memory_swap_or_reads(self):
        for field,value in [("process_after","VmHWM: 160000 kB\nVmSwap: 4 kB"),
                            ("process_after","VmHWM: 2000000 kB\nVmSwap: 0 kB"),
                            ("io_after","read_bytes: 100"),("payload_bytes",0)]:
            r=report();r[field]=value
            with self.assertRaises(ValueError):gate.validate(r,"i5")

    def test_secondary_scope_does_not_rescue_primary_failure(self):
        r=report()
        for row in r["samples"]:
            if row["scope"]=="expert_tasks8":
                row["elapsed_ns"]=100 if row["variant"]=="separate_simd" else 120
        gate.validate(r,"i5")
        self.assertFalse(gate.summarize(r)["passes_compute_rule"])
        r=report()
        for row in r["samples"]:
            if row["layer"]==15 and row["seed"]==991 and row["scope"]=="expert_tasks8" and row["pair"]<3:
                row["elapsed_ns"]=100 if row["variant"]=="separate_simd" else 120
        self.assertFalse(gate.summarize(r)["passes_compute_rule"])

    def test_preview_never_loads_or_builds_and_requires_four_threads(self):
        args=runner.parser().parse_args(["--testbed","i5","--gpu","cpu","--gate-up-resident",
                                         "--threads","4","--allow-dirty","--dry-run"])
        with patch.object(gate.common,"checked_text",return_value="test"),              patch.object(gate.common,"checkpoint_manifest",side_effect=AssertionError("load")),              patch.object(gate.common,"run_command",side_effect=AssertionError("build")),              contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.execute(args,Path.cwd()),0)
            args.threads=1
            with self.assertRaises(ValueError):gate.execute(args,Path.cwd())
        args.threads=4;args.gate_up_model_ab=True
        with self.assertRaises(ValueError):runner.execute(args)


    def test_batch_protocol_and_both_sizes_must_pass(self):
        r=batch_report()
        gate.validate(r,"i5",batch=True)
        self.assertTrue(gate.summarize(r,batch=True)["passes_compute_rule"])
        for row in r["samples"]:
            if row["scope"]=="batch2":
                row["elapsed_ns"]=100 if row["variant"]=="separate_tokens" else 120
        self.assertFalse(gate.summarize(r,batch=True)["passes_compute_rule"])

    def test_batch_rejects_scratch_baseline_or_variant_drift(self):
        for key,value in (("scratch_peak_capacity_bytes",300*1024),("baseline_gate_up","separate"),
                          ("expert_ids",list(range(8)))):
            r=batch_report();r[key]=value
            with self.assertRaises(ValueError):gate.validate(r,"i5",batch=True)
        r=batch_report();r["samples"][0]["variant"]="separate_simd"
        with self.assertRaises(ValueError):gate.validate(r,"i5",batch=True)

    def test_batch_preview_dispatches_to_shared_harness_without_loading(self):
        args=runner.parser().parse_args(["--testbed","i5","--gpu","cpu","--expert-batch-kernel",
                                         "--threads","4","--allow-dirty","--dry-run"])
        output=io.StringIO()
        with patch.object(gate.common,"checked_text",return_value="test"), \
             patch.object(gate.common,"checkpoint_manifest",side_effect=AssertionError("load")), \
             patch.object(gate.common,"run_command",side_effect=AssertionError("build")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(runner.execute(args),0)
        import json
        preview=json.loads(output.getvalue())
        self.assertEqual(preview["scopes"],["batch2","batch4"])
        self.assertEqual(preview["test"],gate.BATCH_TEST)


if __name__=="__main__":
    unittest.main()
