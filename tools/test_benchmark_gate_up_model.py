"""Model Gate-Up protocol checks with fake records only; no inference."""
import copy
import os
import unittest
from unittest.mock import patch

import benchmark_gate_up_model as model


def record(mode):
    trace = {"outcome": "completed", "samples": [{"stage": str(i)} for i in range(7)]}
    status = {"gate_up_mode": mode, "expert_quantization": "int8", "kv_fp8": False,
              "gpu_mode_at_load": "cpu", "lookahead_enabled": False,
              "kv_buffer_reuse_enabled": False, "session_dir": None,
              "gpu": {"enabled": False, "allocated_vram_bytes": 0},
              "last_memory_trace": trace}
    metrics = {"gate_up_mode": mode, "cached_prompt_tokens": 0, "memory_trace": trace,
               **{k: 2 for k in model.WORK_COUNTERS},
               **{k: 10.0 for k in ("total_ms", "prefill_ms", "decode_ms",
                                   "time_to_first_token_ms", "expert_compute_ms")}}
    return {"status": "completed", "watchdog_violations": [], "status_start": status,
            "cpu_start_library_probe": {"available": True, "cuda_libraries": []},
            "source_sha256": {"src": "hash"}, "binary_sha256": "hash", "checkpoint": {"hash": "hash"},
            "config": {"gate_up_mode": mode, "output_root": mode, "threads": 4},
            "turns": [{"session_id": str(i), "messages": [{"role": "user", "content": prompt}],
                       "status": copy.deepcopy(status),
                       "response": {"text": "same", "usage": {"completion_tokens": 64},
                                    "finish_reason": "length", "metrics": copy.deepcopy(metrics), "client_wall_ms": 11}}
                      for i, prompt in enumerate(("Where is Mars", "Explain a Python dictionary."))]}


def batch_record(mode):
    r=record("paired")
    enabled=mode=="on"
    r["config"].update(expert_batch_prefill=mode,max_tokens=1)
    for status in [r["status_start"],*(t["status"] for t in r["turns"])]:
        status["expert_batch_prefill_enabled"]=enabled
    for t in r["turns"]:
        t["response"]["usage"]["completion_tokens"]=1
        t["response"]["metrics"].update(expert_batch_prefill_enabled=enabled,
                                       prefill_expert_batch2_calls=2 if enabled else 0,
                                       prefill_expert_batch4_calls=1 if enabled else 0)
    return r


class GateUpModelTests(unittest.TestCase):
    def test_block_and_cross_mode_parity(self):
        a, b = record("separate"), record("paired")
        model.validate_block(a, "separate")
        model.validate_block(b, "paired")
        model.compare_blocks(a, b)

    def test_refuse_precision_or_missing_trace_or_watchdog(self):
        for change in ("precision", "trace", "watchdog", "same_session"):
            r = record("paired")
            if change == "precision":
                r["status_start"]["expert_quantization"] = "int4"
            elif change == "trace":
                r["turns"][0]["response"]["metrics"]["memory_trace"] = None
            elif change == "watchdog":
                r["watchdog_violations"] = ["swap"]
            else:
                r["turns"][1]["session_id"] = r["turns"][0]["session_id"]
            with self.assertRaises(ValueError):
                model.validate_block(r, "paired")

    def test_refuse_source_output_or_workload_drift(self):
        for change in ("source", "output", "loads", "config"):
            a, b = record("separate"), record("paired")
            if change == "source":
                b["source_sha256"] = {"src": "changed"}
            elif change == "output":
                b["turns"][0]["response"]["text"] = "different"
            elif change == "loads":
                b["turns"][0]["response"]["metrics"]["expert_file_loads"] += 1
            else:
                b["config"]["threads"] = 8
            with self.assertRaises(ValueError):
                model.compare_blocks(a, b)

    def test_abba_pairs_use_last_baseline_for_second_pair(self):
        records = [record(m) for m in model.MODES]
        records[3]["turns"][0]["response"]["metrics"]["total_ms"] = 20
        records[2]["turns"][0]["response"]["metrics"]["total_ms"] = 5
        summary = model.paired_summary(records)
        self.assertEqual(summary[0]["timers"]["total_ms"]["separate_over_paired"], 1)
        self.assertEqual(summary[2]["timers"]["total_ms"]["separate_over_paired"], 4)

    def test_environment_restores_after_failure(self):
        with patch.dict(os.environ, {"MOE_TEMPERATURE": "0.8", "MOE_SESSION_DIR": "fixture"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "trigger"):
                with model.fixed_environment():
                    self.assertEqual(os.environ["MOE_TEMPERATURE"], "0")
                    self.assertNotIn("MOE_SESSION_DIR", os.environ)
                    raise RuntimeError("trigger")
            self.assertEqual(os.environ["MOE_TEMPERATURE"], "0.8")
            self.assertEqual(os.environ["MOE_SESSION_DIR"], "fixture")


    def test_expert_batch_selector_calls_and_same_work(self):
        a,b=batch_record("off"),batch_record("on")
        model.validate_block(a,"off",expert_batch=True)
        model.validate_block(b,"on",expert_batch=True)
        model.compare_blocks(a,b,expert_batch=True)
        b["turns"][0]["response"]["metrics"]["prefill_expert_batch2_calls"]=0
        b["turns"][0]["response"]["metrics"]["prefill_expert_batch4_calls"]=0
        with self.assertRaises(ValueError):model.validate_block(b,"on",expert_batch=True)

    def test_expert_batch_summary_ignores_decode_for_max1(self):
        records=[batch_record(m) for m in ("off","on","on","off")]
        for r in records:
            for turn in r["turns"]:
                turn["response"]["metrics"]["decode_ms"]=0
        pairs=model.paired_summary(records,expert_batch=True)
        self.assertEqual(pairs[0]["timers"]["time_to_first_token_ms"]["baseline_over_candidate"],1)
        self.assertNotIn("decode_ms",pairs[0]["timers"])

    def test_expert_batch_refuses_mode_and_token_limit_drift(self):
        for key,value in (("completion_tokens",2),("mode",False)):
            r=batch_record("on")
            if key=="completion_tokens":r["turns"][0]["response"]["usage"][key]=value
            else:r["status_start"]["expert_batch_prefill_enabled"]=value
            with self.assertRaises(ValueError):model.validate_block(r,"on",expert_batch=True)

    def test_long_prefill_workload_frozen_and_independent(self):
        import check_ram_stability as ram
        spec = ram.workload_spec("long-prefill")
        self.assertEqual(spec["expected_prompt_tokens"], [64, 128, 64, 128])
        self.assertEqual(spec["prompts"][:2], spec["prompts"][2:])
        self.assertEqual((spec["warmup_turns"], spec["measured_turns"]), ([0, 1], [2, 3]))
        # Frozen text: any edit changes the experiment identity recorded in SCENARIOS.
        import hashlib
        self.assertEqual([hashlib.sha256(s.encode()).hexdigest()[:16] for s in spec["prompts"][:2]],
                         ["4d14aaf710d91fe6", "fcdccc1028f29461"])
        plan = [ram.turn_plan(spec, i, True) for i in range(4)]
        self.assertEqual([p[1] for p in plan], [True] * 4)
        self.assertEqual([p[2] for p in plan], [True, False, False, False])
        args = ram.parser().parse_args(["--testbed", "i5", "--gpu", "cpu", "--cache", "64",
                                        "--workload", "long-prefill"])
        self.assertEqual(args.workload, "long-prefill")

    def long_record(self, mode):
        import check_ram_stability as ram
        spec = ram.workload_spec("long-prefill")
        r = batch_record(mode)
        base = r["turns"][0]
        r["turns"] = []
        for i, (prompt, tokens) in enumerate(zip(spec["prompts"], spec["expected_prompt_tokens"])):
            turn = copy.deepcopy(base)
            turn["session_id"] = str(i)
            turn["messages"] = [{"role": "user", "content": prompt}]
            turn["response"]["usage"]["prompt_tokens"] = tokens
            r["turns"].append(turn)
        return r

    def test_long_prefill_block_validation_and_drift(self):
        a, b = self.long_record("off"), self.long_record("on")
        model.validate_block(a, "off", expert_batch=True, workload="long-prefill")
        model.validate_block(b, "on", expert_batch=True, workload="long-prefill")
        model.compare_blocks(a, b, expert_batch=True)
        for change in ("tokens", "short", "session"):
            r = self.long_record("on")
            if change == "tokens":
                r["turns"][3]["response"]["usage"]["prompt_tokens"] = 127
            elif change == "short":
                r["turns"] = r["turns"][:2]
            else:
                r["turns"][2]["session_id"] = r["turns"][0]["session_id"]
            with self.assertRaises(ValueError):
                model.validate_block(r, "on", expert_batch=True, workload="long-prefill")
        with self.assertRaises(ValueError):
            model.validate_block(self.long_record("on"), "on", expert_batch=True)

    def test_long_prefill_summary_uses_measured_turns_only(self):
        records = [self.long_record(m) for m in ("off", "on", "on", "off")]
        records[0]["turns"][0]["response"]["metrics"]["prefill_ms"] = 1000.0  # warmup outlier
        records[0]["turns"][2]["response"]["metrics"]["prefill_ms"] = 30.0
        pairs = model.paired_summary(records, expert_batch=True, turns=model.measured_turns("long-prefill"))
        self.assertEqual([p["turn"] for p in pairs], [2, 3, 2, 3])
        self.assertEqual(pairs[0]["timers"]["prefill_ms"]["baseline_over_candidate"], 3)
        self.assertEqual(model.measured_turns("kv-switch"), (0, 1))

    def test_long_prefill_dry_run_protocol_and_scope(self):
        import io, contextlib, json, types
        import run_benchmark as common
        args = common.parser().parse_args(["--testbed", "i5", "--gpu", "cpu", "--expert-batch-model-ab",
                                           "--expert-batch-workload", "long-prefill", "--threads", "4",
                                           "--cache", "64", "--context", "512", "--max-tokens", "1",
                                           "--lookahead", "off", "--allow-dirty", "--dry-run"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(model.execute(args, common.WORKSPACE_DIR, expert_batch=True), 0)
        protocol = json.loads(out.getvalue())
        self.assertEqual((protocol["requests_per_block"], protocol["total_requests"]), (4, 16))
        self.assertEqual((protocol["measured_turns"], protocol["warmup_turns"]), ([2, 3], [0, 1]))
        self.assertEqual(protocol["workload"], "long-prefill-v1")
        default = common.parser().parse_args(["--testbed", "i5", "--gpu", "cpu", "--expert-batch-model-ab"])
        self.assertEqual(default.expert_batch_workload, "kv-switch")
        gate = types.SimpleNamespace(**vars(args))
        with self.assertRaises(ValueError):
            model.execute(gate, common.WORKSPACE_DIR, expert_batch=False)


if __name__ == "__main__":
    unittest.main()
