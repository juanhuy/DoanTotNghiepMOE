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

    I5_TOKENIZER = "b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167"
    I7_TOKENIZER = "d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea"

    def long_record(self, mode, tokenizer=I5_TOKENIZER, counts=None):
        import check_ram_stability as ram
        spec = ram.workload_spec("long-prefill")
        r = batch_record(mode)
        r["checkpoint"] = {"files": [{"path": "tokenizer.json", "sha256": tokenizer}]}
        base = r["turns"][0]
        r["turns"] = []
        counts = spec["expected_prompt_tokens"] if counts is None else [*counts, *counts]
        for i, (prompt, tokens) in enumerate(zip(spec["prompts"], counts)):
            turn = copy.deepcopy(base)
            turn["session_id"] = str(i)
            turn["messages"] = [{"role": "user", "content": prompt}]
            turn["response"]["usage"]["prompt_tokens"] = tokens
            r["turns"].append(turn)
        return r

    def test_long_prefill_token_counts_follow_checkpoint_tokenizer(self):
        # i5 and i7 run different OLMoE checkpoints on purpose; counts are per tokenizer.
        model.validate_block(self.long_record("on", self.I7_TOKENIZER, (61, 125)), "on",
                             expert_batch=True, workload="long-prefill")
        for tokenizer, counts in ((self.I7_TOKENIZER, (64, 128)), (self.I5_TOKENIZER, (61, 125)),
                                  ("0" * 64, (64, 128))):
            with self.assertRaises(ValueError):
                model.validate_block(self.long_record("on", tokenizer, counts), "on",
                                     expert_batch=True, workload="long-prefill")
        r = self.long_record("on")
        r["checkpoint"] = {"files": []}
        with self.assertRaises(ValueError):
            model.validate_block(r, "on", expert_batch=True, workload="long-prefill")

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

    def backend_record(self, mode):
        r = self.long_record("off")
        r["config"].pop("expert_batch_prefill")
        r["config"]["gpu"] = mode
        gpu = ({"enabled": False, "allocated_vram_bytes": 0, "dense_offloaded": False} if mode == "cpu"
               else {"enabled": True, "allocated_vram_bytes": 1 << 30, "dense_offloaded": True})
        for status in [r["status_start"], *(t["status"] for t in r["turns"])]:
            status["gpu_mode_at_load"] = mode
            status["gpu"] = dict(gpu)
        if mode == "auto":
            r.pop("cpu_start_library_probe")
            for t in r["turns"]:
                t["response"]["metrics"]["expert_file_loads"] = 1  # VRAM tier changes work: allowed
        for t in r["turns"]:
            t["response"]["metrics"].update(attention_ms=5.0, lm_head_ms=2.0, expert_io_ms=1.0,
                                           expert_vram_loads=3 if mode == "auto" else 0)
        return r

    def test_backend_block_validation_and_composite_work(self):
        a, b = self.backend_record("cpu"), self.backend_record("auto")
        model.validate_block(a, "cpu", workload="long-prefill", backend=True)
        model.validate_block(b, "auto", workload="long-prefill", backend=True)
        model.compare_blocks(a, b, backend=True)  # work counters may differ, outputs may not
        with self.assertRaises(ValueError):
            model.compare_blocks(a, b, expert_batch=True)
        bad_text = self.backend_record("auto")
        bad_text["turns"][2]["response"]["text"] = "different"
        with self.assertRaises(ValueError):
            model.compare_blocks(a, bad_text, backend=True)

    def test_backend_refuses_wrong_backend_state(self):
        for change in ("auto_disabled", "auto_no_dense", "cpu_cuda_libs", "batch_on", "mode_label"):
            mode = "cpu" if change == "cpu_cuda_libs" else "auto"
            r = self.backend_record(mode)
            if change == "auto_disabled":
                r["turns"][1]["status"]["gpu"]["enabled"] = False
            elif change == "auto_no_dense":
                r["status_start"]["gpu"]["dense_offloaded"] = False
            elif change == "cpu_cuda_libs":
                r["cpu_start_library_probe"]["cuda_libraries"] = ["libcuda.so.1"]
            elif change == "batch_on":
                r["turns"][0]["response"]["metrics"]["prefill_expert_batch2_calls"] = 4
            else:
                r["status_start"]["gpu_mode_at_load"] = "cpu"
            with self.assertRaises(ValueError):
                model.validate_block(r, mode, workload="long-prefill", backend=True)

    def test_backend_summary_names_and_work_record(self):
        records = [self.backend_record(m) for m in ("cpu", "auto", "auto", "cpu")]
        records[0]["turns"][2]["response"]["metrics"]["attention_ms"] = 20.0
        pairs = model.paired_summary(records, turns=(2, 3), backend=True)
        self.assertEqual(pairs[0]["timers"]["attention_ms"]["baseline_over_candidate"], 4)
        self.assertEqual(pairs[0]["work"]["expert_vram_loads"], {"cpu": 0, "auto": 3})
        self.assertIn("lm_head_ms", pairs[0]["timers"])

    def test_backend_dry_run_scope(self):
        import io, contextlib, json
        import run_benchmark as common
        base = ["--testbed", "i5", "--backend-model-ab", "--expert-batch-workload", "long-prefill",
                "--threads", "4", "--cache", "64", "--context", "512", "--max-tokens", "1",
                "--lookahead", "off", "--allow-dirty", "--dry-run"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(common.execute(common.parser().parse_args(["--gpu", "auto", *base])), 0)
        protocol = json.loads(out.getvalue())
        self.assertEqual(protocol["kind"], "backend_model_ab")
        self.assertEqual(protocol["modes"], ["cpu", "auto", "auto", "cpu"])
        self.assertEqual(protocol["total_requests"], 16)
        for bad in (["--gpu", "cpu", *base],
                    ["--gpu", "auto", *[x if x != "long-prefill" else "kv-switch" for x in base]]):
            with self.assertRaises(ValueError):
                common.execute(common.parser().parse_args(bad))

    def tier_record(self, mode):
        r = self.backend_record("auto")
        r["config"]["gpu_expert_tier"] = mode
        for status in [r["status_start"], *(t["status"] for t in r["turns"])]:
            status["gpu_expert_tier_enabled"] = mode == "on"
            status["gpu"]["vram_experts_resident"] = 298 if mode == "on" else 0
        for t in r["turns"]:
            t["response"]["metrics"].update(expert_vram_loads=70 if mode == "on" else 0,
                                           expert_vram_upload_bytes=5 if mode == "on" else 0)
        return r

    def test_gpu_tier_block_validation_and_parity(self):
        on, off = self.tier_record("on"), self.tier_record("off")
        model.validate_block(on, "on", workload="long-prefill", gpu_tier=True)
        model.validate_block(off, "off", workload="long-prefill", gpu_tier=True)
        model.compare_blocks(on, off, gpu_tier=True)
        with self.assertRaises(ValueError):
            model.compare_blocks(on, off, backend=True)  # tier config key differs: not a backend pair
        for change in ("traffic", "label", "cpu"):
            r = self.tier_record("off")
            if change == "traffic":
                r["turns"][3]["response"]["metrics"]["expert_vram_upload_bytes"] = 1
            elif change == "label":
                r["status_start"]["gpu_expert_tier_enabled"] = True
            else:
                r["status_start"]["gpu"]["enabled"] = False
            with self.assertRaises(ValueError):
                model.validate_block(r, "off", workload="long-prefill", gpu_tier=True)

    def test_gpu_tier_summary_and_dry_run(self):
        import io, contextlib, json
        import run_benchmark as common
        records = [self.tier_record(m) for m in ("on", "off", "off", "on")]
        pairs = model.paired_summary(records, turns=(2, 3), gpu_tier=True)
        self.assertEqual(pairs[0]["work"]["expert_vram_loads"], {"on": 70, "off": 0})
        self.assertEqual(pairs[0]["timers"]["time_to_first_token_ms"]["baseline_over_candidate"], 1)
        base = ["--testbed", "i5", "--gpu-tier-model-ab", "--expert-batch-workload", "long-prefill",
                "--threads", "4", "--cache", "64", "--context", "512", "--max-tokens", "1",
                "--lookahead", "off", "--allow-dirty", "--dry-run"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(common.execute(common.parser().parse_args(["--gpu", "auto", *base])), 0)
        protocol = json.loads(out.getvalue())
        self.assertEqual((protocol["kind"], protocol["modes"]), ("gpu_tier_model_ab", ["on", "off", "off", "on"]))
        with self.assertRaises(ValueError):
            common.execute(common.parser().parse_args(["--gpu", "cpu", *base]))
        import check_ram_stability as ram
        self.assertEqual(ram.parser().parse_args(["--testbed", "i5", "--gpu", "auto", "--cache", "64"]).gpu_expert_tier, "on")

    def prefetch_record(self, mode):
        r = self.long_record("off")
        r["config"]["prefill_prefetch"] = mode
        for status in [r["status_start"], *(t["status"] for t in r["turns"])]:
            status["prefill_prefetch_enabled"] = mode == "on"
        for t in r["turns"]:
            t["response"]["metrics"].update(prefill_prefetch_enabled=mode == "on",
                                           prefill_prefetch_loads=40 if mode == "on" else 0,
                                           prefill_prefetch_bytes=4 if mode == "on" else 0,
                                           prefill_prefetch_load_errors=0, attention_ms=5.0, lm_head_ms=2.0)
        return r

    def test_prefetch_block_validation_requires_same_work(self):
        on, off = self.prefetch_record("on"), self.prefetch_record("off")
        model.validate_block(on, "on", workload="long-prefill", prefetch=True)
        model.validate_block(off, "off", workload="long-prefill", prefetch=True)
        model.compare_blocks(on, off, prefetch=True)
        off["turns"][2]["response"]["metrics"]["expert_file_loads"] += 1
        with self.assertRaises(ValueError):
            model.compare_blocks(on, off, prefetch=True)  # unlike backend/tier, work must match
        for change in ("no_loads", "off_loads", "errors", "label"):
            mode = "off" if change == "off_loads" else "on"
            r = self.prefetch_record(mode)
            m = r["turns"][3]["response"]["metrics"]
            if change == "no_loads":
                m["prefill_prefetch_loads"] = 0
            elif change == "off_loads":
                m["prefill_prefetch_loads"] = 1
            elif change == "errors":
                m["prefill_prefetch_load_errors"] = 1
            else:
                r["status_start"]["prefill_prefetch_enabled"] = mode != "on"
            with self.assertRaises(ValueError):
                model.validate_block(r, mode, workload="long-prefill", prefetch=True)

    def test_prefetch_summary_and_dry_run(self):
        import io, contextlib, json
        import run_benchmark as common
        records = [self.prefetch_record(m) for m in ("on", "off", "off", "on")]
        pairs = model.paired_summary(records, turns=(2, 3), prefetch=True)
        self.assertEqual(pairs[0]["work"]["prefill_prefetch_loads"], {"on": 40, "off": 0})
        base = ["--testbed", "i5", "--gpu", "cpu", "--prefill-prefetch-model-ab", "--expert-batch-workload",
                "long-prefill", "--threads", "4", "--cache", "64", "--context", "512", "--max-tokens", "1",
                "--lookahead", "off", "--allow-dirty", "--dry-run"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(common.execute(common.parser().parse_args(base)), 0)
        protocol = json.loads(out.getvalue())
        self.assertEqual((protocol["kind"], protocol["modes"]), ("prefill_prefetch_model_ab", ["on", "off", "off", "on"]))
        with self.assertRaises(ValueError):
            common.execute(common.parser().parse_args([x if x != "cpu" else "auto" for x in base]))
        import check_ram_stability as ram
        self.assertEqual(ram.parser().parse_args(["--testbed", "i5", "--gpu", "cpu", "--cache", "64"]).prefill_prefetch, "on")

    def chat_record(self, mode):
        prompts, _ = model.workload_prompts("mixed")
        r = self.tier_record(mode)
        base = r["turns"][0]
        r["turns"], session = [], None
        for i, prompt in enumerate(prompts):
            turn = copy.deepcopy(base)
            follow = i % 4 == 1
            session = session if follow else f"s{i}"
            turn["session_id"] = session
            turn["messages"] = ([{"role": "user", "content": prompts[i - 1]}, {"role": "assistant", "content": "same"}]
                                if follow else []) + [{"role": "user", "content": prompt}]
            turn["response"]["usage"]["completion_tokens"] = 64
            turn["response"]["metrics"].update(decode_ms=100.0, decode_tokens_per_second=6.0,
                                               decode_expert_io_ms=1.0, cached_prompt_tokens=11 if follow else 0)
            r["turns"].append(turn)
        return r

    def test_chat_tier_block_validation(self):
        on, off = self.chat_record("on"), self.chat_record("off")
        model.validate_block(on, "on", workload="mixed", gpu_tier=True)
        model.validate_block(off, "off", workload="mixed", gpu_tier=True)
        model.compare_blocks(on, off, gpu_tier=True)
        self.assertEqual(model.measured_turns("mixed"), (4, 5, 6, 7))
        for change in ("history", "session", "traffic", "tokens", "dense"):
            r = self.chat_record("off")
            if change == "history":
                r["turns"][5]["messages"] = r["turns"][5]["messages"][-1:]
            elif change == "session":
                r["turns"][6]["session_id"] = r["turns"][5]["session_id"]
            elif change == "traffic":
                r["turns"][7]["response"]["metrics"]["expert_vram_loads"] = 1
            elif change == "tokens":
                r["turns"][3]["response"]["usage"]["completion_tokens"] = 65
            else:
                r["status_start"]["gpu"]["dense_offloaded"] = False
            with self.assertRaises(ValueError):
                model.validate_block(r, "off", workload="mixed", gpu_tier=True)

    def test_chat_tier_summary_and_dry_run(self):
        import io, contextlib, json
        import run_benchmark as common
        records = [self.chat_record(m) for m in ("on", "off", "off", "on")]
        pairs = model.paired_summary(records, turns=model.measured_turns("mixed"), gpu_tier=True)
        self.assertEqual(len(pairs), 8)
        self.assertEqual(pairs[0]["work"]["decode_tokens_per_second"], {"on": 6.0, "off": 6.0})
        base = ["--testbed", "i5", "--gpu", "auto", "--gpu-tier-model-ab", "--expert-batch-workload", "mixed",
                "--threads", "4", "--cache", "64", "--context", "512", "--max-tokens", "64",
                "--lookahead", "off", "--allow-dirty", "--dry-run"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(common.execute(common.parser().parse_args(base)), 0)
        protocol = json.loads(out.getvalue())
        self.assertEqual((protocol["total_requests"], protocol["measured_turns"], protocol["max_tokens"]),
                         (32, [4, 5, 6, 7], 64))
        max1 = list(base)
        max1[max1.index("--max-tokens") + 1] = "1"
        for bad in (max1,
                    [x if x != "--gpu-tier-model-ab" else "--backend-model-ab" for x in base]):
            with self.assertRaises(ValueError):
                common.execute(common.parser().parse_args(bad))

    def compute_record(self, mode):
        r = self.chat_record(mode)
        enabled = mode == "on"
        r["config"]["gpu_expert_compute"] = mode
        for status in [r["status_start"], *(t["status"] for t in r["turns"])]:
            status["gpu_expert_compute_enabled"] = enabled
        for t in r["turns"]:
            t["response"]["metrics"].update(expert_gpu_computes=40 if enabled else 0, expert_gpu_compute_ms=1.0,
                                           expert_vram_loads=0)
        return r

    def test_gpu_compute_block_validation_and_dry_run(self):
        import io, contextlib, json
        import run_benchmark as common
        on, off = self.compute_record("on"), self.compute_record("off")
        model.validate_gpu_compute_block(on, "on")
        model.validate_gpu_compute_block(off, "off")
        model.compare_blocks(on, off, gpu_tier=True, gpu_compute=True)
        with self.assertRaises(ValueError):
            model.compare_blocks(on, off, gpu_tier=True)  # compute key differs: only ignored for A1
        for change in ("status", "download", "never", "off_compute"):
            r = self.compute_record("off" if change == "off_compute" else "on")
            if change == "status":
                r["turns"][3]["status"]["gpu_expert_compute_enabled"] = False
            elif change == "download":
                r["turns"][2]["response"]["metrics"]["expert_vram_loads"] = 3
            elif change == "never":
                for t in r["turns"]:
                    t["response"]["metrics"]["expert_gpu_computes"] = 0
            else:
                r["turns"][1]["response"]["metrics"]["expert_gpu_computes"] = 2
            with self.assertRaises(ValueError):
                model.validate_gpu_compute_block(r, "off" if change == "off_compute" else "on")
        base = ["--testbed", "i5", "--gpu", "auto", "--gpu-compute-model-ab", "--expert-batch-workload", "mixed",
                "--threads", "4", "--cache", "64", "--context", "512", "--max-tokens", "64",
                "--lookahead", "off", "--allow-dirty", "--dry-run"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(common.execute(common.parser().parse_args(base)), 0)
        protocol = json.loads(out.getvalue())
        self.assertEqual((protocol["kind"], protocol["modes"], protocol["total_requests"]),
                         ("gpu_compute_model_ab", ["on", "off", "off", "on"], 32))
        for bad in (["--expert-batch-workload", "long-prefill"], ["--max-tokens", "1"]):
            argv = list(base)
            argv[argv.index(bad[0]) + 1] = bad[1]
            with self.assertRaises(ValueError):
                common.execute(common.parser().parse_args(argv))
        import check_ram_stability as ram
        self.assertEqual(ram.parser().parse_args(["--testbed", "i5", "--gpu", "auto", "--cache", "64"]).gpu_expert_compute, "off")


if __name__ == "__main__":
    unittest.main()
