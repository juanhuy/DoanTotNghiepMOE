#!/usr/bin/env python3
"""Runner protocol tests using fake subprocesses, never model/GPU measurements."""

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("run_benchmark", Path(__file__).with_name("run_benchmark.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"fixture": true}', encoding="utf-8")
        (self.model / "tokenizer.json").write_text('{}', encoding="utf-8")
        (self.model / "weights.safetensors").write_bytes(b"fixture weights")
        self.calls = []
        self.dirty = False
        self.build_code = 0
        self.benchmark_code = 0
        self.stdout = None
        self.output_text = "fixture answer"
        self.artifact = True
        self.patches = [patch.object(runner, "run_command", side_effect=self.command),
                        patch.object(runner, "hardware_observed", return_value={"logical_cpus": 7})]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def command(self, argv, root, env):
        self.calls.append((argv, env["RAYON_NUM_THREADS"]))
        code = 0
        error = ""
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            output = "a" * 40
        elif argv[:2] == ["git", "status"]:
            output = " M fixture" if self.dirty else ""
        elif argv[0] == "rustc":
            output = "rustc fixture\nhost: fixture"
        elif argv[0] == "cargo":
            code = self.build_code
            error = "fake compiler stderr"
            output = json.dumps({"reason": "compiler-artifact", "target": {"name": argv[5]},
                                 "executable": "/fake/current-binary"}) if self.artifact else ""
        else:
            code = self.benchmark_code
            error = "fake benchmark stderr"
            output = self.stdout if self.stdout is not None else json.dumps(self.report(argv))
        return subprocess.CompletedProcess(argv, code, output, error)

    def report(self, argv):
        timers = ("tokenizer_ms", "prefill_ms", "time_to_first_token_ms", "decode_ms", "total_ms",
                  "decode_tokens_per_second", "attention_ms", "expert_compute_ms", "lm_head_ms", "expert_io_ms")
        counters = ("expert_cache_hits", "expert_cache_misses", "expert_cache_evictions", "expert_bytes_read",
                    "expert_cache_bytes", "kv_cache_bytes", "cached_prompt_tokens")
        response = {"text": self.output_text, "prompt_tokens": 3, "token_ids": [1],
                    "finish_reason": "stop", "metrics": {key: 1.0 for key in timers}}
        response["metrics"].update({key: 0 for key in counters})
        report = {"schema_version": 1, "model_directory": argv[1], "load_ms": 1.0,
                  "gpu_mode": argv[7], "gpu_status_start": None, "gpu_status_end": None}
        settings = {"cache_bytes_per_layer": int(argv[5]) * 1024 * 1024, "context_limit": int(argv[6])}
        if Path(argv[3]).is_file():
            suite = json.loads(Path(argv[3]).read_text())
            selected = None if argv[4] == "*" else argv[4]
            report.update({"suite": suite, "selected_case": selected, "rounds": int(argv[2]),
                           "settings": settings, "runs": [
                               {"round": round_id, "case_id": case["id"], "response": response}
                               for round_id in range(int(argv[2])) for case in suite["cases"]
                               if selected is None or case["id"] == selected]})
        else:
            report.update(settings)
            report.update({"prompt": argv[2], "max_tokens": int(argv[3]), "runs": [
                {"index": index, "cache_state": "engine-cold" if index == 0 else "engine-warm",
                 "response": response} for index in range(int(argv[4]))]})
        return report

    def args(self, *extra):
        return runner.parser().parse_args(["--testbed", "i5", "--gpu", "cpu",
                                           "--model-dir", "model", *extra])

    def execute(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.execute(self.args(*extra), self.root)

    def records(self):
        return [json.loads(path.read_text()) for path in
                sorted((self.root / "benchmarks/runs/i5").glob("*/result.json"))]

    def test_explicit_machine_and_gpu_are_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in ([], ["--testbed", "i5"], ["--gpu", "cpu"],
                         ["--testbed", "auto", "--gpu", "cpu"]):
                with self.assertRaises(SystemExit):
                    runner.parser().parse_args(argv)

    def test_build_before_run_metadata_and_content_hashes(self):
        self.assertEqual(self.execute("--threads", "3", "--context", "256"), 0)
        record = self.records()[0]
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["testbed"], "i5")
        self.assertEqual(record["hardware_observed"], {"logical_cpus": 7})
        execution = record["execution"]
        self.assertEqual(execution["rayon_num_threads"], 3)
        self.assertFalse(execution["git_dirty"])
        self.assertIn("--locked", execution["build_argv"])
        self.assertEqual(execution["argv"], ["/fake/current-binary", str(self.model),
            "What is 2 + 2?", "16", "3", "128", "256", "cpu"])
        commands = [call[0][0] for call in self.calls]
        self.assertLess(commands.index("cargo"), commands.index("/fake/current-binary"))
        files = execution["checkpoint"]["files"]
        self.assertEqual(len(files), 3)
        config = next(entry for entry in files if entry["path"] == "config.json")
        self.assertEqual(config["sha256"], hashlib.sha256((self.model / "config.json").read_bytes()).hexdigest())
        self.assertEqual(execution["checkpoint"]["config_sha256"], config["sha256"])
        self.assertEqual(config["size"], (self.model / "config.json").stat().st_size)
        self.assertTrue(all(threads == "3" for _, threads in self.calls))

    def test_same_tag_runs_are_unique_and_do_not_overwrite(self):
        self.assertEqual(self.execute("--tag", "same"), 0)
        self.output_text = "new fixture answer"
        self.assertEqual(self.execute("--tag", "same"), 0)
        records = self.records()
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["run_id"] for record in records}), 2)
        texts = {record["result"]["runs"][0]["response"]["text"] for record in records}
        self.assertEqual(texts, {"fixture answer", "new fixture answer"})
        self.assertEqual(len(list(self.root.rglob("stderr.log"))), 2)
        self.assertEqual(len(list(self.root.rglob("*.tmp"))), 0)

    def test_failed_binary_has_diagnostics_but_no_successful_result(self):
        self.benchmark_code = 17
        self.assertEqual(self.execute(), 1)
        record = self.records()[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["execution"]["returncode"], 17)
        self.assertNotIn("result", record)
        self.assertIn("17", record["error"])
        log = next(self.root.rglob("stderr.log")).read_text()
        self.assertIn("fake benchmark stderr", log)

    def test_build_failure_or_missing_artifact_never_executes_old_binary(self):
        for build_code, artifact in ((1, True), (0, False)):
            self.calls.clear()
            self.build_code, self.artifact = build_code, artifact
            self.assertEqual(self.execute(), 1)
            self.assertNotIn("/fake/current-binary", [call[0][0] for call in self.calls])
        self.assertTrue(all("result" not in record for record in self.records()))

    def test_invalid_json_never_publishes_result(self):
        for output in ('not json', '[]', '{"bad":NaN}', '{}', '{"new":true}'):
            self.stdout = output
            self.assertEqual(self.execute(), 1)
        for record in self.records():
            self.assertEqual(record["status"], "failed")
            self.assertNotIn("result", record)
        self.assertEqual(len(list(self.root.rglob("stdout.failed.log"))), 5)

    def test_dirty_tree_requires_explicit_opt_in(self):
        self.dirty = True
        with self.assertRaisesRegex(ValueError, "--allow-dirty"):
            self.execute()
        self.assertEqual(self.records(), [])
        self.assertEqual(self.execute("--allow-dirty"), 0)
        self.assertTrue(self.records()[0]["execution"]["git_dirty"])

    def test_dry_run_needs_no_checkpoint_and_performs_no_build_or_hash(self):
        with patch.object(runner, "checkpoint_manifest", side_effect=AssertionError("must not hash")):
            self.assertEqual(self.execute("--dry-run", "--model-dir", "missing"), 0)
        self.assertEqual([argv[0] for argv, _ in self.calls], ["git", "git"])
        self.assertFalse((self.root / "benchmarks/runs").exists())

    def test_suite_hash_and_all_case_placeholder(self):
        suite = self.root / "suite.json"
        suite.write_text('{"version":1,"cases":[{"id":"hello","max_tokens":4,"messages":[{"role":"user","content":"hello"}]}]}')
        self.assertEqual(self.execute("--suite", "suite.json", "--cache", "64"), 0)
        execution = self.records()[0]["execution"]
        self.assertEqual(execution["argv"], ["/fake/current-binary", str(self.model),
            "3", str(suite), "*", "64", "512", "cpu"])
        self.assertEqual(execution["suite"]["sha256"], hashlib.sha256(suite.read_bytes()).hexdigest())
        self.assertIsNone(execution["config"]["max_tokens"])

    def test_report_settings_mismatch_is_failed_and_stdout_retained(self):
        argv = ["/fake/current-binary", str(self.model), "What is 2 + 2?", "16", "3", "128", "512", "cpu"]
        for key, value in (("cache_bytes_per_layer", 1), ("context_limit", 1),
                           ("gpu_mode", "auto"), ("runs", []), ("schema_version", 99)):
            report = self.report(argv)
            report[key] = value
            self.stdout = json.dumps(report)
            self.assertEqual(self.execute(), 1)
        self.assertTrue(all(record["status"] == "failed" and "result" not in record for record in self.records()))
        self.assertEqual(len(list(self.root.rglob("stdout.failed.log"))), 5)

    def test_report_invalid_metrics_and_tokens_are_failed(self):
        argv = ["/fake/current-binary", str(self.model), "What is 2 + 2?", "16", "3", "128", "512", "cpu"]
        for key, value in (("prompt_tokens", -1), ("token_ids", [True]), ("metrics", {})):
            report = self.report(argv)
            report["runs"][0]["response"][key] = value
            self.stdout = json.dumps(report)
            self.assertEqual(self.execute(), 1)

    def test_selected_suite_case_and_result_order_must_match(self):
        suite = self.root / "suite.json"
        suite.write_text('{"version":1,"cases":[{"id":"hello","max_tokens":4,"messages":[{"role":"user","content":"hello"}]},{"id":"bye","max_tokens":2,"messages":[{"role":"user","content":"bye"}]}]}')
        self.assertEqual(self.execute("--suite", "suite.json", "--case", "bye"), 0)
        self.assertEqual(len(self.records()[0]["result"]["runs"]), 3)
        argv = ["/fake/current-binary", str(self.model), "3", str(suite), "bye", "128", "512", "cpu"]
        report = self.report(argv)
        report["selected_case"] = "hello"
        self.stdout = json.dumps(report)
        self.assertEqual(self.execute("--suite", "suite.json", "--case", "bye"), 1)
        report = self.report(argv)
        report["runs"][0]["round"] = 2
        self.stdout = json.dumps(report)
        self.assertEqual(self.execute("--suite", "suite.json", "--case", "bye"), 1)

    def test_reject_case_without_suite_and_unsafe_tags(self):
        for extra in (("--case", "hello"), ("--tag", "../../oops")):
            with self.assertRaises(ValueError):
                self.execute(*extra)
        self.assertFalse((self.root / "benchmarks/runs").exists())

    def test_exclusive_atomic_publisher_preserves_existing_file(self):
        path = self.root / "result.json"
        runner.write_exclusive_json(path, {"first": True})
        with self.assertRaises(FileExistsError):
            runner.write_exclusive_json(path, {"second": True})
        self.assertEqual(json.loads(path.read_text()), {"first": True})
        self.assertFalse(path.with_name("result.json.tmp").exists())

    def test_checkpoint_hash_detects_changed_content(self):
        before = runner.checkpoint_manifest(self.model)
        (self.model / "weights.safetensors").write_bytes(b"changed weights")
        after = runner.checkpoint_manifest(self.model)
        self.assertNotEqual(before["files"], after["files"])


class HttpSessionStreamTests(unittest.TestCase):
    def test_memory_provenance_snapshot_is_independent_of_runner_state(self):
        import benchmark_http_session as session
        with patch.object(Path, "read_text", return_value="MemAvailable: 123 kB\nSwapFree: 456 kB\n"):
            self.assertEqual(session.memory_snapshot(), {"MemAvailable": 123, "SwapFree": 456})

    def parse(self, events):
        import benchmark_http_session as session
        data = b"".join(b"data: " + (e.encode() if isinstance(e, str) else json.dumps(e).encode())
                        + b"\n\n" for e in events)
        return session.read_stream(io.BytesIO(data))

    def terminal(self):
        return {"choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                "metrics": {"total_ms": 10, "time_to_first_token_ms": 8,
                            "decode_tokens_per_second": 0, "expert_io_ms": 3}}

    def test_complete_stream_keeps_content_and_metrics(self):
        result = self.parse([{"choices": [{"delta": {"content": "Sao Hỏa"}}]},
                             self.terminal(), "[DONE]"])
        self.assertEqual(result["text"], "Sao Hỏa")
        self.assertEqual(result["usage"]["completion_tokens"], 1)

    def test_generation_error_is_not_a_successful_measurement(self):
        with self.assertRaisesRegex(RuntimeError, "generation error"):
            self.parse([{"error": {"message": "failed"}}, "[DONE]"])

    def test_truncated_stream_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing finish or DONE"):
            self.parse([self.terminal()])
        with self.assertRaisesRegex(RuntimeError, "missing finish or DONE"):
            self.parse(["[DONE]"])

    def test_invalid_metric_is_rejected(self):
        terminal = self.terminal()
        terminal["metrics"]["total_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid metric"):
            self.parse([terminal, "[DONE]"])

    def test_extra_event_after_finish_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "after terminal"):
            self.parse([self.terminal(), self.terminal(), "[DONE]"])


class HttpScenarioTests(unittest.TestCase):
    def test_http_watchdog_terminates_only_owned_child_and_keeps_failure(self):
        import benchmark_http_session as session
        from unittest.mock import Mock
        for broken_sampler in (False, True):
            with tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                child = Mock(pid=123)
                child.poll.return_value = None
                watcher = session.MemoryWatchdog(child, directory)
                with patch.object(session, 'process_snapshot', return_value={'VmRSS': 6000 * 1024, 'VmSwap': 0},
                                  side_effect=RuntimeError('probe failed') if broken_sampler else None), \
                        patch.object(session, 'memory_snapshot', return_value={'MemAvailable': 4000 * 1024}):
                    watcher.thread.start()
                    watcher.thread.join(timeout=1)
                    self.assertFalse(watcher.thread.is_alive())
                    watcher.finish()
                    watcher.finish()  # Cleanup must be idempotent.
                child.terminate.assert_called_once_with()
                report = json.loads((directory / 'watchdog.json').read_text())
                self.assertEqual(len(report['violations']), 1)
                self.assertIn('watchdog', report['violations'][0]['reason'])

    def test_ram_watchdog_stops_on_pressure_or_missing_telemetry(self):
        import check_ram_stability as stability
        args = stability.parser().parse_args(['--testbed', 'i5', '--gpu', 'cpu', '--cache', '64'])
        normal = {'process': {'VmRSS': 1024, 'VmSwap': 0}, 'system': {'MemAvailable': 4 * 1024 * 1024}}
        self.assertIsNone(stability.pressure_reason(normal, args))
        for process, system in [({'VmRSS': 5633 * 1024}, normal['system']),
                                (normal['process'], {'MemAvailable': 1000}),
                                ({'VmRSS': 1024, 'VmSwap': 129 * 1024}, normal['system']),
                                ({}, normal['system']), (normal['process'], {})]:
            self.assertIsNotNone(stability.pressure_reason({'process': process, 'system': system}, args))

    def cycle(self, name, fail_at=None):
        import benchmark_http_session as session
        calls, resets, runs, warmups = [], [], [], []
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)

        def request(port, path, body=None):
            resets.append(path)
            return json.dumps({"expert_cache_policy": "victim"}).encode()

        def stream(port, messages, session_id, limit):
            calls.append(([dict(m) for m in messages], session_id, limit))
            if len(calls) == fail_at:
                raise RuntimeError("generation failed")
            return {"text": "answer", "metrics": {"total_ms": 1}}

        with patch.object(session, "request", side_effect=request), \
                patch.object(session, "stream_chat", side_effect=stream), \
                patch.object(session, "process_snapshot", return_value={}):
            if fail_at:
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    session.run_cycle(1, 1, "victim", 0, session.scenario_spec(name), 7, directory, runs, warmups)
            else:
                session.run_cycle(1, 1, "victim", 0, session.scenario_spec(name), 7, directory, runs, warmups)
        return calls, resets, runs, warmups, directory

    def test_warmup_is_excluded_and_new_session_prevents_kv_reuse(self):
        calls, resets, runs, warmups, directory = self.cycle("warm")
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(warmups), 1)
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertNotEqual(calls[0][1], calls[1][1])
        self.assertEqual(len([p for p in resets if "reset?" in p]), 1)
        self.assertTrue((directory / "warmup-0-0.json").exists())
        self.assertTrue((directory / "turn-0-0.json").exists())
        self.assertIsNone(runs[0]["process_read_bytes_delta"])

    def test_conversation_replays_assistant_in_same_session(self):
        calls, _, runs, warmups, _ = self.cycle("conversation")
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual([m["role"] for m in calls[1][0]], ["user", "assistant", "user"])
        self.assertEqual(runs[1]["messages"][1]["content"], "answer")
        self.assertEqual(warmups, [])

    def test_topics_keep_expert_cache_but_use_independent_histories(self):
        calls, resets, runs, _, _ = self.cycle("topics")
        self.assertEqual(len(runs), 4)
        self.assertEqual(len({call[1] for call in calls}), 4)
        self.assertTrue(all(len(call[0]) == 1 and call[2] == 7 for call in calls))
        self.assertEqual(len([p for p in resets if "reset?" in p]), 1)

    def test_cold_and_legacy_workload(self):
        import benchmark_http_session as session
        calls, _, runs, warmups, _ = self.cycle("cold")
        self.assertEqual((len(calls), len(runs), len(warmups)), (1, 1, 0))
        self.assertEqual(session.scenario_spec(), session.scenario_spec("conversation"))

    def test_failure_keeps_completed_sample(self):
        _, _, runs, _, directory = self.cycle("conversation", fail_at=2)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(list(directory.glob("turn-*.json"))), 1)

    def test_policy_order_alternates_and_http_flag_is_required(self):
        import benchmark_http_session as session
        args = runner.parser().parse_args(["--testbed", "i5", "--gpu", "cpu", "--runs", "2",
                                          "--compare-cache-policies", "--http-session"])
        self.assertEqual(session.policy_plan(args), ["inclusive", "victim", "victim", "inclusive"])
        args = runner.parser().parse_args(["--testbed", "i5", "--gpu", "cpu", "--http-scenario", "cold"])
        with self.assertRaisesRegex(ValueError, "requires --http-session"):
            runner.execute(args)

    def test_cache_sweep_forwards_settings_and_stops_on_failure(self):
        import run_scenarios as scenarios
        args = scenarios.parser().parse_args(["cache-sweep", "--testbed", "i5", "--gpu", "cpu",
                                             "--threads", "2", "--runs", "1", "--allow-dirty"])
        commands = scenarios.commands(args)
        self.assertEqual([c[c.index("--cache") + 1] for c in commands], ["128", "192", "256"])
        for command in commands:
            forwarded = runner.parser().parse_args(command[2:])
            self.assertEqual((forwarded.http_scenario, forwarded.threads, forwarded.max_tokens),
                             ("conversation", 2, 128))
            self.assertTrue(forwarded.compare_cache_policies)
        with patch.object(scenarios.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0),
                                                                  subprocess.CompletedProcess([], 1)]) as run:
            self.assertEqual(scenarios.execute(args), 1)
            self.assertEqual(run.call_count, 2)

    def test_single_policy_and_custom_workload_guard(self):
        import run_scenarios as scenarios
        base = ["warm", "--testbed", "i5", "--gpu", "cpu"]
        args = scenarios.parser().parse_args(base + ["--single-policy"])
        self.assertNotIn("--compare-cache-policies", scenarios.commands(args)[0])
        for extra in (["--prompt", "custom"], ["--tag", "custom"], ["--suite", "custom.json"]):
            with self.assertRaises(ValueError):
                scenarios.commands(scenarios.parser().parse_args(base + extra))


if __name__ == "__main__":
    unittest.main()
