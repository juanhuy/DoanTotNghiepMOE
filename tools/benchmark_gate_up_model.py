"""Fixed CPU INT8 Gate-Up ABBA protocol, reusing supervised RAM runner."""
import contextlib
import datetime
import json
import math
import os
from pathlib import Path
import uuid

import benchmark_gate_up as kernel
import check_ram_stability as ram
import run_benchmark as common

MODES = ("separate", "paired", "paired", "separate")
PINS = {
    "MOE_TEMPERATURE": "0", "MOE_TOP_P": "1", "MOE_TOP_K": "0",
    "MOE_REPETITION_PENALTY": "1", "MOE_KV_FP8": "0",
    "MOE_HOT_PIN_COUNT": "0", "MOE_PILOT_SMOOTH": "0.35",
}
REMOVED_ENV = ("MOE_SESSION_DIR", "ROUTE_TRACE", "MOE_ROUTE_TRACE",
               "MOE_USAGE_PATH", "MOE_PILOT_CONF_LIMIT")
WORK_COUNTERS = ("expert_cache_hits", "expert_cache_misses", "expert_cache_evictions",
                 "expert_file_loads", "expert_file_bytes_read")


@contextlib.contextmanager
def fixed_environment():
    old = {key: os.environ.get(key) for key in (*PINS, *REMOVED_ENV)}
    try:
        os.environ.update(PINS)
        for key in REMOVED_ENV:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_block(record, mode, expert_batch=False):
    gate_mode = "paired" if expert_batch else mode
    if record.get("status") != "completed" or record.get("watchdog_violations"):
        raise ValueError("block failed or watchdog stopped; stop protocol")
    turns = record.get("turns", [])
    if len(turns) != 2 or len({t["session_id"] for t in turns}) != 2:
        raise ValueError("expected two independent sessions")
    if not record.get("cpu_start_library_probe", {}).get("available"):
        raise ValueError("CPU library probe unavailable")
    if record["cpu_start_library_probe"]["cuda_libraries"]:
        raise ValueError("CPU block mapped CUDA libraries")
    for status in [record["status_start"], *(t["status"] for t in turns)]:
        if (status.get("gate_up_mode") != gate_mode or status.get("expert_quantization") != "int8"
                or status.get("kv_fp8") is not False or status.get("gpu_mode_at_load") != "cpu"
                or status.get("lookahead_enabled") is not False
                or status.get("kv_buffer_reuse_enabled") is not False
                or status.get("session_dir") is not None
                or status["gpu"]["enabled"] or status["gpu"]["allocated_vram_bytes"]):
            raise ValueError("actual mode/precision/backend mismatch")
    if expert_batch:
        enabled = mode == "on"
        for status in [record["status_start"], *(t["status"] for t in turns)]:
            if status.get("expert_batch_prefill_enabled") is not enabled:
                raise ValueError("expert batch actual mode mismatch")
        for turn in turns:
            if turn["response"]["usage"].get("completion_tokens") != 1:
                raise ValueError("max1 TTFT workload not observed")
            m=turn["response"]["metrics"]
            calls=[m.get("prefill_expert_batch2_calls"),m.get("prefill_expert_batch4_calls")]
            if (m.get("expert_batch_prefill_enabled") is not enabled
                    or any(type(c) is not int or c<0 for c in calls)
                    or (sum(calls)==0 if enabled else sum(calls)!=0)):
                raise ValueError("expert batch did not execute expected path")
    for turn, prompt in zip(turns, ("Where is Mars", "Explain a Python dictionary.")):
        if turn["messages"] != [{"role": "user", "content": prompt}]:
            raise ValueError("workload drift")
        metrics = turn["response"]["metrics"]
        if metrics.get("gate_up_mode") != gate_mode or metrics.get("cached_prompt_tokens") != 0:
            raise ValueError("request mode or prefix mismatch")
        trace = metrics.get("memory_trace")
        if (not isinstance(trace, dict) or trace.get("outcome") != "completed"
                or len(trace.get("samples", [])) != 7
                or trace != turn["status"].get("last_memory_trace")):
            raise ValueError("trace/status mismatch")


def compare_blocks(reference, actual, expert_batch=False):
    for key in ("source_sha256", "binary_sha256", "checkpoint"):
        if reference[key] != actual[key]:
            raise ValueError("source/binary/checkpoint drift")
    ignore = (("expert_batch_prefill" if expert_batch else "gate_up_mode"), "output_root")
    if ({k: v for k, v in reference["config"].items() if k not in ignore}
            != {k: v for k, v in actual["config"].items() if k not in ignore}):
        raise ValueError("config drift")
    for before, after in zip(reference["turns"], actual["turns"]):
        if before["messages"] != after["messages"]:
            raise ValueError("input drift")
        for key in ("text", "usage", "finish_reason"):
            if before["response"][key] != after["response"][key]:
                raise ValueError("HTTP output parity failed")
        for key in WORK_COUNTERS:
            if before["response"]["metrics"][key] != after["response"]["metrics"][key]:
                raise ValueError("expert workload changed")


def paired_summary(records, expert_batch=False):
    summary = []
    baseline_name,candidate_name = ("off","on") if expert_batch else ("separate","paired")
    keys = ("total_ms","prefill_ms","time_to_first_token_ms","expert_compute_ms") if expert_batch else (
        "total_ms","prefill_ms","decode_ms","time_to_first_token_ms","expert_compute_ms")
    for pair, (baseline, candidate) in enumerate(((records[0], records[1]), (records[3], records[2]))):
        for position, (a, b) in enumerate(zip(baseline["turns"], candidate["turns"])):
            metrics = {}
            for key in keys:
                av, bv = a["response"]["metrics"][key], b["response"]["metrics"][key]
                if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (av, bv)):
                    raise ValueError("invalid timer")
                metrics[key] = {baseline_name: av, candidate_name: bv, "baseline_over_candidate" if expert_batch else "separate_over_paired": av / bv}
            metrics["client_wall_ms"] = {baseline_name: a["response"]["client_wall_ms"],
                                          candidate_name: b["response"]["client_wall_ms"]}
            summary.append({"pair": pair, "prompt_position": position, "timers": metrics})
    return summary


def execute(args, root, expert_batch=False):
    expected_max = 1 if expert_batch else 64
    modes = ("off","on","on","off") if expert_batch else MODES
    if (args.gpu != "cpu" or args.threads != 4 or args.cache != 64 or args.context != 512
            or args.max_tokens != expected_max or args.lookahead != "off" or args.http_session
            or args.suite or args.case or args.compare_lookahead or args.compare_cache_policies):
        raise ValueError("model Gate-Up protocol requires CPU/threads4/cache64/context512/max64/lookahead off; no other suite")
    root = Path(root).resolve()
    model = common.absolute(args.model_dir, root)
    if (model / ".moe_usage").exists():
        raise ValueError("persistent .moe_usage exists; protocol requires a checkpoint without usage side state")
    env = os.environ.copy()
    commit = common.checked_text(["git", "rev-parse", "HEAD"], root, env)
    dirty = bool(common.checked_text(["git", "status", "--porcelain", "--untracked-files=normal"], root, env))
    if dirty and not args.allow_dirty:
        raise ValueError("dirty tree; --allow-dirty is exploratory only")
    protocol = {"kind": "expert_batch_model_ab" if expert_batch else "gate_up_model_ab", "testbed": args.testbed, "git_commit": commit,
                "git_dirty": dirty, "modes": modes, "model_dir": str(model),
                "requests_per_block": 2, "total_requests": 8, "threads": 4,
                "cache_mib_per_layer": 64, "context": 512, "max_tokens": expected_max,
                "workload": "kv-switch-v1", "environment_pins": PINS,
                "removed_environment_keys": REMOVED_ENV,
                "guards_mib": {"rss": 5120, "headroom": 1024, "watchdog_rss": 5632,
                               "available": 1536, "process_swap": 128},
                "notes": ["four fresh processes, fixed ABBA, two pairs only; not a population speedup estimate",
                          "checkpoint hashing warms OS cache; compare same binary with one selector changed",
                          "HTTP text/usage/finish parity plus expert workload; no HTTP logits/token-ID parity"]}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + ("-expert-batch-model-" if expert_batch else "-gate-up-model-") + uuid.uuid4().hex
    directory = common.absolute(args.output_root, root) / args.testbed / run_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {**protocol, "run_id": run_id, "status": "failed", "blocks": []}
    print("Artifact: " + str(directory), flush=True)
    records = []
    try:
        before = kernel.source_files(root)
        with fixed_environment():
            for index, mode in enumerate(modes):
                block_root = directory / f"block-{index}-{mode}"
                argv = ["--testbed", args.testbed, "--gpu", "cpu", "--lookahead", "off",
                        "--memory-trace", "on", "--kv-buffer-reuse", "off", "--gate-up-mode", "paired" if expert_batch else mode,
                        "--workload", "kv-switch", "--cache", "64", "--context", "512",
                        "--cycles", "2", "--max-tokens", str(expected_max), "--keep-cache", "--threads", "4",
                        "--model-dir", str(model), "--output-root", str(block_root),
                        "--ram-limit-mib", "5120", "--headroom-mib", "1024",
                        "--stop-rss-mib", "5632", "--min-available-mib", "1536", "--stop-swap-mib", "128"]
                if expert_batch:
                    argv.extend(["--expert-batch-prefill", mode])
                if args.allow_dirty:
                    argv.append("--allow-dirty")
                code = ram.execute(ram.parser().parse_args(argv))
                paths = list(block_root.rglob("result.json"))
                if len(paths) != 1:
                    raise ValueError("missing block result")
                current = json.loads(paths[0].read_text())
                record["blocks"].append({"mode": mode, "path": str(paths[0].relative_to(directory)),
                                         "run_id": current["run_id"], "status": current["status"]})
                if code:
                    raise ValueError("block failed; no further blocks")
                validate_block(current, mode, expert_batch=expert_batch)
                if records:
                    compare_blocks(records[0], current, expert_batch=expert_batch)
                records.append(current)
                if kernel.source_files(root) != before:
                    raise ValueError("source changed during protocol")
        record["pairs"] = paired_summary(records, expert_batch=expert_batch)
        if expert_batch:
            record["passes_ttft_rule"] = all(
                p["timers"]["time_to_first_token_ms"]["baseline_over_candidate"] >= 1.05
                and p["timers"]["prefill_ms"]["baseline_over_candidate"] >= 1.05
                for p in record["pairs"])
            record["decision_rule"] = "each pair/prompt TTFT and prefill ratio>=1.05; same outputs/work counters; limited exploratory window"

        record["status"] = "completed"
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        common.write_exclusive_json(directory / "result.json", record)
    print(record["status"] + ": " + str(directory / "result.json"), flush=True)
    return 0 if record["status"] == "completed" else 1
