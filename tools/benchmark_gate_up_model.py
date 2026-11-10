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
# Backend A/B changes dense compute AND adds the VRAM expert tier, so these differ by design;
# they are recorded per sample instead of being required equal.
BACKEND_COUNTERS = (*WORK_COUNTERS, "expert_io_ms", "expert_file_read_ms", "expert_vram_loads",
                    "expert_vram_upload_bytes", "expert_vram_download_bytes",
                    "expert_vram_upload_ms", "expert_vram_download_ms")
DECODE_COUNTERS = ("decode_tokens_per_second", "decode_ms", "decode_expert_io_ms", "cached_prompt_tokens")
PREFETCH_COUNTERS = ("prefill_prefetch_loads", "prefill_prefetch_bytes", "prefill_prefetch_load_errors",
                     "expert_io_ms", "expert_file_read_ms", "prefill_expert_io_ms")


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


# S5 chat/decode protocol: two mixed groups per process, group 1 warmup, group 2 measured.
CHAT_GROUPS = 2


def workload_prompts(workload):
    if workload == "kv-switch":
        return ("Where is Mars", "Explain a Python dictionary."), None
    if workload == "mixed":
        return tuple(ram.workload_spec("mixed")["prompts"]) * CHAT_GROUPS, None
    spec = ram.workload_spec(workload)
    if workload != "long-prefill":
        raise ValueError("expert batch model workload must be kv-switch or long-prefill")
    return tuple(spec["prompts"]), tuple(spec["expected_prompt_tokens"])


def measured_turns(workload):
    if workload == "mixed":
        group = len(ram.workload_spec("mixed")["prompts"])
        return tuple(range(group * (CHAT_GROUPS - 1), group * CHAT_GROUPS))
    return (0, 1) if workload == "kv-switch" else tuple(ram.workload_spec(workload)["measured_turns"])


def validate_chat_tier_block(record, mode):
    """GPU auto, tier selector, mixed chat with follow-up history, max64 decode."""
    prompts, _ = workload_prompts("mixed")
    if record.get("status") != "completed" or record.get("watchdog_violations"):
        raise ValueError("block failed or watchdog stopped; stop protocol")
    turns = record.get("turns", [])
    if len(turns) != len(prompts):
        raise ValueError("expected two mixed groups")
    enabled = mode == "on"
    for status in [record["status_start"], *(t["status"] for t in turns)]:
        gpu = status["gpu"]
        if (status.get("gate_up_mode") != "paired" or status.get("expert_quantization") != "int8"
                or status.get("kv_fp8") is not False or status.get("gpu_mode_at_load") != "auto"
                or status.get("lookahead_enabled") is not False
                or status.get("kv_buffer_reuse_enabled") is not False
                or status.get("expert_batch_prefill_enabled") is not False
                or status.get("session_dir") is not None
                or status.get("gpu_expert_tier_enabled") is not enabled
                or gpu["enabled"] is not True or gpu.get("dense_offloaded") is not True
                or not gpu["allocated_vram_bytes"]):
            raise ValueError("actual mode/precision/backend/tier mismatch")
    for index, (turn, prompt) in enumerate(zip(turns, prompts)):
        follow_up = index % 4 == 1
        if turn["messages"][-1] != {"role": "user", "content": prompt}:
            raise ValueError("workload drift")
        if follow_up != (len(turn["messages"]) == 3) or follow_up != (turn["session_id"] == turns[index - 1]["session_id"]):
            raise ValueError("follow-up session/history structure mismatch")
        if not follow_up and index and turn["session_id"] in {t["session_id"] for t in turns[:index]}:
            raise ValueError("expected a new session")
        m = turn["response"]["metrics"]
        if (m.get("gate_up_mode") != "paired" or m.get("prefill_expert_batch2_calls")
                or m.get("prefill_expert_batch4_calls") or not 0 < turn["response"]["usage"]["completion_tokens"] <= 64):
            raise ValueError("request mode or token limit mismatch")
        if not enabled and (m.get("expert_vram_upload_bytes") or m.get("expert_vram_loads")
                            or turn["status"]["gpu"].get("vram_experts_resident")):
            raise ValueError("tier off but VRAM expert traffic observed")
        trace = m.get("memory_trace")
        if (not isinstance(trace, dict) or trace.get("outcome") != "completed"
                or trace != turn["status"].get("last_memory_trace")):
            raise ValueError("trace/status mismatch")


def validate_block(record, mode, expert_batch=False, workload="kv-switch", backend=False, gpu_tier=False,
                   prefetch=False):
    if prefetch:
        # CPU, batch off, max1 path shared with the batch protocol; then the prefetch selector.
        validate_block(record, "off", expert_batch=True, workload=workload)
        enabled = mode == "on"
        for status in [record["status_start"], *(t["status"] for t in record["turns"])]:
            if status.get("prefill_prefetch_enabled") is not enabled:
                raise ValueError("prefill prefetch actual mode mismatch")
        for turn in record["turns"]:
            m = turn["response"]["metrics"]
            loads = m.get("prefill_prefetch_loads")
            if (m.get("prefill_prefetch_enabled") is not enabled or type(loads) is not int
                    or (loads <= 0 if enabled else loads != 0) or m.get("prefill_prefetch_load_errors")):
                raise ValueError("prefill prefetch did not execute expected path")
        return
    if gpu_tier and workload == "mixed":
        return validate_chat_tier_block(record, mode)
    if gpu_tier:
        validate_block(record, "auto", workload=workload, backend=True)
        enabled = mode == "on"
        turns = record["turns"]
        for status in [record["status_start"], *(t["status"] for t in turns)]:
            if status.get("gpu_expert_tier_enabled") is not enabled:
                raise ValueError("GPU expert tier actual mode mismatch")
        if not enabled:
            for turn in turns:
                m = turn["response"]["metrics"]
                if (m.get("expert_vram_upload_bytes") or m.get("expert_vram_loads")
                        or turn["status"]["gpu"].get("vram_experts_resident")):
                    raise ValueError("tier off but VRAM expert traffic observed")
        return
    gate_mode = "paired" if (expert_batch or backend) else mode
    gpu_mode = mode if backend else "cpu"
    prompts, expected_tokens = workload_prompts(workload)
    if record.get("status") != "completed" or record.get("watchdog_violations"):
        raise ValueError("block failed or watchdog stopped; stop protocol")
    turns = record.get("turns", [])
    if len(turns) != len(prompts) or len({t["session_id"] for t in turns}) != len(prompts):
        raise ValueError("expected independent sessions, one per request")
    if gpu_mode == "cpu":
        if not record.get("cpu_start_library_probe", {}).get("available"):
            raise ValueError("CPU library probe unavailable")
        if record["cpu_start_library_probe"]["cuda_libraries"]:
            raise ValueError("CPU block mapped CUDA libraries")
    for status in [record["status_start"], *(t["status"] for t in turns)]:
        gpu = status["gpu"]
        backend_ok = (not gpu["enabled"] and not gpu["allocated_vram_bytes"]) if gpu_mode == "cpu" else (
            gpu["enabled"] is True and gpu.get("dense_offloaded") is True and gpu["allocated_vram_bytes"] > 0)
        if (status.get("gate_up_mode") != gate_mode or status.get("expert_quantization") != "int8"
                or status.get("kv_fp8") is not False or status.get("gpu_mode_at_load") != gpu_mode
                or status.get("lookahead_enabled") is not False
                or status.get("kv_buffer_reuse_enabled") is not False
                or status.get("session_dir") is not None or not backend_ok):
            raise ValueError("actual mode/precision/backend mismatch")
    if backend:
        for status in [record["status_start"], *(t["status"] for t in turns)]:
            if status.get("expert_batch_prefill_enabled") is not False:
                raise ValueError("backend A/B requires batch prefill off")
        for turn in turns:
            m = turn["response"]["metrics"]
            if (turn["response"]["usage"].get("completion_tokens") != 1
                    or m.get("prefill_expert_batch2_calls") or m.get("prefill_expert_batch4_calls")):
                raise ValueError("max1/batch-off backend workload not observed")
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
    for index, (turn, prompt) in enumerate(zip(turns, prompts)):
        if turn["messages"] != [{"role": "user", "content": prompt}]:
            raise ValueError("workload drift")
        if expected_tokens and turn["response"]["usage"].get("prompt_tokens") != expected_tokens[index]:
            raise ValueError("prompt token count drift")
        metrics = turn["response"]["metrics"]
        if metrics.get("gate_up_mode") != gate_mode or metrics.get("cached_prompt_tokens") != 0:
            raise ValueError("request mode or prefix mismatch")
        trace = metrics.get("memory_trace")
        if (not isinstance(trace, dict) or trace.get("outcome") != "completed"
                or len(trace.get("samples", [])) != 7
                or trace != turn["status"].get("last_memory_trace")):
            raise ValueError("trace/status mismatch")


def compare_blocks(reference, actual, expert_batch=False, backend=False, gpu_tier=False, prefetch=False):
    for key in ("source_sha256", "binary_sha256", "checkpoint"):
        if reference[key] != actual[key]:
            raise ValueError("source/binary/checkpoint drift")
    ignore = (("prefill_prefetch" if prefetch else "gpu_expert_tier" if gpu_tier else "gpu" if backend else "expert_batch_prefill" if expert_batch else "gate_up_mode"), "output_root")
    if ({k: v for k, v in reference["config"].items() if k not in ignore}
            != {k: v for k, v in actual["config"].items() if k not in ignore}):
        raise ValueError("config drift")
    for before, after in zip(reference["turns"], actual["turns"]):
        if before["messages"] != after["messages"]:
            raise ValueError("input drift")
        for key in ("text", "usage", "finish_reason"):
            if before["response"][key] != after["response"][key]:
                raise ValueError("HTTP output parity failed")
        for key in () if (backend or gpu_tier) else WORK_COUNTERS:
            if before["response"]["metrics"][key] != after["response"]["metrics"][key]:
                raise ValueError("expert workload changed")


def paired_summary(records, expert_batch=False, turns=(0, 1), backend=False, gpu_tier=False, prefetch=False):
    summary = []
    counters = PREFETCH_COUNTERS if prefetch else BACKEND_COUNTERS + DECODE_COUNTERS if gpu_tier else BACKEND_COUNTERS
    if gpu_tier or prefetch:
        backend = True
    baseline_name,candidate_name = ("on","off") if (gpu_tier or prefetch) else ("cpu","auto") if backend else ("off","on") if expert_batch else ("separate","paired")
    keys = ("total_ms","prefill_ms","time_to_first_token_ms","expert_compute_ms","attention_ms","lm_head_ms") if backend else (
        "total_ms","prefill_ms","time_to_first_token_ms","expert_compute_ms") if expert_batch else (
        "total_ms","prefill_ms","decode_ms","time_to_first_token_ms","expert_compute_ms")
    ratio_name = "baseline_over_candidate" if (expert_batch or backend) else "separate_over_paired"
    for pair, (baseline, candidate) in enumerate(((records[0], records[1]), (records[3], records[2]))):
        for position, index in enumerate(turns):
            a, b = baseline["turns"][index], candidate["turns"][index]
            metrics = {}
            for key in keys:
                av, bv = a["response"]["metrics"][key], b["response"]["metrics"][key]
                if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (av, bv)):
                    raise ValueError("invalid timer")
                metrics[key] = {baseline_name: av, candidate_name: bv, ratio_name: av / bv}
            metrics["client_wall_ms"] = {baseline_name: a["response"]["client_wall_ms"],
                                          candidate_name: b["response"]["client_wall_ms"]}
            entry = {"pair": pair, "prompt_position": position, "turn": index, "timers": metrics}
            if backend:
                entry["work"] = {key: {baseline_name: a["response"]["metrics"].get(key),
                                       candidate_name: b["response"]["metrics"].get(key)}
                                 for key in counters}
            summary.append(entry)
    return summary


def execute(args, root, expert_batch=False, backend=False, gpu_tier=False, prefetch=False):
    workload = getattr(args, "expert_batch_workload", "kv-switch") or "kv-switch"
    if sum((expert_batch, backend, gpu_tier, prefetch)) > 1:
        raise ValueError("choose one model A/B selector")
    chat = gpu_tier and workload == "mixed"
    if workload == "mixed" and not gpu_tier:
        raise ValueError("mixed chat workload is only defined for --gpu-tier-model-ab")
    if backend or gpu_tier or prefetch:
        if expert_batch or (workload != "long-prefill" and not chat):
            raise ValueError("backend A/B uses long-prefill workload only (GPU tier also mixed chat), batch prefill off")
    elif workload != "kv-switch" and not expert_batch:
        raise ValueError("--expert-batch-workload applies only to --expert-batch-model-ab")
    prompts, expected_tokens = workload_prompts(workload)
    measured = measured_turns(workload)
    expected_max = 64 if chat else 1 if (expert_batch or backend or gpu_tier or prefetch) else 64
    modes = ("on","off","off","on") if (gpu_tier or prefetch) else ("cpu","auto","auto","cpu") if backend else ("off","on","on","off") if expert_batch else MODES
    if ((args.gpu != ("auto" if (backend or gpu_tier) else "cpu")) or args.threads != 4 or args.cache != 64 or args.context != 512
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
    protocol = {"kind": "prefill_prefetch_model_ab" if prefetch else "gpu_tier_model_ab" if gpu_tier else "backend_model_ab" if backend else "expert_batch_model_ab" if expert_batch else "gate_up_model_ab", "testbed": args.testbed, "git_commit": commit,
                "git_dirty": dirty, "modes": modes, "model_dir": str(model),
                "requests_per_block": len(prompts), "total_requests": 4 * len(prompts), "threads": 4,
                "cache_mib_per_layer": 64, "context": 512, "max_tokens": expected_max,
                "workload": workload + "-v1", "measured_turns": list(measured),
                "warmup_turns": [i for i in range(len(prompts)) if i not in measured],
                "expected_prompt_tokens": list(expected_tokens) if expected_tokens else None,
                "environment_pins": PINS,
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
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + ("-prefetch-model-" if prefetch else "-gpu-tier-model-" if gpu_tier else "-backend-model-" if backend else "-expert-batch-model-" if expert_batch else "-gate-up-model-") + uuid.uuid4().hex
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
                gpu_arg = "auto" if gpu_tier else mode if backend else "cpu"
                argv = ["--testbed", args.testbed, "--gpu", gpu_arg, "--lookahead", "off",
                        "--memory-trace", "on", "--kv-buffer-reuse", "off",
                        "--gate-up-mode", "paired" if (expert_batch or backend or gpu_tier or prefetch) else mode,
                        "--workload", workload, "--cache", "64", "--context", "512",
                        "--cycles", str(len(prompts)), "--max-tokens", str(expected_max), "--keep-cache", "--threads", "4",
                        "--model-dir", str(model), "--output-root", str(block_root),
                        "--ram-limit-mib", "5120", "--headroom-mib", "1024",
                        "--stop-rss-mib", "5632", "--min-available-mib", "1536", "--stop-swap-mib", "128"]
                if expert_batch:
                    argv.extend(["--expert-batch-prefill", mode])
                if backend or gpu_tier or prefetch:
                    argv.extend(["--expert-batch-prefill", "off"])
                if prefetch:
                    argv.extend(["--prefill-prefetch", mode])
                if gpu_tier:
                    argv.extend(["--gpu-expert-tier", mode])
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
                validate_block(current, mode, expert_batch=expert_batch, workload=workload, backend=backend,
                               gpu_tier=gpu_tier, prefetch=prefetch)
                if records:
                    compare_blocks(records[0], current, expert_batch=expert_batch, backend=backend, gpu_tier=gpu_tier,
                                   prefetch=prefetch)
                records.append(current)
                if kernel.source_files(root) != before:
                    raise ValueError("source changed during protocol")
        record["pairs"] = paired_summary(records, expert_batch=expert_batch, turns=measured, backend=backend,
                                         gpu_tier=gpu_tier, prefetch=prefetch)
        if record["warmup_turns"]:
            record["warmup_pairs_excluded"] = paired_summary(records, expert_batch=expert_batch,
                                                             turns=tuple(record["warmup_turns"]), backend=backend,
                                                             gpu_tier=gpu_tier, prefetch=prefetch)
        if prefetch:
            ratios = [(p["timers"]["time_to_first_token_ms"]["baseline_over_candidate"],
                       p["timers"]["prefill_ms"]["baseline_over_candidate"]) for p in record["pairs"]]
            record["prefetch_helps"] = all(t <= 1 / 1.05 and f <= 1 / 1.05 for t, f in ratios)
            record["prefetch_hurts"] = all(t >= 1.05 and f >= 1.05 for t, f in ratios)
            record["decision_rule"] = ("on/off TTFT and prefill ratio<=1/1.05 in all 4 => prefetch helps; >=1.05 in all 4 "
                                       "=> prefetch hurts; otherwise no consistent difference. Same work counters required")
        if chat:
            record["gpu_status_start"] = {r["status_start"]["gpu_expert_tier_enabled"]: r["status_start"]["gpu"] for r in records}

            def decode_ratio(p):
                d = p["work"]["decode_ms"]
                if any(type(v) not in (int, float) or not v > 0 for v in d.values()):
                    raise ValueError("invalid decode timer")
                return d["on"] / d["off"]
            ratios = [(p["timers"]["total_ms"]["baseline_over_candidate"], decode_ratio(p)) for p in record["pairs"]]
            record["off_faster"] = all(t >= 1.05 and d >= 1.05 for t, d in ratios)
            record["on_faster"] = all(t <= 1 / 1.05 and d <= 1 / 1.05 for t, d in ratios)
            record["decision_rule"] = ("chat max64: tier on/off total_ms and decode_ms ratio>=1.05 in all 8 => off faster; "
                                       "<=1/1.05 in all 8 => on faster; otherwise no consistent difference. GPU dense both sides")
        elif gpu_tier:
            record["gpu_status_start"] = {r["status_start"]["gpu_expert_tier_enabled"]: r["status_start"]["gpu"] for r in records}
            record["off_faster"] = all(
                p["timers"]["time_to_first_token_ms"]["baseline_over_candidate"] >= 1.05
                and p["timers"]["prefill_ms"]["baseline_over_candidate"] >= 1.05 for p in record["pairs"])
            record["on_faster"] = all(
                p["timers"]["time_to_first_token_ms"]["baseline_over_candidate"] <= 1 / 1.05
                and p["timers"]["prefill_ms"]["baseline_over_candidate"] <= 1 / 1.05 for p in record["pairs"])
            record["decision_rule"] = ("tier on/off TTFT and prefill ratio>=1.05 in all 4 => off faster; <=1/1.05 in all 4 "
                                       "=> on faster; otherwise no consistent difference. GPU dense on both sides")
        if backend:
            record["gpu_status_start"] = {r["status_start"]["gpu_mode_at_load"]: r["status_start"]["gpu"] for r in records}
            record["passes_backend_rule"] = all(
                p["timers"]["time_to_first_token_ms"]["baseline_over_candidate"] >= 1.05
                and p["timers"]["prefill_ms"]["baseline_over_candidate"] >= 1.05
                for p in record["pairs"])
            record["decision_rule"] = ("each pair/prompt TTFT and prefill cpu/auto ratio>=1.05; same text/usage/finish; "
                                       "composite backend (GPU dense + VRAM expert tier), expert SwiGLU stays CPU")
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
