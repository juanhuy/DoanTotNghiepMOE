"""A3 thread sweep (threads-chat-v1): recommended config, only RAYON threads change between fresh processes.

Palindrome order over a per-testbed thread set (i5 4,6,8,12,12,8,6,4), mixed chat x2 groups per process
(group 1 warmup, group 2 measured), GPU dense on, VRAM expert tier off. Each N is compared with the
4-thread baseline in two mirrored pairs (first and last occurrences) over four prompts = 8 comparisons.
"""
import datetime
import json
import statistics
import uuid

import benchmark_gate_up as kernel
import benchmark_gate_up_model as model_ab
import check_ram_stability as ram
import run_benchmark as common

BASELINE = 4
THREAD_SETS = {"i5": (4, 6, 8, 12), "i7": (4, 8, 16, 24)}
RATIO_KEYS = ("total_ms", "decode_ms", "time_to_first_token_ms", "expert_compute_ms", "attention_ms")
RULE = ("N vs 4 threads: total_ms and decode_ms ratio 4/N >= 1.05 in all 8 => N faster; <= 1/1.05 in all 8 "
        "=> N slower; otherwise no consistent difference. Recommended = fastest-median-decode among N faster, else 4")


def order(thread_set):
    return (*thread_set, *reversed(thread_set))


def compare_blocks(reference, actual):
    for key in ("source_sha256", "binary_sha256", "checkpoint"):
        if reference[key] != actual[key]:
            raise ValueError("source/binary/checkpoint drift")
    ignore = ("threads", "output_root")
    if ({k: v for k, v in reference["config"].items() if k not in ignore}
            != {k: v for k, v in actual["config"].items() if k not in ignore}):
        raise ValueError("config drift")
    for before, after in zip(reference["turns"], actual["turns"]):
        if before["messages"] != after["messages"]:
            raise ValueError("input drift")
        for key in ("text", "usage", "finish_reason"):
            if before["response"][key] != after["response"][key]:
                raise ValueError("HTTP output parity failed across thread counts")
        for key in model_ab.WORK_COUNTERS:
            if before["response"]["metrics"][key] != after["response"]["metrics"][key]:
                raise ValueError("expert workload changed across thread counts")


def timer(turn, key):
    value = turn["response"]["metrics"][key]
    if type(value) not in (int, float) or not value > 0:
        raise ValueError("invalid timer " + key)
    return value


def summarize(records, sequence, measured):
    first = {n: sequence.index(n) for n in set(sequence)}
    last = {n: len(sequence) - 1 - sequence[::-1].index(n) for n in set(sequence)}
    per_threads = {}
    for n in sorted(set(sequence)):
        rates = [timer(records[i]["turns"][t], "decode_tokens_per_second")
                 for i in (first[n], last[n]) for t in measured]
        per_threads[str(n)] = {"median_decode_tokens_per_second": statistics.median(rates),
                               "median_total_ms": statistics.median(
                                   timer(records[i]["turns"][t], "total_ms") for i in (first[n], last[n]) for t in measured)}
    comparisons = {}
    for n in sorted(set(sequence) - {BASELINE}):
        pairs = []
        for pair, (b, c) in enumerate(((first[BASELINE], first[n]), (last[BASELINE], last[n]))):
            for position, t in enumerate(measured):
                a, z = records[b]["turns"][t], records[c]["turns"][t]
                ratios = {k: {"base": timer(a, k), "candidate": timer(z, k), "base_over_candidate": timer(a, k) / timer(z, k)}
                          for k in RATIO_KEYS}
                pairs.append({"pair": pair, "prompt_position": position, "turn": t, "blocks": [b, c], "timers": ratios,
                              "decode_tokens_per_second": {"base": timer(a, "decode_tokens_per_second"),
                                                           "candidate": timer(z, "decode_tokens_per_second")}})
        both = [(p["timers"]["total_ms"]["base_over_candidate"], p["timers"]["decode_ms"]["base_over_candidate"]) for p in pairs]
        comparisons[str(n)] = {"pairs": pairs,
                               "faster_than_4": all(t >= 1.05 and d >= 1.05 for t, d in both),
                               "slower_than_4": all(t <= 1 / 1.05 and d <= 1 / 1.05 for t, d in both)}
    faster = [int(n) for n, c in comparisons.items() if c["faster_than_4"]]
    recommended = max(faster, key=lambda n: per_threads[str(n)]["median_decode_tokens_per_second"]) if faster else BASELINE
    return per_threads, comparisons, recommended


def execute(args, root):
    if args.testbed not in THREAD_SETS:
        raise ValueError("thread sweep needs --testbed i5 or i7")
    if (args.gpu != "auto" or args.cache != 64 or args.context != 512 or args.max_tokens != 64 or args.lookahead != "off"
            or getattr(args, "expert_batch_workload", None) != "mixed" or args.http_session or args.suite or args.case
            or args.compare_lookahead or args.compare_cache_policies):
        raise ValueError("threads-chat-v1 requires --gpu auto --cache 64 --context 512 --max-tokens 64 --lookahead off "
                         "--expert-batch-workload mixed; no other suite")
    sequence = order(THREAD_SETS[args.testbed])
    prompts, _ = model_ab.workload_prompts("mixed")
    measured = model_ab.measured_turns("mixed")
    root = common.Path(root).resolve()
    model = common.absolute(args.model_dir, root)
    if (model / ".moe_usage").exists():
        raise ValueError("persistent .moe_usage exists; protocol requires a checkpoint without usage side state")
    env = common.os.environ.copy()
    commit = common.checked_text(["git", "rev-parse", "HEAD"], root, env)
    dirty = bool(common.checked_text(["git", "status", "--porcelain", "--untracked-files=normal"], root, env))
    if dirty and not args.allow_dirty:
        raise ValueError("dirty tree; --allow-dirty is exploratory only")
    protocol = {"kind": "thread_sweep_model", "protocol": "threads-chat-v1", "testbed": args.testbed,
                "git_commit": commit, "git_dirty": dirty, "thread_order": list(sequence), "baseline_threads": BASELINE,
                "model_dir": str(model), "requests_per_block": len(prompts), "total_requests": len(sequence) * len(prompts),
                "gpu": "auto", "gpu_expert_tier": "off", "cache_mib_per_layer": 64, "context": 512, "max_tokens": 64,
                "workload": "mixed-v1", "measured_turns": list(measured),
                "warmup_turns": [i for i in range(len(prompts)) if i not in measured],
                "environment_pins": model_ab.PINS, "removed_environment_keys": model_ab.REMOVED_ENV,
                "guards_mib": {"rss": 5120, "headroom": 1024, "watchdog_rss": 5632, "available": 1536, "process_swap": 128},
                "decision_rule": RULE,
                "notes": ["fresh process per block, one palindrome run; not a population estimate",
                          "only RAYON_NUM_THREADS changes; dense on GPU, expert SwiGLU on CPU",
                          "HTTP text/usage/finish parity plus expert work counters across thread counts"]}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-threads-" + uuid.uuid4().hex
    directory = common.absolute(args.output_root, root) / args.testbed / run_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {**protocol, "run_id": run_id, "status": "failed", "blocks": []}
    print("Artifact: " + str(directory), flush=True)
    records = []
    try:
        before = kernel.source_files(root)
        with model_ab.fixed_environment():
            for index, threads in enumerate(sequence):
                block_root = directory / f"block-{index}-t{threads}"
                argv = ["--testbed", args.testbed, "--gpu", "auto", "--lookahead", "off", "--memory-trace", "on",
                        "--kv-buffer-reuse", "off", "--gate-up-mode", "paired", "--workload", "mixed",
                        "--cache", "64", "--context", "512", "--cycles", str(len(prompts)), "--max-tokens", "64",
                        "--keep-cache", "--threads", str(threads), "--model-dir", str(model),
                        "--output-root", str(block_root), "--ram-limit-mib", "5120", "--headroom-mib", "1024",
                        "--stop-rss-mib", "5632", "--min-available-mib", "1536", "--stop-swap-mib", "128",
                        "--expert-batch-prefill", "off", "--gpu-expert-tier", "off"]
                if args.allow_dirty:
                    argv.append("--allow-dirty")
                code = ram.execute(ram.parser().parse_args(argv))
                paths = list(block_root.rglob("result.json"))
                if len(paths) != 1:
                    raise ValueError("missing block result")
                current = json.loads(paths[0].read_text())
                record["blocks"].append({"threads": threads, "path": str(paths[0].relative_to(directory)),
                                         "run_id": current["run_id"], "status": current["status"]})
                if code:
                    raise ValueError("block failed; no further blocks")
                if current["config"].get("threads") != threads:
                    raise ValueError("block thread count mismatch")
                model_ab.validate_chat_tier_block(current, "off")
                if records:
                    compare_blocks(records[0], current)
                records.append(current)
                if kernel.source_files(root) != before:
                    raise ValueError("source changed during protocol")
        per_threads, comparisons, recommended = summarize(records, list(sequence), measured)
        warm = [i for i in range(len(prompts)) if i not in measured]
        _, warmup, _ = summarize(records, list(sequence), tuple(warm))
        record.update({"per_threads": per_threads, "comparisons": comparisons, "recommended_threads": recommended,
                       "warmup_comparisons_excluded": warmup, "status": "completed"})
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        common.write_exclusive_json(directory / "result.json", record)
    print(record["status"] + ": " + str(directory / "result.json"), flush=True)
    return 0 if record["status"] == "completed" else 1
