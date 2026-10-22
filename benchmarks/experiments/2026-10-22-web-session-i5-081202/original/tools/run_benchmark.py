#!/usr/bin/env python3
"""Build and record explicit, reproducible benchmark runs using only stdlib.

No machine identity is inferred, no historical files are rewritten, and a failed
execution never becomes a successful measurement. See benchmarks/README.md.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def positive(value):
    number = nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--testbed", choices=("i5", "i7"), required=True,
                     help="Operator-selected machine label; never inferred from core count")
    cli.add_argument("--gpu", choices=("cpu", "auto"), required=True,
                     help="Requested backend; consult result GPU status for actual activation")
    cli.add_argument("--model-dir", type=Path, default=Path("models/olmoe-1b-7b-int8"))
    cli.add_argument("--prompt", default="What is 2 + 2?")
    cli.add_argument("--max-tokens", type=positive, default=16)
    cli.add_argument("--runs", type=positive, default=3)
    cli.add_argument("--cache", type=nonnegative, default=128, help="MiB per layer")
    cli.add_argument("--context", type=positive, default=512)
    cli.add_argument("--threads", type=positive,
                     help="Rayon workers; convenience defaults i5=4, i7=8, not measured optima")
    cli.add_argument("--tag", default="chat")
    cli.add_argument("--suite", type=Path)
    cli.add_argument("--case", help="Suite case ID (requires --suite); omitted means all")
    cli.add_argument("--allow-dirty", action="store_true")
    cli.add_argument("--output-root", type=Path, default=Path("benchmarks/runs"))
    cli.add_argument("--dry-run", action="store_true",
                     help="Print preflight and commands without building, hashing or recording")
    return cli


def absolute(path, root):
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def run_command(argv, root, env):
    return subprocess.run(argv, cwd=root, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=False)


def checked_text(argv, root, env):
    completed = run_command(argv, root, env)
    if completed.returncode:
        raise RuntimeError(f"{argv[0]} failed ({completed.returncode}): {completed.stderr.strip()}")
    return completed.stdout.strip()


def checkpoint_manifest(model):
    """Hash content, including JSON tokenizer/config/index files, not just paths."""
    if not (model / "config.json").is_file():
        raise ValueError(f"Missing checkpoint config: {model / 'config.json'}")
    paths = sorted(p for p in model.rglob("*")
                   if p.is_file() and p.suffix in (".json", ".safetensors"))
    if not any(p.suffix == ".safetensors" for p in paths):
        raise ValueError(f"No Safetensors checkpoint files under {model}")
    files = []
    for path in paths:
        digest = hashlib.sha256()
        before = path.stat()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Checkpoint changed during hashing: {path}")
        files.append({"path": path.relative_to(model).as_posix(),
                      "size": after.st_size, "sha256": digest.hexdigest()})
    config = next(entry for entry in files if entry["path"] == "config.json")
    return {"config_sha256": config["sha256"], "files": files}


def hardware_observed(root, env):
    observed = {"logical_cpus": os.cpu_count(), "cpu_model": platform.processor() or None,
                "physical_memory_bytes": None,
                "gpu_inventory": {"status": "unavailable", "devices": []}}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                observed["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        observed["physical_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, timeout=10)
        if probe.returncode == 0:
            observed["gpu_inventory"] = {"status": "observed", "format": "name,memory_MiB,driver",
                                         "devices": probe.stdout.strip().splitlines()}
        else:
            observed["gpu_inventory"]["reason"] = probe.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        observed["gpu_inventory"]["reason"] = str(exc)
    return observed


def command_arguments(args, model, suite):
    if suite:
        return [str(model), str(args.runs), str(suite), args.case or "*",
                str(args.cache), str(args.context), args.gpu]
    return [str(model), args.prompt, str(args.max_tokens), str(args.runs),
            str(args.cache), str(args.context), args.gpu]


def validate_result(result, args, model, suite_content=None):
    """Accept the examples' actual report contract, not arbitrary exit-0 JSON."""
    def require(condition, message):
        if not condition:
            raise ValueError("Invalid benchmark report: " + message)

    def integer(value):
        return type(value) is int and value >= 0

    def finite_nonnegative(value):
        try:
            return type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            return False

    require(isinstance(result, dict), "expected one JSON object")
    require(type(result.get("schema_version")) is int and result["schema_version"] == 1,
            "schema_version must equal 1")
    require(result.get("model_directory") == str(model), "model_directory does not match request")
    require(result.get("gpu_mode") == args.gpu, "gpu_mode does not match request")
    for key in ("gpu_status_start", "gpu_status_end"):
        require(key in result and (result[key] is None or isinstance(result[key], dict)),
                key + " must be an object or null")
    load_ms = result.get("load_ms")
    require(finite_nonnegative(load_ms),
            "load_ms must be finite and nonnegative")
    runs = result.get("runs")
    require(isinstance(runs, list) and bool(runs), "runs must be a nonempty list")
    if suite_content is None:
        require(result.get("prompt") == args.prompt, "prompt does not match request")
        require(result.get("max_tokens") == args.max_tokens, "max_tokens does not match request")
        settings = result
        require(len(runs) == args.runs, "run count does not match request")
        budgets = [args.max_tokens] * args.runs
        for index, run in enumerate(runs):
            require(isinstance(run, dict) and type(run.get("index")) is int
                    and run["index"] == index, "generation indexes must be ordered from zero")
            require(run.get("cache_state") == ("engine-cold" if index == 0 else "engine-warm"),
                    "cache_state does not match run order")
    else:
        require(result.get("suite") == suite_content, "embedded suite does not match hashed suite")
        selected = args.case if args.case and args.case != "*" else None
        require("selected_case" in result and result["selected_case"] == selected,
                "selected_case does not match request")
        require(type(result.get("rounds")) is int and result["rounds"] == args.runs,
                "round count does not match request")
        require(isinstance(suite_content, dict) and isinstance(suite_content.get("cases"), list),
                "suite must contain cases")
        cases = [case for case in suite_content["cases"]
                 if isinstance(case, dict) and (selected is None or case.get("id") == selected)]
        require(bool(cases), "no selected suite cases")
        expected = [(round_id, case) for round_id in range(args.runs) for case in cases]
        require(len(runs) == len(expected), "suite run count does not match rounds/cases")
        budgets = []
        for run, (round_id, case) in zip(runs, expected):
            require(isinstance(run, dict) and type(run.get("round")) is int
                    and run["round"] == round_id and run.get("case_id") == case.get("id"),
                    "suite round/case order does not match request")
            require(integer(case.get("max_tokens")) and case["max_tokens"] > 0,
                    "suite case max_tokens must be positive")
            budgets.append(case["max_tokens"])
        settings = result.get("settings")
        require(isinstance(settings, dict), "suite settings must be an object")
    require(type(settings.get("cache_bytes_per_layer")) is int
            and settings["cache_bytes_per_layer"] == args.cache * 1024 * 1024,
            "cache setting does not match request")
    require(type(settings.get("context_limit")) is int and settings["context_limit"] == args.context,
            "context setting does not match request")
    timers = ("tokenizer_ms", "prefill_ms", "time_to_first_token_ms", "decode_ms", "total_ms",
              "decode_tokens_per_second", "attention_ms", "expert_compute_ms", "lm_head_ms", "expert_io_ms")
    counters = ("expert_cache_hits", "expert_cache_misses", "expert_cache_evictions", "expert_bytes_read",
                "expert_cache_bytes", "kv_cache_bytes", "cached_prompt_tokens")
    for run, budget in zip(runs, budgets):
        response = run.get("response")
        require(isinstance(response, dict), "run response must be an object")
        require(isinstance(response.get("text"), str), "response text must be a string")
        require(response.get("finish_reason") in ("stop", "length"), "invalid finish_reason")
        require(integer(response.get("prompt_tokens")) and response["prompt_tokens"] > 0,
                "prompt_tokens must be positive")
        tokens = response.get("token_ids")
        require(isinstance(tokens, list) and len(tokens) <= budget and all(integer(token) for token in tokens),
                "token_ids must be nonnegative integers within the requested budget")
        metrics = response.get("metrics")
        require(isinstance(metrics, dict), "response metrics must be an object")
        for key in timers:
            value = metrics.get(key)
            require(finite_nonnegative(value),
                    key + " must be finite and nonnegative")
        for key in counters:
            require(integer(metrics.get(key)), key + " must be a nonnegative integer")


def write_exclusive_json(path, data):
    """Publish a complete file atomically, and refuse an existing destination."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute(args, root=WORKSPACE_DIR):
    root = Path(root).resolve()
    env = os.environ.copy()
    threads = args.threads if args.threads is not None else (4 if args.testbed == "i5" else 8)
    env["RAYON_NUM_THREADS"] = str(threads)
    model = absolute(args.model_dir, root)
    suite = absolute(args.suite, root) if args.suite else None
    if args.case and not suite:
        raise ValueError("--case requires --suite")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.tag):
        raise ValueError("--tag must be 1-64 letters, digits, dots, underscores or hyphens")
    commit = checked_text(["git", "rev-parse", "HEAD"], root, env)
    dirty = bool(checked_text(["git", "status", "--porcelain", "--untracked-files=normal"], root, env))
    if dirty and not args.allow_dirty:
        raise ValueError("Working tree is dirty; commit changes or explicitly use --allow-dirty")
    example = "benchmark_suite" if suite else "benchmark_olmoe"
    build = ["cargo", "build", "--release", "--locked", "--example", example, "--message-format=json"]
    benchmark_args = command_arguments(args, model, suite)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "testbed": args.testbed, "git_commit": commit,
                          "git_dirty": dirty, "rayon_num_threads": threads,
                          "build_argv": build, "example": example,
                          "benchmark_arguments": benchmark_args,
                          "output_root": str(absolute(args.output_root, root)),
                          "checkpoint_hashing": "not performed"}, indent=2))
        return 0
    if not (model / "config.json").is_file():
        raise ValueError(f"Missing checkpoint config: {model / 'config.json'}")
    if suite and not suite.is_file():
        raise ValueError(f"Missing suite: {suite}")
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + args.tag + "-" + uuid.uuid4().hex
    directory = absolute(args.output_root, root) / args.testbed / run_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 2, "run_id": run_id,
              "recorded_at_utc": now.isoformat(), "testbed": args.testbed,
              "status": "failed", "tag": args.tag,
              "hardware_observed": {},
              "execution": {"argv": [], "build_argv": build,
                            "git_commit": commit, "git_dirty": dirty,
                            "rustc": None, "rayon_num_threads": threads,
                            "platform": platform.platform(), "checkpoint": None,
                            "requested_gpu": args.gpu, "model_dir": str(model),
                            "cpu_mode_scope": "CPU request computation; model load may initialize GPU and retain VRAM",
                            "os_page_cache": "uncontrolled; checkpoint hashing reads files before measurement",
                            "config": {"cache_mib_per_layer": args.cache, "context": args.context,
                                       "runs": args.runs, "max_tokens": args.max_tokens if not suite else None,
                                       "prompt": args.prompt if not suite else None,
                                       "suite_case": (args.case or "*") if suite else None},
                            "build_environment": {key: env.get(key) for key in
                                ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET")}}}
    error_log = []
    benchmark_stdout = None
    try:
        record["hardware_observed"] = hardware_observed(root, env)
        record["execution"]["rustc"] = checked_text(["rustc", "-Vv"], root, env)
        record["execution"]["checkpoint"] = checkpoint_manifest(model)
        suite_content = None
        if suite:
            suite_bytes = suite.read_bytes()
            suite_content = json.loads(suite_bytes)
            record["execution"]["suite"] = {"path": str(suite),
                "sha256": hashlib.sha256(suite_bytes).hexdigest()}
        built = run_command(build, root, env)
        error_log.append("[build]\n" + built.stderr)
        error_log.append("[cargo JSON diagnostics]\n" + built.stdout)
        if built.returncode:
            raise RuntimeError(f"Build failed with exit code {built.returncode}")
        executable = None
        for line in built.stdout.splitlines():
            artifact = json.loads(line)
            if (artifact.get("reason") == "compiler-artifact"
                    and artifact.get("target", {}).get("name") == example
                    and artifact.get("executable")):
                executable = artifact["executable"]
        if not executable:
            raise RuntimeError("Cargo did not report an example executable; refusing a stale binary")
        argv = [executable] + benchmark_args
        record["execution"]["argv"] = argv
        measured = run_command(argv, root, env)
        benchmark_stdout = measured.stdout
        error_log.append("[benchmark]\n" + measured.stderr)
        record["execution"]["returncode"] = measured.returncode
        if measured.returncode:
            raise RuntimeError(f"Benchmark failed with exit code {measured.returncode}")
        result = json.loads(measured.stdout, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON value: {value}")))
        validate_result(result, args, model, suite_content)
        record["result"] = result
        record["status"] = "success"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        record["error"] = str(exc)
        error_log.append("[runner]\n" + str(exc))
        if benchmark_stdout is not None:
            with (directory / "stdout.failed.log").open("x", encoding="utf-8") as output:
                output.write(benchmark_stdout)
    with (directory / "stderr.log").open("x", encoding="utf-8") as output:
        output.write("\n".join(error_log) + "\n")
    write_exclusive_json(directory / "result.json", record)
    print(f"{record['status']}: {directory / 'result.json'}")
    return 0 if record["status"] == "success" else 1


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
