"""Paired, resident-buffer INT8 Gate-Up protocol; invoked by run_benchmark.py."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid
import zipfile

import run_benchmark as common


TEST = "int8_expert::kernel_tests::benchmark_gate_up_paired"
BUILD = ["cargo", "test", "--release", "--lib", "--locked", "--no-run", "--message-format=json"]
SCOPES = ("gate_up_silu", "full_expert")
SEEDS = (17, 73, 991)


def source_files(root):
    paths = [root / name for name in ("Cargo.toml", "Cargo.lock")]
    paths += sorted((root / "src").rglob("*.rs"))
    paths += sorted((root / "tools").glob("*.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate(report, testbed):
    expected = {
        "schema_version": 1, "kind": "gate_up_kernel", "protocol": "gate-up-int8-v1",
        "testbed": testbed, "hidden": 2048, "intermediate": 1024, "threads": 1,
        "seeds": list(SEEDS), "pairs_per_seed_scope": 9, "iterations": 100,
        "warmup_per_variant_scope": 10, "avx2_fma": True, "scopes": list(SCOPES),
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("kernel report protocol/config mismatch")
    parity = report.get("parity", [])
    if len(parity) != 6:
        raise ValueError("missing parity results")
    seen = set()
    for row in parity:
        key = (row.get("seed"), row.get("scope"))
        error = row.get("max_abs_error")
        if key in seen or key[0] not in SEEDS or key[1] not in SCOPES:
            raise ValueError("invalid/duplicate parity case")
        if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error < 2e-5:
            raise ValueError("parity tolerance failed")
        seen.add(key)
    seen = set()
    for row in report.get("samples", []):
        seed, scope, pair, position = (row.get(k) for k in ("seed", "scope", "pair", "position"))
        key = (seed, scope, pair, position)
        if (key in seen or seed not in SEEDS or scope not in SCOPES
                or type(pair) is not int or not 0 <= pair < 9
                or type(position) is not int or position not in (0, 1)):
            raise ValueError("invalid/duplicate paired sample")
        variant = "paired_simd" if (pair + position) % 2 else "separate_simd"
        if (row.get("variant") != variant or row.get("iterations") != 100
                or type(row.get("elapsed_ns")) is not int or row["elapsed_ns"] <= 0):
            raise ValueError("invalid timing or measurement order")
        seen.add(key)
    if len(seen) != 108:
        raise ValueError("incomplete measurements")


def summarize(report):
    result = {}
    for scope in SCOPES:
        rows = [r for r in report["samples"] if r["scope"] == scope]
        cases = []
        for seed in SEEDS:
            pairs = []
            for pair in range(9):
                values = {r["variant"]: r["elapsed_ns"] for r in rows if r["seed"] == seed and r["pair"] == pair}
                pairs.append(values["separate_simd"] / values["paired_simd"])
            cases.append({"seed": seed, "wins": sum(r > 1 for r in pairs),
                          "median_paired_speed_ratio": statistics.median(pairs), "paired_speed_ratios": pairs})
        result[scope] = {
            "seeds": cases,
            "median_ns_per_call": {
                variant: statistics.median(r["elapsed_ns"] / 100 for r in rows if r["variant"] == variant)
                for variant in ("separate_simd", "paired_simd")},
            "median_paired_speed_ratio": statistics.median(r for c in cases for r in c["paired_speed_ratios"]),
        }
    full = result["full_expert"]
    result["promising_for_model_ab"] = (
        full["median_paired_speed_ratio"] >= 1.05 and all(c["wins"] >= 7 for c in full["seeds"]))
    result["decision_rule"] = "full expert median paired ratio>=1.05 and paired wins>=7/9 for every seed; not inference speedup"
    return result


def execute(args, root):
    if (args.gpu != "cpu" or args.threads not in (None, 1) or args.http_session
            or args.compare_lookahead or args.compare_cache_policies or args.suite or args.case):
        raise ValueError("--gate-up-kernel requires CPU, one thread, no HTTP/suite/policy comparison")
    env = dict(os.environ, RAYON_NUM_THREADS="1", MOE_GATE_UP_TESTBED=args.testbed, MOE_GPU_MODE="cpu")
    commit = common.checked_text(["git", "rev-parse", "HEAD"], root, env)
    dirty = bool(common.checked_text(["git", "status", "--porcelain", "--untracked-files=normal"], root, env))
    if dirty and not args.allow_dirty:
        raise ValueError("Working tree is dirty; --allow-dirty is exploratory only")
    if args.dry_run:
        print(json.dumps({"kind": "gate_up_kernel", "testbed": args.testbed, "git_commit": commit,
                          "git_dirty": dirty, "build_argv": BUILD, "test": TEST, "threads": 1,
                          "dataset": "synthetic; no checkpoint", "seeds": SEEDS, "pairs_per_seed_scope": 9,
                          "iterations": 100, "warmup_per_variant_scope": 10,
                          "scopes": SCOPES, "measured_batches": 108,
                          "decision": "full expert median ratio>=1.05; >=7/9 wins per seed"}, indent=2))
        return 0
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-gate-up-" + uuid.uuid4().hex
    directory = common.absolute(args.output_root, root) / args.testbed / run_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "kind": "gate_up_kernel_run", "run_id": run_id,
              "status": "failed", "testbed": args.testbed, "git_commit": commit, "git_dirty": dirty,
              "checkpoint": None, "dataset": "synthetic; no model weights", "threads": 1, "gpu": "cpu",
              "build_argv": BUILD, "notes": ["Resident buffers; serial expert task; no I/O or allocation in timed region",
                                             "Dirty is exploratory; no full-model latency or speedup claim"]}
    print("Artifact: " + str(directory), flush=True)
    try:
        before = source_files(root)
        record["source_sha256"] = before
        with zipfile.ZipFile(directory / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for name in before:
                archive.write(root / name, name)
        record["hardware_observed"] = common.hardware_observed(root, env)
        record["rustc"] = common.checked_text(["rustc", "-vV"], root, env)
        record["cpu_affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        record["build_rustflags"] = {k: env.get(k) for k in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")}
        build = common.run_command(BUILD, root, env)
        (directory / "build.log").write_text(build.stdout + "\n" + build.stderr)
        if build.returncode:
            raise RuntimeError("release benchmark build failed")
        artifacts = []
        for line in build.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (item.get("reason") == "compiler-artifact" and item.get("target", {}).get("name") == "moe_tier_engine"
                    and item.get("profile", {}).get("test") and item.get("executable")):
                artifacts.append(Path(item["executable"]))
        if len(artifacts) != 1:
            raise ValueError("expected exactly one lib test executable")
        binary = artifacts[0]
        record["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        record["argv"] = [str(binary), TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
        env["MOE_GATE_UP_REPORT"] = str(directory / "kernel.json")
        started = time.monotonic()
        with (directory / "benchmark.log").open("x") as log:
            completed = subprocess.run(record["argv"], cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=180, check=False)
        record["process_wall_s"] = time.monotonic() - started
        record["exit_code"] = completed.returncode
        if completed.returncode:
            raise RuntimeError("kernel benchmark failed; inspect benchmark.log")
        report = json.loads((directory / "kernel.json").read_text())
        validate(report, args.testbed)
        if source_files(root) != before or hashlib.sha256(binary.read_bytes()).hexdigest() != record["binary_sha256"]:
            raise ValueError("source or executable changed during run")
        record["summary"] = summarize(report)
        record["status"] = "completed"
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        common.write_exclusive_json(directory / "result.json", record)
    print(record["status"] + ": " + str(directory / "result.json"), flush=True)
    return 0 if record["status"] == "completed" else 1
