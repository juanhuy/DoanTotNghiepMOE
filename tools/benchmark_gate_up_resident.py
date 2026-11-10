"""Shared resident harness: Gate-Up and batched expert prototypes; no model forward."""
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
from benchmark_gate_up import BUILD, source_files

TEST = "int8_expert::kernel_tests::benchmark_gate_up_resident"
BATCH_TEST = "int8_expert::batch_prototype::benchmark_expert_batch"
BATCH_SCOPES = ("batch2", "batch4")
LAYERS = (0, 7, 15)
SEEDS = (17, 73, 991)
SCOPES = ("expert_tasks8", "row_parallel")


def proc_value(raw, key):
    for line in raw.splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    raise ValueError("missing /proc field: " + key)


def validate(report, testbed, batch=False):
    scopes = BATCH_SCOPES if batch else SCOPES
    expected = {"schema_version":1,"kind":"expert_batch_resident" if batch else "gate_up_resident",
                "protocol":"expert-batch-int8-v1" if batch else "gate-up-resident-v1",
                "testbed":testbed,"hidden":2048,"intermediate":1024,"threads":4,"avx2_fma":True,
                "layers":list(LAYERS),"expert_ids":[0] if batch else list(range(8)),"seeds":list(SEEDS),
                "scopes":list(scopes),"pairs_per_case":9,"iterations":40,
                "warmup_per_variant_scope":10,"payload_budget_bytes":256*1024**2}
    if any(report.get(k) != v for k,v in expected.items()):
        raise ValueError("resident protocol/config mismatch")
    if report.get("payload_bytes") != (3 if batch else 24)*(3*2048*1024+4096*4):
        raise ValueError("resident payload mismatch")
    if batch and (report.get("baseline_gate_up") != "paired"
                  or report.get("baseline_dispatch") != "serial_tokens_row_parallel"
                  or report.get("scratch_budget_bytes") != 256*1024
                  or type(report.get("scratch_peak_capacity_bytes")) is not int
                  or not 0 < report["scratch_peak_capacity_bytes"] <= 256*1024):
        raise ValueError("invalid batch scratch budget or baseline")
    for key in ("process_before","process_after"):
        if proc_value(report[key],"VmSwap") or proc_value(report[key],"VmHWM") > 1024*1024:
            raise ValueError("resident benchmark process pressure")
    if proc_value(report["io_after"],"read_bytes") != proc_value(report["io_before"],"read_bytes"):
        raise ValueError("physical read I/O during compute region")
    parity = report.get("parity",[])
    expected_cases = {(layer,seed,scope) for layer in LAYERS for seed in SEEDS for scope in scopes}
    seen=set()
    for row in parity:
        key=(row.get("layer"),row.get("seed"),row.get("scope"))
        err=row.get("max_abs_error")
        if key not in expected_cases or key in seen or type(err) not in (int,float) or not math.isfinite(err) or not 0<=err<2e-5:
            raise ValueError("invalid resident parity")
        seen.add(key)
    if seen!=expected_cases:
        raise ValueError("missing resident parity")
    seen=set()
    for row in report.get("samples",[]):
        layer,seed,scope,pair,pos=(row.get(k) for k in ("layer","seed","scope","pair","position"))
        key=(layer,seed,scope,pair,pos)
        if ((layer,seed,scope) not in expected_cases or key in seen or type(pair) is not int
                or not 0<=pair<9 or type(pos) is not int or pos not in (0,1)):
            raise ValueError("invalid/duplicate resident sample")
        variant=(("shared_weights" if (pair+pos)%2 else "separate_tokens") if batch
                 else ("paired_simd" if (pair+pos)%2 else "separate_simd"))
        if (row.get("variant")!=variant or row.get("iterations")!=40
                or type(row.get("elapsed_ns")) is not int or row["elapsed_ns"]<=0):
            raise ValueError("invalid resident timer/order")
        seen.add(key)
    if len(seen)!=324:
        raise ValueError("missing resident samples")


def summarize(report, batch=False):
    summary={}
    scopes=BATCH_SCOPES if batch else SCOPES
    baseline,candidate=("separate_tokens","shared_weights") if batch else ("separate_simd","paired_simd")
    ratio_key="median_speed_ratio" if batch else "median_paired_ratio"
    for scope in scopes:
        cases=[]
        for layer in LAYERS:
            for seed in SEEDS:
                ratios=[]
                for pair in range(9):
                    vals={r["variant"]:r["elapsed_ns"] for r in report["samples"]
                          if r["scope"]==scope and r["layer"]==layer and r["seed"]==seed and r["pair"]==pair}
                    ratios.append(vals[baseline]/vals[candidate])
                cases.append({"layer":layer,"seed":seed,"wins":sum(v>1 for v in ratios),
                              "median_ratio":statistics.median(ratios),"ratios":ratios})
        summary[scope]={"cases":cases,ratio_key:statistics.median(v for c in cases for v in c["ratios"]),
                        "wins":sum(c["wins"] for c in cases),"pairs":81}
    if batch:
        summary["passes_compute_rule"]=all(
            summary[scope][ratio_key]>=1.10 and all(c["wins"]>=7 for c in summary[scope]["cases"])
            for scope in scopes)
        summary["decision_rule"]="both batch2/batch4 median ratio>=1.10 and >=7/9 wins in every layer/seed; not model speedup"
        summary["next"]="integrate into prefill only if rule passes; otherwise archive prototype and stop"
        return summary
    tasks=summary["expert_tasks8"]
    summary["passes_compute_rule"]=(tasks["median_paired_ratio"]>=1.05 and all(c["wins"]>=7 for c in tasks["cases"]))
    summary["decision_rule"]="expert_tasks8 pooled median ratio>=1.05 and >=7/9 wins for every layer/seed; row_parallel secondary; no inference speedup"
    summary["next"]="close Gate-Up exploration; retain existing default and move to expert I/O; no further HTTP rerun for speedup"
    return summary


def execute(args, root, batch=False):
    if (args.gpu != "cpu" or args.threads != 4 or args.http_session
            or args.compare_lookahead or args.compare_cache_policies or args.suite or args.case):
        raise ValueError("--gate-up-resident requires CPU, four threads, no HTTP/suite/policy comparison")
    kind="expert_batch_resident" if batch else "gate_up_resident"
    benchmark_test=BATCH_TEST if batch else TEST
    scopes=BATCH_SCOPES if batch else SCOPES
    decision=("both batch2/batch4 median ratio>=1.10 and >=7/9 wins in every layer/seed"
              if batch else "expert_tasks8 median ratio>=1.05 and >=7/9 wins in every layer/seed; close Gate-Up after this run")
    env = dict(os.environ, RAYON_NUM_THREADS="4", MOE_GATE_UP_TESTBED=args.testbed, MOE_GPU_MODE="cpu")
    env["MOE_RESIDENT_MODEL"] = str(common.absolute(args.model_dir, root))
    commit = common.checked_text(["git", "rev-parse", "HEAD"], root, env)
    dirty = bool(common.checked_text(["git", "status", "--porcelain", "--untracked-files=normal"], root, env))
    if dirty and not args.allow_dirty:
        raise ValueError("Working tree is dirty; --allow-dirty is exploratory only")
    if args.dry_run:
        print(json.dumps({"kind":kind,"testbed":args.testbed,"git_commit":commit,
                          "git_dirty":dirty,"threads":4,"build_argv":BUILD,"test":benchmark_test,
                          "layers":LAYERS,"expert_ids":[0] if batch else list(range(8)),"seeds":SEEDS,"scopes":scopes,
                          "pairs_per_case":9,"iterations":40,"payload_budget_mib":256,
                          "model_dir":env["MOE_RESIDENT_MODEL"],
                          "decision":decision},indent=2))
        return 0
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + ("-expert-batch-" if batch else "-gate-up-resident-") + uuid.uuid4().hex
    directory = common.absolute(args.output_root, root) / args.testbed / run_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "kind": kind+"_run", "run_id": run_id,
              "status": "failed", "testbed": args.testbed, "git_commit": commit, "git_dirty": dirty,
              "dataset": "real INT8 weights/scales; synthetic activations", "threads": 4, "gpu": "cpu",
              "build_argv": BUILD, "notes": ["Resident preallocated buffers; CPU4; file reads before timing; no full model forward",
                                             "Dirty is exploratory; no full-model latency or speedup claim"]}
    print("Artifact: " + str(directory), flush=True)
    try:
        record["checkpoint"] = common.checkpoint_manifest(Path(env["MOE_RESIDENT_MODEL"]))
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
        record["argv"] = [str(binary), benchmark_test, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
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
        validate(report, args.testbed, batch=batch)
        if source_files(root) != before or hashlib.sha256(binary.read_bytes()).hexdigest() != record["binary_sha256"]:
            raise ValueError("source or executable changed during run")
        record["summary"] = summarize(report, batch=batch)
        record["status"] = "completed"
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        common.write_exclusive_json(directory / "result.json", record)
    print(record["status"] + ": " + str(directory / "result.json"), flush=True)
    return 0 if record["status"] == "completed" else 1
