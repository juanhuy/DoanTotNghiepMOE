"""HTTP/SSE workload for run_benchmark.py --http-session; stdlib only."""
import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import uuid

import run_benchmark as common


def memory_pressure_reason(sample, limits):
    process, system = sample['process'], sample['system']
    if 'VmRSS' not in process or 'MemAvailable' not in system:
        return 'required memory telemetry unavailable'
    if process['VmRSS'] > limits.stop_rss_mib * 1024:
        return 'RSS watchdog threshold'
    if system['MemAvailable'] < limits.min_available_mib * 1024:
        return 'system available-memory watchdog threshold'
    if process.get('VmSwap', 0) > limits.stop_swap_mib * 1024:
        return 'process swap watchdog threshold'
    return None


class MemoryWatchdog:
    """Supervise only the child owned by this runner; preserve failures on disk."""
    def __init__(self, server, directory):
        from types import SimpleNamespace
        self.server, self.directory = server, directory
        self.limits = SimpleNamespace(stop_rss_mib=5632, min_available_mib=1536, stop_swap_mib=128)
        self.stop = threading.Event()
        self.violations = []
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.finished = False

    def run(self):
        try:
            with (self.directory / 'memory.jsonl').open('x') as output:
                while not self.stop.is_set() and self.server.poll() is None:
                    sample = {'elapsed_s': time.monotonic() - self.started,
                              'process': process_snapshot(self.server.pid), 'system': memory_snapshot()}
                    output.write(json.dumps(sample) + '\n')
                    output.flush()
                    reason = memory_pressure_reason(sample, self.limits)
                    if reason:
                        self.violations.append({'reason': reason, 'sample': sample})
                        self.server.terminate()
                        return
                    self.stop.wait(.25)
        except Exception as error:
            self.violations.append({'reason': f'watchdog error: {type(error).__name__}: {error}'})
            self.server.terminate()

    def finish(self):
        if self.finished:
            return
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('memory watchdog did not stop')
        common.write_exclusive_json(self.directory / 'watchdog.json',
                                    {'limits_mib': vars(self.limits), 'violations': self.violations})
        self.finished = True


def scenario_spec(name=None):
    """Versioned workload embedded in each record; never depend on live web content."""
    specs = {
        "cold": (False, 0, ["Where is Mars"]),
        "warm": (False, 1, ["Where is Mars"]),
        "conversation": (True, 0, ["Where is Mars", "What is love ?"]),
        "topics": (False, 0, ["Where is Mars", "What is love ?",
                               "Explain a Python dictionary.", "Why did the Roman Republic end?"]),
        "long-context": (False, 0, [
            "Read this research note. A small laboratory runs a mixture of experts model on a laptop. "
            "Model weights are stored as INT8 files. Dense layers remain on the GPU when memory permits. "
            "Expert weights can reside in RAM, in VRAM, or in files. Each token selects a small group of experts. "
            "A cache hit avoids a file load but may still require a transfer between host and device. "
            "The team compares inclusive and victim cache policies using the same prompts and token budgets. "
            "They reset application caches before each independent trial, without clearing the operating system page cache. "
            "For warm trials they first run an unmeasured request, then start a new chat with the same prompt. "
            "They record time to first token, decode throughput, total wall time, payload bytes and process memory. "
            "Background activity and temperature can affect results, so policy order is alternated. "
            "A faster isolated operation does not guarantee a faster complete request. "
            "Summarize the experimental controls and explain two limitations of this measurement."]),
    }
    key = name or "conversation"
    conversation, warmups, prompts = specs[key]
    return {"version": 1, "id": key, "session_mode": "conversation" if conversation else "independent",
            "warmup_requests": warmups, "prompts": prompts,
            "reset": "before each policy cycle, before warmup; OS cache unchanged"}


def policy_plan(args):
    if args.compare_cache_policies:
        return [policy for pair in range(args.runs)
                for policy in (("inclusive", "victim") if pair % 2 == 0 else ("victim", "inclusive"))]
    return [args.expert_cache_policy] * args.runs


def run_cycle(port, pid, policy, cycle, spec, max_tokens, directory, runs, warmups):
    """Reset once per trial; warmup never supplies history/KV to measured chats."""
    request(port, "/api/cache/reset?expert_cache_policy=" + policy, {})
    status = json.loads(request(port, "/experts"))
    if status.get("expert_cache_policy") != policy:
        raise ValueError("policy switch was not applied")
    history = []
    session = "benchmark-" + uuid.uuid4().hex
    tasks = [(True, spec["prompts"][0])] * spec["warmup_requests"]
    tasks += [(False, prompt) for prompt in spec["prompts"]]
    for index, (warmup, prompt) in enumerate(tasks):
        if warmup or spec["session_mode"] == "independent":
            history = []
            session = "benchmark-" + uuid.uuid4().hex
        history.append({"role": "user", "content": prompt})
        before = process_snapshot(pid)
        # Save the actual sent history, including earlier model output.
        sent_messages = [dict(message) for message in history]
        response = stream_chat(port, history, session, max_tokens)
        after = process_snapshot(pid)
        turn = index if warmup else index - spec["warmup_requests"]
        sample = {"cycle": cycle, "policy": policy, "turn": turn, "prompt": prompt,
                  "warmup": warmup, "session_id": session, "messages": sent_messages,
                  "response": response, "process_before": before, "process_after": after,
                  "process_read_bytes_delta": (after["read_bytes"] - before["read_bytes"]
                                               if "read_bytes" in after and "read_bytes" in before else None)}
        (warmups if warmup else runs).append(sample)
        history.append({"role": "assistant", "content": response["text"]})
        prefix = "warmup" if warmup else "turn"
        common.write_exclusive_json(directory / f"{prefix}-{cycle}-{index if warmup else turn}.json", sample)
        print(f"cycle={cycle} policy={policy} {prefix}={turn} wall={response['metrics']['total_ms']/1000:.2f}s", flush=True)
        if warmup:
            history = []
            session = "benchmark-" + uuid.uuid4().hex


def process_snapshot(pid):
    result = {}
    for name in ("status", "io"):
        try:
            for line in Path(f"/proc/{pid}/{name}").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmHWM", "VmSwap", "read_bytes", "write_bytes"):
                    result[key] = int(value.split()[0])
        except (OSError, ValueError):
            pass
    return result


def memory_snapshot():
    keys = ("MemAvailable", "SwapTotal", "SwapFree")
    try:
        return {key: int(value.split()[0]) for line in Path("/proc/meminfo").read_text().splitlines()
                for key, _, value in [line.partition(":")] if key in keys}
    except OSError:
        return {}


def request(port, path, body=None, timeout=300):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET" if body is None else "POST", path,
                           body=None if body is None else json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {payload[:400]!r}")
        return payload
    finally:
        connection.close()


def read_stream(response):
    """Reject generation_error, truncated streams and missing terminal metrics."""
    text = []
    terminal = None
    done = False
    first_content_ms = None
    started = time.perf_counter()
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"SSE generation error: {event['error']}")
        if terminal is not None:
            raise RuntimeError("SSE data after terminal event")
        choices = event.get("choices", [])
        if not choices:
            raise ValueError("SSE event missing choices")
        delta = choices[0].get("delta", {}).get("content", "")
        if delta:
            if first_content_ms is None:
                first_content_ms = (time.perf_counter() - started) * 1000
            text.append(delta)
        if choices[0].get("finish_reason") is not None:
            terminal = event
    if not done or terminal is None:
        raise RuntimeError("SSE missing finish or DONE")
    metrics, usage = terminal.get("metrics"), terminal.get("usage")
    if not isinstance(metrics, dict) or not isinstance(usage, dict):
        raise ValueError("SSE missing metrics/usage")
    for key in ("total_ms", "time_to_first_token_ms", "decode_tokens_per_second", "expert_io_ms"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid metric {key}")
    return {"text": "".join(text), "usage": usage, "metrics": metrics,
            "finish_reason": terminal["choices"][0]["finish_reason"],
            "client_first_content_after_headers_ms": first_content_ms}


def stream_chat(port, messages, session, max_tokens):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps({
            "model": "olmoe", "messages": messages, "session_id": session,
            "max_tokens": max_tokens, "stream": True}),
            headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"chat: HTTP {response.status}: {response.read()[:400]!r}")
        result = read_stream(response)
        usage = result["usage"]
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in
               ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise ValueError("invalid token usage")
        if not 1 <= usage["completion_tokens"] <= max_tokens:
            raise ValueError("completion token count outside request budget")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("inconsistent token usage")
        result["client_wall_ms"] = (time.perf_counter() - started) * 1000
        return result
    finally:
        connection.close()


def measure(binary, model, args, env, directory):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env.update(MOE_BIND=f"127.0.0.1:{port}", MOE_MODEL_DIR=str(model),
               MOE_CACHE_BYTES_PER_LAYER=str(args.cache * 1024 * 1024),
               MOE_CONTEXT=str(args.context), MOE_DENSE_BYTES=str(3 * 1024**3),
               MOE_EXPERT_CACHE_POLICY=args.expert_cache_policy)
    env.setdefault('MOE_RAM_LIMIT_BYTES', str(5 * 1024**3))
    env.setdefault('MOE_RAM_HEADROOM_BYTES', str(1024**3))
    with (directory / "server.log").open("x") as log:
        server = subprocess.Popen([str(binary)], env=env, stdout=log, stderr=log)
        watchdog = MemoryWatchdog(server, directory)
        try:
            watchdog.thread.start()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited: {server.returncode}; see server.log")
                try:
                    request(port, "/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("server readiness timeout")
            start = json.loads(request(port, "/experts"))
            if args.gpu == "cpu" and start["gpu"]["enabled"]:
                request(port, "/api/gpu/toggle", {})
                start = json.loads(request(port, "/experts"))
            if start["cache_bytes_per_layer"] != args.cache * 1024 * 1024:
                raise ValueError("server cache mismatch")
            if start.get("expert_cache_policy") != args.expert_cache_policy:
                raise ValueError("server expert cache policy mismatch")
            if args.gpu == "auto" and not start["gpu"]["enabled"]:
                raise RuntimeError("GPU requested but unavailable; refuse CPU fallback for this experiment")
            runs, warmups = [], []
            spec = scenario_spec(args.http_scenario)
            for cycle, policy in enumerate(policy_plan(args)):
                run_cycle(port, server.pid, policy, cycle, spec, args.max_tokens, directory, runs, warmups)
            end = json.loads(request(port, "/experts"))
            watchdog.finish()
            if watchdog.violations:
                raise RuntimeError('memory watchdog stopped the experiment; see watchdog.json')
            return {"runs": runs, "warmups": warmups, "status_start": start,
                    "status_end": end}
        finally:
            try:
                watchdog.finish()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def execute(args, root):
    root = Path(root).resolve()
    if args.suite or args.case:
        raise ValueError("--http-session cannot be combined with --suite/--case")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.tag):
        raise ValueError("invalid tag")
    env = os.environ.copy()
    threads = args.threads or (4 if args.testbed == "i5" else 8)
    env["RAYON_NUM_THREADS"] = str(threads)
    env.setdefault('MOE_RAM_LIMIT_BYTES', str(5 * 1024**3))
    env.setdefault('MOE_RAM_HEADROOM_BYTES', str(1024**3))
    model = common.absolute(args.model_dir, root)
    commit = common.checked_text(["git", "rev-parse", "HEAD"], root, env)
    dirty = bool(common.checked_text(["git", "status", "--porcelain"], root, env))
    if dirty and not args.allow_dirty:
        raise ValueError("Working tree is dirty; use --allow-dirty for exploratory runs")
    build = ["cargo", "build", "--release", "--locked", "--bin", "moe-tier-engine", "--message-format=json"]
    spec = scenario_spec(args.http_scenario)
    plan = policy_plan(args)
    if args.dry_run:
        print(json.dumps({"http_session": True, "testbed": args.testbed, "model": str(model),
                          "build": build, "threads": threads, "cache_mib": args.cache,
                          "expert_cache_policy": args.expert_cache_policy,
                          "compare_cache_policies": args.compare_cache_policies,
                          "max_tokens": args.max_tokens, "policy_plan": plan,
                          "ram_environment": {key: env[key] for key in ('MOE_RAM_LIMIT_BYTES', 'MOE_RAM_HEADROOM_BYTES')},
                          "watchdog_limits_mib": {"stop_rss": 5632, "min_available": 1536, "stop_swap": 128},
                          "measured_requests": len(plan) * len(spec["prompts"]),
                          "warmup_requests": len(plan) * spec["warmup_requests"],
                          "scenario": spec}, indent=2))
        return 0
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + args.tag + "-" + uuid.uuid4().hex
    directory = common.absolute(args.output_root, root) / args.testbed / run_id
    directory.mkdir(parents=True)
    record = {"schema_version": 1, "kind": "http_session", "run_id": run_id, "status": "failed",
              "testbed": args.testbed, "git_commit": commit, "git_dirty": dirty,
              "config": {"threads": threads, "cache_mib": args.cache, "context": args.context,
                         "max_tokens": args.max_tokens, "cycles": args.runs, "gpu_requested": args.gpu,
                         "expert_cache_policy": args.expert_cache_policy},
              "compare_cache_policies": args.compare_cache_policies,
              "ram_environment": {key: env[key] for key in ('MOE_RAM_LIMIT_BYTES', 'MOE_RAM_HEADROOM_BYTES')},
              "scenario": spec,
              "scenario_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
              "policy_plan": plan,
              "memory_start_kib": memory_snapshot(),
              "notes": ["engine reset does not clear OS cache or VRAM pool allocations",
                        "checkpoint hashing reads files before measurement", "desktop load uncontrolled",
                        "process read_bytes is a cumulative OS counter; per-turn deltas reported"]}
    try:
        record["rustc"] = common.checked_text(["rustc", "-Vv"], root, env)
        record["build_environment"] = {key: env.get(key) for key in
                                       ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET")}
        record["checkpoint"] = common.checkpoint_manifest(model)
        record["hardware_observed"] = common.hardware_observed(root, env)
        record["source_sha256"] = {}
        for folder in ("src", "tools"):
            for path in sorted((root / folder).glob("*.rs" if folder == "src" else "*.py")):
                data = path.read_bytes()
                record["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        # The dirty source snapshot is retained, not just a bool/hash.
        for name in ("olmoe.rs", "gpu.rs", "generation.rs"):
            (directory / name).write_bytes((root / "src" / name).read_bytes())
        built = common.run_command(build, root, env)
        (directory / "build.log").write_text(built.stderr + built.stdout)
        if built.returncode:
            raise RuntimeError("build failed; see build.log")
        binaries = [json.loads(line).get("executable") for line in built.stdout.splitlines()
                    if json.loads(line).get("reason") == "compiler-artifact"
                    and json.loads(line).get("target", {}).get("name") == "moe-tier-engine"]
        binary = Path(next(p for p in reversed(binaries) if p))
        record["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        record["result"] = measure(binary, model, args, env, directory)
        record["status"] = "success"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["memory_end_kib"] = memory_snapshot()
    common.write_exclusive_json(directory / "result.json", record)
    print(f"{record['status']}: {directory / 'result.json'}", flush=True)
    return 0 if record["status"] == "success" else 1
