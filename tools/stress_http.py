"""Repeated local HTTP/SSE/cancellation test using a model fixture.

Usage: python3 tools/stress_http.py MODEL_DIR OUTPUT_JSON [CYCLES=50]
The script starts and terminates only its own release server.
"""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def rss_kib(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS missing")


def main():
    if not 3 <= len(sys.argv) <= 4:
        raise SystemExit("usage: stress_http.py MODEL_DIR OUTPUT_JSON [CYCLES=50]")
    model, output = sys.argv[1:3]
    cycles = int(sys.argv[3]) if len(sys.argv) == 4 else 50
    if cycles < 10:
        raise SystemExit("CYCLES must be at least 10")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(
        os.environ,
        MOE_MODEL_DIR=model,
        MOE_BIND=f"127.0.0.1:{port}",
        MOE_CONTEXT="64",
        MOE_CACHE_BYTES_PER_LAYER="10000",
        MOE_DENSE_BYTES="10000000",
        MOE_STREAM_SEND_TIMEOUT_MS="1000",
    )
    with tempfile.TemporaryFile() as log:
        server = subprocess.Popen(
            ["target/release/moe-tier-engine"], env=env, stdout=log, stderr=log
        )

        def request(body=None, path="/v1/chat/completions"):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            connection.request(
                "GET" if body is None else "POST",
                path,
                body=None if body is None else json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            return connection, connection.getresponse()

        def wait_ready():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read().decode())
                try:
                    connection, response = request(path="/health")
                    response.read()
                    connection.close()
                    if response.status == 200:
                        return
                except OSError:
                    time.sleep(0.05)
            raise TimeoutError("server did not become ready")

        body = {
            "model": "olmoe",
            "messages": [{"role": "user", "content": "token3 token4"}],
            "max_tokens": 8,
            "stream": True,
        }
        rss_samples = []
        completed = cancelled = 0
        started = time.monotonic()
        try:
            wait_ready()
            for cycle in range(cycles):
                connection, response = request(body)
                assert response.status == 200
                first = response.readline()
                while first and not first.startswith(b"data: "):
                    first = response.readline()
                assert first
                if cycle % 3 == 0:
                    response.close()
                    connection.close()
                    cancelled += 1
                    deadline = time.monotonic() + 2
                    while True:
                        check, result = request(dict(body, max_tokens=0))
                        status = result.status
                        result.read()
                        check.close()
                        if status == 400:
                            break
                        assert status == 429
                        if time.monotonic() > deadline:
                            raise TimeoutError("cancelled request did not release model")
                        time.sleep(0.005)
                else:
                    payloads = []
                    while True:
                        line = response.readline()
                        if not line:
                            break
                        if line.startswith(b"data: "):
                            payloads.append(line[6:].strip())
                    response.close()
                    connection.close()
                    assert b"[DONE]" in payloads
                    events = [json.loads(item) for item in payloads if item != b"[DONE]"]
                    assert not any("error" in event for event in events)
                    assert any(event.get("usage") for event in events)
                    completed += 1
                health, result = request(path="/health")
                assert result.status == 200 and result.read() == b"ok"
                health.close()
                rss_samples.append(rss_kib(server.pid))
            steady = rss_samples[5:]
            spread = max(steady) - min(steady)
            assert spread <= 64 * 1024, f"steady RSS spread too large: {spread} KiB"
            report = {
                "model_directory": model,
                "cycles": cycles,
                "completed": completed,
                "cancelled": cancelled,
                "elapsed_ms": (time.monotonic() - started) * 1000,
                "rss_kib": {
                    "first": rss_samples[0],
                    "last": rss_samples[-1],
                    "steady_min": min(steady),
                    "steady_max": max(steady),
                    "steady_spread": spread,
                },
                "checks": [
                    "stream completes with usage and DONE",
                    "disconnect releases the single-request permit",
                    "health remains available after every cycle",
                    "steady VmRSS spread is at most 64 MiB",
                ],
                "notes": [
                    "fixture stress test; not model-quality or production-load evaluation",
                    "VmRSS includes allocator/cache behavior and is sampled after each cycle",
                ],
            }
            Path(output).write_text(json.dumps(report, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
