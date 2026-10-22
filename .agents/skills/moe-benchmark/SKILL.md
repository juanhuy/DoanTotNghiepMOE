---
name: moe-benchmark
description: Run, catalog or interpret this project's i5/i7 benchmarks and thesis evidence with explicit provenance and comparable settings. Use for performance claims, raw-record cleanup or benchmark tooling; not routine unit tests.
---

Read `benchmarks/README.md` and `benchmarks/machines.json`. Old raw records are indexed by `benchmarks/manifest.json`; status historical_unverified is not a new measurement.

- Require explicit i5/i7 selection; never identify a CPU or owner from thread count. Missing hardware remains unknown.
- Use `tools/run_benchmark.py`; preview with `--dry-run`. Normal measurements need a clean revision, checkpoint/suite hashes, exact settings, requested and observed GPU status, UTC run ID and logs. Dirty runs are exploratory only.
- One measurement process per machine. Finish builds/tests before timing. Do not run heavy validation concurrently with a benchmark.
- Hashing checkpoint content reads files before measurement; engine-cold does not mean SSD/OS-cache-cold. CPU mode disables compute after load, which may already allocate GPU memory.
- Compare same revision/checkpoint/workload/config when studying hardware; compare same hardware/config when studying an optimization. Record differences and refuse unsupported causal percentages.
- Report per-run samples before summaries. Separate load/prefill/TTFT/decode, wall time from overlapping timers, payload from physical SSD, capacity from RSS, logical from reserved VRAM. Current engine counters have known review limitations.
- Preserve unique raw payloads. Deduplicate only matching SHA-256 copies; retain aliases/checksums. Validate catalog via `python3 tools/export_benchmark_evidence.py --check`; do not repair raw numbers.
- For thesis claims, link exact raw artifact and source/test, state sample size, error tolerance and unmeasured limits. Quality smoke is not standardized model quality.

Owner records commands, success/failure/skipped, raw path and limits in PROGRESS. No invented benchmark, optimal thread count, zero-copy or zero-allocation claim.
