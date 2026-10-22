---
name: moe-engine
description: Modify or review this repository's Rust OLMoE inference, caches, numerical kernels, HTTP cancellation or CUDA FFI. Use project invariants and fixture checks; not for general Rust tutorials or benchmark-only work.
---

Read `docs/ARCHITECTURE.md` and the applicable open findings in `docs/PLAN.md`; do not assume historical completion means the invariant holds.

- OLMoE and tiny/reference are separate model paths. Use the matching fixture and call path.
- For caches, count capacity plus recycled/inflight allocations, not just logical entries. Identify ownership across CPU/VRAM/pool and what reset retains.
- For CUDA, trace context activation on each thread, allocation length and mutability, pointer lifetime, Drop order, scratch synchronization and error fallback. Public safe APIs must preserve the invariant without requiring callers to know hidden unsafe assumptions.
- For prefill/decode, preserve causal positions, Top-K tie ordering and deterministic reduction. Existing fixture tolerance is the reference; text parity alone does not prove logits parity.
- For streaming, distinguish validation before headers from errors after SSE begins. Exercise disconnect, full queue, timeout and next-request reuse when related to the change.
- Regression tests should exercise the real implementation and trigger. Start with a focused test; then `python3 tools/check.py --rust`, adding release/HTTP/CUDA checks by risk.
- GPU tests may succeed without initializing CUDA. State availability and whether GPU compute actually ran. Allocation/I/O speed needs measurement; `unsafe`, Clippy and comments do not establish performance.

Review output: actionable findings with file/symbol, trigger, impact and evidence; distinguish reproduced, statically established and uncertain. The owner records checks/limits in PROGRESS.
