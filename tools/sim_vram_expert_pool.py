#!/usr/bin/env python3
"""Offline replay of the expert RAM cache with an extra VRAM expert pool.

Input is a routing trace written by the engine with ``MOE_ROUTE_TRACE=<path>``
(one line per token and layer: ``<call> <row> <layer> <expert>:<gate> ...``).
Routing does not depend on cache state, so one CPU-only run gives the expert
sequence for every pool size. The replay mirrors ``src/olmoe.rs``:

* RAM cache per layer: byte budget, 70% protected segment, eviction by the
  first minimum of ``(heat, last_used)``, decode order "hits, misses, then
  admit hits and misses", prefill order "ascending expert id, acquire then
  admit". Hot pinning is assumed off (``MOE_HOT_PIN_COUNT`` unset).
* Pool ``a1_first``: A1 semantics (``MOE_GPU_EXPERT_COMPUTE=on``): victim
  exchange disabled, upload on SSD miss and on RAM hit, fill-only (never
  evicts), one global slot budget. A routed expert in the pool and not in RAM
  is computed on the GPU and does not touch the RAM cache.
* Pool ``a1_quota``: as ``a1_first`` but each layer may fill at most
  ``slots // layers`` (+1 for the first ``slots % layers`` layers).
* Pool ``oracle_static``: the ``slots`` most-routed (layer, expert) pairs of the
  whole trace are resident from the start. It sees the future, so it is an
  upper bound for a static placement, not a policy the engine can run.

With ``pool none`` the replay must reproduce the engine's per-request
``expert_cache_hits/misses/evictions`` exactly (``--validate``); otherwise the
simulator does not describe the engine and its pool numbers are not usable.
Counts are expert loads/routes, not time: GPU compute, PCIe and page-cache
effects are not modelled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_trace(path: Path):
    """Return a list of calls; each call is (prefill_rows, decode_steps).

    prefill_rows[layer] = list of top-k lists (one per prompt token);
    decode_steps = list of per-layer top-k lists.
    """
    calls: dict[int, list[tuple[int, list[int]]]] = {}
    order: list[int] = []
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            call, layer = int(parts[0]), int(parts[2])
            experts = [int(item.split(":", 1)[0]) for item in parts[3:]]
            if call not in calls:
                calls[call] = []
                order.append(call)
            calls[call].append((layer, experts))
    layers = 1 + max(layer for lines in calls.values() for layer, _ in lines)
    parsed = []
    for call in order:
        lines = calls[call]
        batch = 0
        while batch < len(lines) and lines[batch][0] == 0:
            batch += 1
        prefill_lines = lines[: batch * layers]
        prefill = [[] for _ in range(layers)]
        for index, (layer, experts) in enumerate(prefill_lines):
            if layer != index // batch:
                raise ValueError(f"call {call}: prefill is not layer-major")
            prefill[layer].append(experts)
        rest = lines[batch * layers :]
        if len(rest) % layers:
            raise ValueError(f"call {call}: decode lines not a multiple of layers")
        steps = []
        for start in range(0, len(rest), layers):
            group = rest[start : start + layers]
            if [layer for layer, _ in group] != list(range(layers)):
                raise ValueError(f"call {call}: decode step out of layer order")
            steps.append([experts for _, experts in group])
        parsed.append((call, prefill, steps))
    return parsed, layers


class Layer:
    def __init__(self, experts: int):
        self.cache: list[list] = []  # [id, last_used, hits_in_cache, is_protected]
        self.heat = [0] * experts
        self.clock = 0

    def find(self, expert: int):
        for index, entry in enumerate(self.cache):
            if entry[0] == expert:
                return index
        return None


class Sim:
    def __init__(self, layers, experts, expert_bytes, budget, policy, slots, oracle_set):
        self.layers = [Layer(experts) for _ in range(layers)]
        self.expert_bytes = expert_bytes
        self.budget = budget
        self.protected_budget = budget * 7 // 10
        self.policy = policy
        self.slots = slots
        self.pool: set[tuple[int, int]] = set(oracle_set or ())
        self.pool_per_layer = Counter(layer for layer, _ in self.pool)
        if policy == "a1_quota":
            base, extra = divmod(slots, layers)
            self.quota = [base + (1 if layer < extra else 0) for layer in range(layers)]
        self.reset_counters()

    def reset_counters(self):
        self.hits = self.misses = self.evictions = 0
        self.gpu_routes = self.uploads = 0

    # Mirrors promote_to_vram + GpuDenseStorage::upload_and_cache with fill_only.
    def promote(self, layer_id: int, expert: int):
        if self.policy not in ("a1_first", "a1_quota"):
            return
        key = (layer_id, expert)
        if key in self.pool or len(self.pool) >= self.slots:
            return
        if self.policy == "a1_quota" and self.pool_per_layer[layer_id] >= self.quota[layer_id]:
            return
        self.pool.add(key)
        self.pool_per_layer[layer_id] += 1
        self.uploads += 1

    def on_gpu(self, layer_id: int, layer: Layer, expert: int) -> bool:
        return (
            self.policy != "none"
            and (layer_id, expert) in self.pool
            and layer.find(expert) is None
        )

    def acquire(self, layer_id: int, layer: Layer, expert: int, uses: int):
        index = layer.find(expert)
        if index is not None:
            self.hits += uses
            entry = layer.cache.pop(index)
            result = (True, entry[2] + 1, entry[3])
            self.promote(layer_id, expert)  # l2_hit && !victim
            return result
        self.misses += 1
        self.hits += uses - 1
        self.promote(layer_id, expert)  # SSD miss && !victim
        return (False, 1, False)

    def admit(self, layer: Layer, expert: int, l2_hit: bool, hits_in_cache: int, protected: bool):
        size = self.expert_bytes
        if size > self.budget:
            return
        if l2_hit:
            protected = True
            current = sum(size for entry in layer.cache if entry[3])
            if current + size > self.protected_budget:
                candidates = [e for e in layer.cache if e[3]]
                if candidates:
                    victim = min(candidates, key=lambda e: (layer.heat[e[0]], e[1]))
                    victim[3] = False
        while len(layer.cache) * size > self.budget - size:
            pool = [i for i, e in enumerate(layer.cache) if not e[3]] or list(range(len(layer.cache)))
            if not pool:
                break
            victim = min(pool, key=lambda i: (layer.heat[layer.cache[i][0]], layer.cache[i][1]))
            layer.cache.pop(victim)
            self.evictions += 1
        if len(layer.cache) * size <= self.budget - size:
            layer.cache.append([expert, layer.clock, hits_in_cache, protected])

    def touch(self, layer: Layer, expert: int):
        layer.clock += 1
        if layer.clock % 65_536 == 0:
            layer.heat = [h // 2 for h in layer.heat]
        layer.heat[expert] += 1

    def prefill(self, rows_per_layer):
        for layer_id, rows in enumerate(rows_per_layer):
            layer = self.layers[layer_id]
            uses = Counter(expert for row in rows for expert in row)
            for expert in sorted(uses):
                count = uses[expert]
                before = layer.clock
                layer.clock += count
                if layer.clock // 65_536 != before // 65_536:
                    layer.heat = [h // 2 for h in layer.heat]
                layer.heat[expert] += count
                if self.on_gpu(layer_id, layer, expert):
                    self.hits += count
                    self.gpu_routes += count
                    continue
                l2_hit, hits_in_cache, protected = self.acquire(layer_id, layer, expert, count)
                self.admit(layer, expert, l2_hit, hits_in_cache, protected)

    def decode(self, step):
        for layer_id, experts in enumerate(step):
            layer = self.layers[layer_id]
            in_ram = [layer.find(e) is not None for e in experts]
            gpu = [
                e for e, ram in zip(experts, in_ram) if not ram and (layer_id, e) in self.pool
            ] if self.policy != "none" else []
            hit_slots = [e for e, ram in zip(experts, in_ram) if ram]
            miss_slots = [e for e, ram in zip(experts, in_ram) if not ram and e not in gpu]
            ready = []
            for expert in hit_slots + miss_slots:
                self.touch(layer, expert)
                ready.append((expert, *self.acquire(layer_id, layer, expert, 1)))
            for expert in gpu:
                self.touch(layer, expert)
            self.hits += len(gpu)
            self.gpu_routes += len(gpu)
            for expert, l2_hit, hits_in_cache, protected in ready:
                self.admit(layer, expert, l2_hit, hits_in_cache, protected)


def run(calls, layers, experts, expert_bytes, budget, policy, slots, oracle_set=None):
    sim = Sim(layers, experts, expert_bytes, budget, policy, slots, oracle_set)
    per_call = []
    for call, prefill, steps in calls:
        sim.reset_counters()
        sim.prefill(prefill)
        for step in steps:
            sim.decode(step)
        per_call.append(
            {
                "call": call,
                "prefill_tokens": len(prefill[0]),
                "decode_steps": len(steps),
                "hits": sim.hits,
                "misses": sim.misses,
                "evictions": sim.evictions,
                "gpu_routes": sim.gpu_routes,
                "uploads": sim.uploads,
            }
        )
    return per_call, sim


def route_counts(calls) -> Counter:
    counts = Counter()
    for _, prefill, steps in calls:
        for layer_id, rows in enumerate(prefill):
            counts.update((layer_id, e) for row in rows for e in row)
        for step in steps:
            for layer_id, experts in enumerate(step):
                counts.update((layer_id, e) for e in experts)
    return counts


def call_range(text: str, calls):
    first, _, last = text.partition("-")
    wanted = range(int(first), int(last or first) + 1)
    return [call for call in calls if call[0] in wanted]


def load_turns(run_dir: Path):
    turns = []
    for path in sorted(run_dir.glob("turn-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        metrics = json.loads(path.read_text())["response"]["metrics"]
        turns.append(
            {
                "hits": metrics["expert_cache_hits"],
                "misses": metrics["expert_cache_misses"],
                "evictions": metrics["expert_cache_evictions"],
                "file_loads": metrics["expert_file_loads"],
                "file_bytes": metrics["expert_file_bytes_read"],
            }
        )
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--experts", type=int, required=True, help="experts per layer")
    parser.add_argument("--expert-bytes", type=int, required=True, help="allocated bytes per expert")
    parser.add_argument("--cache-mib", type=int, required=True, help="RAM cache MiB per layer")
    parser.add_argument("--slots", type=int, nargs="*", default=[], help="VRAM pool sizes to replay")
    parser.add_argument("--validate", type=Path, help="run dir with turn-*.json from the traced run")
    parser.add_argument(
        "--heldout",
        nargs=2,
        metavar=("TRAIN", "TEST"),
        help="learned static placement: top-N from calls TRAIN (e.g. 0-1), replayed on calls TEST "
        "(e.g. 2-3) from an empty cache",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    calls, layers = parse_trace(args.trace)
    budget = args.cache_mib * 1024 * 1024
    report = {
        "trace": str(args.trace),
        "layers": layers,
        "experts": args.experts,
        "expert_bytes": args.expert_bytes,
        "cache_mib_per_layer": args.cache_mib,
        "calls": len(calls),
    }

    baseline, _ = run(calls, layers, args.experts, args.expert_bytes, budget, "none", 0)
    report["baseline"] = baseline
    ok = True
    if args.validate:
        turns = load_turns(args.validate)
        mismatches = []
        if len(turns) != len(baseline):
            mismatches.append(f"turns {len(turns)} != trace calls {len(baseline)}")
        for index, (sim, real) in enumerate(zip(baseline, turns)):
            for key in ("hits", "misses", "evictions"):
                if sim[key] != real[key]:
                    mismatches.append(f"turn {index} {key}: sim {sim[key]} real {real[key]}")
        report["validation"] = {"run_dir": str(args.validate), "mismatches": mismatches}
        ok = not mismatches
        print("validation:", "exact match" if ok else f"{len(mismatches)} mismatches", file=sys.stderr)
        for line in mismatches[:20]:
            print("  ", line, file=sys.stderr)

    counts = route_counts(calls)

    base_misses = sum(c["misses"] for c in baseline)
    scenarios = []
    for slots in args.slots:
        oracle = [key for key, _ in counts.most_common(slots)]
        for policy in ("a1_first", "a1_quota", "oracle_static"):
            per_call, sim = run(
                calls, layers, args.experts, args.expert_bytes, budget, policy, slots,
                oracle if policy == "oracle_static" else None,
            )
            misses = sum(c["misses"] for c in per_call)
            scenarios.append(
                {
                    "policy": policy,
                    "slots": slots,
                    "pool_filled": len(sim.pool),
                    "layers_with_pool": sum(1 for v in Counter(l for l, _ in sim.pool).values() if v),
                    "ssd_loads": misses,
                    "ssd_loads_vs_baseline": misses / base_misses if base_misses else None,
                    "gpu_routes": sum(c["gpu_routes"] for c in per_call),
                    "uploads": sum(c["uploads"] for c in per_call),
                    "per_call_misses": [c["misses"] for c in per_call],
                }
            )
    report["baseline_ssd_loads"] = base_misses
    report["scenarios"] = scenarios

    print(f"baseline SSD loads: {base_misses}")
    print(f"{'policy':14} {'slots':>5} {'filled':>6} {'layers':>6} {'ssd_loads':>9} {'ratio':>6} {'gpu_routes':>10}")
    for s in scenarios:
        print(
            f"{s['policy']:14} {s['slots']:>5} {s['pool_filled']:>6} {s['layers_with_pool']:>6} "
            f"{s['ssd_loads']:>9} {s['ssd_loads_vs_baseline']:>6.3f} {s['gpu_routes']:>10}"
        )
    if args.heldout:
        train, test = (call_range(text, calls) for text in args.heldout)
        learned_counts, test_counts = route_counts(train), route_counts(test)
        base_test = sum(c["misses"] for c in run(test, layers, args.experts, args.expert_bytes, budget, "none", 0)[0])
        rows = []
        for slots in args.slots:
            row = {"slots": slots}
            for name, source in (("learned_static", learned_counts), ("oracle_static", test_counts)):
                placement = [key for key, _ in source.most_common(slots)]
                per_call, _ = run(test, layers, args.experts, args.expert_bytes, budget, "oracle_static", slots, placement)
                row[name] = sum(c["misses"] for c in per_call) / base_test
            for policy in ("a1_first", "a1_quota"):
                per_call, _ = run(test, layers, args.experts, args.expert_bytes, budget, policy, slots)
                row[policy] = sum(c["misses"] for c in per_call) / base_test
            rows.append(row)
        report["heldout"] = {"train_calls": args.heldout[0], "test_calls": args.heldout[1],
                             "baseline_ssd_loads": base_test, "ratios": rows}
        print(f"held-out: learn on calls {args.heldout[0]}, replay calls {args.heldout[1]} "
              f"(baseline SSD loads {base_test}); SSD loads / baseline")
        for row in rows:
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
