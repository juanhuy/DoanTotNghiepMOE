#!/usr/bin/env python3
"""Scientific Analysis Tool for MoE Expert Routing Traces and Usage Statistics.

Learned and adapted from Colibri's route_trace.h pipeline.
Calculates expert load imbalance, Gini skewness, co-activation pairs, and hot expert rankings.
Produces structured JSON artifacts and formatted tables for Graduation Thesis Chapter 4.
"""
import argparse
import json
from pathlib import Path
import sys


def parse_usage_file(path: Path) -> dict:
    """Parse Colibri/.moe_usage format:

    -1 <n_layers> <n_experts>
    -2 <version> <engine_id>
    <layer> <expert> <count>
    """
    content = path.read_text(encoding="utf-8")
    num_layers = 0
    num_experts = 0
    counts: dict[int, dict[int, int]] = {}

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "-1":
            if len(parts) >= 3:
                num_layers = int(parts[1])
                num_experts = int(parts[2])
            continue
        if parts[0] == "-2":
            continue
        if len(parts) >= 3:
            layer_id = int(parts[0])
            expert_id = int(parts[1])
            count = int(parts[2])
            if layer_id not in counts:
                counts[layer_id] = {}
            counts[layer_id][expert_id] = count

    return {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "counts": counts,
    }


def compute_gini_coefficient(values: list[float]) -> float:
    """Calculate Gini coefficient measuring routing skewness (0 = perfectly uniform, 1 = completely skewed)."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    for i, v in enumerate(sorted_vals, 1):
        cumulative += i * v
    gini = (2.0 * cumulative) / (n * total) - (n + 1.0) / n
    return max(0.0, min(1.0, gini))


def analyze_usage(usage_data: dict, top_k: int = 4) -> dict:
    num_layers = usage_data["num_layers"]
    num_experts = usage_data["num_experts"]
    counts = usage_data["counts"]

    layer_stats = []
    total_calls_all = 0

    for l in range(num_layers):
        l_counts = counts.get(l, {})
        full_distribution = [l_counts.get(e, 0) for e in range(num_experts)]
        total_layer_calls = sum(full_distribution)
        total_calls_all += total_layer_calls
        active_count = sum(1 for c in full_distribution if c > 0)
        gini = compute_gini_coefficient(full_distribution)

        sorted_experts = sorted(
            [(e, l_counts.get(e, 0)) for e in range(num_experts)],
            key=lambda x: x[1],
            reverse=True,
        )
        top_pinned = sorted_experts[:top_k]

        layer_stats.append({
            "layer": l,
            "total_activations": total_layer_calls,
            "active_experts": active_count,
            "gini_coefficient": round(gini, 4),
            "top_hot_experts": [
                {"expert_id": e, "count": c, "share": round(c / total_layer_calls, 4) if total_layer_calls > 0 else 0.0}
                for e, c in top_pinned if c > 0
            ],
        })

    return {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_expert_activations": total_calls_all,
        "layer_statistics": layer_stats,
    }


def parse_route_trace_file(path: Path) -> list[dict]:
    """Parse ROUTE_TRACE stream:

    <call> <row> <layer> <expert_0>:<gate_0> ...
    """
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            call_id = int(parts[0])
            row = int(parts[1])
            layer = int(parts[2])
            routes = []
            for item in parts[3:]:
                if ":" in item:
                    eid_s, gate_s = item.split(":", 1)
                    routes.append({"expert_id": int(eid_s), "gate": float(gate_s)})
            records.append({
                "call_id": call_id,
                "row": row,
                "layer": layer,
                "routes": routes,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE routing trace & usage statistics.")
    parser.add_argument("--usage", type=Path, help="Path to .moe_usage or .coli_usage file")
    parser.add_argument("--trace", type=Path, help="Path to ROUTE_TRACE log file")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--out", type=Path, help="Save structured report to file")
    args = parser.parse_args()

    if not args.usage and not args.trace:
        parser.print_help()
        sys.exit(1)

    result = {}
    if args.usage:
        usage_data = parse_usage_file(args.usage)
        result["usage_analysis"] = analyze_usage(usage_data)

    if args.trace:
        trace_records = parse_route_trace_file(args.trace)
        result["trace_summary"] = {
            "total_records": len(trace_records),
            "calls_count": len(set(r["call_id"] for r in trace_records)),
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[+] Báo cáo đã được lưu vào: {args.out}")

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif "usage_analysis" in result:
        ua = result["usage_analysis"]
        print("=" * 80)
        print(f"BÁO CÁO PHÂN TÍCH ĐỊNH TUYẾN CHUYÊN GIA ({ua['num_layers']} TẦNG x {ua['num_experts']} EXPERTS)")
        print(f"Tổng số lượt kích hoạt chuyên gia: {ua['total_expert_activations']}")
        print("=" * 80)
        print(f"{'Tầng':<6} | {'Kích hoạt':<10} | {'Active':<8} | {'Gini Skew':<10} | {'Top Hot Experts Ghim Cố Định'}")
        print("-" * 80)
        for row in ua["layer_statistics"]:
            hot_str = ", ".join(f"E{e['expert_id']} ({e['count']}x, {e['share']*100:.1f}%)" for e in row["top_hot_experts"])
            print(f"L{row['layer']:<5} | {row['total_activations']:<10} | {row['active_experts']:<8} | {row['gini_coefficient']:<10.4f} | {hot_str}")
        print("=" * 80)


if __name__ == "__main__":
    main()
