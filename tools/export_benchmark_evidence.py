#!/usr/bin/env python3
"""
tools/export_benchmark_evidence.py
Tự động tổng hợp dữ liệu benchmark từ các thư mục phân chia theo Testbed:
- benchmarks/results/testbed_a_i7_14650hx/ (Máy 1: Core i7, RTX 5060)
- benchmarks/results/testbed_b_i5_10400h/   (Máy 2: Core i5, GTX 1650)
Xuất ra docs/BENCHMARKS.md và các file CSV phân tách rõ ràng cho Luận văn Tốt nghiệp.
"""

import json
import os
import glob
import sys
from pathlib import Path
import csv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE_DIR / "benchmarks" / "results"
REPORTS_DIR = WORKSPACE_DIR / "benchmarks" / "reports"
DOCS_DIR = WORKSPACE_DIR / "docs"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def parse_benchmark_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    path_obj = Path(filepath)
    parent_dir = path_obj.parent.name
    testbed_name = "Testbed A (i7-14650HX)" if "testbed_a" in parent_dir else ("Testbed B (Core i5)" if "testbed_b" in parent_dir else "General")

    filename = path_obj.name
    date_prefix = filename[:10] if len(filename) >= 10 and filename[4] == '-' and filename[7] == '-' else "unknown"
    tag = filename[11:-5] if date_prefix != "unknown" else filename[:-5]

    runs = data.get("runs", [])
    if not runs:
        if "metrics" in data:
            runs = [{"cache_state": "single", "response": {"metrics": data["metrics"]}}]
        elif "results" in data:
            runs = data["results"]
        else:
            return None

    parsed_runs = []
    for idx, r in enumerate(runs):
        resp = r.get("response", {})
        metrics = resp.get("metrics", r.get("metrics", {}))
        if not metrics:
            continue
        
        parsed_runs.append({
            "run_index": idx,
            "cache_state": r.get("cache_state", f"run_{idx}"),
            "text": resp.get("text", "")[:40].replace("\n", " "),
            "prompt_tokens": resp.get("prompt_tokens", 0),
            "tokenizer_ms": metrics.get("tokenizer_ms", 0),
            "prefill_ms": metrics.get("prefill_ms", 0),
            "ttft_ms": metrics.get("time_to_first_token_ms", 0),
            "decode_ms": metrics.get("decode_ms", 0),
            "total_ms": metrics.get("total_ms", 0),
            "decode_tok_s": metrics.get("decode_tokens_per_second", 0),
            "attention_ms": metrics.get("attention_ms", 0),
            "expert_compute_ms": metrics.get("expert_compute_ms", 0),
            "lm_head_ms": metrics.get("lm_head_ms", 0),
            "expert_io_ms": metrics.get("expert_io_ms", 0),
            "cache_hits": metrics.get("expert_cache_hits", 0),
            "cache_misses": metrics.get("expert_cache_misses", 0),
            "cache_evictions": metrics.get("expert_cache_evictions", 0),
            "bytes_read_mb": metrics.get("expert_bytes_read", 0) / (1024 * 1024),
            "expert_cache_mb": metrics.get("expert_cache_bytes", 0) / (1024 * 1024),
            "kv_cache_mb": metrics.get("kv_cache_bytes", 0) / (1024 * 1024),
        })

    return {
        "filepath": filepath,
        "filename": filename,
        "parent_dir": parent_dir,
        "testbed": testbed_name,
        "date": date_prefix,
        "tag": tag,
        "cache_bytes_per_layer": data.get("cache_bytes_per_layer", 0),
        "context_limit": data.get("context_limit", 0),
        "logical_cpus": data.get("logical_cpus", 0),
        "prompt": str(data.get("prompt", ""))[:30].replace("\n", " "),
        "runs": parsed_runs,
    }

def main():
    # Find all JSON files recursively in benchmarks/results/
    files = sorted(glob.glob(str(RESULTS_DIR / "**" / "*.json"), recursive=True))
    print(f"Found {len(files)} benchmark JSON files in {RESULTS_DIR}")

    flat_rows = []
    testbed_a_rows = []
    testbed_b_rows = []

    for f in files:
        res = parse_benchmark_file(f)
        if not res or not res["runs"]:
            continue
        for r in res["runs"]:
            row = {
                "testbed": res["testbed"],
                "filename": res["filename"],
                "date": res["date"],
                "tag": res["tag"],
                "cpus": res["logical_cpus"],
                "cache_layer_mb": int(res["cache_bytes_per_layer"] / (1024 * 1024)),
                "context_limit": res["context_limit"],
                "run_state": r["cache_state"],
                "ttft_ms": round(r["ttft_ms"], 2),
                "total_ms": round(r["total_ms"], 2),
                "decode_ms": round(r["decode_ms"], 2),
                "decode_tok_s": round(r["decode_tok_s"], 2),
                "attn_ms": round(r["attention_ms"], 2),
                "expert_compute_ms": round(r["expert_compute_ms"], 2),
                "expert_io_ms": round(r["expert_io_ms"], 2),
                "lm_head_ms": round(r["lm_head_ms"], 2),
                "cache_hits": r["cache_hits"],
                "cache_misses": r["cache_misses"],
                "cache_evictions": r["cache_evictions"],
                "hit_rate_pct": round(r["cache_hits"] / max(1, r["cache_hits"] + r["cache_misses"]) * 100, 2),
                "bytes_read_mb": round(r["bytes_read_mb"], 2),
                "text_sample": r["text"],
            }
            flat_rows.append(row)
            if "Testbed A" in res["testbed"]:
                testbed_a_rows.append(row)
            elif "Testbed B" in res["testbed"]:
                testbed_b_rows.append(row)

    # 1. Export CSVs
    def write_csv(path, rows):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Exported CSV: {path} ({len(rows)} runs)")

    write_csv(REPORTS_DIR / "all_benchmarks.csv", flat_rows)
    write_csv(REPORTS_DIR / "testbed_a_results.csv", testbed_a_rows)
    write_csv(REPORTS_DIR / "testbed_b_results.csv", testbed_b_rows)

    # 2. Generate docs/BENCHMARKS.md with clean testbed separation
    md_path = DOCS_DIR / "BENCHMARKS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đo Lường & Bằng Chứng Thực Nghiệm Đa Phần Cứng\n\n")
        f.write("> **Tài liệu Bằng chứng Thực nghiệm cho Đồ án Tốt nghiệp & Nghiên cứu Khoa học**  \n")
        f.write(f"> Tự động trích xuất từ **{len(files)} tệp tin benchmark JSON thô** phân loại theo 2 Testbed.\n\n")
        
        f.write("## 1. Môi Trường Thực Nghiệm Của 2 Thành Viên (Hardware Testbeds)\n\n")
        f.write("| Thông số Phần cứng | Testbed A (Thành viên 1) | Testbed B (Thành viên 2) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| **Vi xử lý (CPU)** | Intel Core i7-14650HX (16 Cores / 24 Threads) | Intel Core i5 (6 Cores / 12 Threads) |\n")
        f.write("| **Bộ nhớ trong (RAM)** | 16 GB DDR5 (~60 GB/s) | 16 GB DDR4 (~25-30 GB/s) |\n")
        f.write("| **Card đồ họa (GPU)** | NVIDIA GeForce RTX 5060 Laptop (8 GB GDDR6) | NVIDIA GeForce GTX 1650 Max-Q (4 GB GDDR5) |\n")
        f.write("| **Ổ cứng lưu trữ** | SSD NVMe PCIe 4.0 | SSD NVMe PCIe 3.0 |\n")
        f.write("| **Luồng song song (`RAYON_NUM_THREADS`)** | **8 Threads** | **4 Threads** |\n")
        f.write("| **Thư mục lưu kết quả** | `benchmarks/results/testbed_a_i7_14650hx/` | `benchmarks/results/testbed_b_i5_10400h/` |\n\n")

        # Section 2: Side-by-side comparison on identical prompt
        f.write("## 2. Bảng So Sánh Đối Đầu (Cross-Hardware Side-by-Side Comparison)\n\n")
        f.write("So sánh trực diện hiệu năng khi chạy cùng kịch bản (`What is 2 + 2?`, Cache 128 MiB/layer, Max 16 tokens):\n\n")
        f.write("| Chỉ số Đánh giá | Testbed A (i7-14650HX) Cold | Testbed B (Core i5) Cold | Testbed A Warm | Testbed B Warm |\n")
        f.write("| :--- | ---: | ---: | ---: | ---: |\n")

        # Extract sample comparable runs
        a_cold = next((r for r in testbed_a_rows if "cache-128" in r["filename"] and r["run_state"] == "engine-cold"), None)
        b_cold = next((r for r in testbed_b_rows if r["run_state"] == "engine-cold"), None)
        a_warm = next((r for r in testbed_a_rows if "cache-128" in r["filename"] and r["run_state"] == "engine-warm"), None)
        b_warm = next((r for r in testbed_b_rows if r["run_state"] == "engine-warm"), None)

        if a_cold and b_cold and a_warm and b_warm:
            f.write(f"| **Time to First Token (TTFT)** | {round(a_cold['ttft_ms']/1000, 2)} s | **{round(b_cold['ttft_ms']/1000, 2)} s** | {round(a_warm['ttft_ms']/1000, 2)} s | **{round(b_warm['ttft_ms']/1000, 2)} s** |\n")
            f.write(f"| **Tốc độ Decode (tokens/s)** | {a_cold['decode_tok_s']} | **{b_cold['decode_tok_s']}** | {a_warm['decode_tok_s']} | **{b_warm['decode_tok_s']}** |\n")
            f.write(f"| **Tổng thời gian sinh token** | {round(a_cold['total_ms']/1000, 2)} s | **{round(b_cold['total_ms']/1000, 2)} s** | {round(a_warm['total_ms']/1000, 2)} s | **{round(b_warm['total_ms']/1000, 2)} s** |\n")
            f.write(f"| **Thời gian Causal Attention** | {round(a_cold['attn_ms']/1000, 2)} s | {round(b_cold['attn_ms']/1000, 2)} s | {round(a_warm['attn_ms']/1000, 2)} s | {round(b_warm['attn_ms']/1000, 2)} s |\n")
            f.write(f"| **Thời gian Tính Expert** | {round(a_cold['expert_compute_ms']/1000, 2)} s | {round(b_cold['expert_compute_ms']/1000, 2)} s | {round(a_warm['expert_compute_ms']/1000, 2)} s | {round(b_warm['expert_compute_ms']/1000, 2)} s |\n")
            f.write(f"| **Thời gian Đọc SSD (I/O)** | {round(a_cold['expert_io_ms']/1000, 2)} s | {round(b_cold['expert_io_ms']/1000, 2)} s | {round(a_warm['expert_io_ms']/1000, 2)} s | {round(b_warm['expert_io_ms']/1000, 2)} s |\n")
            f.write(f"| **Tỷ lệ Cache Hit (%)** | {a_cold['hit_rate_pct']}% | **{b_cold['hit_rate_pct']}%** | {a_warm['hit_rate_pct']}% | **{b_warm['hit_rate_pct']}%** |\n")
            f.write(f"| **Dung lượng đọc từ SSD** | {round(a_cold['bytes_read_mb']/1024, 2)} GB | {round(b_cold['bytes_read_mb']/1024, 2)} GB | {round(a_warm['bytes_read_mb']/1024, 2)} GB | {round(b_warm['bytes_read_mb']/1024, 2)} GB |\n\n")

        # Section 3: Testbed B Details
        f.write("## 3. Nhật Ký Kết Quả Đo Lường Chi Tiết: Testbed B (Máy Core i5)\n\n")
        f.write("Thư mục nguồn: `benchmarks/results/testbed_b_i5_10400h/`\n\n")
        f.write("| Tệp Benchmark | State | TTFT (s) | Decode (tok/s) | Tổng (s) | Attention (s) | Expert Compute (s) | Expert I/O (s) | Cache Hit % | SSD Đọc (GB) |\n")
        f.write("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in testbed_b_rows:
            f.write(f"| `{row['filename'][:28]}` | {row['run_state']} | {round(row['ttft_ms']/1000, 2)} | {row['decode_tok_s']} | {round(row['total_ms']/1000, 2)} | {round(row['attn_ms']/1000, 2)} | {round(row['expert_compute_ms']/1000, 2)} | {round(row['expert_io_ms']/1000, 2)} | {row['hit_rate_pct']}% | {round(row['bytes_read_mb']/1024, 2)} |\n")
        f.write("\n\n")

        # Section 4: Testbed A Details
        f.write("## 4. Nhật Ký Kết Quả Đo Lường Chi Tiết: Testbed A (Máy Core i7)\n\n")
        f.write("Thư mục nguồn: `benchmarks/results/testbed_a_i7_14650hx/`\n\n")
        f.write("| Tệp Benchmark | State | TTFT (s) | Decode (tok/s) | Tổng (s) | Attention (s) | Expert Compute (s) | Expert I/O (s) | Cache Hit % | SSD Đọc (GB) |\n")
        f.write("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in testbed_a_rows:
            f.write(f"| `{row['filename'][:28]}` | {row['run_state']} | {round(row['ttft_ms']/1000, 2)} | {row['decode_tok_s']} | {round(row['total_ms']/1000, 2)} | {round(row['attn_ms']/1000, 2)} | {round(row['expert_compute_ms']/1000, 2)} | {round(row['expert_io_ms']/1000, 2)} | {row['hit_rate_pct']}% | {round(row['bytes_read_mb']/1024, 2)} |\n")
        f.write("\n\n")

        f.write("## 5. Quy Ước Vận Hành Cho 2 Thành Viên (Team Protocol)\n\n")
        f.write("1. **Thành viên 1** khi chạy benchmark lưu vào: `benchmarks/results/testbed_a_i7_14650hx/`.\n")
        f.write("2. **Thành viên 2** khi chạy benchmark lưu vào: `benchmarks/results/testbed_b_i5_10400h/`.\n")
        f.write("3. Chạy `python tools/run_benchmark.py` để script tự động phát hiện máy và lưu vào đúng folder mà không lo nhầm lẫn!\n")

    print(f"Generated Docs: {md_path}")

if __name__ == "__main__":
    main()
