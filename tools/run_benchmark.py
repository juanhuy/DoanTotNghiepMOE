#!/usr/bin/env python3
"""
tools/run_benchmark.py
Công cụ chạy benchmark tự động dành cho 2 người 2 máy:
- Tự động nhận diện phần cứng (Core i7 -> Testbed A; Core i5 -> Testbed B).
- Tự động gán số luồng Rayon tối ưu (i7 -> 8 luồng; i5 -> 4 luồng).
- Tự động lưu file kết quả JSON vào đúng thư mục testbed của từng người.
- Tự động đồng bộ báo cáo docs/BENCHMARKS.md sau khi chạy xong.

Cách dùng:
  # Chạy tự động với prompt mặc định:
  python tools/run_benchmark.py

  # Chạy với prompt tùy chọn:
  python tools/run_benchmark.py --prompt "Explain quantum computing briefly" --max-tokens 32

  # Chỉ định rõ máy A hoặc B:
  python tools/run_benchmark.py --testbed a
  python tools/run_benchmark.py --testbed b
"""

import argparse
import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = WORKSPACE_DIR / "models" / "olmoe-1b-7b-int8"
RESULTS_DIR = WORKSPACE_DIR / "benchmarks" / "results"

TESTBED_A_DIR = RESULTS_DIR / "testbed_a_i7_14650hx"
TESTBED_B_DIR = RESULTS_DIR / "testbed_b_i5_10400h"

TESTBED_A_DIR.mkdir(parents=True, exist_ok=True)
TESTBED_B_DIR.mkdir(parents=True, exist_ok=True)

def detect_testbed():
    """Tự động phát hiện cấu hình máy dựa trên số lõi CPU."""
    count = os.cpu_count() or 1
    if count >= 20:
        return "a", 8, "Intel Core i7-14650HX (24 threads)"
    else:
        return "b", 4, f"Intel Core i5 ({count} threads)"

def main():
    parser = argparse.ArgumentParser(description="Chạy benchmark MoE tự động phân chia theo Testbed máy.")
    parser.add_argument("--testbed", choices=["auto", "a", "b"], default="auto", help="Chọn máy đo: a, b hoặc auto (mặc định)")
    parser.add_argument("--prompt", default="What is 2 + 2?", help="Câu prompt để kiểm thử")
    parser.add_argument("--max-tokens", type=int, default=16, help="Số token tối đa sinh ra")
    parser.add_argument("--runs", type=int, default=2, help="Số lượt chạy liên tiếp (1 cold, các lượt sau warm)")
    parser.add_argument("--cache", type=int, default=128, help="Ngân sách cache mỗi layer (MiB)")
    parser.add_argument("--tag", default="chat", help="Tên nhãn lưu file (ví dụ: smoke, near-context, chat)")
    parser.add_argument("--suite", help="Đường dẫn file suite JSON (nếu chạy benchmark suite)")
    parser.add_argument("--case", help="Case ID trong suite (nếu chạy benchmark suite)")

    args = parser.parse_args()

    # 1. Xác định Testbed
    auto_tb, auto_threads, cpu_desc = detect_testbed()
    tb = auto_tb if args.testbed == "auto" else args.testbed
    threads = auto_threads if args.testbed == "auto" else (8 if tb == "a" else 4)

    target_dir = TESTBED_A_DIR if tb == "a" else TESTBED_B_DIR
    tb_name = "Testbed A (Core i7)" if tb == "a" else "Testbed B (Core i5)"

    print("=" * 70)
    print(f"🚀 KHỞI CHẠY BENCHMARK MOE-TIERENGINE")
    print(f"   • Cấu hình nhận diện: {cpu_desc}")
    print(f"   • Đích lưu kết quả: {tb_name} -> {target_dir.relative_to(WORKSPACE_DIR)}")
    print(f"   • Số worker Rayon: {threads} threads")
    print(f"   • Ngân sách Cache: {args.cache} MiB/layer")
    print("=" * 70)

    # 2. Tạo tên file kết quả
    today = datetime.date.today().isoformat()
    safe_tag = args.tag.replace(" ", "-").lower()
    filename = f"{today}-{safe_tag}.json"
    output_file = target_dir / filename

    # 3. Chuẩn bị câu lệnh thực thi
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)

    if args.suite:
        exe_path = WORKSPACE_DIR / "target" / "release" / "examples" / "benchmark_suite.exe"
        if not exe_path.exists():
            exe_path = WORKSPACE_DIR / "target" / "release" / "examples" / "benchmark_suite"
        cmd = [str(exe_path), str(MODEL_DIR), str(args.runs), args.suite]
        if args.case:
            cmd.append(args.case)
    else:
        exe_path = WORKSPACE_DIR / "target" / "release" / "examples" / "benchmark_olmoe.exe"
        if not exe_path.exists():
            exe_path = WORKSPACE_DIR / "target" / "release" / "examples" / "benchmark_olmoe"
        cmd = [str(exe_path), str(MODEL_DIR), args.prompt, str(args.max_tokens), str(args.runs), str(args.cache)]

    if not exe_path.exists():
        print(f"❌ Không tìm thấy binary đã biên dịch tại {exe_path}. Hãy chạy `cargo build --release --examples` trước.")
        sys.exit(1)

    print(f"\n▶ Đang chạy: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(WORKSPACE_DIR))

    if result.returncode != 0:
        print(f"❌ Lỗi khi chạy benchmark (Exit code {result.returncode}):")
        print(result.stderr)
        sys.exit(result.returncode)

    # 4. Ghi dữ liệu JSON ra file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"\n✅ Đã lưu kết quả đo vào: {output_file.relative_to(WORKSPACE_DIR)}")

    # 5. Tự động đồng bộ báo cáo docs/BENCHMARKS.md
    print("\n🔄 Đang đồng bộ báo cáo docs/BENCHMARKS.md...")
    export_script = WORKSPACE_DIR / "tools" / "export_benchmark_evidence.py"
    subprocess.run([sys.executable, str(export_script)], cwd=str(WORKSPACE_DIR))
    print("🎉 Hoàn tất!")

if __name__ == "__main__":
    main()
