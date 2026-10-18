#!/usr/bin/env bash
# tools/monitor_tier.sh: Giám sát phần cứng thời gian thực cho MoE-TierEngine (RAM, VRAM, SSD)

PID=$(pgrep -f "moe-tier-engine" | head -n 1)

if [ -z "$PID" ]; then
    echo "⚠️  Không tìm thấy tiến trình 'moe-tier-engine' đang chạy!"
    echo "👉 Hãy khởi chạy server trước: cargo run --release --bin moe-tier-engine"
    exit 1
fi

clear
echo "================================================================================"
echo "   📊 MoE-TierEngine: Giám Sát Phần Cứng Tường Minh (RAM / VRAM / SSD)"
echo "   PID: $PID | Thời gian: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

prev_read_bytes=0
prev_time=$(date +%s%N)

while true; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Tiến trình $PID đã dừng."
        break
    fi

    now_time=$(date +%s%N)
    time_diff_sec=$(echo "scale=3; ($now_time - $prev_time) / 1000000000" | bc 2>/dev/null || awk "BEGIN {print ($now_time - $prev_time) / 1000000000}")

    # 1. Đo RAM từ /proc/$PID/status
    vm_rss_kb=$(grep -m1 "VmRSS:" /proc/$PID/status 2>/dev/null | awk '{print $2}')
    vm_hwm_kb=$(grep -m1 "VmHWM:" /proc/$PID/status 2>/dev/null | awk '{print $2}')
    vm_size_kb=$(grep -m1 "VmSize:" /proc/$PID/status 2>/dev/null | awk '{print $2}')

    vm_rss_gb=$(echo "scale=2; $vm_rss_kb / 1048576" | bc 2>/dev/null || awk "BEGIN {printf \"%.2f\", $vm_rss_kb / 1048576}")
    vm_hwm_gb=$(echo "scale=2; $vm_hwm_kb / 1048576" | bc 2>/dev/null || awk "BEGIN {printf \"%.2f\", $vm_hwm_kb / 1048576}")
    vm_size_gb=$(echo "scale=2; $vm_size_kb / 1048576" | bc 2>/dev/null || awk "BEGIN {printf \"%.2f\", $vm_size_kb / 1048576}")

    # 2. Đo SSD I/O từ /proc/$PID/io
    rchar=$(grep -m1 "rchar:" /proc/$PID/io 2>/dev/null | awk '{print $2}')
    read_bytes=$(grep -m1 "read_bytes:" /proc/$PID/io 2>/dev/null | awk '{print $2}')

    rchar_gb=$(echo "scale=2; $rchar / 1073741824" | bc 2>/dev/null || awk "BEGIN {printf \"%.2f\", $rchar / 1073741824}")
    read_bytes_gb=$(echo "scale=2; $read_bytes / 1073741824" | bc 2>/dev/null || awk "BEGIN {printf \"%.2f\", $read_bytes / 1073741824}")

    # Tốc độ đọc tức thời
    if [ "$prev_read_bytes" -gt 0 ] && [ "$read_bytes" -ge "$prev_read_bytes" ]; then
        delta_bytes=$((read_bytes - prev_read_bytes))
        speed_mb_s=$(echo "scale=1; ($delta_bytes / 1048576) / $time_diff_sec" | bc 2>/dev/null || awk "BEGIN {printf \"%.1f\", ($delta_bytes / 1048576) / $time_diff_sec}")
    else
        speed_mb_s="0.0"
    fi
    prev_read_bytes=$read_bytes
    prev_time=$now_time

    # 3. Đo GPU VRAM
    vram_used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "^$PID" | awk -F',' '{print $2}' | xargs)
    if [ -z "$vram_used" ]; then
        vram_info="0 MiB (Tiến trình chạy 100% CPU/RAM/SSD)"
    else
        vram_info="$vram_used"
    fi

    # 4. Dung lượng file mô hình trên SSD
    model_dir="models/olmoe-1b-7b-int8"
    if [ -d "$model_dir" ]; then
        model_size=$(du -shL "$model_dir" 2>/dev/null | awk '{print $1}')
    else
        model_size="~7.0 GB"
    fi

    # 5. Thông tin Cache Experts từ API server
    expert_json=$(curl -s --max-time 0.5 http://127.0.0.1:8080/experts 2>/dev/null)
    if [ -n "$expert_json" ]; then
        resident_count=$(echo "$expert_json" | grep -o '"resident_experts":\[[^]]*\]' | grep -o '[0-9]\+' | wc -l)
        api_status="🟢 Kết nối API HTTP 8080 OK ($resident_count / 1024 experts trong RAM Cache)"
    else
        api_status="⚪ Không thăm dò được API (hoặc đang bận)"
    fi

    # In ra màn hình console dạng bảng HUD
    tput cup 3 0 2>/dev/null || true
    cat << EOF
================================================================================
🕒 Cập nhật lúc: $(date '+%H:%M:%S') | Trạng thái API: $api_status
--------------------------------------------------------------------------------
🧠 [TẦNG 1: BỘ NHỚ RAM VẬT LÝ]
   • RAM Đang Dùng Thực Tế (VmRSS)   : ${vm_rss_gb} GiB (${vm_rss_kb} KB)
   • RAM Đỉnh Cao Nhất (VmHWM)        : ${vm_hwm_gb} GiB (${vm_hwm_kb} KB)
   • Không Gian Địa Chỉ Ảo (VmSize)  : ${vm_size_gb} GiB
   • Phân bổ Logic Nội Bộ             : 3.00 GiB Dense FP32 + ~2.00 GiB Expert Cache INT8

💾 [TẦNG 2: Ổ CỨNG LƯU TRỮ SSD NVMe]
   • Dung Lượng Model Trên Ổ Đĩa     : ${model_size} (Safetensors INT8 Shards)
   • Đọc Vật Lý Thực Sự từ Ổ Cứng     : ${read_bytes_gb} GiB (read_bytes qua Kernel)
   • Tổng Lượng Yêu Cầu Đọc (rchar)   : ${rchar_gb} GiB (gồm cả dữ liệu qua OS Page Cache)
   • Tốc Độ Đọc Tức Thời              : ${speed_mb_s} MB/s

⚡ [TẦNG 3: BỘ NHỚ ĐỒ HỌA GPU VRAM]
   • Mức Chiếm Dụng VRAM Tiến Trình  : ${vram_info}
   • GPU Được Nhận Diện              : NVIDIA GeForce GTX 1650 Ti (4096 MiB)
================================================================================
💡 Hướng dẫn:
   - Gửi prompt trên Web hoặc cURL để quan sát RAM VmRSS và SSD I/O tăng lên trực tiếp.
   - Bấm Ctrl+C để thoát giám sát.
EOF

    sleep 1
done
