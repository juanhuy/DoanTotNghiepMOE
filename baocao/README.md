# Hướng dẫn bàn giao & Dữ liệu thực nghiệm i7 cho Báo cáo định kỳ lần 1

Tài liệu này tổng hợp toàn bộ thông tin kỹ thuật, số liệu thực nghiệm đã kiểm chứng trên testbed `i7` và các lưu ý khoa học để thành viên phụ trách tích hợp hoàn thiện [BAO_CAO_DINH_KY_LAN_1.md](BAO_CAO_DINH_KY_LAN_1.md).

Mục tiêu đã thống nhất:
- Đánh giá khả năng vận hành riêng từng cấu hình: testbed `i5` với checkpoint `0924-Instruct` và testbed `i7` với checkpoint `0125-Instruct`.
- Không xếp hạng hai máy hay hai mô hình; không quy trực tiếp chênh lệch hiệu năng cho riêng CPU hay GPU khi chưa có thí nghiệm đối chứng có kiểm soát.
- Mọi số liệu trong tài liệu này được trích xuất trực tiếp từ các file nhật ký gốc (`result.json`, `memory.jsonl`, `watchdog.json`, `server.log`), mã băm SHA-256 và truy vấn API Hugging Face chính thức; giới hạn kết luận trong phạm vi và điều kiện đo thực tế.

---

## 1. Thông tin thành viên & Phân công nhiệm vụ

*Cần điền vào Bìa và Bảng phân công nhiệm vụ (dòng 174 của [BAO_CAO_DINH_KY_LAN_1.md](BAO_CAO_DINH_KY_LAN_1.md)):*

- **Họ và tên thành viên i7:** [CẦN BỔ SUNG họ và tên thành viên i7]
- **Mã số sinh viên (MSSV):** [CẦN BỔ SUNG MSSV]
- **Lớp / Khóa:** [CẦN BỔ SUNG lớp]
- **Trách nhiệm / Nhiệm vụ đảm nhận:**
  - Tiếp nhận và triển khai hệ thống trên phần cứng máy `i7` (môi trường máy đồng nghiệp).
  - Thực hiện kiểm thử độ ổn định bộ nhớ (preflight RAM stability) có giám sát.
  - Thu thập dữ liệu thực nghiệm chuẩn hóa 3 kịch bản `cold`, `warm`, `conversation` theo giao thức `phase1-v1`.
  - Đối chiếu tính toàn vẹn (provenance) và rà soát nguồn gốc checkpoint Hugging Face trên máy i7.

---

## 2. Thông số phần cứng Testbed i7 (Quan sát thực tế)

*Nguồn: Lệnh `lscpu`, `free -h`, `nvidia-smi` và ghi nhận trong file raw [cold/result.json](../benchmarks/experiments/2026-10-23-phase1-i7/cold/result.json):*

| Thành phần | Thông số quan sát thực tế |
| :--- | :--- |
| **Nhãn testbed** | `i7` (máy đồng nghiệp, xác nhận bởi người dùng) |
| **CPU Model** | `Intel(R) Core(TM) i7-14650HX` |
| **Cấu trúc CPU** | 16 nhân vật lý, 24 luồng logic, xung nhịp tối đa 5.20 GHz, L3 Cache 30 MiB |
| **RAM vật lý** | 15,910,060,032 bytes (≈ 14.817 GiB / 15.910 GB; Available lúc bắt đầu: ~9.6 GiB) |
| **GPU Model** | `NVIDIA GeForce RTX 5060 Laptop GPU` |
| **VRAM & Driver** | 8,151 MiB VRAM (7.53 GB khả dụng), Driver 595.91.07, CUDA 13.2 |
| **GPU Engine** | Yêu cầu `auto` -> Nhận diện thành công `cuBLAS SGEMV Active`, offload 1.38 GB Dense lên VRAM |
| **Hệ điều hành** | Ubuntu Linux (Kernel `7.0.0-38-generic`, x86_64) |

---

## 3. Dữ liệu thực nghiệm Testbed i7 (Bộ đo Phase1-v1)

*Toàn bộ 51 file trong thư mục [benchmarks/experiments/2026-10-23-phase1-i7/](../benchmarks/experiments/2026-10-23-phase1-i7/) đã được kiểm tra tính toàn vẹn chữ ký số SHA-256 (`sha256sum -c CHECKSUMS.sha256` đạt 100%).*

- **Git Commit nguồn:** `d4adefd9e53627bdc1fd250eabab436c3ac54b0d`, `git_dirty: false`.
- **Cấu hình chạy:** 8 Rayon threads, expert cache policy `victim`, 64 MiB/layer (tổng 1 GiB / 16 layers), context 512, max output 128 tokens, RSS guard 5.0 GiB, headroom 1.0 GiB.
- **Độ ổn định bộ nhớ (Watchdog):** Không có vi phạm (`violations: []`), `VmSwap = 0 KiB` suốt các lượt chạy.
- **HWM đỉnh quan sát được:** **3.278 GiB** (trong phạm vi 3 kịch bản đã đo; không cam đoan an toàn tuyệt đối cho mọi khối lượng công việc dài hạn).
- **Điều kiện kết thúc:** Toàn bộ 20 mẫu đo đều kết thúc với `finish_reason=length` tại 128 completion tokens (đạt giới hạn token sinh chủ động, không dừng bằng EOS).

### 3.1. Bảng tổng hợp hiệu năng i7 (Trung vị 5 mẫu/nhóm - Bảng 5.13 báo cáo)

| Nhóm kịch bản | TTFT (s) | Decode (steps/s) | Wall trung vị (s) | Khoảng Wall min–max (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Cold Mars** | **4.062** | **4.261** | **33.420** | 32.429 – 34.979 |
| **Warm Mars** | **2.141** | **4.862** | **28.135** | 27.721 – 30.167 |
| **Conversation Mars** | **3.912** | **4.172** | **34.543** | 33.279 – 40.893 |
| **Conversation Love** | **2.929** | **3.603** | **38.173** | 36.391 – 44.732 |

### 3.2. Bảng phân tích chi phí I/O và Tính toán i7 (Bảng 5.14 báo cáo)

*Lưu ý: Các timer có thể chồng lấn, không cộng thành tổng thời gian Wall. File payload là lượng byte ứng dụng yêu cầu đọc qua pread/syscalls, có thể đến từ OS page cache; không đồng nhất với lượng dữ liệu đọc vật lý từ đĩa SSD.*

| Nhóm kịch bản | Chờ nạp Expert (s) | Tính toán Expert (s) | Xử lý Attention (s) | File payload đọc đĩa (GB) |
| :--- | :---: | :---: | :---: | :---: |
| **Cold Mars** | 13.485 | 8.312 | 0.864 | 5.757 |
| **Warm Mars** | 9.573 | 7.664 | 0.855 | **0.000** |
| **Conversation Mars** | 13.745 | 8.990 | 0.910 | 5.757 |
| **Conversation Love** | 13.718 | 9.083 | 1.922 | **0.176** |

### 3.3. Bảng quan sát độ ổn định bộ nhớ RAM trên i7 (Bảng 5.15 báo cáo)

| Kịch bản | Đỉnh RAM HWM quan sát (GiB) | Bộ nhớ tráo VmSwap (KiB) | RAM khả dụng thấp nhất (GiB) | Thời gian quan sát (s) | Vi phạm Watchdog |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cold** | 3.161 | 0 | 5.463 | 169.1 | 0 (`[]`) |
| **Warm** | **3.278** *(Cao nhất)* | 0 | 6.347 | 314.1 | 0 (`[]`) |
| **Conversation** | 3.164 | 0 | 5.607 | 378.2 | 0 (`[]`) |

### 3.4. Bảng đối chiếu số liệu vận hành giữa Testbed i5 và Testbed i7

*Trình bày kết quả hai cấu hình riêng biệt; công bố rõ khác biệt về model/workload/backend:*

| Chỉ số / Đặc tính | Testbed i5 (GTX 1650 Max-Q) | Testbed i7 (RTX 5060 Laptop) | Ghi chú đối chiếu thực nghiệm |
| :--- | :---: | :---: | :--- |
| **Checkpoint nguồn** | `0924-Instruct` (commit `7f1c97f4...`) | `0125-Instruct` (commit `b89a7c4b...`) | Hai phiên bản phát hành khác nhau từ AllenAI; cấu trúc 2,195 tensors giống nhau nhưng trọng số khác nhau. |
| **Prompt tokens Mars** | **19 tokens** | **16 tokens** | Khác biệt về số lượng token đầu vào do bộ tách từ và template của hai phiên bản. |
| **KV Cache Reuse Love** | **11 tokens** (TTFT: 9.099s) | **143 tokens** (TTFT: **2.929s**) | Quan sát thấy trên i7 lượt Love tái sử dụng được 143 tokens tiền tố; đưa ra giả thuyết cấu trúc template bản 0125 tương thích tốt hơn với chuỗi lịch sử. |
| **Thời gian Attention** | ~1.86 s | **~0.86 s** | Khác biệt về năng lực tính toán và dung lượng VRAM của hai thiết bị GPU; chưa có thí nghiệm cô lập nguyên nhân. |
| **Tốc độ giải mã (Decode)**| 3.66 – 3.99 steps/s | **4.26 – 4.86 steps/s** | Tốc độ giải mã quan sát trên hai hệ thống có sự khác nhau về phần cứng và xung nhịp. |
| **File payload Warm Mars** | **34.176 GB** (31.829 GiB) | **0.000 GB** | Trên i5 với cache 64 MiB/layer các misses tiếp tục yêu cầu đọc file; trên i7 victim cache trên VRAM đáp ứng các misses qua PCIe DMA. |
| **Bộ nhớ RAM quan sát** | Đỉnh HWM 3.494 GiB, VmSwap = 0 | Đỉnh HWM 3.278 GiB, VmSwap = 0 | Cả hai cấu hình vận hành trong giới hạn an toàn ở các lượt đo đã thực hiện. |

---

## 4. Hồ sơ nguồn gốc Checkpoint Testbed i7 (Model Provenance)

Hồ sơ và bằng chứng chi tiết được lưu trữ tại [benchmarks/experiments/2026-10-23-phase1-i7/provenance/](../benchmarks/experiments/2026-10-23-phase1-i7/provenance/):

1. **Đường dẫn runtime và đối chiếu tính toàn vẹn:**
   - Trong `server.log`, đường dẫn mô hình là `/home/nhat-huy/DOANTN/MoE/models/olmoe-1b-7b-int8`.
   - Đối chiếu SHA-256 (xem [checkpoint_sha256_comparison.json](../benchmarks/experiments/2026-10-23-phase1-i7/provenance/checkpoint_sha256_comparison.json)): Toàn bộ 10/10 file trên đĩa khớp 100% mã băm SHA-256 với trường `checkpoint.files` trong file raw `cold/result.json`. Checkpoint hiện tại chính là checkpoint đã được chạy trong bộ đo.
2. **Xác nhận Repo ID và Revision từ Hugging Face:**
   - Metadata tải về lưu tại `.cache/huggingface/download/*.metadata` (xem [download_metadata.json](../benchmarks/experiments/2026-10-23-phase1-i7/provenance/download_metadata.json)) ghi nhận commit hash `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
   - Đối chiếu API Hugging Face chính thức (xem [hf_api_model_info.json](../benchmarks/experiments/2026-10-23-phase1-i7/provenance/hf_api_model_info.json)), commit hash này xác nhận chính xác mô hình: **`allenai/OLMoE-1B-7B-0125-Instruct`** (phát hành tháng 01/2025).
3. **Đặc điểm kỹ thuật bản 0125-Instruct:**
   - Trong `tokenizer_config.json` và `special_tokens_map.json`: token `bos_token` và `eos_token` được định nghĩa là `|||IP_ADDRESS|||` (ID: 50279, token mặt nạ PII của tập dữ liệu Dolma) thay cho `<|endoftext|>`.
   - File `tokenizer.json` có kích thước 3,565,064 bytes (so với 2,115,598 bytes của bản 0924).
   - Cả hai checkpoint đều gồm đúng 2,195 tensors phân bố trên 5 shard: `[513, 513, 513, 512, 144]`. Kích thước từng file safetensors khớp từng byte nhưng mã băm SHA-256 khác nhau do giá trị số học của trọng số xuất phát từ hai đợt huấn luyện khác nhau.

---

## 5. Checklist chỉnh trang văn bản BAO_CAO_DINH_KY_LAN_1.md

Người phụ trách tích hợp cần thực hiện các điều chỉnh trên file báo cáo:

1. **Sửa lỗi trùng số thứ tự bảng:**
   - Tại dòng 990 (Mục 5.11): Đổi `**Bảng 5.12. Kế hoạch kỳ tiếp theo**` thành `**Bảng 5.16. Kế hoạch kỳ tiếp theo**` (tránh trùng với Bảng 5.12 ở dòng 908 về từng mẫu đo i7).
2. **Cập nhật danh mục bảng ở Lời mở đầu (dòng 146–147):**
   - Thay thế mục `- Bảng 5.11. Dữ liệu i7 cần bổ sung` bằng danh sách đầy đủ:
     - `Bảng 5.11. Môi trường và cấu hình thực nghiệm trên máy i7.`
     - `Bảng 5.12. Từng mẫu đo của bộ i7 phase1-v1.`
     - `Bảng 5.13. Tổng hợp hiệu năng i7, trung vị của 5 mẫu/nhóm.`
     - `Bảng 5.14. Phân tích chi phí expert, attention và file payload trên i7.`
     - `Bảng 5.15. Quan sát bộ nhớ máy i7 trong bộ đo chính.`
     - `Bảng 5.16. Kế hoạch kỳ tiếp theo.`
3. **Cập nhật câu chữ phản ánh tiến độ thực tế:**
   - Tại dòng 161 (Mở đầu): Xóa câu *"Kết quả i7 chưa có tại thời điểm lập báo cáo và được dành mục bổ sung riêng"*, thay bằng: *"Kết quả vận hành trên testbed i7 được thu thập độc lập và trình bày tại Mục 5.10."*
   - Tại dòng 205: Cập nhật tình trạng câu hỏi nghiên cứu i7 thành đã đo và trình bày tại Mục 5.10.
   - Tại dòng 988 (Mục 5.11): Xóa cụm từ *"dữ liệu i7 còn chờ bổ sung"*.
   - Giữ nguyên các vị trí `[CẦN BỔ SUNG họ tên/MSSV/lớp]` để thành viên tự điền chính thức.
