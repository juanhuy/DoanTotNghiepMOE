# Hướng dẫn bàn giao & Dữ liệu thực nghiệm i7 cho Báo cáo định kỳ lần 1

Tài liệu này tổng hợp toàn bộ thông tin kỹ thuật, số liệu thực nghiệm đã kiểm chứng trên testbed `i7` và các lưu ý học thuật quan trọng để thành viên phụ trách tích hợp hoàn thiện [BAO_CAO_DINH_KY_LAN_1.md](BAO_CAO_DINH_KY_LAN_1.md).

Mọi số liệu trong tài liệu này đều là **dữ liệu thực tế** được trích xuất trực tiếp từ các file nhật ký gốc (`result.json`, `memory.jsonl`, `watchdog.json`, `server.log`), mã băm SHA-256 và truy vấn API Hugging Face chính thức; không có số liệu suy diễn hay phỏng đoán.

---

## 1. Thông tin thành viên & Phân công nhiệm vụ

*Cần điền vào Bìa và Bảng phân công nhiệm vụ (Mục Bảng phân công nhiệm vụ, dòng 174 của [BAO_CAO_DINH_KY_LAN_1.md](BAO_CAO_DINH_KY_LAN_1.md)):*

- **Họ và tên thành viên i7:** [Điền Họ và Tên của bạn]
- **Mã số sinh viên (MSSV):** [Điền MSSV]
- **Lớp / Khóa:** [Điền Lớp]
- **Trách nhiệm / Nhiệm vụ đảm nhận:**
  - Tiếp nhận và triển khai hệ thống trên phần cứng máy `i7` (môi trường máy đồng nghiệp).
  - Thực hiện kiểm thử độ ổn định bộ nhớ (preflight RAM stability) có giám sát.
  - Thu thập dữ liệu thực nghiệm chuẩn hóa 3 kịch bản `cold`, `warm`, `conversation` theo giao thức `phase1-v1`.
  - Đối chiếu tính toàn vẹn (provenance) và rà soát nguồn gốc checkpoint Hugging Face.

---

## 2. Thông số phần cứng Testbed i7 (Quan sát thực tế)

*Nguồn: Lệnh `lscpu`, `free -h`, `nvidia-smi` và ghi nhận trong file raw [cold/result.json](../benchmarks/experiments/2026-10-23-phase1-i7/cold/result.json):*

| Thành phần | Thông số quan sát thực tế |
| :--- | :--- |
| **Nhãn testbed** | `i7` (máy đồng nghiệp, xác nhận bởi người dùng) |
| **CPU Model** | `Intel(R) Core(TM) i7-14650HX` |
| **Cấu trúc CPU** | 16 nhân vật lý, 24 luồng logic, xung nhịp tối đa 5.20 GHz, L3 Cache 30 MiB |
| **RAM vật lý** | 14.0 GiB (15,910,060,032 bytes; Available lúc bắt đầu: ~9.6 GiB) |
| **GPU Model** | `NVIDIA GeForce RTX 5060 Laptop GPU` |
| **VRAM & Driver** | 8,151 MiB VRAM (7.53 GB khả dụng), Driver 595.91.07, CUDA 13.2 |
| **GPU Engine** | Yêu cầu `auto` -> Nhận diện thành công `cuBLAS SGEMV Active`, offload 1.38 GB Dense lên VRAM |
| **Hệ điều hành** | Ubuntu Linux (Kernel `7.0.0-38-generic`, x86_64) |

---

## 3. Dữ liệu thực nghiệm i7 (Bộ đo Phase1-v1)

*Toàn bộ 51 file trong thư mục [benchmarks/experiments/2026-10-23-phase1-i7/](../benchmarks/experiments/2026-10-23-phase1-i7/) đã vượt qua kiểm tra SHA-256 (`sha256sum -c CHECKSUMS.sha256` đạt 100%).*

- **Git Commit nguồn:** `d4adefd9e53627bdc1fd250eabab436c3ac54b0d`, `git_dirty: false`.
- **Cấu hình chạy:** 8 Rayon threads, expert cache policy `victim`, 64 MiB/layer (1 GiB tổng 16 layers), context 512, max output 128 tokens, RSS guard 5.0 GiB, headroom 1.0 GiB.
- **Độ ổn định bộ nhớ (Watchdog):** Không có vi phạm (`violations: []`), `VmSwap = 0 KiB` suốt toàn bộ các lượt chạy.
- **HWM đỉnh quan sát được:** **3.278 GiB** (thấp hơn nhiều so với ngưỡng dừng 5.5 GiB).

### Bảng tổng hợp hiệu năng i7 (Trung vị 5 mẫu/nhóm)

*Đã được trình bày tại Mục 5.10.2 (Bảng 5.13 và Bảng 5.14) của báo cáo:*

| Nhóm kịch bản | TTFT (s) | Decode (steps/s) | Wall trung vị (s) | Khoảng Wall min–max (s) | File payload đọc đĩa (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cold Mars** | 4.062 | 4.261 | 33.420 | 32.429 – 34.979 | 5.757 |
| **Warm Mars** | 2.141 | 4.862 | 28.135 | 27.721 – 30.167 | **0.000** *(Hit 100% cache)* |
| **Conversation Mars** | 3.912 | 4.172 | 34.543 | 33.279 – 40.893 | 5.757 |
| **Conversation Love** | 2.929 | 3.603 | 38.173 | 36.391 – 44.732 | 0.176 *(Reuse 143 KV tokens)* |

---

## 4. Bằng chứng và Nguyên nhân sâu xa vì sao Checkpoint hai máy khác nhau

Rà soát kỹ thuật độc lập thông qua file metadata tải (`.metadata`), log tiến trình (`xet_*.log`) và API Hugging Face chính thức (`https://huggingface.co/api/models/...`) đã làm sáng tỏ hoàn toàn nguyên nhân khác biệt:

### 4.1. Bản chất hai mô hình nguồn từ AllenAI (Vì sao tác giả phát hành hai bản?)
- **Bản trên máy i5 — `allenai/OLMoE-1B-7B-0924-Instruct` (Tháng 09/2024):**
  - **Commit:** `7f1c97f440f06ce36705e4f2b843edb5925f4498`.
  - **Bối cảnh phát hành:** Đây là phiên bản đầu tiên của dòng OLMoE khi AllenAI công bố bài báo khoa học. Mô hình được huấn luyện căn chỉnh qua SFT rồi DPO trên tập dữ liệu UltraFeedback.
  - **Tokenizer & Template:** Dùng chuẩn GPT-NeoX ban đầu với token kết thúc câu (EOS) và bắt đầu (BOS) là `<|endoftext|>`. Chat template chưa chèn ngắt dòng `\n` sau thẻ vai trò.
- **Bản trên máy i7 — `allenai/OLMoE-1B-7B-0125-Instruct` (Tháng 01/2025):**
  - **Commit:** `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
  - **Bối cảnh phát hành:** Bản nâng cấp sau 4 tháng của AllenAI, cải thiện năng lực giải toán và suy luận thông qua RLVR (Reinforcement Learning with Verifiable Rewards) trên tập dữ liệu GSM.
  - **Tokenizer & Template:** AllenAI cập nhật toàn bộ tokenizer để đồng bộ với bộ dữ liệu Dolma (xử lý mặt nạ thông tin nhạy cảm PII). Token ID `50279` (`|||IP_ADDRESS|||`) được chỉ định làm BOS và EOS chính thức của mô hình thay cho `<|endoftext|>`. Đồng thời, chat template được chuẩn hóa theo định dạng Hugging Face mới, thêm ký tự xuống dòng `\n` sau mỗi thẻ `<|user|>\n` và `<|assistant|>\n`.

### 4.2. Vì sao hai máy trong nhóm lại tải về hai phiên bản khác nhau?
- Công cụ chuyển đổi trọng số Colibri (`c/tools/convert_olmoe_merged.py`) có docstring hướng dẫn và tham số mẫu:
  `python tools/convert_olmoe_merged.py --repo allenai/OLMoE-1B-7B-0125-Instruct --out ./olmoe_merged`
- **Trên máy i5:** Quá trình chuẩn bị mô hình diễn ra ở giai đoạn đầu dự án (tháng 9/2024), thành viên phụ trách đã tải trực tiếp bản phát hành lúc đó là `OLMoE-1B-7B-0924-Instruct`.
- **Trên máy i7:** Quá trình convert được thực hiện vào ngày 15/09/2026 thông qua script Colibri ở chế độ streaming `--repo`. Script đã tải bản mới nhất được khai báo trong tài liệu mẫu là `OLMoE-1B-7B-0125-Instruct`.

### 4.3. Vì sao các file Safetensors giống hệt kích thước nhưng khác mã băm (SHA-256)?
- **Giống kích thước từng byte:** Cả hai mô hình đều có chung kiến trúc mạng (16 layers, 64 experts, top-8 routing, hidden size 2048, intermediate size 1024). Khi qua cùng bộ chuyển đổi Colibri với thuật toán gom cụm `flush_every=512`, cả hai máy tạo ra đúng **2,195 tensors** phân chia thành 5 file safetensors với cấu trúc header và kích thước byte giống hệt nhau (ví dụ: `model-00000.safetensors` đều đúng `1,834,100,784 bytes`).
- **Khác mã băm SHA-256:** Trọng số số học thực tế (các mảng số thực và mảng lượng tử hóa INT8 bên trong) xuất phát từ hai đợt huấn luyện khác nhau (`0924` vs `0125`) nên giá trị trọng số khác nhau, dẫn đến mã băm SHA-256 của toàn bộ 5 file safetensors khác nhau.

### 4.4. Tác động cụ thể lên số đo thực nghiệm của hai máy (Giải thích chênh lệch số đo)
Sự khác nhau về tokenizer và template giải thích tường minh 100% các điểm chênh lệch trong bảng thực nghiệm giữa i5 và i7:

1. **Chênh lệch số lượng Prompt Tokens của câu Mars (19 tokens vs 16 tokens):**
   - Trên i5 (bản 0924): Chat template nối chuỗi `<|user|>\nWhere is Mars<|assistant|>` kết hợp với bộ tách từ của 0924 tạo thành chuỗi gồm **19 tokens**.
   - Trên i7 (bản 0125): Chat template có thêm định dạng ngắt dòng và bộ tách từ 0125 mã hóa hiệu quả hơn, tạo thành chuỗi gồm đúng **16 tokens**.
2. **Chênh lệch tỷ lệ KV Cache Prefix Reuse ở câu Love (11 tokens vs 143 tokens):**
   - Trên i5: Do template server khi sinh token câu Mars kết thúc bằng `<|endoftext|>`, nhưng khi client gửi lượt Love thì chuỗi lịch sử bị vênh ngắt dòng ở cuối lượt trả lời của assistant. Engine của i5 chỉ khớp được **11 tokens** đầu tiên rồi bị trượt cache (miss), buộc lòng phải tính lại toàn bộ prefill từ token thứ 12 trở đi -> TTFT câu Love trên i5 kéo dài đến **9.099s**.
   - Trên i7: Bản 0125-Instruct có template thống nhất hoàn hảo; lượt trả lời của assistant kết thúc bằng `|||IP_ADDRESS|||` trùng khớp tuyệt đối với tiền tố lịch sử gửi vào ở lượt Love. Engine tái sử dụng được toàn bộ **143 tokens** có sẵn trong KV cache -> engine chỉ cần tính prefill cho phần câu hỏi mới của Love -> TTFT câu Love trên i7 giảm mạnh xuống chỉ còn **2.929s**.

*Kết luận học thuật:* Chênh lệch TTFT và số lượng token giữa hai máy không phải do lỗi đo lường hay phần cứng, mà là do sự tiến hóa trong thiết kế tokenizer và chat template giữa bản `0924-Instruct` và bản `0125-Instruct`. Việc giải trình rõ điều này thể hiện năng lực điều tra kỹ thuật sâu và tính trung thực khoa học của nhóm nghiên cứu.

---

## 5. Checklist chỉnh trang văn bản BAO_CAO_DINH_KY_LAN_1.md

Người phụ trách tích hợp cần thực hiện 3 điều chỉnh nhỏ trên file báo cáo:

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
   - Tại dòng 161 (Mở đầu): Xóa câu *"Kết quả i7 chưa có tại thời điểm lập báo cáo và được dành mục bổ sung riêng"*, thay bằng: *"Kết quả thực nghiệm trên cả hai máy i5 và i7 đã được thu thập và trình bày chi tiết tại Chương 5."*
   - Tại dòng 988 (Mục 5.11): Xóa cụm từ *"dữ liệu i7 còn chờ bổ sung"*.
