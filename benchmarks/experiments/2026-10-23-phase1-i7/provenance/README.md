# Hồ sơ nguồn gốc Checkpoint Testbed i7 (Model Provenance)

Thư mục này lưu trữ các bằng chứng kỹ thuật và dữ liệu nguồn gốc của checkpoint đã được sử dụng trong bộ đo benchmark `phase1-v1` trên testbed `i7`.

---

## 1. Đường dẫn mô hình trong bộ đo và tính toàn vẹn hiện tại

1. **Đường dẫn runtime đã sử dụng:**
   - Trong nhật ký khởi động máy chủ [cold/server.log](../cold/server.log) và `warm/server.log`:
     ```text
     moe_tier_engine: MoE server ready address="127.0.0.1:..." directory="/home/nhat-huy/DOANTN/MoE/models/olmoe-1b-7b-int8" model_name="olmoe"
     ```
   - Thư mục trên đĩa: `/home/nhat-huy/DOANTN/MoE/models/olmoe-1b-7b-int8` là thư mục vật lý (không phải symlink).

2. **Đối chiếu giữa "Mô hình hiện tại trên đĩa" và "Mô hình khi đo":**
   - Xem [checkpoint_sha256_comparison.json](checkpoint_sha256_comparison.json):
   - Toàn bộ 10/10 tệp (gồm `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` và 5 tệp `model-00000.safetensors` đến `model-00004.safetensors`) hiện có trên đĩa khớp **100% từng byte mã băm SHA-256** với trường `checkpoint.files` được ghi nhận trong file thô [cold/result.json](../cold/result.json).
   - Xác nhận: Checkpoint hiện tại trên máy hoàn toàn trùng khớp với checkpoint thực tế đã chạy trong bộ đo ngày 23/10/2026.

---

## 2. Nguồn gốc Hugging Face và Revision đã xác minh

1. **Revision và ETag tải về từ Hugging Face:**
   - File [download_metadata.json](download_metadata.json) trích xuất từ bộ nhớ đệm tải `.cache/huggingface/download/*.metadata` ghi nhận:
     - **Commit Hash:** `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` cho toàn bộ các tệp cấu hình và tokenizer.
   - Đối chiếu với cơ sở dữ liệu Hugging Face API chính thức tại [hf_api_model_info.json](hf_api_model_info.json), commit hash `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` xác nhận chính xác mô hình:
     - **Hugging Face Repo ID:** `allenai/OLMoE-1B-7B-0125-Instruct` (phát hành tháng 01/2025).

2. **Đặc điểm Tokenizer và Chat Template của bản 0125-Instruct:**
   - Trong `tokenizer_config.json` và `special_tokens_map.json` gốc của bản 0125:
     - `bos_token`: `|||IP_ADDRESS|||` (ID: 50279).
     - `eos_token`: `|||IP_ADDRESS|||` (ID: 50279).
     - `pad_token`: `<pad>` (ID: 50280).
     - `added_tokens`: 29 tokens (bao gồm các token mặt nạ PII của tập dữ liệu Dolma).
     - `chat_template`: Thêm tường minh ký tự ngắt dòng `\n` sau các thẻ vai trò `{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}` và `{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}`.
   - Dung lượng `tokenizer.json` là 3,565,064 bytes (so với 2,115,598 bytes của bản 0924).

3. **Công cụ đóng gói (Converter):**
   - Cả hai checkpoint i5 và i7 đều sử dụng thuật toán đóng gói gom cụm của Colibri (`flush_every=512`), tạo ra đúng 2,195 tensors phân bố trên 5 shard: `[513, 513, 513, 512, 144]`.
   - Dung lượng byte của 5 tệp safetensors trên i7 trùng khớp từng byte với i5 (`1834100784`, `1657854168`, `1628059584`, `1821755792`, `471116928`).
   - Tuy nhiên, do trọng số số học gốc xuất phát từ hai đợt huấn luyện khác nhau (`0924` vs `0125`), giá trị số học bên trong khác nhau dẫn đến mã băm SHA-256 của các tệp safetensors khác nhau.
