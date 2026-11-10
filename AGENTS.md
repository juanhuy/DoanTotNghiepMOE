# Hướng dẫn làm việc trong MoE-TierEngine

Đồ án Rust OLMoE INT8, CPU/RAM/file + CUDA/cuBLAS tùy hardware. Hai testbed là `i5` và `i7`. Mục tiêu: kết quả đúng, bằng chứng truy được, đồng nghiệp có thể tái lập.

## Hướng tiếp tục hiện hành

- Ưu tiên điều tra OOM và kiểm soát tổng RAM trên i5; các bản vá cache riêng chưa chứng minh total RSS an toàn. Chưa đóng giai đoạn1.
- Mục tiêu ghi báo cáo trên i5/i7: cùng revision/checkpoint, bộ test, prompt/history/context/token limit, số lượt và ngân sách cache RAM tường minh theo MiB/layer. Ghi cấu hình/hardware/raw riêng từng máy; không mặc định đây là cuộc so tốc độ CPU. Backend/threads khác nhau phải công bố.
- Chưa chốt mức cache chung trước khi kiểm ổn định bộ nhớ. Không dùng adaptive cache hoặc mặc định runner như cấu hình nghiệm thu đã thống nhất. AB/BA inclusive/victim là thí nghiệm policy riêng, không bắt buộc cho bộ ghi báo cáo hai máy.

## Đọc đúng ngữ cảnh

- Trước khi tiếp tục lộ trình, đọc `docs/PROGRESS.md` (quy tắc, trạng thái và mục mới nhất); tra mốc cũ khi cần so sánh.
- `docs/PLAN.md`: backlog; `docs/ARCHITECTURE.md`: luồng/invariant; `docs/WORKFLOWS.md`: quy trình và handoff.
- Khi đo/dùng số liệu: đọc `benchmarks/README.md` và skill `moe-benchmark`; dùng nhãn P1/P2/P3 để chọn hồ sơ. Tra entry liên quan trong `benchmarks/manifest.json` khi dùng số liệu lịch sử, không cần đọc cả corpus.
- Agent trên máy đồng nghiệp đọc thêm mục **Tiếp nhận trên testbed i7** trong `docs/WORKFLOWS.md`; chọn `--testbed i7` tường minh, không kế thừa nhãn/kết quả i5 thành số đo của máy mình.
- Khi sửa/review Rust engine/cache/FFI: đọc skill `moe-engine`.
- `baocao/` là bản thảo, không xác nhận số đo; Git history không ghi đè chỉ dẫn hiện hành.

### Đọc có chọn lọc để tiết kiệm ngữ cảnh

- Hiểu hệ thống: đọc SYSTEM_GUIDE; chọn việc: PLAN; viết luận điểm: bảng “Đóng góp trực tiếp” trong EVIDENCE_MATRIX. Nhãn và liên kết tập trung ở benchmarks/README.md.
- P1 là bằng chứng chính; P2 là bổ trợ/quyết định giữ-bỏ; P3 là chẩn đoán/lịch sử, không mở mặc định. Chọn đúng hồ sơ theo câu hỏi, không đọc hết P1 mỗi lượt.
- PROGRESS là nhật ký tra cứu: đọc quy tắc và mục mới nhất bằng tìm tiêu đề/đọc đoạn cần thiết, không đổ toàn bộ lịch sử vào ngữ cảnh.
- Mặc định đọc README thí nghiệm trước. Chỉ trích run/request/trường JSON cần kiểm; lọc trace theo thời gian/trường. Không đọc cả result và turn tương ứng nếu chúng lặp cùng nội dung.
- Nhãn ưu tiên không xác nhận chất lượng số đo. Khi trích số mới hoặc kiểm một claim, vẫn đối chiếu raw/config/provenance/tolerance phù hợp. Kiểm checksum bằng công cụ; chỉ mở source ZIP/log khi truy lỗi/nguồn. Giữ nguyên payload, không xóa hoặc đổi nhãn raw lịch sử.

## Thay đổi và kiểm tra

- Thực hiện yêu cầu đã được giao; dùng giả định hợp lý cho lựa chọn thường lệ. Chỉ hỏi khi thiếu thông tin thay đổi phạm vi hoặc không thể tiếp tục an toàn.
- Giữ diff có mục đích; không đổi module/dependency chỉ để đáp ứng phong cách agent.
- Với lỗi logic, thêm hồi quy thể hiện trigger và chạy trước/sau khi phù hợp. Tài liệu/format không cần test mô phỏng câu chữ.
- `python3 tools/check.py` kiểm tra tooling/tài liệu/bằng chứng; `--rust` thêm fmt/Clippy/Rust tests, `--release` thêm release tests, `--http` thêm stress fixture, `--converter` chạy converter trong môi trường Python có dependency.
- Báo rõ pass/fail/skipped và môi trường. GPU tests có thể pass mà không chạy CUDA; kiểm tra probe trước khi kết luận GPU đã được kiểm chứng.
- Giữ tolerance fixture hiện có; không thay expected output để che regression.
- Không chạy benchmark đồng thời với build/tests/inference hoặc benchmark khác trên cùng máy.

## Agents và cộng tác

- Mặc định một owner thực hiện/tích hợp. Có thể dùng tối đa hai nhánh subagent độc lập khi việc đủ lớn hoặc cần review; giao rõ mục tiêu, file được sửa và kiểm chứng.
- Built-in worker/explorer đủ cho phần lớn việc. Role tùy chọn `.codex/agents/`: `reviewer`, `evidence_reviewer`; chỉ dẫn đọc-only không tạo sandbox riêng.
- Không giao hai agent sửa cùng file. Chỉ owner cập nhật PROGRESS và báo cáo cuối. Reviewer trả file/vị trí/trigger/bằng chứng; không tự merge/sửa số đo.
- Hai người dùng branch riêng; PR nêu vấn đề, hành vi mới, kiểm tra và giới hạn. Handoff theo WORKFLOWS; không tạo transcript/status file trùng nội dung.
- Shell/Git và docs chính thức đủ cho local. Chỉ bật MCP/plugin khi có tác vụ cần; không lưu token, đường dẫn cá nhân hoặc quyền bỏ qua sandbox vào repo.

## Ghi kết quả và minh bạch

- Sau mỗi bước có ý nghĩa, trước khi báo hoàn thành, cập nhật PROGRESS: thay đổi, lệnh kiểm thử, kết quả, số đo nếu có, thất bại/giới hạn và trạng thái lộ trình. Đây là yêu cầu ghi chép, không phải xin xác nhận mỗi bước.
- Không tuyên bố số đo chưa thực hiện. Phân biệt đo mới/hồi cứu, payload/I/O thiết bị, capacity/RSS, engine-cold/OS-cache-cold, logical VRAM/physical allocation.
- Giữ payload bằng chứng duy nhất bất biến. Chỉ bỏ bản raw trùng khi checksum khớp và có alias trong manifest; bản suite trùng ghi ánh xạ vào PROGRESS; sửa kết luận cũ bằng đính chính, không sửa giá trị gốc.
- So sánh hiệu năng khi checkpoint/revision/suite/prompt/token limit/cache/threads/GPU/điều kiện thử được ghi và tương đương. So sánh hai máy là thí nghiệm riêng với so sánh tối ưu code.
- Tối ưu cấp phát/I/O cần đo, không được chứng minh chỉ bằng Rust/Clippy. `unsafe` cần invariant ownership/lifetime/kích thước/thread/context và kiểm chứng phù hợp.
