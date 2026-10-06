# Quy tắc Ghi chép Tiến độ (Progress Logging)

Tuân thủ nghiêm ngặt chỉ thị cốt lõi của dự án trong `AGENTS.md`:

## 1. Đọc trước khi làm
- Luôn đọc `docs/PROGRESS.md` trước khi tiếp tục bất kỳ bước nào trong lộ trình nâng cấp.

## 2. Ghi kết quả sau mỗi bước có ý nghĩa
- Cập nhật nhật ký trước khi báo hoàn thành công việc:
  - Mục tiêu, thay đổi thực tế và các file liên quan.
  - Lệnh kiểm thử, kết quả pass/fail, số test và lỗi còn lại.
  - Số đo hiệu năng nếu có (checkpoint, prompt, số token, cấu hình, trạng thái cache, cách đo, số lần chạy, trước/sau).
  - Thất bại, sự cố và các giới hạn còn tồn tại.
  - Cập nhật trạng thái trong bảng lộ trình.

## 3. Trung thực về số liệu
- Không tuyên bố số đo chưa thực hiện (nếu chưa đo phải ghi rõ **chưa đo**).
- Phân biệt rõ số đo mới vừa chạy thực tế với số liệu hồi cứu từ quá khứ.
- Giữ nguyên các mốc cũ để so sánh. Chỉ so sánh hiệu năng khi điều kiện thử nghiệm (hardware, prompt, thread, cache) là tương đương.
