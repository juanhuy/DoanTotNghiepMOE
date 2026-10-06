# Quy tắc Kỷ luật Kỹ thuật (Engineering Discipline)

Hệ thống tuân thủ nghiêm ngặt quy trình AI-First Engineering lấy cảm hứng từ `mattpocock/skills` và Best Practices trong phát triển phần mềm hiệu năng cao:

## 1. Không Vibe-Coding (No Assumptions)
- Trước khi sửa đổi hoặc viết tính năng mới, phải làm rõ input/output, kiểu dữ liệu, giới hạn bộ nhớ (RAM/SSD), và các edge-cases.
- Nếu yêu cầu chưa rõ ràng hoặc có nhiều hướng tiếp cận đánh đổi (trade-offs), phải đặt câu hỏi làm rõ (Socratic questioning / Grill-me).

## 2. Quy trình TDD (Test-Driven Development)
- **Red Phase**: Tạo test case hoặc benchmark fixture thể hiện hành vi mong muốn trước khi can thiệp code chính. Đảm bảo test thất bại đúng lý do.
- **Green Phase**: Viết mã tối thiểu, chính xác để pass test.
- **Refactor Phase**: Tối ưu hóa hiệu năng, dọn dẹp mã nguồn, kiểm tra memory allocation mà không làm gãy test.

## 3. Đánh giá Phản biện (Anti-Sycophancy Review)
- Code review phải chỉ ra thẳng thắn các nguy cơ:
  - Cấp phát bộ nhớ không cần thiết trong hot-loop.
  - Race condition, deadlock hoặc contention khi dùng Rayon/Mutex.
  - Sai số tính toán vượt ngưỡng dung sai so với baseline (`2e-5`).
  - Khả năng gây OOM khi nạp nhiều expert.

## 4. Bảo tồn Ngữ cảnh (Context Handoff)
- Mọi mốc cải tiến hoặc thử nghiệm đều phải được lưu trữ có cấu trúc để các phiên làm việc sau kế thừa liền mạch.
