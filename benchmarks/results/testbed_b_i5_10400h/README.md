# Testbed B: Cấu Hình Máy Đo Intel Core i5

Thư mục này lưu trữ toàn bộ dữ liệu benchmark thô được thực thi trên **Máy B**.

## Thông Số Kỹ Thuật (Hardware Specifications)
* **Thành viên phụ trách**: Thành viên 2 (kasiz)
* **Vi xử lý (CPU)**: Intel Core i5 (6 nhân / 12 luồng logic), Max Turbo 4.5+ GHz, 12MB Cache
* **Bộ nhớ trong (RAM)**: 16 GB DDR4 (Băng thông ~25-30 GB/s)
* **Card đồ họa (GPU)**: NVIDIA GeForce GTX 1650 Max-Q (4 GB GDDR5/GDDR6 VRAM, kiến trúc Turing, Compute Capability 7.5)
* **Ổ cứng lưu trữ**: SSD M.2 NVMe PCIe 3.0
* **Số luồng song song tối ưu**: `RAYON_NUM_THREADS=4` (hoặc 6)

## Quy Ước Đặt Tên Tệp
`YYYY-MM-DD-<kịch-bản-đo>.json` (ví dụ: `2026-10-10-testbed-i5-10400h.json`)
