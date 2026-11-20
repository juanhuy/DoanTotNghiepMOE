# Giới hạn và hướng phát triển (bản thảo cho báo cáo định kỳ lần 2)

Mục này dùng cho chương kết luận của báo cáo lần 2 và để chuẩn bị trả lời hội đồng. Mỗi nhận định ghi rõ là **đo** (có hồ sơ) hay **suy ra** (tính từ số đo, chưa thực nghiệm riêng). Số liệu lấy từ hồ sơ đã khóa checksum; không trích số khác ngày hoặc khác máy để so tốc độ.

## 1. Qwen3-30B-A3B chưa dùng được GPU trên i5

**Hiện trạng (đo).**
- Trên i5 (GTX 1650 4GB), Qwen3 chạy hoàn toàn bằng CPU.
- Chế độ GPU của engine đưa toàn bộ trọng số dense (attention, router, LM head) lên VRAM dưới dạng F32 để cuBLAS SGEMV tính. Dense của Qwen3 lưu BF16 là 2,46GB, khi nới sang F32 thành khoảng 4,9GB. Trên i7 lượt GPU cấp phát 4643MiB cho phần này ([hồ sơ i7](../../benchmarks/experiments/2026-11-18-qwen3-ram-cap-i7/README.md)).
- Lượng đó vượt VRAM của GTX 1650, nên engine phát hiện lúc khởi động và tự chạy CPU, không lỗi (`tools/demo.sh qwen3-gpu` trên i5, PROGRESS 20/11).
- Pool expert trên GPU hiện gắn với khối dense trên GPU, nên cũng không được bật.

**Vì sao lợi ích dự kiến có giới hạn.**
- **Đo:** với Qwen3 trên i5, 58–76% thời gian request là chờ đọc expert từ SSD; mỗi request 64 token đọc lại 60–91GB payload expert ([hồ sơ i5](../../benchmarks/experiments/2026-11-17-qwen3-ram-cap-i5/README.md)). Phần dense (attention và LM head) chiếm 12–23% thời gian mỗi request (`attention_ms + lm_head_ms` trên `total_ms`, 16 request ở hai mức trần).
- **Suy ra:** theo định luật Amdahl, kể cả khi phần dense nhanh vô hạn, toàn request nhanh hơn tối đa 1/(1 − 0,12) đến 1/(1 − 0,23), tức khoảng 1,13–1,30 lần. Thực tế thấp hơn vì còn chi phí chép dữ liệu qua PCIe.
- **Đo:** trong cùng trần RAM, tăng cache expert từ 64 lên 96MiB/layer làm hit rate tăng từ 50 lên 60% và payload đọc giảm 20%, nhưng tốc độ không cải thiện nhất quán (A/B kiểu ABBA). Nguyên nhân là page cache bị thu hẹp tương ứng ([hồ sơ cache A/B](../../benchmarks/experiments/2026-11-18-qwen3-cache-ab-i5/README.md)).
- **Suy ra:** 4GB VRAM, nếu chỉ dùng chứa expert, giữ được khoảng 700 trong 6.144 expert (khoảng 11%). Khác với thí nghiệm cache A/B, đây là bộ nhớ thêm chứ không lấy chỗ của page cache, nên có thể có lợi. Nhưng chưa đo, và mức lợi bị giới hạn bởi tỉ lệ 11%.

**GPU có ích khi đủ dung lượng (đo).**
- OLMoE-1B-7B trên chính GTX 1650: tính expert trên GPU nhanh hơn tắt GPU ở 8/8 request, A/B chính thức ([hồ sơ A1 i5](../../benchmarks/experiments/2026-11-14-gpu-compute-i5/README.md)).
- Qwen3 trên i7 (RTX 5060 8GB): lượt khám phá với dense và pool 565 expert trên GPU đạt decode trung vị 2,22 tok/s, so với 1,53 tok/s khi chạy CPU cùng máy. Lượt này đổi nhiều yếu tố cùng lúc, không phải A/B, nên không quy được phần tăng cho thành phần nào.

**Kết luận cho báo cáo.** Hạn chế trên i5 là hạn chế về **dung lượng VRAM**, không phải về kiến trúc: cùng engine dùng được GPU với OLMoE trên i5 và với Qwen3 trên i7. Với Qwen3 trên i5, nút thắt chính là I/O đọc expert, nên đưa dense lên GPU chỉ cải thiện được tối đa khoảng 13–30%.

## 2. Các giới hạn khác của bằng chứng hiện có

- **Tính đúng:** parity trên model thật dùng 6 chuỗi ngắn (159 vị trí với Qwen3, 186 với OLMoE), qua đường `model.logits` trên CPU. Đây là độ lệch so với Transformers và với BF16 gốc, không phải điểm chất lượng tác vụ (perplexity hay benchmark chuẩn) ([hồ sơ parity](../../benchmarks/experiments/2026-11-20-real-model-parity-i5/README.md)).
- **Số lượt:** các hồ sơ Qwen3 dưới trần RAM là một lượt mỗi mức, 8 request × 64 token. Đây là quan sát ổn định có giới hạn, không phải phân phối thống kê.
- **Hai máy:** i5 và i7 dùng chung checkpoint (hash 478/478 khớp với Qwen3, 2195/2195 tensor với OLMoE) và cho text giống nhau. Tốc độ tuyệt đối không được so giữa hai máy vì khác CPU, SSD và ngày đo.
- **Tốc độ đọc:** tốc độ đọc được ghi là payload engine chia thời gian đọc của engine, không phải băng thông thiết bị đo độc lập.

## 3. Hướng phát triển

1. **Dùng VRAM làm kho expert khi dense không vừa** (đã đánh giá khả thi, chưa làm).
   - Kernel tính expert trên GPU nhận đầu vào và trả kết quả qua RAM, không cần dense trên GPU (`src/olmoe.rs`, A1).
   - Chỉ cần tách pool expert khỏi khối dense GPU.
   - Cần đo A/B kiểu ABBA trên Qwen3 i5 để biết 11% sức chứa thêm có bù được I/O không.
2. **Dense 16-bit trên GPU.** Giữ BF16/FP16 trên VRAM (khoảng 2,5GB thay vì 4,9GB) bằng GEMM nửa độ chính xác; khi đó cả dense lẫn một phần expert vừa 4GB. Cần kiểm lại logits so với CPU.
3. **Expert INT4.** Giảm một nửa lượng byte phải đọc mỗi lần, đánh thẳng vào nút thắt I/O. Code có đường INT4, nhưng chưa có hồ sơ đo đủ provenance cho INT4; cần parity hoặc độ lệch logits (như hồ sơ parity 20/11) trước khi đo tốc độ.
4. **Định tuyến có tính đến cache.** Ưu tiên expert đã có trong RAM/VRAM khi router gần hòa. Hồ sơ parity cho thấy các vị trí đổi top-1 có biên rất nhỏ. Hướng này đổi đầu ra của model, nên phải đánh giá chất lượng song song với tốc độ.

## 4. Câu trả lời ngắn khi được hỏi "sao Qwen3 không dùng GPU trên máy này?"

> Do dung lượng VRAM: dense của Qwen3 khi đưa lên GPU ở dạng F32 khoảng 4,9GB, lớn hơn GTX 1650 4GB, nên engine tự chuyển sang CPU. Kể cả đưa được thì lợi ích nhỏ: bọn em đo được 58–76% thời gian là chờ đọc expert từ SSD, phần dense chỉ 12–23%, nên theo Amdahl tối đa nhanh khoảng 1,13–1,30 lần. Khi VRAM đủ thì GPU có ích: OLMoE trên chính máy này nhanh hơn 8/8 request khi tính expert trên GPU, và Qwen3 trên RTX 5060 8GB đạt 2,22 so với 1,53 tok/s (lượt khám phá). Hướng tiếp theo là dùng VRAM làm kho expert, dense 16-bit trên GPU và expert INT4.

Không nên nói:
- "GPU không giúp gì cho Qwen3": chưa đo trên i5, và số liệu i7 cho thấy có giúp.
- Một con số tăng tốc cho các hướng phát triển: chưa đo.
