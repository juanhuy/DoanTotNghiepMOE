# Mô phỏng pool expert trên VRAM cho Qwen3-30B-A3B, i5, 24/11/2026

**Nhãn: P2** (quyết định có làm bước 2 hay không). Đây là mô phỏng số lần nạp expert, **không phải số đo thời gian**.

## Câu hỏi

Trên i5, Qwen3 chạy CPU-only vì dense F32 không vừa GTX 1650 4GB. Nếu tách pool expert VRAM khỏi dense GPU (A1 tính expert trên GPU, dense vẫn trên CPU), pool khoảng 700 slot giảm được bao nhiêu lần đọc expert từ SSD? Ngưỡng đặt trước khi chạy (PROGRESS 23/11): giảm dưới 25% thì đề nghị dừng hướng này.

## Cách làm

- **Trace:** chạy lại đúng cấu hình `qwen3-ram-cap-v1` ở trần 10240MiB (CPU4, cache 64MiB/layer, dense BF16, mixed × 8, keep-cache, max64, victim, lookahead off), thêm `MOE_ROUTE_TRACE`. Routing không phụ thuộc trạng thái cache, nên một trace dùng được cho mọi kích thước pool. Lượt chạy ([run/](run/), [console.log](console.log)) không xóa page cache trước khi chạy và có ghi trace, nên **thời gian của nó không dùng để so sánh**.
- **Nguồn:** engine ở HEAD `1b35cdf`; `git_dirty` true chỉ vì docs/tools chưa commit, `src/` không đổi. Text và số cache miss 8/8 turn giống hệt hồ sơ P1 [qwen3-ram-cap-i5](../2026-11-17-qwen3-ram-cap-i5/README.md) (trần 10240).
- **Công cụ:** [`tools/sim_vram_expert_pool.py`](../../../tools/sim_vram_expert_pool.py) tái hiện chính sách cache RAM của engine và ba chính sách pool:
  - `a1_first`: đúng hành vi A1 hiện tại (inclusive, `fill_only`): pool nhận N expert đầu tiên được đọc, không bao giờ evict;
  - `a1_quota`: như trên nhưng chia đều slot cho 48 layer;
  - `oracle_static`: N cặp (layer, expert) được gọi nhiều nhất trong chính trace, nạp sẵn. Nhìn trước tương lai, chỉ là **cận trên**.
  - Thêm phép held-out `learned_static`: chọn top-N từ turn 0–1 (Mars, follow-up), phát lại trên turn 2–3 (Python dictionary, love) từ cache rỗng.
- **Kiểm chứng công cụ:** với pool 0, mô phỏng khớp **tuyệt đối** `expert_cache_hits/misses/evictions` của engine ở 8/8 turn (cũng khớp 3/3 turn OLMoE ngày 23/11). Phần pool chưa được đối chiếu với run A1 thật; nó dựa trên đọc code (`src/gpu.rs` `upload_and_cache`, `src/olmoe.rs` nhánh A1).
- Lệnh: `python3 tools/sim_vram_expert_pool.py <(gzip -dc qwen3-route-trace.log.gz) --experts 128 --expert-bytes 4732928 --cache-mib 64 --validate run --slots 256 512 700 1024 --heldout 0-1 2-3 --output sim.json`. Kết quả: [sim.json](sim.json).

## Kết quả

Số lần đọc expert từ SSD so với không có pool (baseline 128.467 lần cho 8 turn):

| Slot | `a1_first` | `a1_quota` | `oracle_static` (cận trên) | Layer có slot (`a1_first`) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0,926 | 0,929 | 0,847 | 5/48 |
| 512 | 0,861 | 0,856 | 0,706 | 10/48 |
| 700 | **0,813** | 0,803 | 0,613 | 14/48 |
| 1024 | 0,734 | 0,706 | 0,475 | 21/48 |

Held-out (học từ turn 0–1, phát lại turn 2–3, baseline 30.563 lần):

| Slot | `learned_static` | `oracle_static` | `a1_first` | `a1_quota` |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0,909 | 0,836 | 0,935 | 0,936 |
| 512 | 0,784 | 0,689 | 0,876 | 0,869 |
| 700 | **0,692** | 0,593 | 0,835 | 0,820 |
| 1024 | 0,542 | 0,451 | 0,764 | 0,738 |

700 slot là ước tính cho GTX 1650 (VRAM trống ~3,9GB, trừ 512MiB dự trữ của engine, chia 4.732.928 byte/expert); số thật phụ thuộc VRAM trống lúc khởi động.

## Diễn giải

- **Chính sách A1 hiện tại không đạt ngưỡng:** ở 700 slot chỉ giảm 19% lần đọc (17% trên held-out). Chia đều theo layer gần như không giúp: vấn đề là chọn sai expert, không phải dồn sai layer.
- **Chọn expert theo tần suất thì đạt:** học từ prompt khác vẫn giảm 31% ở 700 slot; cận trên là 39–41%. Engine đã có `MOE_USAGE_PATH` lưu tần suất dùng expert, có thể dùng làm nguồn chọn.
- **Suy ra, chưa đo:** nếu thời gian I/O tỉ lệ với số lần đọc, với tỉ lệ I/O 0,63–0,76 của hồ sơ P1, giảm 31% lần đọc tương ứng nhanh hơn tối đa khoảng 1,24–1,31 lần toàn request. Con số thực sẽ thấp hơn: chưa tính chi phí chép hidden state qua PCIe mỗi layer, RAM của CUDA context làm page cache nhỏ đi, và page cache vốn đã phục vụ một phần lần đọc.

## Giới hạn

- Workload chỉ 4 prompt tiếng Anh ngắn; held-out học từ 2 turn, thử trên 2 turn. Chưa biết placement học được có tổng quát cho workload khác.
- Đếm lần nạp, không đo thời gian; không mô hình hóa GPU compute, PCIe, page cache.
- Phần pool của mô phỏng chưa đối chiếu với run GPU thật.
