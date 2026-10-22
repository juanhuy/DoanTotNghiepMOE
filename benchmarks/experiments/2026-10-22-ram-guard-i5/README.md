# Kiểm RAM i5 sau guard hợp tác — 2026-10-22

Các raw mới là **dirty exploratory stability tests**, không baseline tốc độ, không chứng minh hết leak/OOM hoặc đạt nghiệm thu giai đoạn1. Chỉ một model7B tại một thời điểm; server người dùng đã dừng trước thử, không build/tests/benchmark đồng thời. Không thay corpus40 raw lịch sử.

## Điều kiện và nguồn

- i5 do người dùng xác nhận, GTX1650 Max-Q enabled; threads8/context512/victim. Hai lượt hoàn tất dùng cache64MiB/layer, RSS guard5GiB, system headroom1GiB. Default server sau sửa cũng cache64, nhưng default RSS guard có thể là6GiB; không đồng nhất default với config thử5GiB.
- [Lượt32-token xen kẽ reset](reset-32/result.json), [lượt128-token giữ cache](keep-128/result.json), [kiểm âm tính256MiB/layer](preflight-256/result.json). Mỗi folder giữ source ZIP, source/binary/checkpoint hashes, config, server.log, memory.jsonl và từng input/output/status. Source engine/binary hai lượt hoàn tất khớp; Python runner lượt sau bổ sung keep-cache, được lưu hash riêng.
- Runner lấy mẫu /proc250ms; watchdog dừng child riêng khi RSS>5,5GiB, MemAvailable<1,5GiB, process VmSwap>128MiB hoặc thiếu telemetry. Host/cgroup dùng cho guard vẫn có giới hạn đã công bố trong ARCHITECTURE. Hash weights trước load làm ấm OS cache. GPU enabled không thay thế profiler; targeted CUDA fixture đã chạy riêng trước đo.
- Lượt32 reset trước các phase0/3, phase1 giữ history Mars; lượt128 chỉ reset đầu rồi giữ expert cache, mỗi nhóm4 câu có một follow-up cùng session, câu khác session riêng. Đây là12 request/lượt, không12 session dài hoặc24 lần cold load.

## Tóm tắt quan sát

- **reset-32**: status completed, 12 request hoàn tất, thời gian quan sát 212.1s/848 samples; RSS sample max 3.4686GiB; observed HWM max 3.4686GiB; process swap max 0KiB; MemAvailable min 6.471GiB; watchdog violations 0.
- **preflight-256**: status failed, 0 request hoàn tất, thời gian quan sát 0.0s/1 samples; RSS sample max 0.0019GiB; observed HWM max 0.0019GiB; process swap max 0KiB; MemAvailable min 10.361GiB; watchdog violations 0.
- **keep-128**: status completed, 12 request hoàn tất, thời gian quan sát 488.7s/1952 samples; RSS sample max 3.5256GiB; observed HWM max 3.5256GiB; process swap max 0KiB; MemAvailable min 6.520GiB; watchdog violations 0.

Status failed ở preflight-256 là **kết quả âm tính mong đợi**, giữ nguyên raw. Server.log báo guard từ chối reserve7.288.299.520B trước đọc weights lớn với RSS lúc kiểm8.208.384B. Sampling250ms không bắt trúng peak startup rất ngắn, nên RSS sample max không dùng để khẳng định peak thực của kiểm này. Lệnh server không watchdog trước đó bị automatic approval review từ chối vì nguy cơ OOM nếu preflight sai; phép thử thực được thực hiện qua runner có watchdog, không bỏ qua kiểm soát.

## Từng request hoàn tất

| Run | Turn | Completion tokens | RSS sau request GiB | VmSwap KiB |
| --- | --- | --- | --- | --- |
| reset-32 | 0 | 32 | 3.2150 | 0 |
| reset-32 | 1 | 32 | 3.2400 | 0 |
| reset-32 | 2 | 32 | 3.3021 | 0 |
| reset-32 | 3 | 32 | 3.3607 | 0 |
| reset-32 | 4 | 32 | 3.3607 | 0 |
| reset-32 | 5 | 32 | 3.3607 | 0 |
| reset-32 | 6 | 32 | 3.4049 | 0 |
| reset-32 | 7 | 32 | 3.4174 | 0 |
| reset-32 | 8 | 32 | 3.4174 | 0 |
| reset-32 | 9 | 32 | 3.4174 | 0 |
| reset-32 | 10 | 32 | 3.4569 | 0 |
| reset-32 | 11 | 32 | 3.4452 | 0 |
| keep-128 | 0 | 128 | 3.2349 | 0 |
| keep-128 | 1 | 91 | 3.3028 | 0 |
| keep-128 | 2 | 128 | 3.3796 | 0 |
| keep-128 | 3 | 128 | 3.4048 | 0 |
| keep-128 | 4 | 128 | 3.4414 | 0 |
| keep-128 | 5 | 91 | 3.4573 | 0 |
| keep-128 | 6 | 128 | 3.4749 | 0 |
| keep-128 | 7 | 128 | 3.4903 | 0 |
| keep-128 | 8 | 128 | 3.5077 | 0 |
| keep-128 | 9 | 91 | 3.5102 | 0 |
| keep-128 | 10 | 128 | 3.5217 | 0 |
| keep-128 | 11 | 128 | 3.5256 | 0 |

## Diễn giải và giới hạn

24 requests hoàn tất, không watchdog stop/process swap trong hai lượt. RSS vẫn tăng dần một phần trong lượt giữ cache; số mẫu/thời gian hữu hạn không chứng minh plateau tuyệt đối, allocator retention hay không leak. Phân tích RSS còn phải xét dense host, expert capacity, KV capacity, tokenizer, driver và allocator. Không cộng các vùng thành RSS chính xác nếu chưa profiler.

Không so theo phần trăm với OOM9,93GiB cũ: cache/source/history/điều kiện của lượt lỗi chưa đầy đủ, hai phép thử mới đã ép64MiB và guard5GiB. Chưa xác định duy nhất nguyên nhân tăng RAM cũ. Guard hợp tác có thể overshoot giữa sample/boundaries, không hard RSS cap, không thay thế OS/cgroup isolation. System headroom có thể đổi vì ứng dụng khác. Không có i7 run mới, không so tốc độ hay chất lượng.

Checks trước thử: Rust debug/release76 pass/profile,5 ignored/profile, fmt/Clippy; Python39 +Node7/catalog40; HTTPfixture30 vòng20 complete/10 cancel; CUDA probe available và victim cache test in EXECUTED với logits2e-5. Các kiểm này khác smoke RAM model thật. Xem [PROGRESS](../../../docs/PROGRESS.md), [PLAN](../../../docs/PLAN.md) và [WORKFLOWS](../../../docs/WORKFLOWS.md).

Kiểm toàn bộ payload bất biến: `sha256sum -c CHECKSUMS.sha256` trong thư mục này. Raw copy từ runs đã kiểm byte SHA khớp; không xóa payload gốc ignored hoặc đổi manifest lịch sử.
