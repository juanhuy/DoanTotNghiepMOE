# Cửa sổ RAM hữu hạn S1 trên i5, 10/11/2026

## Câu hỏi và protocol

Bước S1 của [PLAN](../../../docs/PLAN.md), protocol `ram-window-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#cửa-sổ-ram-hữu-hạn-s1-ram-window-v1-1011) (commit 2f64752) trước khi chạy. Một process mới, workload mixed v1 ×6 nhóm = 24 request (Mars, follow-up cùng phiên, Python phiên mới, Love phiên mới), max64, reset chỉ trước request1/keepcache. CPU trước load/4threads/victim/cache64MiB-layer/context512, Gate–Up paired, lookahead off, KVreuse off, batch prefill off, memorytrace on. Guard RSS5120/headroom1024MiB; watchdog RSS5632/available1536/processswap128MiB. Đây là kiểm vận hành hữu hạn, không đo tốc độ, không chọn cache chung, không so i5/i7.

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --threads 4 \
  --cache 64 --context 512 --cycles 24 --max-tokens 64 --workload mixed --keep-cache \
  --lookahead off --memory-trace on --kv-buffer-reuse off --expert-batch-prefill off \
  --gate-up-mode paired --ram-limit-mib 5120 --headroom-mib 1024 \
  --stop-rss-mib 5632 --min-available-mib 1536 --stop-swap-mib 128
```

Không dùng --allow-dirty: tree sạch. Đúng một lượt, không rerun.

## Provenance và preflight

- Run 20261110T043843.817290Z-ram-9d3521ed86f647f592c2ba161c0468e2, HEAD 2f64752, git_dirty false. Binary d12a3939a7405d483a665e8642fa8c3404176e31decb0dad6eece7cd7b90c3fc và 56 source hash trùng snapshot model 09/11 (source/tools không đổi từ bf5c905); checkpoint manifest trùng hoàn toàn record 09/11. Toolchain rustc 1.99.1 không đổi. Tái sử dụng kiểm Rust debug/release 133 tests/profile 09/11 của cùng source; tools/check.py PASS 87 Python ngày 10/11.
- [preflight-initial](preflight-initial.json): available ≥5610MiB khi nhiều ứng dụng mở; người dùng đóng bớt ứng dụng trước lượt chạy. [preflight](preflight.json) cuối 3 mẫu cách 5s: available ≥8636MiB (ngưỡng chốt ≥5120MiB), pswpout 6364 không đổi, không engine/build khác. [preview](preview.json) là dry-run.
- Hardware i5-10500H; CPU mode, GPU không dùng cho compute. Hash checkpoint làm ấm OS cache, không SSD-cold.

## Kết quả từng request

COMPLETED 24/24, watchdog_violations rỗng, không guard từ chối/OOM. Mọi request 64 completion/finish length; follow-up 107 prompt/11 cached prompt. Text là smoke, không đánh giá chất lượng.

| Req | RSS (MiB) | Req | RSS (MiB) | Req | RSS (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2887.734 | 9 | 3160.754 | 17 | 3174.641 |
| 2 | 2920.492 | 10 | 3145.992 | 18 | 3177.332 |
| 3 | 2982.562 | 11 | 3153.941 | 19 | 3177.938 |
| 4 | 3021.504 | 12 | 3159.945 | 20 | 3178.609 |
| 5 | 3065.590 | 13 | 3163.949 | 21 | 3178.609 |
| 6 | 3063.805 | 14 | 3171.965 | 22 | 3178.613 |
| 7 | 3103.926 | 15 | 3156.762 | 23 | 3184.613 |
| 8 | 3133.953 | 16 | 3174.637 | 24 | 3188.621 |

RSS là process VmRSS sau request (KiB/1024). Cuối nhóm: 3021.504 → 3133.953 → 3159.945 → 3174.637 → 3178.609 → 3188.621 MiB (tăng 112.4, 26.0, 14.7, 4.0, 10.0). Request1→24 tăng 300.887MiB.

- Gate an toàn **ĐẠT**: sampled RSS max 3188.617MiB, VmHWM max 3188.621MiB ≤5120; process VmSwap max 0 trong 1320 mẫu/335.5s; MemAvailable min 6596.8MiB.
- Chỉ báo độ phẳng: spread request17–24 = **13.980MiB** (3174.641–3188.621) ≤64MiB → "phẳng trong cửa sổ" theo phân loại đã chốt. RSS vẫn tăng nhẹ cuối chuỗi; không phải plateau dài hạn.
- Trace 7 stage/request; tracked host tại after_generate_inner cố định 2909.011MiB ở mọi request (cache 962.5MiB, retained session ~128.004MiB, recycled0). RSS−tracked từ −21.3 tới +279.6MiB. Phù hợp với bằng chứng allocator giữ vùng free ở [hồ sơ 08/11](../2026-11-08-allocator-i5/README.md), nhưng lượt này không có allocator telemetry nên không tự chứng minh nguyên nhân.

## Giới hạn: swap-out toàn hệ thống

pswpout hệ thống tăng 6364 → 241812 page giữa preflight và mẫu sampler đầu (trùng khoảng hash checkpoint 7GB, trước khi server chạy), rồi 241812 → 480777 trong run (238965 page ≈ 933.5MiB với page 4096B, chủ yếu khoảng 13–100s; hai bước nhảy lớn ở ~52s). Swap là zram, swappiness 60. Process VmSwap của engine bằng 0 trong mọi mẫu, nên swap-out thuộc process khác; tác nhân reclaim chưa xác định. PSI some.total tăng 1045145µs. Kết quả không chứng minh hệ thống "không swap", chỉ chứng minh tiến trình engine không bị swap trong cửa sổ.

## Kết luận

Vận hành được trong cấu hình/cửa sổ đã thử: 24/24 request, RSS ≤3188.6MiB, spread 8 request cuối 14MiB. Đủ gate an toàn S1 để mở S2 có giám sát. Không phải hard cap, không chứng minh hết leak/plateau dài hạn, không xác nhận mức cache chung hoặc i7, không đóng giai đoạn1. Không so tốc độ/TTFT với lượt khác.

## Bảo toàn payload

Chuyển nguyên 29 file từ benchmarks/runs/i5/ vào run/ (đường dẫn gốc trong record/log giữ nguyên), thêm preflight-initial/preflight/preview; khóa 32 file bằng CHECKSUMS.sha256. README là diễn giải ngoài payload khóa.

```bash
sha256sum -c CHECKSUMS.sha256
```
