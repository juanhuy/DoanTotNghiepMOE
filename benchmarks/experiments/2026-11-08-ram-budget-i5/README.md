# Kiểm ngân sách RAM i5, 08/11/2026

## Câu hỏi và protocol

Theo [thiết kế chốt trước tải](../../../docs/RAM_BUDGET.md), kiểm một process/12request mixed (Mars, follow-up, Python mới, Love mới ×3), không reset giữa nhóm; cache64MiB/layer, context512, max64, CPU trước load/4threads/victim, lookahead off/KVreuse off/memorytrace on. Đây là kiểm cửa sổ vận hành, không đối chứng speedup, không chọn cache tối ưu hoặc nghiệm thu dài hạn.

Lệnh preview rồi thực thi (bỏ --dry-run):

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse off --workload mixed --cache 64 \
  --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

[Preflight](preflight.json) có3mẫu cách5s: available khoảng10,8GiB, swap0/pswpin-out0, PSIavg10=0, không có engine/build liên quan. Source44hash và binary khớp bản candidate đã kiểm95Rust/profile và runner đã kiểm61Python/7Node; không build/tests lại hoặc chạy song song. Được cấp localhost rồi chạy đúng một lượt, không nới guard/đổi OS.

## Provenance

[Record](run/result.json) run20261108T083112.269185Z-ram-ab563b56eb0b47d6b7c78b84aaf5573c, HEAD75d5a62/source dirty exploratory. SourceZIP/44hash và binarySHA256c45617f4f626f94af219c1d40ea86d6774735d09d3739bd9e3aa1e97aff4c313 khớp nguồn/binary sau đo và hồ sơ KV05/11; checkpoint files/hash cũng khớp. Không lấy thời gian khác ngày làm speedup.

Hardware i5-10500H/12logicalCPU/RAM16571326464B. GPU GTX1650 nằm trong inventory, nhưng CPUmode0VRAM/probe available khônglibCUDA, không kiểm GPUcompute. Cấu hình guard/watchdog ghi đủ trong record.

## Kết quả từng request

COMPLETED12/12, không watchdogviolation/guardOOM. Mỗi request64completion/finishlength. Follow-up có107prompt/11cachedprompt; client gửi lịch sử thực trong turnJSON. Đây là smoke và kiểm giao thức runner, không parity với model tham chiếu hoặc đánh giá chất lượng chuẩn hóa.

| Request | Prompt tokens | Cached prompt | RSS sau request (MiB) | VmSwap (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 19 | 0 | 2898.262 | 0.000 |
| 2 | 107 | 11 | 2926.105 | 0.000 |
| 3 | 22 | 0 | 2988.812 | 0.000 |
| 4 | 20 | 0 | 3030.082 | 0.000 |
| 5 | 19 | 0 | 3073.609 | 0.000 |
| 6 | 107 | 11 | 3061.613 | 0.000 |
| 7 | 22 | 0 | 3077.730 | 0.000 |
| 8 | 20 | 0 | 3117.770 | 0.000 |
| 9 | 19 | 0 | 3153.113 | 0.000 |
| 10 | 107 | 11 | 3158.426 | 0.000 |
| 11 | 22 | 0 | 3161.094 | 0.000 |
| 12 | 20 | 0 | 3169.430 | 0.000 |

797samplerows/mốc cuối201,839s; sampledRSSmax3169,426MiB, HWMmax từ sampler+turn3169,430MiB. ProcessVmSwap0 trong mọi mẫu, MemAvailablemin7,905GiB. Sampler và turn khác thời điểm; không ghép các peak khác mốc thành một sample. Chênh RSS request1→12 **271,168MiB**; cuối nhóm3030,082→3117,770→3169,430MiB (tăng87,688 rồi51,660MiB). RSS có giảm ở request6, nhưng cuối nhóm vẫn tăng: chưa chứng minh plateau.

Trace7stage/request/84stage, trùng last_memory_trace; tổngtracked không đếm retainedKV hai lần. Cache cuối mọirequest962,5MiB/recycled0, retainedsession128,001–128,003MiB; trackedhost cuối khoảng2909,010MiB gần cố định. Cache+recycled từnglayer và sessionadmission đúngbudget. Vì accounting chưa bao phủ mọi vùng và capacity khác resident pages, không suy RSS trừtracked thành leak hoặc quy phần tăng cho tokenizer/allocator/KV riêng lẻ.

Globalpswpin/out không tăng trong sampler; PSI some.total tăng75769microsecond. Cgrouppartial do7file rootlevel5 thiếu; memory.events/events.local đọc được không tăng high/max/oom/oom_kill. Counter/cgroup có thể gồm process khác, không cộng ancestor với child và không coi thiếufile là0. Các quan sát này không loại trừ nguyên nhân sự cố swap trước đây.

## Kết luận và việc tiếp theo

**Hoàn tất cửa sổ kiểm trong ngưỡng, nhưng bộ nhớ còn tăng.** Cache64MiB/layer vận hành được cho đúng workload/cửa sổ này; chưa xác nhận mức cache chung, ổn định dài hạn, hếtOOM/leak hoặc i7. Kết quả không chứng minh mọi phần RAM đã nằm dưới một hardcap.

Chưa có cơ sở nâng cache để giảm I/O. Bước tiếp là phân tích capacity/RSS theo từng stage từ raw này để khoanh vùng mức tăng sau khi cache ổn định; nếu vẫn chưa phân biệt allocation còn sống với allocator/trang giữ lại thì thiết kế profiling sâu hơn, không chạy lại12request ngay hoặc thêm bản vá thu hồi trang theo phỏng đoán.

## Bảo toàn payload

Chuyển nguyên17file từ runs vào run/, thêm preflight riêng và khóa18file bằng CHECKSUMS.sha256; một payload duy nhất, đường dẫn gốc trong record/log giữ nguyên. README là diễn giải ngoài payload khóa. Không sửa raw/checksum cũ.

```bash
sha256sum -c CHECKSUMS.sha256
```

## Phân tích hồi cứu theo stage (08/11, không chạy thêm model)

Mỗi ô là RSSstage_sau − RSSstage_trước, đơn vịMiB; số âm là RSS giảm giữa hai mốc, không phải số byte free đo trực tiếp. Request1 loại khỏi bảng để không gộp quá trình nạp đầy cache đầu tiên vào các request sau. Tên cột thể hiện khoảng thời gian, không quy byte tăng cho duy nhất thao tác mang tên cột.

| Request | before_request→after_kv_lookup | prefill_buffers_live→after_prefill | after_prefill→before_session_admission | admission→after_generate_inner |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.012 | 16.492 | 11.316 | 0.000 |
| 3 | 2.078 | 51.270 | 9.355 | 0.000 |
| 4 | 0.004 | 34.164 | 7.094 | 0.000 |
| 5 | 1.508 | 41.988 | 0.004 | 0.000 |
| 6 | 0.000 | -16.457 | 4.445 | 0.000 |
| 7 | 0.000 | 28.012 | -11.895 | 0.000 |
| 8 | 0.000 | 40.012 | 0.004 | 0.000 |
| 9 | 0.000 | 35.344 | 0.000 | 0.000 |
| 10 | 0.000 | 1.375 | 3.938 | 0.000 |
| 11 | 0.000 | 2.668 | 0.000 | 0.000 |
| 12 | 0.000 | 8.328 | 0.000 | 0.000 |

Tổng các chênh lệch request2–12: khoảnglookup **+3,602MiB**, after_kv_lookup→prefill_buffers_live **0MiB**, khoảngprefill **+243,195MiB**, khoảngsauprefill tới admission **+24,262MiB**, admission/thoátinner **0MiB**. Các khoảng giữa request từafter_generate_inner trước tới before_request sau cộng **+0,164MiB**. Đây là tổng delta có cả âm, không tổng byte cấp phát hoặc tổng peak.

Traceafter_generate_inner request1→12 tăng271,223MiB. SnapshotHTTP cuối request tăng271,168MiB như bảng phía trên; khác endpoint/thời điểm đọc nên chênh0,055MiB, không sửa raw để ép bằng nhau. Kiểm số nguyên byte: tổng từngdelta cộng đúng chênh hai endpointtrace.

Snapshotrequest1→12: RssAnon+271,168MiB, RssFile+0, RssShmem+0, VmSwap+0; VmData+194,012MiB/VmSize+192MiB. Các trường /proc khác định nghĩa, không cộng hoặc lấy chênh giữa chúng làm số leak. Không có bằng chứng phần RSS tăng này là tăng ánh xạ file trong hai endpoint; cũng không suy file I/O không ảnh hưởng allocator/pagecache hệ thống.

Capacity sau mỗi request gần cố định; các request mở session mới có lại overlapactiveKV+retainedKV128MiB mỗi bộ ở đườngreuseoff. Tại admission, trackedcapacity giảm khoảng128MiB nhưng RSS tại mốc không giảm; điều này phù hợp drop buffer không đồng nghĩa OS lấy lại resident pages ngay. Không phân biệt được allocator giữ trang, buffer live ngoàiaccounting hoặc trang capacity cũ mới được chạm chỉ bằng các counter này.

Ràsource: `prefill_buffers_live` ghi sau khi các vectorprefill chính đã tạo; `after_prefill` ghi sau khi hàmprefill trở về, vectorlocal đã ra scope. Khoảng này gồm cả forward/layer/expertload/tạm/scratch/KVwrite và giải phónglocal, không thuần một loại cấp phát. Khoảngsauprefill gồm decode và hoàn tấtoutput, không timerdecode thuần. Request6prefilldelta−16,457MiB vàrequest7khoảngsauprefill−11,895MiB chứng tỏ RSS có thể giảm, chưa chứng minh leak hoặc khôngleak.

### Quyết định kỹ thuật từ phân tích

Đã đủ căn cứ ưu tiên **vòng đời cấp phát ở prefill**, thay vì nới cache hoặc tăng số request để tìm plateau. Chưa đủ căn cứ chọn tokenizerclone/KV/contributions/loader làm nguyênnhân duy nhất hoặc thêm malloc_trim theo phỏng đoán.

Câu hỏi profiling tiếp: *sauprefill, heap còn sống tăng ở callstack nào, hay sốlive gần cố định nhưng allocator giữ thêm trang?* Muốn trả lời cần profiler allocation/free theo callstack và sốlive tại boundary (ví dụ heap profiler tương thích allocator/binary thực tế), cộng RSS/anonymous mapping theo mốc; /proc smaps_rollup một mình không chứng minh liveheap. Profiler phải công bố overhead/coverage, runinstrumented là protocolchẩnđoán khác, không so tốc độ với runnày. Trước hết kiểm khả năng profiler trên fixture; chỉ chọn một workload ngắn thực sự kích hoạt phần tăng sau đó. Chưa cài/chạyprofiler hoặc sửaengine trong bước phân tích này.

### Rà cấp phát cụ thể trong source (08/11)

Các điểm dưới là kết quả rà mã, không phải profiler đã quy được271MiB về một callstack:

| Vùng | Cơ chế / giới hạn | Mức ưu tiên kiểm chứng |
| --- | --- | --- |
| Vectorprefill chính | batch/hidden/QKV/norm/context/projected/contributions/expert_uses là local của hàm prefill, capacityđã chọn được trace ởprefill_buffers_live. Ra scope sauhàm, không được lưu vào session. | Có churn mỗi request; cần xem allocation/free thực, không suy leak từvec. |
| Prefetch expert | `prefetch_missing_experts` chọn tối đaTopK8 miss; nếuít2miss thì trảempty. Poprecycled hoặc cấp cặpweights/scales mới rồi load bằngRayon vàgiữ trongHashMappreloaded tạm. Tối đa8payload≈48,125MiB/lần gọi, không phải tổngallocation suốtrequest. | Ưu tiên vì có nhiều lần load/layer và allocation lớn/đa luồng. |
| Demand expert load | Nếu khônghit/preloaded, poprecycled hoặc cấp cặpweights/scales mới rồi load; `load_reusing` chỉ tăngcapacity nếu bufferthiếu. | Cần callstack phân biệt prefetch/demand/resize với free/admission. |
| Recycledpool | Cache+recycled cùngbudget; cache 10 expert=60,15625MiB/layer, slack3,84375MiB < một expert6,015625MiB. Khi cache ởmức này, poolkhônggiữđược thêm một cặp payload chuẩn đầyđủ. | Giải thích source vì sao recycled0 vẫn có thể đi kèm nhiều lần cấp/free; không phải bugbudget hoặc bằng chứng leak. |
| KV | Buffer128MiB cấp trước prefill ởsessionmới, trang có thể mới resident khi đượcghi trongprefill/decode. Reuseoff có overlapđãbiết. | Không loại trừKV chỉ vì deltaởlookupnhỏ; profiling phải phânbiệt mapping/trangchạm với allocation live. |

Phần 48,125MiB là payload prefetch đang giữ cho một layer, không nhân16layer làmpeak, không cộngvàoRSSgiống sốđãđo. Churnlớn có thể liênquan allocatorgiữtrang/phânmảnh, nhưng chưa chứngminh; không tựnới cache để chứa pool vì tăngbudget làm thayđổi câu hỏi và tổng RAM.

Môi trườngi5 lúc rà: cógdb, chưa tìm thấyheaptrack/heaptrack_print/valgrind/perf trong PATH. Chưa cài package hoặc chạy công cụ. Repo không khai báo custom global allocator trong source đã rà; cần xác nhận allocator/binary/OS thực khi chọnprofiler, chưa chốt công cụ theo giảđịnh. Bước cóthể bàn giao choi7 ngay là kiểm công cụ/fixture vàbáo kết quả môi trường, chưa tự chạy thêm model7B. [Handoff](../../../docs/WORKFLOWS.md#bàn-giao-điều-tra-prefillram-08112026).

### Đối chiếu phân tích độc lập do người dùng cung cấp

Tính lại từ raw: request1 VmData3015,566MiB/RssAnon2889,258MiB; request3 3155,574/2979,809; request9 3209,578/3144,109; request12 3209,578/3160,426. Request2→3 VmData+128,008MiB/VmSize+128MiB; toànrequest1→12 VmData+194,012MiB/RssAnon+271,168MiB. Những số này củng cố giả thuyết vùng ánh xạ đã cấp trở thành resident và allocator giữ bộ nhớ, nhưng không tự phân rã được heap live/free.

Cần giữ các giới hạn khi trích phân tích:
- VmData là kích thước phân đoạn dữ liệu riêng theo kernel, RssAnon là resident anonymous memory; hiệu hai counter khác phạm vi không phải một phép đo chính xác “heap chưa chạm”. Phần77,156MiB chênhdelta là đẳng thức số học, chưa chứng minh đó đều là trang cũ mới chạm. Tương tự, bước128MiB trùng thời điểm KV là liên hệ mạnh, chưa định danh mapping/chunk bằng smaps hoặc allocationtrace.
- VmSize thay đổi64MiB phù hợp giả thuyết arenaheap, chưa chứng minh arena cụ thể hoặc thread nào tạo nó. gettid ở boundary chỉ nhận diện worker gọi trace, không đo thread Rayon phân bổ/free mọi buffer.
- Môi trường lúc đối chiếu báo glibc2.44; binaryELF dynamically linked. Raw cũ chưa có bản ghi phiên bản libc/interposition đủ để mặc định runtime lúc đo giống hệt môi trường hiện tại.
- mallinfo2 có thể cung cấp arena/uordblks/fordblks/hblkhd hữu ích. Disassembly libc hiện tại có vòng lặp arena; nguồn glibc master cũng có vòng đó. Manual có cảnh báo coverage khác, vì vậy phải xác nhận trên libc cụ thể, không suy từ tên API hoặc một trang hướng dẫn chung.
- uordblks+hblkhd là allocator-accounted in-use/mmap indicator, không phải tổng requestedbytes của mọi đối tượng Rust còn live: bookkeeping, tcache và mapping ngoài malloc có giới hạn. Các arena được đọc tuần tự, không phải snapshot nguyên tử toàn process. Không gán coverage toànheap hoặc overhead<1ms khi chưa đo.
- Nếu chỉ báo in-use gần cố định trong khi RSS tăng và allocatorfree tăng, bằng chứng giữ trang mạnh hơn, nhưng chưa loại trừ leak nhỏ hay mapping ngoài malloc. Nếu in-use tăng thì cần callsite, không tự gọi mọi tăng đó là leak. Các ngưỡng12MiB/150MiB/50% là tiêu chí chẩn đoán đề xuất, không định luật hay hai kết quả bao phủ mọi trường hợp; phải có nhánh chưa phân biệt được. Không bảo đảm request3 ở lượt mới có đúng bước128MiB như lượt cũ.

Bước tiếp được ưu tiên trong PLAN: thiết kế trace allocator opt-in trên Linux/glibc và threadID, kiểm ABI/availability/coverage trên fixture trước. Chưa thêm FFI, chạy fixture hoặc lặp model trong lần đối chiếu này. Nguồn ngữ nghĩa: [kernel proc](https://docs.kernel.org/filesystems/proc.html), [mallinfo manual](https://man7.org/linux/man-pages/man3/mallinfo.3.html), [glibc source master](https://raw.githubusercontent.com/bminor/glibc/master/malloc/malloc.c). Nguồn master phục vụ rà thuật toán, không thay đúng source/build libc2.44 của máy.
