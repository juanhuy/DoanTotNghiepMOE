# Danh mục bằng chứng benchmark

Tạo bằng `python3 tools/export_benchmark_evidence.py`; kiểm tra bằng `--check`.

Có **40 artifact thô duy nhất** trong `benchmarks/manifest.json`. SHA-256 bảo vệ tính toàn vẹn sau khi lập danh mục, không xác minh nguồn gốc phép đo.

`historical_unverified`: dữ liệu lịch sử được giữ nguyên; chưa tái chạy hoặc xác minh đầy đủ provenance. `i5`/`i7` là phân loại lịch sử theo danh mục, không phải chứng nhận phần cứng. `unknown` là chưa đủ bằng chứng gán máy.

Không có benchmark mới trong bước chuẩn hóa. Không suy ra tăng tốc hoặc so sánh hai máy từ danh mục này. Byte expert là payload reader, không phải số byte SSD vật lý; engine-cold không đồng nghĩa OS page cache cold.

Các artifact lịch sử chưa có bộ metadata provenance chuẩn: commit và dirty state tại lúc chạy, lệnh đầy đủ, hash checkpoint, nhận diện phần cứng/driver thực tế, trạng thái OS cache và điều kiện tải nền. Giá trị thiếu là **chưa ghi nhận**, không thay bằng 0. Cần phép đo mới theo cùng giao thức trước khi so sánh i5/i7.

| Artifact thô | Loại | Máy (phân loại lịch sử) | Trạng thái | SHA-256 |
| --- | --- | --- | --- | --- |
| [2026-10-20-step22-fused-swiglu.json](../benchmarks/results/2026-10-20-step22-fused-swiglu.json) | generation | unknown | historical_unverified | `fc2144faf3d3708fa53d45467739b546ce9ad2be5b92080c2081f56f63670a94` |
| [2026-10-20-step23-inter-layer-lookahead.json](../benchmarks/results/2026-10-20-step23-inter-layer-lookahead.json) | generation | unknown | historical_unverified | `a87a750f7663647bbcf6d926f04b7e3495bef6a1098038527f4b74c123909f34` |
| [2026-10-20-step24-256mb.json](../benchmarks/results/2026-10-20-step24-256mb.json) | generation | unknown | historical_unverified | `6d836bea09de9446bb5575fe1a7baa7d32af647b5b696a19ffeb6025979808b5` |
| [2026-10-20-step24-coarse-grained-multi-expert.json](../benchmarks/results/2026-10-20-step24-coarse-grained-multi-expert.json) | generation | unknown | historical_unverified | `d401bb6fccdabe7c57d8c3054c712f724e3e13844eb026a8bfe62671d8abc023` |
| [2026-10-21-step25-256mb.json](../benchmarks/results/2026-10-21-step25-256mb.json) | generation | unknown | historical_unverified | `6c95865ab9a77c147f53cdc74df33c4fa55f1d00434d4b15ed3f20a5de398f21` |
| [2026-10-21-step25-zero-alloc-io.json](../benchmarks/results/2026-10-21-step25-zero-alloc-io.json) | generation | unknown | historical_unverified | `e6bfde39979f21eeca92a01cd694c035367781b10b7d74b39dbf04a12e5eb9da` |
| [2026-09-22-baseline.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-22-baseline.json) | generation | i7 | historical_unverified | `ba141800da27b4f1a135b799e49e9b73a56b008d5ab3701d60f6dbe4354a7724` |
| [2026-09-22-cache-128mib.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-22-cache-128mib.json) | generation | i7 | historical_unverified | `fc57572446400ef391d8023fd4856cddccd7099b64693801a53eb35deb8dc8e8` |
| [2026-09-22-hot-cache-reused-buffers.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-22-hot-cache-reused-buffers.json) | generation | i7 | historical_unverified | `1d37d66cbd0409debbe616a6f3b28b44aed3c6058dfce4b23a5afe87f2c105d0` |
| [2026-09-22-mixed-workload-open-shards.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-22-mixed-workload-open-shards.json) | generation | i7 | historical_unverified | `b01515f15a1091d4f467b2124ebd7a96c83d5c8ab872ce4871c41c65dd27f41d` |
| [2026-09-22-skip-dense-revalidation.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-22-skip-dense-revalidation.json) | generation | i7 | historical_unverified | `e85fbbe5d33e441239cebb095577760221f5345933e217fd363793a256573712` |
| [2026-09-23-batched-prefill-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-batched-prefill-context.json) | suite | i7 | historical_unverified | `93da60767f4ea2f06faa286e381b09ebb08b0eaff3f645e38b41c9b9e53dfa4e` |
| [2026-09-23-batched-prefill-near-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-batched-prefill-near-context.json) | suite | i7 | historical_unverified | `1f0e73b708fcc9067ca2cc639622e2043eeff004d44ebe73cf83123188e1316b` |
| [2026-09-23-dense-after-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-dense-after-context.json) | suite | i7 | historical_unverified | `b770e0725197611b89351bded61da46e908ed262772ebf3bc0ddbefa09dc37bf` |
| [2026-09-23-dense-before-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-dense-before-context.json) | suite | i7 | historical_unverified | `761b0fa1a9f49e79eabd5eec0c06582899e872c4494f197451d6328991d57527` |
| [2026-09-23-dense-kernel.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-dense-kernel.json) | microbenchmark | i7 | historical_unverified | `431f4daef305e485db3dcbec3dfa4f862764bfa9242158b4d56643550f256ea7` |
| [2026-09-23-flat-kv-context-3x.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-flat-kv-context-3x.json) | suite | i7 | historical_unverified | `bb416260fea605730d5360a70a898f174547c64be98d726dc743a3e5a97454cd` |
| [2026-09-23-flat-kv-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-flat-kv-context.json) | suite | i7 | historical_unverified | `274f3908f0c6c657ded74eec5c596b1d016eae0a415109ef8cfbe8e915293900` |
| [2026-09-23-http-stress-fixture.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-http-stress-fixture.json) | http | i7 | historical_unverified | `cb8ba976dae9197f7ebab0645f773caf900a5416327119bd302a5e4ee5194c9e` |
| [2026-09-23-int8-after.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-int8-after.json) | generation | i7 | historical_unverified | `ade40338b8561e116f7c4b328b024e4cf5dd0efeeec3c0cafe32f9b28a6720d1` |
| [2026-09-23-int8-before.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-int8-before.json) | generation | i7 | historical_unverified | `0c9f71e483a795c6cb32d953678bc3a8f8963734dcaef522b250e4efa46fe0a8` |
| [2026-09-23-int8-kernel.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-int8-kernel.json) | microbenchmark | i7 | historical_unverified | `5a96eb3c2de5a30d7e73035df6f6830639004baecea3b4646be1fef001ef0208` |
| [2026-09-23-quality-v1.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-quality-v1.json) | quality | i7 | historical_unverified | `8d8ba6f79c3e3f4d0d32c13ccb4ca1dcacf3a6775d7da54b92fdc3cff1ab30ae` |
| [2026-09-23-rayon-1-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-rayon-1-context.json) | suite | i7 | historical_unverified | `67d8859acefbeed7e145d5911860ffa66f6af06821912a4b6dfb69833cf9246b` |
| [2026-09-23-rayon-24-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-rayon-24-context.json) | suite | i7 | historical_unverified | `1fd7671a7756f1d9f2db4012276d8929977a7c0b74a49c91c250de6f6cfa656b` |
| [2026-09-23-rayon-8-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-rayon-8-context.json) | suite | i7 | historical_unverified | `80ddd33afa195c3fe2894bf352d586ce8e31291cecd32839184969d75e21c0c8` |
| [2026-09-23-session-cache-http.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-session-cache-http.json) | http | i7 | historical_unverified | `463b3820786d0a91415536e12258978d9b876fbb2c13cd17107ec691d1761e0e` |
| [2026-09-23-streaming-smoke.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-streaming-smoke.json) | http | i7 | historical_unverified | `ee7a79b765b6fdd3a4a4ab6d9e100da316ecffeb84be0e7249f7d6394ec279e0` |
| [2026-09-23-suite-v1.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-suite-v1.json) | suite | i7 | historical_unverified | `7bb7beef295385bb4f2151cd9247b30512b7e6be530af975d29b1afa432cefd9` |
| [2026-09-23-suite-v2-long-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-suite-v2-long-context.json) | suite | i7 | historical_unverified | `60ac68c539bf04c810edc33bd1f4cd1578f08edc531547f09cc35601ccc936a5` |
| [2026-09-23-suite-v2-near-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-suite-v2-near-context.json) | suite | i7 | historical_unverified | `c01ea6f08ad2794d5c220d52b397974345ddce80a16956ff28ce4bc67477c4c2` |
| [2026-09-24-attention-online-vs-materialized.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-24-attention-online-vs-materialized.json) | microbenchmark | i7 | historical_unverified | `27d30dd4f365e6ae67afdd5f9cbc6085edb16d92e41469028b1e21c2de8ba843` |
| [2026-09-24-blocked-attention-near-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-24-blocked-attention-near-context.json) | suite | i7 | historical_unverified | `5c3a654b417999d3dd376e251fe14bb5c0e214afc234eea9184f352132e95a2d` |
| [2026-09-24-online-softmax-near-context-3x.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-24-online-softmax-near-context-3x.json) | suite | i7 | historical_unverified | `e728f4684a57e2b4a169ee531a43a119087f358844e022a4223dca9570994775` |
| [2026-09-24-parallel-attention-near-context-3x.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-24-parallel-attention-near-context-3x.json) | suite | i7 | historical_unverified | `2f4257924d579a060e0b430ab35419191f7150eef70bd1a7fb06c77d153989f2` |
| [2026-09-24-parallel-attention-near-context.json](../benchmarks/results/testbed_a_i7_14650hx/2026-09-24-parallel-attention-near-context.json) | suite | i7 | historical_unverified | `11ea66472c6f492ce760348520b8e6e5e535168442e2ca14e9d488d58594aa5e` |
| [2026-10-10-testbed-i5-10400h.json](../benchmarks/results/testbed_b_i5_10400h/2026-10-10-testbed-i5-10400h.json) | generation | i5 | historical_unverified | `2b47cedb13d22e0cf37f875470ce785495a23b4aa7969febae65267db9716857` |
| [2026-10-19-phase3-l1-vram-expert-cache.json](../benchmarks/results/testbed_b_i5_10400h/2026-10-19-phase3-l1-vram-expert-cache.json) | generation | i5 | historical_unverified | `f3464974468be8a951122b5434fb83b3170e97d5a6cc2220faba75ba22a7f667` |
| [2026-10-19-phase4-cold-pool-prefetch.json](../benchmarks/results/testbed_b_i5_10400h/2026-10-19-phase4-cold-pool-prefetch.json) | generation | i5 | historical_unverified | `546b36ab21b047434e7e4f84afa28e706098e945720cd782b8041e9b5596d8c9` |
| [2026-10-19-phase4-cpu-pure.json](../benchmarks/results/testbed_b_i5_10400h/2026-10-19-phase4-cpu-pure.json) | generation | i5 | historical_unverified | `f217ad6f3f3784523851e93888b57990713b134be693d0c65479abdb394752cb` |

Alias trong manifest ghi đường dẫn cũ sau khi bỏ bản sao trùng byte. Dữ liệu raw là nguồn gốc; bảng này không trích hay làm tròn số đo.
