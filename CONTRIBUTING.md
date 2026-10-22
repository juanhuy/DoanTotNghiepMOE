# Cộng tác trên hai máy

Chạy lệnh từ root repo. Hai người dùng cùng commit, `Cargo.lock`, suite/config khi đối chiếu. Nhãn `i5` là máy Sơn, `i7` là máy đồng nghiệp; xem [profiles](benchmarks/machines.json).

## Setup tối thiểu

1. Clone repo, tạo branch cho thay đổi. Cài Rust stable (rustfmt/Clippy), Python 3.11+. Không cần Node/CUDA/checkpoint để chạy fixture tests Rust.
2. Chạy `python3 tools/check.py --rust`. Đây cũng là lệnh CI. `Cargo.toml` khai báo MSRV 1.85; stable CI không chứng minh MSRV nếu chưa chạy riêng toolchain đó.
3. Đọc [AGENTS.md](AGENTS.md), [workflow](docs/WORKFLOWS.md) và [setup coding agent](docs/AGENT_SETUP.md). Không cần cài nhiều agent framework.
4. Chuẩn bị checkpoint chỉ khi cần demo/benchmark thật. Model không commit; có thể dùng thư mục local/symlink, runner hash nội dung thực.

## Python converter/reference theo tác vụ

Core tooling chỉ dùng stdlib. Converter cần `torch`, `numpy`, `safetensors`; `huggingface_hub` chỉ cho tải nguồn HF. Tạo virtualenv Python 3.11/3.12, CI converter chọn direct dependencies trong `tools/requirements-converter.txt` trên Python 3.12 (CPU wheel); chưa phải lock toàn bộ dependency bắc cầu. Máy khác cài PyTorch phù hợp hardware theo hướng dẫn chính thức; lưu `python -m pip freeze` khi tạo bằng chứng model. Fixture gốc dùng Transformers **4.51.3**, không tự tái tạo bằng bản khác rồi gọi cùng baseline.

```bash
python3 -m venv .venv
# CPU-only converter fixture, cùng direct dependencies của CI:
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r tools/requirements-converter.txt
.venv/bin/python tools/test_prepare_olmoe.py
```

Windows dùng `.venv\Scripts\python.exe`. CUDA hiện dynamic-load `libcuda.so`/`libcublas.so` qua Unix; không coi Windows GPU được hỗ trợ chỉ vì Rust compile. Linux là môi trường kiểm chứng chính. Nếu máy i7 dùng Windows, dùng cùng checkout/toolchain trong WSL2 Linux cho engine và các checks; native Windows chưa nằm trong phạm vi nghiệm thu. Coding agent cần mở đúng workspace Linux đó.

## Chuẩn bị checkpoint thật (chỉ khi cần demo/benchmark)

Nếu đã có merged INT8 checkpoint hợp lệ, sao chép thư mục hoặc tạo symlink local; không convert lại một checkpoint đã merged. Nếu có weights HF gốc ở máy:

```bash
.venv/bin/python tools/prepare_olmoe.py --source /path/to/hf-original --out models/olmoe-1b-7b-int8
```

Nếu cần tải nguồn, cài thêm `huggingface_hub`. Ví dụ [nguồn Instruct của Ai2](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct); chọn một **commit SHA nguồn cố định** trên trang Files/History, thống nhất cùng SHA cho hai máy. Ví dụ này không xác nhận checkpoint lịch sử trong repo là cùng biến thể:

```bash
.venv/bin/python -m pip install huggingface_hub
# Đặt MODEL_SOURCE_REVISION thành commit SHA đã chọn, không dùng main cho thí nghiệm đối chiếu.
: "${MODEL_SOURCE_REVISION:?Set a pinned Hugging Face source commit SHA first}"
.venv/bin/python tools/prepare_olmoe.py --repo allenai/OLMoE-1B-7B-0125-Instruct --revision "$MODEL_SOURCE_REVISION" --out models/olmoe-1b-7b-int8
```

Thư mục đích phải chưa tồn tại. Converter giữ `INCOMPLETE` khi chưa hoàn tất; không xóa marker để ép loader chạy. Cần dung lượng cho cả nguồn, cache tải HF và bản converted; không tự dọn checkpoint nguồn của đồng nghiệp. Ghi repo/revision nguồn, commit converter và pip freeze vào bằng chứng chuẩn bị model; runner sau đó hash checkpoint thực dùng.

```bash
cargo run --release --locked --example inspect_model -- models/olmoe-1b-7b-int8
```

Inspect chỉ kiểm index/một expert với input tổng hợp; tiếp tục kiểm chat/template qua demo, không coi inspect là nghiệm thu model đầy đủ. Các lệnh tải/convert trên chưa được chạy trong lần chuẩn hóa repo.

## Kiểm tra và PR

```bash
python3 tools/check.py --rust --release
python3 tools/check.py --http
```

`--http` build server, chạy fixture localhost và cần quyền socket. Converter/CUDA/quality model thật kiểm tra riêng khi tác vụ liên quan; bỏ qua không phải pass.

Mỗi thay đổi có một owner. PR dùng template `.github/`; cập nhật PROGRESS trước khi báo hoàn thành. Không chạy benchmark song song trên cùng máy. Dữ liệu mới qua [runner](benchmarks/README.md), không redirect vào tên cũ.

Không có auto-push/merge/deploy hoặc hook tự chạy shell khi clone. Tài khoản/token/model coding agent/quyền hệ thống là lựa chọn cá nhân, không bị repo ghi đè.
