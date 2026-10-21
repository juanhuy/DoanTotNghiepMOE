# HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG
## KHOA CÔNG NGHỆ THÔNG TIN
***

# BÁO CÁO TIẾN ĐỘ THỰC HIỆN ĐỒ ÁN TỐT NGHIỆP (LẦN 1)

### ĐỀ TÀI:
**NGHIÊN CỨU, THIẾT KẾ VÀ HIỆN THỰC HÓA HỆ THỐNG SUY LUẬN TỐI ƯU CHO MÔ HÌNH NGÔN NGỮ LỚN KIẾN TRÚC SPARSE MIXTURE-OF-EXPERTS TRÊN PHẦN CỨNG GIỚI HẠN**

* **Tên hệ thống:** `MoE-TierEngine`
* **Mã nguồn:** Ngôn ngữ Rust (Systems Programming, Rayon Parallelism, cuBLAS FFI, POSIX Direct DMA)
* **Mô hình thực nghiệm trọng tâm:** OLMoE-1B-7B-0924 (16 Layers, 64 Experts/Layer, Top-8 Active Routing, Trọng số INT8 SwiGLU)

---

## DANH MỤC CÁC KÝ HIỆU VÀ CHỮ VIẾT TẮT

| Chữ viết tắt | Cụm từ đầy đủ tiếng Anh | Ý nghĩa tiếng Việt |
| :--- | :--- | :--- |
| **LLM** | Large Language Model | Mô hình ngôn ngữ lớn |
| **MoE** | Mixture of Experts | Mô hình hỗn hợp chuyên gia |
| **FFN** | Feed-Forward Network | Mạng truyền thẳng |
| **SwiGLU** | Swish-Gated Linear Unit | Khối biến đổi tuyến tính cổng Swish |
| **TTFT** | Time to First Token | Thời gian phản hồi token đầu tiên (độ trễ Prefill) |
| **VRAM** | Video Random Access Memory | Bộ nhớ video card đồ họa |
| **RAM** | Random Access Memory | Bộ nhớ trong khả biến của máy chủ/máy trạm |
| **NVMe** | Non-Volatile Memory Express | Giao thức truyền dữ liệu ổ cứng thể rắn tốc độ cao |
| **SSD** | Solid State Drive | Ổ lưu trữ thể rắn |
| **DMA** | Direct Memory Access | Cơ chế truy xuất bộ nhớ trực tiếp không qua CPU |
| **SIMD** | Single Instruction, Multiple Data | Xử lý đơn lệnh nhiều dữ liệu trên CPU |
| **FMA** | Fused Multiply-Add | Phép toán nhân-cộng gộp trong một chu kỳ xung nhịp |
| **RoPE** | Rotary Position Embedding | Cơ chế nhúng vị trí quay vector |
| **RMSNorm** | Root Mean Square Normalization | Chuẩn hóa phương sai trung bình bình phương |
| **GQA** | Grouped-Query Attention | Cơ chế chú ý nhóm truy vấn |
| **MHA** | Multi-Head Attention | Cơ chế chú ý đa đầu truyền thống |
| **SLRU** | Segmented Least Recently Used | Thuật toán thay thế bộ đệm phân đoạn độ mới |
| **2Q** | Two Queue Cache Algorithm | Thuật toán bộ đệm hai hàng đợi phân tách |
| **RSS** | Resident Set Size | Kích thước bộ nhớ vật lý thực tế tiến trình chiếm dụng |
| **POSIX** | Portable Operating System Interface | Chuẩn giao tiếp hệ điều hành di động |
| **FFI** | Foreign Function Interface | Giao diện liên kết gọi hàm ngoại ngữ (Rust $\leftrightarrow$ C/CUDA) |
| **TLS** | Thread-Local Storage | Bộ nhớ cục bộ theo từng luồng |

---

## DANH MỤC CÁC BẢNG BIỂU

* **Bảng 1.1:** Phân tích thông số hình học và bộ nhớ của mô hình OLMoE-1B-7B.
* **Bảng 2.1:** So sánh độ phức tạp tính toán và I/O giữa Token-Major và Layer-Major Prefill.
* **Bảng 3.1:** So sánh chi phí đồng bộ luồng giữa Fine-Grained Fork-Join và Coarse-Grained Task Parallel.
* **Bảng 4.1:** Cấu hình phần cứng hai trạm đo kiểm nghiệm thực tế (Testbed A & Testbed B).
* **Bảng 4.2:** Sai số tuyệt đối cực đại của Logits đối chiếu giữa Engine Rust và PyTorch Transformers 4.51.3.
* **Bảng 4.3:** Lịch sử biến thiên hiệu năng qua các mốc phát triển trọng yếu (từ Baseline đến Bước 25).
* **Bảng 4.4:** Phân rã thời gian suy luận (Latency Breakdown) trên ngân sách 128 MiB/layer và 256 MiB/layer.
* **Bảng 4.5:** Ma trận đối đầu thực tế giữa hai nền tảng phần cứng Testbed A và Testbed B.

---

## DANH MỤC CÁC HÌNH VẼ VÀ SƠ ĐỒ

* **Hình 1.1:** Sơ đồ luồng tính toán Transformer Decoder với khối kích hoạt thưa Sparse MoE SwiGLU.
* **Hình 2.1:** Kiến trúc bộ nhớ phân tầng 3 cấp (L1 GPU VRAM $\leftrightarrow$ L2 Host RAM $\leftrightarrow$ NVMe SSD).
* **Hình 2.2:** Sơ đồ phân vùng và dịch chuyển phần tử trong Segmented Frequency-Recency Expert Cache (SF-Cache).
* **Hình 2.3:** Cơ chế tách rời và chồng lấn Hit-First Decoupled Compute-I/O Scheduling qua `rayon::join`.
* **Hình 2.4:** Sơ đồ khối nạp trước Opportunistic Inter-Layer Lookahead Prefetching dựa trên ma trận chuyển tiếp Markov.
* **Hình 3.1:** Đường ống SIMD unrolling 32-lane với 4 ngân hàng cộng dồn FMA độc lập trong kernel `dot_i8_f32`.
* **Hình 4.1:** Biểu đồ so sánh thời gian phản hồi token đầu tiên (TTFT) và thông lượng giải mã (Decode throughput).

---

# MỞ ĐẦU

### 1. Bối cảnh và Tính cấp thiết của Đề tài
Sự phát triển vượt bậc của các mô hình ngôn ngữ lớn (Large Language Models - LLMs) đã định hình lại toàn bộ nền tảng trí tuệ nhân tạo hiện đại. Tuy nhiên, việc tăng quy mô tham số từ hàng chục tỷ đến hàng trăm tỷ tham số để đạt được năng lực suy luận phức tạp đã tạo ra rào cản phần cứng khổng lồ, thường được gọi là "Bức tường Bộ nhớ" (Memory Wall). Đối với các mô hình dày đặc (Dense Models), toàn bộ trọng số mạng phải được nạp đầy đủ vào bộ nhớ video card đồ họa (VRAM) tốc độ cao và tham gia tính toán cho mỗi token được sinh ra. Chi phí trang bị các hệ thống máy chủ GPU chuyên dụng (như NVIDIA A100, H100) vượt xa khả năng tiếp cận của người dùng cá nhân, các phòng thí nghiệm nghiên cứu độc lập và doanh nghiệp vừa và nhỏ.

Kiến trúc hỗn hợp chuyên gia thưa (Sparse Mixture-of-Experts - MoE), tiêu biểu như Mixtral 8x7B, DBRX, DeepSeek-V2/V3 và OLMoE, nổi lên như một giải pháp mang tính cách mạng. Bằng cách thay thế các khối mạng truyền thẳng (Feed-Forward Networks - FFN) truyền thống bằng một tập hợp các mạng chuyên gia độc lập ($N$ experts) và sử dụng một bộ định tuyến thưa (Sparse Router) để chỉ kích hoạt một tập hợp con ($k \ll N$) cho mỗi token, MoE cho phép mở rộng dung lượng tri thức của mô hình tương đương một mạng khổng lồ nhưng chỉ tiêu tốn chi phí số học (FLOPs) tương đương một mạng nhỏ hơn nhiều lần.

### 2. Vấn đề Kỹ thuật Cốt lõi
Dù chi phí tính toán FLOPs giảm mạnh, rào cản bộ nhớ của kiến trúc Sparse MoE lại chuyển dịch từ **năng lực tính toán (Compute-Bound)** sang **băng thông bộ nhớ và vào/ra dữ liệu (Memory/IO-Bound)**:
1. **Dung lượng tham số tổng thể không suy giảm:** Toàn bộ $N$ chuyên gia vẫn phải thường trực hoặc sẵn sàng được truy xuất. Một mô hình như OLMoE-1B-7B sở hữu 64 chuyên gia mỗi tầng trên 16 tầng transformer, tổng dung lượng trọng số lên tới hơn 6.4 GB (ở định dạng INT8), vượt quá dung lượng VRAM khả dụng của các dòng GPU laptop phổ thông (4 GB – 6 GB).
2. **Hiện tượng rung lắc bộ nhớ đệm (Cache Thrashing):** Khi bộ nhớ RAM/VRAM không đủ để lưu trữ toàn bộ các chuyên gia, hệ thống buộc phải nạp động các chuyên gia từ ổ đĩa SSD. Do hành vi kích hoạt chuyên gia thay đổi liên tục theo từng token và từng tầng mạng (Routing Drift), nếu chính sách bộ nhớ đệm không tối ưu, các chuyên gia vừa được nạp sẽ bị trục xuất ngay lập tức, dẫn tới hiện tượng đọc đĩa lặp lại liên tục, đẩy độ trễ suy luận lên hàng chục giây cho một câu trả lời.
3. **Hiện tượng bão hòa I/O và nghẽn luồng (Prefetch Contention):** Việc cố gắng nạp trước dữ liệu một cách ngây thơ (naive eager lookahead) sẽ gây tranh chấp băng thông ổ đĩa NVMe với các chuyên gia đang cần nạp khẩn cấp cho token hiện tại, làm suy giảm nghiêm trọng thời gian tạo token đầu tiên (TTFT).

### 3. Mục tiêu Nghiên cứu của Đề tài
Mục tiêu xuyên suốt của đồ án là: **Nghiên cứu, thiết kế kiến trúc và hiện thực hóa một Inference Engine độc lập, viết hoàn toàn bằng ngôn ngữ Rust, tối ưu hóa suy luận mô hình Sparse MoE trên phần cứng máy tính cá nhân phổ thông (RAM 16 GB, GPU 4 GB - 8 GB, ổ cứng NVMe SSD).** Hệ thống phải đảm bảo:
* **Tính toàn vẹn số học tuyệt đối (Numerical Parity):** Sai số tính toán logits so với thư viện chuẩn PyTorch / Hugging Face Transformers phải tiệm cận $0.0$ (ngưỡng sai số tuyệt đối $< 2 \times 10^{-5}$).
* **Không cấp phát động trong vòng lặp suy luận (Zero-Allocation Hot-Loop):** Loại bỏ hoàn toàn chi phí dọn rác (GC pauses) và giảm thiểu tối đa các ngắt chuyển ngữ cảnh (kernel page faults).
* **Phân tầng lưu trữ thông minh (Tiered Storage Engine):** Khai thác tối đa cấu trúc phân cấp bộ nhớ từ GPU VRAM (L1), Host RAM (L2) đến NVMe SSD (Secondary Storage) thông qua cơ chế DMA Direct Pread.

### 4. Phạm vi Nghiên cứu và Đối tượng Thực nghiệm
* **Mô hình mục tiêu:** Mô hình mã nguồn mở kiến trúc MoE thuần túy **OLMoE-1B-7B-0924** do Viện AI Allen (Allen Institute for AI) phát triển. Mô hình gồm 16 tầng Transformer, mỗi tầng có 64 chuyên gia FFN, định tuyến Top-8 ($k=8$), độ dài ngữ cảnh chuẩn 512–1024 tokens.
* **Môi trường phần cứng kiểm thử:** Triển khai độc lập trên 2 trạm máy tính thực tế với các thế hệ vi kiến trúc khác nhau:
  - *Testbed A:* CPU Intel Core i7-14650HX (16 nhân, 24 luồng), RAM 16 GB DDR5, GPU NVIDIA GeForce RTX 5060 Laptop (8 GB GDDR6), NVMe PCIe 4.0 SSD.
  - *Testbed B:* CPU Intel Core i5 (6 nhân, 12 luồng), RAM 16 GB DDR4, GPU NVIDIA GeForce GTX 1650 Max-Q (4 GB GDDR5), NVMe PCIe 3.0 SSD.

### 5. Tóm tắt Đóng góp Kỹ thuật Nổi bật trong Giai đoạn 1
Trong giai đoạn nghiên cứu và hiện thực hóa đầu tiên (tương ứng từ Bước 1 đến Bước 25 trong nhật ký tiến độ), đồ án đã đạt được các kết quả mang tính đột phá sau:
1. **Thiết kế và hiện thực thành công hệ thống lưu trữ phân tầng 3 cấp (L1 VRAM $\leftrightarrow$ L2 RAM $\leftrightarrow$ NVMe SSD):** Tận dụng vùng VRAM nhàn rỗi làm L1 Hot Expert Cache với chi phí sao chép PCIe DMA siêu tốc (~0.4 ms), nạp Dense weights lên GPU qua thư viện tự phát triển tích hợp cuBLAS FFI.
2. **Đề xuất thuật toán bộ đệm Segmented Frequency-Recency Expert Cache (SF-Cache):** Phân vùng cố định 70% ngân sách cho tập chuyên gia tổng quát (Protected Generalists) và 30% cho chuyên gia cục bộ (Probationary Specialists), giải quyết triệt để vấn đề Cache Thrashing do Routing Drift.
3. **Phát triển cơ chế lập lịch tính toán - I/O tách rời và chồng lấn (Hit-First Decoupled Compute-I/O Scheduling):** Tách luồng xử lý Hit và Miss bằng `rayon::join`, ẩn giấu 100% thời gian trễ đọc ổ đĩa NVMe dưới thời gian tính toán của CPU.
4. **Xây dựng mô hình Markov trực tuyến dự đoán định tuyến liên tầng (Opportunistic Lookahead MoE Prefetching):** Khai thác xác suất chuyển tiếp $P(\text{Expert}^{(L+1)}_j \mid \text{Expert}^{(L)}_i)$ cập nhật online trong quá trình giải mã, chỉ nạp trước khi tầng hiện tại đạt 100% Cache Hit nhằm triệt tiêu hoàn toàn hiện tượng nghẽn I/O (Prefetch Contention).
5. **Tối ưu hóa vi kiến trúc CPU SIMD và điều phối đa luồng cấp thô (Coarse-Grained Task Parallelism):** Giảm 93.7% rào cản đồng bộ luồng Rayon, tích hợp kernel Fused Gate-Up SwiGLU với đường ống SIMD 32-lane 4-bank FMA unrolling, đi kèm hồ đệm Scratch Pool (96 KB) nằm trọn trong CPU L2 Cache.
6. **Triệt tiêu gánh nặng cấp phát bộ nhớ I/O (Zero-Allocation Expert I/O):** Kết hợp Buffer Recycling Pool với POSIX `pread64` Direct DMA, triệt tiêu 100% các lệnh `malloc/free` 6.3 MB và loại bỏ 1.37 triệu Minor Page Faults của kernel Linux.
7. **Chứng minh thực nghiệm toàn diện:** Tốc độ giải mã (Decode throughput) tăng từ **0.82 tokens/s lên 4.40 tokens/s** (trên cấu hình chuẩn) và đạt **8.51 tokens/s** (trên GPU warm), thời gian sinh phản hồi giảm từ **29.86 giây xuống còn 5.89 giây** (giảm 80.3%), giảm dung lượng đọc đĩa từ **18.66 GB xuống 2.08 GB** (giảm 88.8%), tỷ lệ Cache Hit đạt **93.0%**, toàn bộ 59/59 test suites đạt kết quả chuẩn xác tuyệt đối.

---

# CHƯƠNG 1: CƠ SỞ LÝ THUYẾT VÀ MÔ HÌNH TOÁN HỌC CỦA SPARSE MOE

## 1.1. Cấu trúc Mô hình Transformer Decoder với Nhánh Sparse MoE
Kiến trúc tổng thể của mô hình giải mã Transformer bao gồm một chuỗi các tầng nối tiếp nhau ($L = 1, \dots, N_{layers}$). Mỗi tầng bao gồm hai khối tính toán chính: Khối Chú ý Bản thân Có hướng (Causal Self-Attention) và Khối Mạng Nơ-ron Truyền thẳng (FFN). Đối với kiến trúc Sparse MoE, khối FFN dày đặc truyền thống được thay thế bằng một khối Hỗn hợp Chuyên gia thưa.

```
Input Tokens: x_0, x_1, ..., x_t
        │
        ▼
┌──────────────────────────────┐
│  Input Embedding + RMSNorm   │
└──────────────┬───────────────┘
               │
        ┌──────▼──────┐◄─────────────────────────┐
        │  Layer l    │                          │
        │  ┌──────────┴───────────────────────┐  │
        │  │ Causal Self-Attention (GQA+RoPE) │  │  Residual
        │  └──────────┬───────────────────────┘  │  Connection
        │             │                          │
        │  ┌──────────▼───────────────────────┐  │
        │  │ RMSNorm                          │  │
        │  └──────────┬───────────────────────┘  │
        │             │                          │
        │  ┌──────────▼───────────────────────┐  │
        │  │ Sparse MoE Layer                 │  │
        │  │  - Gating Network: Top-k Routing │  │
        │  │  - Active Experts: SwiGLU FFNs   │  │
        │  └──────────┬───────────────────────┘  │
        │             │                          │
        │             ▼                          │
        │  ┌──────────────────────────────────┐  │
        │  │ Add Residual & RMSNorm           ├──┘
        │  └──────────┬───────────────────────┘
        └─────────────┼──────────────────────────┘
                      │  (Lặp lại qua N tầng)
                      ▼
        ┌──────────────────────────────┐
        │   Final RMSNorm & LM Head    │
        └─────────────┬────────────────┘
                      │
                      ▼
               Output Logits
```
*Hình 1.1: Sơ đồ luồng tính toán Transformer Decoder với khối kích hoạt thưa Sparse MoE SwiGLU.*

### 1.1.1. Cơ chế Chú ý Nhóm Truy vấn (Grouped-Query Attention - GQA)
Tại mỗi tầng mạng, vector ẩn đầu vào $x \in \mathbb{R}^{d}$ được chuẩn hóa bằng RMSNorm, sau đó chiếu tuyến tính thành các ma trận Truy vấn ($Q$), Khóa ($K$) và Giá trị ($V$):
$$Q = x W_Q, \quad K = x W_K, \quad V = x W_V$$
Trong đó $W_Q \in \mathbb{R}^{d \times (n_q \cdot d_h)}$, $W_K \in \mathbb{R}^{d \times (n_{kv} \cdot d_h)}$, $W_V \in \mathbb{R}^{d \times (n_{kv} \cdot d_h)}$. Với kiến trúc GQA, số lượng đầu khóa và giá trị $n_{kv}$ nhỏ hơn số lượng đầu truy vấn $n_q$ (theo tỷ lệ nhóm $g = n_q / n_{kv}$), giúp giảm thiểu kích thước lưu trữ của bộ đệm KV Cache trong quá trình suy luận.

### 1.1.2. Nhúng Vị trí Quay (Rotary Position Embedding - RoPE)
Để bảo toàn thông tin thứ tự từ ngữ mà không cần thêm ma trận trọng số vị trí tĩnh, vector truy vấn $Q$ và khóa $K$ tại vị trí token $m$ được quay trong không gian phức hai chiều:
$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$
Với ma trận quay trực giao cấp 2:
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}, \quad \theta_i = b^{-2(i-1)/d}, \quad b = 10000$$
Tích vô hướng giữa truy vấn tại vị trí $m$ và khóa tại vị trí $n$ suy biến thành một hàm phụ thuộc duy nhất vào khoảng cách tương đối $(m - n)$:
$$\langle \mathbf{R}_{\Theta, m}^d q_m, \mathbf{R}_{\Theta, n}^d k_n \rangle = \text{Re} \left( q_m k_n^* e^{j(m-n)\Theta} \right)$$

### 1.1.3. Chuẩn hóa Root Mean Square Normalization (RMSNorm)
Thay vì sử dụng LayerNorm truyền thống đòi hỏi tính toán cả giá trị kỳ vọng (mean) và phương sai (variance), RMSNorm chỉ điều chỉnh theo căn bậc hai trung bình bình phương, giúp tiết kiệm chu kỳ xung nhịp CPU:
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad \text{với} \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
Trong đó $\gamma \in \mathbb{R}^d$ là vector trọng số học tỉ lệ, và $\epsilon$ là hằng số chống tràn số thực (thường là $10^{-6}$).

---

## 1.2. Mô hình Toán học của Bộ định tuyến Chuyên gia (Expert Routing Mechanism)

Khối Sparse MoE thay thế biến đổi FFN truyền thống bằng một tổ hợp gồm $N$ chuyên gia song song: $\{\text{Expert}_i\}_{i=0}^{N-1}$. Một mạng định tuyến tuyến tính (Router Gating Network) với trọng số $W_g \in \mathbb{R}^{d \times N}$ tiếp nhận biểu diễn ẩn $x \in \mathbb{R}^d$ và tính toán vector điểm số định tuyến:
$$H(x) = x \cdot W_g \in \mathbb{R}^N$$

### 1.2.1. Phân phối Xác suất Softmax và Lọc Top-$k$
Điểm số thô được đưa qua hàm Softmax để chuyển đổi thành phân phối xác suất trên toàn bộ $N$ chuyên gia:
$$P(x) = \text{Softmax}(H(x)), \quad P_i(x) = \frac{\exp(H_i(x))}{\sum_{j=0}^{N-1} \exp(H_j(x))}$$
Bộ chọn lựa thưa Top-$k$ sẽ trích xuất tập hợp chỉ mục $I_k(x)$ chứa đúng $k$ chuyên gia có giá trị xác suất cao nhất:
$$I_k(x) = \operatorname{argtopk}\left( \{ P_i(x) \}_{i=0}^{N-1}, \, k \right), \quad |I_k(x)| = k \ll N$$

### 1.2.2. Tái Chuẩn hóa Trọng số Định tuyến (Renormalization)
Để đảm bảo tổng trọng số kích hoạt của các chuyên gia được chọn bằng $1.0$, hệ thống thực hiện tái chuẩn hóa phân phối trên tập con $I_k(x)$:
$$\tilde{w}_i(x) = \begin{cases} \dfrac{P_i(x)}{\sum_{j \in I_k(x)} P_j(x)} = \dfrac{\exp(H_i(x))}{\sum_{j \in I_k(x)} \exp(H_j(x))}, & \text{nếu } i \in I_k(x) \\ 0, & \text{nếu } i \notin I_k(x) \end{cases}$$
Đầu ra tổng hợp của khối MoE là tổ hợp tuyến tính có trọng số từ các chuyên gia được kích hoạt:
$$y = x + \sum_{i \in I_k(x)} \tilde{w}_i(x) \cdot \text{Expert}_i(x)$$

---

## 1.3. Cấu trúc Tính toán Khối SwiGLU Feed-Forward Network

Mỗi chuyên gia $\text{Expert}_i$ trong mô hình OLMoE được cấu hình theo kiến trúc SwiGLU (Swish-Gated Linear Unit). Cấu trúc này bao gồm 3 ma trận trọng số độc lập:
* Ma trận Cổng (Gate Projection): $W_{gate} \in \mathbb{R}^{d \times d_{ff}}$
* Ma trận Tăng chiều (Up Projection): $W_{up} \in \mathbb{R}^{d \times d_{ff}}$
* Ma trận Hạ chiều (Down Projection): $W_{down} \in \mathbb{R}^{d_{ff} \times d}$

### 1.3.1. Hàm kích hoạt SiLU (Swish-1)
Hàm kích hoạt SiLU được xác định bởi công thức:
$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

### 1.3.2. Công thức Tổng hợp SwiGLU FFN
Đầu ra của chuyên gia đối với vector đầu vào $x \in \mathbb{R}^d$ được tính toán thông qua tích Hadamard (element-wise multiplication) giữa nhánh cổng đã kích hoạt và nhánh tăng chiều, sau đó nhân với ma trận hạ chiều:
$$\text{Activated}(x) = \text{SiLU}(x \cdot W_{gate}) \odot (x \cdot W_{up}) \in \mathbb{R}^{d_{ff}}$$
$$\text{Expert}(x) = \text{Activated}(x) \cdot W_{down} \in \mathbb{R}^d$$

---

## 1.4. Lượng tử hóa Trọng số Số nguyên INT8 Đối xứng theo Hàng

Để giảm tải áp lực bộ nhớ và băng thông I/O, toàn bộ trọng số của các chuyên gia được nén sang định dạng số nguyên có dấu 8-bit (INT8) thông qua phương pháp lượng tử hóa đối xứng theo từng hàng (Per-row Symmetric Quantization).

### 1.4.1. Công thức Lượng tử hóa và Tính Tỉ lệ Co giãn (Scale Factor)
Xét một ma trận trọng số $W \in \mathbb{R}^{M \times K}$. Với mỗi hàng $i \in [0, M-1]$, hệ số tỉ lệ $s_i \in \mathbb{R}^+$ được xác định dựa trên giá trị tuyệt đối lớn nhất của các phần tử trong hàng đó:
$$s_i = \frac{\max_{j \in [0, K-1]} |W_{i, j}|}{127.0}$$
Ma trận trọng số lượng tử hóa $Q \in \mathbb{Z}^{M \times K}$ được tạo ra bởi phép làm tròn và giới hạn biên (clamping):
$$Q_{i, j} = \text{clip}\left( \left\lfloor \frac{W_{i, j}}{s_i} \right\rceil, \, -128, \, 127 \right)$$

### 1.4.2. Tích vô hướng Khôi phục Trực tiếp (Dequantized Dot Product)
Trong quá trình suy luận, tích vô hướng giữa hàng ma trận lượng tử hóa $Q_i$ và vector kích hoạt số thực $x \in \mathbb{R}^K$ được tính toán trực tiếp mà không cần giải nén toàn bộ ma trận về bộ nhớ:
$$y_i = W_i \cdot x \approx s_i \cdot \sum_{j=0}^{K-1} \left( Q_{i, j} \cdot x_j \right)$$
Kỹ thuật này cho phép chuyển đổi phần lớn các phép nhân số thực dấu phẩy động (FP32 MUL) thành phép nhân số nguyên mở rộng dấu (INT8 $\times$ FP32 $\to$ FP32), bảo toàn độ chính xác số học với sai số cực đại nhỏ hơn $2 \times 10^{-5}$ so với tính toán FP32 gốc.

---

## 1.5. Phân tích Định lượng Rào cản Bộ nhớ và Nút thắt I/O

Mô hình thực nghiệm **OLMoE-1B-7B-0924** có các tham số hình học được thống kê chi tiết trong Bảng 1.1:

*Bảng 1.1: Phân tích thông số hình học và bộ nhớ của mô hình OLMoE-1B-7B.*

| Thông số hình học | Ký hiệu | Giá trị |
| :--- | :--- | :--- |
| Số lượng tầng mạng (Transformer Layers) | $L$ | 16 |
| Kích thước vector ẩn (Hidden Size) | $d$ | 2,048 |
| Kích thước lớp trung gian FFN (Intermediate Size) | $d_{ff}$ | 1,024 |
| Tổng số chuyên gia mỗi tầng (Total Experts / Layer) | $N$ | 64 |
| Số chuyên gia tích cực mỗi token (Active Experts / Token) | $k$ | 8 |
| Số đầu chú ý Truy vấn / Khóa-Giá trị ($n_q / n_{kv}$) | $n_q / n_{kv}$ | 16 / 16 |
| Tổng số chuyên gia toàn mạng ($L \times N$) | $N_{total}$ | 1,024 experts |

### 1.5.1. Kích thước Trọng số của một Chuyên gia
Mỗi chuyên gia gồm 3 ma trận $W_{gate}$, $W_{up}$, $W_{down}$. Với $d = 2048$ và $d_{ff} = 1024$:
$$\text{Số lượng tham số} = 2 \times (2048 \times 1024) + (1024 \times 2048) = 6,291,456 \text{ phần tử}$$
Ở định dạng INT8 (1 byte/phần tử) cộng thêm vector hệ số co giãn FP32 ($d_{ff} + d_{ff} + d = 4096$ floats $\times$ 4 bytes $= 16,384$ bytes):
$$\text{Dung lượng 1 Expert} = 6,291,456 \text{ bytes} + 16,384 \text{ bytes} \approx 6.307 \text{ MB}$$

### 1.5.2. Tính toán Dung lượng Toàn mạng và Active Set
* **Tổng dung lượng của 1,024 chuyên gia:**
  $$\text{Dung lượng Experts} = 1,024 \times 6.307 \text{ MB} \approx 6,458 \text{ MB} \approx 6.46 \text{ GB}$$
* **Dung lượng tập chuyên gia hoạt động (Active Set) cho 1 token:**
  Tại mỗi tầng, có $k=8$ chuyên gia được kích hoạt:
  $$\text{Active Set / Layer} = 8 \times 6.307 \text{ MB} \approx 50.46 \text{ MB}$$
  Toàn bộ 16 tầng yêu cầu một tập hợp hoạt động tức thời:
  $$\text{Total Active Set / Token} = 16 \times 50.46 \text{ MB} \approx 807.36 \text{ MB}$$

### 1.5.3. Nghịch lý Cache Thrashing tại Cấu hình Ngân sách Cũ (32 MiB/layer)
Khi bắt đầu nghiên cứu (Mốc Bước 1 trong PROGRESS.md), hệ thống gán mặc định ngân sách cache là $32\text{ MiB/layer}$. Khi đối chiếu với kích thước Active Set ($50.46\text{ MB}$), ta nhận thấy:
$$\text{Ngân sách Cache (32 MiB)} < \text{Active Set cần thiết (50.46 MiB)}$$
Do đó, ngay trong quá trình giải mã 1 token duy nhất, bộ nhớ đệm đã bị tràn. Hệ thống liên tục nạp 5 chuyên gia đầu tiên, sau đó buộc phải trục xuất (evict) các chuyên gia này để lấy chỗ trống nạp 3 chuyên gia còn lại. Sang token kế tiếp, các chuyên gia vừa bị trục xuất lại được định tuyến chọn lại, gây ra hiện tượng rung lắc bộ nhớ đệm (Cache Thrashing) cực kỳ nghiêm trọng. Bằng chứng thực nghiệm tại Bước 1 ghi nhận: chỉ để giải mã một câu hỏi ngắn sinh ra 7 tokens, hệ thống đã thực hiện **2,878 lần trục xuất** và phải đọc tới **18.66 GB dữ liệu từ SSD**, khiến thời gian sinh kéo dài tới **29.86 giây**.

---

# CHƯƠNG 2: THIẾT KẾ KIẾN TRÚC HỆ THỐNG VÀ CÁC GIẢI THUẬT ĐỘT PHÁ

Để giải quyết triệt để rào cản băng thông bộ nhớ và hiện tượng I/O Thrashing đã phân tích ở Chương 1, hệ thống `MoE-TierEngine` được xây dựng dựa trên kiến trúc phân tầng 3 cấp kết hợp các giải thuật phối hợp luồng dữ liệu độc đáo.

```
┌────────────────────────────────────────────────────────┐
│               L1: GPU VRAM TIER (~3.4 GB)              │
│  - Dense Weights & Router (1.4 GB)                     │
│  - L1 Hot Expert Cache (~2.0 GB, ~300 Hot Experts)     │
└───────────────────────────▲────────────────────────────┘
                            │ PCIe 3.0/4.0 DMA (~0.4 ms)
┌───────────────────────────▼────────────────────────────┐
│               L2: HOST RAM TIER (~2.5 - 4.5 GB)        │
│  - SF-Cache (128 - 256 MiB/layer: 70% Prot / 30% Prob) │
│  - KV Cache & Layer Scratches (96 KB L2 Resident)      │
│  - Buffer Recycling Pool (16 slots/layer)              │
└───────────────────────────▲────────────────────────────┘
                            │ POSIX pread64 DMA (Direct)
┌───────────────────────────▼────────────────────────────┐
│            SECONDARY STORAGE: NVMe SSD TIER            │
│  - 1,024 Experts (INT8 SwiGLU in Safetensors, 6.4 GB)  │
│  - Shared Open File Descriptors (Zero Seek Lock)       │
└────────────────────────────────────────────────────────┘
```
*Hình 2.1: Kiến trúc bộ nhớ phân tầng 3 cấp (L1 GPU VRAM $\leftrightarrow$ L2 Host RAM $\leftrightarrow$ NVMe SSD).*

## 2.1. Kiến trúc Bộ nhớ Phân tầng 3 Cấp (Tiered Storage Engine)

Hệ thống phân chia toàn bộ không gian tham số và bộ nhớ đệm thành 3 phân vùng vật lý:
1. **Phân tầng 1 (L1 GPU VRAM Tier):**
   * Lưu trữ toàn bộ các tham số dày đặc (Dense Weights): Embedding, Attention ($W_Q, W_K, W_V, W_O$), Router Gate ($W_g$), RMSNorm và ma trận chiếu từ vựng LM Head (~1.4 GB FP32).
   * Khai thác vùng VRAM còn lại (khoảng 2.0 GB trên GPU 4GB/8GB) để thiết lập **L1 VRAM Expert Cache**, lưu trữ cố định khoảng 300 chuyên gia "nóng" nhất. Tốc độ chuyển giao trọng số từ L1 VRAM xuống bộ đệm Host CPU qua chuẩn PCIe DMA chỉ mất **$\approx 0.4\text{ ms}$**, nhanh gấp hơn 25 lần so với đọc đĩa SSD.
2. **Phân tầng 2 (L2 Host RAM Tier):**
   * Quản lý bộ nhớ đệm chuyên gia cấp tầng với dung lượng cấu hình linh hoạt từ $128\text{ MiB}$ đến $256\text{ MiB}$ mỗi tầng (tương đương 20 đến 40 chuyên gia/tầng).
   * Hồ đệm tái chế bộ nhớ I/O (Buffer Recycling Pool) và các vector kích hoạt tạm thời (Scratches).
3. **Phân tầng Lưu trữ Thứ cấp (Secondary NVMe SSD Tier):**
   * Toàn bộ 1,024 chuyên gia INT8 được đóng gói trong tệp nhị phân Safetensors duy nhất. Các file descriptor được mở sẵn (`open`) khi khởi tạo engine, duy trì trạng thái thường trực trong bảng quản lý tệp của tiến trình để triệt tiêu chi phí mở/đóng tệp hệ điều hành.

---

## 2.2. Thuật toán Tiền xử lý Gom cụm Chuyên gia theo Tầng (Layer-Major Expert-Batched Prefill)

Trong giai đoạn nạp ngữ cảnh đầu vào (Prefill Phase) với chuỗi gồm $T$ tokens, phương pháp xử lý tuần tự từng token theo chiều sâu mạng (Token-Major) dẫn đến sự suy giảm nghiêm trọng về hiệu năng I/O.

### 2.2.1. Phân tích Hạn chế của Token-Major Prefill
Với cách tiếp cận ngây thơ (Token-Major):
```text
Với mỗi token t = 1..T:
    Với mỗi tầng l = 1..L:
        Tính Attention, Router -> Lấy 8 experts -> Nạp từ SSD/Cache -> Tính toán
```
Nếu chuỗi prompt có độ dài $T = 512$ tokens và mỗi tầng có $k=8$ chuyên gia hoạt động: số lượt truy vấn chuyên gia là $T \times L \times k = 512 \times 16 \times 8 = 65,536$ lượt. Do dung lượng cache của một tầng không thể chứa toàn bộ 64 chuyên gia, các chuyên gia sẽ bị nạp và trục xuất hàng ngàn lần.

### 2.2.2. Thiết kế Thuật toán Layer-Major Expert-Batched
`MoE-TierEngine` đảo ngược thứ tự vòng lặp, duyệt theo từng tầng trước (Layer-Major) và thực hiện gom cụm toàn bộ $T$ tokens của prompt theo chuyên gia mục tiêu:

```text
Thuật toán 1: Layer-Major Expert-Batched Prefill
Đầu vào: Chuỗi vector ẩn X = [x_1, x_2, ..., x_T] ∈ R^(T x d)
1: Với mỗi tầng l = 1..L:
2:    X_norm = RMSNorm(X)
3:    X_attn = Batched_Causal_Attention(X_norm) + X
4:    X_norm2 = RMSNorm(X_attn)
5:    H = X_norm2 · W_g                        // Điểm số Router cho toàn bộ T tokens
6:    Khởi tạo bảng danh sách gom cụm: Buckets[e] = [] với e = 0..N-1
7:    Với mỗi token t = 1..T:
8:        Top_k = argtopk(Softmax(H[t]), k)
9:        Với mỗi e ∈ Top_k:
10:           Thêm (t, Trọng_số_chuẩn_hóa(t, e)) vào Buckets[e]
11:   X_moe_out = Ma trận zeros(T, d)
12:   Với mỗi chuyên gia e có Buckets[e] không rỗng:
13:       Expert_e = Nạp_Expert(l, e)          // CHỈ NẠP DUY NHẤT 1 LẦN CHO CẢ CHUỖI PROMPT
14:       Với mỗi (t, weight) ∈ Buckets[e]:
15:           Out_t = Expert_e(X_norm2[t])
16:           X_moe_out[t] += weight * Out_t
17:   X = X_attn + X_moe_out
Đầu ra: Chuỗi vector ẩn X sẵn sàng cho tầng kế tiếp
```

*Bảng 2.1: So sánh độ phức tạp tính toán và I/O giữa Token-Major và Layer-Major Prefill.*

| Tiêu chí so sánh | Token-Major Prefill (Truyền thống) | Layer-Major Expert-Batched Prefill (Đề tài) |
| :--- | :--- | :--- |
| **Số lần nạp Expert tối đa từ đĩa** | $O(T \cdot L \cdot k)$ (Lên tới hàng vạn lần) | **$O(L \cdot \min(N, T \cdot k))$ (Tối đa $16 \times 64 = 1,024$ lần)** |
| **Tần suất nạp lại 1 Expert trong 1 Layer** | $O(T)$ lần (Bị nạp/trục xuất lặp lại) | **Chính xác 1 lần duy nhất** |
| **Hành vi nạp đĩa SSD** | Rời rạc, phân mảnh, gây nghẽn hàng đợi I/O | Tuyến tính, tuần tự, gom khối lớn |

---

## 2.3. Chiến lược Quản lý Bộ nhớ đệm Phân đoạn Tần suất - Độ mới (SF-Cache)

Giải thuật LRU (Least Recently Used) truyền thống chỉ đánh giá yếu tố thời gian truy cập gần nhất, do đó rất dễ bị tổn thương trước hiện tượng "quét ngẫu nhiên" (scan pollution) khi bộ định tuyến chuyển dịch chủ đề (Routing Drift). Khi một từ ngữ hiếm kích hoạt một tập hợp các chuyên gia hẹp (Specialists), LRU sẽ đẩy các chuyên gia tổng quát (Generalists - thường xuyên được gọi ở hầu hết mọi câu) ra khỏi cache.

Để khắc phục, đề tài phát triển giải thuật **Segmented Frequency-Recency Expert Cache (SF-Cache)** dựa trên nguyên lý kết hợp giữa 2Q và Segmented-LRU:

```
                  Expert mới nạp từ SSD / L1
                             │
                             ▼
            ┌──────────────────────────────────┐
            │   PROBATIONARY SEGMENT (30%)     │
            │   (Cửa sổ thử thách độ mới)      │
            └────────────────┬─────────────────┘
                             │ Hit lần 2 (Thăng hạng)
                             ▼
            ┌──────────────────────────────────┐
            │     PROTECTED SEGMENT (70%)      │◄──┐
            │   (Tập chuyên gia Generalists)   │   │ Hit lặp lại
            └────────────────┬─────────────────┘───┘
                             │ Vượt ngân sách 70% (Giáng hạng)
                             ▼
            ┌──────────────────────────────────┐
            │   PROBATIONARY SEGMENT (30%)     │
            └────────────────┬─────────────────┘
                             │ Victim trục xuất khi đầy
                             ▼
                       TRỤC XUẤT (EVICT)
```
*Hình 2.2: Sơ đồ phân vùng và dịch chuyển phần tử trong SF-Cache.*

### 2.3.1. Phân vùng Ngân sách Bộ đệm
Ngân sách bộ đệm của mỗi tầng ($C_{budget}$) được chia thành 2 phân vùng vật lý:
1. **Phân vùng Bảo vệ (Protected Segment - $70\% C_{budget}$):** Dành riêng cho các chuyên gia có tần suất xuất hiện cao (Generalists). Các chuyên gia trong vùng này không bao giờ bị trục xuất trực tiếp ra đĩa khi có chuyên gia mới nạp vào.
2. **Phân vùng Thử thách (Probationary Segment - $30\% C_{budget}$):** Đóng vai trò là cửa sổ thẩm định cho các chuyên gia mới được nạp vào lần đầu.

### 2.3.2. Quy tắc Dịch chuyển và Trục xuất (Admission, Promotion & Eviction)
* **Tiếp nhận ban đầu (Admission):** Một chuyên gia nạp từ SSD vào L2 sẽ được gán nhãn `is_protected = false` và đưa vào phân vùng Probationary.
* **Thăng hạng (Promotion):** Khi một chuyên gia trong cache ghi nhận thêm một lượt truy cập (Cache Hit), nó ngay lập tức được thăng hạng lên phân vùng Protected (`is_protected = true`).
* **Giáng hạng (Demotion):** Nếu phân vùng Protected vượt quá hạn mức $70\%$, phần tử có độ ưu tiên thấp nhất trong vùng Protected (xác định bởi cặp khóa `(heat, last_used)`) sẽ bị tước nhãn bảo vệ và giáng hạng xuống vùng Probationary:
  $$\text{Victim}_{\text{demote}} = \operatorname{argmin}_{e \in \text{Protected}} \left( \text{Heat}[e], \, \text{Timestamp}[e] \right)$$
* **Trục xuất ra khỏi RAM (Eviction):** Khi tổng kích thước bộ đệm vượt quá $C_{budget}$, hệ thống luôn ưu tiên tìm và giải phóng phần tử trong phân vùng Probationary trước:
  $$\text{Victim}_{\text{evict}} = \operatorname{argmin}_{e \in \text{Probationary}} \left( \text{Heat}[e], \, \text{Timestamp}[e] \right)$$
  Chỉ khi phân vùng Probationary hoàn toàn rỗng, hệ thống mới buộc phải trục xuất phần tử nguội nhất của vùng Protected.

### 2.3.3. Cơ chế Giải trừ Tần suất (Heat Aging)
Để thích ứng linh hoạt khi ngữ cảnh người dùng chuyển dịch hẳn sang một chủ đề mới, đồng hồ tầng (`layer.clock`) tăng thêm 1 sau mỗi lần truy xuất. Khi đồng hồ chạm chu kỳ $2^{16} = 65,536$ bước, toàn bộ mảng nhiệt lượng của 64 chuyên gia được chia đôi:
$$\text{Heat}[e] \leftarrow \left\lfloor \frac{\text{Heat}[e]}{2} \right\rfloor, \quad \forall e \in [0, N-1]$$
Công thức này ngăn chặn hiện tượng các chuyên gia của câu hỏi cũ giữ nhiệt lượng quá cao và chiếm dụng vĩnh viễn vùng Protected.

---

## 2.4. Thuật toán Tách rời Tính toán & I/O Chồng lấn (Hit-First Decoupled Compute-I/O Scheduling)

Trong giai đoạn giải mã từng token (Decode Phase), với mỗi tầng mạng, tập hợp 8 chuyên gia tích cực $I_k(x)$ được chia tách thành hai tập con không giao nhau:
$$I_k(x) = S_{hit} \cup S_{miss}, \quad S_{hit} \cap S_{miss} = \emptyset$$
Trong đó:
* $S_{hit} = \{ e \in I_k(x) \mid e \in \text{Cache}_{L2} \lor e \in \text{Cache}_{L1} \}$ (Các chuyên gia đã có sẵn trong bộ nhớ).
* $S_{miss} = \{ e \in I_k(x) \mid e \notin \text{Cache}_{L2} \land e \notin \text{Cache}_{L1} \}$ (Các chuyên gia bắt buộc phải đọc từ SSD).

```
Token Input x
      │
      ▼
┌──────────────┐
│  Router Top-8│
└──────┬───────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
Tập S_hit (Có sẵn trong RAM/VRAM)     Tập S_miss (Bị thiếu, cần nạp SSD)
       │                                 │
       └────────────────┬────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │      rayon::join      │
            └───────────┬───────────┘
       ┌────────────────┴────────────────┐
       ▼ Nhánh CPU Compute               ▼ Nhánh Async I/O
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ Tính toán song song S_hit    │   │ Nạp song song S_miss từ SSD  │
│ (8 luồng worker trên CPU)    │   │ (POSIX pread64 qua Direct DMA│
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               │                                  │
               └────────────────┬─────────────────┘
                                │ Đồng bộ hoàn tất
                                ▼
                   ┌──────────────────────────────┐
                   │ Tính toán song song S_miss   │
                   └────────────┬─────────────────┘
                                │
                                ▼
                   ┌──────────────────────────────┐
                   │ Deterministic Fused Reduction│
                   └──────────────────────────────┘
```
*Hình 2.3: Cơ chế tách rời và chồng lấn Hit-First Decoupled Compute-I/O Scheduling qua `rayon::join`.*

### 2.4.1. Thực thi Bất đồng bộ với `rayon::join`
Thay vì thực hiện nạp toàn bộ các chuyên gia bị thiếu rồi mới tiến hành tính toán (gây ra tình trạng CPU nhàn rỗi chờ đợi I/O), hệ thống kích hoạt đồng thời hai nhánh tác vụ song song thông qua cấu trúc Fork-Join của Rayon:
```rust
let (io_res, compute_res) = rayon::join(
    // Nhánh 1: I/O Worker Pool thực hiện nạp S_miss từ NVMe SSD
    || {
        miss_ids.into_par_iter()
            .zip(curr_buffers.into_par_iter())
            .filter_map(|(eid, (w, s))| {
                Int8Expert::load_reusing(index, id, eid, d, intermediate, w, s).ok()
            }).collect()
    },
    // Nhánh 2: Compute Worker Pool thực hiện tính toán FFN cho các Hit Experts
    || {
        hit_ready.par_iter()
            .zip(hit_scratches.par_iter_mut())
            .try_for_each(|(ready, scratch_buf)| {
                ready.expert.forward_single_threaded(hidden, scratch_buf)
            })
    }
);
```

### 2.4.2. Hiệu quả Triệt tiêu Độ trễ I/O
Gọi $T_{\text{compute}}(S_{hit})$ là thời gian CPU tính toán cho tập $S_{hit}$, và $T_{I/O}(S_{miss})$ là thời gian đọc ổ đĩa NVMe để nạp tập $S_{miss}$. Độ trễ quan sát được của giai đoạn này là:
$$T_{\text{phase1}} = \max\left( T_{\text{compute}}(S_{hit}), \, T_{I/O}(S_{miss}) \right)$$
Do thời gian tính toán FFN của $S_{hit}$ trên CPU thường dao động từ $15\text{ ms}$ đến $25\text{ ms}$, trong khi thời gian đọc 2–3 chuyên gia từ ổ SSD NVMe qua POSIX DMA chỉ mất từ $10\text{ ms}$ đến $18\text{ ms}$, toàn bộ độ trễ I/O đã được **ẩn giấu hoàn toàn (100% hidden overhead)** dưới thời gian thực thi tính toán của CPU.

---

## 2.5. Mô hình Markov Trực tuyến Dự đoán Định tuyến Liên tầng (Opportunistic Lookahead MoE Prefetching)

Mặc dù việc chồng lấn I/O tại mục 2.4 đã giải quyết được độ trễ khi $S_{hit} \neq \emptyset$, trường hợp xấu nhất xảy ra khi một tầng gặp **100% Cache Miss** ($S_{hit} = \emptyset$). Khi đó, CPU buộc phải dừng hoàn toàn để chờ đọc toàn bộ 8 chuyên gia từ SSD. Để khắc phục rào cản tuần tự liên tầng này, hệ thống hiện thực cơ chế nạp trước dựa trên suy đoán (Lookahead Speculative Prefetching).

```
Tầng L (100% Cache Hit)
┌──────────────────────────────────────────────┐
│ Đang tính toán 8 Hit Experts trên CPU        │
└──────────────────────┬───────────────────────┘
                       │ Kích hoạt Lookahead Prefetching
                       ▼
┌──────────────────────────────────────────────┐
│ Ma trận Chuyển tiếp Markov P(E^(L+1) | E^(L))│
│ Kết hợp Heatmap L2 Cache của Tầng L+1        │
└──────────────────────┬───────────────────────┘
                       │ Dự đoán 2 ứng viên có xác suất cao nhất
                       ▼
┌──────────────────────────────────────────────┐
│ NVMe I/O Channel (Đang hoàn toàn rảnh rỗi)   │
│ Nạp trước Candidate 1 & 2 của Tầng L+1       │
└──────────────────────────────────────────────┘
```
*Hình 2.4: Sơ đồ khối nạp trước Opportunistic Inter-Layer Lookahead Prefetching dựa trên ma trận chuyển tiếp Markov.*

### 2.5.1. Ma trận Xác suất Chuyển tiếp Markov Liên tầng
Giữa hai tầng liên tiếp $L$ và $L+1$, hành vi kích hoạt của bộ định tuyến có sự tương quan ngữ nghĩa mạnh mẽ. Hệ thống duy trì một ma trận chuyển tiếp Markov $M^{(L)} \in \mathbb{N}^{64 \times 64}$ cho mỗi cặp tầng:
$$M^{(L)}[i, j] = \text{Số lần chuyên gia } j \text{ tại tầng } L+1 \text{ được kích hoạt khi chuyên gia } i \text{ tại tầng } L \text{ hoạt động}$$
Ma trận này được khởi tạo và cập nhật động (online learning) trực tiếp trong quá trình giải mã mà không cần huấn luyện lại mô hình:
$$M^{(L)}[i, j] \leftarrow M^{(L)}[i, j] + 1, \quad \forall i \in I_k^{(L)}, \, \forall j \in I_k^{(L+1)}$$
Để tránh hiện tượng tràn số nguyên $16$-bit (`u16`), khi tổng lượt truy cập đạt $60,000$, ma trận được chuẩn hóa tỉ lệ bằng phép dịch bit phải.

### 2.5.2. Hàm Đánh giá Điểm số Ứng viên (Candidate Scoring Function)
Tại tầng hiện tại $L$, khi tập chuyên gia $I_k^{(L)}$ đã được xác định, điểm số dự báo cho chuyên gia ứng viên $j$ ở tầng kế tiếp $L+1$ được tính toán kết hợp giữa xác suất chuyển tiếp Markov và tín hiệu nhiệt LFU/LRU của tầng $L+1$:
$$\text{Score}(j) = \sum_{i \in I_k^{(L)}} M^{(L)}[i, j] + \left\lfloor \frac{\text{Heat}_{L+1}[j]}{4} \right\rfloor$$
Tập ứng viên nạp trước được lọc ra bởi điều kiện:
$$\text{Candidates} = \operatorname{argtopm} \left( \{ \text{Score}(j) \mid j \notin \text{Cache}_{L2}^{(L+1)} \land j \notin \text{Cache}_{L1}^{(L+1)} \land \text{Score}(j) > 0 \}, \, m \right)$$
Trong đó $m = 2$ là số lượng ứng viên tối ưu được lựa chọn qua thực nghiệm.

### 2.5.3. Nguyên lý Cơ hội (The Opportunistic Principle) để Triệt tiêu Tranh chấp I/O
*Bài học thất bại đắt giá từ Bước 23 trong PROGRESS.md* chỉ ra rằng: Nếu kích hoạt nạp trước tầng $L+1$ một cách ồ ạt ngay trong lúc tầng $L$ đang bị miss và cần đọc đĩa, băng thông NVMe PCIe sẽ bị chia sẻ và bão hòa. Hậu quả là thời gian nạp các chuyên gia cấp bách của tầng $L$ bị kéo dài từ $20\text{ ms}$ lên $45\text{ ms}$, làm giảm thông lượng giải mã tổng thể.

Do đó, đồ án đề xuất **Chiến lược Nạp trước Cơ hội (Opportunistic Prefetching):**
$$\text{Kích hoạt Lookahead } (L+1) \iff S_{miss}^{(L)} = \emptyset \quad (\text{Tầng } L \text{ đạt 100\% Cache Hit})$$
Khi tầng $L$ đạt 100% Hit, đường truyền I/O từ SSD hoàn toàn rảnh rỗi. Việc chiếm dụng kênh truyền rảnh rỗi này để nạp trước cho tầng $L+1$ đạt hiệu quả chi phí bằng 0 (Zero observable cost), giúp tầng $L+1$ kế tiếp tăng tỷ lệ Cache Hit lên thêm **$12.5\% - 25.0\%$**.

---

## 2.6. Điều phối Chuyên gia Song song Cấp thô (Coarse-Grained Task Parallel Dispatching)

Trong các bản dựng ban đầu (từ Bước 1 đến Bước 23), việc tính toán ma trận SwiGLU cho từng chuyên gia sử dụng cơ chế chia nhỏ tác vụ mức vi mô (Fine-Grained Parallelism): 12 luồng CPU của Rayon cùng xâu xé 1,024 hàng ma trận của 1 chuyên gia. Khi chuyên gia đó tính xong, các luồng lại đồng bộ (barrier) và chuyển sang chuyên gia kế tiếp.

### 2.6.1. Chi phí Rào cản Đồng bộ của Fine-Grained Parallelism
* Với 8 chuyên gia mỗi tầng và 16 tầng mạng:
  $$\text{Số lượng rào cản luồng} = 16 \text{ tầng} \times 8 \text{ chuyên gia} \times 2 \text{ pha} = 256 \text{ barriers / token}$$
* Hiện tượng rung lắc bộ nhớ đệm CPU L1/L2: Do 12 lõi liên tục ghi đè dữ liệu của các chuyên gia khác nhau vào bộ nhớ đệm L1/L2 của từng lõi, hiệu suất vector hóa SIMD bị giảm sút do xung đột bộ nhớ đệm (Cache Line Invalidation).

### 2.6.2. Chuyển đổi sang Coarse-Grained Task Parallel Dispatching
Tại Bước 24, hệ thống chuyển dịch hoàn toàn sang kiến trúc song song cấp tác vụ thô: Mỗi chuyên gia trong số 8 chuyên gia được gán độc quyền cho 1 worker thread duy nhất của Rayon chạy từ đầu đến cuối thông qua hàm `forward_single_threaded`.

*Bảng 3.1: So sánh chi phí đồng bộ luồng giữa Fine-Grained Fork-Join và Coarse-Grained Task Parallel.*

| Chỉ số kỹ thuật | Fine-Grained Parallelism (Cũ) | Coarse-Grained Parallelism (Mới) | Mức độ cải thiện |
| :--- | :--- | :--- | :--- |
| **Số rào cản Rayon Barrier / Layer** | 16 barriers | **Đúng 1 barrier duy nhất** | **Giảm 93.75% rào cản** |
| **Số rào cản Rayon / Token** | 256 barriers | **16 barriers** | **Triệt tiêu overhead đồng bộ** |
| **Tranh chấp CPU L1/L2 Data Cache** | Cao (12 lõi tranh nhau 1 expert) | **Bằng 0 (Mỗi lõi xử lý 1 expert độc lập)** | Tối ưu hóa lưu trú L1/L2 |
| **Bộ đệm tạm thời (Scratch Buffer)** | Cấp phát động trong vòng lặp | **Pre-allocated 8 slots (96 KB L2)** | **Zero-Allocation tuyệt đối** |

### 2.6.3. Thuật toán Quy tụ Kết quả Xác định (Deterministic Fused Reduction)
Việc tính toán song song 8 chuyên gia trên các luồng độc lập có thời gian hoàn thành khác nhau. Nếu gom kết quả vào vector ẩn $x$ theo thứ tự luồng hoàn thành trước, phép cộng số thực dấu phẩy động sẽ bị mất tính giao hoán:
$$(a + b) + c \neq a + (b + c) \quad \text{trong chuẩn IEEE 754}$$
Điều này sẽ dẫn tới sai lệch logits qua từng token và phá vỡ tính tương thích số học với PyTorch.

Để khắc phục, hệ thống thiết kế mảng ánh xạ ngăn xếp tĩnh `slot_to_scratch: [usize; 64]`. Bất kể luồng nào hoàn thành trước, kết quả đầu ra luôn được lưu vào đúng ô nhớ tương ứng với vị trí xếp hạng Top-8 ban đầu. Vòng lặp quy tụ (reduction loop) được duyệt tuần tự và tất định từ slot 0 đến slot 7:
```rust
for (slot, &expert_id) in top_experts.iter().enumerate() {
    let scratch_idx = slot_to_scratch[slot];
    let weight = scratch.router[expert_id] / normalizer;
    let out = &scratch.expert_scratches[scratch_idx].output;
    for (a, b) in scratch.x.iter_mut().zip(out.iter().copied()) {
        *a += weight * b;
    }
}
```
Nhờ cơ chế này, hệ thống bảo toàn **100% tính xác định (Bit-Exact Parity)** trên mọi cấu hình đa luồng.

---

# CHƯƠNG 3: HIỆN THỰC HÓA VÀ CÁC KỸ THUẬT TỐI ƯU HÓA MỨC THẤP

Hệ thống `MoE-TierEngine` được hiện thực hóa hoàn toàn bằng ngôn ngữ lập trình hệ thống Rust, tuân thủ nghiêm ngặt các nguyên tắc quản lý bộ nhớ an toàn (Safe Rust), không sử dụng Garbage Collection và triệt tiêu các lỗi truy cập đồng thời tại thời điểm biên dịch.

## 3.1. Kỷ luật Lập trình Hệ thống với Rust (Zero-Allocation & Memory Safety)

### 3.1.1. Kỹ thuật Phân chia Lát cắt Khả biến `split_at_mut`
Một trong những thách thức lớn nhất khi lập trình hệ thống suy luận MoE bằng Rust là tuân thủ quy tắc mượn duy nhất (Aliasing Rule: chỉ cho phép một tham chiếu khả biến `&mut` tại một thời điểm). Trong kỹ thuật nạp trước liên tầng (Lookahead Prefetching), hệ thống cần truy xuất đồng thời trạng thái bộ đệm của tầng hiện tại ($L$) và tầng kế tiếp ($L+1$). Việc gọi `&mut self.layers[L]` và `&mut self.layers[L+1]` sẽ bị trình biên dịch Rust từ chối do vi phạm quy tắc an toàn bộ nhớ.

Thay vì sử dụng các khối `unsafe` tiềm ẩn rủi ro lỗi bộ nhớ, hệ thống áp dụng kỹ thuật chia lát cắt an toàn `split_at_mut`:
```rust
let (current_slice, remaining) = self.layers.split_at_mut(id + 1);
let layer = &mut current_slice[id];
let next_layer_opt = remaining.first_mut();
```
Phương thức này cho phép chia đôi mảng cấu trúc thành hai lát cắt không chồng lấn, cung cấp cho borrow checker bằng chứng toán học rằng hai vùng nhớ hoàn toàn độc lập, đảm bảo an toàn tuyệt đối 100% Safe Rust.

### 3.1.2. Buffer Recycling Pool và POSIX Zero-Copy DMA Pread
Tại Bước 25 trong PROGRESS.md, việc phân tích kernel trace cho thấy khi nạp 893 lượt cache miss, hệ thống đã gọi hàng ngàn lệnh `malloc` và `free` cho các mảng vector trọng số $6.3\text{ MB}$, kích hoạt hơn **1.37 triệu Minor Page Faults** trong nhân Linux. Nguyên nhân là các trang bộ nhớ vật lý mới cấp phát chưa được ánh xạ vào bảng trang (Page Table), buộc kernel phải can thiệp để cấp phát khung trang (page frames) và ghi đè số 0.

Hệ thống đã giải quyết triệt để vấn đề này thông qua cơ chế **Buffer Recycling Pool** kết hợp với **POSIX `read_exact_at` (`pread64`)**:
1. Mỗi tầng duy trì một ngăn xếp `layer.recycled_buffers: Vec<(Vec<u8>, Vec<f32>)>` chứa tối đa 16 bộ đệm đã được "làm ấm" (pre-faulted memory frames).
2. Khi cần nạp expert từ SSD, thay vì cấp phát mới, hệ thống rút một bộ đệm có sẵn từ pool.
3. Sử dụng lệnh gọi hệ thống `nix::sys::uio::pread` để nạp trực tiếp dữ liệu từ file descriptor vào địa chỉ bộ đệm thông qua cơ chế Direct DMA của bộ điều khiển NVMe, không cần qua bộ đệm trung gian của tiến trình (Zero-Copy):
```rust
pub fn read_raw_into(&self, name: &str, buf: &mut [u8]) -> io::Result<()> {
    let tensor = self.find(name)?;
    let file_lock = &self.files[tensor.file_index];
    let file = file_lock.lock().unwrap();
    file.read_exact_at(buf, tensor.data_offset)?;
    Ok(())
}
```
Sau khi chuyên gia hoàn thành nhiệm vụ và bị trục xuất khỏi L2 cache, các mảng bộ nhớ không bị hủy (`drop`) mà được trả lại vào `recycled_buffers`, triệt tiêu hoàn toàn hiện tượng phân mảnh heap và page faults.

### 3.1.3. Kỹ thuật Gối đầu Bộ đệm (Overlapping Scratch Buffer)
Mỗi tệp chuyên gia lưu trữ cả ma trận trọng số INT8 (6.29 MB) và vector hệ số co giãn FP32 (16 KB). Để đọc và chuyển đổi vector hệ số co giãn mà không tốn thêm một byte cấp phát heap nào, hệ thống tận dụng chính 16 KB đầu tiên của mảng `weights` (`&mut weights[..16384]`) làm bộ đệm nhị phân tạm để đọc dữ liệu từ SSD, giải mã sang mảng `scales`, sau đó mới đọc đè toàn bộ dữ liệu trọng số INT8 lên mảng `weights`.

---

## 3.2. Tối ưu hóa Vi kiến trúc CPU & SIMD Vectorization

### 3.2.1. Kernel Tích Vô Hướng `dot_i8_f32` với 4 Ngân Hàng FMA Unrolling
Phép tính toán chiếm hơn 80% thời gian chạy trên CPU là nhân vô hướng giữa hàng trọng số INT8 $Q \in \mathbb{Z}^K$ và vector kích hoạt FP32 $x \in \mathbb{R}^K$. Một vòng lặp tuần tự đơn giản sẽ gặp hiện tượng nghẽn đường ống lệnh (Instruction Pipeline Stall) do chuỗi phụ thuộc dữ liệu của thanh ghi tích lũy.

Hệ thống thiết kế kernel `dot_i8_f32` mở rộng 32 phần tử mỗi bước với 4 ngân hàng cộng dồn FMA độc lập:
```rust
#[inline(always)]
fn dot_i8_f32(row: &[u8], input: &[f32]) -> f32 {
    let mut s0 = [0.0f32; 8];
    let mut s1 = [0.0f32; 8];
    let mut s2 = [0.0f32; 8];
    let mut s3 = [0.0f32; 8];
    let mut qs32 = row.chunks_exact(32);
    let mut xs32 = input.chunks_exact(32);
    for (q, x) in qs32.by_ref().zip(xs32.by_ref()) {
        for lane in 0..8 {
            s0[lane] += (q[lane] as i8 as f32) * x[lane];
            s1[lane] += (q[lane + 8] as i8 as f32) * x[lane + 8];
            s2[lane] += (q[lane + 16] as i8 as f32) * x[lane + 16];
            s3[lane] += (q[lane + 24] as i8 as f32) * x[lane + 24];
        }
    }
    // Thu gọn kết quả từ 4 ngân hàng cộng dồn
    let mut sum = 0.0f32;
    for lane in 0..8 {
        sum += s0[lane] + s1[lane] + s2[lane] + s3[lane];
    }
    sum
}
```

```
Dòng dữ liệu 32 phần tử (32 Bytes INT8 & 128 Bytes FP32)
  ├── Phần tử 00..07 ──► Bank s0 [lane 0..7] (Ánh xạ thanh ghi YMM0/ZMM0)
  ├── Phần tử 08..15 ──► Bank s1 [lane 0..7] (Ánh xạ thanh ghi YMM1/ZMM1)
  ├── Phần tử 16..23 ──► Bank s2 [lane 0..7] (Ánh xạ thanh ghi YMM2/ZMM2)
  └── Phần tử 24..31 ──► Bank s3 [lane 0..7] (Ánh xạ thanh ghi YMM3/ZMM3)
```
*Hình 3.1: Đường ống SIMD unrolling 32-lane với 4 ngân hàng cộng dồn FMA độc lập trong kernel `dot_i8_f32`.*

Việc phân tách thành 4 mảng tích lũy độc lập (`s0`, `s1`, `s2`, `s3`) cho phép trình biên dịch LLVM ánh xạ trực tiếp vào 4 thanh ghi vector SIMD 256-bit (AVX2 `ymm`) hoặc 512-bit (AVX-512 `zmm`), rút ngắn chuỗi phụ thuộc lệnh từ 8 chu kỳ xung nhịp xuống còn 1 chu kỳ, đạt hiệu suất xử lý tối đa của bộ xử lý số học.

### 3.2.2. Kernel Fused Gate-Up-Activation SwiGLU
Trong triển khai chuẩn, phép chiếu Gate và Up được thực hiện tuần tự qua 2 lần duyệt bộ nhớ riêng biệt, ghi kết quả trung gian ra hai vector `gate_buf` và `up_buf`, sau đó duyệt lần thứ 3 để tính toán $\text{SiLU}(\text{gate}) \odot \text{up}$. Điều này dẫn tới hiện tượng trôi dữ liệu (cache eviction) khỏi bộ nhớ đệm L1 CPU.

Kernel `fused_gate_up_swiglu` gộp cả hai phép tính Gate và Up trong cùng một vòng lặp, tái sử dụng vector `input` đang cư trú sẵn trong L1 Cache, tính toán trực tiếp giá trị SiLU inline và ghi ngay vào vector kích hoạt mà không tốn bất kỳ bộ nhớ đệm trung gian nào:
```rust
let compute_lane = |i: usize, target: &mut f32| {
    let gate_row = &gate_weights[i * hidden..(i + 1) * hidden];
    let up_row = &up_weights[i * hidden..(i + 1) * hidden];
    let g = dot_i8_f32(gate_row, input) * gate_scales[i];
    let u = dot_i8_f32(up_row, input) * up_scales[i];
    let silu = g / (1.0 + (-g).exp());
    *target = silu * u;
};
```
Kỹ thuật này giúp cắt giảm **22.5% thời gian tính toán FFN** và giảm thiểu 66% lưu lượng đọc/ghi bộ nhớ nội bộ của CPU.

---

## 3.3. Tích hợp Tăng tốc GPU Lai ghép (Hybrid cuBLAS SGEMM Engine)

Để khai thác tối đa sức mạnh của card đồ họa rời (NVIDIA GeForce RTX 5060 hoặc GTX 1650) có sẵn trên máy tính người dùng mà không bị giới hạn bởi dung lượng VRAM nhỏ, hệ thống phát triển mô hình tính toán lai ghép (Hybrid CPU-GPU Compute Engine).

### 3.3.1. Phân bổ Thác nước (Waterfall Allocation)
Hệ thống tự động thăm dò năng lực bộ nhớ VRAM của GPU thông qua CUDA Driver API:
1. **Nạp Dense Weights:** Toàn bộ các tầng Attention ($W_Q, W_K, W_V, W_O$) và ma trận LM Head (kích thước $2048 \times 32000 \approx 262\text{ MB}$) được ưu tiên nạp vĩnh viễn lên VRAM.
2. **Khai thác cuBLAS SGEMM Batched:** Các phép nhân ma trận dày đặc trong pha Prefill được thực thi thông qua kernel `cublasSgemmStridedBatched`, tận dụng hàng ngàn nhân CUDA Core để giải phóng hoàn toàn gánh nặng cho CPU.
3. **Quản lý Ngữ cảnh Đa luồng CUDA An toàn (CUDA TLS Context Safety):** Trong môi trường máy chủ bất đồng bộ Tokio đa luồng, mỗi worker thread có thể gọi API GPU tại các thời điểm ngẫu nhiên. Nếu ngữ cảnh CUDA Context không được gắn kết đúng với Thread-Local Storage của luồng đó, lệnh gọi `cuMemAlloc` hoặc `cublasCreate` sẽ trả về mã lỗi `CUDA_ERROR_INVALID_CONTEXT`. Hệ thống cài đặt bộ ngắt mạch tự động (Circuit Breaker) và lớp bọc TLS để đảm bảo ngữ cảnh CUDA luôn được kích hoạt trước mỗi lời gọi FFI, đảm bảo an toàn tuyệt đối 100% trong môi trường concurrency.

---

# CHƯƠNG 4: THỰC NGHIỆM, KẾT QUẢ VÀ ĐÁNH GIÁ ĐỐI CHỨNG

Toàn bộ kết quả thực nghiệm trong chương này được trích xuất trực tiếp và trung thực từ **31 tệp dữ liệu benchmark JSON thô** và bảng tổng hợp lịch sử 25 bước trong `docs/PROGRESS.md` cùng `docs/BENCHMARKS.md`. Mọi số liệu đều có khả năng tái lập độc lập 100%.

## 4.1. Thiết lập Môi trường Đo kiểm Đa Phần cứng

Hệ thống được kiểm thử độc lập trên hai cấu hình máy tính cá nhân đại diện cho hai phân khúc người dùng phổ biến:

*Bảng 4.1: Cấu hình phần cứng hai trạm đo kiểm nghiệm thực tế (Testbed A & Testbed B).*

| Thông số phần cứng | Testbed A (Máy tính cá nhân Thành viên 1) | Testbed B (Máy tính cá nhân Thành viên 2) |
| :--- | :--- | :--- |
| **Bộ vi xử lý (CPU)** | Intel Core i7-14650HX (16 nhân / 24 luồng, Turbo 5.2 GHz) | Intel Core i5-10400H (6 nhân / 12 luồng, Turbo 4.6 GHz) |
| **Bộ nhớ trong (RAM)** | 16 GB DDR5 5600 MHz (Băng thông ~60 GB/s) | 16 GB DDR4 2933 MHz (Băng thông ~25-30 GB/s) |
| **Card đồ họa (GPU)** | NVIDIA GeForce RTX 5060 Laptop (8 GB GDDR6) | NVIDIA GeForce GTX 1650 Max-Q (4 GB GDDR5) |
| **Ổ cứng lưu trữ** | SSD NVMe PCIe 4.0 M.2 (Đọc tuần tự ~5000 MB/s) | SSD NVMe PCIe 3.0 M.2 (Đọc tuần tự ~2400 MB/s) |
| **Hệ điều hành** | Linux (Ubuntu 24.04 LTS, Kernel 6.8, Native POSIX) | Linux (Ubuntu 22.04 LTS, Kernel 5.15, Native POSIX) |
| **Số luồng song song** | `RAYON_NUM_THREADS = 8` | `RAYON_NUM_THREADS = 4` |

---

## 4.2. Thẩm định Tính đúng đắn Số học (Numerical Parity)

Trước khi đánh giá tốc độ, tiêu chí tối thượng là đảm bảo chất lượng mô hình không bị suy thoái. Toàn bộ các tầng mạng, từ Embedding, RoPE, Causal Attention đến Router Softmax và SwiGLU FFN đều được kiểm tra đối chiếu từng ma trận kết quả (tensor-by-tensor) với thư viện tham chiếu chuẩn **Hugging Face Transformers 4.51.3** trên nền Python PyTorch.

*Bảng 4.2: Sai số tuyệt đối cực đại của Logits đối chiếu giữa Engine Rust và PyTorch Transformers 4.51.3.*

| Thành phần kiểm thử | Tệp kiểm thử đơn vị | Số lượng test cases | Sai số tuyệt đối lớn nhất ($\max |y_{\text{Rust}} - y_{\text{PyTorch}}|$) | Trạng thái kiểm định |
| :--- | :--- | :---: | :---: | :---: |
| **Multi-Head Attention (MHA)** | `tests/olmoe_reference.rs` | 18 unit tests | $1.19 \times 10^{-6}$ | **ĐẠT (PASS)** |
| **Grouped-Query Attention (GQA)** | `tests/olmoe_reference.rs` | 12 unit tests | $1.43 \times 10^{-6}$ | **ĐẠT (PASS)** |
| **SwiGLU INT8 Expert FFN** | `tests/int8_loader.rs` | 6 integration tests | $1.85 \times 10^{-5}$ | **ĐẠT (PASS)** |
| **End-to-End Output Logits** | `tests/olmoe_reference.rs` | 23 end-to-end tests | **$1.92 \times 10^{-5}$ ($< 2 \times 10^{-5}$)** | **ĐẠT (PASS)** |
| **Toàn bộ Test Suite hệ thống** | `cargo test --all-targets` | **59 / 59 tests** | **0.0 (Bit-Exact Text Output)** | **ĐẠT 100%** |

Ngoài ra, bộ kiểm thử suy luận ngôn ngữ thực tế gồm 12 câu hỏi đa miền (`benchmarks/suites/quality-v1.json`) bao gồm toán học, lập trình Python, kiến thức tổng quát và tiếng Việt đều cho kết quả đầu ra chuẩn xác, dừng chính xác tại token kết thúc câu (`<|endoftext|>`).

---

## 4.3. Đánh giá Hiệu năng Toàn diện qua 25 Mốc Phát triển

Bảng 4.3 tổng hợp quá trình tiến hóa hiệu năng của `MoE-TierEngine` từ phiên bản mộc ban đầu (Baseline) đến phiên bản hoàn thiện hiện tại (Bước 25). Toàn bộ dữ liệu đo kiểm với prompt chuẩn: `"Where is Mars ?"` (hoặc `"What is 2 + 2?"`), sinh 16 tokens.

*Bảng 4.3: Lịch sử biến thiên hiệu năng qua các mốc phát triển trọng yếu.*

| Giai đoạn phát triển | Kỹ thuật cốt lõi áp dụng | TTFT (s) | Decode (tok/s) | Tổng thời gian sinh (s) | Lưu lượng SSD đọc (GB) | Tỷ lệ Cache Hit (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Mốc 1 (Baseline)** | Ngân sách cache 32 MiB, tuần tự CPU | 22.52 | 0.82 | 29.86 | 17.38 | 7.56% |
| **Mốc 2a** | Bỏ xác thực dense lặp lại | 19.21 | 1.01 | 25.16 | 17.38 | 7.56% |
| **Mốc 4a** | Tăng cache 128 MiB/layer (Chống Thrashing) | 13.90 | 1.40 | 18.18 | 7.34 | 60.97% |
| **Mốc 4b** | Heat Cache + Buffer reuse | 12.50 | 1.41 | 16.77 | 4.01 | 78.69% |
| **Mốc 5** | Layer-Major Batched Prefill | 8.54 | 2.21 | 11.26 | 6.29 | 66.56% |
| **Mốc 10** | cuBLAS GPU Dense Offloading | 6.64 | 5.69 | 8.92 | 4.01 | 78.69% |
| **Mốc 14** | L1 VRAM Expert Cache (3-Tier Engine) | 4.60 | 6.20 | 7.15 | 2.50 | 88.50% |
| **Mốc 19** | cuBLAS Batched SGEMM Prefill | 2.45 | 3.88 | 6.57 | 2.50 | 94.90% |
| **Mốc 21** | Hit-First Decoupled Compute-I/O | 3.51 | 3.21 | 8.43 | 8.35 | 79.50% |
| **Mốc 22** | Fused SwiGLU + 32-Lane SIMD | 3.02 | 4.59 | 7.62 | 8.21 | 79.80% |
| **Mốc 23** | Opportunistic Lookahead Prefetching | 2.96 | 3.37 | 7.41 | 8.08 | 80.10% |
| **Mốc 24** | Coarse-Grained Task Parallel Dispatching | 2.48 | 4.14 | 6.10 | 2.08 | 93.00% |
| **Mốc 25 (Hiện tại)** | **Zero-Alloc I/O + POSIX Direct DMA** | **2.48** | **4.40** | **5.89** | **2.08** | **93.00%** |
| **Tăng tốc tổng thể** | **Mốc 25 so với Baseline ban đầu** | **Nhanh gấp 9.08x** | **Nhanh gấp 5.36x** | **Giảm 80.3% thời gian** | **Cắt giảm 88.0% I/O** | **Tăng từ 7.5% lên 93%** |

```
Thời gian phản hồi Token đầu tiên (TTFT - Giây) [Càng thấp càng tốt]
Baseline (Mốc 1)   : ████████████████████████████████ 22.52 s
Sau Prefill Batch  : ████████████ 8.54 s
Sau cuBLAS GPU     : ████ 2.45 s
Hoàn thiện Bước 25 : ████ 2.48 s (Giảm 89.0%)

Tốc độ giải mã (Decode Throughput - tokens/giây) [Càng cao càng tốt]
Baseline (Mốc 1)   : █ 0.82 tok/s
Sau Cache 128MB    : ██ 1.40 tok/s
Sau Prefill Batch  : ███ 2.21 tok/s
Sau Fused SwiGLU   : ██████ 4.59 tok/s
Hoàn thiện Bước 25 : ██████ 4.40 tok/s (CPU Production) / 8.51 tok/s (GPU Warm)
```
*Hình 4.1: Biểu đồ so sánh thời gian phản hồi token đầu tiên (TTFT) và thông lượng giải mã (Decode throughput).*

---

## 4.4. Phân rã Độ trễ (Latency Breakdown)

Để làm rõ đóng góp của từng thành phần kỹ thuật, Bảng 4.4 trình bày chi tiết phân rã thời gian suy luận giữa hai cấu hình ngân sách bộ nhớ đệm: Cấu hình khắc nghiệt ($128\text{ MiB/layer}$) và Cấu hình chuẩn triển khai thực tế ($256\text{ MiB/layer}$).

*Bảng 4.4: Phân rã thời gian suy luận (Latency Breakdown) tại Bước 25.*

| Thành phần thời gian | Ngân sách khắc nghiệt (128 MiB/layer) | Ngân sách chuẩn Production (256 MiB/layer) | Nhận xét kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Prefill TTFT (Warm)** | 2.92 s | **2.48 s** | Tận dụng cuBLAS SGEMM Batched |
| **Expert Compute Time** | 1,594 ms | **1,323 ms** | Giảm mạnh nhờ Fused SwiGLU & SIMD unrolling |
| **Expert I/O Time** | 4,240 ms | **2,533 ms** | Giảm 40.2% nhờ Cache Hit đạt 93% |
| **Causal Attention Time** | 820 ms | **780 ms** | Chạy trực tiếp trên GPU VRAM qua cuBLAS |
| **LM Head Projection** | 240 ms | **210 ms** | Chiếu từ vựng 32k tokens trên VRAM |
| **Tổng thời gian sinh (Total Wall Time)** | 7.75 s | **5.89 s** | **Lần đầu tiên phá vỡ mốc 6 giây** |
| **Số lượt Cache Miss** | 893 misses | **314 misses** | **Cắt giảm 64.8% số lần đọc đĩa** |
| **Dung lượng đọc từ SSD** | 7.90 GB | **2.08 GB** | **Cắt giảm 73.6% băng thông đọc** |
| **Tỷ lệ Cache Hit** | 80.1% | **93.0%** | SF-Cache bảo vệ vững chắc Generalists |

---

## 4.5. Ma trận Đối đầu Thực tế giữa Hai Nền tảng Phần cứng

Bảng 4.5 thể hiện kết quả đo đạc đối đầu trực tiếp giữa Testbed A (i7-14650HX) và Testbed B (Core i5-10400H) khi chạy cùng kịch bản:

*Bảng 4.5: Ma trận đối đầu thực tế giữa hai nền tảng phần cứng Testbed A và Testbed B.*

| Chỉ số đánh giá | Testbed A (Core i7) Cold | Testbed B (Core i5) Cold | Testbed A Warm | Testbed B Warm |
| :--- | :---: | :---: | :---: | :---: |
| **Time to First Token (TTFT)** | 13.90 s | **8.83 s** | 13.89 s | **4.90 s** |
| **Tốc độ Decode (tokens/s)** | 1.40 | **1.85** | 1.28 | **2.31** |
| **Tổng thời gian sinh** | 18.18 s | **16.95 s** | 18.59 s | **11.39 s** |
| **Thời gian Attention** | 3.01 s | **1.98 s** | 3.04 s | **1.46 s** |
| **Thời gian Tính toán Expert** | 8.61 s | **2.89 s** | 8.85 s | **2.43 s** |
| **Thời gian Đọc SSD (I/O)** | 5.41 s | 11.27 s | 5.55 s | 6.84 s |
| **Tỷ lệ Cache Hit (%)** | 60.97% | **68.69%** | 64.38% | **76.39%** |
| **Dung lượng đọc từ SSD** | 7.34 GB | 8.71 GB | 6.70 GB | 6.57 GB |

*Nhận định khoa học:* Trên Testbed B, mặc dù CPU có số nhân ít hơn (6 nhân vs 16 nhân) và ổ SSD PCIe 3.0 chậm hơn, nhưng việc điều phối số luồng Rayon ở mức tối ưu (`RAYON_NUM_THREADS = 4`) giúp hệ thống không gặp hiện tượng bão hòa băng thông RAM DDR4. Điều này chứng minh tính đúng đắn của việc cân chỉnh tỷ lệ song song phù hợp với năng lực phần cứng thực tế.

---

## 4.6. Phân tích Thử nghiệm Thất bại và Bài học Kỹ thuật (Negative Results & Engineering Insights)

Một báo cáo khoa học chân chính không chỉ ghi nhận thành công mà phải phản ánh trung thực các thử nghiệm thất bại và bài học rút ra:

1. **Thất bại với Naive Eager Lookahead (Hiện tượng Prefetch Contention tại Bước 23):**
   * *Hiện tượng:* Ban đầu, hệ thống kích hoạt nạp trước cho tầng $L+1$ bất kể tầng $L$ có bị miss hay không. Việc này dẫn đến việc ổ SSD NVMe phải phục vụ đồng thời cả yêu cầu khẩn cấp của tầng $L$ lẫn yêu cầu suy đoán của tầng $L+1$, gây tắc nghẽn hàng đợi I/O.
   * *Hậu quả:* Thời gian trễ I/O của tầng $L$ tăng gấp đôi, làm giảm thông lượng decode.
   * *Giải pháp:* Đưa ra nguyên lý nạp trước cơ hội (Opportunistic Lookahead): chỉ nạp trước khi tầng hiện tại đạt 100% Cache Hit.
2. **Hiện tượng Bão hòa Băng thông RAM khi Mở rộng Luồng quá mức (Rayon Thread Saturation tại Bước 5/EVIDENCE_MATRIX):**
   * *Hiện tượng:* Thử nghiệm tăng số luồng Rayon lên 24 luồng (bằng số luồng luận lý của Core i7-14650HX).
   * *Kết quả:* Hiệu năng không những không tăng mà còn giảm nhẹ so với 8 luồng. Phân tích phần cứng cho thấy bus bộ nhớ RAM DDR5 (~60 GB/s) đã đạt ngưỡng bão hòa truyền tải dữ liệu giữa các nhân CPU.
   * *Bài học:* Điểm ngọt (sweet spot) của hệ thống là 8 luồng cho Testbed A và 4 luồng cho Testbed B.
3. **Lệch Thứ tự Cộng Dấu Phẩy Động trong Multi-Expert Reduction (Bước 24):**
   * *Hiện tượng:* Khi thu thập kết quả song song từ 8 luồng Rayon, nếu luồng nào xong trước cộng vào trước, sai số làm tròn số thực dấu phẩy động tích lũy qua 16 tầng đã làm trôi dạt logits (`max diff = 0.001`), làm thay đổi từ ngữ được sinh ra.
   * *Giải pháp:* Áp dụng bảng ánh xạ ngăn xếp tĩnh `slot_to_scratch: [usize; 64]` và quy tụ kết quả theo đúng thứ tự Top-K xác định ban đầu, khôi phục tính Bit-Exact Parity tuyệt đối.
4. **Xung đột Ngữ cảnh Đa luồng CUDA Driver Context trong Tokio Runtime (Bước 22-Fix):**
   * *Hiện tượng:* Các worker threads của máy chủ Axum/Tokio khi tiếp nhận request HTTP mới gọi tới thư viện cuBLAS đã gây crash tiến trình với mã lỗi `CUDA_ERROR_INVALID_CONTEXT`.
   * *Giải pháp:* Thiết lập cơ chế Thread-Local Storage (TLS) tự động kiểm tra và kích hoạt `cuCtxSetCurrent` trước mọi lệnh gọi kernel FFI.

---

# KẾT LUẬN VÀ KẾ HOẠCH GIAI ĐOẠN TIẾP THEO (GIAI ĐOẠN 2)

### 1. Tổng kết Kết quả Đạt được trong Giai đoạn 1
Trong Báo cáo Tiến độ Lần 1, toàn bộ các mục tiêu đặt ra cho giai đoạn khởi động và tối ưu hóa nền tảng hệ thống đã được hoàn thành xuất sắc:
1. Xây dựng hoàn chỉnh từ con số 0 một Inference Engine chuyên biệt cho kiến trúc Sparse MoE bằng ngôn ngữ Rust, độc lập hoàn toàn với Python runtime, tuân thủ kỷ luật Zero-Allocation và an toàn bộ nhớ tuyệt đối.
2. Hiện thực thành công kiến trúc bộ nhớ phân tầng 3 cấp (L1 GPU VRAM $\leftrightarrow$ L2 Host RAM $\leftrightarrow$ Secondary NVMe SSD) với cơ chế nạp POSIX Direct DMA `pread64`, triệt tiêu 100% chi phí cấp phát heap động và kernel page faults khi nạp trọng số.
3. Giải quyết triệt để vấn đề I/O Thrashing thông qua thuật toán Layer-Major Expert-Batched Prefill và bộ đệm phân đoạn SF-Cache (70% Protected / 30% Probationary).
4. Khai phá các giải thuật song song hóa tiên tiến: Chồng lấn tính toán và I/O Hit-First, Nạp trước liên tầng Markov Opportunistic Lookahead, và Điều phối đa chuyên gia Coarse-Grained Parallelism.
5. Chứng minh thực nghiệm thuyết phục trên 2 hệ thống phần cứng thực tế: Rút ngắn thời gian sinh phản hồi từ **29.86 giây xuống còn 5.89 giây** (giảm 80.3%), đưa thông lượng giải mã lên **4.40 tokens/s** (CPU production) và **8.51 tokens/s** (GPU warm), bảo toàn 100% tính đúng đắn số học so với PyTorch ($< 2 \times 10^{-5}$) trên toàn bộ 59/59 test suites.

### 2. Kế hoạch và Lộ trình Triển khai Giai đoạn 2
Trên nền tảng vững chắc của Giai đoạn 1, nhóm sinh viên đề xuất kế hoạch nghiên cứu chuyên sâu cho Giai đoạn 2 nhằm nâng cao hơn nữa hiệu năng của hệ thống:
1. **Tối ưu hóa Lệnh Vector cấp phần cứng Intel VNNI (Vector Neural Network Instructions):**
   * Chuyển đổi kernel tích vô hướng `dot_i8_f32` sang sử dụng trực tiếp chỉ thị phần cứng `_mm256_dpbusd_epi32` (AVX2 VNNI / AVX-512 VNNI). Cho phép thực hiện đồng thời 4 phép nhân số nguyên INT8 và cộng dồn vào thanh ghi INT32 trong duy nhất 1 chu kỳ máy, kỳ vọng tăng gấp 2.5x tốc độ tính toán FFN trên CPU.
2. **Hiện thực hóa Cơ chế Flash-Decoding và Paged KV Cache trên Host RAM:**
   * Tái cấu trúc cơ chế quản lý KV Cache theo kiến trúc trang nhớ phân mảnh (PagedAttention), loại bỏ hoàn toàn việc cấp phát mảng tuần tự liên tục, hỗ trợ suy luận ngữ cảnh dài (2048 - 4096 tokens) mà không làm tăng bộ nhớ vật lý.
3. **Nghiên cứu Kỹ thuật Lượng tử hóa Chuyên gia INT4 (AWQ / GPTQ INT4 Expert Offloading):**
   * Nén kích thước của mỗi chuyên gia từ $6.3\text{ MB}$ xuống còn $\approx 3.15\text{ MB}$, cho phép nhân đôi số lượng chuyên gia lưu trú trong L1 VRAM Cache và L2 RAM Cache với cùng một ngân sách phần cứng.
4. **Đóng gói Thư viện và Chuẩn bị Tài liệu Khóa luận Tốt nghiệp:**
   * Chuẩn hóa API theo chuẩn C FFI và đóng gói Docker container tối ưu hóa cho môi trường Edge AI.
   * Hoàn thiện toàn văn quyển Đồ án tốt nghiệp theo quy cách chuẩn của Quyết định 449/QĐ-HV.

---

# TÀI LIỆU THAM KHẢO

### Tiếng Việt:
1. Học viện Công nghệ Bưu chính Viễn thông (2026), *Quy định về Hình thức, Quy cách và Cấu trúc Quyển Đồ án Khóa luận Tốt nghiệp Đại học*, Ban hành kèm theo Quyết định số 449/QĐ-HV ngày 09/03/2026 của Giám đốc Học viện.
2. Nguyễn Kim Khánh (2018), *Giáo trình Kiến trúc Máy tính và Hệ điều hành*, Nhà xuất bản Thông tin và Truyền thông, Hà Nội.

### Tiếng Anh:
3. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", *Advances in Neural Information Processing Systems (NeurIPS)*, 35, pp. 16344-16359.
4. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022), "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", *Advances in Neural Information Processing Systems (NeurIPS)*, 35, pp. 30318-30332.
5. Eliseev, D., & Mazur, D. (2023), "Fast Inference of Mixture-of-Experts Language Models on Constrained Hardware", *arXiv preprint arXiv:2312.17238*.
6. Fedus, W., Zoph, B., & Shazeer, N. (2022), "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", *Journal of Machine Learning Research (JMLR)*, 23(120), pp. 1-39.
7. Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., ... & Sayed, W. E. (2024), "Mixtral of Experts", *arXiv preprint arXiv:2401.04088*.
8. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023), "Efficient Memory Management for Large Language Model Serving with PagedAttention", *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)*, pp. 611-626.
9. Muennighoff, N., Liu, A., Rashkin, H., Morrison, N., Faridi, S., Khandelwal, U., ... & Raffel, C. (2024), "OLMoE: Open Mixture-of-Experts Language Models", *Allen Institute for AI Technical Report*, arXiv preprint arXiv:2409.02060.
10. Shazeer, N. (2020), "GLU Variants Improve Transformer", *arXiv preprint arXiv:2002.05202*.
11. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024), "RoFormer: Enhanced Transformer with Rotary Position Embedding", *Neurocomputing*, 568, pp. 127063.
12. Williams, S., Waterman, A., & Patterson, D. (2009), "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *Communications of the ACM*, 52(4), pp. 65-76.
