#!/usr/bin/env python3
"""
tools/build_all_chapters.py
Tự động trích xuất các chương từ BAO_CAO_DINH_KY_LAN_1.md và dàn trang LaTeX chuẩn QĐ 449.
"""

import os
import re
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", nargs="+", help="Chỉ sinh các tên file chương được chọn; giữ các file khác.")
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(BASE_DIR, "baocao", "BAO_CAO_DINH_KY_LAN_1.md")
LATEX_DIR = os.path.join(BASE_DIR, "baocao", "latex")
CHAPTERS_DIR = os.path.join(LATEX_DIR, "chapters")

with open(MD_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

# Cắt bỏ phần hướng dẫn ban đầu <!-- ... -->
if full_text.startswith("<!--"):
    full_text = full_text.split("-->", 1)[1]

# Chia tài liệu theo các tiêu đề H1 (# ...)
h1_sections = re.split(r'\n(?=# [^\n]+)', full_text)

section_dict = {}
for s in h1_sections:
    s = s.strip()
    if not s:
        continue
    match = re.match(r'^# ([^\n]+)', s)
    if match:
        title = match.group(1).strip()
        section_dict[title] = s

print("Found H1 sections:")
for title in section_dict.keys():
    print(f" - {title}")

# TikZ code cho 4 sơ đồ
TIKZ_HINH_2_1 = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=0.35cm, auto, box/.append style={minimum height=0.65cm}, proc/.append style={minimum height=0.65cm}]
    \node [box, fill=blue!10] (emb) {Đầu vào lớp (embedding ở lớp đầu)};
    \node [box, below=of emb] (attn) {RMSNorm \& Causal Multi-Head Attention};
    \node [box, fill=gray!15, below=of attn] (res1) {Residual Connection (+)};
    \node [box, below=of res1] (norm2) {RMSNorm \& Router (Top-$k$)};
    \node [proc, fill=orange!15, below=of norm2] (select) {Kích hoạt $k=8$ trên $E=64$ Expert};
    \node [box, fill=purple!10, below=of select] (swiglu) {SwiGLU CPU (trọng số INT8)};
    \node [box, fill=gray!15, below=of swiglu] (res2) {Tổng có trọng số $\sum g_i E_i(x)$ \& Residual (+)};
    \node [box, fill=green!10, below=of res2] (out) {Lớp tiếp theo / RMSNorm cuối \& LM Head};

    \draw [arrow] (emb) -- (attn);
    \draw [arrow] (attn) -- (res1);
    \draw [arrow] (emb.west) to [out=180, in=180] (res1.west);
    \draw [arrow] (res1) -- (norm2);
    \draw [arrow] (norm2) -- (select);
    \draw [arrow] (select) -- (swiglu);
    \draw [arrow] (swiglu) -- (res2);
    \draw [arrow] (res1.east) to [out=0, in=0] (res2.east);
    \draw [arrow] (res2) -- (out);
\end{tikzpicture}
\caption{Khối suy luận Transformer-MoE minh họa}
\label{fig:moe_block}
\par\smallskip{\footnotesize\RaggedRight Nguồn: nhóm dựng từ kiến trúc Transformer và OLMoE \cite{vaswani_attention,olmoe}. Hình lược bỏ các norm phụ của Q/K và chi tiết tensor; INT8 chỉ định dạng trọng số expert.\par}
\end{figure}
"""

TIKZ_HINH_3_1 = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[box/.append style={text width=3.9cm, minimum height=0.85cm, align=center}]
    \node [box, fill=blue!10] (ui) at (0,0) {Giao diện web};
    \node [box, fill=blue!5] (http) at (5,0) {HTTP và luồng SSE};
    \node [box, fill=orange!10] (engine) at (10,0) {Điều phối yêu cầu\\InferenceEngine};
    \node [box, fill=yellow!15] (model) at (5,-1.65) {Mô hình OLMoE\\OlmoeModel};
    \node [box, fill=red!10] (guard) at (0,-1.65) {Kiểm tra kế hoạch bộ nhớ\\và bảo vệ RAM};
    \node [box, fill=gray!10] (dense) at (0,-3.55) {Phép chiếu và LM head\\CPU hoặc GPU/cuBLAS};
    \node [box, fill=purple!10] (router) at (5,-3.3) {Router chọn top-$k$};
    \node [box, fill=gray!15] (kv) at (10,-3.3) {Bộ đệm KV và\\tiền tố phiên hội thoại};
    \node [box, fill=purple!20] (expert) at (5,-4.85) {Expert INT8\\Tính trên CPU (Rayon)};
    \node [box, fill=cyan!10] (ram) at (5,-6.4) {Bộ đệm expert RAM và\\vùng đệm tái sử dụng};
    \node [box, fill=green!10] (vram) at (10,-6.4) {Bộ đệm expert VRAM\\khi bật GPU};
    \node [box, fill=red!10] (file) at (0,-6.4) {Tệp trọng số Safetensors};
    \draw [arrow] (ui) -- (http);
    \draw [arrow] (http) -- (engine);
    \draw [arrow] (engine.south) |- (model.east);
    \draw [arrow] (model) -- (guard);
    \draw [arrow] (model.south west) -- (dense.north east);
    \draw [arrow] (model.south east) -- (kv.north west);
    \draw [arrow] (model) -- (router);
    \draw [arrow] (router) -- (expert);
    \draw [arrow] (ram) -- node[right, font=\footnotesize] {trọng số} (expert);
    \draw [arrow] (file) -- (ram);
    \draw [arrow, <->] (ram) -- (vram);
\end{tikzpicture}
\caption{Kiến trúc tổng thể hệ thống MoE-TierEngine}
\label{fig:system_arch}
\par\smallskip{\footnotesize\RaggedRight Nguồn: nhóm dựng theo mã nguồn \cite{moe_tierengine_arch}. Phần trên thể hiện quan hệ điều phối; các liên kết tệp/RAM/VRAM thể hiện đường cung cấp trọng số. Tác vụ nạp có thể dùng bộ đệm tạm trước khi được giữ trong cache. Hình lược bỏ chi tiết nạp trước và đường trả kết quả; không biểu diễn lịch chạy song song.\par}
\end{figure}
"""

TIKZ_HINH_3_2 = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[box/.append style={text width=3.7cm, minimum height=0.9cm, align=center}, decision/.append style={font=\small, inner sep=1pt}]
    \node [box, fill=blue!10] (start) at (0,0) {Nhận expert từ Router};
    \node [decision] (ram) at (0,-1.25) {Có ở RAM?};
    \node [decision] (vram) at (0,-3.35) {VRAM dùng được?};
    \node [box, fill=yellow!15] (download) at (4.7,-3.35) {Truyền từ GPU về RAM};
    \node [box, fill=red!10] (file) at (-4.7,-3.35) {Lấy bản đã nạp trước;\\nếu chưa có thì đọc tệp};
    \node [box, fill=purple!10] (compute) at (0,-5.25) {Tính expert trên CPU};
    \node [box, fill=cyan!10, text width=8.7cm, minimum height=1.4cm] (admit) at (0,-7.25) {Sau khi tính: xét lưu lại expert trong RAM.\\Nếu cần chỗ, loại expert cũ và thử giữ expert cũ ở VRAM trước khi tái sử dụng vùng đệm.\\Nếu GPU đang bật, bỏ bản VRAM tương ứng khi giữ expert vừa dùng trong RAM.};
    \draw [arrow] (start) -- (ram);
    \draw [arrow] (ram) -- node[right] {Không} (vram);
    \draw [arrow] (ram.west) -- node[above] {Có} (-7,-1.25) |- (compute.west);
    \draw [arrow] (vram) -- node[above] {Có} (download);
    \draw [arrow] (vram) -- node[above, pos=0.2, font=\footnotesize] {Không} (file);
    \draw [arrow] (download.south) |- (compute.east);
    \draw [arrow] (file.south) |- (compute.west);
    \draw [arrow] (compute) -- (admit);
\end{tikzpicture}
\caption{Truy xuất và lưu lại expert theo chính sách victim}
\label{fig:tiering_victim}
\par\smallskip{\footnotesize\RaggedRight Nguồn: các đường lấy, tính và lưu lại expert trong mã nguồn \cite{moe_tierengine_arch}. Hình lược bỏ lịch nạp trước và chi tiết xử lý lỗi. Truyền từ VRAM thất bại thì có thể dùng bản đã nạp trước hoặc đọc tệp. Nếu expert không vừa ngân sách RAM, hệ thống không giữ trong cache RAM và có thể thử giữ ở VRAM. Mọi đường VRAM phụ thuộc GPU được bật và tài nguyên khả dụng.\par}
\end{figure}
"""

TIKZ_HINH_3_3 = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[node distance=1.0cm, auto]
    \node [box, fill=blue!10] (user) {Client (Web / Runner)};
    \node [box, fill=orange!10, right=2.5cm of user] (http) {HTTP Facade};
    \node [box, fill=purple!10, right=2.5cm of http] (worker) {Blocking Worker};

    \node [below=5.0cm of user] (u_end) {};
    \node [below=5.0cm of http] (h_end) {};
    \node [below=5.0cm of worker] (w_end) {};

    \draw [dashed] (user) -- (u_end);
    \draw [dashed] (http) -- (h_end);
    \draw [dashed] (worker) -- (w_end);

    % Message sequence
    \draw [arrow] ($(user.south)+(0,-0.4)$) -- node [above, scale=0.8] {POST (messages, session)} ($(http.south)+(0,-0.4)$);
    \draw [arrow] ($(http.south)+(0,-0.9)$) -- node [above, scale=0.8] {Acquire gate, spawn task} ($(worker.south)+(0,-0.9)$);
    \draw [arrow] ($(worker.south)+(0,-1.5)$) -- node [above, scale=0.8] {Worker ready (preflight OK)} ($(http.south)+(0,-1.5)$);
    \draw [arrow] ($(http.south)+(0,-2.0)$) -- node [above, scale=0.8] {SSE Stream Header (200 OK)} ($(user.south)+(0,-2.0)$);
    \draw [arrow] ($(worker.south)+(0,-2.6)$) -- node [above, scale=0.8] {Delta token chunks} ($(http.south)+(0,-2.6)$);
    \draw [arrow] ($(http.south)+(0,-3.1)$) -- node [above, scale=0.8] {Content chunk (SSE)} ($(user.south)+(0,-3.1)$);
    \draw [arrow] ($(worker.south)+(0,-3.7)$) -- node [above, scale=0.8] {Finished / Terminal status} ($(http.south)+(0,-3.7)$);
    \draw [arrow] ($(http.south)+(0,-4.1)$) -- node [above, scale=0.8] {Terminal result (SSE)} ($(user.south)+(0,-4.1)$);
    \draw [arrow] ($(http.south)+(0,-4.6)$) -- node [above, scale=0.8] {data: [DONE]} ($(user.south)+(0,-4.6)$);
\end{tikzpicture}
\caption{Vòng đời xử lý yêu cầu suy luận streaming qua HTTP/SSE}
\label{fig:streaming_lifecycle}
\par\smallskip{\footnotesize\RaggedRight Nguồn: nhóm dựng theo HTTP handler và blocking worker \cite{moe_tierengine_arch}. Endpoint: \nolinkurl{/v1/chat/completions}. Hình mô tả luồng thành công, lược bỏ nhánh lỗi và hủy.\par}
\end{figure}
"""

def process_markdown_text(content):
    """Xử lý cú pháp Markdown thành cú pháp LaTeX tiêu chuẩn."""
    protected_blocks = []
    def protect(text):
        protected_blocks.append(text)
        return f"LATEXPROTBLOCKALPHA{len(protected_blocks)-1}OMEGA"

    # 1. Thay thế trích dẫn [X] và bảo vệ
    cite_bib_map = {
        '1': 'ptit_qd449', '2': 'switch_transformers', '3': 'olmoe',
        '4': 'swiglu', '5': 'roformer', '6': 'vaswani_attention',
        '7': 'rmsnorm', '8': 'nvidia_cublas', '9': 'moe_tierengine_arch',
        '10': 'exp_phase1_i5', '11': 'exp_web_session_i5', '12': 'exp_prefill_i5',
        '13': 'exp_ram_guard_i5', '14': 'moe_progress_matrix', '15': 'oom_log_i5',
        '16': 'exp_smoke_validation', '17': 'bench_protocols', '18': 'exp_phase1_i7',
        '19': 'quantization_survey', '20': 'hf_kv_cache', '21': 'cuda_memory',
        '22': 'linux_proc', '23': 'olmoe_0924_config', '24': 'olmoe_0125_config',
        '25': 'ostep_policies', '26': 'moe_offloading'
    }
    for k, v in cite_bib_map.items():
        content = re.sub(rf"\[{k}, (mục [^\]\n]+)\]", lambda m, key=v: protect("\\cite[" + m.group(1) + "]{" + key + "}"), content)

    for k, v in cite_bib_map.items():
        content = re.sub(rf'\[{k}\]', lambda m, key=v: protect(f"\\cite{{{key}}}"), content)

    # 2. Thay thế các sơ đồ Mermaid bằng TikZ và bảo vệ
    content = re.sub(r'```mermaid\nflowchart TD\s+A\[(?:Token embedding|Đầu vào lớp: embedding ở lớp đầu)\].*?```\n\n\*Hình 2\.1.*?\*', lambda m: protect(TIKZ_HINH_2_1), content, flags=re.DOTALL)
    content = re.sub(r'```mermaid\nflowchart LR\s+UI\[Web local\].*?```\n\n\*Hình 3\.1.*?\*', lambda m: protect(TIKZ_HINH_3_1), content, flags=re.DOTALL)
    content = re.sub(r'```mermaid\nflowchart TD\s+A\[Nhận expert ID.*?```\n\n\*Hình 3\.2.*?\*', lambda m: protect(TIKZ_HINH_3_2), content, flags=re.DOTALL)
    content = re.sub(r'```mermaid\nsequenceDiagram\s+participant U.*?```\n\n\*Hình 3\.3.*?\*', lambda m: protect(TIKZ_HINH_3_3), content, flags=re.DOTALL)

    # 3. Xử lý code blocks và bảo vệ
    def repl_cb_bash(m):
        return protect(f"\\begin{{lstlisting}}[language=bash]\n{m.group(1)}\\end{{lstlisting}}")
    def repl_cb_rust(m):
        return protect(f"\\begin{{lstlisting}}[language=Rust]\n{m.group(1)}\\end{{lstlisting}}")
    def repl_cb_gen(m):
        return protect(f"\\begin{{lstlisting}}\n{m.group(1)}\\end{{lstlisting}}")

    def repl_cb_text(m):
        return protect("\\begin{Verbatim}[samepage=true,breaklines=true,breakanywhere=true,fontsize=\\small,frame=single,framesep=3mm]\n" + m.group(1) + "\\end{Verbatim}")

    content = re.sub(r'^```text[^\S\n]*\n(.*?)^```[^\S\n]*$', repl_cb_text, content, flags=re.DOTALL | re.MULTILINE)
    content = re.sub(r'```bash\n(.*?)```', repl_cb_bash, content, flags=re.DOTALL)
    content = re.sub(r'```rust\n(.*?)```', repl_cb_rust, content, flags=re.DOTALL)
    content = re.sub(r'```json\n(.*?)```', repl_cb_gen, content, flags=re.DOTALL)
    content = re.sub(r'^```[^\S\n]*\n(.*?)^```[^\S\n]*$', repl_cb_gen, content, flags=re.DOTALL | re.MULTILINE)

    # 4. Xử lý inline code `some_code` TRƯỚC KHI chuyển bảng và bảo vệ
    def repl_inline_code(match):
        code_text = match.group(1)
        if len(code_text) >= 20 or "/" in code_text:
            return protect("\\nolinkurl{" + code_text + "}")
        code_text = code_text.replace(r'\_', '_').replace('_', r'\_')
        code_text = code_text.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
        return protect(f"\\texttt{{{code_text}}}")

    content = re.sub(r'`([^`\n]+)`', repl_inline_code, content)

    # 5. Đổi công thức $$ ... \tag{X.Y} $$ sang equation và bảo vệ
    def repl_eq(match):
        body = match.group(1).strip()
        tag_match = re.search(r'\\tag\{([^\}]+)\}', body)
        if tag_match:
            tag = tag_match.group(1)
            body_clean = re.sub(r'\\tag\{[^\}]+\}', '', body).strip()
            return protect(f"\\begin{{equation}}\n\\label{{eq:{tag}}}\n{body_clean}\n\\end{{equation}}")
        return protect(f"\\begin{{equation}}\n{body}\n\\end{{equation}}")

    content = re.sub(r'\$\$(.*?)\$\$', repl_eq, content, flags=re.DOTALL)

    # 6. Bảo vệ các công thức toán inline $...$
    content = re.sub(r'\$([^\$\n]+)\$', lambda m: protect(m.group(0)), content)

    # 7. Chuyển bảng Markdown thành LaTeX tabularx và bảo vệ
    table_pattern = re.compile(r'((?:^[ \t]*\|[^\n]+\|[ \t]*(?:\n|$)){2,})', re.MULTILINE)
    def repl_table(match):
        raw_table = match.group(1).strip()
        lines = [l.strip() for l in raw_table.split('\n') if l.strip()]
        if len(lines) < 2:
            return raw_table
        headers = [c.strip() for c in lines[0].strip('|').split('|')]
        num_cols = len(headers)
        if num_cols == 2:
            spec = r"p{4.5cm} X"
        elif num_cols == 3:
            spec = r"p{3.5cm} p{4cm} X"
        elif num_cols == 4:
            spec = r"p{2.5cm} p{3.5cm} p{3.5cm} X"
        else:
            spec = "X " * num_cols
            
        spec = re.sub(r"p\{[^}]+\}|X", lambda m: r">{\RaggedRight\arraybackslash}" + m.group(0), spec)
        tex = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", f"\\begin{{tabularx}}{{\\textwidth}}{{{spec}}}", r"\toprule"]
        h_row = " & ".join([f"\\textbf{{{h}}}" for h in headers]) + r" \\"
        tex.append(h_row)
        tex.append(r"\midrule")
        for r_line in lines[2:]:
            cols = [c.strip() for c in r_line.strip('|').split('|')]
            while len(cols) < num_cols:
                cols.append("")
            cols = cols[:num_cols]
            # Escape _ và % trong các ô bảng nếu chưa escape
            escaped_cols = []
            for c in cols:
                # Nếu c không phải token bảo vệ thì escape _ và %
                if not "LATEXPROTBLOCKALPHA" in c:
                    c = c.replace(r'\_', '_').replace('_', r'\_')
                    c = c.replace(r'\%', '%').replace('%', r'\%')
                    c = c.replace(r'\&', '&').replace('&', r'\&')
                escaped_cols.append(c)
            tex.append(" & ".join(escaped_cols) + r" \\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabularx}")
        tex.append(r"\end{table}")
        return "\n" + protect("\n".join(tex)) + "\n"

    content = table_pattern.sub(repl_table, content)

    # 8. Đổi các đề mục:
    # LaTeX tự đánh số; không giữ tiền tố Markdown gây lặp "1.1 1.1".
    content = re.sub(r'^##\s+[0-9]+\.[0-9]+\.?\s+([^\n]+)', r'\\section{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^###\s+[0-9]+\.[0-9]+\.[0-9]+\.?\s+([^\n]+)', r'\\subsection{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^####\s+[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\.?\s+([^\n]+)', r'\\subsubsection{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^##\s+(PHỤ LỤC [A-Z]\. [^\n]+)', lambda m: protect('\\section*{' + m.group(1) + '}'), content, flags=re.MULTILINE)

    # 9. In đậm và in nghiêng
    content = re.sub(r'\*\*([^\*]+)\*\*', lambda m: f"\\textbf{{{m.group(1)}}}", content)
    content = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', lambda m: f"\\textit{{{m.group(1)}}}", content)

    # 10. Escape các ký tự đặc biệt còn lại trên text văn xuôi thường
    content = re.sub(r'(?<!\\)%', r'\%', content)
    content = re.sub(r'(?<!\\)_', r'\_', content)
    content = re.sub(r'(?<!\\)&', r'\&', content)
    content = re.sub(r'(?<!\\)#', r'\#', content)

    # 11. Khôi phục tất cả protected blocks (lặp nhiều lượt cho đến khi giải nén hết các khối lồng nhau)
    max_passes = 10
    passes = 0
    while "LATEXPROTBLOCKALPHA" in content and passes < max_passes:
        for idx in range(len(protected_blocks) - 1, -1, -1):
            content = content.replace(f"LATEXPROTBLOCKALPHA{idx}OMEGA", protected_blocks[idx])
        passes += 1

    return content

print("Processor loaded successfully.")

# Trích xuất từng chương và lưu vào file
chapter_mapping = [
    ("CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI VÀ BÀI TOÁN NGHIÊN CỨU", "ch1_tongquan.tex", "TỔNG QUAN ĐỀ TÀI VÀ BÀI TOÁN NGHIÊN CỨU"),
    ("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ LIÊN QUAN", "ch2_lythuyet.tex", "CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ LIÊN QUAN"),
    ("CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG", "ch3_thietke.tex", "PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG"),
    ("CHƯƠNG 4. THUẬT TOÁN VÀ CÁC GIẢI PHÁP KHẢO SÁT", "ch4_thuattoan.tex", "THUẬT TOÁN VÀ CÁC GIẢI PHÁP KHẢO SÁT"),
    ("CHƯƠNG 5. THIẾT KẾ THỰC NGHIỆM VÀ KẾT QUẢ BƯỚC ĐẦU", "ch5_thucnghiem.tex", "THIẾT KẾ THỰC NGHIỆM VÀ KẾT QUẢ BƯỚC ĐẦU"),
    ("KẾT LUẬN, KIẾN NGHỊ", "ketluan.tex", "KẾT LUẬN VÀ KIẾN NGHỊ"),
    ("PHỤ LỤC", "phuluc.tex", "PHỤ LỤC"),
]

for section_key, filename, chapter_title in chapter_mapping:
    if args.only and filename not in args.only:
        continue
    raw_content = ""
    for k, v in section_dict.items():
        if section_key in k:
            raw_content = v
            break
            
    if not raw_content:
        print(f"WARNING: Section '{section_key}' not found!")
        continue

    # Loại bỏ dòng tiêu đề H1 gốc (# CHƯƠNG X...)
    lines = raw_content.split('\n')
    body_lines = lines[1:] if lines[0].startswith('# ') else lines
    body = '\n'.join(body_lines)

    # Chuyển đổi nội dung
    processed_body = process_markdown_text(body)
    # Gắn tên/nguồn vào float để không tách khỏi bảng khi LaTeX dời trang.
    def attach_caption(match):
        title, table, source = match.groups()
        table = table.replace(r"\centering", r"\centering" + "\n" + r"\caption{" + title + "}", 1)
        return table.replace(r"\end{table}", "\n" + (source or "").strip() + "\n" + r"\end{table}") + "\n\n"
    processed_body = re.sub(
        r"\\textbf\{Bảng \d+\.\d+\. ([^\n]+)\}\s*\n(\\begin\{table\}.*?\\end\{table\})(?:\s*(\\textit\{Nguồn:[^\n]*\}))?",
        attach_caption, processed_body, flags=re.DOTALL,
    )

    # Danh sách đánh số Markdown cần item để không dính thành một đoạn.
    def numbered_list(match):
        items = [re.sub(r"^\d+\.\s+", r"\\item ", line) for line in match.group(0).strip().splitlines()]
        return r"\begin{enumerate}" + "\n" + "\n".join(items) + "\n" + r"\end{enumerate}" + "\n"
    # Không chuyển các dòng đánh số nằm trong giả mã/verbatim thành lệnh LaTeX.
    parts = re.split(r"(\\begin\{(?:Verbatim|lstlisting)\}.*?\\end\{(?:Verbatim|lstlisting)\})", processed_body, flags=re.DOTALL)
    processed_body = "".join(
        part if i % 2 else re.sub(r"(?:^\d+\. [^\n]+\n)+", numbered_list, part, flags=re.MULTILINE)
        for i, part in enumerate(parts)
    )

    def bulleted_list(match):
        items = [r"\item " + line[2:] for line in match.group(0).strip().splitlines()]
        return r"\begin{itemize}" + "\n" + "\n".join(items) + "\n" + r"\end{itemize}" + "\n"
    parts = re.split(r"(\\begin\{(?:Verbatim|lstlisting)\}.*?\\end\{(?:Verbatim|lstlisting)\})", processed_body, flags=re.DOTALL)
    processed_body = "".join(
        part if i % 2 else re.sub(r"(?:^- [^\n]+\n)+", bulleted_list, part, flags=re.MULTILINE)
        for i, part in enumerate(parts)
    )

    # Tạo tiêu đề chương
    if "KẾT LUẬN" in section_key:
        header = f"\\chapter*{{{chapter_title}}}\n\\addcontentsline{{toc}}{{chapter}}{{{chapter_title}}}\n\\markboth{{{chapter_title}}}{{{chapter_title}}}\n\n"
    elif "PHỤ LỤC" in section_key:
        header = f"\\chapter*{{{chapter_title}}}\n\\addcontentsline{{toc}}{{chapter}}{{{chapter_title}}}\n\\markboth{{{chapter_title}}}{{{chapter_title}}}\n\n"
    else:
        header = f"\\chapter{{{chapter_title}}}\n\n"

    out_file = os.path.join(CHAPTERS_DIR, filename)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(header + processed_body)
    print(f"Generated: {out_file}")

print("All chapters generated successfully.")
