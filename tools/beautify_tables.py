#!/usr/bin/env python3
"""
tools/beautify_tables.py
Format lại toàn bộ các bảng trong baocao/latex/chapters/ để giao diện đẹp mắt,
khoảng cách thoáng đãng, số liệu căn chỉnh chuẩn mực và KHÔNG THAY ĐỔI BẤT KỲ DỮ LIỆU NÀO.
"""

import os
import re

CHAPTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baocao", "latex", "chapters")

def beautify_ch5():
    ch5_path = os.path.join(CHAPTERS_DIR, "ch5_thucnghiem.tex")
    with open(ch5_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Bảng 1: Phân loại các nhóm bằng chứng (3 cột)
    # Cũ: \begin{tabularx}{\textwidth}{p{3.5cm} p{4.0cm} X}
    content = content.replace(
        r"\begin{tabularx}{\textwidth}{p{3.5cm} p{4.0cm} X}",
        r"\begin{tabularx}{\textwidth}{>{\RaggedRight\arraybackslash}p{3.2cm} >{\RaggedRight\arraybackslash}p{4.0cm} >{\RaggedRight\arraybackslash}X}"
    )

    # Bảng 2: Môi trường và cấu hình bộ i5 phase1-v1 (2 cột)
    content = content.replace(
        r"\begin{tabularx}{\textwidth}{p{4.5cm} X}",
        r"\begin{tabularx}{\textwidth}{>{\RaggedRight\arraybackslash}p{4.6cm} >{\RaggedRight\arraybackslash}X}"
    )

    # Bảng 3: Từng mẫu của bộ i5 phase1-v1 (7 cột)
    # Nhóm | Lượt | Prompt tokens | Output tokens | TTFT(s) | Decode(steps/s) | Wall(s)
    content = re.sub(
        r'(\\caption\{Từng mẫu của bộ i5 phase1-v1\}\s*\\label\{tab:bang_5_3\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.2cm} >{\\centering\\arraybackslash}p{0.8cm} >{\\centering\\arraybackslash}p{1.3cm} >{\\centering\\arraybackslash}p{1.3cm} >{\\centering\\arraybackslash}p{1.4cm} >{\\centering\\arraybackslash}p{2.2cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 4: Tổng hợp hiệu năng i5 (5 cột)
    # Nhóm | TTFT(s) | Decode(steps/s) | Wall(s) | Wall min–max(s)
    content = re.sub(
        r'(\\caption\{Tổng hợp hiệu năng i5, trung vị của 5 mẫu/nhóm\}\s*\\label\{tab:bang_5_4\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.8cm} >{\\centering\\arraybackslash}p{1.6cm} >{\\centering\\arraybackslash}p{2.6cm} >{\\centering\\arraybackslash}p{1.8cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 5: Chi phí expert, attention và file payload i5 (5 cột)
    # Nhóm | Expert load wait(s) | Expert compute(s) | Attention(s) | File payload(GB)
    content = re.sub(
        r'(\\caption\{Chi phí expert, attention và file payload, trung vị 5 mẫu/nhóm\}\s*\\label\{tab:bang_5_5\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.8cm} >{\\centering\\arraybackslash}p{2.6cm} >{\\centering\\arraybackslash}p{2.6cm} >{\\centering\\arraybackslash}p{2.0cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 6: Quan sát bộ nhớ trong bộ đo chính (5 cột)
    # Scenario | HWM cao nhất quan sát(GiB) | VmSwap cao nhất(KiB) | MemAvailable thấp nhất(GiB) | Thời gian quan sát(s)
    content = re.sub(
        r'(\\caption\{Quan sát bộ nhớ trong bộ đo chính\}\s*\\label\{tab:bang_5_6\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{2.8cm} >{\\centering\\arraybackslash}p{3.0cm} >{\\centering\\arraybackslash}p{2.4cm} >{\\centering\\arraybackslash}p{3.0cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 7: Thí nghiệm inclusive/victim: từng wall mẫu và trung vị (5 cột)
    # Policy | Prompt | Mẫu 1(s) | Mẫu 2(s) | Trung vị(s)
    content = re.sub(
        r'(\\caption\{Thí nghiệm inclusive/victim: từng wall mẫu và trung vị\}\s*\\label\{tab:bang_5_7\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.0cm} >{\\RaggedRight\\arraybackslash}p{3.0cm} >{\\centering\\arraybackslash}p{2.0cm} >{\\centering\\arraybackslash}p{2.0cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 8: Thí nghiệm KV prefix reuse (4 cột)
    # Lượt | Prompt/tình huống | TTFT(s) | Wall(s)
    content = re.sub(
        r'(\\caption\{Thí nghiệm KV prefix reuse\}\s*\\label\{tab:bang_5_8\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\centering\\arraybackslash}p{1.2cm} >{\\RaggedRight\\arraybackslash}p{7.0cm} >{\\centering\\arraybackslash}p{2.5cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 9: Thí nghiệm dispatch prefill theo nhóm (5 cột)
    # Lượt cặp | Prompt | Baseline wall(s) | Chunked wall(s) | Ghi nhận
    content = re.sub(
        r'(\\caption\{Thí nghiệm dispatch prefill theo nhóm\}\s*\\label\{tab:bang_5_9\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\centering\\arraybackslash}p{1.5cm} >{\\RaggedRight\\arraybackslash}p{3.2cm} >{\\centering\\arraybackslash}p{2.4cm} >{\\centering\\arraybackslash}p{2.4cm} >{\\RaggedRight\\arraybackslash}X}',
        content
    )

    # Bảng 10: Hai lượt kiểm RAM có giám sát (3 cột)
    # Thuộc tính/quan sát | reset-32 (mẫu 2) | keep-128 (mẫu 3)
    content = re.sub(
        r'(\\caption\{Hai lượt kiểm RAM có giám sát\}\s*\\label\{tab:bang_5_10\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{5.5cm} >{\\centering\\arraybackslash}p{4.2cm} >{\\RaggedRight\\arraybackslash}X}',
        content
    )

    # Bảng 11: Môi trường và cấu hình thực nghiệm trên máy i7 (2 cột)
    content = re.sub(
        r'(\\caption\{Môi trường và cấu hình thực nghiệm trên máy i7\}\s*\\label\{tab:bang_5_11\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{4.6cm} >{\\RaggedRight\\arraybackslash}X}',
        content
    )

    # Bảng 12: Từng mẫu đo của bộ i7 phase1-v1 (7 cột)
    content = re.sub(
        r'(\\caption\{Từng mẫu đo của bộ i7 phase1-v1[^\}]*\}\s*\\label\{tab:bang_5_12\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.2cm} >{\\centering\\arraybackslash}p{0.8cm} >{\\centering\\arraybackslash}p{1.3cm} >{\\centering\\arraybackslash}p{1.3cm} >{\\centering\\arraybackslash}p{1.4cm} >{\\centering\\arraybackslash}p{2.2cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 13: Tổng hợp hiệu năng i7, trung vị của 5 mẫu/nhóm (5 cột)
    content = re.sub(
        r'(\\caption\{Tổng hợp hiệu năng i7, trung vị của 5 mẫu/nhóm\}\s*\\label\{tab:bang_5_13\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.8cm} >{\\centering\\arraybackslash}p{1.6cm} >{\\centering\\arraybackslash}p{2.6cm} >{\\centering\\arraybackslash}p{1.8cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 14: Phân tích chi phí expert, attention và file payload trên i7 (5 cột)
    content = re.sub(
        r'(\\caption\{Phân tích chi phí expert, attention và file payload trên i7[^\}]*\}\s*\\label\{tab:bang_5_14\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.8cm} >{\\centering\\arraybackslash}p{2.6cm} >{\\centering\\arraybackslash}p{2.6cm} >{\\centering\\arraybackslash}p{2.0cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 15: Quan sát bộ nhớ máy i7 trong bộ đo chính (5 cột)
    content = re.sub(
        r'(\\caption\{Quan sát bộ nhớ máy i7 trong bộ đo chính\}\s*\\label\{tab:bang_5_15\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{2.8cm} >{\\centering\\arraybackslash}p{3.0cm} >{\\centering\\arraybackslash}p{2.4cm} >{\\centering\\arraybackslash}p{3.0cm} >{\\RaggedLeft\\arraybackslash}X}',
        content
    )

    # Bảng 16: Kế hoạch kỳ tiếp theo (3 cột)
    content = re.sub(
        r'(\\caption\{Kế hoạch kỳ tiếp theo\}\s*\\label\{tab:bang_5_16\}\s*)\\begin\{tabularx\}\{\\textwidth\}\{[^\}]+\}',
        r'\1\\begin{tabularx}{\\textwidth}{>{\\RaggedRight\\arraybackslash}p{3.5cm} >{\\RaggedRight\\arraybackslash}p{8.0cm} >{\\centering\\arraybackslash}X}',
        content
    )

    # Thêm \rowcolor{tableheader} vào ngay sau \toprule của mọi bảng nếu chưa có
    content = re.sub(r'(\\toprule\s*\n)(?!\s*\\rowcolor)', r'\1\\rowcolor{tableheader}\n', content)

    # Đưa các dòng \textit{Nguồn: ...} nằm ngoài \end{table} vào bên trong \end{table}
    # Pattern: \end{tabularx}\n\end{table}\textit{Nguồn: ...} -> \end{tabularx}\n\vspace{3pt}\par\noindent\footnotesize\textit{Nguồn: ...}\n\end{table}
    content = re.sub(
        r'\\end\{tabularx\}\s*\n\\end\{table\}\s*\\textit\{(Nguồn:[^\}]+)\}',
        r'\\end{tabularx}\n\\vspace{3pt}\\par\\noindent\\footnotesize\\textit{\1}\n\\end{table}',
        content
    )

    # Pattern có sẵn \vspace: chuẩn hóa
    content = re.sub(
        r'\\end\{tabularx\}\s*\n(\\vspace\{3pt\}[^\n]+)\s*\n\\end\{table\}',
        r'\\end{tabularx}\n\1\n\\end{table}',
        content
    )

    with open(ch5_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("ch5_thucnghiem.tex tables beautified successfully.")

if __name__ == "__main__":
    beautify_ch5()

