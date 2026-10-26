#!/usr/bin/env python3
"""
tools/convert_report_to_latex.py
Chuyển đổi toàn diện BAO_CAO_DINH_KY_LAN_1.md thành cấu trúc LaTeX theo chuẩn QĐ 449/QĐ-HV.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(BASE_DIR, "baocao", "BAO_CAO_DINH_KY_LAN_1.md")
LATEX_DIR = os.path.join(BASE_DIR, "baocao", "latex")
CHAPTERS_DIR = os.path.join(LATEX_DIR, "chapters")

os.makedirs(CHAPTERS_DIR, exist_ok=True)

# Bản đồ ánh xạ trích dẫn sang BibTeX key
CITE_MAP = {
    r'\[1\]': r'\cite{ptit_qd449}',
    r'\[2\]': r'\cite{switch_transformers}',
    r'\[3\]': r'\cite{olmoe}',
    r'\[4\]': r'\cite{swiglu}',
    r'\[5\]': r'\cite{roformer}',
    r'\[6\]': r'\cite{vaswani_attention}',
    r'\[7\]': r'\cite{rmsnorm}',
    r'\[8\]': r'\cite{nvidia_cublas}',
    r'\[9\]': r'\cite{moe_tierengine_arch}',
    r'\[10\]': r'\cite{exp_phase1_i5}',
    r'\[11\]': r'\cite{exp_web_session_i5}',
    r'\[12\]': r'\cite{exp_prefill_i5}',
    r'\[13\]': r'\cite{exp_ram_guard_i5}',
    r'\[14\]': r'\cite{moe_progress_matrix}',
    r'\[15\]': r'\cite{oom_log_i5}',
    r'\[16\]': r'\cite{exp_smoke_validation}',
    r'\[17\]': r'\cite{bench_protocols}',
}

def escape_latex(text):
    """Escape các ký tự đặc biệt của LaTeX ngoài toán và lệnh."""
    # Không escape trong môi trường toán hoặc code
    parts = re.split(r'(\$\$.*?\$\$|\$.*?\$|\\cite\{.*?\}|\\ref\{.*?\})', text, flags=re.DOTALL)
    for i in range(len(parts)):
        if not (parts[i].startswith('$') or parts[i].startswith('\\cite') or parts[i].startswith('\\ref')):
            # Escape & % # _
            # Lưu ý không phá vỡ các macro đã sinh
            p = parts[i]
            p = p.replace('&', r'\&')
            p = p.replace('%', r'\%')
            p = p.replace('#', r'\#')
            p = re.sub(r'(?<!\\)_', r'\_', p)
            parts[i] = p
    return ''.join(parts)

def convert_md_table_to_latex(table_text, caption="", label=""):
    """Chuyển đổi bảng Markdown sang LaTeX tabularx."""
    lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
    if len(lines) < 2:
        return ""
    
    # Parse header
    header_cols = [c.strip() for c in lines[0].strip('|').split('|')]
    num_cols = len(header_cols)
    
    # Alignments
    # Dùng X cho cột nội dung dài, l/c/r cho cột ngắn
    col_spec = "l " + "X " * (num_cols - 1) if num_cols > 1 else "X"
    if num_cols >= 4:
        col_spec = "l " * (num_cols - 1) + "X"
    if num_cols == 2:
        col_spec = "p{4.5cm} X"
    if num_cols == 3:
        col_spec = "p{3.5cm} p{4cm} X"

    latex = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        latex.append(f"\\caption{{{caption}}}")
    if label:
        latex.append(f"\\label{{{label}}}")
    latex.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    latex.append(r"\toprule")
    
    # Header row
    escaped_headers = [f"\\textbf{{{escape_latex(h)}}}" for h in header_cols]
    latex.append(" & ".join(escaped_headers) + r" \\")
    latex.append(r"\midrule")
    
    # Body rows (bỏ qua separator ở lines[1])
    for line in lines[2:]:
        cols = [c.strip() for c in line.strip('|').split('|')]
        # Đảm bảo đủ số cột
        while len(cols) < num_cols:
            cols.append("")
        cols = cols[:num_cols]
        escaped_cols = [escape_latex(c) for c in cols]
        latex.append(" & ".join(escaped_cols) + r" \\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabularx}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

print("Convert tool initialized.")

