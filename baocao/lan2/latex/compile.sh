#!/usr/bin/env bash
# Script biên dịch nhanh báo cáo LaTeX theo chuẩn XeLaTeX + BibTeX
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=== [1/4] Chạy XeLaTeX lần 1 (khởi tạo cấu trúc) ==="
xelatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "=== [2/4] Chạy BibTeX (nạp tài liệu tham khảo) ==="
bibtex main > /dev/null

echo "=== [3/4] Chạy XeLaTeX lần 2 (cập nhật Mục lục, Bảng, Hình) ==="
xelatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "=== [4/4] Chạy XeLaTeX lần 3 (chốt số trang và liên kết) ==="
xelatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "===> Hoàn tất! File PDF đã được xuất tại: $DIR/main.pdf"

