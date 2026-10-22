#!/usr/bin/env python3
"""Validate immutable benchmark provenance and render its deterministic catalog.

Raw artifacts are never rewritten. The manifest records historical attribution;
that attribution alone does not establish comparable experimental conditions.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent.parent
KINDS = {"generation", "suite", "quality", "http", "microbenchmark", "unknown"}
TESTBEDS = {"i5", "i7", "unknown"}
STATUSES = {"historical_unverified"}
FIELDS = {"path", "sha256", "kind", "testbed", "status", "aliases"}


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def classify(data):
    """Classify shapes without assigning hardware or manufacturing measurements."""
    if not isinstance(data, dict):
        return "unknown"
    if "pass_rate" in data and "results" in data:
        return "quality"
    if "checks" in data or "fresh_second_turn" in data:
        return "http"
    if "iterations_per_sample" in data:
        return "microbenchmark"
    if "suite" in data and "runs" in data:
        return "suite"
    if "runs" in data and ("prompt" in data or "max_tokens" in data):
        return "generation"
    return "unknown"


def safe_relative(value):
    return (isinstance(value, str) and bool(value)
            and "\\" not in value and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts
            and PurePosixPath(value).as_posix() == value)


def validate_manifest(manifest, root):
    """Return actionable errors; historical aliases need not still exist."""
    root = Path(root).resolve()
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "artifacts"}:
        return ["manifest requires exactly schema_version and artifacts"]
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        return ["schema_version must be 1"]
    if not isinstance(manifest["artifacts"], list):
        return ["artifacts must be an array"]
    errors, paths, hashes, aliases = [], set(), set(), set()
    for index, artifact in enumerate(manifest["artifacts"]):
        prefix = f"artifacts[{index}]"
        if not isinstance(artifact, dict) or set(artifact) != FIELDS:
            errors.append(f"{prefix}: requires exactly {', '.join(sorted(FIELDS))}")
            continue
        path = artifact["path"]
        if not safe_relative(path):
            errors.append(f"{prefix}: unsafe relative path")
            continue
        if path in paths:
            errors.append(f"{path}: duplicate canonical path")
        paths.add(path)
        digest = artifact["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append(f"{path}: sha256 must be 64 lowercase hex characters")
        elif digest in hashes:
            errors.append(f"{path}: duplicate artifact content")
        else:
            hashes.add(digest)
        for field, choices in [("kind", KINDS), ("testbed", TESTBEDS), ("status", STATUSES)]:
            if not isinstance(artifact[field], str) or artifact[field] not in choices:
                errors.append(f"{path}: unsupported {field}: {artifact[field]!r}")
        if not isinstance(artifact["aliases"], list):
            errors.append(f"{path}: aliases must be an array")
        else:
            for alias in artifact["aliases"]:
                if not safe_relative(alias):
                    errors.append(f"{path}: unsafe alias")
                elif alias in aliases:
                    errors.append(f"{path}: duplicate alias {alias}")
                else:
                    aliases.add(alias)
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            errors.append(f"{path}: resolved path escapes repository")
            continue
        try:
            payload = target.read_bytes()
            data = json.loads(payload)
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: cannot read JSON: {exc}")
            continue
        if hashlib.sha256(payload).hexdigest() != digest:
            errors.append(f"{path}: sha256 mismatch")
        if classify(data) != artifact["kind"]:
            errors.append(f"{path}: kind must be {classify(data)!r} for recorded JSON shape")
    for collision in sorted(paths & aliases):
        errors.append(f"{collision}: alias collides with canonical path")
    return errors


def render_report(manifest, root):
    artifacts = sorted(manifest["artifacts"], key=lambda item: item["path"])
    lines = ["# Danh mục bằng chứng benchmark", "",
             "Tạo bằng `python3 tools/export_benchmark_evidence.py`; kiểm tra bằng `--check`.", "",
             f"Có **{len(artifacts)} artifact thô duy nhất** trong `benchmarks/manifest.json`. SHA-256 bảo vệ tính toàn vẹn sau khi lập danh mục, không xác minh nguồn gốc phép đo.", "",
             "`historical_unverified`: dữ liệu lịch sử được giữ nguyên; chưa tái chạy hoặc xác minh đầy đủ provenance. `i5`/`i7` là phân loại lịch sử theo danh mục, không phải chứng nhận phần cứng. `unknown` là chưa đủ bằng chứng gán máy.", "",
             "Không có benchmark mới trong bước chuẩn hóa. Không suy ra tăng tốc hoặc so sánh hai máy từ danh mục này. Byte expert là payload reader, không phải số byte SSD vật lý; engine-cold không đồng nghĩa OS page cache cold.", "",
             "Các artifact lịch sử chưa có bộ metadata provenance chuẩn: commit và dirty state tại lúc chạy, lệnh đầy đủ, hash checkpoint, nhận diện phần cứng/driver thực tế, trạng thái OS cache và điều kiện tải nền. Giá trị thiếu là **chưa ghi nhận**, không thay bằng 0. Cần phép đo mới theo cùng giao thức trước khi so sánh i5/i7.", "",
             "| Artifact thô | Loại | Máy (phân loại lịch sử) | Trạng thái | SHA-256 |", "| --- | --- | --- | --- | --- |"]
    for artifact in artifacts:
        path = artifact["path"]
        lines.append(f"| [{Path(path).name}](../{path}) | {artifact['kind']} | {artifact['testbed']} | {artifact['status']} | `{artifact['sha256']}` |")
    lines += ["", "Alias trong manifest ghi đường dẫn cũ sau khi bỏ bản sao trùng byte. Dữ liệu raw là nguồn gốc; bảng này không trích hay làm tròn số đo.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifest.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/BENCHMARKS.md")
    parser.add_argument("--check", action="store_true", help="Check integrity and report freshness without writing")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        errors = validate_manifest(manifest, args.root)
        if not errors:
            recorded = {item["path"] for item in manifest["artifacts"]}
            actual = {p.relative_to(args.root).as_posix() for p in (args.root / "benchmarks/results").rglob("*.json")}
            if recorded != actual:
                errors.append(f"raw catalog coverage mismatch: missing={sorted(actual - recorded)}, absent={sorted(recorded - actual)}")
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        report = render_report(manifest, args.root)
        if args.check:
            if not args.output.exists() or args.output.read_text(encoding="utf-8") != report:
                print(f"stale or missing report: {args.output}", file=sys.stderr)
                return 1
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        print(f"Validated {len(manifest['artifacts'])} unique evidence artifacts")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Evidence catalog error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
