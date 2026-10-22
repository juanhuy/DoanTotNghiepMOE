#!/usr/bin/env python3
"""Shared local/CI entry point. Default checks use Python stdlib, no model/GPU."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent


def project_errors(root=ROOT):
    errors = []
    documents = [root / name for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md")]
    documents += list((root / "docs").rglob("*.md"))
    documents += list((root / "benchmarks").rglob("README.md"))
    documents += list((root / ".agents/skills").rglob("SKILL.md"))
    for path in documents:
        text = path.read_text(encoding="utf-8")
        # Check Markdown link destinations, not historical shell examples or prose.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for link in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
            link = link.strip().strip("<>")
            parsed = urlsplit(link)
            if parsed.scheme in ("http", "https", "mailto") or not parsed.path:
                continue
            if parsed.scheme or parsed.path.startswith("/"):
                errors.append(f"{path.relative_to(root)}: nonportable link {link}")
            elif not (path.parent / unquote(parsed.path)).exists():
                errors.append(f"{path.relative_to(root)}: broken link {link}")
    for path in (root / ".codex").rglob("*.toml"):
        try:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
            if path.parent.name == "agents":
                if set(config) != {"name", "description", "developer_instructions"}:
                    errors.append(f"{path.relative_to(root)}: unsupported project role fields")
                elif not all(isinstance(value, str) and value.strip() for value in config.values()):
                    errors.append(f"{path.relative_to(root)}: empty role field")
        except (OSError, ValueError) as exc:
            errors.append(f"{path.relative_to(root)}: invalid TOML: {exc}")
    skills = list((root / ".agents/skills").glob("*/SKILL.md"))
    if {path.parent.name for path in skills} != {"moe-engine", "moe-benchmark"}:
        errors.append("Expected the two maintained project skills")
    for path in skills:
        # Deliberately narrow subset: two plain scalar YAML fields, no YAML dependency.
        text = path.read_text(encoding="utf-8")
        pieces = text.split("---", 2)
        if len(pieces) != 3 or pieces[0] != "":
            errors.append(f"{path.relative_to(root)}: missing skill frontmatter")
            continue
        lines = pieces[1].strip().splitlines()
        if len(lines) != 2 or any(": " not in line for line in lines):
            errors.append(f"{path.relative_to(root)}: expected name/description scalar fields")
            continue
        fields = dict(line.split(": ", 1) for line in lines)
        if (set(fields) != {"name", "description"}
                or fields.get("name") != path.parent.name or not fields.get("description")):
            errors.append(f"{path.relative_to(root)}: invalid skill identity")
    machines = json.loads((root / "benchmarks/machines.json").read_text(encoding="utf-8"))
    if machines.get("schema_version") != 1 or set(machines.get("machines", {})) != {"i5", "i7"}:
        errors.append("Machine profiles must declare exactly i5 and i7")
    return errors


def run(command):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--converter", action="store_true")
    parser.add_argument("--web", action="store_true", help="Browser SSE protocol tests (requires Node.js)")
    args = parser.parse_args(argv)
    errors = project_errors()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    commands = [
        [sys.executable, "tools/export_benchmark_evidence.py", "--check"],
        [sys.executable, "tools/test_benchmark_evidence.py"],
        [sys.executable, "tools/test_run_benchmark.py"],
    ]
    if args.rust:
        commands += [["cargo", "fmt", "--all", "--", "--check"],
                     ["cargo", "clippy", "--all-targets", "--locked", "--", "-D", "warnings"],
                     ["cargo", "test", "--all-targets", "--locked"]]
    if args.release:
        commands.append(["cargo", "test", "--release", "--all-targets", "--locked"])
    if args.converter:
        commands.append([sys.executable, "tools/test_prepare_olmoe.py"])
    if args.web:
        commands.append(["node", "tools/test_chat_stream.cjs"])
    if not all(run(command) for command in commands):
        return 1
    if args.http:
        if not run(["cargo", "build", "--release", "--locked", "--bin", "moe-tier-engine"]):
            return 1
        with tempfile.TemporaryDirectory(prefix="moe-http-check-") as output:
            report = Path(output) / "stress.json"
            if not run([sys.executable, "tools/stress_http.py",
                        "tests/fixtures/olmoe/unnormalized-mha", str(report), "30"]):
                return 1
            data = json.loads(report.read_text(encoding="utf-8"))
            print(f"HTTP fixture: {data['completed']} completed, {data['cancelled']} cancelled; "
                  f"steady RSS spread {data['rss_kib']['steady_spread']} KiB")
    print("Selected checks passed. Unselected CUDA/converter/model benchmarks are not validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
