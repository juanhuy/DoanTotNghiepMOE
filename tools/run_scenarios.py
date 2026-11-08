#!/usr/bin/env python3
"""Ready HTTP/SSE experiments, executed sequentially through run_benchmark.py."""
import subprocess
import sys

import run_benchmark as common


def parser():
    cli = common.parser()
    cli.description = __doc__
    cli.add_argument("scenario", choices=("cold", "warm", "conversation", "topics", "long-context", "cache-sweep"))
    cli.add_argument("--single-policy", dest="compare_cache_policies", action="store_false",
                     help="Measure only --expert-cache-policy (the default; kept for old commands)")
    cli.add_argument("--cache-levels", type=common.nonnegative, nargs="+",
                     help="Explicit ordered MiB/layer budgets; required for cache-sweep")
    cli.set_defaults(runs=5, max_tokens=128, cache=64, compare_cache_policies=False)
    return cli


def commands(args):
    if args.compare_lookahead:
        raise ValueError("use run_benchmark.py --http-session --http-scenario topics --compare-lookahead")
    if args.suite or args.case or args.http_scenario:
        raise ValueError("scenario runner does not accept --suite/--case/--http-scenario")
    if args.prompt != "What is 2 + 2?" or args.tag != "chat":
        raise ValueError("scenario runner fixes prompts/tags; use run_benchmark.py for custom workloads")
    scenario = "conversation" if args.scenario == "cache-sweep" else args.scenario
    if args.scenario == "cache-sweep":
        if not args.cache_levels:
            raise ValueError("cache-sweep requires --cache-levels; choose budgets after RAM preflight")
        caches = args.cache_levels
    else:
        if args.cache_levels is not None:
            raise ValueError("--cache-levels is only valid for cache-sweep")
        caches = (args.cache,)
    result = []
    for cache in caches:
        command = [sys.executable, str(common.WORKSPACE_DIR / "tools/run_benchmark.py"),
                   "--http-session", "--http-scenario", scenario,
                   "--testbed", args.testbed, "--gpu", args.gpu,
                   "--model-dir", str(args.model_dir), "--output-root", str(args.output_root),
                   "--cache", str(cache), "--context", str(args.context),
                   "--max-tokens", str(args.max_tokens), "--runs", str(args.runs),
                   "--expert-cache-policy", args.expert_cache_policy,
                   "--lookahead", args.lookahead,
                   "--tag", f"{args.scenario}-{cache}"]
        if args.threads:
            command += ["--threads", str(args.threads)]
        for flag in ("compare_cache_policies", "allow_dirty", "dry_run"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        result.append(command)
    return result


def execute(args):
    # Stop on failure/OOM; never start another model while a child is running.
    for command in commands(args):
        completed = subprocess.run(command, cwd=common.WORKSPACE_DIR, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    try:
        sys.exit(execute(parser().parse_args()))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
