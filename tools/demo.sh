#!/usr/bin/env bash
# Start the web demo with a named preset; Ctrl+C stops it, rerun with another preset to switch.
#
#   tools/demo.sh PRESET [--cache MIB] [--port PORT] [--context N] [--model-dir DIR] [--ram-limit-mib MIB]
#
# Presets (cache defaults to 64 MiB/layer, the measured configuration for both models):
#   olmoe-gpu   OLMoE-1B-7B, GPU auto + VRAM expert tier + GPU expert compute (README demo config)
#   olmoe-cpu   OLMoE-1B-7B, CPU only
#   qwen3       Qwen3-30B-A3B, CPU only, dense BF16, RAM guard 9GiB (measured on i5)
#   qwen3-gpu   Qwen3-30B-A3B, GPU auto + tier + compute (needs ~5GB free VRAM, e.g. i7 RTX 5060 8GB)
#
# Model directories: --model-dir, or OLMOE_DIR / QWEN3_DIR, else the first existing default.
# The engine's RAM guard refuses a configuration that does not fit (e.g. a too large cache):
# the server then exits with a "RAM guard" message instead of risking an OOM.
set -euo pipefail

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[[ $# -ge 1 ]] || usage 1
preset=$1; shift
[[ $preset == -h || $preset == --help ]] && usage 0
cache=64 port=8080 context=512 model_dir="" ram_limit_mib=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --cache) cache=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --context) context=$2; shift 2 ;;
    --model-dir) model_dir=$2; shift 2 ;;
    --ram-limit-mib) ram_limit_mib=$2; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done
[[ $cache =~ ^[0-9]+$ && $port =~ ^[0-9]+$ && $context =~ ^[0-9]+$ ]] || { echo "cache/port/context must be integers" >&2; exit 1; }

root=$(cd "$(dirname "$0")/.." && pwd)
first_dir() { for d in "$@"; do [[ -n $d && -f $d/config.json ]] && { echo "$d"; return; }; done; }

unset MOE_GPU_MODE MOE_GPU_EXPERT_TIER MOE_GPU_EXPERT_COMPUTE MOE_DENSE_STORAGE MOE_RAM_LIMIT_BYTES
case $preset in
  olmoe-gpu|olmoe-cpu)
    model_dir=${model_dir:-$(first_dir "${OLMOE_DIR:-}" "$root/models/olmoe-1b-7b-int8" "$root/../Models/olmoe_merged")}
    if [[ $preset == olmoe-gpu ]]; then
      export MOE_GPU_MODE=auto MOE_GPU_EXPERT_TIER=on MOE_GPU_EXPERT_COMPUTE=on
    else
      export MOE_GPU_MODE=cpu
    fi ;;
  qwen3|qwen3-gpu)
    model_dir=${model_dir:-$(first_dir "${QWEN3_DIR:-}" "$root/models/qwen3-30b-a3b-int8" "$root/../Models/qwen3-30b-a3b-int8")}
    export MOE_DENSE_STORAGE=bf16 MOE_RAM_LIMIT_BYTES=$(( ${ram_limit_mib:-9216} * 1048576 ))
    if [[ $preset == qwen3-gpu ]]; then
      export MOE_GPU_MODE=auto MOE_GPU_EXPERT_TIER=on MOE_GPU_EXPERT_COMPUTE=on
    else
      export MOE_GPU_MODE=cpu
    fi ;;
  *) echo "unknown preset: $preset" >&2; usage 1 ;;
esac
[[ -n $ram_limit_mib && $preset == olmoe-* ]] && export MOE_RAM_LIMIT_BYTES=$(( ram_limit_mib * 1048576 ))
[[ -n $model_dir && -f $model_dir/config.json ]] || { echo "model directory not found for $preset; pass --model-dir" >&2; exit 1; }

binary=$root/target/release/moe-tier-engine
# Always (re)build: a no-op when current, and the web page is embedded in the binary.
(cd "$root" && cargo build --release --locked --bin moe-tier-engine)

export MOE_MODEL_DIR=$model_dir MOE_BIND=127.0.0.1:$port MOE_CONTEXT=$context
export MOE_CACHE_BYTES_PER_LAYER=$(( cache * 1048576 ))
echo "preset=$preset model=$model_dir cache=${cache}MiB/layer context=$context gpu=${MOE_GPU_MODE}" \
     "ram_limit=${MOE_RAM_LIMIT_BYTES:-default}" >&2
echo "open http://127.0.0.1:$port  (Ctrl+C to stop)" >&2
exec "$binary"
