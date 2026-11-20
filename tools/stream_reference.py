"""Transformers reference logits for a checkpoint larger than RAM, one decoder layer at a time.

Each layer is built on the meta device from the Transformers class of the model
(OLMoE or Qwen3-MoE, eager attention), filled with that layer's weights only, run over
every sequence, then freed. Math is Transformers' own forward in F32; only weight loading
differs from `AutoModelForCausalLM`, so peak RAM is about one layer instead of the model.

Experts come either from the original BF16 checkpoint (`--experts bf16`, the quantization
baseline) or are dequantized from the converted INT8 checkpoint (`--experts int8`, the exact
weights the engine multiplies), using `merged_weight`/`qs` as written by prepare_olmoe.py.

usage:
  python tools/stream_reference.py --source HF_DIR [--converted INT8_DIR] --experts bf16|int8 \
      --ids ids.json --out OUT_DIR
Writes seq-<i>.f32 (little-endian F32, positions x vocab) and reference.json.
"""
import argparse
import ctypes
import gc
import json
import platform
import resource
import time
from pathlib import Path

import torch
import transformers
import transformers.utils.import_utils as iu
iu._torchvision_available = False
from safetensors import safe_open


def release_freed_memory_to_os():
    """glibc keeps freed per-expert tensors (~6MB) in its heap and RSS grows layer after layer;
    a fixed mmap threshold makes every large tensor its own mapping, returned on free."""
    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.mallopt(-3, 1 << 20)  # M_MMAP_THRESHOLD = 1MiB (also disables the dynamic threshold)
        return libc
    except OSError:
        return None


def rss_mib():
    with open('/proc/self/status') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0.0


def tensor_index(directory):
    index = {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                index[name] = path
    if not index:
        raise SystemExit(f'no safetensors in {directory}')
    return index


class Reader:
    """Opens the shard per read: a long-lived safe_open handle keeps every tensor read through it
    resident (RSS grew by one layer of weights per layer), which would exceed RAM on 30B models."""
    def __init__(self, directory):
        self.index = tensor_index(directory)

    def __contains__(self, name):
        return name in self.index

    def get(self, name):
        with safe_open(str(self.index[name]), framework='pt', device='cpu') as handle:
            return handle.get_tensor(name).clone()

    def rows(self, name, ids):
        with safe_open(str(self.index[name]), framework='pt', device='cpu') as handle:
            sliced = handle.get_slice(name)
            return torch.stack([sliced[i:i + 1][0].clone() for i in ids])


def classes(model_type):
    if model_type == 'olmoe':
        from transformers.models.olmoe import modeling_olmoe as m
        return m.OlmoeConfig, m.OlmoeDecoderLayer, m.OlmoeRMSNorm, m.OlmoeRotaryEmbedding
    if model_type == 'qwen3_moe':
        from transformers.models.qwen3_moe import modeling_qwen3_moe as m
        return m.Qwen3MoeConfig, m.Qwen3MoeDecoderLayer, m.Qwen3MoeRMSNorm, m.Qwen3MoeRotaryEmbedding
    raise SystemExit(f'unsupported model_type {model_type}')


def expert_width(config):
    return config.moe_intermediate_size if config.model_type == 'qwen3_moe' else config.intermediate_size


def dequantized_experts(converted, layer, config):
    """F32 gate/up/down per expert from INT8 rows and per-row scales (prepare_olmoe.py layout)."""
    hidden, width = config.hidden_size, expert_width(config)
    out = {}
    for expert in range(config.num_experts):
        prefix = f'model.layers.{layer}.mlp.experts.{expert}'
        q = converted.get(prefix + '.merged_weight').float()
        s = converted.get(prefix + '.qs').float()
        gate = q[:width * hidden].view(width, hidden) * s[:width, None]
        up = q[width * hidden:2 * width * hidden].view(width, hidden) * s[width:2 * width, None]
        down = q[2 * width * hidden:].view(hidden, width) * s[2 * width:, None]
        local = f'mlp.experts.{expert}'
        out[f'{local}.gate_proj.weight'] = gate
        out[f'{local}.up_proj.weight'] = up
        out[f'{local}.down_proj.weight'] = down
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True,
                        help='dense weights (and BF16 experts unless --experts int8)')
    parser.add_argument('--converted', type=Path, help='INT8 checkpoint for --experts int8 (default: --source)')
    parser.add_argument('--experts', choices=['bf16', 'int8'], required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    libc = release_freed_memory_to_os()
    torch.set_num_threads(args.threads)
    raw_config = json.loads((args.source / 'config.json').read_text())
    Config, Layer, Norm, Rotary = classes(raw_config['model_type'])
    config = Config.from_dict(raw_config)
    config._attn_implementation = 'eager'
    sequences = json.loads(args.ids.read_text())['sequences']
    source = Reader(args.source)
    converted = Reader(args.converted or args.source) if args.experts == 'int8' else None
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    layer_seconds = []
    with torch.inference_mode():
        hidden = [source.rows('model.embed_tokens.weight', ids).float()[None] for ids in sequences]
        rotary = Rotary(config=config)
        masks, positions, rope = [], [], []
        for ids in sequences:
            n = len(ids)
            mask = torch.full((n, n), torch.finfo(torch.float32).min).triu(1)[None, None]
            pos = torch.arange(n)[None]
            masks.append(mask)
            positions.append(pos)
            rope.append(rotary(hidden[0], pos))
        for layer in range(config.num_hidden_layers):
            t = time.time()
            prefix = f'model.layers.{layer}.'
            state = {name[len(prefix):]: source.get(name).float() for name in source.index
                     if name.startswith(prefix) and (args.experts == 'bf16' or '.mlp.experts.' not in name)}
            if args.experts == 'int8':
                state.update(dequantized_experts(converted, layer, config))
            with torch.device('meta'):
                module = Layer(config, layer)
            module.load_state_dict(state, strict=True, assign=True)
            module.eval()
            del state
            for i in range(len(sequences)):
                hidden[i] = module(hidden[i], attention_mask=masks[i], position_ids=positions[i],
                                   position_embeddings=rope[i])[0]
            del module
            gc.collect()
            if libc is not None:
                libc.malloc_trim(0)
            layer_seconds.append(time.time() - t)
            print(f'layer {layer + 1}/{config.num_hidden_layers} {layer_seconds[-1]:.1f}s rss {rss_mib():.0f}MiB',
                  flush=True)
        norm = Norm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight.data = source.get('model.norm.weight').float()
        head_name = 'lm_head.weight' if 'lm_head.weight' in source else 'model.embed_tokens.weight'
        head = source.get(head_name).float()
        shapes = []
        for i, h in enumerate(hidden):
            logits = (norm(h)[0] @ head.T).contiguous()
            if not torch.isfinite(logits).all():
                raise SystemExit(f'non-finite reference logits in sequence {i}')
            (args.out / f'seq-{i}.f32').write_bytes(logits.numpy().astype('<f4').tobytes())
            shapes.append({'positions': logits.shape[0], 'vocab': logits.shape[1]})
    meta = {
        'source': str(args.source), 'converted': str(args.converted) if converted else None,
        'experts': args.experts, 'compute_dtype': 'float32', 'attention': 'eager',
        'model_type': config.model_type, 'layers': config.num_hidden_layers, 'shapes': shapes,
        'seconds': time.time() - started, 'layer_seconds': layer_seconds, 'threads': args.threads,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'python': platform.python_version(),
    }
    (args.out / 'reference.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
