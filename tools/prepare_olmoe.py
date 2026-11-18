#!/usr/bin/env python3
"""Standalone HF OLMoE / Qwen3-MoE -> merged row-wise INT8 Safetensors converter.
Reads one tensor at a time; creates a new output directory, never overwrites.
Dependencies: torch, numpy, safetensors; huggingface_hub for --repo only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(source, output):
    source, output = Path(source), Path(output)
    config = json.loads((source / 'config.json').read_text())
    model_type = config.get('model_type')
    if model_type == 'olmoe':
        expert_width = config['intermediate_size']
    elif model_type == 'qwen3_moe':
        # Every layer must be sparse; the dense `intermediate_size` is unused by the engine.
        if config.get('decoder_sparse_step', 1) != 1 or config.get('mlp_only_layers'):
            raise ValueError('Qwen3-MoE with dense MLP layers is not supported')
        expert_width = config['moe_intermediate_size']
    else:
        raise ValueError('expected OLMoE or Qwen3-MoE source checkpoint')
    paths = sorted(source.glob('*.safetensors'))
    index = {}
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                if name in index:
                    raise ValueError(f'duplicate tensor: {name}')
                index[name] = path
    if not index:
        raise ValueError('no source tensors')
    if any(name.endswith('.merged_weight') for name in index):
        raise ValueError('source is already converted; use its directory directly')
    # Ensure all expert projections exist before creating output.
    for layer in range(config['num_hidden_layers']):
        for expert in range(config['num_experts']):
            for projection in ['gate_proj', 'up_proj', 'down_proj']:
                name = f'model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'
                if name not in index:
                    raise ValueError(f'missing tensor: {name}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').write_text('Conversion in progress; do not run this directory.\n')
    buffer, size, shard_id = {}, 0, 0

    def read(name):
        with safe_open(str(index[name]), framework='pt', device='cpu') as handle:
            return handle.get_tensor(name).clone()

    def flush():
        nonlocal buffer, size, shard_id
        if not buffer:
            return
        target = output / f'model-{shard_id:05d}.safetensors'
        temp = target.with_suffix('.tmp')
        save_file(buffer, str(temp))
        os.replace(temp, target)
        buffer, size = {}, 0
        shard_id += 1

    def add(group):
        nonlocal size
        group_size = sum(t.numel() * t.element_size() for t in group.values())
        if size + group_size > 64 * 1024**2:
            flush()
        buffer.update(group)
        size += group_size
        if size >= 64 * 1024**2:
            flush()

    for name in sorted(index):
        if '.mlp.experts.' not in name and '.rotary_emb.' not in name:
            add({name: read(name).contiguous()})
    flush()
    for layer in range(config['num_hidden_layers']):
        for expert in range(config['num_experts']):
            prefix = f'model.layers.{layer}.mlp.experts.{expert}'
            weights, scales = [], []
            for projection in ['gate_proj', 'up_proj', 'down_proj']:
                tensor = read(f'{prefix}.{projection}.weight').float()
                shape = (config['hidden_size'], expert_width) if projection == 'down_proj' else (expert_width, config['hidden_size'])
                if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                    raise ValueError(f'invalid expert tensor {prefix}.{projection}')
                scale = tensor.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                quantized = (tensor / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
                weights.append(quantized.flatten())
                scales.append(scale)
            add({prefix + '.merged_weight': torch.cat(weights), prefix + '.qs': torch.cat(scales)})
        print(f'Converted layer {layer + 1}/{config["num_hidden_layers"]}', flush=True)
    flush()
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json',
                 'vocab.json', 'merges.txt']:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    (output / 'INCOMPLETE').unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source', type=Path)
    sources.add_argument('--repo', help='Hugging Face model ID; downloads original weights into the HF cache')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists; choose a new directory')
    source = args.source
    if args.repo:
        from huggingface_hub import snapshot_download
        source = snapshot_download(args.repo, revision=args.revision,
            allow_patterns=['*.json', '*.safetensors'])
    torch.set_num_threads(1)
    convert(source, args.out)


if __name__ == '__main__':
    main()
