#!/usr/bin/env python3
"""Standalone HF OLMoE -> merged row-wise INT4 Safetensors converter.
Quantizes expert weights into signed 4-bit nibble-packed tensors (2 weights per uint8 byte).
Reduces expert storage and NVMe streaming footprint by ~50% (from 6.01 MB to 3.01 MB per expert).
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


def pack_int4_nibbles(flat_q: torch.Tensor) -> torch.Tensor:
    """Packs signed 4-bit integers [-8, 7] into uint8 bytes (low nibble first).

    - Element 2*i is stored in bits 0..3 (low nibble)
    - Element 2*i + 1 is stored in bits 4..7 (high nibble)
    """
    if flat_q.numel() % 2 != 0:
        raise ValueError(f"expected even element count for INT4 packing, got {flat_q.numel()}")
    flat_q = flat_q.contiguous()
    lo = flat_q[0::2] & 0x0F
    hi = (flat_q[1::2] & 0x0F) << 4
    return (lo | hi).to(torch.uint8)


def convert(source, output):
    source, output = Path(source), Path(output)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "olmoe":
        raise ValueError("expected OLMoE source checkpoint")
    paths = sorted(source.glob("*.safetensors"))
    index = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in index:
                    raise ValueError(f"duplicate tensor: {name}")
                index[name] = path
    if not index:
        raise ValueError("no source tensors")

    is_already_merged = any(name.endswith(".merged_weight") for name in index)

    # Validate that all required expert tensors exist before creating output
    for layer in range(config["num_hidden_layers"]):
        for expert in range(config["num_experts"]):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            if is_already_merged:
                w_name = f"{prefix}.merged_weight"
                s_name = f"{prefix}.qs"
                if w_name not in index or s_name not in index:
                    raise ValueError(f"missing merged tensor: {w_name} or {s_name}")
            else:
                for projection in ["gate_proj", "up_proj", "down_proj"]:
                    name = f"{prefix}.{projection}.weight"
                    if name not in index:
                        raise ValueError(f"missing tensor: {name}")

    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Conversion in progress; do not run this directory.\n")
    buffer, size, shard_id = {}, 0, 0

    def read(name):
        with safe_open(str(index[name]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).clone()

    def flush():
        nonlocal buffer, size, shard_id
        if not buffer:
            return
        target = output / f"model-{shard_id:05d}.safetensors"
        temp = target.with_suffix(".tmp")
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
        if ".mlp.experts." not in name and ".rotary_emb." not in name:
            add({name: read(name).contiguous()})
    flush()

    d, i = config["hidden_size"], config["intermediate_size"]
    n = d * i

    for layer in range(config["num_hidden_layers"]):
        for expert in range(config["num_experts"]):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            weights, scales = [], []

            if is_already_merged:
                # Direct INT8 -> INT4 re-quantization
                merged_int8 = read(f"{prefix}.merged_weight")
                qs_int8 = read(f"{prefix}.qs").float()

                projections = [
                    (merged_int8[0:n].reshape(i, d), qs_int8[0:i]),
                    (merged_int8[n : 2 * n].reshape(i, d), qs_int8[i : 2 * i]),
                    (merged_int8[2 * n : 3 * n].reshape(d, i), qs_int8[2 * i : 2 * i + d]),
                ]
                for tensor_int8, scale_int8 in projections:
                    # ratio maps [-max, max] to [-7, 7]
                    max_val = tensor_int8.float().abs().amax(dim=1).clamp(min=1.0)
                    ratio = max_val / 7.0
                    quantized = (tensor_int8.float() / ratio[:, None]).round().clamp(-8, 7).to(torch.int8)
                    scale = scale_int8 * ratio
                    weights.append(quantized.flatten())
                    scales.append(scale)
            else:
                # Conversion from unmerged FP16/FP32 source
                for projection in ["gate_proj", "up_proj", "down_proj"]:
                    tensor = read(f"{prefix}.{projection}.weight").float()
                    shape = (
                        (d, i)
                        if projection == "down_proj"
                        else (i, d)
                    )
                    if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                        raise ValueError(f"invalid expert tensor {prefix}.{projection}")
                    scale = tensor.abs().amax(dim=1).clamp(min=1e-12) / 7.0
                    quantized = (tensor / scale[:, None]).round().clamp(-8, 7).to(torch.int8)
                    weights.append(quantized.flatten())
                    scales.append(scale)

            flat_weights = torch.cat(weights)
            packed_weights = pack_int4_nibbles(flat_weights)
            add({
                prefix + ".merged_weight": packed_weights,
                prefix + ".qs": torch.cat(scales),
            })
        print(f"Converted layer {layer + 1}/{config['num_hidden_layers']} (INT4)", flush=True)
    flush()
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    (output / "INCOMPLETE").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path)
    sources.add_argument(
        "--repo",
        help="Hugging Face model ID; downloads original weights into the HF cache",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new directory")
    source = args.source
    if args.repo:
        from huggingface_hub import snapshot_download

        source = snapshot_download(
            args.repo,
            revision=args.revision,
            allow_patterns=["*.json", "*.safetensors"],
        )
    torch.set_num_threads(1)
    convert(source, args.out)


if __name__ == "__main__":
    main()
