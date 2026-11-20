"""Tensor-by-tensor equality of two safetensors checkpoints (shard layout may differ).

usage: python tools/compare_checkpoint_tensors.py DIR_A DIR_B [--json OUT]
Exit code 0 only when both hold the same tensor names with identical dtype, shape and bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def digests(directory):
    out = {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name).contiguous()
                raw = tensor.view(-1).view(dtype=__import__('torch').uint8).numpy().tobytes()
                out[name] = (str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    a, b = digests(args.a), digests(args.b)
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    differ = sorted(name for name in set(a) & set(b) if a[name] != b[name])
    result = {'a': str(args.a), 'b': str(args.b), 'tensors_a': len(a), 'tensors_b': len(b),
              'only_a': only_a, 'only_b': only_b, 'differ': differ,
              'identical': not (only_a or only_b or differ)}
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in result.items()}))
    raise SystemExit(0 if result['identical'] else 1)


if __name__ == '__main__':
    main()
