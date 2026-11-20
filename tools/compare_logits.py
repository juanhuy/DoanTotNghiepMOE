"""Compare two logit dumps (seq-<i>.f32, positions x vocab) from dump_logits / stream_reference.

usage: python tools/compare_logits.py REFERENCE_DIR CANDIDATE_DIR --vocab N [--json OUT]
Reports per sequence and overall: max/mean |delta logit|, top-1 agreement, top-5 overlap,
KL(reference || candidate) on softmax, and the reference top-1 margin where top-1 differs.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(directory, index, vocab):
    data = np.fromfile(Path(directory) / f'seq-{index}.f32', dtype='<f4')
    if data.size % vocab:
        raise SystemExit(f'{directory}/seq-{index}.f32 is not a multiple of vocab {vocab}')
    return data.reshape(-1, vocab).astype(np.float64)


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def compare(ref, cand):
    if ref.shape != cand.shape:
        raise SystemExit(f'shape mismatch {ref.shape} vs {cand.shape}')
    delta = np.abs(ref - cand)
    lr, lc = log_softmax(ref), log_softmax(cand)
    kl = (np.exp(lr) * (lr - lc)).sum(axis=1)
    top_r, top_c = ref.argmax(axis=1), cand.argmax(axis=1)
    top5_r = np.argsort(-ref, axis=1)[:, :5]
    top5_c = np.argsort(-cand, axis=1)[:, :5]
    overlap = [len(set(a) & set(b)) / 5 for a, b in zip(top5_r, top5_c)]
    sorted_ref = np.sort(ref, axis=1)
    margin = sorted_ref[:, -1] - sorted_ref[:, -2]
    differ = np.nonzero(top_r != top_c)[0]
    return {
        'positions': int(ref.shape[0]),
        'max_abs_delta': float(delta.max()),
        'mean_abs_delta': float(delta.mean()),
        'max_abs_delta_per_position_median': float(np.median(delta.max(axis=1))),
        'reference_logit_abs_max': float(np.abs(ref).max()),
        'top1_agree': int((top_r == top_c).sum()),
        'top5_overlap_mean': float(np.mean(overlap)),
        'kl_mean': float(kl.mean()),
        'kl_max': float(kl.max()),
        'top1_differs_at': [{'position': int(p), 'reference_top1': int(top_r[p]), 'candidate_top1': int(top_c[p]),
                             'reference_margin': float(margin[p])} for p in differ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--vocab', type=int, required=True)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    count = len(list(args.reference.glob('seq-*.f32')))
    if count == 0 or count != len(list(args.candidate.glob('seq-*.f32'))):
        raise SystemExit('reference and candidate must hold the same non-zero number of sequences')
    per = []
    for i in range(count):
        per.append(compare(load(args.reference, i, args.vocab), load(args.candidate, i, args.vocab)))
    positions = sum(p['positions'] for p in per)
    overall = {
        'sequences': count,
        'positions': positions,
        'max_abs_delta': max(p['max_abs_delta'] for p in per),
        'mean_abs_delta': sum(p['mean_abs_delta'] * p['positions'] for p in per) / positions,
        'top1_agree': sum(p['top1_agree'] for p in per),
        'top5_overlap_mean': sum(p['top5_overlap_mean'] * p['positions'] for p in per) / positions,
        'kl_mean': sum(p['kl_mean'] * p['positions'] for p in per) / positions,
        'kl_max': max(p['kl_max'] for p in per),
    }
    result = {'reference': str(args.reference), 'candidate': str(args.candidate), 'overall': overall,
              'per_sequence': per}
    text = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(text + '\n')
    print(json.dumps(overall, indent=2))


if __name__ == '__main__':
    main()
