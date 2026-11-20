"""Layer-streamed reference equals a full Transformers forward (BF16 and INT8-dequant experts).

Tiny random OLMoE and Qwen3-MoE models are saved with BF16 weights, converted with
prepare_olmoe.py, and streamed; logits must equal the whole-model F32 forward exactly.
Requires torch, transformers>=4.51, safetensors (same environment as --converter).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import transformers.utils.import_utils as iu
iu._torchvision_available = False
from transformers import OlmoeConfig, OlmoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from prepare_olmoe import convert

TOOLS = Path(__file__).resolve().parent
IDS = [[1, 5, 9, 2, 7, 3], [4, 4, 8]]


def tiny(model_type):
    torch.manual_seed(7)
    common = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  num_experts=6, num_experts_per_tok=2, vocab_size=32, max_position_embeddings=64,
                  rms_norm_eps=1e-6, tie_word_embeddings=False)
    if model_type == 'qwen3_moe':
        config = Qwen3MoeConfig(**common, head_dim=8, intermediate_size=40, moe_intermediate_size=8,
                                norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[])
        cls = Qwen3MoeForCausalLM
    else:
        config = OlmoeConfig(**common, intermediate_size=8, norm_topk_prob=False)
        cls = OlmoeForCausalLM
    config._attn_implementation = 'eager'
    model = cls(config).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith('norm.weight'):
                param.copy_(1.0 + 0.5 * torch.randn_like(param))
            param.copy_(param.bfloat16().float())  # weights on disk are BF16
    return model


def quantize_experts_in_place(model):
    with torch.no_grad():
        for layer in model.model.layers:
            for expert in layer.mlp.experts:
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    w = getattr(expert, proj).weight
                    scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                    w.copy_((w / scale[:, None]).round().clamp(-128, 127) * scale[:, None])


class StreamReferenceTest(unittest.TestCase):
    def check(self, model_type):
        model = tiny(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, converted = root / 'source', root / 'int8'
            model.to(torch.bfloat16).save_pretrained(source, safe_serialization=True)
            model.float()
            convert(source, converted)
            (root / 'ids.json').write_text(json.dumps({'sequences': IDS}))
            for mode in ['bf16', 'int8']:
                if mode == 'int8':
                    quantize_experts_in_place(model)
                out = root / mode
                subprocess.run([sys.executable, str(TOOLS / 'stream_reference.py'), '--source', str(source),
                                '--converted', str(converted), '--experts', mode, '--ids', str(root / 'ids.json'),
                                '--out', str(out), '--threads', '1'], check=True, capture_output=True)
                for i, ids in enumerate(IDS):
                    with torch.no_grad():
                        expected = model(torch.tensor([ids]), use_cache=False).logits[0].numpy()
                    actual = np.fromfile(out / f'seq-{i}.f32', dtype='<f4').reshape(expected.shape)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5,
                                               err_msg=f'{model_type} {mode} sequence {i}')

    def test_olmoe(self):
        self.check('olmoe')

    def test_qwen3_moe(self):
        self.check('qwen3_moe')


if __name__ == '__main__':
    unittest.main()
