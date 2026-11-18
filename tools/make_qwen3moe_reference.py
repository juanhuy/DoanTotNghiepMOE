"""Generate tiny quantized Qwen3-MoE fixtures and Transformers logits (CPU only).
Covers per-head Q/K norm, head_dim != hidden/heads, GQA, norm_topk_prob and >64 experts.
Requires torch, transformers>=4.51, safetensors, tokenizers.
Usage: python tools/make_qwen3moe_reference.py tests/fixtures/qwen3moe
"""
import json
import sys
from pathlib import Path
import torch
import transformers.utils.import_utils as iu
iu._torchvision_available = False
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

torch.set_num_threads(1)
root = Path(sys.argv[1])
VARIANTS = [
    # name, heads, kv_heads, head_dim, experts, top_k, BF16 dense tensors on disk
    ('gqa-headdim', 4, 2, 6, 8, 2, False),
    ('many-experts', 2, 1, 12, 72, 4, False),
    ('bf16-dense', 4, 2, 6, 8, 2, True),
]
for name, heads, kv_heads, head_dim, experts, top_k, bf16_dense in VARIANTS:
    torch.manual_seed(321)
    config = Qwen3MoeConfig(hidden_size=16, intermediate_size=40, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=kv_heads,
        head_dim=head_dim, num_experts=experts, num_experts_per_tok=top_k, norm_topk_prob=True,
        vocab_size=32, max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-6,
        eos_token_id=31, bos_token_id=1, pad_token_id=0, tie_word_embeddings=False,
        attention_bias=False, attention_dropout=0.0, decoder_sparse_step=1, mlp_only_layers=[],
        use_sliding_window=False)
    config._attn_implementation = 'eager'
    model = Qwen3MoeForCausalLM(config).float().eval()
    tensors = {}
    with torch.no_grad():
        # Random norm gains so per-head vs whole-vector Q/K norm differ measurably.
        for module_name, param in model.named_parameters():
            if module_name.endswith('norm.weight'):
                param.copy_(1.0 + 0.5 * torch.randn_like(param))
            if bf16_dense and '.mlp.experts.' not in module_name:
                # Reference uses exactly the BF16 values stored on disk (BF16 -> F32 is exact).
                param.copy_(param.bfloat16().float())
        for layer_id, layer in enumerate(model.model.layers):
            for expert_id, expert in enumerate(layer.mlp.experts):
                quantized, scales = [], []
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    weight = getattr(expert, proj).weight
                    scale = weight.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                    q = (weight / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
                    weight.copy_(q.float() * scale[:, None])
                    quantized.append(q.flatten())
                    scales.append(scale)
                prefix = f'model.layers.{layer_id}.mlp.experts.{expert_id}'
                tensors[prefix + '.merged_weight'] = torch.cat(quantized)
                tensors[prefix + '.qs'] = torch.cat(scales)
        for tensor_name, value in model.state_dict().items():
            if '.mlp.experts.' not in tensor_name:
                tensors[tensor_name] = (value.bfloat16() if bf16_dense else value).contiguous()
        ids = [1, 3, 4, 5, 9, 2]
        logits = model(torch.tensor([ids]), use_cache=False).logits[0].tolist()
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    config.to_json_file(out / 'config.json')
    save_file(tensors, str(out / 'model.safetensors'))
    (out / 'reference.json').write_text(json.dumps({'ids': ids, 'logits': logits}))
    vocab = {f'token{i}': i for i in range(32)}
    vocab.pop('token0'); vocab['[UNK]'] = 0
    vocab.pop('token1'); vocab['[BOS]'] = 1
    vocab.pop('token31'); vocab['[EOS]'] = 31
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(['[BOS]', '[EOS]'])
    tokenizer.save(str(out / 'tokenizer.json'))
    template = "{{ bos_token }}{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
    # gqa-headdim carries the real Qwen3 chat template (copied from the HF repo); keep it.
    if not (out / 'tokenizer_config.json').exists():
        (out / 'tokenizer_config.json').write_text(json.dumps({'bos_token': '[BOS]', 'eos_token': '[EOS]', 'chat_template': template}))
    print(out)
