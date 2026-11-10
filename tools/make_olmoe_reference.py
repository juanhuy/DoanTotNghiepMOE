"""Generate tiny quantized OLMoE fixtures and Transformers logits (CPU only).
Requires torch, transformers==4.51.3, safetensors, tokenizers.
Usage: python tools/make_olmoe_reference.py tests/fixtures/olmoe
"""
import json
import sys
from pathlib import Path
import torch
import transformers.utils.import_utils as iu
# The reference is text-only and does not require optional torchvision kernels.
iu._torchvision_available = False
from transformers import OlmoeConfig, OlmoeForCausalLM
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

torch.set_num_threads(1)
root = Path(sys.argv[1])
for normalize, kv_heads in [(False, 2), (True, 1)]:
    torch.manual_seed(123)
    config = OlmoeConfig(hidden_size=16, intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=kv_heads, num_experts=4,
        num_experts_per_tok=2, vocab_size=32, max_position_embeddings=64,
        norm_topk_prob=normalize, eos_token_id=31, pad_token_id=0,
        tie_word_embeddings=False, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    model = OlmoeForCausalLM(config).float().eval()
    tensors = {}
    with torch.no_grad():
        for layer_id, layer in enumerate(model.model.layers):
            for expert_id, expert in enumerate(layer.mlp.experts):
                quantized, scales = [], []
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    weight = getattr(expert, name).weight
                    scale = weight.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                    q = (weight / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
                    weight.copy_(q.float() * scale[:, None])
                    quantized.append(q.flatten())
                    scales.append(scale)
                prefix = f'model.layers.{layer_id}.mlp.experts.{expert_id}'
                tensors[prefix + '.merged_weight'] = torch.cat(quantized)
                tensors[prefix + '.qs'] = torch.cat(scales)
        for name, value in model.state_dict().items():
            if '.mlp.experts.' not in name:
                tensors[name] = value.contiguous()
        ids = [1, 3, 4, 5]
        logits = model(torch.tensor([ids]), use_cache=False).logits[0].tolist()
    out = root / ('normalized-gqa' if normalize else 'unnormalized-mha')
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
    (out / 'tokenizer_config.json').write_text(json.dumps({'bos_token': '[BOS]', 'eos_token':'[EOS]', 'chat_template':template}))
    print(out)
