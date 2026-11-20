"""Fixed parity prompt set (real-model-parity-v1) tokenized with the model's own tokenizer.

Chat prompts use the checkpoint's chat template with a generation prompt (Qwen3: thinking
off, as in every Qwen3 benchmark); plain prompts are raw text with the tokenizer's default
special tokens. Writes {"sequences": [...], "prompts": [...], "tokenizer": ...}.
usage: python tools/make_parity_ids.py TOKENIZER_DIR OUT.json
"""
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

CHAT = ['Where is Mars', 'What is love ?', 'Explain a Python dictionary.',
        'Thủ đô của Việt Nam là gì? Trả lời ngắn gọn.']
PLAIN = ['The quick brown fox jumps over the lazy dog. In 1969, Apollo 11 landed on the Moon, '
         'and Neil Armstrong became the first person to walk on its surface.',
         'def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n'
         'print([fibonacci(i) for i in range(10)])']


def main():
    directory, out = Path(sys.argv[1]), Path(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(directory)
    config = json.loads((directory / 'config.json').read_text())
    extra = {'enable_thinking': False} if config['model_type'] == 'qwen3_moe' else {}
    sequences, prompts = [], []
    for text in CHAT:
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True,
                                            tokenize=True, **extra)
        sequences.append(list(ids))
        prompts.append({'kind': 'chat', 'text': text})
    for text in PLAIN:
        sequences.append(tokenizer(text)['input_ids'])
        prompts.append({'kind': 'plain', 'text': text})
    tokenizer_sha = hashlib.sha256((directory / 'tokenizer.json').read_bytes()).hexdigest()
    out.write_text(json.dumps({'sequences': sequences, 'prompts': prompts, 'tokenizer_dir': str(directory),
                               'tokenizer_json_sha256': tokenizer_sha, 'chat_kwargs': extra}, indent=1))
    print([len(s) for s in sequences], sum(map(len, sequences)))


if __name__ == '__main__':
    main()
