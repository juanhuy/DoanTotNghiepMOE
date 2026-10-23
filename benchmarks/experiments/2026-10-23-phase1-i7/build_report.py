import hashlib
import json
import statistics
import shutil
from pathlib import Path

ROOT = Path.cwd()
SOURCE = 'd4adefd9e53627bdc1fd250eabab436c3ac54b0d'
base = ROOT / 'benchmarks/runs/phase1-v1/i7'
records = {}
paths = {}
for p in base.glob('*/result.json'):
    r = json.loads(p.read_text())
    key = r['scenario']['id']
    assert key not in records, f'duplicate scenario record: {key}'
    records[key] = r
    paths[key] = p.parent

assert set(records) == {'cold', 'warm', 'conversation'}, f'suite incomplete: {set(records)}'
reference = records['cold']
for key, r in records.items():
    assert r['status'] == 'success' and r['git_dirty'] is False, f'{key} status failed or dirty'
    assert r['git_commit'] == SOURCE, f'{key} commit mismatch'
    assert r['checkpoint'] == reference['checkpoint'], f'{key} checkpoint mismatch'
    assert r['binary_sha256'] == reference['binary_sha256'], f'{key} binary mismatch'
    assert r['source_sha256'] == reference['source_sha256'], f'{key} source mismatch'
    assert r['compare_cache_policies'] is False, f'{key} compare_cache_policies mismatch'
    for setting, expected in [('cache_mib', 64), ('context', 512), ('threads', 8), ('max_tokens', 128), ('cycles', 5), ('expert_cache_policy', 'victim'), ('gpu_requested', 'auto')]:
        assert r['config'][setting] == expected, (key, setting, r['config'][setting], expected)
    assert r['ram_environment'] == {'MOE_RAM_LIMIT_BYTES': '5368709120', 'MOE_RAM_HEADROOM_BYTES': '1073741824'}
    assert len(r['result']['runs']) == (10 if key == 'conversation' else 5)
    assert len(r['result']['warmups']) == (5 if key == 'warm' else 0)
    status = r['result']['status_start']
    assert status['gpu']['enabled'] and status['ram_rss_limit_bytes'] == 5 * 1024**3
    assert status['cache_bytes_per_layer'] == 64 * 1024**2
    watch = json.loads((paths[key] / 'watchdog.json').read_text())
    assert watch['violations'] == [], f'{key} watchdog violations: {watch["violations"]}'

summary = {
    'protocol': 'phase1-v1',
    'source_commit': SOURCE,
    'testbed': 'i7',
    'status': 'completed',
    'hardware': {
        'cpu': 'Intel(R) Core(TM) i7-14650HX',
        'logical_cores': 24,
        'gpu': 'NVIDIA GeForce RTX 5060 Laptop GPU',
        'vram_total_bytes': reference['result']['status_start']['gpu']['total_vram_bytes'],
        'os': 'Linux x86_64'
    },
    'config': reference['config'],
    'ram_environment': reference['ram_environment'],
    'checkpoint': reference['checkpoint'],
    'binary_sha256': reference['binary_sha256'],
    'groups': [],
    'memory': []
}

groups = [
    ('cold Mars', 'cold', 0),
    ('warm Mars', 'warm', 0),
    ('conversation Mars', 'conversation', 0),
    ('conversation Love', 'conversation', 1)
]

for label, key, turn in groups:
    samples = [x for x in records[key]['result']['runs'] if x['turn'] == turn]
    assert len(samples) == 5, f'expected 5 samples for {label}, got {len(samples)}'
    hashes = {hashlib.sha256(x['response']['text'].encode()).hexdigest() for x in samples}
    usages = {json.dumps(x['response']['usage'], sort_keys=True) for x in samples}
    assert len(hashes) == len(usages) == 1, f'{label} output/usage drift: {len(hashes)} {len(usages)}'
    assert {x['response']['finish_reason'] for x in samples} == {'length'}, f'{label} unexpected finish reason'
    row = {
        'group': label,
        'samples': samples,
        'text_sha256': next(iter(hashes)),
        'usage': samples[0]['response']['usage'],
        'metrics': {}
    }
    for metric in [
        'time_to_first_token_ms', 'decode_tokens_per_second', 'total_ms',
        'expert_file_loads', 'expert_file_bytes_read',
        'expert_vram_download_bytes', 'expert_vram_upload_bytes',
        'expert_io_ms', 'expert_compute_ms', 'attention_ms', 'lm_head_ms',
        'cached_prompt_tokens'
    ]:
        values = [x['response']['metrics'][metric] for x in samples]
        row['metrics'][metric] = {
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values)
        }
    summary['groups'].append(row)

assert len({r['text_sha256'] for r in summary['groups'][:3]}) == 1, 'Mars output differs across scenarios'

for key in ['cold', 'warm', 'conversation']:
    series = [json.loads(line) for line in (paths[key] / 'memory.jsonl').read_text().splitlines()]
    summary['memory'].append({
        'scenario': key,
        'samples': len(series),
        'max_rss_gib': max(x['process'].get('VmRSS', 0) for x in series) / 1048576,
        'max_observed_hwm_gib': max(x['process'].get('VmHWM', 0) for x in series) / 1048576,
        'max_process_swap_kib': max(x['process'].get('VmSwap', 0) for x in series),
        'min_available_gib': min(x['system']['MemAvailable'] for x in series) / 1048576,
        'elapsed_s': series[-1]['elapsed_s']
    })

out = ROOT / 'benchmarks/experiments/2026-10-23-phase1-i7'
out.mkdir(parents=True, exist_ok=True)
for key, path in paths.items():
    dest = out / key
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(path, dest)
    for target in dest.rglob('*'):
        if target.is_file():
            original = path / target.relative_to(dest)
            assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()

(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copyfile(__file__, out / 'build_report.py')

text = f'''# Báo cáo giai đoạn1 — bộ đo i7 phase1-v1

Đây là **bộ số liệu i7 của checkpoint hiện tại**, thực hiện trên cấu hình máy đồng nghiệp với cùng workload và thiết lập đã đo trên máy i5. Không đặt mục tiêu so sánh xếp hạng CPU hay suy diễn tăng tốc giữa hai máy.

## Phiên bản và cấu hình

Source commit `{SOURCE}`, mọi run `git_dirty=false`, cùng source/binary/checkpoint hashes. Testbed i7 do người dùng xác nhận, hardware runner quan sát Intel(R) Core(TM) i7-14650HX (24 threads) / NVIDIA GeForce RTX 5060 Laptop GPU (7.53 GB VRAM); GPU enabled (cuBLAS SGEMV Active), 8 Rayon threads, victim policy, 64MiB cache/layer (~1GiB/16layer), context 512, max 128 output tokens. Guard RSS 5GiB / headroom 1GiB; watchdog RSS 5,5GiB / MemAvailable 1,5GiB / process swap 128MiB / 250ms.

Cold 5 requests; warm 5 requests có 5 warmup riêng; conversation 5 vòng Mars→Love (10 requests). Tổng 20 measured + 5 warmup, từng nhóm 5 mẫu. Reset chỉ xóa cache ứng dụng, không xóa OS cache; checkpoint hashing trước load làm ấm page cache. Warm dùng session mới giữa warmup và mẫu để không reuse KV. Conversation gửi đầy đủ assistant output trước trong history. Một loaded server cho mỗi scenario, không build/tests/inference khác đồng thời.

<details>
<summary>Từng mẫu đo (20 request, không gồm warmup)</summary>

| Nhóm | Cycle | Prompt tokens | Completion tokens | TTFT s | Decode steps/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''

for row in summary['groups']:
    for sample in row['samples']:
        response = sample['response']
        m = response['metrics']
        u = response['usage']
        text += f"| {row['group']} | {sample['cycle']} | {u['prompt_tokens']} | {u['completion_tokens']} | {m['time_to_first_token_ms']/1000:.3f} | {m['decode_tokens_per_second']:.3f} | {m['total_ms']/1000:.3f} |\n"

text += '''
</details>

## Kết quả chính

Mỗi ô là trung vị của 5 mẫu. Khoảng wall là min–max, không phải confidence interval. Decode là engine model steps/s sau token đầu, không số chunk SSE. Prompt là `Where is Mars` và `What is love ?` đúng spec trong raw.

| Nhóm | TTFT s | Decode steps/s | Wall s | Wall min–max s |
| --- | ---: | ---: | ---: | ---: |
'''

for row in summary['groups']:
    m = row['metrics']
    text += f"| {row['group']} | {m['time_to_first_token_ms']['median']/1000:.3f} | {m['decode_tokens_per_second']['median']:.3f} | {m['total_ms']['median']/1000:.3f} | {m['total_ms']['min']/1000:.3f}–{m['total_ms']['max']/1000:.3f} |\n"

text += '''
## Diễn giải chi phí

Love có 166 prompt tokens so với 19 của Mars, chỉ reuse 11 prompt tokens; cùng session không đồng nghĩa toàn bộ lịch sử đã có KV dùng lại. TTFT cao hơn đi cùng phần prefill dài hơn, không đủ bằng chứng quy hết cho SSD.

Các timer dưới đây là median, có thể chồng lấn, không cộng thành wall. File payload là bytes ứng dụng yêu cầu đọc, có thể đến từ OS page cache; không phải SSD physical I/O.

| Nhóm | Expert load wait s | Expert compute s | Attention s | File payload GB |
| --- | ---: | ---: | ---: | ---: |
'''

for row in summary['groups']:
    m = row['metrics']
    file_bytes_gb = m['expert_file_bytes_read']['median'] / (1024**3)
    io_wait_s = (m['expert_io_ms']['median']) / 1000
    compute_s = m['expert_compute_ms']['median'] / 1000
    attn_s = m['attention_ms']['median'] / 1000
    text += f"| {row['group']} | {io_wait_s:.3f} | {compute_s:.3f} | {attn_s:.3f} | {file_bytes_gb:.3f} |\n"

text += '''
## RAM trong bộ đo

| Scenario | Observed HWM max GiB | Process VmSwap max KiB | MemAvailable min GiB | Thời gian quan sát s |
| --- | ---: | ---: | ---: | ---: |
'''

for m in summary['memory']:
    text += f"| {m['scenario']} | {m['max_observed_hwm_gib']:.3f} | {m['max_process_swap_kib']} | {m['min_available_gib']:.3f} | {m['elapsed_s']:.1f} |\n"

text += '''
Ba record status success, watchdog violations rỗng; output text và usage khớp 5 mẫu trong mỗi nhóm, Mars khớp cả ba scenario. Cả 20 mẫu kết thúc reason `length` ở 128 completion tokens; đây là giới hạn sinh chủ động, không chứng minh câu trả lời hoàn chỉnh. HTTP không trả token IDs/logits nên đây không phải bit-exact logits7B. Chỉ chứng minh vận hành trong cấu hình/thời gian đã đo, không bảo đảm mọi workload không OOM/leak. RSS/HWM khác cache capacity, GPU logical khác physical allocation.

## Phạm vi nghiệm thu và phần còn lại

Bộ đo hoàn tất việc thu thập số liệu trên testbed i7 bằng đúng checkpoint, source revision và protocol đã đo trên i5. Kết quả phản ánh năng lực vận hành thực tế của engine trên cấu hình phần cứng i7 (Intel Core i7-14650HX / RTX 5060 Laptop GPU). Không quy trực tiếp mức chênh lệch giữa hai máy cho riêng CPU vì khác biệt kiến trúc GPU, RAM bandwidth và tải nền.

<details>
<summary>Hồ sơ tái lập/audit</summary>

- [Summary machine-readable](summary.json), [script tổng hợp](build_report.py).
- [Cold raw](cold/result.json), [warm raw](warm/result.json), [conversation raw](conversation/result.json).
- Từng folder có build/server log, memory.jsonl/watchdog.json, source snapshots và từng turn; warmup chỉ ở warmup files/result.warmups.
- Kiểm byte bất biến bằng `sha256sum -c CHECKSUMS.sha256` trong thư mục này.
- Protocol/lệnh ở [SCENARIOS](../../SCENARIOS.md), trạng thái ở [PLAN](../../../docs/PLAN.md), lịch sử checks ở [PROGRESS](../../../docs/PROGRESS.md).

</details>
'''

(out / 'README.md').write_text(text)

chk_file = out / 'CHECKSUMS.sha256'
if chk_file.exists():
    chk_file.unlink()
with chk_file.open('w') as manifest:
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name != 'CHECKSUMS.sha256':
            manifest.write(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(out)) + '\n')

print('REPORT GENERATED SUCCESSFULLY:', out)
for row in summary['groups']:
    print(row['group'], json.dumps({k: v['median'] for k, v in row['metrics'].items()}))
print('MEMORY', json.dumps(summary['memory']))
