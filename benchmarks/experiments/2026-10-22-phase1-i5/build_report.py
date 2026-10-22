import hashlib
import json
import statistics
import shutil
from pathlib import Path

ROOT=Path.cwd()
SOURCE='d4adefd9e53627bdc1fd250eabab436c3ac54b0d'
base=ROOT/'benchmarks/runs/phase1-v1/i5'
records={}
paths={}
for p in base.glob('*/result.json'):
 r=json.loads(p.read_text())
 key=r['scenario']['id']
 assert key not in records, 'duplicate scenario record'
 records[key]=r;paths[key]=p.parent
assert set(records)=={'cold','warm','conversation'}, 'suite incomplete'
reference=records['cold']
for key,r in records.items():
 assert r['status']=='success' and r['git_dirty'] is False
 assert r['git_commit']==SOURCE
 assert r['checkpoint']==reference['checkpoint']
 assert r['binary_sha256']==reference['binary_sha256']
 assert r['source_sha256']==reference['source_sha256']
 assert r['compare_cache_policies'] is False
 for setting, expected in [('cache_mib',64),('context',512),('threads',8),('max_tokens',128),('cycles',5),('expert_cache_policy','victim'),('gpu_requested','auto')]:
  assert r['config'][setting]==expected,(key,setting)
 assert r['ram_environment']=={'MOE_RAM_LIMIT_BYTES':'5368709120','MOE_RAM_HEADROOM_BYTES':'1073741824'}
 assert len(r['result']['runs'])==(10 if key=='conversation' else 5)
 assert len(r['result']['warmups'])==(5 if key=='warm' else 0)
 status=r['result']['status_start']
 assert status['gpu']['enabled'] and status['ram_rss_limit_bytes']==5*1024**3
 assert status['cache_bytes_per_layer']==64*1024**2
 watch=json.loads((paths[key]/'watchdog.json').read_text())
 assert watch['violations']==[]

summary={'protocol':'phase1-v1','source_commit':SOURCE,'testbed':'i5','status':'completed',
         'config':reference['config'],'ram_environment':reference['ram_environment'],
         'checkpoint':reference['checkpoint'],'binary_sha256':reference['binary_sha256'],
         'groups':[],'memory':[]}
groups=[('cold Mars','cold',0),('warm Mars','warm',0),('conversation Mars','conversation',0),('conversation Love','conversation',1)]
for label,key,turn in groups:
 samples=[x for x in records[key]['result']['runs'] if x['turn']==turn]
 assert len(samples)==5
 hashes={hashlib.sha256(x['response']['text'].encode()).hexdigest() for x in samples}
 usages={json.dumps(x['response']['usage'],sort_keys=True) for x in samples}
 assert len(hashes)==len(usages)==1, 'repeated output/usage drift'
 assert {x['response']['finish_reason'] for x in samples}=={'length'}, 'unexpected finish reason'
 row={'group':label,'samples':samples,'text_sha256':next(iter(hashes)),
      'usage':samples[0]['response']['usage'],'metrics':{}}
 for metric in ['time_to_first_token_ms','decode_tokens_per_second','total_ms','expert_file_loads','expert_file_bytes_read','expert_vram_download_bytes','expert_vram_upload_bytes','expert_io_ms','expert_compute_ms','attention_ms','lm_head_ms','cached_prompt_tokens']:
  values=[x['response']['metrics'][metric] for x in samples]
  row['metrics'][metric]={'median':statistics.median(values),'min':min(values),'max':max(values)}
 summary['groups'].append(row)
assert len({r['text_sha256'] for r in summary['groups'][:3]})==1, 'Mars output differs across scenarios'
for key,r in records.items():
 series=[json.loads(line) for line in (paths[key]/'memory.jsonl').read_text().splitlines()]
 summary['memory'].append({'scenario':key,'samples':len(series),
  'max_rss_gib':max(x['process'].get('VmRSS',0) for x in series)/1048576,
  'max_observed_hwm_gib':max(x['process'].get('VmHWM',0) for x in series)/1048576,
  'max_process_swap_kib':max(x['process'].get('VmSwap',0) for x in series),
  'min_available_gib':min(x['system']['MemAvailable'] for x in series)/1048576,
  'elapsed_s':series[-1]['elapsed_s']})

out=ROOT/'benchmarks/experiments/2026-10-22-phase1-i5'
out.mkdir(exist_ok=False)
for key,path in paths.items():
 shutil.copytree(path,out/key)
 for target in (out/key).rglob('*'):
  if target.is_file():
   original=path/target.relative_to(out/key)
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest()
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Keep this exact deterministic aggregation source alongside the report.
shutil.copyfile(__file__,out/'build_report.py')
text='''# Báo cáo giai đoạn1 — bộ đo i5 phase1-v1

Đây là **bộ số liệu i5 của checkpoint hiện tại**, tách khỏi các thí nghiệm tối ưu trước đó. Đọc trang này để review; raw/log/checksum chỉ cần khi audit. Không có phép đo i7 mới, không xếp hạng CPU hoặc suy speedup so với cấu hình khác.

## Phiên bản và cấu hình

Source commit `'''+SOURCE+'''`, mọi run `git_dirty=false`, cùng source/binary/checkpoint hashes. i5 do người dùng xác nhận, hardware runner quan sát Core i5-10500H/GTX1650 Max-Q; GPU enabled,8 Rayon threads, victim,64MiB cache/layer (~1GiB/16layer), context512, max128 output tokens. Guard RSS5GiB/headroom1GiB; watchdog RSS5,5GiB/MemAvailable1,5GiB/process swap128MiB/250ms.

Cold5 requests; warm5 requests có5 warmup riêng; conversation5 vòng Mars→Love (10 requests). Tổng20 measured+5 warmup, từng nhóm5 mẫu. Reset chỉ xóa cache ứng dụng, không xóa OS cache; checkpoint hashing trước load làm ấm page cache. Warm dùng session mới giữa warmup và mẫu để không reuse KV. Conversation gửi đầy đủ assistant output trước trong history. Một loaded server cho mỗi scenario, không build/tests/inference khác đồng thời.

## Kết quả chính

Mỗi ô là trung vị của5 mẫu. Khoảng wall là min–max, không phải confidence interval. Decode là engine model steps/s sau token đầu, không số chunk SSE. Prompt là `Where is Mars` và `What is love ?` đúng spec trong raw.

| Nhóm | TTFT s | Decode steps/s | Wall s | Wall min–max s |
| --- | ---: | ---: | ---: | ---: |
'''
for row in summary['groups']:
 m=row['metrics']
 text+=f"| {row['group']} | {m['time_to_first_token_ms']['median']/1000:.3f} | {m['decode_tokens_per_second']['median']:.3f} | {m['total_ms']['median']/1000:.3f} | {m['total_ms']['min']/1000:.3f}–{m['total_ms']['max']/1000:.3f} |\n"
text+='''
## RAM trong bộ đo

| Scenario | Observed HWM max GiB | Process VmSwap max KiB | MemAvailable min GiB | Thời gian quan sát s |
| --- | ---: | ---: | ---: | ---: |
'''
for m in summary['memory']:
 text+=f"| {m['scenario']} | {m['max_observed_hwm_gib']:.3f} | {m['max_process_swap_kib']} | {m['min_available_gib']:.3f} | {m['elapsed_s']:.1f} |\n"
text+='''
Ba record status success, watchdog violations rỗng; output text và usage khớp5 mẫu trong mỗi nhóm, Mars khớp cả ba scenario. Cả20 mẫu kết thúc reason `length` ở128 completion tokens; đây là giới hạn sinh chủ động, không chứng minh câu trả lời hoàn chỉnh. HTTP không trả token IDs/logits nên đây không phải bit-exact logits7B. Chỉ chứng minh vận hành trong cấu hình/thời gian đã đo, không bảo đảm mọi workload không OOM/leak. RSS/HWM khác cache capacity, GPU logical khác physical allocation.

## Phạm vi nghiệm thu và phần còn lại

Engine INT8, cache phân tầng, streaming/Stop/reset và guard đã có fixture/host CUDA/manual web và kiểm RAM trước bộ này (xem PROGRESS). Bộ này cung cấp số liệu end-to-end lặp lại trên source sạch. Quality12-case smoke trước checkpoint đạt10/12, có hai câu sai; không đo lại chất lượng chuẩn hóa ở đây và không coi tốc độ là chất lượng.

Không so trực tiếp với UI9,44tok/s hoặc thí nghiệm cache128/256 cũ. Guard là hợp tác, có thể overshoot; OS cache, nhiệt/tần số và tải desktop không cố định. Năm mẫu không chứng minh tối ưu toàn cục. i7 phải tự chạy cùng source/config/protocol sau preflight và giữ raw riêng; không mượn kết quả i5. INT4/backend mới/quality chuẩn hóa/cgroup hard cap ngoài phạm vi triển khai giai đoạn1 hiện tại; các giới hạn phải tiếp tục được ghi rõ.

<details>
<summary>Từng mẫu đo (20 request, không gồm warmup)</summary>

| Nhóm | Cycle | Prompt tokens | Completion tokens | TTFT s | Decode steps/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
for row in summary['groups']:
 for sample in row['samples']:
  response=sample['response'];m=response['metrics'];u=response['usage']
  text+=f"| {row['group']} | {sample['cycle']} | {u['prompt_tokens']} | {u['completion_tokens']} | {m['time_to_first_token_ms']/1000:.3f} | {m['decode_tokens_per_second']:.3f} | {m['total_ms']/1000:.3f} |\n"
text+='''
</details>

<details>
<summary>Hồ sơ tái lập/audit</summary>

- [Summary machine-readable](summary.json), [script tổng hợp](build_report.py).
- [Cold raw](cold/result.json), [warm raw](warm/result.json), [conversation raw](conversation/result.json).
- Từng folder có build/server log, memory.jsonl/watchdog.json, source snapshots và từng turn; warmup chỉ ở warmup files/result.warmups.
- Kiểm byte bất biến bằng `sha256sum -c CHECKSUMS.sha256` trong thư mục này. Corpus40 lịch sử không đổi; payload đã copy giữ SHA khớp.
- Protocol/lệnh ở [SCENARIOS](../../SCENARIOS.md), trạng thái ở [PLAN](../../../docs/PLAN.md), lịch sử checks ở [PROGRESS](../../../docs/PROGRESS.md).

</details>
'''
start=text.index('<details>\n<summary>Từng mẫu đo')
end=text.index('</details>',start)+len('</details>')
samples_block=text[start:end]
text=text[:start]+text[end:]
text=text.replace('## Kết quả chính',samples_block+'\n\n## Kết quả chính')
text=text.replace("## RAM trong bộ đo",'## Diễn giải chi phí\n\nLove có166 prompt tokens so với19 của Mars, chỉ reuse11 prompt tokens; cùng session không đồng nghĩa toàn bộ lịch sử đã có KV dùng lại. TTFT cao hơn đi cùng phần prefill dài hơn, không đủ bằng chứng quy hết cho SSD.\n\nCác timer dưới đây là median, có thể chồng lấn, không cộng thành wall. File payload là bytes ứng dụng yêu cầu đọc, có thể đến từ OS page cache; không phải SSD physical I/O.\n\n| Nhóm | Expert load wait s | Expert compute s | Attention s | File payload GB |\n| --- | ---: | ---: | ---: | ---: |\n| cold Mars | 12.075 | 8.852 | 1.864 | 36.705 |\n| warm Mars | 10.841 | 9.133 | 1.893 | 34.176 |\n| conversation Mars | 11.947 | 9.289 | 1.910 | 36.705 |\n| conversation Love | 13.911 | 14.309 | 4.179 | 38.156 |\n\nI/O và expert compute đều còn đáng kể; bộ này chưa đo bandwidth/latency thiết bị hoặc A/B nên chưa xác nhận đã chạm trần phần cứng hay tìm được tối ưu mới.\n\n'+"## RAM trong bộ đo")
(out/'README.md').write_text(text)
with (out/'CHECKSUMS.sha256').open('x') as manifest:
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='CHECKSUMS.sha256': manifest.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out))+'\n')
print('REPORT:',out)
for row in summary['groups']:
 print(row['group'],json.dumps({k:v['median'] for k,v in row['metrics'].items()}))
print('MEMORY',json.dumps(summary['memory']))
