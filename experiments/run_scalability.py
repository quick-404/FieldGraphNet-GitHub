"""Scalability: end-to-end NEMESYS-first pipeline time vs input size.

Merges k BenignTraffic chunks into a single pcap (50 flows / ~4900 messages per
chunk) and runs the full pipeline, recording wall-clock time per flow count.
Confirms near-linear scaling of the detection pipeline.
"""
import sys, os, json, time, subprocess, tempfile

sys.path.insert(0, 'src')

MERGECAP = r'D:/Wireshark/mergecap.exe'
CHUNK_DIR = 'data/data/23pcap_chunks'
OUT = 'eval_results/fusion_comparison/scalability.json'
CHUNKS = [f'BenignTraffic_chunk{i}.pcap' for i in range(1, 11)]
POINTS = [(1, 50), (2, 100), (4, 200), (6, 300), (8, 400), (10, 500)]

from nemesys_gnn4id.pipeline import TrafficPipeline

p = TrafficPipeline(device='cpu', model_path='models/model.pth',
                    proto_model_path='field_proto_model_bcdg.pth')

results = []
for nchunks, max_flows in POINTS:
    inputs = [os.path.join(CHUNK_DIR, c) for c in CHUNKS[:nchunks]]
    merged = os.path.join(tempfile.gettempdir(), f'scal_bt_{nchunks}.pcap')
    subprocess.run([MERGECAP, '-w', merged, *inputs], check=True, capture_output=True)

    t0 = time.time()
    result = p.analyze(merged, max_flows=max_flows)
    elapsed = time.time() - t0

    total_flows = result.get('gnn', {}).get('total_flows', 0)
    nemesys = result.get('nemesys', {})
    messages = len(nemesys.get('segments', [])) if nemesys else 0
    results.append({
        'chunks': nchunks,
        'max_flows': max_flows,
        'flows': total_flows,
        'messages': messages,
        'elapsed_seconds': round(elapsed, 2),
    })
    print(f'{max_flows} flows ({nchunks} chunks, {messages} msgs): {elapsed:.2f}s',
          flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump({'protocol': 'BenignTraffic', 'points': results}, f, indent=2)
print('Saved', OUT)
