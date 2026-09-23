"""
Split CIC-2023 PCAPs into manageable chunks and run NEMESYS-first fusion evaluation.
"""
import sys, os, json, time
sys.path.insert(0, 'src')

PCAP_DIR = 'data/data/23pcap'
CHUNK_DIR = 'data/data/23pcap_chunks'
EVAL_OUT = 'eval_results/fusion_comparison'
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(EVAL_OUT, exist_ok=True)

# PCAP -> attack class mapping (GNN4ID 8-class)
PCAP_LABELS = {
    'BenignTraffic.pcap': 0,          # Benign
    'DNS_Spoofing.pcap': 2,           # Spoofing
    'Mirai-udpplain.pcap': 4,         # Mirai
    'SqlInjection.pcap': 1,           # WebBased
    'DDoS-HTTP_Flood-.pcap': 6,       # DDos
    'DictionaryBruteForce.pcap': 7,   # BruteForce
    'Recon-PortScan.pcap': 3,         # Recon
}
MAX_PKTS = 50000   # 每个 PCAP 最多处理 50000 包
PKTS_PER_CHUNK = 5000

print('=' * 60)
print('Step 1: Splitting PCAPs')
print('=' * 60)

from scapy.all import PcapReader, wrpcap

chunk_info = {}  # chunk_path -> (attack_label, source_pcap)

for pcap_name, attack_label in PCAP_LABELS.items():
    pcap_path = os.path.join(PCAP_DIR, pcap_name)
    if not os.path.exists(pcap_path):
        print(f'  [SKIP] {pcap_name} not found')
        continue

    print(f'\n  Streaming {pcap_name}...')
    base = os.path.splitext(pcap_name)[0]
    reader = PcapReader(pcap_path)
    chunk_pkts = []
    chunk_idx = 1
    total_pkts = 0

    for pkt in reader:
        if total_pkts >= MAX_PKTS:
            break
        chunk_pkts.append(pkt)
        if len(chunk_pkts) >= PKTS_PER_CHUNK:
            chunk_name = f'{base}_chunk{chunk_idx}.pcap'
            chunk_path = os.path.join(CHUNK_DIR, chunk_name)
            wrpcap(chunk_path, chunk_pkts)
            chunk_info[chunk_path] = (attack_label, pcap_name)
            print(f'    chunk {chunk_idx}: {len(chunk_pkts)} pkts')
            chunk_pkts = []
            chunk_idx += 1
        total_pkts += 1

    # Last partial chunk
    if chunk_pkts:
        chunk_name = f'{base}_chunk{chunk_idx}.pcap'
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        wrpcap(chunk_path, chunk_pkts)
        chunk_info[chunk_path] = (attack_label, pcap_name)
        print(f'    chunk {chunk_idx}: {len(chunk_pkts)} pkts')

    reader.close()
    print(f'  {base}: {total_pkts} total packets, {chunk_idx} chunks')

# Save chunk index
index_path = os.path.join(CHUNK_DIR, '_chunk_index.json')
with open(index_path, 'w') as f:
    json.dump({os.path.basename(k): v for k, v in chunk_info.items()}, f, indent=2)
print(f'\nSaved chunk index: {index_path}')

print(f'\nTotal chunks: {len(chunk_info)}')
print('Split complete.')
