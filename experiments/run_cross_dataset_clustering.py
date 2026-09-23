"""
Lightweight cross-dataset validation on CICIDS2017.
No torch/torch_geometric imports — only BCDG + clustering.

Uses scapy PcapReader to sample packets incrementally (memory-safe).
"""
import sys, os, json, time
sys.path.insert(0, 'src')

DATA_DIR = 'data/data/17pcap'
EVAL_DIR = 'eval_results/fusion_comparison'
os.makedirs(EVAL_DIR, exist_ok=True)

from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.nemesys.cluster import ClusterAnalyzer
from nemesys_gnn4id.pipeline import _segments_to_dicts

SAMPLE_PACKETS = 5000


def sample_pcap_packets(pcap_path, n_packets=5000, offset=0):
    """Extract N packets from a large pcap using PcapReader (memory safe)."""
    from scapy.utils import PcapReader, wrpcap
    import io

    temp_dir = os.path.join(EVAL_DIR, 'temp_samples')
    os.makedirs(temp_dir, exist_ok=True)

    base = os.path.basename(pcap_path)
    base = base.replace('.pcapng', '').replace('.pcap', '')
    temp_path = os.path.join(temp_dir, f'{base}_sample{offset}_{n_packets}.pcap')

    if os.path.exists(temp_path):
        print(f"  Using cached: {os.path.basename(temp_path)}")
        return temp_path

    print(f"  Reading {n_packets} packets @ offset {offset}...")
    try:
        reader = PcapReader(pcap_path)
    except Exception:
        # Try pcapng reader
        from scapy.utils import PcapNgReader
        reader = PcapNgReader(pcap_path)

    packets = []
    skipped = 0
    for i, pkt in enumerate(reader):
        if i < offset:
            continue
        if len(packets) >= n_packets:
            break
        # Only keep IP packets
        if pkt and hasattr(pkt, 'haslayer') and pkt.haslayer('IP'):
            packets.append(pkt)
        else:
            skipped += 1

    reader.close()

    print(f"  Collected {len(packets)} IP packets ({skipped} skipped)")
    if not packets:
        print("  [ERROR] No IP packets found!")
        return None

    wrpcap(temp_path, packets)
    print(f"  Saved to {os.path.basename(temp_path)}")
    return temp_path


def analyze_sample(pcap_path, analyzer, cluster_analyzer, sample_label):
    """Run BCDG + clustering on one pcap sample."""
    print(f"\n--- {sample_label} ---")
    try:
        nemesys_result = analyzer.analyze(pcap_path)
    except Exception as e:
        print(f"  [ERROR] BCDG failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    summary = nemesys_result.get('summary', {})
    proto_dist = summary.get('protocol_distribution', {})
    print(f"  Messages: {summary.get('total_messages', 0)}")
    print(f"  Avg fields: {summary.get('avg_fields', 0):.1f}")
    print(f"  Protocols: {dict(sorted(proto_dist.items(), key=lambda x: -x[1])[:8])}")

    # Clustering
    segments = nemesys_result.get('segments', [])
    if not segments:
        print("  No segments, skip")
        return None

    seg_dicts = _segments_to_dicts(segments)
    clustering = cluster_analyzer.analyze(seg_dicts)
    cl_summary = clustering.get('summary', {})

    print(f"  Clusters: {cl_summary.get('total_clusters', 0)}")
    print(f"  Outliers: {cl_summary.get('total_outliers', 0)}")
    print(f"  Risk: {cl_summary.get('risk_level', 'none')}")

    clusters = clustering.get('clusters', [])
    for c in clusters[:5]:
        indicators = c.get('indicators', [])
        flag = f" [{'|'.join(indicators)}]" if indicators else ""
        print(f"    C{c.get('cluster_id')}: {c.get('inferred_name', '?')} "
              f"({c.get('size')}, {c.get('cohesion')}){flag}")

    return {
        'sample_label': sample_label,
        'total_messages': summary.get('total_messages', 0),
        'avg_fields': round(summary.get('avg_fields', 0), 1),
        'protocol_distribution': dict(proto_dist),
        'cluster_count': cl_summary.get('total_clusters', 0),
        'outlier_count': cl_summary.get('total_outliers', 0),
        'risk_level': cl_summary.get('risk_level', 'none'),
        'top_clusters': [
            {'id': c.get('cluster_id'), 'name': c.get('inferred_name'),
             'size': c.get('size'), 'cohesion': c.get('cohesion'),
             'indicators': c.get('indicators', []),
             'outliers': c.get('outlier_count', 0)}
            for c in clusters[:5]
        ],
    }


def main():
    t_start = time.time()
    analyzer = NEMESYSAnalyzer(sigma=0.6)
    cluster_analyzer = ClusterAnalyzer()

    files = [
        ('Wednesday-workingHours.pcap', 'CICIDS2017-Wed'),
        ('Friday-WorkingHours.pcap', 'CICIDS2017-Fri'),
    ]

    all_results = []
    for fname, label in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f'[SKIP] {fname} not found')
            continue

        size_gb = os.path.getsize(path) / (1024**3)
        print(f'\n{"="*60}')
        print(f'  {label} ({size_gb:.1f} GB)')
        print(f'{"="*60}')

        for i in range(3):  # 3 samples from each file
            offset = i * 8000
            temp_pcap = sample_pcap_packets(path, SAMPLE_PACKETS, offset)
            if temp_pcap is None:
                continue

            result = analyze_sample(
                temp_pcap, analyzer, cluster_analyzer,
                f'{label}_sample{i+1}'
            )
            if result:
                all_results.append(result)

            # Clean up temp
            try:
                os.remove(temp_pcap)
            except OSError:
                pass

    # Print summary
    elapsed = time.time() - t_start
    print(f'\n{"="*60}')
    print('  CROSS-DATASET CLUSTERING SUMMARY')
    print(f'{"="*60}')
    print(f'  Time: {elapsed:.0f}s')
    print(f'\n  {"Sample":35s} {"Msgs":>5s} {"Fields":>6s} '
          f'{"Clusters":>9s} {"Outliers":>8s} {"Risk":>6s}')
    print(f'  {"-"*69}')
    for r in all_results:
        print(f'  {r["sample_label"]:35s} {r["total_messages"]:>5d} '
              f'{r["avg_fields"]:>6.1f} {r["cluster_count"]:>9d} '
              f'{r["outlier_count"]:>8d} {r["risk_level"]:>6s}')

    # Save
    out_path = os.path.join(EVAL_DIR, 'cross_dataset_clustering.json')
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'sigma': 0.6,
                'packets_per_sample': SAMPLE_PACKETS,
            },
            'results': all_results,
            'elapsed_seconds': elapsed,
        }, f, indent=2, ensure_ascii=False)

    # Clean up temp dir
    temp_dir = os.path.join(EVAL_DIR, 'temp_samples')
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except OSError:
        pass

    print(f'\n  Saved to {out_path}')
    print(f'  Done in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
