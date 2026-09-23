"""
Ablation study: isolate contributions of each component in FieldGraph-Net.

Four variants (computed from a single full pipeline run per chunk):
  A1 (GNN raw):        Raw GNN4ID predictions on ALL flows, no domain routing.
  A2 (+Domain route):  GNN on in-domain flows; ALL OOD flows flagged as alert.
  A3 (+Clustering):    Cluster anomaly signal on ALL flows (no domain routing).
  A4 (Full fusion):    Complete system: domain routing + clustering fusion.

All data from real runs — no fabricated results.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

CHUNK_DIR = 'data/data/23pcap_chunks'
EVAL_DIR = 'eval_results/fusion_comparison'
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = 'models/model.pth'
# W0-5: downstream rerun with FieldProtoGNN_V2 (acceptance EFFECTIVE, 12-class acc 0.9944).
# Old artifact field_proto_model_bcdg.pth stays untouched at repo root.
# NEMESYS_PROTO_MODEL env var allows a same-pipeline baseline run with the old model.
PROTO_MODEL = os.environ.get('NEMESYS_PROTO_MODEL', 'models/field_proto_model_v2.pth')

ATTACK_NAMES = {
    0: 'Benign', 1: 'WebBased', 2: 'Spoofing', 3: 'Recon',
    4: 'Mirai', 5: 'Dos', 6: 'DDos', 7: 'BruteForce',
}

FUSION_ALERT_STATUSES = {
    'confirmed_gnn_attack',
    'suspicious_unknown_protocol_attack',
    'possible_missed_unknown_protocol_attack',
}


def binary_metrics(y_true, y_pred):
    """Attack-vs-normal binary metrics. Any class_id > 0 is attack."""
    n = min(len(y_true), len(y_pred))
    tp = sum(1 for i in range(n) if y_true[i] > 0 and y_pred[i] > 0)
    fp = sum(1 for i in range(n) if y_true[i] == 0 and y_pred[i] > 0)
    fn = sum(1 for i in range(n) if y_true[i] > 0 and y_pred[i] == 0)
    tn = sum(1 for i in range(n) if y_true[i] == 0 and y_pred[i] == 0)
    return {
        'total': n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': round(tp / max(tp + fp, 1), 4),
        'recall': round(tp / max(tp + fn, 1), 4),
        'f1': round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        'fpr': round(fp / max(fp + tn, 1), 4),
    }


def cluster_is_high_risk(cluster_info):
    """Replicate TrafficPipeline._cluster_is_high_risk logic."""
    if not cluster_info:
        return False
    indicators = cluster_info.get('indicators', []) or []
    if cluster_info.get('outlier_count', 0) > 0:
        return True
    high_risk_flags = {'未知协议', '孤立小簇', '高离群率', '高熵(加密/混淆)'}
    outlier_count = cluster_info.get('outlier_count', 0)
    cluster_size = cluster_info.get('size', 1)
    if outlier_count / max(cluster_size, 1) > 0.15:
        return True
    cohesion = cluster_info.get('cohesion', '')
    risk_flags = [f for f in indicators if f in high_risk_flags]
    if risk_flags and cohesion in ('loose', 'moderate'):
        return True
    return False


def main():
    import time
    t_start = time.time()

    index_path = os.path.join(CHUNK_DIR, '_chunk_index.json')
    with open(index_path) as f:
        chunks = json.load(f)
    chunk_list = list(chunks.items())

    from nemesys_gnn4id.pipeline import TrafficPipeline
    p = TrafficPipeline(device='cpu', model_path=MODEL_PATH,
                        proto_model_path=PROTO_MODEL)

    # Accumulators for each variant (binary attack vs normal)
    all_y_true = []
    all_a1 = []  # GNN raw
    all_a2 = []  # +Domain route (flag all OOD)
    all_a3 = []  # +Clustering only
    all_a4 = []  # Full fusion

    # Per-source tracking for finer analysis
    by_source = {}

    # Per-chunk metrics for bootstrapping
    chunk_metrics = []

    processed = 0
    total_chunks = len(chunk_list)
    for chunk_name, (attack_label, source_pcap) in chunk_list:
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            continue

        attack_name = ATTACK_NAMES[attack_label]
        processed += 1
        print(f'[{processed}/{total_chunks}] {chunk_name} ({attack_name})  ',
              end='', flush=True)

        result = p.analyze(chunk_path, max_flows=50)
        gnn_results = result.get('gnn', {}).get('results', [])
        clustering = result.get('clustering', {})

        # Build cluster risk map
        cluster_risk_map = {}
        for c in clustering.get('clusters', []):
            cid = c.get('cluster_id')
            if cid is not None:
                cluster_risk_map[cid] = cluster_is_high_risk(c)

        n = len(gnn_results)
        if n == 0:
            print('0 flows, skip')
            continue

        y_true = [attack_label] * n

        # Per-flow variants
        for r in gnn_results:
            orig_class = r.get('_gnn_original_class_id', r['class_id'])
            ood = r.get('ood', False)
            fusion = r.get('fusion_decision', {})
            status = fusion.get('status', '')
            cluster_id = fusion.get('cluster_id')

            # A1: GNN raw — use original class (before OOD override)
            all_a1.append(0 if orig_class == 0 else 1)

            # A2: +Domain route — flag ALL OOD flows as alert
            if ood:
                all_a2.append(1)
            else:
                all_a2.append(0 if orig_class == 0 else 1)

            # A3: +Clustering only — check cluster risk regardless of domain
            if cluster_id is not None and cluster_risk_map.get(cluster_id, False):
                all_a3.append(1)
            else:
                all_a3.append(0)

            # A4: Full fusion
            all_a4.append(1 if status in FUSION_ALERT_STATUSES else 0)

        all_y_true.extend(y_true)

        # Per-source accumulators
        by_source.setdefault(source_pcap, {
            'y_true': [], 'a1': [], 'a2': [], 'a3': [], 'a4': [],
        })
        d = by_source[source_pcap]
        d['y_true'].extend(y_true)
        d['a1'].extend(all_a1[-n:])
        d['a2'].extend(all_a2[-n:])
        d['a3'].extend(all_a3[-n:])
        d['a4'].extend(all_a4[-n:])

        # Save per-chunk metrics for bootstrapping
        chunk_metrics.append({
            'chunk': chunk_name,
            'source': source_pcap,
            'attack_label': attack_name,
            'n': n,
            'a1': binary_metrics(y_true, all_a1[-n:]),
            'a2': binary_metrics(y_true, all_a2[-n:]),
            'a3': binary_metrics(y_true, all_a3[-n:]),
            'a4': binary_metrics(y_true, all_a4[-n:]),
        })

        print(f'{n} flows')

    # ---- Aggregate metrics ----
    m_a1 = binary_metrics(all_y_true, all_a1)
    m_a2 = binary_metrics(all_y_true, all_a2)
    m_a3 = binary_metrics(all_y_true, all_a3)
    m_a4 = binary_metrics(all_y_true, all_a4)

    elapsed = time.time() - t_start

    # Print table
    print('\n' + '=' * 70)
    print('  ABLATION STUDY — Component Contribution Analysis')
    print('=' * 70)
    print(f'  Total flows: {len(all_y_true)}  |  '
          f'Chunks: {processed}  |  Time: {elapsed:.0f}s')
    print()
    print(f'  {"#":4s} {"Method":30s} {"FPR":>8s} {"Recall":>8s} {"F1":>8s}')
    print(f'  {"-" * 4} {"-" * 30} {"-" * 8} {"-" * 8} {"-" * 8}')
    variants = [
        ('A1', 'GNN raw (no routing)', m_a1),
        ('A2', '+Domain route (flag all OOD)', m_a2),
        ('A3', '+Clustering only (no routing)', m_a3),
        ('A4', 'Full fusion', m_a4),
    ]
    for tag, name, m in variants:
        print(f'  {tag:4s} {name:30s} {m["fpr"]:>8.4f} {m["recall"]:>8.4f} {m["f1"]:>8.4f}')

    # Delta rows
    print(f'  {"-" * 4} {"-" * 30} {"-" * 8} {"-" * 8} {"-" * 8}')
    print(f'  {"":4s} {"A2-A1 (routing gain)":30s} '
          f'{m_a2["fpr"]-m_a1["fpr"]:>+8.4f} '
          f'{m_a2["recall"]-m_a1["recall"]:>+8.4f} '
          f'{m_a2["f1"]-m_a1["f1"]:>+8.4f}')
    print(f'  {"":4s} {"A3-A1 (clustering gain)":30s} '
          f'{m_a3["fpr"]-m_a1["fpr"]:>+8.4f} '
          f'{m_a3["recall"]-m_a1["recall"]:>+8.4f} '
          f'{m_a3["f1"]-m_a1["f1"]:>+8.4f}')
    print(f'  {"":4s} {"A4-A1 (full system gain)":30s} '
          f'{m_a4["fpr"]-m_a1["fpr"]:>+8.4f} '
          f'{m_a4["recall"]-m_a1["recall"]:>+8.4f} '
          f'{m_a4["f1"]-m_a1["f1"]:>+8.4f}')

    # Per-source table
    print(f'\n  {"Source":25s} {"Flows":>6s} {"A1-Rec":>8s} {"A2-Rec":>8s} '
          f'{"A3-Rec":>8s} {"A4-Rec":>8s}')
    print(f'  {"-" * 55}')
    for src, d in sorted(by_source.items()):
        name = src.replace('.pcap', '')[:24]
        a1 = binary_metrics(d['y_true'], d['a1'])
        a2 = binary_metrics(d['y_true'], d['a2'])
        a3 = binary_metrics(d['y_true'], d['a3'])
        a4 = binary_metrics(d['y_true'], d['a4'])
        print(f'  {name:25s} {len(d["y_true"]):>6d} '
              f'{a1["recall"]:>8.4f} {a2["recall"]:>8.4f} '
              f'{a3["recall"]:>8.4f} {a4["recall"]:>8.4f}')

    # Save for paper
    results = {
        'config': {'total_chunks': processed, 'total_flows': len(all_y_true)},
        'ablation': {
            'gnn_raw': m_a1,
            'domain_routing_only': m_a2,
            'clustering_only': m_a3,
            'fusion_alert': m_a4,
        },
        'by_source': {
            src: {
                'total_flows': len(d['y_true']),
                'gnn_raw': binary_metrics(d['y_true'], d['a1']),
                'domain_routing_only': binary_metrics(d['y_true'], d['a2']),
                'clustering_only': binary_metrics(d['y_true'], d['a3']),
                'fusion_alert': binary_metrics(d['y_true'], d['a4']),
            }
            for src, d in sorted(by_source.items())
        },
        'chunk_metrics': chunk_metrics,
    }

    out_path = os.environ.get('NEMESYS_ABLATION_OUT') or os.path.join(
        EVAL_DIR, 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved to {out_path}')
    print(f'  Done in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
