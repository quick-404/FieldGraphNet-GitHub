"""Full CIC-2023 fusion evaluation with confidence threshold and ablation."""
import sys, os, json, argparse
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


def compute_metrics(y_true, y_pred):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-flows', type=int, default=50)
    parser.add_argument('--max-chunks', type=int, default=30)
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                        help='Min confidence for GNN attack prediction (0.0=disabled)')
    parser.add_argument('--out', default=None,
                        help='output JSON path (default: eval_results/fusion_comparison/'
                             'nemesys_first_eval.json). Use an explicit path when running a '
                             'robustness variant so the published result is not overwritten.')
    args = parser.parse_args()

    index_path = os.path.join(CHUNK_DIR, '_chunk_index.json')
    if not os.path.exists(index_path):
        print(f'ERROR: {index_path} not found')
        sys.exit(1)

    with open(index_path) as f:
        chunks = json.load(f)
    chunk_list = list(chunks.items())

    print(f'Chunks: {len(chunk_list)}, max={args.max_chunks}, max_flows={args.max_flows}')
    print(f'Confidence threshold: {args.confidence_threshold}\n')

    from nemesys_gnn4id.pipeline import TrafficPipeline
    p = TrafficPipeline(device='cpu', model_path=MODEL_PATH, proto_model_path=PROTO_MODEL)

    # Per-source tracking for ablation
    by_source = {}  # source_pcap -> {y_true, y_pred_raw, y_pred_thresh, y_pred_alert, y_pred_cluster}
    all_y_true = []
    all_y_pred_raw = []
    all_y_pred_thresh = []
    all_y_pred_alert = []
    all_y_pred_cluster = []
    chunk_results = []

    processed = 0
    for chunk_name, (attack_label, source_pcap) in chunk_list:
        if processed >= args.max_chunks:
            break
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            continue

        aname = ATTACK_NAMES[attack_label]
        print(f'--- [{processed+1}/{min(args.max_chunks, len(chunk_list))}] {chunk_name} ({aname}) ---')

        result = p.analyze(chunk_path, max_flows=args.max_flows)
        gnn_results = result.get('gnn', {}).get('results', [])
        n = len(gnn_results)
        if n == 0:
            print('  [SKIP] No flows')
            processed += 1
            continue

        y_true = [attack_label] * n
        y_pred_raw = []       # Original GNN class (before OOD override)
        y_pred_thresh = []    # GNN class after confidence threshold
        y_pred_alert = []     # Fusion alert decision
        y_pred_cluster = []   # Clustering-only alert (possible_missed or suspicious)
        ood_flags = []
        confidences = []

        for r in gnn_results:
            orig_class = r.get('_gnn_original_class_id', r['class_id'])
            confidence = r.get('confidence', 0)
            fusion = r.get('fusion_decision', {})
            status = fusion.get('status', '')
            ood = r.get('ood', False)

            # GNN raw
            y_pred_raw.append(orig_class)

            # GNN with confidence threshold
            if orig_class != 0 and confidence < args.confidence_threshold:
                y_pred_thresh.append(0)  # Too low confidence, treat as benign
            else:
                y_pred_thresh.append(orig_class)

            # Fusion alert
            alert = 1 if status in FUSION_ALERT_STATUSES else 0
            y_pred_alert.append(alert)

            # Clustering-only: alert if it's a possible_missed/suspicious (cluster-driven)
            y_pred_cluster.append(1 if status in {
                'suspicious_unknown_protocol_attack',
                'possible_missed_unknown_protocol_attack',
            } else 0)

            ood_flags.append(ood)
            confidences.append(confidence)

        # Compute metrics
        raw_m = compute_metrics(y_true, y_pred_raw)
        thresh_m = compute_metrics(y_true, y_pred_thresh)
        alert_m = compute_metrics(y_true, y_pred_alert)
        cluster_m = compute_metrics(y_true, y_pred_cluster)
        in_domain = sum(1 for f in ood_flags if not f)
        ood_count = sum(1 for f in ood_flags if f)
        avg_conf = sum(confidences) / max(len(confidences), 1)

        cr = {
            'chunk': chunk_name, 'source': source_pcap,
            'attack_label': aname, 'total_flows': n,
            'in_domain': in_domain, 'ood': ood_count,
            'avg_confidence': round(avg_conf, 4),
            'gnn_raw': raw_m,
            'gnn_threshold': thresh_m,
            'fusion_alert': alert_m,
            'cluster_only': cluster_m,
        }
        chunk_results.append(cr)

        # Accumulate
        all_y_true.extend(y_true)
        all_y_pred_raw.extend(y_pred_raw)
        all_y_pred_thresh.extend(y_pred_thresh)
        all_y_pred_alert.extend(y_pred_alert)
        all_y_pred_cluster.extend(y_pred_cluster)

        # Per-source
        by_source.setdefault(source_pcap, {'y_true': [], 'y_pred_raw': [], 'y_pred_alert': []})
        by_source[source_pcap]['y_true'].extend(y_true)
        by_source[source_pcap]['y_pred_raw'].extend(y_pred_raw)
        by_source[source_pcap]['y_pred_alert'].extend(y_pred_alert)

        print(f'  Flows={n} in={in_domain} ood={ood_count} avg_conf={avg_conf:.3f}')
        print(f'  GNN raw:  FPR={raw_m["fpr"]:.4f} Recall={raw_m["recall"]:.4f} F1={raw_m["f1"]:.4f}')
        if args.confidence_threshold > 0:
            print(f'  GNN(thr): FPR={thresh_m["fpr"]:.4f} Recall={thresh_m["recall"]:.4f} F1={thresh_m["f1"]:.4f}')
        print(f'  Fusion:   FPR={alert_m["fpr"]:.4f} Recall={alert_m["recall"]:.4f} F1={alert_m["f1"]:.4f}')
        print()
        processed += 1

    # Aggregate
    agg_raw = compute_metrics(all_y_true, all_y_pred_raw)
    agg_thresh = compute_metrics(all_y_true, all_y_pred_thresh)
    agg_alert = compute_metrics(all_y_true, all_y_pred_alert)
    agg_cluster = compute_metrics(all_y_true, all_y_pred_cluster)

    # Per-source aggregates
    source_agg = {}
    for src, data in by_source.items():
        source_agg[src] = {
            'gnn_raw': compute_metrics(data['y_true'], data['y_pred_raw']),
            'fusion_alert': compute_metrics(data['y_true'], data['y_pred_alert']),
            'total_flows': len(data['y_true']),
        }

    summary = {
        'config': {
            'max_chunks': args.max_chunks,
            'max_flows_per_chunk': args.max_flows,
            'confidence_threshold': args.confidence_threshold,
            'alert_statuses': list(FUSION_ALERT_STATUSES),
        },
        'total_chunks': len(chunk_results),
        'total_flows': len(all_y_true),
        'ablation': {
            'gnn_raw': agg_raw,
            'gnn_threshold': agg_thresh,
            'cluster_only': agg_cluster,
            'fusion_alert': agg_alert,
        },
        'by_source': source_agg,
        'chunk_results': chunk_results,
    }

    summary_path = args.out or os.path.join(EVAL_DIR, 'nemesys_first_eval.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('=' * 60)
    print('ABLATION TABLE')
    print('=' * 60)
    print(f'Total flows: {len(all_y_true)}')
    print(f'{"Method":25s} {"FPR":>8s} {"Recall":>8s} {"F1":>8s}')
    print('-' * 50)
    for name, m in [('GNN raw', agg_raw), ('GNN + threshold', agg_thresh),
                    ('Cluster only', agg_cluster), ('Fusion alert', agg_alert)]:
        print(f'{name:25s} {m["fpr"]:>8.4f} {m["recall"]:>8.4f} {m["f1"]:>8.4f}')
    print()
    print(f'Saved to {summary_path}')

    if args.confidence_threshold == 0:
        print()
        print('TIP: Use --confidence-threshold 0.5 to filter low-confidence GNN predictions')


if __name__ == '__main__':
    main()
