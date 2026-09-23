"""
Cross-dataset validation on CICIDS2017.

Runs the full FieldGraph-Net pipeline on 2017 pcap data and reports:
1. Protocol identification distribution (FieldProtoGNN generalization)
2. Structural cluster analysis (anomaly detection)
3. GNN4ID predictions (distribution shift)

Note: Per-flow labels are not available in the CICIDS2017 CSVs (no IPs).
Validation is qualitative: does the system detect structural anomalies in
a completely different dataset from its training distribution?
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

DATA_DIR = 'data/data/17pcap'
EVAL_DIR = 'eval_results/fusion_comparison'
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = 'models/model.pth'
PROTO_MODEL = 'field_proto_model_bcdg.pth'

# CSV label summaries for context (from MachineLearningCVE)
PCAP_LABELS = {
    'Wednesday-workingHours.pcap': 'DoS (Hulk+GoldenEye+slowloris+Slowhttptest+Heartbleed)',
    'Friday-WorkingHours.pcap': 'Morning:Bot; Afternoon:DDoS+PortScan',
}

PCAP_DESCRIPTIONS = {
    'Wednesday-workingHours.pcap': 'CICIDS2017 Wed = DoS attacks on IT infrastructure',
    'Friday-WorkingHours.pcap': 'CICIDS2017 Fri = Bot, DDoS, PortScan attacks',
}

FUSION_ALERT_STATUSES = {
    'confirmed_gnn_attack',
    'suspicious_unknown_protocol_attack',
    'possible_missed_unknown_protocol_attack',
}


def main():
    import time
    t_start = time.time()

    from nemesys_gnn4id.pipeline import TrafficPipeline
    p = TrafficPipeline(device='cpu', model_path=MODEL_PATH,
                        proto_model_path=PROTO_MODEL)

    pcap_files = [
        'Wednesday-workingHours.pcap',
        'Friday-WorkingHours.pcap',
    ]

    results = {}

    for pcap_name in pcap_files:
        pcap_path = os.path.join(DATA_DIR, pcap_name)
        if not os.path.exists(pcap_path):
            print(f'[SKIP] {pcap_name} not found')
            continue

        pcap_size_gb = os.path.getsize(pcap_path) / (1024**3)
        print(f'\n{"="*70}')
        print(f'  Processing: {pcap_name} ({pcap_size_gb:.1f} GB)')
        print(f'  Labels: {PCAP_LABELS.get(pcap_name, "unknown")}')
        print(f'{"="*70}')

        # Run pipeline with limited flows
        try:
            result = p.analyze(pcap_path, max_flows=50)
        except Exception as e:
            print(f'  [ERROR] Pipeline failed: {e}')
            import traceback
            traceback.print_exc()
            continue

        gnn_results = result.get('gnn', {}).get('results', [])
        protocol_inference = result.get('protocol_inference', {}) or {}
        clustering = result.get('clustering', {}) or {}

        # --- Protocol identification ---
        pi_summary = protocol_inference.get('summary', {})
        proto_dist = pi_summary.get('protocol_distribution', {})

        print(f'\n  --- FieldProtoGNN Protocol Distribution ---')
        print(f'  Known: {pi_summary.get("known_messages", 0)} msgs, '
              f'Unknown: {pi_summary.get("unknown_messages", 0)} msgs')
        if proto_dist:
            for proto, count in sorted(proto_dist.items(), key=lambda x: -x[1]):
                print(f'    {proto}: {count}')

        # --- GNN results ---
        ood_count = sum(1 for r in gnn_results if r.get('ood', False))
        in_domain_count = sum(1 for r in gnn_results if not r.get('ood', False))
        attack_count = sum(1 for r in gnn_results
                           if r.get('_gnn_original_class_id', r['class_id']) != 0)

        print(f'\n  --- GNN4ID Predictions (trained on CIC IoT 2023) ---')
        print(f'  Total flows: {len(gnn_results)}')
        print(f'  In-domain: {in_domain_count}  |  OOD: {ood_count}')
        print(f'  GNN predicted attacks: {attack_count}')

        if gnn_results:
            # Show top classes predicted
            from collections import Counter
            class_counter = Counter()
            for r in gnn_results:
                cls = r.get('class_name', 'unknown')
                class_counter[cls] += 1
            print(f'  GNN class distribution:')
            for cls, cnt in class_counter.most_common():
                print(f'    {cls}: {cnt}')

        # --- Clustering analysis ---
        cl_summary = clustering.get('summary', {})
        clusters = clustering.get('clusters', [])

        print(f'\n  --- Structural Clustering ---')
        print(f'  Segments: {cl_summary.get("total_messages", 0)}')
        print(f'  Clusters: {cl_summary.get("total_clusters", 0)}')
        print(f'  Outliers: {cl_summary.get("total_outliers", 0)}')
        print(f'  Risk level: {cl_summary.get("risk_level", "none")}')

        # Fusion decisions summary
        fusion_counts = {}
        for r in gnn_results:
            status = r.get('fusion_decision', {}).get('status', 'none')
            fusion_counts[status] = fusion_counts.get(status, 0) + 1

        alert_count = sum(
            cnt for status, cnt in fusion_counts.items()
            if status in FUSION_ALERT_STATUSES
        )

        print(f'\n  --- Fusion Decisions ---')
        print(f'  Total alerts: {alert_count}/{len(gnn_results)}')
        for status, cnt in sorted(fusion_counts.items(), key=lambda x: -x[1]):
            print(f'    {status}: {cnt}')

        # Cluster details
        if clusters:
            print(f'\n  Top clusters:')
            for c in clusters[:5]:
                cid = c.get('cluster_id', 0)
                size = c.get('size', 0)
                name = c.get('inferred_name', 'unknown')
                cohesion = c.get('cohesion', '')
                indicators = c.get('indicators', [])
                flag = f" [{'|'.join(indicators)}]" if indicators else ""
                outliers = c.get('outlier_count', 0)
                out_str = f', outliers={outliers}' if outliers else ''
                print(f'    Cluster {cid}: {name} ({size}{out_str}, {cohesion}){flag}')

        # Collect results
        results[pcap_name] = {
            'pcap_size_gb': round(pcap_size_gb, 1),
            'label_context': PCAP_LABELS.get(pcap_name, ''),
            'total_flows': len(gnn_results),
            'in_domain': in_domain_count,
            'ood': ood_count,
            'gnn_attack_predictions': attack_count,
            'protocol_distribution': proto_dist,
            'cluster_risk_level': cl_summary.get('risk_level', 'none'),
            'cluster_count': cl_summary.get('total_clusters', 0),
            'cluster_outliers': cl_summary.get('total_outliers', 0),
            'fusion_alerts': alert_count,
            'fusion_decisions': dict(fusion_counts),
            'top_clusters': [
                {
                    'id': c.get('cluster_id'),
                    'name': c.get('inferred_name'),
                    'size': c.get('size'),
                    'cohesion': c.get('cohesion'),
                    'indicators': c.get('indicators', []),
                    'outliers': c.get('outlier_count', 0),
                }
                for c in clusters[:5]
            ],
        }

    elapsed = time.time() - t_start

    print(f'\n{"="*70}')
    print('  CROSS-DATASET VALIDATION SUMMARY')
    print(f'{"="*70}')
    print(f'  Time: {elapsed:.0f}s')
    print()
    print(f'  {"File":40s} {"Flows":>6s} {"OOD":>5s} {"Alerts":>6s} '
          f'{"Clusters":>9s} {"Risk":>6s}')
    print(f'  {"-"*72}')
    for pcap_name, r in results.items():
        short_name = pcap_name[:39]
        print(f'  {short_name:40s} {r["total_flows"]:>6d} {r["ood"]:>5d} '
              f'{r["fusion_alerts"]:>6d} {r["cluster_count"]:>9d} '
              f'{r["cluster_risk_level"]:>6s}')

    # Save
    out_path = os.path.join(EVAL_DIR, 'cross_dataset_validation.json')
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'model_path': MODEL_PATH,
                'proto_model': PROTO_MODEL,
                'max_flows': 50,
            },
            'results': results,
            'elapsed_seconds': elapsed,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved to {out_path}')
    print(f'  Done in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
