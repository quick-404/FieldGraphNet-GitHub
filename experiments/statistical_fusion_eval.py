# experiments/statistical_fusion_eval.py
"""Isolation Forest statistical arm for the fusion — feasibility test first."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sklearn.ensemble import IsolationForest
from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer

CHUNK_DIR = 'data/data/23pcap_chunks'
MODEL_PATH = 'models/model.pth'
IN_DOMAIN = ['BenignTraffic.pcap', 'DDoS-HTTP_Flood-.pcap']
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
# SVM targets (supervised oracle) from per_protocol_baselines.json
SVM_TARGETS = {'DNS_Spoofing.pcap': 0.306, 'Mirai-udpplain.pcap': 0.820,
               'SqlInjection.pcap': 0.448, 'DictionaryBruteForce.pcap': 0.392,
               'Recon-PortScan.pcap': 0.652}


def extract_features_by_source():
    """Return {source_pcap: list[82-dim drop-ports feature vectors]}."""
    with open(os.path.join(CHUNK_DIR, '_chunk_index.json'), encoding='utf-8') as f:
        chunks = json.load(f)
    analyzer = GNN4IDAnalyzer(model_path=MODEL_PATH)
    out = {}
    for chunk_name, (_label, src) in chunks.items():
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            continue
        df = analyzer.extract_features(chunk_path, max_flows=50)
        if df.empty or 'flow_features' not in df.columns:
            continue
        for _, row in df.iterrows():
            feat = list(row['flow_features'])
            for z in range(77, min(82, len(feat))):
                feat[z] = 0.0  # drop ports
            out.setdefault(src, []).append(feat)
    return out


def if_anomaly_recall(clf, X_test):
    """Fraction of test flows flagged as anomalies (all are attack => recall)."""
    pred = (clf.predict(X_test) == -1)  # -1 = anomaly
    return float(pred.sum() / max(len(pred), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['feasibility', 'full'], default='feasibility')
    parser.add_argument('--contamination', type=float, default=0.15)
    parser.add_argument('--out', default='eval_results/fusion_comparison/statistical_fusion.json')
    args = parser.parse_args()

    features = extract_features_by_source()
    X_train = np.array([f for s in IN_DOMAIN for f in features.get(s, [])])
    clf = IsolationForest(n_estimators=200, contamination=args.contamination, random_state=42)
    clf.fit(X_train)
    print(f'IF trained on {len(X_train)} in-domain flows (contamination={args.contamination})')

    if args.mode == 'feasibility':
        print('=== FEASIBILITY: IF anomaly recall vs SVM oracle ===')
        for ood in ['Mirai-udpplain.pcap', 'Recon-PortScan.pcap']:
            X_test = np.array(features.get(ood, []))
            rec = if_anomaly_recall(clf, X_test)
            print(f'  {ood}: IF recall={rec:.3f}  SVM={SVM_TARGETS[ood]:.3f}  '
                  f'(delta={rec-SVM_TARGETS[ood]:+.3f})')
        return

    # full mode: per-protocol recall for IF alone, saved for Task 2
    results = {}
    for ood in OOD_PROTOCOLS:
        X_test = np.array(features.get(ood, []))
        results[ood] = {'if_recall': round(if_anomaly_recall(clf, X_test), 4),
                        'n_flows': len(X_test),
                        'svm_target': SVM_TARGETS[ood]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Saved IF-only recall to {args.out}')


if __name__ == '__main__':
    main()
