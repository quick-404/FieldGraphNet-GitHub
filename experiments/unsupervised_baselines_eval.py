# experiments/unsupervised_baselines_eval.py
"""k-means and DBSCAN unsupervised baselines for per-protocol OOD comparison.

Both are unsupervised (no attack labels), trained on in-domain 82-dim drop-ports
flow features, and flag OOD flows as anomalies via a distance threshold
calibrated on in-domain (target FPR ~20%). Compared against the fusion's
structural-clustering recall (from per_protocol_baselines.json).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sklearn.cluster import KMeans, DBSCAN

# reuse the feature extractor from the statistical-arm script
from statistical_fusion_eval import extract_features_by_source

CHUNK_DIR = 'data/data/23pcap_chunks'
IN_DOMAIN = ['BenignTraffic.pcap', 'DDoS-HTTP_Flood-.pcap']
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
K_CLUSTERS = 10
DBSCAN_EPS = 3.0
TARGET_FPR = 0.20


def min_dist_to_centroids(X, centroids):
    """Per-row min Euclidean distance to the k-means centroids."""
    d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return d.min(axis=1)


def kmeans_baseline(X_train, X_test):
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit(X_train)
    train_d = min_dist_to_centroids(X_train, km.cluster_centers_)
    thr = np.quantile(train_d, 1 - TARGET_FPR)  # ~20% of in-domain flagged
    test_d = min_dist_to_centroids(X_test, km.cluster_centers_)
    return (test_d > thr).astype(int), float(thr)


def dbscan_baseline(X_train, X_test):
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=5).fit(X_train)
    # use all in-domain points as the density model; flag OOD flows far from them
    train_d = np.linalg.norm(X_train[:, None, :] - X_test[None, :, :], axis=2)
    # nearest in-domain distance per OOD flow
    test_nn = train_d.min(axis=0)
    # calibrate threshold on in-domain self-nearest-neighbor distances
    self_d = np.linalg.norm(X_train[:, None, :] - X_train[None, :, :], axis=2)
    np.fill_diagonal(self_d, np.inf)
    thr = np.quantile(self_d.min(axis=1), 1 - TARGET_FPR)
    return (test_nn > thr).astype(int), float(thr)


def main():
    features = extract_features_by_source()
    X_in = np.array([f for s in IN_DOMAIN for f in features.get(s, [])])

    with open('eval_results/fusion_comparison/per_protocol_baselines.json', encoding='utf-8') as f:
        baselines = json.load(f)

    results = {}
    print(f'{"protocol":<24s} {"kmeans":>7s} {"dbscan":>7s} {"fusion":>7s}  kmeans>=fus  dbscan>=fus')
    for ood in OOD_PROTOCOLS:
        X_test = np.array(features.get(ood, []))
        n = len(X_test)
        km_flag, km_thr = kmeans_baseline(X_in, X_test)
        db_flag, db_thr = dbscan_baseline(X_in, X_test)
        km_rec = float(km_flag.sum() / n)
        db_rec = float(db_flag.sum() / n)
        fus_rec = baselines[ood.replace('.pcap', '')]['FieldGraph-Net (Fusion)']['recall']
        results[ood] = {
            'n_flows': n,
            'kmeans_recall': round(km_rec, 4),
            'dbscan_recall': round(db_rec, 4),
            'kmeans_threshold': round(km_thr, 3),
            'dbscan_threshold': round(db_thr, 3),
            'target_fpr': TARGET_FPR,
            'fusion_recall': round(fus_rec, 4),
            'kmeans_ge_fusion': km_rec >= fus_rec,
            'dbscan_ge_fusion': db_rec >= fus_rec,
        }
        print(f'{ood:<24s} {km_rec:>7.3f} {db_rec:>7.3f} {fus_rec:>7.3f}  '
              f'{results[ood]["kmeans_ge_fusion"]!s:<9} {results[ood]["dbscan_ge_fusion"]!s}')

    os.makedirs('eval_results/fusion_comparison', exist_ok=True)
    with open('eval_results/fusion_comparison/unsupervised_baselines.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Saved: eval_results/fusion_comparison/unsupervised_baselines.json')


if __name__ == '__main__':
    main()
