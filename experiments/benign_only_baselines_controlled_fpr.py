# -*- coding: utf-8 -*-
"""Control experiment: supervised baselines trained WITHOUT any attack labels.

Motivation (reviewer-proofing): the main controlled-FPR comparison trains SVM/RF/KNN
on in-domain Benign + DDoS (i.e. with one attack class labelled). A reviewer may ask
whether that gives supervised baselines an unfair edge over the label-free structural
fusion. This script answers it directly by training the SAME baselines on Benign ONLY
(no attack labels at all), which is the strictly label-free setting — the same
information budget the structural arm has.

Expected (honest): with only one class, discriminative classifiers degenerate; their
OOD detection performance drops sharply. That would demonstrate the supervised
baselines' advantage comes from attack labels, not from a superior signal.

Same split / preprocessing / 20%-FPR calibration as unknown_protocol_detection.py.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from nemesys_gnn4id.nemesys.cluster import cluster_fingerprints
from flow_key_alignment import load_aligned
from unknown_protocol_detection import BENIGN_TRAIN, TARGET_FPR

IN_DOMAIN_ATTACK = 'DDoS-HTTP_Flood-.pcap'
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
OUT = 'eval_results/fusion_comparison/benign_only_baselines_controlled_fpr.json'


def metrics(score, y):
    auc = roc_auc_score(y, score)
    in_s = score[y == 0]
    thr = np.quantile(in_s, 1 - TARGET_FPR)
    pred = (score > thr).astype(int)
    tp = ((y == 1) & (pred == 1)).sum(); fp = ((y == 0) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum()
    rec = tp / max(tp + fn, 1); fpr = fp / max(int((y == 0).sum()), 1)
    prec = tp / max(tp + fp, 1); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {'auc': round(float(auc), 4), 'recall': round(float(rec), 4),
            'fpr': round(float(fpr), 4), 'precision': round(float(prec), 4),
            'f1': round(float(f1), 4)}


def main():
    # key-aligned pair of containers: struct[src][i] is the fingerprint of the very
    # flow described by feat[src][i] (see experiments/flow_key_alignment.py)
    feat, struct = load_aligned()
    benign_feats = feat['BenignTraffic.pcap']
    benign_struct = struct.get('BenignTraffic.pcap', [])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(benign_feats))
    tr_idx, te_idx = idx[:BENIGN_TRAIN], idx[BENIGN_TRAIN:]

    X_benign = np.array([benign_feats[i] for i in tr_idx])  # benign only, NO attack labels

    # Benign-only (strictly label-free) detectors — the fair comparison to the
    # label-free structural arm.
    detectors = {
        # one-class / unsupervised anomaly detectors trained on benign only
        'OneClassSVM': OneClassSVM(kernel='rbf', nu=0.1),
        'IsolationForest': IsolationForest(n_estimators=200, random_state=42),
        'LOF': LocalOutlierFactor(n_neighbors=20, novelty=True),
    }
    for name, d in detectors.items():
        d.fit(X_benign)
    print(f'benign-only detectors trained on {len(X_benign)} flows (no attack labels)')

    # structural arm (same as fusion): centroids from benign-only fingerprints
    _lbl, centroids = cluster_fingerprints(np.array([benign_struct[i] for i in tr_idx]), threshold=0.50)
    centroids = np.array(centroids)

    def struct_score(fps):
        if centroids.size == 0:
            return np.zeros(len(fps))
        d = np.linalg.norm(np.array(fps)[:, None, :] - centroids[None, :, :], axis=2)
        return d.min(axis=1)

    results = {}
    for ood in OOD_PROTOCOLS:
        ben_test_feats = [benign_feats[i] for i in te_idx]
        ben_test_struct = [benign_struct[i] for i in te_idx]
        att_struct = struct.get(ood, [])
        n_att = min(len(feat[ood]), len(att_struct))
        att_struct = att_struct[:n_att]
        X_mix = np.array(ben_test_feats + feat[ood][:n_att])
        y_mix = np.array([0] * len(ben_test_feats) + [1] * n_att)

        fus = np.concatenate([struct_score(ben_test_struct), struct_score(att_struct)])
        out = {'n_benign_test': len(ben_test_feats), 'n_attack': n_att,
               'Fusion-struct (label-free)': metrics(fus, y_mix)}
        for name, d in detectors.items():
            if hasattr(d, 'decision_function'):
                s = -d.decision_function(X_mix)   # higher = more anomalous
            else:
                s = -d.score_samples(X_mix)
            out[name + ' (benign-only)'] = metrics(s, y_mix)
        results[ood] = out
        print(f'{ood:<24} fusion f1={out["Fusion-struct (label-free)"]["f1"]:.3f} | '
              + ' '.join(f'{k.split()[0]}={out[k]["f1"]:.3f}' for k in out if 'benign-only' in k))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
