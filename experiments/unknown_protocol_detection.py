"""Redesigned experiment: unknown-protocol attack detection on MIXED traffic.

For each unknown (OOD) protocol, test = held-out in-domain benign flows +
that protocol's attack flows. Every method is trained/calibrated on in-domain
only (Benign + DDoS). Reports per-protocol AUC (threshold-free separation) and
Recall/FPR/Precision/F1 at a threshold calibrated to ~20% in-domain FPR.

Signals compared:
  - SVM (supervised, 82-dim flow stats): decision_function score
  - Random Forest (supervised, 82-dim): P(attack)
  - Fusion structural (unsupervised, 24-dim BCDG fingerprint): min distance to
    in-domain structural cluster centroids (higher = more anomalous)
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.nemesys.cluster import extract_structural_fingerprint, cluster_fingerprints
from nemesys_gnn4id.pipeline import canonical_flow_tuple

STRUCT_CACHE = '/tmp/upd_struct_flows.json'

# canonical flow-key alignment: feat[src][i] and struct[src][i] describe the SAME flow
from flow_key_alignment import load_aligned

CHUNK_DIR = 'data/data/23pcap_chunks'
IN_DOMAIN_ATTACK = 'DDoS-HTTP_Flood-.pcap'
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
BENIGN_TRAIN = 350  # hold out 150 benign for fair FPR
TARGET_FPR = 0.20


def flow_structural_fingerprints():
    """Return {source: {flow_key: mean 24-dim fingerprint}} via BCDG (cached)."""
    if os.path.exists(STRUCT_CACHE):
        with open(STRUCT_CACHE, encoding='utf-8') as f:
            raw = json.load(f)
        return {src: {tuple(k.split('|')): np.array(v, dtype=np.float32)
                      for k, v in flows.items()}
                for src, flows in raw.items()}
    analyzer = NEMESYSAnalyzer()
    with open(os.path.join(CHUNK_DIR, '_chunk_index.json'), encoding='utf-8') as f:
        chunks = json.load(f)
    out = defaultdict(dict)
    for chunk_name, (_label, src) in chunks.items():
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            continue
        result = analyzer.analyze(chunk_path)
        per_flow = defaultdict(list)
        for seg in result.get('segments', []):
            if not isinstance(seg, dict) or 'src' not in seg:
                continue
            key = canonical_flow_tuple(seg.get('src', ''), seg.get('dst', ''),
                                       seg.get('sport', 0), seg.get('dport', 0),
                                       seg.get('proto', 0))
            per_flow[key].append(extract_structural_fingerprint(seg))
        for key, fps in per_flow.items():
            out[src][key] = np.mean(fps, axis=0)
    save = {src: {'|'.join(str(x) for x in k): v.tolist() for k, v in fv.items()}
            for src, fv in out.items()}
    with open(STRUCT_CACHE, 'w', encoding='utf-8') as f:
        json.dump(save, f)
    return out


def main():
    # key-aligned pair of containers: struct[src][i] is the fingerprint of the very
    # flow described by feat[src][i] (see experiments/flow_key_alignment.py)
    feat, struct = load_aligned()

    # --- split benign into train/test ---
    benign_feats = feat['BenignTraffic.pcap']
    benign_struct = struct.get('BenignTraffic.pcap', [])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(benign_feats))
    tr_idx, te_idx = idx[:BENIGN_TRAIN], idx[BENIGN_TRAIN:]

    # --- supervised: all sklearn baselines on 82-dim, train on train-benign + DDoS ---
    X_in = np.array([benign_feats[i] for i in tr_idx] + feat[IN_DOMAIN_ATTACK])
    y_in = np.array([0] * BENIGN_TRAIN + [1] * len(feat[IN_DOMAIN_ATTACK]))
    classifiers = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'RF': RandomForestClassifier(n_estimators=200, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=4),
        'NaiveBayes': GaussianNB(),
        'C4.5': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    }
    for name, clf in classifiers.items():
        clf.fit(X_in, y_in)

    # --- fusion structural model: cluster in-domain fingerprints, get centroids ---
    X_struct_in = np.array([benign_struct[i] for i in tr_idx]
                           + struct.get(IN_DOMAIN_ATTACK, []))
    _labels, centroids = cluster_fingerprints(X_struct_in, threshold=0.50)
    centroids = np.array(centroids)

    def struct_score(fps):
        """Min distance of each flow fingerprint to in-domain centroids."""
        if centroids.size == 0:
            return np.zeros(len(fps))
        d = np.linalg.norm(np.array(fps)[:, None, :] - centroids[None, :, :], axis=2)
        return d.min(axis=1)

    def eval_protocol(ood):
        ben_test_feats = [benign_feats[i] for i in te_idx]
        ben_test_struct = [benign_struct[i] for i in te_idx]
        att_feats = feat[ood]
        att_struct = struct.get(ood, [])
        n_att = min(len(att_feats), len(att_struct))
        att_feats, att_struct = att_feats[:n_att], att_struct[:n_att]

        # scores for benign + attack
        X_mix = np.array(ben_test_feats + att_feats)
        y_mix = np.array([0] * len(ben_test_feats) + [1] * n_att)
        fus_s = np.concatenate([struct_score(ben_test_struct), struct_score(att_struct)])

        def metrics(score, y):
            auc = roc_auc_score(y, score)
            in_s = score[y == 0]
            thr = np.quantile(in_s, 1 - TARGET_FPR)
            pred = (score > thr).astype(int)
            tp = ((y == 1) & (pred == 1)).sum()
            fp = ((y == 0) & (pred == 1)).sum()
            fn = ((y == 1) & (pred == 0)).sum()
            rec = tp / max(tp + fn, 1); fpr = fp / max(fp + (y == 0).sum() - fp, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            return {'auc': round(auc, 4), 'recall': round(rec, 4), 'fpr': round(fpr, 4),
                    'precision': round(prec, 4), 'f1': round(f1, 4)}

        out = {'Fusion-struct': metrics(fus_s, y_mix),
               'n_benign_test': len(ben_test_feats), 'n_attack': n_att}
        for name, clf in classifiers.items():
            if hasattr(clf, 'decision_function'):
                s = clf.decision_function(X_mix)
            else:
                s = clf.predict_proba(X_mix)[:, 1]
            out[name] = metrics(s, y_mix)
        return out

    results = {}
    order = ['SVM', 'RF', 'KNN', 'NaiveBayes', 'C4.5', 'AdaBoost', 'Fusion-struct']
    print(f'{"protocol":<22s} ' + ''.join(f'{n:>10s}' for n in order))
    print('  F1 (AUC)')
    for ood in OOD_PROTOCOLS:
        r = eval_protocol(ood)
        results[ood] = r
        print(f'{ood:<22s} ' + ''.join(
            f'{r[n]["f1"]:.3f}({r[n]["auc"]:.2f})' for n in order))

    os.makedirs('eval_results/fusion_comparison', exist_ok=True)
    with open('eval_results/fusion_comparison/unknown_protocol_detection.json', 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Saved: eval_results/fusion_comparison/unknown_protocol_detection.json')


if __name__ == '__main__':
    main()
