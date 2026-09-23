"""Multi-seed CI for the per-protocol mixed-traffic unknown-protocol detection.

Runs the per-protocol F1/AUC evaluation (from unknown_protocol_detection.py)
over multiple benign-split seeds and reports mean +/- std per protocol/method,
giving confidence bounds for the 4.3 numbers.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from nemesys_gnn4id.nemesys.cluster import cluster_fingerprints

from flow_key_alignment import load_aligned
from unknown_protocol_detection import TARGET_FPR, IN_DOMAIN_ATTACK, OOD_PROTOCOLS

SEEDS = [42, 123, 2024, 7, 99]
BENIGN_TRAIN = 350


def eval_per_protocol(feat, struct, seed):
    rng = np.random.RandomState(seed)
    ben_feats = feat['BenignTraffic.pcap']
    ben_struct = struct.get('BenignTraffic.pcap', [])
    idx = rng.permutation(len(ben_feats))
    tr_idx, te_idx = idx[:BENIGN_TRAIN], idx[BENIGN_TRAIN:]

    X_in = np.array([ben_feats[i] for i in tr_idx] + feat[IN_DOMAIN_ATTACK])
    y_in = np.array([0] * BENIGN_TRAIN + [1] * len(feat[IN_DOMAIN_ATTACK]))
    svm = SVC(kernel='rbf', random_state=seed).fit(X_in, y_in)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X_in, y_in)

    X_struct_in = np.array([ben_struct[i] for i in tr_idx]
                           + struct.get(IN_DOMAIN_ATTACK, []))
    _labels, centroids = cluster_fingerprints(X_struct_in, threshold=0.50)
    centroids = np.array(centroids)

    def struct_score(fps):
        if centroids.size == 0:
            return np.zeros(len(fps))
        d = np.linalg.norm(np.array(fps)[:, None, :] - centroids[None, :, :], axis=2)
        return d.min(axis=1)

    te_ben = [ben_feats[i] for i in te_idx]
    te_ben_struct = [ben_struct[i] for i in te_idx]

    out = {}
    for ood in OOD_PROTOCOLS:
        att_feats = feat[ood]
        att_struct = struct.get(ood, [])
        n_att = min(len(att_feats), len(att_struct))
        att_feats, att_struct = att_feats[:n_att], att_struct[:n_att]
        X_mix = np.array(te_ben + att_feats)
        y_mix = np.array([0] * len(te_ben) + [1] * n_att)
        svm_s = svm.decision_function(X_mix)
        rf_s = rf.predict_proba(X_mix)[:, 1]
        fus_s = np.concatenate([struct_score(te_ben_struct), struct_score(att_struct)])

        def m(scores, y):
            auc = roc_auc_score(y, scores)
            thr = np.quantile(scores[y == 0], 1 - TARGET_FPR)
            pred = (scores > thr).astype(int)
            tp = ((y == 1) & (pred == 1)).sum()
            fp = ((y == 0) & (pred == 1)).sum()
            fn = ((y == 1) & (pred == 0)).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            rec = tp / max(tp + fn, 1)
            return {'f1': round(f1, 4), 'auc': round(auc, 4), 'recall': round(rec, 4)}

        out[ood] = {'SVM': m(svm_s, y_mix), 'RF': m(rf_s, y_mix),
                    'Fusion-struct': m(fus_s, y_mix)}
    return out


def main():
    # key-aligned pair of containers: struct[src][i] is the fingerprint of the very
    # flow described by feat[src][i] (see experiments/flow_key_alignment.py)
    feat, struct = load_aligned()

    # accumulate per protocol per method
    acc = {}
    for ood in OOD_PROTOCOLS:
        acc[ood] = {'SVM': [], 'RF': [], 'Fusion-struct': []}
    for seed in SEEDS:
        r = eval_per_protocol(feat, struct, seed)
        for ood, methods in r.items():
            for mname, mm in methods.items():
                acc[ood][mname].append(mm)

    results = {}
    print(f'{"protocol":<22s} {"Fusion F1 (mean±std)":>22s} {"Fusion AUC (mean±std)":>24s}  '
          f'SVM F1 RF F1')
    for ood in OOD_PROTOCOLS:
        fus_f1 = [x['f1'] for x in acc[ood]['Fusion-struct']]
        fus_auc = [x['auc'] for x in acc[ood]['Fusion-struct']]
        svm_f1 = np.mean([x['f1'] for x in acc[ood]['SVM']])
        rf_f1 = np.mean([x['f1'] for x in acc[ood]['RF']])
        results[ood] = {
            'Fusion-struct': {'f1_mean': round(np.mean(fus_f1), 4),
                              'f1_std': round(np.std(fus_f1), 4),
                              'auc_mean': round(np.mean(fus_auc), 4),
                              'auc_std': round(np.std(fus_auc), 4),
                              'f1_std_err': round(np.std(fus_f1) / len(fus_f1) ** 0.5, 4)},
            'SVM': {'f1_mean': round(svm_f1, 4)},
            'RF': {'f1_mean': round(rf_f1, 4)},
            'n_seeds': len(SEEDS),
        }
        print(f'{ood:<22s} {np.mean(fus_f1):>10.3f}±{np.std(fus_f1):<5.3f} '
              f'{np.mean(fus_auc):>10.3f}±{np.std(fus_auc):<5.3f}  '
              f'{svm_f1:.3f} {rf_f1:.3f}')

    os.makedirs('eval_results/fusion_comparison', exist_ok=True)
    with open('eval_results/fusion_comparison/unknown_protocol_multiseed.json', 'w',
              encoding='utf-8') as f:
        json.dump({'seeds': SEEDS, 'results': results}, f, indent=2, ensure_ascii=False)
    print('Saved: eval_results/fusion_comparison/unknown_protocol_multiseed.json')


if __name__ == '__main__':
    main()
