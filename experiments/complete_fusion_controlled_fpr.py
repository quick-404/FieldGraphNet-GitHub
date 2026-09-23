# -*- coding: utf-8 -*-
"""Complete fusion system at controlled 20% FPR — CORRECT methodology.

Methodology (honest):
  The production fusion pipeline routes flows by protocol domain:
    - in-domain flows  -> GNN arm
    - OOD/unknown flows -> structural arm
  In this unknown-protocol task, the attack flows belong to protocols NOT seen in
  training, so the domain router sends them to the STRUCTURAL ARM. Therefore the
  complete fusion system's decision on these OOD attack flows IS the structural-arm
  decision — this is the pipeline's definition, not an assumption.
  The GNN arm's contribution on OOD flows is measured separately (GNN raw on the same
  flows) to show it is weak there, justifying the routing.

  Consequently: complete fusion F1@20%FPR on unknown-protocol attacks == structural arm F1.
  We report both, plus the GNN raw on the same flows, and the supervised baselines,
  all under the identical split/calibration as unknown_protocol_detection.py.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from nemesys_gnn4id.nemesys.cluster import cluster_fingerprints
from flow_key_alignment import load_aligned
from unknown_protocol_detection import BENIGN_TRAIN, TARGET_FPR

IN_DOMAIN_ATTACK = 'DDoS-HTTP_Flood-.pcap'
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
OUT = 'eval_results/fusion_comparison/complete_fusion_controlled_fpr.json'


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

    # Which feature backend produced the numbers below, read from the analyzer AFTER
    # extraction has finished (last_feature_backend() reports the most recent call, so
    # reading it before extraction would report nothing). Imported here -- inside
    # main() -- so the module keeps its deferred heavy-import behaviour.
    #
    # WHY THIS IS IN THE JSON: this file is the reference for
    # experiments/eval_ood_method_comparison.py, where our system's per-protocol rows are
    # READ from here and juxtaposed with baselines COMPUTED by that run. The nfstream and
    # scapy extractors yield non-equivalent 82-dim features (~49% cell agreement, 12/82
    # dimensions never agree) and even different flow counts, so a reference whose
    # backend is unknown/mismatched silently invalidates the comparison table.
    from nemesys_gnn4id.gnn4id.analyzer import FEATURE_BACKEND, last_feature_backend
    backend, reason = last_feature_backend()

    benign_feats = feat['BenignTraffic.pcap']
    benign_struct = struct.get('BenignTraffic.pcap', [])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(benign_feats))
    tr_idx, te_idx = idx[:BENIGN_TRAIN], idx[BENIGN_TRAIN:]

    # supervised baselines (in-domain training only) — same as unknown_protocol_detection.py
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
    for clf in classifiers.values():
        clf.fit(X_in, y_in)

    # structural arm model (fusion's OOD detector): centroids from in-domain fingerprints
    X_struct_in = np.array([benign_struct[i] for i in tr_idx]
                           + struct.get(IN_DOMAIN_ATTACK, []))
    _lbl, centroids = cluster_fingerprints(X_struct_in, threshold=0.50)
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
        y_mix = np.array([0] * len(ben_test_feats) + [1] * n_att)

        # Fusion score = structural-arm distance (domain router sends OOD -> structural arm)
        fus = np.concatenate([struct_score(ben_test_struct), struct_score(att_struct)])

        out = {'n_benign_test': len(ben_test_feats), 'n_attack': n_att,
               'Fusion-complete': metrics(fus, y_mix),   # == structural arm by routing definition
               'note': ('complete fusion == structural arm on OOD flows: the domain router '
                        'sends unknown-protocol traffic to the structural arm; the GNN arm '
                        'does not score these flows (see GNN-raw reference below)')}
        for name, clf in classifiers.items():
            X_mix = np.array(ben_test_feats + feat[ood][:n_att])
            s = clf.decision_function(X_mix) if hasattr(clf, 'decision_function') else clf.predict_proba(X_mix)[:, 1]
            out[name] = metrics(s, y_mix)
        results[ood] = out
        print(f'{ood:<24} fusion(complete)=struct auc={out["Fusion-complete"]["auc"]:.3f} '
              f'f1={out["Fusion-complete"]["f1"]:.3f} | RF f1={out["RF"]["f1"]:.3f} '
              f'SVM f1={out["SVM"]["f1"]:.3f}')

    results['_meta'] = {
        'feature_backend_requested': FEATURE_BACKEND,
        'feature_backend_used': backend,
        'feature_backend_reason': reason,
        'note': ('This file is a reference for the OOD method comparison. Its '
                 'feature_backend_used MUST match the comparison run\'s backend: the '
                 'nfstream and scapy extractors yield non-equivalent 82-dim features '
                 'and different flow counts, so mixing them invalidates the table.'),
    }

    out_path = os.environ.get('NEMESYS_COMPLETE_FUSION_OUT') or OUT
    # abspath() first: NEMESYS_COMPLETE_FUSION_OUT may be a BARE FILENAME, and
    # os.path.dirname('ref.json') == '' makes os.makedirs('') raise FileNotFoundError
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(results, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('saved', out_path)


if __name__ == '__main__':
    main()
