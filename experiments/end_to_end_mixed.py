"""End-to-end mixed-traffic evaluation on the FULL CIC IoT 2023 dataset.

Unifies the headline with the fair methodology: all methods scored per-flow on
held-out benign + ALL attack flows (500-benign pool, 150 held out for testing;
2733 attack flows). Reports AUC (threshold-free) and Recall/FPR/F1 at a
calibrated 20% in-domain FPR. GNN4ID row uses its real aggregate from
ablation_results.json (native operating point, FPR 0.220).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.nemesys.cluster import extract_structural_fingerprint, cluster_fingerprints
from nemesys_gnn4id.pipeline import canonical_flow_tuple

from flow_key_alignment import load_aligned

CHUNK_DIR = 'data/data/23pcap_chunks'
IN_DOMAIN_ATTACK = 'DDoS-HTTP_Flood-.pcap'
ALL_SOURCES = ['BenignTraffic.pcap', 'DDoS-HTTP_Flood-.pcap', 'DNS_Spoofing.pcap',
               'Mirai-udpplain.pcap', 'SqlInjection.pcap',
               'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
BENIGN_TRAIN = 350
TARGET_FPR = 0.20
SEED = 42
STRUCT_CACHE = '/tmp/end_to_end_struct.json'


def flow_structural_fingerprints():
    """Return {source: {flow_key: mean 24-dim fingerprint}} via BCDG (cached).

    SUPERSEDED: main() now uses flow_key_alignment.load_aligned(), which returns the
    same fingerprints as a list index-aligned with the 82-dim feature rows. This raw
    key->fingerprint mapping has no shared index space with `feat`, so indexing it
    positionally would pair unrelated flows. Kept only as the extraction reference.
    """
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
    with open(STRUCT_CACHE, 'w', encoding='utf-8') as f:
        json.dump({src: {'|'.join(str(x) for x in k): v.tolist()
                         for k, v in fv.items()} for src, fv in out.items()}, f)
    return out


def metrics_at_fpr(score, y, benign_scores):
    auc = roc_auc_score(y, score)
    thr = np.quantile(benign_scores, 1 - TARGET_FPR)
    pred = (score > thr).astype(int)
    tp = ((y == 1) & (pred == 1)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum()
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + (y == 0).sum() - fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {'auc': round(auc, 4), 'recall': round(rec, 4), 'fpr': round(fpr, 4),
            'precision': round(prec, 4), 'f1': round(f1, 4)}


def main():
    # key-aligned pair of containers: struct[src][i] is the fingerprint of the very
    # flow described by feat[src][i] (see experiments/flow_key_alignment.py)
    feat, struct = load_aligned()

    # benign train/test split
    ben_feats = feat['BenignTraffic.pcap']
    ben_struct = struct.get('BenignTraffic.pcap', [])
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(ben_feats))
    tr_idx, te_idx = idx[:BENIGN_TRAIN], idx[BENIGN_TRAIN:]

    # train SVM/RF on train-benign + DDoS
    X_in = np.array([ben_feats[i] for i in tr_idx] + feat[IN_DOMAIN_ATTACK])
    y_in = np.array([0] * BENIGN_TRAIN + [1] * len(feat[IN_DOMAIN_ATTACK]))
    svm = SVC(kernel='rbf', random_state=SEED).fit(X_in, y_in)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_in, y_in)

    # fusion structural model from in-domain fingerprints
    X_struct_in = np.array([ben_struct[i] for i in tr_idx]
                           + struct.get(IN_DOMAIN_ATTACK, []))
    _labels, centroids = cluster_fingerprints(X_struct_in, threshold=0.50)
    centroids = np.array(centroids)

    def struct_score(fps):
        if centroids.size == 0:
            return np.zeros(len(fps))
        d = np.linalg.norm(np.array(fps)[:, None, :] - centroids[None, :, :], axis=2)
        return d.min(axis=1)

    # test set: held-out benign + ALL attack flows
    te_ben_feats = [ben_feats[i] for i in te_idx]
    te_ben_struct = [ben_struct[i] for i in te_idx]
    att_feats = [f for src in ALL_SOURCES if src not in ('BenignTraffic.pcap',)
                 for f in feat[src]]
    att_struct = [fp for src in ALL_SOURCES if src not in ('BenignTraffic.pcap',)
                  for fp in struct.get(src, [])]
    n_att = min(len(att_feats), len(att_struct))
    att_feats, att_struct = att_feats[:n_att], att_struct[:n_att]

    X_test = np.array(te_ben_feats + att_feats)
    y_test = np.array([0] * len(te_ben_feats) + [1] * n_att)

    # benign scores for calibration (train-side) per method
    svm_b = svm.decision_function(np.array([ben_feats[i] for i in tr_idx]))
    rf_b = rf.predict_proba(np.array([ben_feats[i] for i in tr_idx]))[:, 1]
    fus_b = struct_score([ben_struct[i] for i in tr_idx])

    results = {}
    results['SVM'] = metrics_at_fpr(svm.decision_function(X_test), y_test, svm_b)
    results['RF'] = metrics_at_fpr(rf.predict_proba(X_test)[:, 1], y_test, rf_b)
    results['Fusion-struct'] = metrics_at_fpr(
        np.concatenate([struct_score(te_ben_struct), struct_score(att_struct)]),
        y_test, fus_b)

    # GNN4ID: real aggregate from ablation (native operating point, FPR 0.220)
    ab = json.load(open('eval_results/fusion_comparison/ablation_results.json', encoding='utf-8'))
    g = ab['ablation']['gnn_raw']
    results['GNN4ID'] = {'recall': round(g['recall'], 4), 'fpr': round(g['fpr'], 4),
                         'f1': round(g['f1'], 4),
                         'note': 'native operating point (FPR 0.220), from full eval'}
    results['_test'] = {'n_benign': len(te_ben_feats), 'n_attack': n_att}

    print(f'Test set: {len(te_ben_feats)} held-out benign + {n_att} attack flows')
    print(f'{"Method":<15s} {"AUC":>6s} {"Recall":>7s} {"FPR":>6s} {"F1":>6s}')
    for m, r in results.items():
        if m.startswith('_'): continue
        print(f'{m:<15s} {r.get("auc", "—"):>6} {r["recall"]:>7.3f} '
              f'{r["fpr"]:>6.3f} {r["f1"]:>6.3f}')

    os.makedirs('eval_results/fusion_comparison', exist_ok=True)
    with open('eval_results/fusion_comparison/end_to_end_mixed.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Saved: eval_results/fusion_comparison/end_to_end_mixed.json')


if __name__ == '__main__':
    main()
