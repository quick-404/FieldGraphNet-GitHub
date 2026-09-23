# -*- coding: utf-8 -*-
# experiments/pu_two_step_detection.py
"""PU two-step structural detection + incremental vocabulary self-extension.

Completes the incremental unknown-protocol discovery framework with PU
Learning's Step 2 (final classifier). After Phase 1A identifies reliable
negatives (RN) from a deployment-realistic unlabeled pool U (in-domain
messages dominate, unknown attacks are the minority arrival), a binary
classifier is trained on the labeled positives (known training-domain
protocols) and the RN, producing P(unknown|x) for new messages. Compared
against two zero-label PU baselines under identical data:
  - elkan_noto: P(known|x) = P(s=1|x)/c (Elkan-Noto), inverted to unknown.
  - naive: treats ALL unlabeled traffic as negatives (classic PU strawman;
    with a known-majority U it is heavily mislabeled and should degrade).
Protocol: disjoint P / U / val-K / test partitions; the 20%-FPR threshold is
calibrated on a held-out val-K set (disjoint from training); test = held-out
known + OOD with labels revealed.
"""
import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from nemesys_gnn4id.nemesys.cluster import cluster_fingerprints

from incremental_unknown_protocol import (extract_fingerprints, build_known_model,
                                          split_known_unknown, pseudo_label,
                                          KNOWN_PROTOCOLS, KNOWN_THRESHOLD)

SEED = 42
TARGET_FPR = 0.20
OUT = 'eval_results/fusion_comparison/pu_two_step.json'

# deployment-realistic protocol sizes (fractions of available pools, auto-capped)
N_P_FRAC = 0.35        # labeled positives (known train)
N_U_KNOWN_FRAC = 0.28  # known messages inside the unlabeled pool U
N_VAL_FRAC = 0.18      # held-out known for threshold calibration
N_U_OOD = 100          # per-OOD unknown messages inside U
N_TEST_OOD = 300       # per-OOD unknown messages in the test batch

# multi-class appendix: discovery-oriented U (OOD-rich so clusters form)
MC_U_OOD = 500        # per-OOD OOD messages inside the appendix unlabeled pool
MC_MIN_CLUSTER = 40   # pseudo-label min cluster size for the appendix


def metrics_at_fpr(scores, y, benign_scores):
    """AUC + recall/F1 at a threshold tuned to TARGET_FPR on held-out benign scores."""
    auc = roc_auc_score(y, scores)
    thr = np.quantile(benign_scores, 1 - TARGET_FPR)
    pred = (scores > thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    return {'auc': round(auc, 4), 'recall': round(rec, 4), 'precision': round(prec, 4),
            'f1': round(f1, 4), 'fpr': round(fpr, 4)}


def make_training_sets(X_P, U_fps, rn):
    """Return {method: (X, y, kind)} for the three PU schemes.

    X_P: labeled positives P (known train).
    U_fps: deployment-realistic unlabeled pool (known-majority mix).
    rn: reliable negatives from Phase 1A (all unknown-split messages).
    kind: 'cls' trains P(unknown|x) directly; 'en' trains the s-classifier.
    """
    X_RN = np.array(rn)
    return {
        'fusion': (np.concatenate([X_P, X_RN]),
                   np.array([0] * len(X_P) + [1] * len(X_RN)), 'cls'),
        'naive': (np.concatenate([X_P, np.array(U_fps)]),
                  np.array([0] * len(X_P) + [1] * len(U_fps)), 'cls'),
        'elkan_noto': (np.concatenate([X_P, np.array(U_fps)]),
                       np.array([1] * len(X_P) + [0] * len(U_fps)), 'en'),
    }


def fit_and_score(kind, X, y, X_P, X_val, X_test, clf):
    """Train clf; return (scores_on_val_known, test_scores) both as P(unknown|x)."""
    clf.fit(X, y)
    if kind == 'en':
        c = float(np.clip(np.mean(clf.predict_proba(X_P)[:, 1]), 1e-6, 1.0))

        def score(Xn, clf=clf, c=c):
            return 1 - np.clip(clf.predict_proba(Xn)[:, 1] / c, 0.0, 1.0)
    else:
        def score(Xn, clf=clf):
            return clf.predict_proba(Xn)[:, 1]
    return score(X_val), score(X_test)


def clf_factory(clf_name, seed=SEED):
    if clf_name == 'SVM':
        return SVC(kernel='rbf', probability=True, random_state=seed)
    return RandomForestClassifier(n_estimators=200, random_state=seed)


def eval_seed(track_feat, known_centroids, seed):
    """Full train/calibrate/test evaluation for one seed on one feature track."""
    rng = np.random.RandomState(seed)
    known_pool = [fp for p in KNOWN_PROTOCOLS if p in track_feat for fp in track_feat[p]]
    ood_pools = {p: track_feat[p] for p in track_feat if p not in KNOWN_PROTOCOLS}

    # disjoint partition of the known pool: P / U_known / val_K / test_K
    nk = len(known_pool)
    kidx = rng.permutation(nk)
    n_P = min(int(nk * N_P_FRAC), nk)
    n_U_known = min(int(nk * N_U_KNOWN_FRAC), nk - n_P)
    n_val = min(int(nk * N_VAL_FRAC), nk - n_P - n_U_known)
    P = np.array([known_pool[i] for i in kidx[:n_P]])
    U_known = [known_pool[i] for i in kidx[n_P:n_P + n_U_known]]
    val_K = [known_pool[i] for i in kidx[n_P + n_U_known:n_P + n_U_known + n_val]]
    test_K = [known_pool[i] for i in kidx[n_P + n_U_known + n_val:]]

    # OOD pools: train (feeds U) / test (feeds batch2)
    OOD_train, OOD_test = {}, {}
    for p, pool in ood_pools.items():
        oidx = rng.permutation(len(pool))
        cut = int(len(pool) * 0.7)
        OOD_train[p] = [pool[i] for i in oidx[:cut]]
        OOD_test[p] = [pool[i] for i in oidx[cut:]]

    # U: deployment-realistic unlabeled pool (known-majority)
    U_fps = list(U_known)
    for p in sorted(ood_pools):
        U_fps += OOD_train[p][:min(N_U_OOD, len(OOD_train[p]))]
    U_fps = [U_fps[i] for i in rng.permutation(len(U_fps))]

    # Phase 1A: RN = ALL unknown-split messages (no cohesion filter)
    _km, rn, _fl = split_known_unknown(U_fps, known_centroids)

    # batch2 test: held-out known + controlled OOD test sample
    te_fps = list(test_K)
    te_true = ['known'] * len(test_K)
    for p in sorted(ood_pools):
        n = min(N_TEST_OOD, len(OOD_test[p]))
        te_fps += OOD_test[p][:n]
        te_true += [p] * n
    X_test = np.array(te_fps)
    y_test = np.array([0 if t == 'known' else 1 for t in te_true])
    X_val = np.array(val_K)

    sets = make_training_sets(P, U_fps, rn)
    results = {}
    raw = {}
    for method, (X, y, kind) in sets.items():
        for clf_name in ('SVM', 'RF'):
            clf = clf_factory(clf_name, seed)
            ben, te = fit_and_score(kind, X, y, P, X_val, X_test, clf)
            results[f'{method}/{clf_name}'] = metrics_at_fpr(te, y_test, ben)
            raw[f'{method}/{clf_name}'] = {
                'ben_scores': [float(v) for v in ben],
                'test_scores': [float(v) for v in te],
                'test_true': [int(v) for v in y_test],
            }
    results['_meta'] = {'seed': seed, 'n_P': len(P), 'n_U': len(U_fps),
                        'n_known_in_U': len(U_known), 'n_rn': len(rn),
                        'n_val': len(val_K), 'n_test': len(te_fps),
                        'n_known_test': int((y_test == 0).sum()),
                        'n_unknown_test': int((y_test == 1).sum())}
    return results, raw


SEEDS = [42, 123, 2024, 7, 99]


def extract_flow_features():
    """Return {label: list[82-dim drop-ports flow vectors]} for known pcaps + OOD."""
    from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer
    from statistical_fusion_eval import extract_features_by_source
    import incremental_unknown_protocol as iup

    analyzer = GNN4IDAnalyzer(model_path='models/model.pth')
    out = {}
    for stem, proto in zip(iup.KNOWN_PCAPS, KNOWN_PROTOCOLS):
        pcap = os.path.join(iup.DATA_DIR, stem)
        if not os.path.exists(pcap):
            continue
        df = analyzer.extract_features(pcap, max_flows=1000)
        if df.empty or 'flow_features' not in df.columns:
            continue
        feats = []
        for _, row in df.iterrows():
            feat = list(row['flow_features'])
            for z in range(77, min(82, len(feat))):
                feat[z] = 0.0  # drop ports
            feats.append(feat)
        out[proto] = feats
    ood = extract_features_by_source()
    for src in iup.OOD_SOURCES:
        out[src.replace('.pcap', '')] = ood.get(src, [])
    return out


def aggregate(seed_results):
    """Collapse per-seed results into mean/std per method/clf."""
    methods = sorted({k for r in seed_results for k in r if not k.startswith('_')})
    out = {}
    for m in methods:
        aucs = [r[m]['auc'] for r in seed_results]
        f1s = [r[m]['f1'] for r in seed_results]
        recs = [r[m]['recall'] for r in seed_results]
        out[m] = {'auc_mean': round(float(np.mean(aucs)), 4),
                  'auc_std': round(float(np.std(aucs)), 4),
                  'f1_mean': round(float(np.mean(f1s)), 4),
                  'f1_std': round(float(np.std(f1s)), 4),
                  'recall_mean': round(float(np.mean(recs)), 4)}
    return out


def multiclass_eval(track_feat, known_centroids, seed=42):
    """Multi-class appendix: known + pseudo-labeled unknown clusters, per-class F1."""
    rng = np.random.RandomState(seed)
    known_pool = [fp for p in KNOWN_PROTOCOLS if p in track_feat for fp in track_feat[p]]
    ood_pools = {p: track_feat[p] for p in track_feat if p not in KNOWN_PROTOCOLS}
    nk = len(known_pool)
    kidx = rng.permutation(nk)
    n_P = min(int(nk * N_P_FRAC), nk)
    n_U_known = min(int(nk * N_U_KNOWN_FRAC), nk - n_P)
    n_val = min(int(nk * N_VAL_FRAC), nk - n_P - n_U_known)
    P = [known_pool[i] for i in kidx[:n_P]]
    U_known = [known_pool[i] for i in kidx[n_P:n_P + n_U_known]]
    test_K = [known_pool[i] for i in kidx[n_P + n_U_known + n_val:]]
    OOD_train, OOD_test = {}, {}
    for p, pool in ood_pools.items():
        oidx = rng.permutation(len(pool))
        cut = int(len(pool) * 0.7)
        OOD_train[p] = [pool[i] for i in oidx[:cut]]
        OOD_test[p] = [pool[i] for i in oidx[cut:]]

    # U with per-message source labels (needed to map clusters to true protocols)
    U_pairs = [(fp, 'known') for fp in U_known]
    for p in sorted(ood_pools):
        U_pairs += [(fp, p) for fp in OOD_train[p][:min(MC_U_OOD, len(OOD_train[p]))]]
    rng.shuffle(U_pairs)
    U_fps = [fp for fp, _s in U_pairs]
    U_src = [_s for _fp, _s in U_pairs]
    _km, unknown_m, flags = split_known_unknown(U_fps, known_centroids)
    unknown_true = [U_src[i] for i in range(len(U_fps)) if flags[i] == 0]

    labels, _cent = cluster_fingerprints(np.array(unknown_m), threshold=KNOWN_THRESHOLD)
    clusters = defaultdict(list)
    cluster_true = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)
        cluster_true[int(lbl)].append(unknown_true[i])
    plabels = pseudo_label({cid: [unknown_m[i] for i in idxs]
                            for cid, idxs in clusters.items()},
                           min_size=MC_MIN_CLUSTER)
    cl_to_proto = {cid: Counter(cluster_true[cid]).most_common(1)[0][0]
                   for cid in plabels}
    classes = ['known'] + sorted({p for p in cl_to_proto.values()})

    tr_idx = [i for cid in plabels for i in clusters[cid]]
    X_tr = np.array(P + [unknown_m[i] for i in tr_idx])
    y_tr = (['known'] * len(P)
            + [cl_to_proto[cid] for cid in plabels for _ in clusters[cid]])
    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X_tr, y_tr)

    te_fps = list(test_K)
    te_true = ['known'] * len(test_K)
    for p in sorted(ood_pools):
        n = min(N_TEST_OOD, len(OOD_test[p]))
        te_fps += OOD_test[p][:n]
        te_true += [p] * n
    keep = [i for i, t in enumerate(te_true) if t == 'known' or t in classes]
    X_te = np.array([te_fps[i] for i in keep])
    y_te = [te_true[i] for i in keep]
    pred = rf.predict(X_te)

    per_class = {}
    for c in classes:
        yt = np.array([1 if v == c else 0 for v in y_te])
        yp = np.array([1 if v == c else 0 for v in pred])
        tp = int((yt & yp).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_class[c] = {'f1': round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                        'precision': round(prec, 4), 'recall': round(rec, 4),
                        'n_test': int(yt.sum())}
    f1s = [v['f1'] for c, v in per_class.items() if v['n_test'] > 0]
    return {'per_class': per_class,
            'macro_f1': round(float(np.mean(f1s)), 4) if f1s else 0.0,
            'classes': classes, 'n_pseudo_classes': len(plabels)}


def run_track(track_feat, track_name):
    """Run all seeds on one track; add multi-class appendix for the 24-dim track."""
    known_fps = {p: track_feat[p] for p in KNOWN_PROTOCOLS if p in track_feat}
    known_centroids = build_known_model(known_fps)
    seed_results = []
    raw_by_seed = {}
    for s in SEEDS:
        res, raw = eval_seed(track_feat, known_centroids, s)
        seed_results.append(res)
        raw_by_seed[str(s)] = raw
    multiclass = (multiclass_eval(track_feat, known_centroids)
                  if track_name == '24dim' else None)
    return {'seeds': SEEDS, 'per_seed': seed_results,
            'mean_std': aggregate(seed_results), 'multiclass': multiclass}, raw_by_seed


def _print_track(tag, track):
    print(f'=== {tag} (mean over {len(SEEDS)} seeds) ===')
    for m, v in sorted(track['mean_std'].items()):
        print(f'{m:<22s} AUC {v["auc_mean"]:.3f}±{v["auc_std"]:.3f} '
              f'F1 {v["f1_mean"]:.3f}±{v["f1_std"]:.3f} recall {v["recall_mean"]:.3f}')


def main():
    out = {}
    raw_by_track = {}
    fps = extract_fingerprints()
    out['24dim'], raw_by_track['24dim'] = run_track(fps, '24dim')
    _print_track('24-dim structural track', out['24dim'])

    flow = extract_flow_features()
    known_flow = sum(len(v) for p, v in flow.items() if p in KNOWN_PROTOCOLS)
    if known_flow < 30:
        print(f'WARNING: known flow pool small ({known_flow}); 82-dim results are weak.')
    out['82dim'], raw_by_track['82dim'] = run_track(flow, '82dim')
    _print_track('82-dim flow track', out['82dim'])

    mc = out['24dim']['multiclass']
    print('=== multi-class appendix (24-dim, seed 42) ===')
    print(f'macro F1={mc["macro_f1"]}  pseudo-classes={mc["n_pseudo_classes"]}')
    for c, v in sorted(mc['per_class'].items()):
        print(f'  {c:<20s} F1={v["f1"]:.4f} n_test={v["n_test"]}')

    os.makedirs('eval_results/fusion_comparison', exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'Saved: {OUT}')

    scores_out = os.path.splitext(OUT)[0] + '_scores.json'
    with open(scores_out, 'w', encoding='utf-8') as f:
        json.dump(raw_by_track, f, indent=2, ensure_ascii=False)
    print(f'[SAVE] per-sample scores: {scores_out}')


if __name__ == '__main__':
    main()
