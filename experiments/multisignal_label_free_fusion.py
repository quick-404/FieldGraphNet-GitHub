# -*- coding: utf-8 -*-
"""Multi-signal label-free fusion (3 arms) at controlled 20% FPR.

Arms (all fitted on BENIGN FLOWS ONLY, zero attack labels):
  1. structural : 24-dim BCDG fingerprint, min L2 distance to benign centroids
  2. flow_lof   : LocalOutlierFactor on 82-dim flow features
  3. flow_if    : IsolationForest on 82-dim flow features

Calibration: each arm's raw score -> benign percentile via the calibration split's
empirical CDF (benign only, still zero attack labels).

Fusion rules:
  R1 (headline, ZERO tuning) : max of the three calibrated arm scores
  R2 (robustness only)       : weighted sum, weights selected on a validation half
                               of the attack flows, then FROZEN, evaluated on the
                               held-out test half.
  F0/F1/F2/F3 (pre-registered rule comparison, spec 2026-08-19): F0=max (same as R1,
                               kept as the reference), F1=mean, F2=median,
                               F3=Fisher combination of the benign percentiles.
                               All four are pure functions of the three calibrated
                               arm scores — no attack labels, no test-driven rule
                               selection. See multisignal_fusion_rules.json.

Benign split (500 flows total): 280 fit / 70 calibration / 150 test.
The 150-flow benign test set matches the existing protocol (unknown_protocol_detection.py)
so numbers are directly comparable with tab:strongest / the benign-only control.

Anti-overfitting discipline (see spec 4.5): no test-driven tuning, no per-protocol
rule selection, all 5 seeds reported, honest reporting of losses.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from nemesys_gnn4id.nemesys.cluster import cluster_fingerprints
from flow_key_alignment import load_aligned
from unknown_protocol_detection import TARGET_FPR

IN_DOMAIN_ATTACK = 'DDoS-HTTP_Flood-.pcap'
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
SEEDS = [42, 123, 2024, 7, 99]

# benign 3-way split (500 flows total)
N_FIT, N_CALIB, N_TEST = 280, 70, 150

# pre-registered Fisher clipping bound (spec 2026-08-19 §2, fixed before any new
# results): p_i <- clip(p_i, 1/(2*N_CALIB), 1 - 1/(2*N_CALIB)) = [1/140, 139/140].
# Needed because the calibrated scores are quantised to k/N_CALIB, so p_i can be 0
# (ln 0 = -inf). Derived from N_CALIB, never hard-coded.
FISHER_P_MIN = 1.0 / (2.0 * N_CALIB)

# arm hyperparameters — standard defaults, NOT tuned against results (spec 4.3)
LOF_N_NEIGHBORS = 20
IF_N_ESTIMATORS = 200
STRUCT_THRESHOLD = 0.50

OUT = 'eval_results/fusion_comparison/multisignal_fusion_controlled_fpr.json'


def load_data():
    """Load 82-dim flow features and 24-dim structural fingerprints per source.

    Both containers are KEY-ALIGNED: struct[src][i] is the fingerprint of the very
    flow described by feat[src][i]. The raw containers are NOT index-compatible
    (features are capped at 50/chunk, fingerprints are uncapped), so indexing them
    positionally pairs unrelated flows -- fatal for row-level fusion. See
    experiments/flow_key_alignment.py for the measured evidence.
    """
    return load_aligned()


def split_benign(benign_feats, benign_struct, seed):
    """3-way benign split: fit / calibration / test. Returns index arrays."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(benign_feats))
    fit_idx = idx[:N_FIT]
    calib_idx = idx[N_FIT:N_FIT + N_CALIB]
    test_idx = idx[N_FIT + N_CALIB:N_FIT + N_CALIB + N_TEST]
    return fit_idx, calib_idx, test_idx


def fit_arms(benign_feats, benign_struct, fit_idx, lof_neighbors=LOF_N_NEIGHBORS,
             if_estimators=IF_N_ESTIMATORS, struct_threshold=STRUCT_THRESHOLD):
    """Fit the three label-free arms on benign-only data."""
    X_fit = np.array([benign_feats[i] for i in fit_idx])

    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True)
    lof.fit(X_fit)

    iso = IsolationForest(n_estimators=if_estimators, random_state=42)
    iso.fit(X_fit)

    X_struct_fit = np.array([benign_struct[i] for i in fit_idx])
    _labels, centroids = cluster_fingerprints(X_struct_fit, threshold=struct_threshold)
    centroids = np.array(centroids)

    return {'lof': lof, 'iso': iso, 'centroids': centroids}


def arm_scores(arms, X_flow, fp_struct):
    """Raw (unnormalised) anomaly scores from each arm. Higher = more anomalous.

    X_flow   : (n, 82) flow features
    fp_struct: (n, 24) structural fingerprints
    """
    s_lof = -arms['lof'].score_samples(X_flow)          # higher = more anomalous
    s_if = -arms['iso'].score_samples(X_flow)
    centroids = arms['centroids']
    if centroids.size == 0:
        s_struct = np.zeros(len(fp_struct))
    else:
        d = np.linalg.norm(fp_struct[:, None, :] - centroids[None, :, :], axis=2)
        s_struct = d.min(axis=1)
    return {'structural': s_struct, 'flow_lof': s_lof, 'flow_if': s_if}


def fit_calibrators(arms, benign_feats, benign_struct, calib_idx):
    """Build per-arm benign empirical CDFs from the calibration split (benign only)."""
    X_cal = np.array([benign_feats[i] for i in calib_idx])
    fp_cal = np.array([benign_struct[i] for i in calib_idx])
    raw = arm_scores(arms, X_cal, fp_cal)
    # store the sorted benign scores per arm; percentile = searchsorted / n
    return {name: np.sort(s) for name, s in raw.items()}


def apply_calibration(calibrators, raw_scores):
    """Map raw arm scores to benign percentiles in [0,1] via the stored CDFs."""
    out = {}
    for name, s in raw_scores.items():
        ref = calibrators[name]
        out[name] = np.searchsorted(ref, s, side='right') / max(len(ref), 1)
    return out


def calibrate_for_seed(arms, benign_feats, benign_struct, calib_idx, X_flow, fp_struct):
    """Convenience: fit calibrators on the calibration split, then calibrate given scores."""
    calibrators = fit_calibrators(arms, benign_feats, benign_struct, calib_idx)
    raw = arm_scores(arms, X_flow, fp_struct)
    return apply_calibration(calibrators, raw)


def fuse_max(calibrated):
    """R1: headline zero-tuning fusion — elementwise max of calibrated arm scores."""
    return np.max(np.vstack([calibrated[k] for k in sorted(calibrated)]), axis=0)


def fuse_mean(calibrated):
    """F1 (pre-registered): arithmetic mean of the three calibrated arm scores.

    Symmetric in the arms; pure function of the calibrated scores (no labels).
    """
    return np.mean(np.vstack([calibrated[k] for k in sorted(calibrated)]), axis=0)


def fuse_median(calibrated):
    """F2 (pre-registered): elementwise median of the three calibrated arm scores.

    With three arms this discards the single most extreme arm, so one noisy arm
    cannot dominate. Symmetric in the arms; pure function of the calibrated scores.
    """
    return np.median(np.vstack([calibrated[k] for k in sorted(calibrated)]), axis=0)


def fuse_fisher(calibrated):
    """F3 (pre-registered): Fisher's combination of the three benign percentiles.

    With s_i the benign percentile (larger = more anomalous), the p-value analogue
    is p_i = 1 - s_i. Because the scores are quantised to k/N_CALIB, p_i can reach 0
    (ln 0 = -inf), so p_i is clipped to the pre-declared bound
    [1/(2*N_CALIB), 1 - 1/(2*N_CALIB)] = [1/140, 139/140] (FISHER_P_MIN, derived from
    N_CALIB — no magic number). Then X = -2 * sum_i ln(p_i); larger = more anomalous.
    Used for ranking only (no chi-square p-value conversion). Symmetric in the arms.
    """
    names = sorted(calibrated)
    p = np.clip(1.0 - np.vstack([calibrated[k] for k in names]),
                FISHER_P_MIN, 1.0 - FISHER_P_MIN)
    return -2.0 * np.log(p).sum(axis=0)


def select_weights(calibrated, y, grid=None):
    """R2: pick weights maximising F1@20%FPR on the VALIDATION set, then freeze.

    calibrated : dict arm -> scores for the validation flows
    y          : binary labels for the validation flows (validation only!)
    grid       : candidate weight vectors (defaults: 0.5/3 simplex over 3 arms)
    """
    if grid is None:
        grid = []
        for w0 in (0.0, 0.25, 0.5, 0.75, 1.0):
            for w1 in (0.0, 0.25, 0.5, 0.75, 1.0):
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                grid.append((max(w0, 0.0), max(w1, 0.0), max(w2, 0.0)))
    names = sorted(calibrated)
    best, best_f1 = None, -1.0
    for w in grid:
        s = sum(wi * calibrated[n] for wi, n in zip(w, names))
        m = metrics_at_fpr(s, y, s[y == 0])
        if m['f1'] > best_f1:
            best_f1, best = m['f1'], w
    return {n: float(wi) for n, wi in zip(names, best)}, best_f1


def fuse_weighted(calibrated, weights):
    """R2: weighted sum of calibrated arm scores using frozen weights."""
    names = sorted(calibrated)
    return sum(weights[n] * calibrated[n] for n in names)


def threshold_at_target_fpr(benign_scores, target_fpr=TARGET_FPR):
    """Benign-only threshold whose REALISED FPR is closest to target_fpr.

    Label-free: depends only on benign scores, never on attack labels.

    Why this is needed. The three arms are scored as benign percentiles built from
    only 70 calibration flows, so their scores are quantised to k/70 and tie heavily,
    whereas the supervised baselines produce continuous scores. Under a fixed
    80%-quantile + strict '>' rule this made each method operate at a DIFFERENT
    realised FPR (measured, seed 42: R1_max 0.1267, flow_lof 0.1533, flow_if 0.1933,
    structural/RF/SVM 0.2000, NaiveBayes 0.0000). Reporting those F1 scores as a
    "controlled 20% FPR" comparison would be misleading: tied benign flows sitting
    exactly on the threshold were never flagged.

    Fixing the operating point at the same realised FPR for every method makes the
    comparison honest. Applied identically to all methods, using benign scores only.
    """
    b = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(b)
    if n == 0:
        return np.inf
    best_thr, best_gap = np.inf, np.inf
    # candidate thresholds = just above each distinct benign score
    for t in np.append(np.unique(b), b[-1] + 1e-12):
        gap = abs(float((b > t).sum()) / n - target_fpr)
        if gap < best_gap - 1e-12:
            best_thr, best_gap = float(t), gap
    return best_thr


def metrics_at_fpr(scores, y, benign_scores):
    """AUC + recall/F1/precision at a benign-only threshold near TARGET_FPR.

    Metric formulas mirror unknown_protocol_detection.py so numbers stay comparable;
    only the operating-point rule differs, for the fairness reason documented on
    `threshold_at_target_fpr`. The realised `fpr` is reported, not assumed.
    """
    auc = roc_auc_score(y, scores)
    thr = threshold_at_target_fpr(benign_scores)
    pred = (scores > thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {'auc': round(float(auc), 4), 'recall': round(float(rec), 4),
            'fpr': round(float(fpr), 4), 'precision': round(float(prec), 4),
            'f1': round(float(f1), 4)}



def evaluate_seed(feat, struct, seed):
    """One seed: fit arms, calibrate, score 5 OOD protocols vs all baselines."""
    benign_feats = feat['BenignTraffic.pcap']
    benign_struct = struct.get('BenignTraffic.pcap', [])   # 修正案 1：等长列表
    fit_idx, calib_idx, test_idx = split_benign(benign_feats, benign_struct, seed)
    arms = fit_arms(benign_feats, benign_struct, fit_idx)

    # supervised baselines (口径 2): trained on benign-fit + in-domain DDoS attack
    X_sup = np.array([benign_feats[i] for i in fit_idx] + feat[IN_DOMAIN_ATTACK])
    y_sup = np.array([0] * len(fit_idx) + [1] * len(feat[IN_DOMAIN_ATTACK]))
    supervised = {
        'SVM': SVC(kernel='rbf', random_state=seed),
        'RF': RandomForestClassifier(n_estimators=200, random_state=seed),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=4),
        'NaiveBayes': GaussianNB(),
        'C4.5': DecisionTreeClassifier(criterion='entropy', random_state=seed),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=seed),
    }
    for clf in supervised.values():
        clf.fit(X_sup, y_sup)

    calib_ben_feats = [benign_feats[i] for i in calib_idx]
    calib_ben_struct = [benign_struct[i] for i in calib_idx]
    test_ben_feats = [benign_feats[i] for i in test_idx]
    test_ben_struct = [benign_struct[i] for i in test_idx]

    # ---- calibration is fitted on the benign CALIBRATION split only ----
    # Hoisted out of the protocol loop: it depends only on `arms` and `calib_idx`,
    # so it is identical for every protocol (same value, 5x less redundant work).
    calibrators = fit_calibrators(arms, benign_feats, benign_struct, calib_idx)

    out = {'seed': seed, 'protocols': {}}
    for ood in OOD_PROTOCOLS:
        att_feat = feat[ood]
        att_struct = struct.get(ood, [])   # 修正案 1：等长列表
        n_att = min(len(att_feat), len(att_struct))
        att_feat, att_struct = att_feat[:n_att], att_struct[:n_att]

        # ---- val/test split of the attack flows (chunk-ordered halves) ----
        cut = n_att // 2
        val_att_feat, val_att_struct = att_feat[:cut], att_struct[:cut]
        test_att_feat, test_att_struct = att_feat[cut:], att_struct[cut:]

        def calibrated_scores(X_flow, fp):
            raw = arm_scores(arms, np.array(X_flow), np.array(fp))
            return apply_calibration(calibrators, raw)

        # validation set: benign-calib flows (as negatives) + attack val half
        X_val = np.array(calib_ben_feats + val_att_feat)
        fp_val = np.array(calib_ben_struct + val_att_struct)
        cal_val = calibrated_scores(X_val, fp_val)
        y_val = np.array([0] * len(calib_ben_feats) + [1] * len(val_att_feat))
        weights, _val_f1 = select_weights(cal_val, y_val)

        # test set: benign-test flows + attack test half   (R2 uses half-size test)
        X_half = np.array(test_ben_feats + test_att_feat)
        fp_half = np.array(test_ben_struct + test_att_struct)
        cal_half = calibrated_scores(X_half, fp_half)
        y_half = np.array([0] * len(test_ben_feats) + [1] * len(test_att_feat))

        rec_half = {
            'R1_max': metrics_at_fpr(fuse_max(cal_half), y_half, fuse_max(cal_half)[y_half == 0]),
            'F1_mean': metrics_at_fpr(fuse_mean(cal_half), y_half,
                                      fuse_mean(cal_half)[y_half == 0]),
            'F2_median': metrics_at_fpr(fuse_median(cal_half), y_half,
                                        fuse_median(cal_half)[y_half == 0]),
            'F3_fisher': metrics_at_fpr(fuse_fisher(cal_half), y_half,
                                        fuse_fisher(cal_half)[y_half == 0]),
            'R2_weighted': metrics_at_fpr(fuse_weighted(cal_half, weights), y_half,
                                          fuse_weighted(cal_half, weights)[y_half == 0]),
            'weights': weights,
        }
        for name in ('structural', 'flow_lof', 'flow_if'):
            s = cal_half[name]
            rec_half[name] = metrics_at_fpr(s, y_half, s[y_half == 0])

        # full test set (R1 headline + arms + baselines; matches existing protocol)
        X_full = np.array(test_ben_feats + att_feat)
        y_full = np.array([0] * len(test_ben_feats) + [1] * n_att)
        cal_full = calibrated_scores(X_full, test_ben_struct + att_struct)
        rec_full = {
            'R1_max': metrics_at_fpr(fuse_max(cal_full), y_full, fuse_max(cal_full)[y_full == 0]),
            'F1_mean': metrics_at_fpr(fuse_mean(cal_full), y_full,
                                      fuse_mean(cal_full)[y_full == 0]),
            'F2_median': metrics_at_fpr(fuse_median(cal_full), y_full,
                                        fuse_median(cal_full)[y_full == 0]),
            'F3_fisher': metrics_at_fpr(fuse_fisher(cal_full), y_full,
                                        fuse_fisher(cal_full)[y_full == 0]),
        }
        for name in ('structural', 'flow_lof', 'flow_if'):
            s = cal_full[name]
            rec_full[name] = metrics_at_fpr(s, y_full, s[y_full == 0])
        for name, clf in supervised.items():
            s = (clf.decision_function(X_full) if hasattr(clf, 'decision_function')
                 else clf.predict_proba(X_full)[:, 1])
            rec_full[name] = metrics_at_fpr(s, y_full, s[y_full == 0])

        out['protocols'][ood] = {'full_test': rec_full, 'half_test_R2': rec_half,
                                 'n_benign_test': len(test_ben_feats), 'n_attack': n_att}
    return out


def run_sensitivity(feat, struct):
    """Vary arm hyperparameters; show the fused result is not fragile (spec §4.3).

    Fixed seed 42, standard grid n_neighbors x struct_threshold. Reports the realised
    benign FPR next to F1: the arms are scored as benign percentiles quantised on 70
    calibration flows, so a tie-aware threshold can quietly move the operating point
    away from the nominal 20%. F1 alone cannot reveal that drift.
    """
    rows = []
    for nn in (10, 20, 40):
        for thr in (0.4, 0.5, 0.6):
            benign_feats = feat['BenignTraffic.pcap']
            benign_struct = struct.get('BenignTraffic.pcap', [])   # 修正案 1：等长列表
            fit_idx, calib_idx, test_idx = split_benign(benign_feats, benign_struct, 42)
            arms = fit_arms(benign_feats, benign_struct, fit_idx, lof_neighbors=nn,
                            struct_threshold=thr)
            calibrators = fit_calibrators(arms, benign_feats, benign_struct, calib_idx)
            row = {'n_neighbors': nn, 'struct_threshold': thr, 'per_protocol': {}}
            for ood in OOD_PROTOCOLS:
                att_feat = feat[ood]
                att_struct = struct.get(ood, [])   # 修正案 1：等长列表
                n_att = min(len(att_feat), len(att_struct))
                X = np.array([benign_feats[i] for i in test_idx] + att_feat[:n_att])
                fp = np.array([benign_struct[i] for i in test_idx] + att_struct[:n_att])
                y = np.array([0] * len(test_idx) + [1] * n_att)
                raw = arm_scores(arms, X, fp)
                cal = apply_calibration(calibrators, raw)
                s = fuse_max(cal)
                m = metrics_at_fpr(s, y, s[y == 0])
                row['per_protocol'][ood] = {'f1': m['f1'], 'fpr': m['fpr']}
            row['f1_mean'] = round(float(np.mean(
                [v['f1'] for v in row['per_protocol'].values()])), 4)
            row['fpr_mean'] = round(float(np.mean(
                [v['fpr'] for v in row['per_protocol'].values()])), 4)
            rows.append(row)
    return rows


def selftest():
    """Verify split sizes and that all three arms produce finite scores."""
    feat, struct = load_data()
    benign_feats = feat['BenignTraffic.pcap']
    benign_struct = struct.get('BenignTraffic.pcap', [])
    assert len(benign_feats) == 500, f'expected 500 benign flows, got {len(benign_feats)}'
    # the two containers must stay index-aligned by construction
    assert len(benign_struct) == len(benign_feats), (
        f'flow features and structural fingerprints are not index-aligned: '
        f'{len(benign_feats)} vs {len(benign_struct)}')
    fit_idx, calib_idx, test_idx = split_benign(benign_feats, benign_struct, 42)
    assert (len(fit_idx), len(calib_idx), len(test_idx)) == (N_FIT, N_CALIB, N_TEST)
    arms = fit_arms(benign_feats, benign_struct, fit_idx)
    X_test = np.array([benign_feats[i] for i in test_idx])
    fp_test = np.array([benign_struct[i] for i in test_idx])
    scores = arm_scores(arms, X_test, fp_test)
    for name, s in scores.items():
        assert len(s) == N_TEST, f'{name} length mismatch'
        assert np.all(np.isfinite(s)), f'{name} has non-finite scores'
        print(f'  {name}: min={s.min():.4f} max={s.max():.4f}')
    calibrated = calibrate_for_seed(arms, benign_feats, benign_struct,
                                    calib_idx, X_test, fp_test)
    for name, s in calibrated.items():
        assert s.min() >= 0.0 and s.max() <= 1.0, f'{name} percentile out of [0,1]'
        # [0,1] above is tautological by construction; this is the real check that the
        # CDFs were fitted on BENIGN calibration flows (benign-vs-benign percentiles
        # must be ~uniform, so the mean sits near 0.5). A CDF fitted on the wrong
        # population would push this mean toward 0 or 1.
        mean_pct = float(np.mean(s))
        assert 0.25 < mean_pct < 0.75, (
            f'{name}: mean benign percentile {mean_pct:.3f} is not ~uniform — '
            f'calibrators were probably fitted on the wrong split')
        print(f'  calibrated {name}: max={s.max():.3f} mean={mean_pct:.3f}')
    m = fuse_max(calibrated)
    assert len(m) == N_TEST, 'fuse_max length mismatch'
    y_dummy = np.array([0] * (N_TEST // 2) + [1] * (N_TEST - N_TEST // 2))
    w, _ = select_weights({k: v[:N_TEST] for k, v in calibrated.items()}, y_dummy)
    assert abs(sum(w.values()) - 1.0) < 1e-6, f'weights must sum to 1, got {w}'
    print(f'  R1 max ok; R2 weights={w}')
    print('SELFTEST OK')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    ap.add_argument('--out', default=OUT,
                    help=f'output JSON path (default: {OUT}); the Task 6 deliverable '
                         f'multisignal_fusion_controlled_fpr.json must not be overwritten')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    feat, struct = load_data()
    seeds = [int(s) for s in args.seeds.split(',')]
    all_runs = [evaluate_seed(feat, struct, s) for s in seeds]

    # aggregate mean over seeds for the headline table
    agg = {}
    for ood in OOD_PROTOCOLS:
        methods = list(all_runs[0]['protocols'][ood]['full_test'].keys())
        agg[ood] = {m: {'f1_mean': round(float(np.mean([r['protocols'][ood]['full_test'][m]['f1']
                                                       for r in all_runs])), 4),
                        'f1_std': round(float(np.std([r['protocols'][ood]['full_test'][m]['f1']
                                                      for r in all_runs])), 4),
                        'auc_mean': round(float(np.mean([r['protocols'][ood]['full_test'][m]['auc']
                                                         for r in all_runs])), 4)}
                    for m in methods}

    print('running sensitivity check (9 configs)...')
    sensitivity = run_sensitivity(feat, struct)
    for r in sensitivity:
        print(f"  n_neighbors={r['n_neighbors']} thr={r['struct_threshold']} "
              f"mean_f1={r['f1_mean']}")

    out = {'seeds': seeds, 'per_seed': all_runs, 'aggregate_full_test': agg,
           'sensitivity': sensitivity,
           'protocols': OOD_PROTOCOLS,
           'notes': ('R1_max = zero-tuning headline (full test set, all attack flows); '
                     'F1_mean/F2_median/F3_fisher = pre-registered rules '
                     '(docs/superpowers/specs/2026-08-19-fusion-rule-preregistration.md), '
                     'evaluated on the identical full and half test sets as R1_max; '
                     'all four are pure functions of the three calibrated arm scores '
                     '(no attack labels, no test-driven rule selection); '
                     'R2_weighted = validation-selected weights, evaluated on a half-size '
                     'held-out attack half; arms and all baselines share the identical '
                     'benign 3-way split and 20%-FPR protocol.')}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n{"protocol":<24}{"R1_max":>9}{"struct":>9}{"LOF":>9}{"IF":>9}'
          f'{"RF":>9}{"KNN":>9}{"SVM":>9}')
    for ood in OOD_PROTOCOLS:
        a = agg[ood]
        print(f'{ood:<24}{a["R1_max"]["f1_mean"]:>9.3f}{a["structural"]["f1_mean"]:>9.3f}'
              f'{a["flow_lof"]["f1_mean"]:>9.3f}{a["flow_if"]["f1_mean"]:>9.3f}'
              f'{a["RF"]["f1_mean"]:>9.3f}{a["KNN"]["f1_mean"]:>9.3f}{a["SVM"]["f1_mean"]:>9.3f}')
    print('saved', args.out)


if __name__ == '__main__':
    main()
