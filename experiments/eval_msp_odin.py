# -*- coding: utf-8 -*-
"""MSP / ODIN OOD detection on our own data (24-dim structural track).

Trains a supervised RF/SVM on known-protocol fingerprints only (multi-class
over the KNOWN_PROTOCOLS in-domain classes), then evaluates MSP thresholding
and ODIN (temperature-scaled softmax) as OOD detectors at a controlled 20% FPR,
against the fusion's structural OOD detector as reference.

MSP (Hendrycks & Gimpel 2017) score = 1 - max_c P(c|x): a confident in-domain
classification yields a low OOD score. ODIN (Liang et al. 2018) applies
temperature scaling to the logits before the softmax, then the same 1 - max
rule. RF has no true logits, so predict_log_proba (clipped) is used as a
documented pseudo-logit proxy; SVM uses decision_function margins. Because the
standard ODIN recipe (T=1000, CNN-scale logits) does not transfer to tree /
margin scores, a temperature sweep is also recorded.

Direct first-hand evidence for the paper's claim that confidence-based
statistical OOD methods fail under protocol covariate shift (extending the
cited Corsini & Yang conclusion). NOTE the honest caveat: the 24-dim structural
space is cleanly separable, so even confidence-based methods can look good
here; the real failure of these methods is on the 82-dim flow track under
covariate shift.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from incremental_unknown_protocol import extract_fingerprints, KNOWN_PROTOCOLS
from pu_two_step_detection import metrics_at_fpr, TARGET_FPR

OUT = 'eval_results/fusion_comparison/msp_odin.json'
T = 1000.0            # ODIN headline temperature (Liang et al. default for CNNs)
T_SWEEP = [1.0, 10.0, 100.0, 1000.0]
SEEDS = [42, 123, 2024, 7, 99]   # same seeds as the fusion PU study
N_TR_FRAC, N_VA_FRAC = 0.5, 0.2


def scaled_softmax(logits, temp):
    """Temperature-scaled softmax over classes (rows)."""
    z = np.asarray(logits, dtype=np.float64) / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ood_scores(p_class):
    """OOD score from a P(class|x) matrix: 1 - max softmax probability (MSP rule)."""
    return 1.0 - p_class.max(axis=1)


def logits_of(clf, clf_name, X):
    """Logit proxy per classifier: SVM decision_function, RF log-proba (clipped)."""
    if clf_name == 'SVM':
        return clf.decision_function(X)
    return np.log(np.clip(clf.predict_proba(X), 1e-12, 1.0))


def eval_seed(fps, seed):
    """One seed: 50/20/30 known train/val/test + all OOD in test; return per-method dict."""
    rng = np.random.RandomState(seed)
    known = [fp for p in KNOWN_PROTOCOLS if p in fps for fp in fps[p]]
    ood = [fp for p in fps if p not in KNOWN_PROTOCOLS for fp in fps[p]]
    X_known = np.array(known, dtype=np.float32)
    X_ood = np.array(ood, dtype=np.float32)
    y_known = np.array([i for i, p in enumerate(KNOWN_PROTOCOLS)
                        for _ in range(len(fps[p]))], dtype=np.int64)

    idx = rng.permutation(len(X_known))
    n_tr = int(len(X_known) * N_TR_FRAC)
    n_va = int(len(X_known) * N_VA_FRAC)
    tr, va = X_known[idx[:n_tr]], X_known[idx[n_tr:n_tr + n_va]]
    te_known = X_known[idx[n_tr + n_va:]]
    y_tr = y_known[idx[:n_tr]]

    X_test = np.concatenate([te_known, X_ood])
    y_test = np.concatenate([np.zeros(len(te_known)), np.ones(len(X_ood))])

    res = {}
    for clf_name, clf in [
        ('RF', RandomForestClassifier(n_estimators=200, random_state=seed)),
        ('SVM', SVC(kernel='rbf', probability=True, random_state=seed)),
    ]:
        clf.fit(tr, y_tr)
        # --- MSP: 1 - max softmax over in-domain classes ---
        msp_val = ood_scores(clf.predict_proba(va))
        msp_te = ood_scores(clf.predict_proba(X_test))
        m = metrics_at_fpr(msp_te, y_test, msp_val)
        res[f'MSP/{clf_name}'] = {'auc': m['auc'], 'f1': m['f1'],
                                  'recall': m['recall'], 'fpr': m['fpr']}
        # --- ODIN: temperature-scaled logits, then the same 1 - max rule ---
        logits_val, logits_te = logits_of(clf, clf_name, va), logits_of(clf, clf_name, X_test)
        odin_val = ood_scores(scaled_softmax(logits_val, T))
        odin_te = ood_scores(scaled_softmax(logits_te, T))
        o = metrics_at_fpr(odin_te, y_test, odin_val)
        res[f'ODIN/{clf_name}'] = {'auc': o['auc'], 'f1': o['f1'],
                                   'recall': o['recall'], 'fpr': o['fpr']}
        # temperature sweep (mean AUC over the test set) for the record
        res[f'_sweep/{clf_name}'] = {
            str(temp): round(roc_auc_score(y_test, ood_scores(scaled_softmax(logits_te, temp))), 4)
            for temp in T_SWEEP}
        res[f'_sweep/{clf_name}']['T_default'] = T
    return res


def mean_std(key, seed_results):
    vals = [r[key] for r in seed_results]
    def ms(f):
        return (round(float(np.mean([f(v) for v in vals])), 4),
                round(float(np.std([f(v) for v in vals])), 4))
    return ms(lambda v: v['auc']), ms(lambda v: v['f1']), ms(lambda v: v['recall']), \
        ms(lambda v: v['fpr'])


def main():
    fps = extract_fingerprints()  # {proto: [fp24]}
    n_known = sum(len(v) for p, v in fps.items() if p in KNOWN_PROTOCOLS)
    n_ood = sum(len(v) for p, v in fps.items() if p not in KNOWN_PROTOCOLS)
    print(f'known: {n_known}  ood: {n_ood}  (per-proto: '
          f'{[f"{p}={len(v)}" for p, v in fps.items()]})')

    seed_results = [eval_seed(fps, s) for s in SEEDS]

    results = {}
    for key in ['MSP/RF', 'MSP/SVM', 'ODIN/RF', 'ODIN/SVM']:
        (auc_m, auc_s), (f1_m, f1_s), (rec_m, rec_s), (fpr_m, fpr_s) = mean_std(key, seed_results)
        results[key] = {'auc_mean': auc_m, 'auc_std': auc_s,
                        'f1_mean': f1_m, 'f1_std': f1_s,
                        'recall_mean': rec_m, 'recall_std': rec_s,
                        'fpr_mean': fpr_m, 'fpr_std': fpr_s}
        print(f'{key:<10s} AUC {auc_m:.4f}±{auc_s:.4f}  F1 {f1_m:.4f}±{f1_s:.4f}  '
              f'recall {rec_m:.4f}±{rec_s:.4f}  fpr {fpr_m:.4f}')

    # temperature sweep: mean AUC over seeds (per T)
    sweep = {}
    for clf_name in ['RF', 'SVM']:
        sweep[clf_name] = {t: round(float(np.mean([r[f'_sweep/{clf_name}'][t]
                                                   for r in seed_results])), 4)
                           for t in [str(x) for x in T_SWEEP]}
        sweep[clf_name]['T_default'] = T
    results['_temperature_sweep_auc'] = sweep
    print('temperature sweep (mean AUC):', json.dumps(sweep, indent=2))

    # reference: the fusion's structural OOD detector on the same 24-dim track
    # (mean over the same 5 seeds from the PU study, pu_two_step.json)
    results['reference_fusion_24dim'] = {
        'fusion/RF': {'auc_mean': 0.9816, 'f1_mean': 0.9702, 'recall_mean': 0.9812},
        'fusion/SVM': {'auc_mean': 0.9364, 'f1_mean': 0.9075, 'recall_mean': 0.8891},
    }

    results['_note'] = (
        'MSP/ODIN on the 24-dim structural track; threshold calibrated on held-out '
        'known at 20% FPR (TARGET_FPR from pu_two_step_detection). MSP = 1 - max '
        'softmax over the 5 in-domain known-protocol classes (Hendrycks & Gimpel); '
        'ODIN = temperature-scaled logits before the same rule (Liang et al.), T=1000 '
        'headline (their CNN-scale default) with a full sweep recorded. SVM logits = '
        'decision_function margins; RF has no logits, so predict_log_proba (clipped) '
        'is a documented pseudo-logit proxy. HONEST CAVEAT: the 24-dim structural '
        'space is cleanly separable, so confidence-based methods can look good here '
        '(and ODIN at CNN-scale T collapses tree/margin scores toward uniform, '
        'AUC~0.5). The paper\'s claim that confidence-based statistical OOD methods '
        'fail under protocol covariate shift (Corsini & Yang) concerns the 82-dim '
        'flow track, not this separable structural space.')
    results['_meta'] = {'seeds': SEEDS, 'temperature': T, 'target_fpr': TARGET_FPR,
                        'n_known': n_known, 'n_ood': n_ood,
                        'split': f'known {N_TR_FRAC}/{N_VA_FRAC}/rest train/val/test, all OOD in test',
                        'reference_source': 'eval_results/fusion_comparison/pu_two_step.json (5-seed mean)'}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
