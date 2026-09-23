# -*- coding: utf-8 -*-
"""Paired significance tests for PU methods across 5 seeds (AUC and F1@20%FPR).

Honest reporting: on the highly separable structural space, differences may be
non-significant — that is the result we report, not something to engineer away.
"""
import json, os
import numpy as np
from scipy import stats

OUT = 'eval_results/fusion_comparison/significance.json'


def load_per_seed(track='24dim'):
    data = json.load(open('eval_results/fusion_comparison/pu_two_step.json', encoding='utf-8'))
    per = data[track]['per_seed']
    pairs = {}
    for seed in per:
        for m in ('fusion', 'naive', 'elkan_noto'):
            for clf in ('SVM', 'RF'):
                k = f'{m}/{clf}'
                pairs.setdefault(k, {'auc': [], 'f1': []})
                pairs[k]['auc'].append(seed[k]['auc'])
                pairs[k]['f1'].append(seed[k]['f1'])
    return pairs


def compare(a_key, b_key, metric, pairs):
    a = np.array(pairs[a_key][metric], dtype=float)
    b = np.array(pairs[b_key][metric], dtype=float)
    t_stat, p_t = stats.ttest_rel(a, b)
    if np.allclose(a, b):
        w_stat, p_w = None, 1.0  # identical rows (EN==naive for RF): Wilcoxon undefined -> report p=1
    else:
        w_stat, p_w = stats.wilcoxon(a, b)
    return {
        'a': a_key, 'b': b_key, 'metric': metric,
        'mean_a': round(float(a.mean()), 4), 'mean_b': round(float(b.mean()), 4),
        'delta': round(float((a - b).mean()), 4),
        'ttest_stat': round(float(t_stat), 4) if t_stat is not None else None,
        'ttest_p': round(float(p_t), 4),
        'wilcoxon_stat': round(float(w_stat), 4) if w_stat is not None else None,
        'wilcoxon_p': round(float(p_w), 4),
        'n': int(len(a)),
    }


def main():
    out = {}
    for track in ('24dim', '82dim'):
        pairs = load_per_seed(track)
        comps = []
        for metric in ('auc', 'f1'):
            comps.append(compare('fusion/RF', 'naive/RF', metric, pairs))
            comps.append(compare('fusion/RF', 'elkan_noto/RF', metric, pairs))
            comps.append(compare('fusion/SVM', 'naive/SVM', metric, pairs))
            comps.append(compare('naive/RF', 'elkan_noto/RF', metric, pairs))
        out[track] = {'pairs': comps,
                      'note': 'paired across 5 seeds (42,123,2024,7,99); '
                              'Wilcoxon p=1.0 when rows are identical (EN==naive for RF, rank-equivalent)'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    for track in out:
        print(f'== {track} ==')
        for p in out[track]['pairs']:
            print(f"{p['a']} vs {p['b']} [{p['metric']}]: delta={p['delta']} "
                  f"t_p={p['ttest_p']} w_p={p['wilcoxon_p']}")


if __name__ == '__main__':
    main()
