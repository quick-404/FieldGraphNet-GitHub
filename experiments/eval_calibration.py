# -*- coding: utf-8 -*-
"""Brier score + reliability histogram for PU methods.

Brier = mean((score - true)^2); reliability bins compare mean predicted
probability vs observed positive fraction per bin (10 bins).
This is the calibration evidence the paper needs to distinguish the PU
schemes beyond rank metrics (where EN==naive for RF).
"""
import json, os
import numpy as np

OUT = 'eval_results/fusion_comparison/calibration.json'


def brier_score(y, scores):
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    return float(np.mean((s - y) ** 2))


def reliability_hist(y, scores, bins=10):
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        mask = (s >= edges[i]) & (s < edges[i + 1])
        if mask.sum() == 0:
            out.append({'bin': i, 'range': [round(edges[i], 3), round(edges[i + 1], 3)], 'n': 0})
            continue
        out.append({'bin': i, 'range': [round(edges[i], 3), round(edges[i + 1], 3)],
                    'n': int(mask.sum()),
                    'mean_pred': round(float(s[mask].mean()), 4),
                    'frac_pos': round(float(y[mask].mean()), 4)})
    return out


def main():
    raw = json.load(open('eval_results/fusion_comparison/pu_two_step_scores.json', encoding='utf-8'))
    out = {'24dim': {}, '82dim': {}}
    for track in ('24dim', '82dim'):
        seeds = raw.get(track, {})
        # aggregate per method across seeds
        methods = sorted({m for seed in seeds.values() for m in seed})
        for m in methods:
            briers = []
            hists = {}
            for seed, methods_map in seeds.items():
                d = methods_map.get(m)
                if d is None:
                    continue
                briers.append(brier_score(d['test_true'], d['test_scores']))
                hists[seed] = reliability_hist(d['test_true'], d['test_scores'])
            if not briers:
                continue
            out[track][m] = {
                'brier_mean': round(float(np.mean(briers)), 4),
                'brier_std': round(float(np.std(briers)), 4),
                'per_seed_brier': {s: round(b, 4) for s, b in zip(seeds, briers)},
                'reliability_seed42': hists.get('42', []),
            }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    for track in out:
        print(f'== {track} ==')
        for m, v in out[track].items():
            print(f"  {m}: Brier {v['brier_mean']}±{v['brier_std']}")


if __name__ == '__main__':
    main()
