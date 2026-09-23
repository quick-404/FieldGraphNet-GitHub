"""
Bootstrap confidence intervals for ablation study.

Reads per-chunk metrics from ablation_results.json, resamples with
replacement 10000 times, and computes 95% CI for each variant.
"""
import sys, os, json, random
# Fixed seed for reproducible bootstrap CIs (paper/delta recalc must match).
random.seed(42)
sys.path.insert(0, 'src')

EVAL_DIR = 'eval_results/fusion_comparison'


def bootstrap_ci(chunk_metrics, variant, n_iter=10000, alpha=0.05):
    """Compute bootstrap 95% CI for a variant's recall, fpr, f1."""
    n_chunks = len(chunk_metrics)

    # Pre-extract per-chunk tp/fp/fn/tn for speed
    tps = [c[variant]['tp'] for c in chunk_metrics]
    fps = [c[variant]['fp'] for c in chunk_metrics]
    fns = [c[variant]['fn'] for c in chunk_metrics]
    tns = [c[variant]['tn'] for c in chunk_metrics]

    recalls, fprs, f1s = [], [], []
    for _ in range(n_iter):
        # Bootstrap sample: resample chunks with replacement
        idxs = [random.randrange(0, n_chunks) for _ in range(n_chunks)]
        tp = sum(tps[i] for i in idxs)
        fp = sum(fps[i] for i in idxs)
        fn = sum(fns[i] for i in idxs)
        tn = sum(tns[i] for i in idxs)

        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        recalls.append(recall)
        fprs.append(fpr)
        f1s.append(f1)

    recalls.sort()
    fprs.sort()
    f1s.sort()

    lo_idx = int(n_iter * alpha / 2)
    hi_idx = int(n_iter * (1 - alpha / 2))

    return {
        'recall_ci': (round(recalls[lo_idx], 4), round(recalls[hi_idx], 4)),
        'fpr_ci': (round(fprs[lo_idx], 4), round(fprs[hi_idx], 4)),
        'f1_ci': (round(f1s[lo_idx], 4), round(f1s[hi_idx], 4)),
        'recall_mean': round(sum(recalls) / n_iter, 4),
        'fpr_mean': round(sum(fprs) / n_iter, 4),
        'f1_mean': round(sum(f1s) / n_iter, 4),
    }


def main():
    path = os.environ.get('NEMESYS_ABLATION_OUT') or os.path.join(
        EVAL_DIR, 'ablation_results.json')
    with open(path) as f:
        data = json.load(f)

    chunk_metrics = data.get('chunk_metrics', [])
    if not chunk_metrics:
        print('ERROR: No chunk_metrics found. Re-run ablation with updated script.')
        return

    print(f'Chunks: {len(chunk_metrics)}, total flows: {data["config"]["total_flows"]}')
    print(f'Bootstrap iterations: 10000')
    print()

    variants = {
        'A1: GNN raw': 'a1',
        'A2: Domain route': 'a2',
        'A3: Clustering only': 'a3',
        'A4: Full fusion': 'a4',
    }

    print(f'{"Method":25s} {"Recall":>8s} {"(95% CI)":>14s} {"FPR":>8s} {"(95% CI)":>14s} {"F1":>8s} {"(95% CI)":>14s}')
    print(f'{"-" * 83}')
    for name, vkey in variants.items():
        ci = bootstrap_ci(chunk_metrics, vkey, n_iter=10000)
        agg = data['ablation'][{'a1': 'gnn_raw', 'a2': 'domain_routing_only',
                                'a3': 'clustering_only', 'a4': 'fusion_alert'}[vkey]]
        print(f'{name:25s} {agg["recall"]:>8.4f}  [{ci["recall_ci"][0]:.4f}, {ci["recall_ci"][1]:.4f}]  '
              f'{agg["fpr"]:>8.4f}  [{ci["fpr_ci"][0]:.4f}, {ci["fpr_ci"][1]:.4f}]  '
              f'{agg["f1"]:>8.4f}  [{ci["f1_ci"][0]:.4f}, {ci["f1_ci"][1]:.4f}]')

    # Confidence interval widths
    print(f'\n{"Method":25s} {"Recall ±":>10s} {"FPR ±":>10s} {"F1 ±":>10s}')
    print(f'{"-" * 55}')
    for name, vkey in variants.items():
        ci = bootstrap_ci(chunk_metrics, vkey, n_iter=10000)
        r_hw = round((ci['recall_ci'][1] - ci['recall_ci'][0]) / 2, 4)
        f_hw = round((ci['fpr_ci'][1] - ci['fpr_ci'][0]) / 2, 4)
        f1_hw = round((ci['f1_ci'][1] - ci['f1_ci'][0]) / 2, 4)
        print(f'{name:25s} ±{r_hw:.4f}   ±{f_hw:.4f}   ±{f1_hw:.4f}')

    # Save
    ci_results = {}
    for name, vkey in variants.items():
        ci_results[vkey] = bootstrap_ci(chunk_metrics, vkey, n_iter=10000)
    out_path = os.environ.get('NEMESYS_BOOTSTRAP_OUT') or os.path.join(
        EVAL_DIR, 'ablation_bootstrap_ci.json')
    with open(out_path, 'w') as f:
        json.dump(ci_results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
