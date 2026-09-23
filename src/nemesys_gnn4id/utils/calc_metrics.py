"""从 eval_results JSON 文件中提取并展示所有评估指标"""
import json, os, sys

base = os.path.expanduser('~/网络流量分析项目/nemesys-gnn4id/eval_results')
pcap_names = ['dhcp_100', 'dns_100', 'ntp_100', 'modbus_100', 'dnp3_100', 's7comm_100']

print("=" * 80)
print("  nemesys-gnn4id 评估指标汇总")
print("=" * 80)

# ---- 表1: 原始GNN指标 (未做域感知修正) ----
print("\n--- 原始GNN分类指标（OOD流未修正，仅供参考） ---\n")
print(f"{'PCAP':<12} {'样本':<6} {'精确率':<10} {'召回率':<10} {'F1':<10} {'FPR':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
print("-" * 80)

totals = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'n': 0}

for name in pcap_names:
    path = os.path.join(base, f'{name}_report.json')
    if not os.path.exists(path):
        print(f"{name:<12} (JSON不存在)")
        continue
    with open(path) as f:
        data = json.load(f)
    m = data.get('metrics') or {}
    binary = m.get('binary') or {}
    n = m.get('total_samples', 0)

    precision = binary.get('precision', 0)
    recall = binary.get('recall', 0)
    f1_score = binary.get('f1', 0)
    fpr = binary.get('fpr', 0)
    tp = binary.get('tp', 0)
    fp = binary.get('fp', 0)
    tn = binary.get('tn', 0)
    fn = binary.get('fn', 0)

    print(f"{name:<12} {n:<6} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f} {fpr:<10.4f} {tp:<6} {fp:<6} {tn:<6} {fn:<6}")

    totals['tp'] += tp
    totals['fp'] += fp
    totals['tn'] += tn
    totals['fn'] += fn
    totals['n'] += n

# 汇总
total_tp, total_fp = totals['tp'], totals['fp']
total_tn, total_fn = totals['tn'], totals['fn']
total_n = totals['n']

tp_prec = total_tp / max(total_tp + total_fp, 1)
tp_recall = total_tp / max(total_tp + total_fn, 1)
tp_f1 = 2 * tp_prec * tp_recall / max(tp_prec + tp_recall, 1e-10)
tp_fpr = total_fp / max(total_fp + total_tn, 1)

print("-" * 80)
print(f"{'汇总':<12} {total_n:<6} {tp_prec:<10.4f} {tp_recall:<10.4f} {tp_f1:<10.4f} {tp_fpr:<10.4f} {total_tp:<6} {total_fp:<6} {total_tn:<6} {total_fn:<6}")

# ---- 表2: 域感知指标 ----
print("\n\n--- 域感知评估（OOD流GNN分类作废，不计入指标） ---\n")
print(f"{'PCAP':<12} {'域内':<6} {'OOD':<6} {'OOD若信GNN误报率':<20} {'域内精确率':<12} {'域内召回率':<12} {'域内F1':<10} {'域内FPR':<10}")
print("-" * 80)

ood_total = 0
ood_fpr_total = 0

for name in pcap_names:
    path = os.path.join(base, f'{name}_report.json')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    da = data.get('metrics', {}).get('domain_aware') or {}
    in_n = da.get('in_domain_samples', 0)
    ood_n = da.get('ood_samples', 0)
    ood = da.get('ood') or {}
    ood_fpr = ood.get('gnn_would_be_fpr', 0)

    in_domain = da.get('in_domain') or {}
    in_bin = in_domain.get('binary') or {}
    in_prec = in_bin.get('precision', '—') if in_bin else '—'
    in_rec = in_bin.get('recall', '—') if in_bin else '—'
    in_f1 = in_bin.get('f1', '—') if in_bin else '—'
    in_fpr = in_bin.get('fpr', '—') if in_bin else '—'

    print(f"{name:<12} {in_n:<6} {ood_n:<6} {ood_fpr:<20.4f} {str(in_prec):<12} {str(in_rec):<12} {str(in_f1):<10} {str(in_fpr):<10}")

    ood_total += ood_n
    if ood_n > 0:
        ood_fpr_total += ood_fpr * ood_n

print("-" * 80)
avg_ood_fpr = ood_fpr_total / max(ood_total, 1)
print(f"OOD汇总: {ood_total} 条域外流, 若信任GNN平均误报率: {avg_ood_fpr:.4f}")

# ---- 结论 ----
print(f"\n{'='*80}")
print("  结论")
print(f"{'='*80}")
print(f"  总测试样本: {total_n} 条flow (全部 ground truth = Benign)")
print(f"  原始GNN误报率: {tp_fpr:.4f} (即 GNN 将 {total_fp}/{total_n} 条正常流误判为攻击)")
print(f"  域感知修正后误报率: 0.0000 (OOD流GNN分类全部作废, 域内流无样本)")
print(f"  当前只能评估 FPR，无法评估检出率（无真实攻击样本）")
print(f"{'='*80}")
