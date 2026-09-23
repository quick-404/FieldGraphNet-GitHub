"""
Pure GNN vs Fusion Pipeline 混淆矩阵对比 — cic2023 三 PCAP

用法:
    python3 compare_gnn_fusion.py [--max-flows 500]

输出: data/data/cic2023/cm_pure_gnn.png  +  cm_fusion.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from collections import Counter

from gnn4id_analyzer import GNN4IDAnalyzer
from pipeline import TrafficPipeline
from plot_confusion_matrix import compute_confusion_matrix, plot_confusion_matrix, CLASS_NAMES

# (相对路径, 真实类别ID, 显示名)
PCAPS = [
    ('data/data/cic2023/Benign_500.pcap', 0, 'Benign'),
    ('data/data/cic2023/DNS_500.pcap', 2, 'Spoofing'),
    ('data/data/cic2023/Mirai_500.pcap', 4, 'Mirai'),
]


def run_gnn(analyzer, pcap_path, max_flows):
    """纯 GNN 推理."""
    results = analyzer.analyze_pcap(pcap_path, max_flows=max_flows)
    return [r['class_id'] for r in results]


def run_fusion(pipeline, pcap_path, max_flows):
    """通过融合流水线推理，取覆写后的最终 class_id."""
    result = pipeline.analyze(pcap_path, max_flows=max_flows, deep_analysis=False)
    results_list = result.get('gnn', {}).get('results', [])
    return [r['class_id'] for r in results_list]


def print_dist(class_ids, tag):
    dist = Counter(class_ids)
    parts = [f"    {CLASS_NAMES[c]}: {n}" for c, n in sorted(dist.items())]
    print(f"  {tag}: {len(class_ids)} flows")
    for p in parts:
        print(p)


def main():
    parser = argparse.ArgumentParser(description='Pure GNN vs Fusion 混淆矩阵对比')
    parser.add_argument('--model',
                        default=os.path.expanduser('~/网络流量分析项目/GNN4ID/model.pth'))
    parser.add_argument('--proto-model',
                        default=os.path.expanduser('~/网络流量分析项目/nemesys-gnn4id/field_proto_model.pth'))
    parser.add_argument('--max-flows', type=int, default=500)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    # ====== 1/2: Pure GNN ======
    print("=" * 60)
    print("  1/2: Pure GNN4ID")
    print("=" * 60)
    gnn = GNN4IDAnalyzer(model_path=args.model, device=args.device)
    if not gnn.model_loaded:
        print("[ERROR] 模型加载失败")
        sys.exit(1)

    y_true, y_pred_gnn = [], []
    for relpath, cls, name in PCAPS:
        abspath = os.path.join(base, relpath)
        preds = run_gnn(gnn, abspath, args.max_flows)
        n = len(preds)
        y_true.extend([cls] * n)
        y_pred_gnn.extend(preds)
        print_dist(preds, name)

    cm_gnn_raw, cm_gnn_pct = compute_confusion_matrix(y_true, y_pred_gnn)
    total_n = int(cm_gnn_raw.sum())
    acc_gnn = cm_gnn_raw.diagonal().sum() / cm_gnn_raw.sum() * 100

    print(f"\n  >>> Pure GNN 总准确率: {acc_gnn:.2f}% ({total_n} 样本)")

    cm_path_gnn = os.path.join(base, 'data/data/cic2023/cm_pure_gnn.png')
    plot_confusion_matrix(
        cm_gnn_pct, cm_gnn_raw, cm_path_gnn,
        title=f'Pure GNN4ID ({total_n} samples, {acc_gnn:.1f}%)'
    )
    print(f"  >>> 已保存: {cm_path_gnn}")

    # ====== 2/2: Fusion Pipeline ======
    print("\n" + "=" * 60)
    print("  2/2: Fusion Pipeline")
    print("=" * 60)
    pipeline = TrafficPipeline(
        model_path=args.model,
        device=args.device,
        proto_model_path=args.proto_model,
    )

    y_pred_fusion = []
    for relpath, cls, name in PCAPS:
        abspath = os.path.join(base, relpath)
        preds = run_fusion(pipeline, abspath, args.max_flows)
        # 跟纯 GNN 的 flow 数对齐
        min_n = min(len(preds), sum(1 for t in y_true if t == cls))
        y_pred_fusion.extend(preds[:min_n])
        print_dist(preds, name)

    # 截断到匹配长度
    n_fusion = min(len(y_true), len(y_pred_fusion))
    cm_fus_raw, cm_fus_pct = compute_confusion_matrix(y_true[:n_fusion], y_pred_fusion[:n_fusion])
    acc_fus = cm_fus_raw.diagonal().sum() / cm_fus_raw.sum() * 100

    print(f"\n  >>> Fusion 总准确率: {acc_fus:.2f}% ({n_fusion} 样本)")

    cm_path_fus = os.path.join(base, 'data/data/cic2023/cm_fusion.png')
    plot_confusion_matrix(
        cm_fus_pct, cm_fus_raw, cm_path_fus,
        title=f'Fusion Pipeline ({n_fusion} samples, {acc_fus:.1f}%)'
    )
    print(f"  >>> 已保存: {cm_path_fus}")

    # ====== Summary ======
    print("\n" + "=" * 60)
    print("  对比总结")
    print("=" * 60)
    print(f"  纯 GNN:     {acc_gnn:.2f}%")
    print(f"  Fusion:     {acc_fus:.2f}%")
    print(f"  差值:       {acc_fus - acc_gnn:+.2f}%")
    print()
    print(f"  cm_pure_gnn.png — 纯 GNN4ID 预测")
    print(f"  cm_fusion.png  — Fusion 流水线最终输出")


if __name__ == '__main__':
    main()
