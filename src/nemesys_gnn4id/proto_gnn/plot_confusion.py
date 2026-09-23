"""
混淆矩阵热力图生成工具

用法:
    # 从 pipeline JSON 结果生成
    python plot_confusion_matrix.py --report report.json -o confusion_matrix.png

    # 从 predictions + labels CSV 生成
    python plot_confusion_matrix.py --preds predictions.csv --labels labels.csv -o cm.png
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from nemesys_gnn4id.config import ATTACK_TYPES

# 8 类别固定顺序
CLASS_NAMES = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce']
CLASS_NAMES_ZH = ['正常流量', '网页攻击', '欺骗攻击', '侦察扫描', 'Mirai', 'Dos', 'DDos', '暴力破解']


def compute_confusion_matrix(y_true, y_pred, num_classes=8):
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    # 按行归一化为百分比
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_pct = cm / row_sums * 100
    return cm, cm_pct


def plot_confusion_matrix(cm_pct, cm_raw, output_path, class_names=None,
                          title='Confusion Matrix', figsize=(10, 8)):
    """
    绘制混淆矩阵热力图

    Args:
        cm_pct: 百分比归一化的混淆矩阵 (8x8)
        cm_raw: 原始计数矩阵 (8x8)
        output_path: 输出图片路径
        class_names: 类别名称列表
        title: 标题
        figsize: 图尺寸
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n = cm_pct.shape[0]

    # 配色：深蓝渐变（白→蓝）
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100,
                   aspect='equal')

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('分类占比 (%)', fontsize=11)

    # 设置刻度
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontsize=9)

    # 在每个格子中填写百分比
    for i in range(n):
        for j in range(n):
            pct = cm_pct[i, j]
            count = int(cm_raw[i, j])
            # 文字颜色：深色背景用白字，浅色背景用黑字
            text_color = 'white' if pct > 50 else 'black'
            ax.text(j, i, f'{pct:.1f}%', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')

    # 标签
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[CONFUSION] 混淆矩阵已保存: {output_path}")


def load_from_report(report_path):
    """从 pipeline JSON 报告加载 predictions 和 labels"""
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gnn_results = data.get('gnn', {}).get('results', [])
    metrics = data.get('metrics')
    if metrics is None:
        print("[CONFUSION] 报告中无评估指标（未提供 --labels 标签）")
        print("[CONFUSION] 请使用 --labels 重新运行分析，或在已有标签数据上运行")
        return None, None

    # 尝试从 metrics 的多分类指标中提取
    multiclass = metrics.get('multiclass', {})
    if multiclass:
        cm_raw = np.array(multiclass.get('confusion_matrix', []))
        if cm_raw.size == 64:
            cm_raw = cm_raw.reshape(8, 8)
            row_sums = cm_raw.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_pct = cm_raw / row_sums * 100
            return cm_raw, cm_pct

    # 尝试从 gnn_results 的 ground truth 中提取
    y_true = []
    y_pred = []
    for r in gnn_results:
        if 'true_label' in r and 'class_id' in r:
            y_true.append(r['true_label'])
            y_pred.append(r['class_id'])

    if y_true and y_pred:
        return compute_confusion_matrix(y_true, y_pred)

    print("[CONFUSION] 报告中未找到 ground truth 标签，无法生成混淆矩阵")
    return None, None


def main():
    parser = argparse.ArgumentParser(description='生成GNN4ID混淆矩阵热力图')
    parser.add_argument('--report', '-r', help='pipeline输出的JSON报告路径')
    parser.add_argument('--preds', help='预测标签CSV（每行一个整数 0-7）')
    parser.add_argument('--labels', help='真实标签CSV（每行一个整数 0-7）')
    parser.add_argument('--output', '-o', default='confusion_matrix.png',
                        help='输出图片路径 (默认: confusion_matrix.png)')
    parser.add_argument('--title', default='GNN4ID 分类混淆矩阵',
                        help='图表标题')
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 8),
                        help='图尺寸 (宽 高)')

    args = parser.parse_args()

    cm_raw = cm_pct = None

    if args.report:
        cm_raw, cm_pct = load_from_report(args.report)
    elif args.preds and args.labels:
        y_true = [int(line.strip()) for line in open(args.labels) if line.strip()]
        y_pred = [int(line.strip()) for line in open(args.preds) if line.strip()]
        if len(y_true) != len(y_pred):
            print(f"[ERROR] 标签数量不匹配: true={len(y_true)}, pred={len(y_pred)}")
            sys.exit(1)
        cm_raw, cm_pct = compute_confusion_matrix(y_true, y_pred)
    else:
        parser.print_help()
        sys.exit(1)

    if cm_raw is not None and cm_pct is not None:
        plot_confusion_matrix(cm_pct, cm_raw, args.output,
                              title=args.title, figsize=tuple(args.figsize))

        # 打印混淆矩阵文本摘要
        print("\n=== 混淆矩阵 (百分比) ===")
        print(f"{'':>12}", end='')
        for name in CLASS_NAMES:
            print(f"{name:>10}", end='')
        print()
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:>12}", end='')
            for j in range(8):
                print(f"{cm_pct[i,j]:>9.1f}%", end='')
            print()


if __name__ == '__main__':
    main()
