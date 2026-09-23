"""
Train the NEMESYS field-graph protocol classifier.

Usage:
    python train_proto_classifier.py --data-dir data/data --output field_proto_model.pth
    python train_proto_classifier.py --data-dir data/data --simulated --sages
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from nemesys_gnn4id.nemesys.field_graph_builder import (
    FIELD_FEAT_DIM,
    PROTO_CLASSES,
    PROTO_FEAT_DIM,
    PROTO_TO_ID,
    build_proto_graph,
)
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE, evaluate, train_epoch
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'data')
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'field_proto_model.pth')
PCAP_STEMS = {
    'DHCP': 'dhcp_100',
    'DNS': 'dns_100',
    'NTP': 'ntp_100',
    'Modbus': 'modbus_100',
    'DNP3': 'dnp3_100',
    'S7Comm': 's7comm_100',
    'MQTT': 'iot_mqtt',
    'HTTP': 'http_100',
    'SSH': 'ssh_100',
    'FTP': 'ftp_100',
    'TLS': 'tls_100',
    'mDNS': 'mdns_100',
}


def _message_to_segments(msg):
    """Return field segments and transport metadata from one NEMESYS message."""
    sport = dport = ip_proto = 0
    msg_segments = msg

    if isinstance(msg, dict):
        sport = int(msg.get('sport', 0))
        dport = int(msg.get('dport', 0))
        ip_proto = int(msg.get('proto', 0))
        msg_segments = msg.get('field_types', [])
    elif isinstance(msg, list) and msg:
        msg_obj = getattr(msg[0], 'message', None)
        if msg_obj:
            sport = int(getattr(msg_obj, 'sport', 0))
            dport = int(getattr(msg_obj, 'dport', 0))
            ip_proto = int(getattr(msg_obj, 'proto', 0))

    return msg_segments, sport, dport, ip_proto


def collect_all_graphs(nemesys_analyzer, data_dir, samples_per_class=100, drop_ports=False):
    """Run NEMESYS over all protocol PCAPs and build field graphs."""
    all_graphs = []
    all_labels = []
    stats = {}

    for proto_name in PROTO_CLASSES:
        pcap_path = os.path.join(data_dir, f'{PCAP_STEMS[proto_name]}.pcap')
        if not os.path.exists(pcap_path):
            print(f'[SKIP] {pcap_path} 不存在')
            stats[proto_name] = 0
            continue

        print(f'[{proto_name}] 运行 NEMESYS BCDG...')
        result = nemesys_analyzer.analyze(pcap_path)
        segments = result.get('segments', [])

        label_id = PROTO_TO_ID[proto_name]
        graphs_built = 0

        for i, msg in enumerate(segments):
            if graphs_built >= samples_per_class:
                break

            msg_segments, sport, dport, ip_proto = _message_to_segments(msg)
            graph_sport = 0 if drop_ports else sport
            graph_dport = 0 if drop_ports else dport
            try:
                graph = build_proto_graph(
                    msg_segments,
                    sport=graph_sport,
                    dport=graph_dport,
                    proto=ip_proto,
                    label=label_id,
                )
                all_graphs.append(graph)
                all_labels.append(label_id)
                graphs_built += 1
            except Exception as e:
                print(f'  [WARN] 消息 {i} 图构建失败: {e}')

        stats[proto_name] = graphs_built
        print(f'  [{proto_name}] 构建 {graphs_built} 张图')

    return all_graphs, all_labels, stats


def stratified_split(graphs, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Stratified train/validation/test split."""
    random.seed(seed)
    np.random.seed(seed)

    split = {
        'train': {'graphs': [], 'labels': []},
        'val': {'graphs': [], 'labels': []},
        'test': {'graphs': [], 'labels': []},
    }

    for c in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == c]
        random.shuffle(indices)
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        parts = {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:],
        }
        for name, part_indices in parts.items():
            for idx in part_indices:
                split[name]['graphs'].append(graphs[idx])
                split[name]['labels'].append(labels[idx])

    return split


def _json_safe(obj):
    """Convert metrics containing numpy scalars to JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def _print_dataset_distribution(labels, prefix):
    counts = Counter(labels)
    print(prefix)
    for i, name in enumerate(PROTO_CLASSES):
        print(f'  {name}: {counts.get(i, 0)}')


def _print_metrics(metrics, title):
    print('=' * 85)
    print(f'  {title}')
    print('=' * 85)
    print(f"整体准确率: {metrics.get('accuracy', 0):.4f}")
    macro = metrics.get('macro', {})
    print(f"Macro Precision: {macro.get('precision', 0):.4f}")
    print(f"Macro Recall:    {macro.get('recall', 0):.4f}")
    print(f"Macro F1:        {macro.get('f1', 0):.4f}")
    print(f"Macro FPR:       {macro.get('fpr', 0):.4f}")
    print()
    print("%-10s %6s %6s %6s %6s %10s %10s %10s %10s" % (
        "协议", "TP", "FP", "FN", "TN", "精确率", "召回率", "F1", "FPR"))
    print("-" * 85)
    for c, proto_name in enumerate(PROTO_CLASSES):
        pc = metrics.get('per_class', {}).get(c) or metrics.get('per_class', {}).get(str(c), {})
        print("%-10s %6d %6d %6d %6d %10.4f %10.4f %10.4f %10.4f" % (
            proto_name,
            pc.get('tp', 0), pc.get('fp', 0), pc.get('fn', 0), pc.get('tn', 0),
            pc.get('precision', 0), pc.get('recall', 0), pc.get('f1', 0), pc.get('fpr', 0)))
    print("=" * 85)


def save_checkpoint(path, model, args, best_metrics, final_metrics, best_epoch, model_class_name):
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_class': model_class_name,
        'hidden_dim': args.hidden,
        'latent_dim': args.latent,
        'num_classes': len(PROTO_CLASSES),
        'proto_classes': list(PROTO_CLASSES),
        'field_feat_dim': FIELD_FEAT_DIM,
        'proto_feat_dim': PROTO_FEAT_DIM,
        'v2_heads': getattr(args, 'v2_heads', None) if getattr(args, 'v2', False) else None,
        'confidence_threshold': args.confidence_threshold,
        'drop_ports': bool(args.drop_ports),
        'best_epoch': best_epoch,
        'best_metrics': _json_safe(best_metrics),
        'final_metrics': _json_safe(final_metrics),
        'training_args': vars(args),
    }
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='Train FieldProtoGNN protocol classifier')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='协议PCAP目录')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='输出checkpoint路径')
    parser.add_argument('--simulated', action='store_true', help='强制使用模拟BCDG/协议指纹模式')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--hidden', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--latent', type=int, default=32, help='latent维度')
    parser.add_argument('--samples-per-class', type=int, default=100, help='每类最多使用消息数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--sages', action='store_true', help='使用纯SAGEConv模型')
    model_group.add_argument('--v2', action='store_true', help='训练 FieldProtoGNN_V2（残差+归一化+池化+多头）')
    parser.add_argument('--v2-heads', type=int, default=4, help='V2 GAT 头数')
    parser.add_argument('--batch-size', type=int, default=32, help='mini-batch 大小')
    parser.add_argument('--drop-ports', action='store_true',
                        help='训练和推理时屏蔽源/目的端口特征，降低端口捷径依赖')
    parser.add_argument('--confidence-threshold', type=float, default=0.65,
                        help='推理时低置信度阈值')
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(f'--batch-size must be >= 1, got {args.batch_size}')
    if args.v2 and args.v2_heads < 1:
        parser.error(f'--v2-heads must be >= 1, got {args.v2_heads}')

    if args.v2 and args.lr == 0.005:
        # V2 (residual+LayerNorm+multi-head GAT+pooling) collapses at lr=0.005;
        # diagnosed: encoder activations die, single-class predictions.
        # v1 (no norm/residual) tolerates 0.005; V2 needs 0.001.
        args.lr = 0.001
        print(f'[INFO] --v2: 学习率从 0.005 调整为 0.001（V2 架构在 0.005 下坍缩，见 W0-4 诊断）')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')
    print(f'数据目录: {args.data_dir}')
    print(f'输出模型: {args.output}')
    print(f'端口特征: {"屏蔽" if args.drop_ports else "保留"}')
    print()

    analyzer = NEMESYSAnalyzer(sigma=0.6)
    if args.simulated:
        print('[INFO] 强制使用模拟 BCDG 模式')
        analyzer.nemere_available = False

    graphs, labels, stats = collect_all_graphs(
        analyzer,
        data_dir=args.data_dir,
        samples_per_class=args.samples_per_class,
        drop_ports=args.drop_ports,
    )

    print(f'\n总图数: {len(graphs)}')
    for proto, count in stats.items():
        print(f'  {proto}: {count}')
    if not graphs:
        raise SystemExit('[ERROR] 无有效图数据')

    split = stratified_split(graphs, labels, seed=args.seed)
    print()
    _print_dataset_distribution(split['train']['labels'], '训练集分布:')
    _print_dataset_distribution(split['val']['labels'], '验证集分布:')
    _print_dataset_distribution(split['test']['labels'], '测试集分布:')
    print()

    class_weight = None
    if args.v2:
        from collections import Counter
        train_counts = Counter(split['train']['labels'])
        n_train = len(split['train']['labels'])
        n_classes = len(PROTO_CLASSES)
        class_weight = torch.tensor(
            [n_train / max(train_counts.get(c, 0), 1) / n_classes for c in range(n_classes)],
            dtype=torch.float32).to(device)
        class_weight = class_weight / class_weight.mean()  # normalize to mean 1
        print('[INFO] 类别加权已启用（V2 训练）')

    if args.v2:
        from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN_V2
        model_cls = FieldProtoGNN_V2
        model_class_name = 'FieldProtoGNN_V2'
        model = model_cls(
            hidden_dim=args.hidden, latent_dim=args.latent,
            num_classes=len(PROTO_CLASSES), dropout=0.3, heads=args.v2_heads,
        ).to(device)
    else:
        model_cls = FieldProtoGNN_SAGE if args.sages else FieldProtoGNN
        model_class_name = 'FieldProtoGNN_SAGE' if args.sages else 'FieldProtoGNN'
        model = model_cls(
            hidden_dim=args.hidden, latent_dim=args.latent,
            num_classes=len(PROTO_CLASSES), dropout=0.3,
        ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型: {model_class_name}, 参数量: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    best_metrics = None
    patience = 15
    no_improve = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, split['train']['graphs'], split['train']['labels'], optimizer, device,
            batch_size=args.batch_size,
            class_weight=(class_weight if args.v2 else None))
        val_metrics = evaluate(model, split['val']['graphs'], split['val']['labels'], device)
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['macro']['f1']
        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
                  f'val_loss={val_loss:.4f} val_acc={val_metrics["accuracy"]:.4f} '
                  f'val_f1={val_f1:.4f}')

        if no_improve >= patience:
            print(f'[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}, best_val_f1={best_val_f1:.4f}')
            break

    elapsed = time.time() - start
    print(f'\n训练完成: {elapsed:.1f}s, 最佳 epoch={best_epoch}, 最佳验证 F1={best_val_f1:.4f}')

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, split['test']['graphs'], split['test']['labels'], device)
    _print_metrics(final_metrics, '最终测试集评估结果')

    save_checkpoint(args.output, model, args, best_metrics, final_metrics, best_epoch, model_class_name)
    print(f'\n[SAVE] FieldProtoGNN checkpoint 已保存: {args.output}')

    metrics_path = os.path.splitext(args.output)[0] + '.metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'best_epoch': best_epoch,
            'best_metrics': _json_safe(best_metrics),
            'final_metrics': _json_safe(final_metrics),
            'training_args': vars(args),
        }, f, indent=2, ensure_ascii=False)
    print(f'[SAVE] 训练指标已保存: {metrics_path}')


if __name__ == '__main__':
    main()
