"""
Validate that the FieldProtoGNN protocol classifier is not dominated by ports.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from nemesys_gnn4id.nemesys.field_graph_builder import PROTO_CLASSES, PROTO_TO_ID, build_proto_graph
from nemesys_gnn4id.proto_gnn.model import evaluate
from field_protocol_classifier import FieldProtocolClassifier
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from train_proto_classifier import DEFAULT_DATA_DIR, PCAP_STEMS, _message_to_segments


PORT_MAP = {
    'DHCP': (9067, 9068),
    'DNS': (9053, 9053),
    'NTP': (9123, 9123),
    'Modbus': (9502, 9502),
    'DNP3': (9200, 9200),
    'S7Comm': (9102, 9102),
    'MQTT': (9183, 9183),
    'HTTP': (9080, 9080),
    'SSH': (9022, 9022),
    'FTP': (9021, 9021),
}


def build_test_sets(analyzer, data_dir, samples_per_class=100, drop_ports=False):
    orig_graphs, rand_graphs = [], []
    labels = []

    for proto_name in PROTO_CLASSES:
        pcap_path = os.path.join(data_dir, f'{PCAP_STEMS[proto_name]}.pcap')
        if not os.path.exists(pcap_path):
            print(f'[SKIP] {pcap_path} 不存在')
            continue

        result = analyzer.analyze(pcap_path)
        segments = result.get('segments', [])
        label_id = PROTO_TO_ID[proto_name]
        fake_sport, fake_dport = PORT_MAP[proto_name]
        built = 0

        for msg in segments:
            if built >= samples_per_class:
                break
            msg_segments, sport, dport, ip_proto = _message_to_segments(msg)
            orig_sport = 0 if drop_ports else sport
            orig_dport = 0 if drop_ports else dport
            rand_sport = 0 if drop_ports else fake_sport
            rand_dport = 0 if drop_ports else fake_dport
            try:
                orig_graphs.append(build_proto_graph(
                    msg_segments, sport=orig_sport, dport=orig_dport, proto=ip_proto, label=label_id))
                rand_graphs.append(build_proto_graph(
                    msg_segments, sport=rand_sport, dport=rand_dport, proto=ip_proto, label=label_id))
                labels.append(label_id)
                built += 1
            except Exception:
                continue

        print(f'[{proto_name}] 构建测试图: {built}')

    return orig_graphs, rand_graphs, labels


def _print_summary(name, metrics):
    macro = metrics.get('macro', {})
    print("%-12s %10.4f %10.4f %10.4f %10.4f %10.4f" % (
        name,
        metrics.get('accuracy', 0),
        macro.get('precision', 0),
        macro.get('recall', 0),
        macro.get('f1', 0),
        macro.get('fpr', 0),
    ))


def main():
    parser = argparse.ArgumentParser(description='Validate port invariance for FieldProtoGNN')
    parser.add_argument('--model', default='field_proto_model.pth', help='FieldProtoGNN checkpoint')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='协议PCAP目录')
    parser.add_argument('--simulated', action='store_true', help='强制使用模拟BCDG/协议指纹模式')
    parser.add_argument('--samples-per-class', type=int, default=100, help='每类最多测试消息数')
    parser.add_argument('--confidence-threshold', type=float, default=None,
                        help='覆盖checkpoint中的低置信度阈值')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')

    classifier = FieldProtocolClassifier(
        args.model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )
    drop_ports = bool(classifier.checkpoint.get('drop_ports', False))
    print(f'端口特征: {"屏蔽" if drop_ports else "保留"}')
    analyzer = NEMESYSAnalyzer(sigma=0.6)
    if args.simulated:
        analyzer.nemere_available = False

    print('===== 1. NEMESYS 分析 + 图构建 =====')
    orig_graphs, rand_graphs, labels = build_test_sets(
        analyzer,
        args.data_dir,
        samples_per_class=args.samples_per_class,
        drop_ports=drop_ports,
    )
    print(f'原始端口数据集: {len(orig_graphs)} 图')
    print(f'随机端口数据集: {len(rand_graphs)} 图')
    if not orig_graphs:
        raise SystemExit('[ERROR] 无有效测试图')

    model = classifier.model
    r_orig = evaluate(model, orig_graphs, labels, device)
    r_rand = evaluate(model, rand_graphs, labels, device)

    print('=' * 85)
    print('  端口无关性验证')
    print('=' * 85)
    print("%-12s %10s %10s %10s %10s %10s" % (
        "", "准确率", "精确率", "召回率", "F1", "FPR"))
    print("-" * 70)
    _print_summary('原始端口', r_orig)
    _print_summary('随机端口', r_rand)

    n = len(labels)
    same_count = sum(1 for i in range(n) if r_orig['preds'][i] == r_rand['preds'][i])
    consistency = same_count / max(n, 1)
    print()
    print(f"预测一致性: {same_count}/{n} = {consistency * 100:.1f}%")

    changed = []
    for i in range(n):
        if r_orig['preds'][i] != r_rand['preds'][i]:
            changed.append((
                i,
                PROTO_CLASSES[labels[i]],
                PROTO_CLASSES[r_orig['preds'][i]],
                PROTO_CLASSES[r_rand['preds'][i]],
            ))

    if changed:
        print(f"预测变化的消息 ({len(changed)}):")
        for idx, true, orig_p, rand_p in changed[:20]:
            print(f"  [{idx}] 真实={true}, 原端口预测={orig_p}, 随机端口预测={rand_p}")

    if consistency > 0.9:
        conclusion = '结构 GNN 对端口不敏感，主要依赖字段结构做协议识别。'
    elif consistency > 0.7:
        conclusion = '端口有一定影响但非决定性，字段结构仍是主要信号。'
    else:
        conclusion = '端口影响较大，模型可能过度依赖端口特征。'
    print(f"结论: {conclusion}")
    print('=' * 85)


if __name__ == '__main__':
    main()
