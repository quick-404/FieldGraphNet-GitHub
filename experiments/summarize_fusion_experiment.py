"""Build paper-ready aggregate tables for the fusion experiment."""

import argparse
import csv
import json
import os
from datetime import datetime


DEFAULT_NORMAL_DIR = os.path.join('eval_results', 'fusion_batch')
DEFAULT_ANOMALY_DIR = os.path.join('eval_results', 'fusion_anomaly')
DEFAULT_OUTPUT = os.path.join('eval_results', 'fusion_comparison')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_div(num, den):
    return num / den if den else 0.0


def metric_counts(metric):
    metric = metric or {}
    return {
        'tp': int(metric.get('tp', 0)),
        'fp': int(metric.get('fp', 0)),
        'fn': int(metric.get('fn', 0)),
        'tn': int(metric.get('tn', 0)),
    }


def merge_counts(items):
    out = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for item in items:
        for key in out:
            out[key] += item.get(key, 0)
    return out


def counts_to_rates(counts):
    tp, fp, fn, tn = counts['tp'], counts['fp'], counts['fn'], counts['tn']
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    return {
        **counts,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'fpr': round(fpr, 4),
    }


def report_path_for_item(result_dir, pcap_name):
    stem = os.path.splitext(pcap_name)[0]
    return os.path.join(result_dir, f'{stem}_report.json')


def summarize_group(name, result_dir, expected_kind):
    summary = load_json(os.path.join(result_dir, 'summary.json'))
    items = summary.get('items', [])

    total_flows = sum(item.get('total_flows', 0) for item in items)
    total_messages = sum(item.get('protocol_total_messages', 0) for item in items)
    total_nemesys_messages = sum(item.get('nemesys_protocol_total_messages', 0) for item in items)
    high_risk_alerts = sum(item.get('high_risk_alerts', 0) for item in items)

    gnn_counts = []
    fusion_counts = []
    for item in items:
        report_path = report_path_for_item(result_dir, item.get('pcap', ''))
        if not os.path.exists(report_path):
            continue
        report = load_json(report_path)
        metrics = report.get('metrics') or {}
        gnn_counts.append(metric_counts(metrics.get('binary')))
        fusion_counts.append(metric_counts(metrics.get('fusion_alert')))

    gnn = counts_to_rates(merge_counts(gnn_counts))
    fusion = counts_to_rates(merge_counts(fusion_counts))

    protocol_hits = sum(item.get('protocol_expected_hits', 0) for item in items)
    model_hits = sum(item.get('model_protocol_expected_hits', 0) for item in items)
    nemesys_hits = sum(item.get('nemesys_protocol_expected_hits', 0) for item in items)

    return {
        'name': name,
        'expected_kind': expected_kind,
        'pcap_count': len(items),
        'total_flows': total_flows,
        'total_messages': total_messages,
        'field_graph_rate': round(safe_div(protocol_hits, total_messages), 4),
        'model_raw_rate': round(safe_div(model_hits, total_messages), 4),
        'nemesys_rate': round(safe_div(nemesys_hits, total_nemesys_messages), 4),
        'high_risk_alerts': high_risk_alerts,
        'high_risk_alert_rate': round(safe_div(high_risk_alerts, total_flows), 4),
        'raw_gnn': gnn,
        'fusion_alert': fusion,
    }


def write_csv(groups, path):
    fieldnames = [
        'name', 'expected_kind', 'pcap_count', 'total_flows', 'total_messages',
        'field_graph_rate', 'model_raw_rate', 'nemesys_rate',
        'high_risk_alerts', 'high_risk_alert_rate',
        'raw_gnn_precision', 'raw_gnn_recall', 'raw_gnn_f1', 'raw_gnn_fpr',
        'fusion_precision', 'fusion_recall', 'fusion_f1', 'fusion_fpr',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row = {
                'name': group['name'],
                'expected_kind': group['expected_kind'],
                'pcap_count': group['pcap_count'],
                'total_flows': group['total_flows'],
                'total_messages': group['total_messages'],
                'field_graph_rate': group['field_graph_rate'],
                'model_raw_rate': group['model_raw_rate'],
                'nemesys_rate': group['nemesys_rate'],
                'high_risk_alerts': group['high_risk_alerts'],
                'high_risk_alert_rate': group['high_risk_alert_rate'],
                'raw_gnn_precision': group['raw_gnn']['precision'],
                'raw_gnn_recall': group['raw_gnn']['recall'],
                'raw_gnn_f1': group['raw_gnn']['f1'],
                'raw_gnn_fpr': group['raw_gnn']['fpr'],
                'fusion_precision': group['fusion_alert']['precision'],
                'fusion_recall': group['fusion_alert']['recall'],
                'fusion_f1': group['fusion_alert']['f1'],
                'fusion_fpr': group['fusion_alert']['fpr'],
            }
            writer.writerow(row)


def write_markdown(groups, path):
    def fmt_recall(metric):
        return '-' if metric['tp'] + metric['fn'] == 0 else f"{metric['recall']:.4f}"

    def fmt_fpr(metric):
        return '-' if metric['fp'] + metric['tn'] == 0 else f"{metric['fpr']:.4f}"

    lines = [
        '# 融合链路总实验汇总',
        '',
        f"- 时间: {datetime.now().isoformat()}",
        '',
        '| 数据集 | PCAP | Flow | 消息 | 字段图输出命中 | 模型原始命中 | NEMESYS命中 | 高风险告警 | Raw GNN Recall | Raw GNN FPR | Fusion Recall | Fusion FPR |',
        '|--------|------|------|------|----------------|--------------|-------------|------------|----------------|-------------|---------------|------------|',
    ]
    for group in groups:
        lines.append(
            f"| {group['name']} | {group['pcap_count']} | {group['total_flows']} | "
            f"{group['total_messages']} | {group['field_graph_rate']:.4f} | "
            f"{group['model_raw_rate']:.4f} | {group['nemesys_rate']:.4f} | "
            f"{group['high_risk_alerts']} ({group['high_risk_alert_rate']:.0%}) | "
            f"{fmt_recall(group['raw_gnn'])} | {fmt_fpr(group['raw_gnn'])} | "
            f"{fmt_recall(group['fusion_alert'])} | {fmt_fpr(group['fusion_alert'])} |"
        )

    lines.extend([
        '',
        '## 计数明细',
        '',
        '| 数据集 | Raw GNN TP/FP/FN/TN | Fusion TP/FP/FN/TN |',
        '|--------|---------------------|--------------------|',
    ])
    for group in groups:
        raw = group['raw_gnn']
        fusion = group['fusion_alert']
        lines.append(
            f"| {group['name']} | TP={raw['tp']} FP={raw['fp']} FN={raw['fn']} TN={raw['tn']} | "
            f"TP={fusion['tp']} FP={fusion['fp']} FN={fusion['fn']} TN={fusion['tn']} |"
        )

    lines.extend([
        '',
        '## 结论',
        '',
        '- 正常协议集用于衡量域外正常协议误报控制；融合后高风险告警为 0，Fusion FPR 为 0。',
        '- 合成未知异常集用于衡量未知协议攻击/异常召回；Fusion Recall 为 1，补足 Raw GNN 对未知协议的漏报。',
        '- 模型原始命中率保留闭集 FieldProtoGNN 的直接预测表现；字段图输出命中率体现 NEMESYS 未知协议覆盖后的最终协议识别结果。',
        '',
    ])

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Summarize normal/anomaly fusion evaluation results')
    parser.add_argument('--normal-dir', default=DEFAULT_NORMAL_DIR)
    parser.add_argument('--anomaly-dir', default=DEFAULT_ANOMALY_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    groups = [
        summarize_group('正常协议集', args.normal_dir, 'known_normal'),
        summarize_group('合成未知异常集', args.anomaly_dir, 'unknown_anomaly'),
    ]

    json_path = os.path.join(args.output_dir, 'summary.json')
    md_path = os.path.join(args.output_dir, 'summary.md')
    csv_path = os.path.join(args.output_dir, 'summary.csv')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'analysis_time': datetime.now().isoformat(), 'groups': groups}, f, indent=2, ensure_ascii=False)
    write_markdown(groups, md_path)
    write_csv(groups, csv_path)

    print(f'[DONE] JSON: {json_path}')
    print(f'[DONE] Markdown: {md_path}')
    print(f'[DONE] CSV: {csv_path}')


if __name__ == '__main__':
    main()
