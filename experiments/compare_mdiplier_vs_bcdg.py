"""Compare FieldProtoGNN protocol-classification accuracy fed by BCDG vs
MDIplier field segmentation, over 6 or 12 protocols.

Usage:
    python experiments/compare_mdiplier_vs_bcdg.py [--classes 12] [--epochs 60]
    [--hidden 64] [--seed 42] [--out eval_results/fusion_comparison/mdiplier_vs_bcdg_12.json]
"""
import argparse
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for mdiplier_utils

from mdiplier_utils import PROTOCOLS_6, collect_mdiplier_messages, align_by_hex
from nemesys_gnn4id.nemesys.field_graph_builder import build_proto_graph
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, train_epoch, evaluate

# TLS is excluded from the default 11-class set: MDIplier's mafft alignment
# does not converge on TLS's large encrypted payloads within a feasible time.
PROTOCOLS_12 = ['dhcp', 'dns', 'ntp', 'modbus', 'dnp3', 's7comm',
                'mqtt', 'http', 'ssh', 'ftp', 'tls', 'mdns']
PROTOCOLS_11 = [p for p in PROTOCOLS_12 if p != 'tls']

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'data')
PCAP_STEMS = {
    'dhcp': 'dhcp_100', 'dns': 'dns_100', 'ntp': 'ntp_100',
    'modbus': 'modbus_100', 'dnp3': 'dnp3_100', 's7comm': 's7comm_100',
    # iot_mqtt_50 / mdns_100_pkt are packet-trimmed subsets: MDIplier's mafft
    # alignment is too slow on the full multi-KB-payload captures.
    'mqtt': 'iot_mqtt_50', 'http': 'http_100', 'ssh': 'ssh_100',
    'ftp': 'ftp_100', 'tls': 'tls_100', 'mdns': 'mdns_100_pkt',
}


def collect_bcdg_messages(analyzer, pcap_path):
    """Return NEMESYS BCDG messages with payload_hex + field_types."""
    result = analyzer.analyze(pcap_path)
    return result.get('segments', [])


def messages_to_graphs(messages, label_id, is_mdiplier):
    """Build field graphs from either BCDG or MDIplier messages."""
    graphs, labels = [], []
    for msg in messages:
        if is_mdiplier:
            segments = msg['segments']
        else:
            segments = msg.get('field_types', [])
        try:
            graph = build_proto_graph(segments, sport=0, dport=0, proto=0, label=label_id)
        except Exception:
            continue
        graphs.append(graph)
        labels.append(label_id)
    return graphs, labels


def stratified_split(graphs, labels, seed=42, train_ratio=0.7):
    random.seed(seed)
    np.random.seed(seed)
    train_g, train_l, test_g, test_l = [], [], [], []
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        random.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        for i in idx[:n_train]:
            train_g.append(graphs[i]); train_l.append(labels[i])
        for i in idx[n_train:]:
            test_g.append(graphs[i]); test_l.append(labels[i])
    return train_g, train_l, test_g, test_l


def train_and_evaluate(name, graphs, labels, epochs, hidden, device, seed, num_classes):
    torch.manual_seed(seed)
    random.seed(seed)
    train_g, train_l, test_g, test_l = stratified_split(graphs, labels, seed=seed)
    model = FieldProtoGNN(
        hidden_dim=hidden, latent_dim=hidden // 2, num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        train_epoch(model, train_g, train_l, optimizer, device)
    metrics = evaluate(model, test_g, test_l, device)
    print(f'[{name}] test accuracy = {metrics.get("accuracy", 0):.4f} '
          f'(n_test={len(test_g)})')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, choices=[6, 11, 12], default=11,
                        help='number of protocol classes (6, 11, or 12; 11 excludes TLS)')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='eval_results/fusion_comparison/mdiplier_vs_bcdg_11.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    protocols = PROTOCOLS_6 if args.classes == 6 else (PROTOCOLS_11 if args.classes == 11 else PROTOCOLS_12)
    proto_to_id = {p: i for i, p in enumerate(protocols)}

    device = torch.device(args.device)
    analyzer = NEMESYSAnalyzer()

    bcdg_graphs, bcdg_labels, mdi_graphs, mdi_labels = [], [], [], []
    alignment_report = {}

    for proto in protocols:
        pcap = os.path.join(DATA_DIR, f'{PCAP_STEMS[proto]}.pcap')
        if not os.path.exists(pcap):
            print(f'[SKIP] {pcap} 不存在')
            continue
        label_id = proto_to_id[proto]

        # BCDG side
        bcdg_msgs = collect_bcdg_messages(analyzer, pcap)
        # MDIplier side
        with tempfile.TemporaryDirectory() as workdir:
            mdi_msgs = collect_mdiplier_messages(pcap, workdir)
        # Align by hex (diagnostic; not gating)
        b_aligned, m_aligned, rate = align_by_hex(bcdg_msgs, mdi_msgs)
        alignment_report[proto] = {'bcdg_msgs': len(bcdg_msgs), 'mdi_msgs': len(mdi_msgs),
                                   'aligned': len(b_aligned), 'match_rate': round(rate, 4)}
        print(f'[{proto}] bcdg={len(bcdg_msgs)} mdi={len(mdi_msgs)} aligned={len(b_aligned)} '
              f'match={rate:.2f}')
        if len(b_aligned) >= 10 and len(m_aligned) >= 10:
            b_g, b_l = messages_to_graphs(b_aligned, label_id, is_mdiplier=False)
            m_g, m_l = messages_to_graphs(m_aligned, label_id, is_mdiplier=True)
        else:
            # fallback: use full message sets (same pcaps => same distribution)
            b_g, b_l = messages_to_graphs(bcdg_msgs, label_id, is_mdiplier=False)
            m_g, m_l = messages_to_graphs(mdi_msgs, label_id, is_mdiplier=True)
            alignment_report[proto]['fallback'] = 'full-sets'
        bcdg_graphs += b_g; bcdg_labels += b_l
        mdi_graphs += m_g; mdi_labels += m_l

    print(f'\nBCDG graphs={len(bcdg_graphs)}  MDIplier graphs={len(mdi_graphs)}')
    bcdg_metrics = train_and_evaluate('BCDG', bcdg_graphs, bcdg_labels, args.epochs, args.hidden, device, args.seed, args.classes)
    mdi_metrics = train_and_evaluate('MDIplier', mdi_graphs, mdi_labels, args.epochs, args.hidden, device, args.seed, args.classes)

    summary = {
        'config': {'classes': args.classes, 'epochs': args.epochs, 'hidden': args.hidden,
                   'seed': args.seed, 'protocols': protocols},
        'alignment': alignment_report,
        'bcdg': bcdg_metrics,
        'mdiplier': mdi_metrics,
        'delta_accuracy': round(bcdg_metrics.get('accuracy', 0) - mdi_metrics.get('accuracy', 0), 4),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {args.out}')
    print(f'BCDG accuracy - MDIplier accuracy = {summary["delta_accuracy"]:.4f}')


if __name__ == '__main__':
    main()
