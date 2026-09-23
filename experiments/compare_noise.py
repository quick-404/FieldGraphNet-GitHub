"""Noise-robustness comparison: does BCDG or MDIplier segmentation degrade less
under payload corruption/truncation when feeding the same FieldProtoGNN?

For each noise config (clean / corrupt X% / truncate Y%), the protocol pcaps are
perturbed deterministically, both segmentations re-run on the noisy messages,
and the same FieldProtoGNN is trained/tested on each side's graphs.

Usage:
    python experiments/compare_noise.py [--epochs 60] [--hidden 64] [--seed 42]
        [--out eval_results/fusion_comparison/mdiplier_vs_bcdg_noise.json]
"""
import argparse
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch
from scapy.all import rdpcap, wrpcap, Raw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdiplier_utils import PROTOCOLS_6, collect_mdiplier_messages, align_by_hex
from compare_mdiplier_vs_bcdg import (
    DATA_DIR, PCAP_STEMS, NEMESYSAnalyzer,
    collect_bcdg_messages, messages_to_graphs, train_and_evaluate,
)

NOISE_CONFIGS = [
    ('clean', 0.0),
    ('corrupt', 0.2),
    ('corrupt', 0.4),
    ('truncate', 0.3),
    ('truncate', 0.5),
]


def make_noisy_pcap(src, dst, noise_type, level, seed=42):
    """Deterministically corrupt/truncate packet payloads and write a new pcap."""
    pkts = rdpcap(src)
    rng = random.Random(seed)
    for p in pkts:
        if not p.haslayer(Raw):
            continue
        payload = bytearray(p[Raw].load)
        if noise_type == 'corrupt':
            n = max(1, int(len(payload) * level))
            for i in rng.sample(range(len(payload)), min(n, len(payload))):
                payload[i] ^= rng.randint(1, 255)
        elif noise_type == 'truncate':
            keep = max(1, int(len(payload) * (1 - level)))
            payload = payload[:keep]
        p[Raw].load = bytes(payload)
        for layer in ('TCP', 'UDP'):
            if p.haslayer(layer):
                try:
                    del p[layer].chksum
                except Exception:
                    pass
    wrpcap(dst, pkts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='eval_results/fusion_comparison/mdiplier_vs_bcdg_noise.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    analyzer = NEMESYSAnalyzer()
    proto_to_id = {p: i for i, p in enumerate(PROTOCOLS_6)}
    num_classes = len(PROTOCOLS_6)

    all_results = []
    for noise_type, level in NOISE_CONFIGS:
        print(f'\n===== noise={noise_type} level={level} =====')
        bcdg_graphs, bcdg_labels, mdi_graphs, mdi_labels = [], [], [], []
        alignment_report = {}
        with tempfile.TemporaryDirectory() as base:
            for proto in PROTOCOLS_6:
                src_pcap = os.path.join(DATA_DIR, f'{PCAP_STEMS[proto]}.pcap')
                if not os.path.exists(src_pcap):
                    continue
                label_id = proto_to_id[proto]

                if noise_type == 'clean':
                    pcap = src_pcap
                else:
                    pcap = os.path.join(base, f'{proto}_noisy_{noise_type}_{level}.pcap')
                    make_noisy_pcap(src_pcap, pcap, noise_type, level, seed=args.seed)

                bcdg_msgs = collect_bcdg_messages(analyzer, pcap)
                try:
                    mdi_msgs = collect_mdiplier_messages(pcap, os.path.join(base, f'mdi_{proto}'))
                except Exception as exc:
                    print(f'  [{proto}] MDIplier FAILED: {exc}')
                    alignment_report[proto] = {'error': str(exc)[:200]}
                    continue

                b_aligned, m_aligned, rate = align_by_hex(bcdg_msgs, mdi_msgs)
                alignment_report[proto] = {
                    'bcdg_msgs': len(bcdg_msgs), 'mdi_msgs': len(mdi_msgs),
                    'aligned': len(b_aligned), 'match_rate': round(rate, 4),
                }
                print(f'  [{proto}] bcdg={len(bcdg_msgs)} mdi={len(mdi_msgs)} '
                      f'aligned={len(b_aligned)} match={rate:.2f}')
                if len(b_aligned) >= 10 and len(m_aligned) >= 10:
                    b_g, b_l = messages_to_graphs(b_aligned, label_id, is_mdiplier=False)
                    m_g, m_l = messages_to_graphs(m_aligned, label_id, is_mdiplier=True)
                else:
                    b_g, b_l = messages_to_graphs(bcdg_msgs, label_id, is_mdiplier=False)
                    m_g, m_l = messages_to_graphs(mdi_msgs, label_id, is_mdiplier=True)
                    alignment_report[proto]['fallback'] = 'full-sets'
                bcdg_graphs += b_g; bcdg_labels += b_l
                mdi_graphs += m_g; mdi_labels += m_l

        print(f'  BCDG graphs={len(bcdg_graphs)}  MDIplier graphs={len(mdi_graphs)}')
        if len(bcdg_graphs) < 20 or len(mdi_graphs) < 20:
            print('  [SKIP] too few graphs for this noise config')
            continue
        bcdg_metrics = train_and_evaluate('BCDG', bcdg_graphs, bcdg_labels,
                                          args.epochs, args.hidden, device, args.seed, num_classes)
        mdi_metrics = train_and_evaluate('MDIplier', mdi_graphs, mdi_labels,
                                         args.epochs, args.hidden, device, args.seed, num_classes)
        entry = {
            'noise': noise_type, 'level': level,
            'bcdg_accuracy': round(bcdg_metrics.get('accuracy', 0), 4),
            'mdiplier_accuracy': round(mdi_metrics.get('accuracy', 0), 4),
            'delta': round(bcdg_metrics.get('accuracy', 0) - mdi_metrics.get('accuracy', 0), 4),
            'n_bcdg_graphs': len(bcdg_graphs), 'n_mdi_graphs': len(mdi_graphs),
            'alignment': alignment_report,
        }
        all_results.append(entry)
        print(f"  BCDG={entry['bcdg_accuracy']:.4f} MDIplier={entry['mdiplier_accuracy']:.4f} "
              f"delta={entry['delta']:+.4f}")

    summary = {
        'config': {'epochs': args.epochs, 'hidden': args.hidden, 'seed': args.seed,
                   'protocols': PROTOCOLS_6, 'noise_configs': NOISE_CONFIGS},
        'results': all_results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {args.out}')
    for r in all_results:
        print(f"  {r['noise']:9s} {r['level']:>5}  BCDG={r['bcdg_accuracy']:.4f}  "
              f"MDIplier={r['mdiplier_accuracy']:.4f}  Δ={r['delta']:+.4f}")


if __name__ == '__main__':
    main()
