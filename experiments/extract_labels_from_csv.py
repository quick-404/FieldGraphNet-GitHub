"""
将 MachineLearningCVE CSV 中的标签提取为与 nfstream output 对齐的 labels.txt

用法:
    python extract_labels_from_csv.py ^
        --csv "data/data/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv" ^
        --pcap "data/data/pcap/Friday-WorkingHours.pcap" ^
        --max-flows 200 ^
        -o labels.txt
"""

import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from config import ATTACK_TYPES, LABEL_DICT

# CICIDS2017 标签 → GNN4ID 8 类映射
LABEL_MAP = {
    'BENIGN': 0,
    'Web Based': 1, 'WebBased': 1, 'Web Attack': 1, 'XSS': 1, 'Sql Injection': 1, 'SqlInjection': 1,
    'Brute Force': 1, 'Web Brute Force': 1,
    'Spoofing': 2,
    'Recon': 3, 'Reconnaissance': 3, 'PortScan': 3,
    'Mirai': 4,
    'DoS': 5, 'Dos': 5, 'DoS Hulk': 5, 'DoS GoldenEye': 5, 'DoS Slowloris': 5, 'Slowhttptest': 5,
    'Heartbleed': 5, 'Slowloris': 5,
    'DDoS': 6, 'DDOS': 6, 'DDOS attack': 6,
    'BruteForce': 7, 'FTP-Patator': 7, 'SSH-Patator': 7, 'Patator': 7,
    'Infiltration': 1, 'Infilteration': 1,
    'Bot': 4,
}


def flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """方向无关的 flow key"""
    ep1 = (str(src_ip), int(src_port))
    ep2 = (str(dst_ip), int(dst_port))
    if ep1 <= ep2:
        key = (str(src_ip), int(src_port), str(dst_ip), int(dst_port), int(proto))
    else:
        key = (str(dst_ip), int(dst_port), str(src_ip), int(src_port), int(proto))
    return key


def main():
    parser = argparse.ArgumentParser(description='从 CIC CSV 提取标签并对齐到 nfstream')
    parser.add_argument('--csv', required=True, help='MachineLearningCVE CSV 文件')
    parser.add_argument('--pcap', required=True, help='对应的 PCAP 文件路径（用于提取 5 元组）')
    parser.add_argument('--max-flows', type=int, default=1000, help='nfstream 最大 flow 数')
    parser.add_argument('-o', '--output', default='labels.txt', help='输出 labels.txt 路径')
    args = parser.parse_args()

    # 1. 从 CSV 加载标签，构建 key→label 映射
    print(f"[EXTRACT] 加载 CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[EXTRACT] CSV 共 {len(df)} 行")

    csv_labels = {}
    csv_label_counts = {}
    for _, row in df.iterrows():
        raw_label = str(row.get(' Label', 'BENIGN')).strip()
        label_id = LABEL_MAP.get(raw_label, 0)
        key = flow_key(
            row.get(' Source IP', '0.0.0.0'),
            row.get(' Destination IP', '0.0.0.0'),
            row.get(' Source Port', 0),
            row.get(' Destination Port', 0),
            row.get(' Protocol', 0),
        )
        if key not in csv_labels:
            csv_labels[key] = label_id
            csv_label_counts[label_id] = csv_label_counts.get(label_id, 0) + 1

    print(f"[EXTRACT] CSV 中唯一 flow 数: {len(csv_labels)}")
    for cls_id, cnt in sorted(csv_label_counts.items()):
        name = ATTACK_TYPES.get(cls_id, {}).get('name', f'Class{cls_id}')
        print(f"  {name}: {cnt}")

    # 2. 用 nfstream 从 pcap 提取 flow 的 5 元组
    print(f"[EXTRACT] nfstream 提取 PCAP: {args.pcap}")
    try:
        from nfstream import NFStreamer
    except ImportError:
        print("[ERROR] nfstream 未安装")
        sys.exit(1)

    streamer = NFStreamer(
        source=args.pcap,
        accounting_mode=1,
        idle_timeout=120,
        n_dissections=0,
    )
    import tempfile
    tmp_csv = os.path.join(tempfile.gettempdir(), f'_nfstream_labels_{os.getpid()}.csv')
    streamer.to_csv(path=tmp_csv)
    ndf = pd.read_csv(tmp_csv)
    os.unlink(tmp_csv)

    if args.max_flows and len(ndf) > args.max_flows:
        ndf = ndf.head(args.max_flows)

    print(f"[EXTRACT] nfstream 提取了 {len(ndf)} 条 flow")

    # 3. 按 5 元组对齐
    matched = 0
    unmatched = 0
    labels_out = []
    for _, row in ndf.iterrows():
        key = flow_key(
            row.get('src_ip', row.get(' Source IP', '0.0.0.0')),
            row.get('dst_ip', row.get(' Destination IP', '0.0.0.0')),
            row.get('src_port', row.get(' Source Port', 0)),
            row.get('dst_port', row.get(' Destination Port', 0)),
            row.get('protocol', row.get(' Protocol', 0)),
        )
        label = csv_labels.get(key)
        if label is not None:
            labels_out.append(label)
            matched += 1
        else:
            labels_out.append(0)  # 未匹配的默认 Benign
            unmatched += 1

    # 4. 保存
    with open(args.output, 'w') as f:
        for label in labels_out:
            f.write(f'{label}\n')

    print(f"[EXTRACT] 对齐结果: {matched} 匹配, {unmatched} 未匹配（默认 Benign）")
    print(f"[EXTRACT] 标签已保存: {args.output} ({len(labels_out)} 行)")

    # 统计
    from collections import Counter
    cnt = Counter(labels_out)
    print("[EXTRACT] 标签分布:")
    for cls_id, n in sorted(cnt.items()):
        name = ATTACK_TYPES.get(cls_id, {}).get('name', f'Class{cls_id}')
        print(f"  {name}: {n}")


if __name__ == '__main__':
    main()
