"""
攻击-协议相关性检查 (快速版)
流式读取 PCAP 前 N 个包，统计攻击类型 × 传输层协议/端口 的关系

用法:
    python3 _check_attack_proto_corr.py --max-pkts 5000
"""

import argparse, os, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from scapy.all import PcapReader, IP, TCP, UDP, ICMP

# CIC 2023 PCAP → 攻击标签
PCAPS = [
    ('data/data/23pcap/BenignTraffic.pcap', 0, 'Benign'),
    ('data/data/23pcap/DNS_Spoofing.pcap', 2, 'Spoofing'),
    ('data/data/23pcap/Mirai-udpplain.pcap', 4, 'Mirai'),
]

ATTACK_NAMES = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'DoS', 'DDoS', 'BruteForce']

# 常见端口→协议映射
PORT_PROTO = {
    80: 'HTTP', 443: 'HTTPS', 8080: 'HTTP',
    22: 'SSH', 23: 'Telnet', 21: 'FTP', 20: 'FTP',
    53: 'DNS', 5353: 'mDNS',
    67: 'DHCP', 68: 'DHCP',
    123: 'NTP',
    502: 'Modbus', 20000: 'DNP3', 20001: 'DNP3',
    102: 'S7Comm',
    1883: 'MQTT', 8883: 'MQTT',
}


def guess_proto_by_port(sport, dport, ip_proto):
    """基于端口推测应用层协议"""
    for p in [dport, sport]:
        if p in PORT_PROTO:
            return PORT_PROTO[p]
    # ICMP
    if ip_proto == 1:
        return 'ICMP'
    # 未知
    iproto_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
    return iproto_map.get(ip_proto, f'IP-{ip_proto}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pkts', type=int, default=5000,
                        help='每个 PCAP 读取的最大包数')
    parser.add_argument('-o', '--output', default='attack_proto_correlation')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    all_records = []

    for relpath, true_class, pcap_name in PCAPS:
        pcap_path = os.path.join(base, relpath)
        if not os.path.exists(pcap_path):
            print(f"[跳过] {pcap_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[PCAP] {pcap_name} → {ATTACK_NAMES[true_class]} (class {true_class})")

        # 流式读取
        pkts_read = 0
        ip_pkts = 0
        flow_pkts = Counter()  # flow_key → packet_count
        # 记录每个 flow 的端口/协议信息
        flow_info = {}  # flow_key → first_seen_dict

        for pkt in PcapReader(pcap_path):
            if pkts_read >= args.max_pkts:
                break
            pkts_read += 1
            if IP not in pkt:
                continue
            ip_pkts += 1
            ip = pkt[IP]
            sport, dport = 0, 0
            if TCP in pkt:
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
            elif UDP in pkt:
                sport, dport = pkt[UDP].sport, pkt[UDP].dport

            # 5元组
            flow_key = (ip.src, ip.dst, sport, dport, ip.proto)
            flow_pkts[flow_key] += 1
            if flow_key not in flow_info:
                flow_info[flow_key] = {
                    'sport': sport, 'dport': dport, 'proto': ip.proto,
                }

        n_flows = len(flow_info)
        print(f"  读取 {pkts_read} 包 ({ip_pkts} IP) → {n_flows} 条 flow")

        # 统计协议分布
        proto_dist = Counter()
        port_dist = Counter()
        for fk, info in flow_info.items():
            proto = guess_proto_by_port(info['sport'], info['dport'], info['proto'])
            proto_dist[proto] += 1
            # 目的端口分布
            port_dist[info['dport']] += 1

        print("  协议分布 (按flow数):")
        for proto, cnt in proto_dist.most_common(15):
            print(f"    {proto:12s}: {cnt:5d} ({cnt/n_flows*100:.1f}%)")
            all_records.append({
                'true_attack': ATTACK_NAMES[true_class],
                'protocol': proto,
                'flow_count': cnt,
                'flow_pct': round(cnt/n_flows*100, 1),
            })

        print("  主要目的端口:")
        for port, cnt in port_dist.most_common(10):
            known = PORT_PROTO.get(port, '?')
            print(f"    {port:6d} ({known:8s}): {cnt:5d}")

    # === 汇总：攻击 × 协议 交叉表 ===
    print(f"\n{'='*60}")
    print("攻 击  ×  协 议  交 叉 表")
    print(f"{'='*60}\n")

    # 按攻击类别聚合
    by_attack = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        by_attack[r['true_attack']][r['protocol']] += r['flow_count']

    all_attacks = sorted(by_attack.keys())
    all_protos = sorted(set(r['protocol'] for r in all_records))

    # 表头
    header = f"{'攻击类型':<14}"
    for p in all_protos:
        header += f"{p:>10}"
    header += f"{'总计':>8}"
    print(header)
    print('-' * len(header))

    for atk in all_attacks:
        total = sum(by_attack[atk].values())
        line = f"{atk:<14}"
        for p in all_protos:
            cnt = by_attack[atk][p]
            if total > 0:
                pct = cnt / total * 100
                line += f"{cnt:>4}({pct:>4.0f}%)"
            else:
                line += f"{'':>10}"
        line += f"{total:>8}"
        print(line)

    # 关键发现
    print(f"\n{'='*60}")
    print("关 键 发 现")
    print(f"{'='*60}")
    print("""
1. Benign 流量通常是混合协议 (DHCP, DNS, TCP/UDP 基础通信)
2. DNS_Spoofing 攻击应该主要使用 DNS 协议 (端口 53)
3. Mirai-udpplain 主要使用 UDP 协议 (典型 IoT 僵尸网络 UDP Flood)
""")

    # 输出保存
    out_path = os.path.join(base, f'{args.output}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'cross_table': {atk: dict(by_attack[atk]) for atk in all_attacks},
            'total_flows_by_pcap': {ATTACK_NAMES[true_class]: None for _, true_class, _ in PCAPS},
        }, f, ensure_ascii=False, indent=2)
    print(f"[保存] {out_path}")


if __name__ == '__main__':
    main()
