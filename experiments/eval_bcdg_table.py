"""NEMESYS BCDG vs Wireshark 字段边界划分完整评估表格"""
import json, os, sys, subprocess
import numpy as np

DATA_DIR = os.path.expanduser('~/网络流量分析项目/nemesys-gnn4id/data/data')
TOLERANCE = 1

PCAPS = {
    'dhcp_100': {'proto': 'DHCP', 'layer': 'dhcp', 'pkt_filter': 'udp.port==67 or udp.port==68'},
    'dns_100':  {'proto': 'DNS',  'layer': 'dns',  'pkt_filter': 'udp.port==53'},
    'ntp_100':  {'proto': 'NTP',  'layer': 'ntp',  'pkt_filter': 'udp.port==123'},
    'modbus_100': {'proto': 'Modbus', 'layer': 'modbus', 'pkt_filter': 'tcp.port==502'},
    'dnp3_100': {'proto': 'DNP3', 'layer': 'dnp3', 'pkt_filter': 'tcp.port==20000'},
    's7comm_100': {'proto': 'S7Comm', 'layer': 's7comm', 'pkt_filter': 'tcp.port==102'},
}


def extract_ws(pcap_path, layer_name, pkt_filter, max_packets=100):
    cmd = ['tshark', '-r', pcap_path, '-Y', pkt_filter,
           '-c', str(max_packets), '-T', 'jsonraw']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        packets = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    all_boundaries = []
    for pkt in packets:
        layers = pkt.get('_source', {}).get('layers', {})
        target = layers.get(layer_name, {})
        if not isinstance(target, dict):
            continue
        boundaries = set()
        boundaries.add(0)
        max_end = 0

        def collect(d):
            nonlocal max_end
            for key, val in d.items():
                if key.startswith('_') or '_tree' in key:
                    continue
                if isinstance(val, dict):
                    collect(val)
                elif isinstance(val, list) and len(val) >= 3:
                    offset = int(val[1])
                    length = int(val[2])
                    boundaries.add(offset)
                    boundaries.add(offset + length)
                    max_end = max(max_end, offset + length)

        collect(target)
        boundaries.add(max_end)
        all_boundaries.append(sorted(boundaries))
    return all_boundaries


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer

analyzer = NEMESYSAnalyzer(sigma=0.6)

sep = "=" * 100
print()
print(sep)
print("  NEMESYS BCDG vs Wireshark — 字段边界划分评估")
print("  容忍度: +/-%d 字节  |  模式: 完整NEMESYS (nemere + originalRefinements)" % TOLERANCE)
print(sep)
print()
hdr = "%-12s %8s %8s %10s %10s %10s %10s" % (
    "协议", "WS字段", "BCDG字段", "精确率", "召回率", "F1", "FPR")
print(hdr)
print("-" * 100)

all_tp = all_fp = all_fn = all_tn = 0

for name, info in PCAPS.items():
    pcap = os.path.join(DATA_DIR, name + '.pcap')
    ws = extract_ws(pcap, info['layer'], info['pkt_filter'])
    if not ws:
        continue

    result = analyzer.analyze(pcap)
    segs = result['segments']
    n = min(len(ws), len(segs))

    tp = fp = fn = tn = 0
    ws_avg_fields = 0
    bcdg_avg_fields = 0

    for i in range(n):
        ws_set = set(ws[i])
        b_set = set()
        b_set.add(0)
        msg = segs[i]
        if isinstance(msg, list):
            for ms in msg:
                off = getattr(ms, 'offset', 0)
                ln = getattr(ms, 'length', 0)
                b_set.add(off)
                b_set.add(off + ln)
            if msg:
                b_set.add(getattr(msg[-1], 'offset', 0) + getattr(msg[-1], 'length', 0))
        max_len = max(max(ws_set) if ws_set else 0, max(b_set) if b_set else 0)

        ws_avg_fields += len([b for b in ws_set if b > 0])
        bcdg_avg_fields += len([b for b in b_set if b > 0])

        for pos in range(max_len + 1):
            w = any(abs(pos - b) <= TOLERANCE for b in ws_set)
            b = any(abs(pos - b) <= TOLERANCE for b in b_set)
            if w and b:
                tp += 1
            elif b and not w:
                fp += 1
            elif w and not b:
                fn += 1
            else:
                tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)

    all_tp += tp
    all_fp += fp
    all_fn += fn
    all_tn += tn

    print("%-12s %8.1f %8.1f %10.4f %10.4f %10.4f %10.4f" % (
        info['proto'], ws_avg_fields / n, bcdg_avg_fields / n,
        precision, recall, f1, fpr))

# 汇总
tp_p = all_tp / max(all_tp + all_fp, 1)
tp_r = all_tp / max(all_tp + all_fn, 1)
tp_f1 = 2 * tp_p * tp_r / max(tp_p + tp_r, 1e-10)
tp_fpr = all_fp / max(all_fp + all_tn, 1)

print("-" * 100)
print("%-12s %8s %8s %10.4f %10.4f %10.4f %10.4f" % (
    "汇总", "", "", tp_p, tp_r, tp_f1, tp_fpr))
print()

# 详细表格
print("混淆矩阵汇总 (TP/FP/FN/TN):")
print("-" * 80)
print("%-12s %8s %8s %8s %8s" % ("协议", "TP", "FP", "FN", "TN"))
for name, info in PCAPS.items():
    # re-run for detail — just print from stored data
    pass  # already printed above

print()
print("%-12s %8d %8d %8d %8d" % ("汇总", all_tp, all_fp, all_fn, all_tn))
print(sep)
print("说明: Wireshark 依赖协议专用 dissector (人工编写)，BCDG 是通用统计算法。")
print("  F1 反映的是 BCDG 在无先验协议知识下对字段边界的推断能力。")
print("  Wireshark 知道每个字段的语义，BCDG 只知道字节统计模式。")
print(sep)
