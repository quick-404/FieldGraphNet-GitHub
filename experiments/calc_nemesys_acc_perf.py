"""计算纯 NEMESYS BCDG 评估指标
Precision / Recall / F1 / Accuracy / Perfection
"""
import json
import os
import sys
import subprocess

DATA_DIR = os.path.expanduser('~/网络流量分析项目/nemesys-gnn4id/data/data')
TOLERANCE = 1

PCAPS = [
    ('dhcp_100',  'DHCP',  'dhcp',  'udp.port==67 or udp.port==68'),
    ('dnp3_100',  'DNP3',  'dnp3',  'tcp.port==20000'),
    ('dns_100',   'DNS',   'dns',   'udp.port==53'),
    ('modbus_100','Modbus','modbus','tcp.port==502'),
    ('ntp_100',   'NTP',   'ntp',   'udp.port==123'),
    ('s7comm_100','S7Comm','s7comm','tcp.port==102'),
]


def extract_ws_boundaries(pcap, layer, pkt_filter):
    """tshark → 每条消息的边界集合 {offset,...} 和字段列表 [(offset,length),...]"""
    cmd = ['tshark', '-r', pcap, '-Y', pkt_filter, '-c', '100', '-T', 'jsonraw']
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if not r.stdout:
        return [], []
    packets = json.loads(r.stdout)

    all_bounds, all_fields = [], []
    for pkt in packets:
        layers = pkt.get('_source', {}).get('layers', {})
        target = layers.get(layer, {})
        if not isinstance(target, dict):
            continue
        bounds = {0}
        fields = []
        max_end = [0]

        def collect(d):
            for k, v in d.items():
                if k.startswith('_') or '_tree' in k:
                    continue
                if isinstance(v, dict):
                    collect(v)
                elif isinstance(v, list) and len(v) >= 3:
                    off, ln = int(v[1]), int(v[2])
                    fields.append((off, ln))
                    bounds.add(off)
                    bounds.add(off + ln)
                    max_end[0] = max(max_end[0], off + ln)

        collect(target)
        bounds.add(max_end[0])
        all_bounds.append(bounds)
        all_fields.append(fields)
    return all_bounds, all_fields


def extract_bcdg_boundaries(segs, use_full):
    """BCDG → 每条消息的边界集合和字段列表"""
    all_bounds, all_fields = [], []
    for msg in segs:
        bounds = {0}
        fields = []
        if use_full and isinstance(msg, list):
            for ms in msg:
                off, ln = getattr(ms, 'offset', 0), getattr(ms, 'length', 0)
                fields.append((off, ln))
                bounds.add(off)
                bounds.add(off + ln)
            if msg:
                last = msg[-1]
                bounds.add(getattr(last, 'offset', 0) + getattr(last, 'length', 0))
        elif isinstance(msg, dict):
            for ft in msg.get('field_types', []):
                off, ln = ft.get('offset', 0), ft.get('length', 0)
                fields.append((off, ln))
                bounds.add(off)
                bounds.add(off + ln)
            bounds.add(msg.get('payload_size', 0))
        all_bounds.append(bounds)
        all_fields.append(fields)
    return all_bounds, all_fields


# ============================================================
if subprocess.run(['which', 'tshark'], capture_output=True).returncode != 0:
    print("[ERROR] 请安装 tshark: sudo apt install tshark"); sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer

analyzer = NEMESYSAnalyzer(sigma=0.6)
use_full = analyzer.nemere_available
print(f"模式: {'完整NEMESYS' if use_full else '简化'}\n")

# 表头
sep = "=" * 108
print(sep)
print(f"{'PCAP':<12} {'协议':<8} {'Precision':>10} {'Recall':>8} {'F1':>10} {'Accuracy':>10} {'Perfection':>12}")
print(sep)

all_tp = all_fp = all_fn = all_tn = 0
all_perf_ok = all_perf_total = 0

for name, proto, layer, pkt_filter in PCAPS:
    pcap = os.path.join(DATA_DIR, f'{name}.pcap')
    if not os.path.exists(pcap):
        print(f"{name:<12} {proto:<8}  PCAP 不存在"); continue

    ws_bounds, ws_fields = extract_ws_boundaries(pcap, layer, pkt_filter)
    if not ws_bounds:
        print(f"{name:<12} {proto:<8}  tshark 未解析"); continue

    result = analyzer.analyze(pcap)
    bcdg_bounds, bcdg_fields = extract_bcdg_boundaries(result['segments'], use_full)

    n = min(len(ws_bounds), len(bcdg_bounds))
    tp = fp = fn = tn = 0
    perf_ok = perf_total = 0

    for i in range(n):
        ws_b = ws_bounds[i]
        bcdg_b = bcdg_bounds[i]
        max_pos = max(max(ws_b), max(bcdg_b))

        # 逐字节 TP/FP/FN/TN
        for pos in range(max_pos + 1):
            w = any(abs(pos - wb) <= TOLERANCE for wb in ws_b)
            b = any(abs(pos - bb) <= TOLERANCE for bb in bcdg_b)
            if w and b:
                tp += 1
            elif b and not w:
                fp += 1
            elif w and not b:
                fn += 1
            else:
                tn += 1

        # Perfection: WS 字段 vs BCDG 字段 (offset, length) 双向匹配
        for w_off, w_len in ws_fields[i]:
            perf_total += 1
            w_end = w_off + w_len
            matched = any(
                abs(b_off - w_off) <= TOLERANCE and abs(b_off + b_len - w_end) <= TOLERANCE
                for b_off, b_len in bcdg_fields[i]
            )
            if matched:
                perf_ok += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    perfection = perf_ok / max(perf_total, 1)

    all_tp += tp; all_fp += fp; all_fn += fn; all_tn += tn
    all_perf_ok += perf_ok; all_perf_total += perf_total

    print(f"{name:<12} {proto:<8} {precision:>10.4f} {recall:>8.4f} {f1:>10.4f} {accuracy:>10.4f} {perfection:>12.4f}")

# 合计
print(sep)
tp = all_tp; fp = all_fp; fn = all_fn; tn = all_tn
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)
accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
perfection = all_perf_ok / max(all_perf_total, 1)
print(f"{'合计':<12} {'':<8} {precision:>10.4f} {recall:>8.4f} {f1:>10.4f} {accuracy:>10.4f} {perfection:>12.4f}")

print(f"\nTP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"PerfectlyInferredFields={all_perf_ok}  TrueFieldCount={all_perf_total}")
print()
print("公式:")
print("  Precision = TP / (TP + FP)")
print("  Recall    = TP / (TP + FN)")
print("  F1        = 2PR / (P+R)")
print("  Accuracy  = (TP + TN) / (TP + TN + FP + FN)")
print("  Perfection = PerfectlyInferredFields / TrueFieldCount")
