"""
用 Wireshark/tshark 作为 ground truth，评估 NEMESYS BCDG 字段划分精度。
比较 BCDG 输出的字段边界 vs Wireshark dissector 的字段边界。
"""
import json, os, sys, subprocess, math
from collections import Counter, defaultdict
import numpy as np

# ===== 配置 =====
DATA_DIR = os.path.expanduser('~/网络流量分析项目/nemesys-gnn4id/data/data')
TOLERANCE = 1  # 边界偏移容忍度（±N字节内算匹配）

PCAPS = {
    'dhcp_100': {'proto': 'DHCP', 'layer': 'dhcp', 'pkt_filter': 'udp.port==67 or udp.port==68'},
    'dns_100':  {'proto': 'DNS',  'layer': 'dns',  'pkt_filter': 'udp.port==53'},
    'ntp_100':  {'proto': 'NTP',  'layer': 'ntp',  'pkt_filter': 'udp.port==123'},
    'modbus_100': {'proto': 'Modbus', 'layer': 'modbus', 'pkt_filter': 'tcp.port==502'},
    'dnp3_100': {'proto': 'DNP3', 'layer': 'dnp3', 'pkt_filter': 'tcp.port==20000'},
    's7comm_100': {'proto': 'S7Comm', 'layer': 's7comm', 'pkt_filter': 'tcp.port==102'},
}

# ===== 从 tshark jsonraw 提取 Wireshark 字段边界 =====
def extract_wireshark_boundaries(pcap_path, layer_name, pkt_filter, max_packets=100):
    """用 tshark 解析 PCAP，提取指定协议层的字段字节边界。"""
    cmd = [
        'tshark', '-r', pcap_path, '-Y', pkt_filter,
        '-c', str(max_packets), '-T', 'jsonraw'
    ]
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
        boundaries.add(0)  # 协议头起始
        max_end = 0

        def _collect_boundaries(d, prefix=''):
            nonlocal max_end
            for key, val in d.items():
                if key.startswith('_') or '_tree' in key:
                    continue
                if isinstance(val, dict):
                    _collect_boundaries(val, key + '.')
                elif isinstance(val, list) and len(val) >= 3:
                    offset = int(val[1])
                    length = int(val[2])
                    boundaries.add(offset)           # 字段起始位置
                    boundaries.add(offset + length)  # 字段结束位置
                    max_end = max(max_end, offset + length)

        _collect_boundaries(target)
        boundaries.add(max_end)  # 协议尾
        all_boundaries.append(sorted(boundaries))

    return all_boundaries


# ===== 直接运行 NEMESYS 获取 per-packet BCDG 结果 =====
def run_nemesys_bcdg(pcap_path, analyzer, max_packets=100, use_full=False):
    """用 NEMESYS 分析 PCAP，获取每条消息的 BCDG 字段分段边界。"""
    result = analyzer.analyze(pcap_path)
    result_segments = result.get('segments', [])

    all_boundaries = []
    for msg in result_segments[:max_packets]:
        boundaries = set()
        boundaries.add(0)

        if use_full and isinstance(msg, list):
            # Full NEMESYS mode: msg is a list of MessageSegment objects
            for ms in msg:
                off = getattr(ms, 'offset', 0)
                ln = getattr(ms, 'length', 0)
                boundaries.add(off)
                boundaries.add(off + ln)
            if msg:
                last = msg[-1]
                boundaries.add(getattr(last, 'offset', 0) + getattr(last, 'length', 0))
        elif isinstance(msg, dict):
            # Simulated mode: msg is a dict with field_types
            field_types = msg.get('field_types', [])
            for ft in field_types:
                off = ft.get('offset', 0)
                ln = ft.get('length', 0)
                boundaries.add(off)
                boundaries.add(off + ln)
            boundaries.add(msg.get('payload_size', 0))

        all_boundaries.append(sorted(boundaries))

    return all_boundaries


# ===== 计算指标 =====
def compute_boundary_metrics(ws_boundaries_list, bcdg_boundaries_list, tolerance=TOLERANCE):
    """
    按字节位置比较字段边界。
    对于每个字节位置，判断 Wireshark 和 BCDG 是否都认为它是字段起始点。
    """
    all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0
    per_packet = []

    for ws_bounds, bcdg_bounds in zip(ws_boundaries_list, bcdg_boundaries_list):
        # 取两个列表的最大覆盖范围
        max_len = max(max(ws_bounds) if ws_bounds else 0,
                      max(bcdg_bounds) if bcdg_bounds else 0)

        tp = fp = fn = tn = 0

        for pos in range(max_len + 1):
            ws_is_boundary = any(abs(pos - b) <= tolerance for b in ws_bounds)
            bcdg_is_boundary = any(abs(pos - b) <= tolerance for b in bcdg_bounds)

            if ws_is_boundary and bcdg_is_boundary:
                tp += 1
            elif bcdg_is_boundary and not ws_is_boundary:
                fp += 1
            elif ws_is_boundary and not bcdg_is_boundary:
                fn += 1
            else:
                tn += 1

        per_packet.append({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_tn += tn

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = all_fp / max(all_fp + all_tn, 1)

    return {
        'tp': all_tp, 'fp': all_fp, 'fn': all_fn, 'tn': all_tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr,
        'per_packet': per_packet,
    }


# ===== 主流程 =====
print("=" * 85)
print("  NEMESYS BCDG 字段划分 vs Wireshark Ground Truth")
print(f"  边界容忍度: ±{TOLERANCE} 字节")
print("=" * 85)
print()

# 初始化 NEMESYS 分析器（只初始化一次）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
nemesys_analyzer = NEMESYSAnalyzer(sigma=0.6)
use_full = nemesys_analyzer.nemere_available
mode_name = '完整NEMESYS (nemere+refinements)' if use_full else '简化BCDG (scapy降级)'
print(f"NEMESYS 模式: {mode_name}")
print()

for name, info in PCAPS.items():
    pcap_path = os.path.join(DATA_DIR, f'{name}.pcap')

    if not os.path.exists(pcap_path):
        print(f"[{name}] PCAP 不存在: {pcap_path}")
        continue

    print(f"[{name}] ({info['proto']})")

    # 1. Wireshark ground truth
    print(f"  提取 Wireshark 字段边界...")
    ws_bounds = extract_wireshark_boundaries(pcap_path, info['layer'], info['pkt_filter'])
    if not ws_bounds:
        print(f"  WARNING: tshark 未提取到 {info['layer']} 层数据")
        continue
    print(f"    解析到 {len(ws_bounds)} 条消息")

    # 2. BCDG 分段
    print(f"  运行 NEMESYS BCDG 分段...")
    bcdg_bounds = run_nemesys_bcdg(pcap_path, nemesys_analyzer,
                                   max_packets=len(ws_bounds), use_full=use_full)
    print(f"    BCDG 分段 {len(bcdg_bounds)} 条消息")

    # 对齐数量
    n = min(len(ws_bounds), len(bcdg_bounds))
    ws_bounds = ws_bounds[:n]
    bcdg_bounds = bcdg_bounds[:n]

    # 3. 计算指标
    metrics = compute_boundary_metrics(ws_bounds, bcdg_bounds)

    # 4. 额外统计
    ws_avg_fields = sum(len([b for b in bs if b > 0]) for bs in ws_bounds) / max(n, 1)
    bcdg_avg_fields = sum(len([b for b in bs if b > 0]) for bs in bcdg_bounds) / max(n, 1)

    print(f"    Wireshark 平均字段数: {ws_avg_fields:.1f}")
    print(f"    BCDG 平均字段数:     {bcdg_avg_fields:.1f}")
    print(f"    精确率: {metrics['precision']:.4f}")
    print(f"    召回率: {metrics['recall']:.4f}")
    print(f"    F1:     {metrics['f1']:.4f}")
    print(f"    FPR:    {metrics['fpr']:.4f}")
    print(f"    TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print()

print("=" * 85)
print("  注意: 此评估基于 tshark 协议解析器的字段边界作为 ground truth。")
print("  容忍度 ±1 字节意味着 BCDG 在 1 字节范围内的切分视为正确。")
print("=" * 85)
