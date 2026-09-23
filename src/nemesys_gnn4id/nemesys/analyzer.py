"""
NEMESYS 分析器：封装 BCDG 协议逆向分析

复用 NEMESYS 项目的核心逻辑 (模型/nemesys/src/):
- SpecimenLoader: 加载PCAP消息
- bcDeltaGaussMessageSegmentation: BCDG分段
- originalRefinements / charRefinements: 分段细化
- symbolsFromSegments: 提取符号
"""

import os
import sys
import math
from collections import Counter

import numpy as np

from nemesys_gnn4id.config import NEMESYS_DEFAULTS


# ============================================================
# BCDG 核心算法 (移植自 nemere/inference/analyzers.py)
# ============================================================

def _bit_congruence(byte_seq):
    """
    计算相邻字节的 bit congruence (Simple Matching 系数)。

    对于 byte[i-1] 和 byte[i]，计算 8 位中相同位的比例。
    返回长度为 len(byte_seq)-1 的数组。
    """
    if len(byte_seq) < 2:
        return np.array([])

    arr = np.array(byte_seq, dtype=np.uint8)
    # XOR 相邻字节，统计不同位的数量
    xor = arr[:-1] ^ arr[1:]
    # 计算 popcount (每个字节中 1 的数量 = 不同位的数量)
    # 8 - popcount = 相同位的数量
    # 用查表法加速
    popcount = np.array([bin(x).count('1') for x in xor], dtype=np.float64)
    congruence = (8.0 - popcount) / 8.0
    return congruence


def _bcdg_segment(payload_bytes, sigma=0.6, min_segment_len=2):
    """
    BCDG (Bit Congruence Delta Gauss) 分段算法。

    1. 计算相邻字节的 bit congruence
    2. 计算 congruence 的 delta（一阶差分）
    3. 用高斯滤波平滑 delta 信号
    4. 取平滑信号的局部极大值作为字段边界

    注意（2026-09-22 修正）：本实现只取局部极大值，**不与前导局部极小值配对**。
    NEMESYS 原文描述的是 min-max 上升对规则，本实现是它的简化版。因此论文中不得
    把本函数描述为"复现了 NEMESYS 的 BCDG 分段算法"，只能说是采用了 BCDG 信号。

    Args:
        payload_bytes: bytes 或 int list，协议消息的原始字节
        sigma: 高斯滤波的 sigma 参数
        min_segment_len: 最小段长度

    Returns:
        list of (offset, length) tuples，每个 segment 的位置和长度
    """
    if len(payload_bytes) < 4:
        return [(0, len(payload_bytes))]

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        # Fallback: simple equal-size segmentation
        seg_len = max(min_segment_len, len(payload_bytes) // 4)
        segments = []
        for i in range(0, len(payload_bytes), seg_len):
            seg_end = min(i + seg_len, len(payload_bytes))
            segments.append((i, seg_end - i))
        return segments

    # Step 1: Bit congruence
    congruence = _bit_congruence(payload_bytes)
    if len(congruence) < 2:
        return [(0, len(payload_bytes))]

    # Step 2: Delta (first difference)
    delta = np.diff(congruence)

    # Step 3: Gaussian smoothing (skip if sigma=0 to avoid ZeroDivisionError)
    if sigma > 0:
        smoothed = gaussian_filter1d(delta, sigma=sigma)
    else:
        smoothed = delta

    # Step 4: Find inflection points (min-max rising pairs)
    # Look for local min followed by local max in the smoothed signal
    segments = []
    last_cut = 0

    # Find positions where smoothed signal crosses from negative to positive
    # (indicating a field boundary)
    for i in range(1, len(smoothed) - 1):
        # 平滑信号的局部极大值即为字段边界。
        # 注意（2026-09-22 修正）：此处**只判局部极大值**，原注释声称"check if there
        # was a preceding local min"及 docstring 的"min-max 上升对"均未实现。NEMESYS
        # 原文的 min-max 配对规则在这里被简化掉了，描述本函数时不得声称已复现该规则。
        # 切点取 i+1。
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            cut_pos = i + 1
            if cut_pos - last_cut >= min_segment_len:
                segments.append((last_cut, cut_pos - last_cut))
                last_cut = cut_pos

    # Add the final segment
    if len(payload_bytes) - last_cut > 0:
        segments.append((last_cut, len(payload_bytes) - last_cut))

    # Filter out segments that are too short
    segments = [(off, ln) for off, ln in segments if ln >= min_segment_len]

    if not segments:
        segments = [(0, len(payload_bytes))]

    return segments


def _classify_segment_type(payload_bytes, offset, length):
    """
    根据 segment 的字节内容推断字段类型。

    Returns:
        str: 'text', 'binary', 'zero', 'fixed'
    """
    segment = payload_bytes[offset:offset + length]
    if not segment:
        return 'empty'

    # Check if all zeros
    if all(b == 0 for b in segment):
        return 'zero'

    # Check if printable ASCII text
    printable_count = sum(1 for b in segment if 32 <= b <= 126 or b in (9, 10, 13))
    if printable_count > length * 0.8:
        return 'text'

    # Check if fixed pattern (same byte repeated)
    if len(set(segment)) == 1:
        return 'fixed'

    return 'binary'


# Well-known port → protocol mapping
PORT_PROTOCOL_MAP = {
    80: 'HTTP', 443: 'TLS', 8080: 'HTTP',
    53: 'DNS', 5353: 'mDNS',
    21: 'FTP', 20: 'FTP-data', 22: 'SSH', 23: 'Telnet',
    25: 'SMTP', 110: 'POP3', 143: 'IMAP',
    3306: 'MySQL', 5432: 'PostgreSQL', 6379: 'Redis',
    1883: 'MQTT', 8883: 'MQTTS',
    123: 'NTP', 161: 'SNMP', 162: 'SNMP-trap',
    3389: 'RDP', 5900: 'VNC',
    67: 'DHCP', 68: 'DHCP',
    502: 'Modbus', 20000: 'DNP3', 102: 'S7Comm', 44818: 'EtherNet/IP',
}


def _identify_protocol_from_payload(payload_hex, sport, dport):
    """Identify protocol from port numbers and payload patterns.

    Returns:
        str: protocol name, or 'UNKNOWN_PROTO' for unrecognized protocols.
             Returns 'EMPTY' for packets with no payload.
    """
    # Port-based identification first
    for port in [sport, dport]:
        if port in PORT_PROTOCOL_MAP:
            return PORT_PROTOCOL_MAP[port]

    # Payload-based identification
    if not payload_hex or payload_hex == '00':
        return 'EMPTY'

    try:
        raw = bytes.fromhex(payload_hex[:60])  # first 30 bytes for matching
    except (ValueError, TypeError):
        return 'UNKNOWN_PROTO'

    # TLS Client Hello / Server Hello
    if len(raw) >= 3 and raw[0] == 0x16 and raw[1] == 0x03:
        return 'TLS'

    # HTTP methods
    if any(raw[:4] == m for m in _HTTP_METHODS):
        return 'HTTP'

    # DNS (QR bit + opcode in first 2 bytes, usually 0x00xx or 0x80xx)
    if len(raw) >= 4 and raw[2] == 0x00 and raw[3] == 0x00:
        if sport == 53 or dport == 53:
            return 'DNS'

    # SSH banner
    if raw[:4] == b'SSH-':
        return 'SSH'

    # SOCKS5: version=0x05, nmethods (0-255)
    if len(raw) >= 3 and raw[0] == 0x05 and raw[1] <= 0xFF:
        if sport == 1080 or dport == 1080:
            return 'SOCKS5'

    # SOCKS4: version=0x04, command=0x01(CONNECT) or 0x02(BIND)
    if len(raw) >= 3 and raw[0] == 0x04 and raw[1] in (0x01, 0x02):
        return 'SOCKS4'

    # VNC RFB protocol banner
    if raw[:4] == b'RFB ':
        return 'VNC-RFB'

    # FTP greeting (220 response on port 21) — check before SMTP since both use '220 '
    if len(raw) >= 4 and raw[:4] == b'220 ' and (sport == 21 or dport == 21):
        return 'FTP'

    # SMTP greeting / commands (EHLO/HELO are SMTP-specific; 220 on non-FTP ports)
    if len(raw) >= 4 and (raw[:4] == b'EHLO' or raw[:4] == b'HELO'):
        return 'SMTP'
    if len(raw) >= 4 and raw[:4] == b'220 ' and (sport == 25 or dport == 25 or sport == 587 or dport == 587):
        return 'SMTP'

    # POP3 response
    if len(raw) >= 4 and raw[:4] == b'+OK ':
        return 'POP3'
    if len(raw) >= 5 and raw[:5] == b'-ERR ':
        return 'POP3'

    # IMAP response
    if len(raw) >= 4 and raw[:4] == b'* OK':
        return 'IMAP'
    if len(raw) >= 5 and raw[:5] == b'* BYE':
        return 'IMAP'

    # Redis protocol (RESP): starts with *, +, -, :, $
    if len(raw) >= 2 and raw[0:1] in (b'*', b'+', b'-', b':', b'$'):
        if sport == 6379 or dport == 6379:
            return 'Redis'

    # MQTT: first byte high 4 bits = 0001 (CONNECT), 0010 (CONNACK), etc.
    if len(raw) >= 2:
        first_nibble = (raw[0] & 0xF0) >> 4
        if first_nibble in (0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                            0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE):
            if sport == 1883 or dport == 1883 or sport == 8883 or dport == 8883:
                return 'MQTT'

    # Unrecognized protocol — return UNKNOWN_PROTO instead of BINARY
    return 'UNKNOWN_PROTO'


def _shannon_entropy(byte_seq):
    """计算字节序列的 Shannon 熵 (bits)"""
    if not byte_seq:
        return 0.0
    counter = Counter(byte_seq)
    total = len(byte_seq)
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


# 模块级常量：HTTP 方法前缀（热路径中每包调用，避免重复创建）
_HTTP_METHODS = [b'GET ', b'POST', b'PUT ', b'DELE', b'HEAD', b'HTTP', b'OPTI', b'PATC']


def _analyze_payload_patterns(payload_hex_list):
    """Analyze payload byte patterns for protocol fingerprinting."""
    if not payload_hex_list:
        return {'entropy': 0, 'avg_size': 0, 'unique_ratio': 0}

    sizes = []
    all_bytes = []
    for ph in payload_hex_list:
        if ph and ph != '00':
            try:
                raw = bytes.fromhex(ph)
                sizes.append(len(raw))
                all_bytes.extend(raw)
            except (ValueError, TypeError):
                pass

    if not sizes:
        return {'entropy': 0, 'avg_size': 0, 'unique_ratio': 0}

    entropy = _shannon_entropy(all_bytes)

    return {
        'entropy': round(entropy, 3),
        'avg_size': round(sum(sizes) / len(sizes), 1),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'payload_count': len(sizes),
        'total_bytes': sum(sizes),
        'unique_bytes': len(set(all_bytes)) if all_bytes else 0,
    }


class NEMESYSAnalyzer:
    """
    NEMESYS 协议逆向分析器

    对PCAP文件做BCDG分段，推断未知协议的消息结构。
    支持两种模式：
    1. 完整模式：nemere/netzob 可用，做真实BCDG分析
    2. 模拟模式：依赖不可用，基于端口和payload做协议识别
    """

    def __init__(self, sigma=None, layer=None, relative_to_ip=None):
        self.sigma = sigma if sigma is not None else NEMESYS_DEFAULTS['sigma']
        self.layer = layer if layer is not None else NEMESYS_DEFAULTS['layer']
        self.relative_to_ip = relative_to_ip if relative_to_ip is not None else NEMESYS_DEFAULTS['relative_to_ip']

        self.nemere_available = False
        self._check_dependencies()

    def _check_dependencies(self):
        """检查 nemere/netzob 是否可用"""
        try:
            nemesys_src = os.path.expanduser('~/nemesys/src')
            if nemesys_src not in sys.path:
                sys.path.insert(0, nemesys_src)
            from nemere.utils.loader import SpecimenLoader
            from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation
            self.nemere_available = True
            print("[NEMESYS] nemere 已加载，使用完整BCDG分析")
        except ImportError:
            print("[NEMESYS] nemere 未安装，使用内置 BCDG 分析模式")

    def segment(self, pcap_path, target_flows=None):
        """
        对PCAP做BCDG分段

        Args:
            pcap_path: PCAP文件路径
            target_flows: list of (src_ip, dst_ip, src_port, dst_port, protocol) tuples.
                         If provided, only analyze packets matching these flows.

        Returns:
            list: segmentsPerMsg - 每条消息的分段结果
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP文件不存在: {pcap_path}")

        if self.nemere_available:
            return self._segment_full(pcap_path, target_flows)
        else:
            return self._segment_builtin(pcap_path, target_flows)

    def _segment_full(self, pcap_path, target_flows=None):
        """完整BCDG分段 + 细化（需要nemere）"""
        from nemere.utils.loader import SpecimenLoader
        from nemere.inference.segmentHandler import (bcDeltaGaussMessageSegmentation,
                                                      originalRefinements)

        print(f"[NEMESYS] 加载PCAP: {pcap_path}")
        specimens = SpecimenLoader(
            pcap_path,
            layer=self.layer,
            relativeToIP=self.relative_to_ip
        )

        print(f"[NEMESYS] BCDG分段 (sigma={self.sigma})...")
        segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, self.sigma)
        print(f"[NEMESYS] 分段完成: {len(segmentsPerMsg)} 条消息")

        # 细化分段（PCA 边界精化）
        print(f"[NEMESYS] 细化分段...")
        refinedPerMsg = originalRefinements(segmentsPerMsg)
        print(f"[NEMESYS] 细化完成: {len(refinedPerMsg)} 条消息")

        # 如果有 target_flows，过滤结果
        if target_flows:
            refinedPerMsg = self._filter_by_flows(refinedPerMsg, target_flows)
            print(f"[NEMESYS] 过滤后: {len(refinedPerMsg)} 条消息匹配目标flow")

        return refinedPerMsg

    @staticmethod
    def _filter_by_flows(segmentsPerMsg, target_flows):
        """按 target_flows 过滤 BCDG 分段结果。"""
        flow_set = set()
        for fl in target_flows:
            src_ip, dst_ip, sport, dport, proto = fl[:5]
            flow_set.add((src_ip, dst_ip, int(sport), int(dport), int(proto)))
            flow_set.add((dst_ip, src_ip, int(dport), int(sport), int(proto)))

        filtered = []
        for msg in segmentsPerMsg:
            src = str(getattr(msg, 'src', ''))
            dst = str(getattr(msg, 'dst', ''))
            sport = int(getattr(msg, 'sport', 0))
            dport = int(getattr(msg, 'dport', 0))
            proto = int(getattr(msg, 'proto', 0))
            if (src, dst, sport, dport, proto) in flow_set:
                filtered.append(msg)
        return filtered

    @staticmethod
    def _merge_text_segments(field_types, raw_payload):
        """合并相邻的纯 ASCII 可打印字符段为 text 类型字段。

        nemere 的 originalRefinements 会对 MessageSegment 列表做类似的
        字符序列合并，这里对标该功能的简化实现。
        """
        if not field_types:
            return field_types
        merged = []
        i = 0
        while i < len(field_types):
            ft = field_types[i]
            off, ln = ft['offset'], ft['length']
            chunk = raw_payload[off:off + ln] if isinstance(raw_payload, bytes) else b''
            # 判断当前段是否全为可打印 ASCII
            if chunk and all(32 <= b < 127 for b in chunk):
                total_off = off
                total_ln = ln
                i += 1
                # 向后合并连续的可打印段
                while i < len(field_types):
                    nxt = field_types[i]
                    noff, nln = nxt['offset'], nxt['length']
                    nchunk = raw_payload[noff:noff + nln] if isinstance(raw_payload, bytes) else b''
                    if nchunk and all(32 <= b < 127 for b in nchunk):
                        total_ln += nln
                        i += 1
                    else:
                        break
                merged.append({
                    'offset': total_off, 'length': total_ln,
                    'type': 'text', 'bytes': raw_payload[total_off:total_off + total_ln],
                })
            else:
                merged.append(ft)
                i += 1
        return merged

    def _segment_builtin(self, pcap_path, target_flows=None):
        """内置 BCDG 分段 — 使用本地 _bcdg_segment（无需 nemere）"""
        print(f"[NEMESYS] 内置 BCDG 分析 (sigma={self.sigma}): {pcap_path}")

        try:
            from scapy.all import rdpcap, IP, TCP, UDP
        except ImportError:
            print("[NEMESYS] scapy 不可用，返回空结果")
            return []

        packets = rdpcap(pcap_path)
        print(f"[NEMESYS] 共 {len(packets)} 个数据包")

        # Build flow lookup set for fast filtering
        flow_set = None
        if target_flows:
            flow_set = set()
            for fl in target_flows:
                src_ip, dst_ip, sport, dport, proto = fl[:5]
                flow_set.add((src_ip, dst_ip, int(sport), int(dport), int(proto)))
                # Also add reverse direction
                flow_set.add((dst_ip, src_ip, int(dport), int(sport), int(proto)))

        segments = []
        matched_count = 0

        for pkt in packets:
            if IP not in pkt:
                continue

            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            sport = dport = 0

            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport

            # Filter by target flows if specified
            if flow_set is not None:
                key = (src_ip, dst_ip, int(sport), int(dport), int(proto))
                if key not in flow_set:
                    continue
                matched_count += 1

            # Extract payload
            raw_payload = b''
            if TCP in pkt and pkt[TCP].payload:
                raw_payload = bytes(pkt[TCP].payload)
            elif UDP in pkt and pkt[UDP].payload:
                raw_payload = bytes(pkt[UDP].payload)
            payload_hex = raw_payload.hex() if raw_payload else '00'

            # Protocol identification
            protocol = _identify_protocol_from_payload(payload_hex, sport, dport)

            # BCDG 分段分析 payload
            bcdg_segments = []
            field_types = []
            if len(raw_payload) >= 4:
                bcdg_segments = _bcdg_segment(list(raw_payload), sigma=self.sigma)
                for off, ln in bcdg_segments:
                    ft = _classify_segment_type(raw_payload, off, ln)
                    field_types.append({
                        'offset': off,
                        'length': ln,
                        'type': ft,
                        'bytes': raw_payload[off:off + ln],
                    })
                # 合并相邻可打印字符段（对标 nemere originalRefinements）
                field_types = self._merge_text_segments(field_types, raw_payload)

            seg = {
                'src': src_ip,
                'dst': dst_ip,
                'sport': sport,
                'dport': dport,
                'proto': proto,
                'protocol': protocol,
                'len': len(pkt),
                'payload_hex': payload_hex,
                'payload_size': len(raw_payload),
                'has_payload': len(raw_payload) > 0,
                'bcdg_segments': len(bcdg_segments),
                'field_types': field_types,
            }
            if TCP in pkt:
                seg['tcp_flags'] = str(pkt[TCP].flags)
            segments.append(seg)

        if flow_set is not None:
            print(f"[NEMESYS] 匹配攻击flow的数据包: {matched_count}/{len(packets)}")

        print(f"[NEMESYS] 分析完成: {len(segments)} 个数据包")
        return segments

    def refine(self, segments):
        """细化分段结果（full BCDG模式已在_segment_full中完成，此处跳过）"""
        if self.nemere_available and segments and not isinstance(segments[0], dict):
            return segments  # already refined in _segment_full
        else:
            return segments

    def _refine_full(self, segmentsPerMsg):
        """完整细化（需要nemere）"""
        from nemere.inference.segmentHandler import originalRefinements
        print("[NEMESYS] 细化分段...")
        refinedPerMsg = originalRefinements(segmentsPerMsg)
        print(f"[NEMESYS] 细化完成: {len(refinedPerMsg)} 条消息")
        return refinedPerMsg

    def get_symbols(self, refined_segments):
        """从细化分段中提取符号"""
        if self.nemere_available and refined_segments and not isinstance(refined_segments[0], dict):
            from nemere.inference.segmentHandler import symbolsFromSegments
            return symbolsFromSegments(refined_segments)
        return refined_segments

    def summary(self, segments_or_symbols):
        """
        生成分段/符号的统计摘要

        Returns:
            dict: 统计信息，包含协议分布、payload分析等
        """
        if not segments_or_symbols:
            return {'total_messages': 0, 'avg_fields': 0, 'details': []}

        # Fingerprint mode (dict list)
        if isinstance(segments_or_symbols[0], dict):
            total = len(segments_or_symbols)

            # Protocol distribution
            protocols = Counter()
            payload_sizes = []
            payloads_by_proto = {}
            field_type_counts = Counter()
            bcdg_seg_counts = []

            for seg in segments_or_symbols:
                proto = seg.get('protocol', seg.get('proto', 'unknown'))
                protocols[proto] += 1

                ps = seg.get('payload_size', 0)
                if ps > 0:
                    payload_sizes.append(ps)

                # Collect payloads per protocol for pattern analysis
                ph = seg.get('payload_hex', '')
                if ph and ph != '00':
                    payloads_by_proto.setdefault(proto, []).append(ph)

                # BCDG segmentation stats
                n_segs = seg.get('bcdg_segments', 0)
                if n_segs > 0:
                    bcdg_seg_counts.append(n_segs)

                # Field type distribution
                for ft in seg.get('field_types', []):
                    field_type_counts[ft.get('type', 'unknown')] += 1

            # Per-protocol payload analysis
            proto_analysis = {}
            for proto, payloads in payloads_by_proto.items():
                proto_analysis[proto] = _analyze_payload_patterns(payloads)

            # Count unknown protocol messages
            from nemesys_gnn4id.config import UNKNOWN_PROTO_CONFIG
            known_set = UNKNOWN_PROTO_CONFIG['known_normal_protocols']
            unknown_count = sum(c for p, c in protocols.items()
                                if p not in known_set and p not in ('EMPTY', 'unknown'))

            return {
                'total_messages': total,
                'avg_fields': round(sum(bcdg_seg_counts) / max(len(bcdg_seg_counts), 1), 2),
                'protocol_distribution': dict(protocols.most_common()),
                'payload_analysis': proto_analysis,
                'field_type_distribution': dict(field_type_counts.most_common()),
                'total_payload_bytes': sum(payload_sizes),
                'avg_payload_size': round(sum(payload_sizes) / max(len(payload_sizes), 1), 1),
                'unknown_protocol_count': unknown_count,
                'mode': 'builtin_bcdg',
            }

        # Full BCDG mode (Symbol list)
        try:
            total = len(segments_or_symbols)
            field_counts = []
            for sym in segments_or_symbols:
                if hasattr(sym, 'segments'):
                    field_counts.append(len(sym.segments))
                elif hasattr(sym, '__len__'):
                    field_counts.append(len(sym))

            avg_fields = sum(field_counts) / len(field_counts) if field_counts else 0

            return {
                'total_messages': total,
                'avg_fields': round(avg_fields, 2),
                'min_fields': min(field_counts) if field_counts else 0,
                'max_fields': max(field_counts) if field_counts else 0,
                'mode': 'full_bcdg',
            }
        except Exception as e:
            return {
                'total_messages': len(segments_or_symbols),
                'avg_fields': 0,
                'error': str(e),
                'mode': 'error',
            }

    def analyze(self, pcap_path, target_flows=None):
        """
        一站式分析：PCAP → 分段 → 细化 → 符号 → 摘要

        Args:
            pcap_path: PCAP文件路径
            target_flows: list of (src_ip, dst_ip, src_port, dst_port, protocol) tuples

        Returns:
            dict: 包含各阶段结果的字典
        """
        segments = self.segment(pcap_path, target_flows=target_flows)
        refined = self.refine(segments)
        symbols = self.get_symbols(refined)
        summary = self.summary(symbols if symbols else segments)

        return {
            'segments': segments,
            'refined': refined,
            'symbols': symbols,
            'summary': summary,
        }


# ============================================================
# 未知协议攻击检测器
# ============================================================

class UnknownProtocolDetector:
    """
    未知协议攻击检测器

    从 NEMESYS 分段结果中识别使用未知协议的通信，
    基于多维指标（端口、熵、字段结构、通信模式）判定攻击风险，
    并对相似的未知协议进行聚类分析。
    """

    def __init__(self, config=None):
        from nemesys_gnn4id.config import UNKNOWN_PROTO_CONFIG
        self.config = config or UNKNOWN_PROTO_CONFIG
        self.known_protocols = self.config['known_normal_protocols']
        self.suspicious_ports = self.config['suspicious_ports']
        self.risk_thresholds = self.config['risk_thresholds']

    def detect_unknown_protocol_attacks(self, segments):
        """
        从 NEMESYS 分段结果中检测未知协议攻击。

        Args:
            segments: NEMESYS 分段结果列表 (每项为 dict，含 protocol/payload 等字段)

        Returns:
            dict: {
                'unknown_protocol_flows': [...],
                'total_unknown': int,
                'attack_indicators': {...},
                'protocol_clusters': [...],
                'risk_assessment': {...},
            }
        """
        _empty = {
            'unknown_protocol_flows': [],
            'total_unknown': 0,
            'attack_indicators': {},
            'protocol_clusters': [],
            'risk_assessment': {'level': 'none', 'score': 0},
        }

        if not segments:
            return _empty

        unknown_segs = []
        for seg in segments:
            proto = seg.get('protocol', seg.get('proto', 'unknown'))
            if proto not in self.known_protocols and proto not in ('EMPTY', 'unknown'):
                unknown_segs.append(seg)

        if not unknown_segs:
            return _empty

        # Step 2: 对每个未知协议流做攻击指标分析
        analyzed_flows = []
        indicator_counts = Counter()

        for seg in unknown_segs:
            indicators = self._analyze_flow_indicators(seg)
            risk_score = sum(1 for v in indicators.values() if v)
            risk_level = self._score_to_risk(risk_score)

            analyzed_flows.append({
                'src': seg.get('src', ''),
                'dst': seg.get('dst', ''),
                'sport': seg.get('sport', 0),
                'dport': seg.get('dport', 0),
                'protocol': seg.get('protocol', 'UNKNOWN_PROTO'),
                'payload_hex': seg.get('payload_hex', ''),
                'payload_size': seg.get('payload_size', 0),
                'entropy': self._compute_entropy(seg.get('payload_hex', '')),
                'bcdg_segments': seg.get('bcdg_segments', 0),
                'field_types': seg.get('field_types', []),
                'indicators': indicators,
                'risk_score': risk_score,
                'risk_level': risk_level,
            })

            for k, v in indicators.items():
                if v:
                    indicator_counts[k] += 1

        # Step 3: 对未知协议进行结构聚类
        clusters = self._cluster_protocols(analyzed_flows)

        # Step 4: 整体风险评估
        risk_assessment = self._assess_overall_risk(analyzed_flows, clusters)

        return {
            'unknown_protocol_flows': analyzed_flows,
            'total_unknown': len(analyzed_flows),
            'attack_indicators': dict(indicator_counts),
            'protocol_clusters': clusters,
            'risk_assessment': risk_assessment,
        }

    def _analyze_flow_indicators(self, seg):
        """分析单条流的攻击指标。

        Returns:
            dict: {indicator_name: bool}
        """
        indicators = {}

        # 1. 端口异常：使用已知可疑端口
        sport = seg.get('sport', 0)
        dport = seg.get('dport', 0)
        indicators['suspicious_port'] = (
            sport in self.suspicious_ports or dport in self.suspicious_ports
        )

        # 2. 非标准端口 (既不在 PORT_PROTOCOL_MAP 也不在 suspicious_ports 中)
        indicators['unusual_port'] = (
            sport not in PORT_PROTOCOL_MAP and dport not in PORT_PROTOCOL_MAP
            and sport not in self.suspicious_ports and dport not in self.suspicious_ports
        )

        # 3. Payload 高熵 (加密/混淆数据)
        payload_hex = seg.get('payload_hex', '')
        entropy = self._compute_entropy(payload_hex)
        indicators['high_entropy'] = entropy >= self.config['high_entropy_threshold']

        # 4. BCDG 字段结构异常
        field_types = seg.get('field_types', [])
        if field_types:
            binary_count = sum(1 for ft in field_types if ft.get('type') == 'binary')
            binary_ratio = binary_count / max(len(field_types), 1)
            indicators['high_binary_ratio'] = binary_ratio >= self.config['attack_indicators']['high_binary_ratio']
        else:
            indicators['high_binary_ratio'] = False

        # 5. Payload 大小异常 (超大或超小且非空)
        payload_size = seg.get('payload_size', 0)
        max_size = self.config['attack_indicators'].get('anomalous_payload_max', 1400)
        min_size = self.config['attack_indicators'].get('anomalous_payload_min', 4)
        indicators['anomalous_payload_size'] = payload_size > max_size or (0 < payload_size < min_size)

        # 6. TCP 标志异常 (NULL scan, XMAS scan 等)
        tcp_flags = seg.get('tcp_flags', '')
        if tcp_flags:
            indicators['anomalous_tcp_flags'] = self._check_anomalous_flags(tcp_flags)
        else:
            indicators['anomalous_tcp_flags'] = False

        return indicators

    def _compute_entropy(self, payload_hex):
        """计算 payload 的 Shannon 熵"""
        if not payload_hex or payload_hex == '00':
            return 0.0
        try:
            raw = bytes.fromhex(payload_hex)
        except (ValueError, TypeError):
            return 0.0
        return round(_shannon_entropy(raw), 3)

    def _check_anomalous_flags(self, flags_str):
        """检查异常 TCP 标志组合"""
        # NULL scan: no flags set
        if flags_str in ('', '0', 'None'):
            return True
        # XMAS scan: FIN+PSH+URG
        if 'F' in flags_str and 'P' in flags_str and 'U' in flags_str:
            return True
        # SYN+FIN: invalid combination
        if 'S' in flags_str and 'F' in flags_str:
            return True
        return False

    def _score_to_risk(self, score):
        """将攻击指标命中数映射到风险等级"""
        if score >= self.risk_thresholds['high']:
            return 'high'
        elif score >= self.risk_thresholds['medium']:
            return 'medium'
        elif score >= self.risk_thresholds['low']:
            return 'low'
        return 'none'

    def _cluster_protocols(self, analyzed_flows):
        """对未知协议进行结构聚类。

        按以下特征将相似的协议流分组：
        - payload 首字节模式
        - BCDG 分段数 (±1)
        - payload 大小范围
        - 字段类型序列

        Returns:
            list of dict: 每个聚类的描述
        """
        if not analyzed_flows:
            return []

        # Build feature vectors for clustering
        # Use a simple grouping by: (segment_count_bucket, size_bucket, dominant_field_type)
        groups = {}
        for flow in analyzed_flows:
            n_segs = flow.get('bcdg_segments', 0)
            seg_bucket = min(n_segs, 10)  # cap at 10

            payload_size = flow.get('payload_size', 0)
            if payload_size == 0:
                size_bucket = 0
            elif payload_size < 32:
                size_bucket = 1
            elif payload_size < 256:
                size_bucket = 2
            elif payload_size < 1024:
                size_bucket = 3
            else:
                size_bucket = 4

            # Dominant field type
            field_types = flow.get('field_types', [])
            if field_types:
                type_counts = Counter(ft.get('type', 'unknown') for ft in field_types)
                dominant = type_counts.most_common(1)[0][0]
            else:
                dominant = 'none'

            key = (seg_bucket, size_bucket, dominant)
            groups.setdefault(key, []).append(flow)

        # Build cluster descriptions
        clusters = []
        for idx, (key, flows) in enumerate(groups.items()):
            seg_bucket, size_bucket, dominant = key

            # Compute cluster-level stats
            entropies = []
            sizes = []
            ports = set()
            for f in flows:
                entropies.append(f.get('entropy', 0))
                sizes.append(f.get('payload_size', 0))
                ports.add(f.get('sport', 0))
                ports.add(f.get('dport', 0))

            avg_entropy = round(sum(entropies) / max(len(entropies), 1), 2)
            avg_size = round(sum(sizes) / max(len(sizes), 1), 1)

            # Determine cluster risk
            high_risk_count = sum(1 for f in flows if f.get('risk_level') in ('high', 'medium'))
            cluster_risk = 'high' if high_risk_count > len(flows) * 0.5 else (
                'medium' if high_risk_count > 0 else 'low'
            )

            # Generate human-readable description
            desc = self._describe_cluster(seg_bucket, size_bucket, dominant, avg_entropy, avg_size)

            clusters.append({
                'cluster_id': idx + 1,
                'flow_count': len(flows),
                'avg_entropy': avg_entropy,
                'avg_payload_size': avg_size,
                'dominant_field_type': dominant,
                'segment_count_range': seg_bucket,
                'ports': sorted(ports),
                'risk_level': cluster_risk,
                'description': desc,
                'sample_flows': [
                    {'src': f['src'], 'dst': f['dst'], 'sport': f['sport'], 'dport': f['dport']}
                    for f in flows[:5]
                ],
                'flow_indices': [f.get('flow_index', i) for i, f in enumerate(flows)],
            })

        # Sort by risk level (high first)
        risk_order = {'high': 0, 'medium': 1, 'low': 2, 'none': 3}
        clusters.sort(key=lambda c: risk_order.get(c['risk_level'], 3))

        return clusters

    def _describe_cluster(self, seg_bucket, size_bucket, dominant, avg_entropy, avg_size):
        """生成聚类的人类可读描述"""
        size_desc = {0: '无载荷', 1: '极小载荷', 2: '小型载荷', 3: '中等载荷', 4: '大型载荷'}
        entropy_desc = '高熵(加密/混淆)' if avg_entropy >= 7.0 else (
            '中熵' if avg_entropy >= 4.0 else '低熵(明文/固定)')

        parts = []
        parts.append(size_desc.get(size_bucket, '未知大小'))
        parts.append(f'平均{avg_size}bytes')
        parts.append(entropy_desc)
        if dominant != 'none':
            parts.append(f'主要字段类型:{dominant}')
        parts.append(f'BCDG分段数~{seg_bucket}')

        return ', '.join(parts)

    def _assess_overall_risk(self, analyzed_flows, clusters):
        """评估整体风险等级"""
        if not analyzed_flows:
            return {'level': 'none', 'score': 0, 'description': '无未知协议流量'}

        high_count = sum(1 for f in analyzed_flows if f['risk_level'] == 'high')
        medium_count = sum(1 for f in analyzed_flows if f['risk_level'] == 'medium')
        total = len(analyzed_flows)

        # Weighted risk score
        score = high_count * 3 + medium_count * 2

        if high_count > total * 0.3 or score >= 10:
            level = 'high'
            desc = f'发现 {high_count} 条高风险未知协议流，可能存在C2通信或数据外泄'
        elif medium_count > 0 or high_count > 0:
            level = 'medium'
            desc = f'发现 {medium_count + high_count} 条可疑未知协议流，建议进一步分析'
        else:
            level = 'low'
            desc = '未知协议流量风险较低，多为常规非标准协议通信'

        # Check for suspicious port usage
        suspicious_port_flows = [f for f in analyzed_flows if f['indicators'].get('suspicious_port')]
        if suspicious_port_flows:
            ports = set()
            for f in suspicious_port_flows:
                ports.add(f['sport'])
                ports.add(f['dport'])
            desc += f'；涉及可疑端口: {sorted(ports)}'

        return {
            'level': level,
            'score': score,
            'total_unknown_flows': total,
            'high_risk_flows': high_count,
            'medium_risk_flows': medium_count,
            'description': desc,
        }
