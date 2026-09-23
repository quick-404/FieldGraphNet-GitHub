"""
协议结构指纹提取 + 无监督聚类 + 离群检测

核心思路：
    不依赖"这是什么协议"的先验知识，仅基于 NEMESYS BCDG 输出的
    字段结构特征（字段数、大小分布、类型分布、熵值、字节模式等）
    对所有消息做相似度聚类。同一协议的流天然聚在一起，
    簇内偏离中心超过阈值的标记为可疑。

维度设计（24 维）：
    字段统计:     num_fields, mean_size, std_size, min_size, max_size      [0-4]
    字段类型分布:  ratio_binary, text, fixed, zero, empty                  [5-9]
    字节级特征:    entropy, unique_byte_ratio, byte[0-3]                  [10-15]
    传输特征:      sport, dport, protocol                                  [16-18]
    字段大小直方图: [1B], [2B], [3-4B], [5-8B], [9+B]的比例              [19-23]
"""

import math
import numpy as np
from collections import Counter


FINGERPRINT_DIM = 24

# 聚类相似度阈值（余弦相似度 >= 此值即归入同一簇）
# 聚类距离阈值（欧氏距离 <= 此值即归入同一簇，归一化向量范围 [0,~3]）
DEFAULT_DISTANCE_THRESHOLD = 0.50

# 离群检测阈值（到簇中心的标准差倍数）
DEFAULT_OUTLIER_SIGMA = 2.5

# 最小簇大小（小于此值的簇不参与离群检测，因为样本太少）
MIN_CLUSTER_SIZE_FOR_OUTLIER = 3


# ============================================================
# 指纹提取
# ============================================================

def _ensure_dict(seg):
    """Convert MessageSegment object to dict if needed."""
    if isinstance(seg, dict):
        return seg
    # Handle raw MessageSegment / list-like object
    if isinstance(seg, (list, tuple)) and seg:
        field_types = []
        payload_bytes = b''
        for ms in seg:
            fb = getattr(ms, 'bytes', b'')
            off = getattr(ms, 'offset', len(payload_bytes))
            ln = getattr(ms, 'length', len(fb))
            field_types.append({
                'offset': off,
                'length': ln,
                'type': 'binary',
                'bytes': fb if isinstance(fb, bytes) else b'',
            })
            payload_bytes += fb if isinstance(fb, bytes) else b''

        src = dst = ''
        sport = dport = proto = 0
        first = seg[0]
        msg_obj = getattr(first, 'message', None)
        if msg_obj is not None:
            src = str(getattr(msg_obj, 'source', ''))
            dst = str(getattr(msg_obj, 'destination', ''))
            l3 = getattr(msg_obj, 'payload', None) or getattr(msg_obj, 'l3', None)
            if l3 is not None:
                proto = int(getattr(l3, 'proto', getattr(l3, 'protocol', 0)))
                src = str(getattr(l3, 'src', getattr(l3, 'source', src)))
                dst = str(getattr(l3, 'dst', getattr(l3, 'destination', dst)))
                sport = int(getattr(l3, 'sport', 0))
                dport = int(getattr(l3, 'dport', 0))

        payload_hex = payload_bytes.hex() if payload_bytes else '00'
        return {
            'src': src,
            'dst': dst,
            'sport': sport,
            'dport': dport,
            'proto': proto,
            'protocol': 'UNKNOWN',
            'payload_hex': payload_hex,
            'payload_size': len(payload_bytes),
            'has_payload': len(payload_bytes) > 0,
            'bcdg_segments': len(field_types),
            'field_types': field_types,
        }
    return seg


def extract_structural_fingerprint(seg):
    """
    从单条 NEMESYS 分段结果中提取 24 维结构指纹。

    Args:
        seg: dict 或 MessageSegment 对象，NEMESYS 分段结果

    Returns:
        np.ndarray: shape (24,), dtype float32, 各维度已归一化到 [0,1]
    """
    if not isinstance(seg, dict):
        seg = _ensure_dict(seg)
    if not isinstance(seg, dict):
        return np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    field_types = seg.get('field_types', [])
    payload_hex = seg.get('payload_hex', '')
    payload_size = seg.get('payload_size', 0)

    n_fields = len(field_types)

    if n_fields == 0:
        # 无 BCDG 分段 → 仅保留端口/proto/熵的弱信号
        entropy = _compute_entropy(payload_hex)
        sport = min(seg.get('sport', 0), 65535)
        dport = min(seg.get('dport', 0), 65535)
        proto = seg.get('proto', 0)
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[0] = 0.0                     # 字段数=0
        fp[10] = min(entropy / 8.0, 1.0)
        sport_raw = min(seg.get('sport', 0), 65535)
        dport_raw = min(seg.get('dport', 0), 65535)
        fp[16] = min(math.log2(sport_raw + 1) / 16.0, 1.0) if sport_raw > 0 else 0.0
        fp[17] = min(math.log2(dport_raw + 1) / 16.0, 1.0) if dport_raw > 0 else 0.0
        fp[18] = max(0.0, min(proto / 255.0, 1.0))
        return fp

    # ── 字段大小统计 [0-4] ──
    field_sizes = np.array([ft.get('length', 0) for ft in field_types], dtype=np.float32)
    mean_fs = float(np.mean(field_sizes))
    std_fs = float(np.std(field_sizes)) if n_fields > 1 else 0.0
    min_fs = float(np.min(field_sizes))
    max_fs = float(np.max(field_sizes))

    # 归一化
    # 注意（2026-09-22 修正）：三种归一化分母**不相同**，不要写成"都按平均字段大小归一"：
    #   第 0 维 = 字段数/30；第 1、2 维除以 (payload_size/n_fields) 即平均字段大小；
    #   第 3、4 维除以 payload_size。
    # 另：字段是对同一 payload 的切分，恒有 sum(field_lengths) == payload_size，
    #   故第 1 维 == mean_fs / mean_fs == 1.0（当 payload_size >= n_fields，而
    #   min_segment_len=2 保证总是成立）。即 24 维中有 1 维恒为常数、不携带信息。
    #   这对欧氏距离无影响（常数维对所有向量贡献相同），故保留不改；
    #   若要修则必须重跑全部依赖指纹的实验。
    denom = max(payload_size, 1)
    mean_fs_norm = min(mean_fs / max(denom / max(n_fields, 1), 1), 1.0)
    std_fs_norm = min(std_fs / max(denom / max(n_fields, 1), 1), 1.0)
    min_fs_norm = min(min_fs / denom, 1.0)
    max_fs_norm = min(max_fs / denom, 1.0)

    # ── 字段类型分布 [5-9] ──
    type_counts = Counter(ft.get('type', 'unknown') for ft in field_types)
    ratio_binary = type_counts.get('binary', 0) / n_fields
    ratio_text = type_counts.get('text', 0) / n_fields
    ratio_fixed = type_counts.get('fixed', 0) / n_fields
    ratio_zero = type_counts.get('zero', 0) / n_fields
    ratio_empty = type_counts.get('empty', 0) / n_fields

    # ── 字节级特征 [10-15] ──
    entropy = _compute_entropy(payload_hex)
    byte0 = 0.0
    byte1 = 0.0
    byte2 = 0.0
    byte3 = 0.0
    unique_ratio = 0.0

    if payload_hex and payload_hex != '00':
        try:
            raw = bytes.fromhex(payload_hex)
            if len(raw) >= 1:
                byte0 = raw[0]  # raw 0-255, protocol headers are highly discriminative
            if len(raw) >= 2:
                byte1 = raw[1]
            if len(raw) >= 3:
                byte2 = raw[2]
            if len(raw) >= 4:
                byte3 = raw[3]
            if raw:
                unique_ratio = len(set(raw)) / len(raw)
        except (ValueError, TypeError):
            pass

    # ── 传输特征 [16-18] ──
    # 用 log2 编码端口，增大低端口差异的权重（53 vs 67 比 50000 vs 50014 更显著）
    sport_raw = min(seg.get('sport', 0), 65535)
    dport_raw = min(seg.get('dport', 0), 65535)
    sport = min(math.log2(sport_raw + 1) / 16.0, 1.0) if sport_raw > 0 else 0.0
    dport = min(math.log2(dport_raw + 1) / 16.0, 1.0) if dport_raw > 0 else 0.0
    proto = max(0.0, min(seg.get('proto', 0) / 255.0, 1.0))

    # ── 字段大小直方图 [19-23]（5 个 bin）──
    bins = [0.0] * 5  # [1], [2], [3-4], [5-8], [9+]
    for sz in field_sizes:
        sz = int(sz)
        if sz == 1:
            bins[0] += 1
        elif sz == 2:
            bins[1] += 1
        elif sz <= 4:
            bins[2] += 1
        elif sz <= 8:
            bins[3] += 1
        else:
            bins[4] += 1
    bins = [b / n_fields for b in bins]

    fp = np.array([
        min(n_fields / 30.0, 1.0), # 0:  字段数 (cap at 30, ±2容忍度)
        mean_fs_norm,             # 1:  平均字段大小 (相对比例)
        std_fs_norm,              # 2:  字段大小标准差 (相对比例)
        min_fs_norm,              # 3:  最小字段 (相对比例)
        max_fs_norm,              # 4:  最大字段 (相对比例)
        ratio_binary,             # 5:  binary 字段比例
        ratio_text,               # 6:  text 字段比例
        ratio_fixed,              # 7:  fixed 字段比例
        ratio_zero,               # 8:  zero 字段比例
        ratio_empty,              # 9:  empty 字段比例
        min(entropy / 8.0, 1.0),  # 10: Shannon 熵 (cap at 8 bits)
        unique_ratio,             # 11: 唯一字节比例
        byte0 / 255.0,            # 12: 首字节 (协议头关键特征)
        byte1 / 255.0,            # 13: 第2字节
        byte2 / 255.0,            # 14: 第3字节
        byte3 / 255.0,            # 15: 第4字节
        sport,                    # 16: 源端口 log2(p+1)/16（非线性，见上方 L178）
        dport,                    # 17: 目的端口 log2(p+1)/16（非线性）
        proto,                    # 18: IP 协议号 / 255
        *bins,                    # 19-24: 大小直方图
    ], dtype=np.float32)

    return fp


def _compute_entropy(payload_hex):
    """计算 payload 的 Shannon 熵 (bits)。"""
    if not payload_hex or payload_hex == '00':
        return 0.0
    try:
        raw = bytes.fromhex(payload_hex)
    except (ValueError, TypeError):
        return 0.0
    if not raw:
        return 0.0
    counter = Counter(raw)
    total = len(raw)
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


# ============================================================
# 聚类
# ============================================================

def euclidean_distance(a, b):
    """两个归一化向量的欧氏距离。"""
    return float(np.linalg.norm(a - b))


def cluster_fingerprints(fingerprints, threshold=DEFAULT_DISTANCE_THRESHOLD):
    """
    基于欧氏距离的贪心聚类。

    对每个未分配的指纹，检查到最近簇中心的距离：
    - 距离 <= threshold → 归入该簇
    - 距离 > threshold → 创建新簇

    Args:
        fingerprints: list of np.ndarray, 每个指纹 shape (24,)
        threshold: float, 欧氏距离阈值（归一化向量推荐 0.3-0.6）

    Returns:
        labels: np.ndarray, 每个指纹的簇标签 (0-based)
        centroids: list of np.ndarray, 每个簇的中心向量
    """
    n = len(fingerprints)
    if n == 0:
        return np.array([], dtype=int), []

    labels = np.full(n, -1, dtype=int)
    centroids = []
    cluster_sizes = []

    for i, fp in enumerate(fingerprints):
        best_cluster = -1
        best_dist = float('inf')

        for c_idx in range(len(centroids)):
            centroid = centroids[c_idx] / cluster_sizes[c_idx]
            dist = euclidean_distance(fp, centroid)
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best_cluster = c_idx

        if best_cluster >= 0:
            labels[i] = best_cluster
            centroids[best_cluster] += fp
            cluster_sizes[best_cluster] += 1
        else:
            labels[i] = len(centroids)
            centroids.append(fp.copy())
            cluster_sizes.append(1)

    centroids = [c / cluster_sizes[i] for i, c in enumerate(centroids)]

    return labels, centroids


def detect_cluster_outliers(fingerprints, labels, centroids,
                            sigma=DEFAULT_OUTLIER_SIGMA,
                            min_cluster_size=MIN_CLUSTER_SIZE_FOR_OUTLIER):
    """
    在每个簇内检测离群点。

    计算每个点到簇中心的欧氏距离，标记距离超过
    (mean + sigma * std) 的点为离群。

    Returns:
        outlier_flags: np.ndarray of bool
        distances: np.ndarray, 每个点到簇中心的距离
    """
    n = len(fingerprints)
    outlier_flags = np.zeros(n, dtype=bool)
    distances = np.zeros(n, dtype=np.float32)

    unique_labels = sorted(set(int(l) for l in labels if l >= 0))

    for c in unique_labels:
        mask = labels == c
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) < min_cluster_size:
            continue

        centroid = centroids[c]
        cluster_dists = np.array([
            euclidean_distance(fingerprints[i], centroid)
            for i in cluster_indices
        ])

        distances[cluster_indices] = cluster_dists

        mean_dist = float(np.mean(cluster_dists))
        std_dist = float(np.std(cluster_dists))
        outlier_threshold = mean_dist + sigma * max(std_dist, 1e-8)

        for j, idx in enumerate(cluster_indices):
            if cluster_dists[j] > outlier_threshold:
                outlier_flags[idx] = True

    return outlier_flags, distances


# ============================================================
# 分析器主类
# ============================================================

class ClusterAnalyzer:
    """
    协议结构聚类分析器。

    用法:
        analyzer = ClusterAnalyzer(distance_threshold=0.45)
        result = analyzer.analyze(nemesys_segments)

        # result 包含:
        #   - clusters: 每个簇的描述
        #   - labels: 每个 segment 的簇标签
        #   - outlier_flags: 每个 segment 是否离群
        #   - summary: 整体分析摘要
    """

    def __init__(self, distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
                 outlier_sigma=DEFAULT_OUTLIER_SIGMA):
        self.distance_threshold = distance_threshold
        self.outlier_sigma = outlier_sigma

    def analyze(self, segments):
        """
        对 NEMESYS 分段结果做完整的结构聚类分析。

        Args:
            segments: list of dict, NEMESYS 分段结果

        Returns:
            dict: {
                'fingerprints': list of np.ndarray,
                'labels': np.ndarray,
                'centroids': list of np.ndarray,
                'outlier_flags': np.ndarray,
                'distances': np.ndarray,
                'clusters': list of dict (per-cluster description),
                'summary': dict,
            }
        """
        if not segments:
            return self._empty_result()

        # Step 1: 提取结构指纹
        fingerprints = []
        valid_indices = []
        for i, seg in enumerate(segments):
            fp = extract_structural_fingerprint(seg)
            if np.any(np.isfinite(fp)):
                fingerprints.append(fp)
                valid_indices.append(i)

        if not fingerprints:
            return self._empty_result()

        # Step 2: 聚类
        labels, centroids = cluster_fingerprints(
            fingerprints, threshold=self.distance_threshold
        )

        # Step 3: 离群检测
        outlier_flags, distances = detect_cluster_outliers(
            fingerprints, labels, centroids, sigma=self.outlier_sigma
        )

        # Step 4: 构建簇描述
        clusters = self._describe_clusters(segments, valid_indices,
                                           fingerprints, labels,
                                           centroids, outlier_flags, distances)

        # Step 5: 汇总
        summary = self._build_summary(clusters, outlier_flags, distances)

        # 扩展 labels 到原始 segments 长度（未提取指纹的标记为 -1）
        full_labels = np.full(len(segments), -1, dtype=int)
        full_outliers = np.zeros(len(segments), dtype=bool)
        full_distances = np.full(len(segments), -1.0, dtype=np.float32)
        for j, orig_idx in enumerate(valid_indices):
            full_labels[orig_idx] = int(labels[j])
            full_outliers[orig_idx] = bool(outlier_flags[j])
            full_distances[orig_idx] = float(distances[j])

        return {
            'fingerprints': fingerprints,
            'valid_indices': valid_indices,
            'labels': full_labels,
            'centroids': centroids,
            'outlier_flags': full_outliers,
            'distances': full_distances,
            'clusters': clusters,
            'summary': summary,
        }

    def _empty_result(self):
        return {
            'fingerprints': [],
            'valid_indices': [],
            'labels': np.array([], dtype=int),
            'centroids': [],
            'outlier_flags': np.array([], dtype=bool),
            'distances': np.array([], dtype=np.float32),
            'clusters': [],
            'summary': {
                'total_clusters': 0,
                'total_outliers': 0,
                'largest_cluster_size': 0,
                'description': '无有效分段数据',
            },
        }

    def _describe_clusters(self, segments, valid_indices, fingerprints,
                           labels, centroids, outlier_flags, distances):
        """为每个簇构建可读描述。"""
        clusters = []
        unique_labels = sorted(set(int(l) for l in labels if l >= 0))

        for c in unique_labels:
            mask = labels == c
            cluster_indices = np.where(mask)[0]
            n_members = len(cluster_indices)

            # 收集该簇的元数据
            protocols = Counter()
            ports = Counter()
            entropies = []
            payload_sizes = []
            field_counts = []
            outlier_count = 0

            for j in cluster_indices:
                orig_idx = valid_indices[j]
                seg = segments[orig_idx]
                protocols[seg.get('protocol', 'UNKNOWN_PROTO')] += 1
                ports[seg.get('sport', 0)] += 1
                ports[seg.get('dport', 0)] += 1

                payload_hex = seg.get('payload_hex', '')
                entropies.append(_compute_entropy(payload_hex))
                payload_sizes.append(seg.get('payload_size', 0))
                field_counts.append(seg.get('bcdg_segments', 0))

                if outlier_flags[j]:
                    outlier_count += 1

            # 簇的统计特征
            avg_entropy = float(np.mean(entropies)) if entropies else 0.0
            avg_payload = float(np.mean(payload_sizes)) if payload_sizes else 0.0
            avg_fields = float(np.mean(field_counts)) if field_counts else 0.0

            # 主导协议和端口
            dominant_proto = protocols.most_common(1)[0][0] if protocols else 'unknown'
            top_ports = [p for p, _ in ports.most_common(3)]

            # 判断簇的性质
            is_known = dominant_proto not in ('UNKNOWN_PROTO', 'unknown', 'EMPTY')

            # 簇的紧密度（平均到中心的欧氏距离）
            centroid = centroids[c]
            cluster_dists = [euclidean_distance(fingerprints[j], centroid)
                             for j in cluster_indices]
            mean_dist = float(np.mean(cluster_dists)) if cluster_dists else 0.0
            cohesion = 'tight' if mean_dist < 0.15 else ('moderate' if mean_dist < 0.35 else 'loose')

            # 协议名推断
            inferred = self._infer_protocol_name(dominant_proto, top_ports, avg_entropy,
                                                 avg_fields, avg_payload)

            clusters.append({
                'cluster_id': int(c) + 1,
                'size': n_members,
                'outlier_count': outlier_count,
                'outlier_ratio': round(outlier_count / n_members, 3),
                'avg_entropy': round(avg_entropy, 2),
                'avg_payload_size': round(avg_payload, 1),
                'avg_fields': round(avg_fields, 1),
                'dominant_protocol': dominant_proto,
                'is_known_protocol': is_known,
                'top_ports': top_ports,
                'cohesion': cohesion,
                'mean_distance': round(mean_dist, 4),
                'inferred_name': inferred,
                'indicators': self._cluster_indicators(
                    avg_entropy, n_members, outlier_count, is_known, cohesion
                ),
            })

        # 按大小降序排列
        clusters.sort(key=lambda c: c['size'], reverse=True)
        # 重编号
        for i, c in enumerate(clusters):
            c['cluster_id'] = i + 1

        return clusters

    def _infer_protocol_name(self, dominant_proto, top_ports, avg_entropy,
                             avg_fields, avg_payload):
        """根据簇特征推断可能的协议名。"""
        if dominant_proto and dominant_proto not in ('UNKNOWN_PROTO', 'unknown', 'EMPTY'):
            return dominant_proto

        # 基于端口推断
        well_known = {53: 'DNS', 67: 'DHCP', 68: 'DHCP', 123: 'NTP',
                      161: 'SNMP', 502: 'Modbus', 20000: 'DNP3',
                      102: 'S7Comm', 44818: 'EtherNet/IP'}
        for port in top_ports:
            if port in well_known:
                return f'{well_known[port]}(推断)'

        # 基于特征推断
        if avg_entropy > 7.0 and avg_payload > 100:
            return '加密隧道(疑似)'
        if avg_fields <= 3 and avg_payload < 64:
            return '简单请求-响应协议'
        if avg_entropy < 3.0 and avg_fields > 5:
            return '结构化明文协议'

        return '未知协议'

    def _cluster_indicators(self, avg_entropy, n_members, outlier_count,
                            is_known, cohesion):
        """生成簇级异常指标。"""
        flags = []
        if avg_entropy > 7.0:
            flags.append('高熵(加密/混淆)')
        if not is_known:
            flags.append('未知协议')
        if n_members <= 2:
            flags.append('孤立小簇')
        if outlier_count > n_members * 0.3:
            flags.append('高离群率')
        if cohesion == 'loose':
            flags.append('结构松散(可能存在协议变种)')
        return flags

    def _build_summary(self, clusters, outlier_flags, distances):
        """构建整体分析摘要。"""
        total_outliers = int(outlier_flags.sum()) if len(outlier_flags) > 0 else 0
        total_clusters = len(clusters)
        largest = clusters[0]['size'] if clusters else 0

        suspicious_clusters = [c for c in clusters if len(c.get('indicators', [])) >= 2]
        suspicious_flow_count = sum(c['size'] for c in suspicious_clusters)
        outlier_flow_count = sum(c['outlier_count'] for c in clusters)

        # 整体评估
        if suspicious_clusters:
            if suspicious_flow_count > outlier_flow_count * 2:
                level = 'high'
                desc = (f'发现 {len(suspicious_clusters)} 个可疑协议簇，共 {suspicious_flow_count} 条流，'
                        f'其中 {outlier_flow_count} 条结构离群')
            else:
                level = 'medium'
                desc = (f'{total_clusters} 个协议簇中 {len(suspicious_clusters)} 个存在可疑特征，'
                        f'{outlier_flow_count} 条流结构异常')
        elif outlier_flow_count > 0:
            level = 'low'
            desc = f'{total_clusters} 个协议簇结构正常，但 {outlier_flow_count} 条个体流存在结构偏差'
        else:
            level = 'none'
            desc = f'{total_clusters} 个协议簇结构均正常，未发现异常'

        return {
            'total_clusters': total_clusters,
            'total_outliers': total_outliers,
            'largest_cluster_size': largest,
            'suspicious_clusters': len(suspicious_clusters),
            'suspicious_flow_count': suspicious_flow_count,
            'outlier_flow_count': outlier_flow_count,
            'risk_level': level,
            'description': desc,
        }


# ============================================================
# GNN4ID 域检测
# ============================================================

# CIC-IoT-2023 训练集的协议范围
# 训练数据中出现的协议：TCP(6) 和 UDP(17) 为主，含少量 ICMP(1)
# 训练数据端口特征：IoT 设备常见端口 + 攻击端口
GNN_IOT_DOMAIN_PORTS = {
    # IoT 常见服务端口
    80, 443, 8080,          # HTTP/HTTPS (Web 管理界面)
    1883, 8883,             # MQTT
    22,                     # SSH
    23,                     # Telnet (IoT 设备常见)
    21, 20,                 # FTP
    # 攻击常见目标端口 (CIC-IoT-2023 攻击流量使用)
    3306,                   # MySQL
    6379,                   # Redis
    3389,                   # RDP
    445, 139,               # SMB
    1433,                   # MSSQL
}
# 注意：DNS(53)、NTP(123)、DHCP(67/68) 不在域内——GNN训练数据CIC-IoT-2023不含这些协议

GNN_TRAINING_CLASSES = {
    'Benign', 'WebBased', 'Spoofing', 'Recon',
    'Mirai', 'Dos', 'DDos', 'BruteForce',
}


def is_in_gnn_domain(protocol_name, sport, dport):
    """
    判断一条流是否在 GNN4ID 训练域内。

    GNN4ID 在 CIC-IoT-2023 上训练，只有 IoT 域流量才是它的有效输入。
    非 IoT 协议（DNS/DHCP/NTP 服务端, Modbus/DNP3/S7Comm 工控）不在域内。

    Args:
        protocol_name: str, 协议名称 (NEMESYS 识别结果)
        sport: int, 源端口
        dport: int, 目的端口

    Returns:
        bool: True if the flow is within GNN training domain
    """
    # 已知的 GNN 训练域协议（CIC-IoT-2023 涵盖）
    iot_protocols = {'HTTP', 'TLS', 'MQTT', 'MQTTS', 'SSH', 'FTP', 'Telnet',
                     'MySQL', 'Redis', 'SMTP', 'POP3', 'IMAP'}
    # 注意：DNS/DHCP/NTP等网络服务协议不在IoT域内

    if protocol_name in iot_protocols:
        return True

    # 端口在 IoT 域范围内（可能是 GNN 见过的模式）
    if sport in GNN_IOT_DOMAIN_PORTS or dport in GNN_IOT_DOMAIN_PORTS:
        return True

    return False


def classify_flow_trust(protocol_name, is_ood, cluster_info=None):
    """
    根据协议域和聚类结果，输出对该流 GNN 分类的信任度。

    Returns:
        dict: {
            'trust_gnn': bool,          # 是否可以信任 GNN 分类
            'reason': str,              # 原因说明
            'recommendation': str,      # 建议处理方式
        }
    """
    if is_ood:
        if cluster_info and cluster_info.get('cohesion') == 'tight':
            return {
                'trust_gnn': False,
                'reason': '训练域外协议，但在结构指纹库中形成紧簇',
                'recommendation': '以聚类结果为准，GNN分类仅作参考',
            }
        else:
            return {
                'trust_gnn': False,
                'reason': '训练域外协议，结构特征与训练集分布不符',
                'recommendation': '忽略GNN分类，仅使用结构异常检测',
            }
    else:
        return {
            'trust_gnn': True,
            'reason': '流量特征在GNN训练域内',
            'recommendation': '正常使用GNN分类结果',
        }


def route_flows_by_protocol(flow_predictions):
    """
    Split flow predictions into in-domain and OOD sets based on protocol.

    Args:
        flow_predictions: dict mapping canonical flow tuple -> {protocol, confidence, ...}
                         (as produced by FieldProtocolClassifier.predict_segments())

    Returns:
        dict: {
            'in_domain_keys': set of canonical flow tuples in GNN training domain,
            'ood_keys': set of canonical flow tuples outside GNN domain,
            'flow_protocols': dict mapping canonical key -> protocol_name,
        }
    """
    in_domain_keys = set()
    ood_keys = set()
    flow_protocols = {}

    for key, pred in flow_predictions.items():
        proto_name = pred.get('protocol', 'UNKNOWN_PROTO')
        flow_protocols[key] = proto_name

        if is_in_gnn_domain(proto_name, 0, 0):
            in_domain_keys.add(key)
        else:
            ood_keys.add(key)

    return {
        'in_domain_keys': in_domain_keys,
        'ood_keys': ood_keys,
        'flow_protocols': flow_protocols,
    }
