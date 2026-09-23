"""
字段级异构图构建：将 NEMESYS BCDG 字段分段结果转换为 HeteroData 图

图结构：
  proto 节点 (1个):  消息级全局特征
  field 节点 (N个):  每个 BCDG 字段的特征
  边:
    proto → field: contain 边
    field_i → field_{i+1}: sequential link 边

用途：训练 GNN 从字段结构识别协议类型（替代端口查表）
"""

import math
import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from collections import Counter


# 字段特征维度
FIELD_FEAT_DIM = 18  # 12 基础 + 6 字段类型 one-hot（text/binary/zero/fixed/empty/other）
# 消息级特征维度
PROTO_FEAT_DIM = 18  # 16 基础 + 2 字段类型占比

# 字段类型 → 6 维 one-hot 向量（与 _classify_field_type 的返回值一一对应）
_FIELD_TYPE_ONE_HOT = {
    'text':   [1, 0, 0, 0, 0, 0],
    'binary': [0, 1, 0, 0, 0, 0],
    'zero':   [0, 0, 1, 0, 0, 0],
    'fixed':  [0, 0, 0, 1, 0, 0],
    'empty':  [0, 0, 0, 0, 1, 0],
    'other':  [0, 0, 0, 0, 0, 1],
}

# 协议类别
PROTO_CLASSES = ['DHCP', 'DNS', 'NTP', 'Modbus', 'DNP3', 'S7Comm', 'MQTT', 'HTTP', 'SSH', 'FTP', 'TLS', 'mDNS']
PROTO_TO_ID = {p: i for i, p in enumerate(PROTO_CLASSES)}


def _compute_entropy(byte_seq):
    """计算字节序列的 Shannon 熵 (bits)"""
    if not byte_seq:
        return 0.0
    counter = Counter(byte_seq)
    total = len(byte_seq)
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def _classify_field_type(raw_bytes):
    """
    将字段字节序列分类为 text / binary / zero / fixed / empty / other。

    判定顺序（与字段级 one-hot 及消息级类型占比保持一致）：
      empty → zero（全 0）→ fixed（单一非零值）→ text（可打印字符 >80%）
      → other（仅含 0/255 边界值）→ binary
    """
    if not raw_bytes:
        return 'empty'
    unique_bytes = set(raw_bytes)
    if len(unique_bytes) == 1:
        return 'zero' if 0 in unique_bytes else 'fixed'
    if sum(1 for b in raw_bytes if 32 <= b <= 126) > len(raw_bytes) * 0.8:
        return 'text'
    if all(b in (0, 255) for b in raw_bytes):
        return 'other'
    return 'binary'


def extract_field_features(msg_segments):
    """
    从一条消息的 NEMESYS 分段结果中提取每个字段的特征向量。

    Args:
        msg_segments: list of MessageSegment 对象（完整NEMESYS输出）
                      或 list of dict（模拟模式，含 offset/length/bytes）

    Returns:
        list of np.ndarray, 每个字段 shape (FIELD_FEAT_DIM,)
    """
    if not msg_segments:
        return []

    # 总 payload 大小
    if hasattr(msg_segments[0], 'offset'):
        total_len = sum(getattr(s, 'length', 0) for s in msg_segments)
    else:
        total_len = sum(s.get('length', 0) for s in msg_segments)

    if total_len == 0:
        return []

    field_feats = []

    for i, seg in enumerate(msg_segments):
        if hasattr(seg, 'offset'):
            offset = getattr(seg, 'offset', 0)
            length = getattr(seg, 'length', 0)
            raw_bytes = getattr(seg, 'bytes', b'')
        else:
            offset = seg.get('offset', 0)
            length = seg.get('length', 0)
            raw_bytes = seg.get('bytes', b'') if isinstance(seg.get('bytes'), bytes) else b''

        # 字段类型 one-hot（text / binary / zero / fixed / empty / other）
        type_oh = _FIELD_TYPE_ONE_HOT[_classify_field_type(raw_bytes)]
        n_unique = len(set(raw_bytes))

        feat = [
            offset / max(total_len, 1),           # 0: 归一化偏移
            length / max(total_len, 1),           # 1: 归一化长度
            math.log2(length + 1) / 10.0,         # 2: log2 长度
            (raw_bytes[0] / 255.0) if len(raw_bytes) >= 1 else 0.0,  # 3
            (raw_bytes[1] / 255.0) if len(raw_bytes) >= 2 else 0.0,  # 4
            (raw_bytes[2] / 255.0) if len(raw_bytes) >= 3 else 0.0,  # 5
            (raw_bytes[3] / 255.0) if len(raw_bytes) >= 4 else 0.0,  # 6
            i / max(len(msg_segments), 1),        # 7: 字段序号/总数
            1.0 if i == 0 else 0.0,               # 8: 是否首字段
            1.0 if i == len(msg_segments) - 1 else 0.0,  # 9: 是否尾字段
            min(_compute_entropy(raw_bytes) / 8.0, 1.0) if raw_bytes else 0.0,  # 10
            (1.0 - n_unique / max(len(raw_bytes), 1)) if raw_bytes else 0.0,  # 11
        ] + type_oh                                # 12-17: 类型 one-hot
        field_feats.append(np.array(feat, dtype=np.float32))

    return field_feats


def extract_proto_features(msg_segments, sport, dport, proto):
    """
    提取消息级全局特征向量。

    Args:
        msg_segments: 字段分段列表
        sport: 源端口
        dport: 目的端口
        proto: IP 协议号

    Returns:
        np.ndarray shape (PROTO_FEAT_DIM,)
    """
    n_fields = len(msg_segments)

    total_len = 0
    if msg_segments:
        if hasattr(msg_segments[0], 'offset'):
            total_len = sum(getattr(s, 'length', 0) for s in msg_segments)
        else:
            total_len = sum(s.get('length', 0) for s in msg_segments)

    # 收集所有字段的字节做全局分析
    all_bytes = b''
    for seg in msg_segments:
        if hasattr(seg, 'bytes'):
            all_bytes += getattr(seg, 'bytes', b'')
        elif isinstance(seg.get('bytes'), bytes):
            all_bytes += seg['bytes']

    # 字段长度统计
    lengths = []
    for seg in msg_segments:
        if hasattr(seg, 'length'):
            lengths.append(getattr(seg, 'length', 0))
        else:
            lengths.append(seg.get('length', 0))

    mean_len = np.mean(lengths) if lengths else 0
    std_len = np.std(lengths) if len(lengths) > 1 else 0

    # 字段类型比例（复用 _classify_field_type，与字段级 one-hot 保持一致）
    type_counts = Counter()
    for seg in msg_segments:
        if hasattr(seg, 'bytes'):
            b = getattr(seg, 'bytes', b'')
        else:
            b = seg.get('bytes', b'')
        type_counts[_classify_field_type(b)] += 1

    n = max(n_fields, 1)

    # 文本/二进制字段占比统计（用于第 16/17 维）
    # 'other'（仅含 0/255 的边界值）也归入非文本侧，与字段级分类不矛盾
    n_text = type_counts.get('text', 0)
    n_binary = (type_counts.get('binary', 0) + type_counts.get('fixed', 0)
                + type_counts.get('zero', 0) + type_counts.get('other', 0))

    feat = [
        min(n_fields / 30.0, 1.0),                 # 0: 字段数
        min(total_len / 1500.0, 1.0),              # 1: 总长度
        min(mean_len / 500.0, 1.0),                # 2: 平均字段长度
        min(std_len / 500.0, 1.0),                 # 3: 字段长度标准差
        type_counts.get('binary', 0) / n,          # 4: binary比例
        type_counts.get('text', 0) / n,            # 5: text比例
        type_counts.get('fixed', 0) / n,           # 6: fixed比例
        type_counts.get('zero', 0) / n,            # 7: zero比例
        type_counts.get('empty', 0) / n,           # 8: empty比例
        min(_compute_entropy(all_bytes) / 8.0, 1.0) if all_bytes else 0.0,  # 9: 全局熵
        min(math.log2(sport + 1) / 16.0, 1.0) if sport > 0 else 0.0,  # 10: sport
        min(math.log2(dport + 1) / 16.0, 1.0) if dport > 0 else 0.0,  # 11: dport
        proto / 255.0,                             # 12: 传输协议
        1.0 if sport < 1024 else 0.0,              # 13: 知名端口标记
        1.0 if dport < 1024 else 0.0,              # 14: 知名端口标记
        (len(set(all_bytes)) / max(len(all_bytes), 1)) if all_bytes else 0.0,  # 15: 唯一字节比
        n_text / n,      # 16: 文本字段占比
        n_binary / n,    # 17: 二进制字段占比
    ]
    return np.array(feat, dtype=np.float32)


def build_proto_graph(msg_segments, sport=0, dport=0, proto=0, label=None):
    """
    从一条消息的 NEMESYS BCDG 分段构建 HeteroData 图。

    图结构:
        proto node:   消息级特征 (1 个)
        field nodes:  字段级特征 (N 个)
        contain 边:   proto → field[i]   (含位置特征)
        link 边:      field[i] → field[i+1]  (含间距特征)

    Args:
        msg_segments: NEMESYS 一条消息的分段列表
        sport: 源端口
        dport: 目的端口
        proto: IP 协议号
        label: 协议类别 ID (0-5)

    Returns:
        HeteroData 图对象
    """
    field_feats = extract_field_features(msg_segments)
    proto_feat = extract_proto_features(msg_segments, sport, dport, proto)
    n_fields = len(field_feats)

    if n_fields == 0:
        # 无字段时返回最小图
        field_feats = [np.zeros(FIELD_FEAT_DIM, dtype=np.float32)]

    field_node_x = torch.tensor(np.stack(field_feats), dtype=torch.float32)
    proto_node_x = torch.tensor([proto_feat], dtype=torch.float32)

    # --- Contain 边: proto → each field ---
    contain_src = np.zeros(n_fields, dtype=int)
    contain_dst = np.arange(n_fields)
    contain_edge_index = torch.tensor(
        np.vstack([contain_src, contain_dst]), dtype=torch.int64
    )
    # 边特征: 字段位置 + 归一化偏移
    contain_edge_attr = []
    for i in range(n_fields):
        contain_edge_attr.append([
            i / max(n_fields, 1),
            field_feats[i][0],  # 归一化 offset
        ])
    contain_edge_attr = torch.tensor(contain_edge_attr, dtype=torch.float32)

    # --- Link 边: field[i] → field[i+1] ---
    if n_fields > 1:
        link_src = np.arange(n_fields - 1)
        link_dst = np.arange(1, n_fields)
        link_edge_index = torch.tensor(
            np.vstack([link_src, link_dst]), dtype=torch.int64
        )
        # 边特征: 间距 + 类型变化
        link_edge_attr = []
        for i in range(n_fields - 1):
            curr_len = field_feats[i][1] * 1500  # 反归一化近似
            link_edge_attr.append([
                field_feats[i + 1][0] - field_feats[i][0],  # offset 差
                abs(field_feats[i + 1][1] - field_feats[i][1]),  # 长度差
            ])
        link_edge_attr = torch.tensor(link_edge_attr, dtype=torch.float32)
    else:
        link_edge_index = torch.zeros((2, 0), dtype=torch.int64)
        link_edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    # --- 组装 HeteroData ---
    data = HeteroData()
    data['proto'].x = proto_node_x
    data['field'].x = field_node_x
    data['proto', 'contain', 'field'].edge_index = contain_edge_index
    data['field', 'link', 'field'].edge_index = link_edge_index
    data['proto', 'contain', 'field'].edge_attr = contain_edge_attr
    data['field', 'link', 'field'].edge_attr = link_edge_attr

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.int64)

    data = T.ToUndirected()(data)
    return data
