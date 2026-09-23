"""
GNN4ID 分析器：封装特征提取、异构图构建、GNN模型推理

复用 GNN4ID 项目的核心逻辑：
- 特征提取: nfstream NFStreamer + My_Custom 插件
- 图构建: HeteroData (flow nodes + packet nodes)
- 模型: HeteroGNN (SAGEConv) / HeteroGNN_Edge (GATConv)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from nemesys_gnn4id.config import ATTACK_TYPES, LABEL_DICT, GNN_DEFAULTS, FLOW_FEATURE_NAMES, PROTO_ONEHOT


# ============================================================
# nfstream 特征提取插件 (复用 GNN4ID/Utility/Feature_extractor_flow_packet_combined.py)
# ============================================================

def _preload_npcap_wpcap():
    """Windows: make nfstream's `_lib_engine` resolve against Npcap's wpcap.dll.

    nfstream's native extension imports `pcap_statustostr` from `wpcap.dll`. That
    symbol is a libpcap 1.0+/Npcap addition and is NOT exported by the legacy
    WinPcap 4.1.x `C:\\Windows\\System32\\wpcap.dll`. When both WinPcap and Npcap
    are installed, System32's stale copy wins the load and `import nfstream` fails
    with ``DLL load failed while importing _lib_engine: 找不到指定的程序``
    (ERROR_PROC_NOT_FOUND, 127) -- which looks like "nfstream is not installed"
    but is really a DLL precedence problem.

    Loading Npcap's wpcap.dll first registers it in the process under the base name
    `wpcap.dll`, so the later implicit dependency resolves to the working one.
    Npcap normally lives in ``%SystemRoot%\\System32\\Npcap``.

    Returns a short status string. Never raises: if this fails, nfstream stays
    unavailable and the scapy fallback is used, exactly as before.
    """
    if sys.platform != 'win32':
        return 'not-windows'
    import ctypes
    candidates = []
    override = os.environ.get('NPCAP_DLL_DIR')
    if override:
        candidates.append(override)
    sysroot = os.environ.get('SystemRoot', r'C:\Windows')
    candidates += [os.path.join(sysroot, 'System32', 'Npcap'),
                   r'C:\Program Files\Npcap']
    for d in candidates:
        p = os.path.join(d, 'wpcap.dll')
        if not os.path.exists(p):
            continue
        try:
            ctypes.WinDLL(p)
            return f'preloaded {p}'
        except OSError as e:
            last = e
    return 'no usable Npcap wpcap.dll found'


# Feature-extraction backend: 'auto' (nfstream if usable, else scapy), or force one
# with NEMESYS_FEATURE_BACKEND=nfstream|scapy. The forced-scapy mode exists so the
# degraded path can be re-run as a robustness control without editing code.
FEATURE_BACKEND = os.environ.get('NEMESYS_FEATURE_BACKEND', 'auto').strip().lower()
_NPCAP_STATUS = _preload_npcap_wpcap() if FEATURE_BACKEND != 'scapy' else 'skipped (scapy forced)'

try:
    from nfstream import NFStreamer, NFPlugin
    _NFSTREAM_IMPORTABLE = True
except ImportError:
    _NFSTREAM_IMPORTABLE = False
    NFStreamer = NFPlugin = None

NFSTREAM_AVAILABLE = _NFSTREAM_IMPORTABLE and FEATURE_BACKEND != 'scapy'

if not NFSTREAM_AVAILABLE:
    if FEATURE_BACKEND == 'scapy':
        print("[GNN4ID] 特征后端被强制为 scapy（NEMESYS_FEATURE_BACKEND=scapy）")
    else:
        print(f"[WARNING] nfstream 不可用，将降级到 scapy 特征提取（{_NPCAP_STATUS}）")
elif FEATURE_BACKEND == 'nfstream' or _NPCAP_STATUS.startswith('preloaded'):
    print(f"[GNN4ID] nfstream 可用（{_NPCAP_STATUS}）")


# Which extractor the most recent extract_features() call used. Recorded because the
# nfstream and scapy paths are NOT equivalent (see last_feature_backend()): any persisted
# result must state its provenance.
#
# Exactly what the record means, branch by branch:
#   'nfstream'        the call RETURNED nfstream features (written after the call)
#   'scapy'           scapy was INVOKED -- in BOTH scapy branches (forced scapy,
#                     nfstream unavailable) the record is written BEFORE the extractor
#                     runs, so it names the extractor that was invoked, not one that is
#                     proven to have produced the returned frame
#   'nfstream-failed' nfstream was attempted and produced nothing (it raised, or it is
#                     unavailable while NEMESYS_FEATURE_BACKEND=nfstream). Written
#                     before the RuntimeError is raised, so a caller that catches the
#                     exception cannot read the previous call's record by mistake.
_LAST_BACKEND = {'backend': None, 'reason': None}


def _record_backend(backend, reason):
    _LAST_BACKEND['backend'] = backend
    _LAST_BACKEND['reason'] = reason


def last_feature_backend():
    """Return (backend, reason) for the most recent extract_features() call.

    backend is one of:
      'nfstream'        that extractor produced the features returned by the call;
      'scapy'           scapy was INVOKED -- in both scapy branches (forced scapy and
                        'nfstream unavailable') the record is written BEFORE the
                        extractor runs, so this does not prove which code produced the
                        returned frame, only which extractor was entered;
      'nfstream-failed' nfstream was attempted and produced nothing: it raised, or it
                        is unavailable while NEMESYS_FEATURE_BACKEND=nfstream;
      None              nothing has been extracted yet.

    Callers that persist results MUST record this: the two backends yield
    materially different 82-dim features (~49% cell agreement, 12/82 dimensions
    never agree), so a result is meaningless without knowing which one produced it.
    """
    return _LAST_BACKEND['backend'], _LAST_BACKEND['reason']


if NFSTREAM_AVAILABLE:
    class FlowPacketExtractor(NFPlugin):
        """从PCAP提取flow级别+packet级别特征"""

        def on_init(self, packet, flow):
            if self.limit == 1:
                flow.expiration_id = -1

            flow.udps.payload_data = []
            if packet.payload_size > 0:
                flow.udps.payload_data.append(packet.ip_packet[-packet.payload_size:].hex())
            else:
                flow.udps.payload_data.append('00')

            flow.udps.delta_time = []
            flow.udps.delta_time.append(packet.delta_time)

            flow.udps.packet_direction = []
            flow.udps.packet_direction.append(packet.direction)

            flow.udps.ip_size = []
            flow.udps.ip_size.append(packet.ip_size)

            flow.udps.transport_size = []
            flow.udps.transport_size.append(packet.transport_size)

            flow.udps.payload_size = []
            flow.udps.payload_size.append(packet.payload_size)

            flow.udps.syn = []
            flow.udps.syn.append(packet.syn)
            flow.udps.cwr = []
            flow.udps.cwr.append(packet.cwr)
            flow.udps.ece = []
            flow.udps.ece.append(packet.ece)
            flow.udps.urg = []
            flow.udps.urg.append(packet.urg)
            flow.udps.ack = []
            flow.udps.ack.append(packet.ack)
            flow.udps.psh = []
            flow.udps.psh.append(packet.psh)
            flow.udps.rst = []
            flow.udps.rst.append(packet.rst)
            flow.udps.fin = []
            flow.udps.fin.append(packet.fin)

        def on_update(self, packet, flow):
            if packet.payload_size > 0:
                flow.udps.payload_data.append(packet.ip_packet[-packet.payload_size:].hex())
            else:
                flow.udps.payload_data.append('00')

            flow.udps.delta_time.append(packet.delta_time)
            flow.udps.packet_direction.append(packet.direction)
            flow.udps.ip_size.append(packet.ip_size)
            flow.udps.transport_size.append(packet.transport_size)
            flow.udps.payload_size.append(packet.payload_size)
            flow.udps.syn.append(packet.syn)
            flow.udps.cwr.append(packet.cwr)
            flow.udps.ece.append(packet.ece)
            flow.udps.urg.append(packet.urg)
            flow.udps.ack.append(packet.ack)
            flow.udps.psh.append(packet.psh)
            flow.udps.rst.append(packet.rst)
            flow.udps.fin.append(packet.fin)

            if self.limit == flow.bidirectional_packets:
                flow.expiration_id = -1


# ============================================================
# 异构图构建 (复用 GNN4ID/Utility/Functions.py 的逻辑)
# ============================================================

class GraphBuilder:
    """将 flow+packet 特征 DataFrame 转换为 HeteroData 图对象"""

    # 需要从 flow node features 中排除的列 (scapy 模式也用这些名字)
    FLOW_DROP_COLS = [
        'udps.payload_data', 'udps.delta_time', 'udps.packet_direction',
        'udps.ip_size', 'udps.transport_size', 'udps.payload_size',
        'udps.syn', 'udps.cwr', 'udps.ece', 'udps.urg',
        'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin',
        # scapy mode list columns (no udps. prefix)
        'payload_data', 'delta_time', 'packet_direction',
        'ip_size', 'transport_size', 'payload_size',
        'syn', 'cwr', 'ece', 'urg', 'ack', 'psh', 'rst', 'fin',
    ]

    # 也排除的非特征列
    META_DROP_COLS = [
        'src_ip', 'src_port', 'dst_ip', 'dst_port', 'ip_version',
        'src_mac', 'src_oui', 'dst_mac', 'dst_oui',
        'vlan_id', 'tunnel_id', 'id',
        # scapy mode metadata
        'first_seen', 'last_seen', 'protocol',
    ]

    PAYLOAD_DIM = 1500  # payload 字节长度

    @staticmethod
    def parse_list_column(series):
        """将字符串形式的列表列解析为真正的列表"""
        return series.map(lambda x: str(x).strip('[]').replace("'", "").split(', '))

    @staticmethod
    def build_single_graph(flow_row, label=None, expected_flow_dim=None):
        """从单条 flow 数据构建一个 HeteroData 图

        Args:
            flow_row: dict or Series with flow features.
                     If 'flow_features' key exists, uses that as the 82-dim vector directly.
                     Otherwise falls back to column-based extraction.
            label: optional label
            expected_flow_dim: if set, pad/truncate flow features to this dim.
        """
        # Helper: try both prefixed and unprefixed column names
        def _get_list(key):
            for k in [f'udps.{key}', key]:
                if k in flow_row:
                    return GraphBuilder.parse_list_column(pd.Series([flow_row[k]]))[0]
            return ['0']

        payload_data = _get_list('payload_data')
        delta_time = _get_list('delta_time')
        pkt_direction = _get_list('packet_direction')
        ip_size = _get_list('ip_size')
        transport_size = _get_list('transport_size')
        payload_size = _get_list('payload_size')

        num_packets = len(payload_data)

        # --- Flow node features ---
        # If pre-computed 82-dim vector is available (from scapy extraction), use it directly
        if 'flow_features' in flow_row and isinstance(flow_row['flow_features'], list):
            flow_feats = [float(x) for x in flow_row['flow_features']]
        else:
            # Fallback: extract from columns (for nfstream mode or CSV data)
            flow_feats_keys = [k for k in flow_row.index
                               if k not in GraphBuilder.FLOW_DROP_COLS
                               and k not in GraphBuilder.META_DROP_COLS
                               and k != 'Label']
            flow_feats = []
            for k in flow_feats_keys:
                v = flow_row[k]
                try:
                    flow_feats.append(float(v))
                except (ValueError, TypeError):
                    flow_feats.append(0.0)

        # Pad/truncate to expected dimension if specified
        if expected_flow_dim is not None:
            if len(flow_feats) < expected_flow_dim:
                flow_feats.extend([0.0] * (expected_flow_dim - len(flow_feats)))
            elif len(flow_feats) > expected_flow_dim:
                flow_feats = flow_feats[:expected_flow_dim]

        flow_node_feats = torch.tensor([flow_feats], dtype=torch.float32)

        # --- Packet node features (payload bytes) ---
        packet_node_feats = []
        for i in range(num_packets):
            try:
                byte_array = bytes.fromhex(payload_data[i])
                byte_lst = list(byte_array)
            except (ValueError, TypeError):
                byte_lst = [0]
            if len(byte_lst) < GraphBuilder.PAYLOAD_DIM:
                feat = np.pad(byte_lst, (0, GraphBuilder.PAYLOAD_DIM - len(byte_lst)), 'constant')
            else:
                feat = np.array(byte_lst[:GraphBuilder.PAYLOAD_DIM])
            packet_node_feats.append(np.abs(np.uint8(feat)).tolist())
        packet_node_feats = torch.tensor(packet_node_feats, dtype=torch.float32)

        # --- Contain edge: flow -> each packet ---
        contain_src = np.zeros(num_packets, dtype=int)
        contain_dst = np.arange(num_packets)
        contain_edge_index = torch.tensor(np.vstack([contain_src, contain_dst]), dtype=torch.int64)

        # Contain edge features: direction, ip_size, transport_size, payload_size
        contain_edge_feats = []
        for i in range(num_packets):
            contain_edge_feats.append([
                int(pkt_direction[i]) if i < len(pkt_direction) else 0,
                int(ip_size[i]) if i < len(ip_size) else 0,
                int(transport_size[i]) if i < len(transport_size) else 0,
                int(payload_size[i]) if i < len(payload_size) else 0,
            ])
        contain_edge_attr = torch.tensor(contain_edge_feats, dtype=torch.float32)

        # --- Link edge: packet_i -> packet_{i+1} ---
        if num_packets > 1:
            link_src = np.arange(num_packets - 1)
            link_dst = np.arange(1, num_packets)
            link_edge_index = torch.tensor(np.vstack([link_src, link_dst]), dtype=torch.int64)
            link_edge_attr = torch.tensor(
                [float(delta_time[i + 1]) if i + 1 < len(delta_time) else 0.0
                 for i in range(num_packets - 1)],
                dtype=torch.float32
            ).unsqueeze(-1)
        else:
            link_edge_index = torch.zeros((2, 0), dtype=torch.int64)
            link_edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        # --- Assemble HeteroData ---
        data = HeteroData()
        data['flow'].x = flow_node_feats
        data['packet'].x = packet_node_feats
        data['flow', 'contain', 'packet'].edge_index = contain_edge_index
        data['packet', 'link', 'packet'].edge_index = link_edge_index
        data['flow', 'contain', 'packet'].edge_attr = contain_edge_attr
        data['packet', 'link', 'packet'].edge_attr = link_edge_attr

        if label is not None:
            data.y = torch.tensor([label], dtype=torch.int64)

        data = T.ToUndirected()(data)
        return data


# ============================================================
# GNN4ID 分析器主类
# ============================================================

class GNN4IDAnalyzer:
    """
    GNN4ID 分析器：从PCAP提取特征，构建异构图，用GNN模型做分类推理

    支持两种模式：
    1. 完整模式：有已训练模型，做真实推理
    2. 模拟模式：无模型，基于规则做模拟检测
    """

    def __init__(self, model_path=None, device=None):
        """
        初始化分析器

        Args:
            model_path: 已训练GNN模型路径 (.pth)。None则使用模拟模式。
            device: 'cuda' 或 'cpu'。None则自动选择。
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False

        if model_path and os.path.exists(model_path):
            try:
                # 将 GNN4ID 目录加入路径，使 pickle 能找到 Utility 模块
                # 搜索多个候选路径: 项目同级目录、模型所在目录的上级
                gnn4id_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                gnn4id_candidates = [
                    os.path.join(gnn4id_root, 'GNN4ID'),
                    os.path.join(os.path.dirname(gnn4id_root), 'GNN4ID'),
                    os.path.join(os.path.dirname(gnn4id_root), 'GNN4IDplus', 'GNN4ID'),
                ]
                # 向上逐级检查模型文件所在目录树，寻找包含 Utility 的目录
                # 模型典型路径: GNN4ID/Saved_Model/GNN4ID_8_Classes/model.pth
                # Utility 所在路径: GNN4ID/Utility/
                model_dir = os.path.dirname(model_path)
                for _ in range(6):  # 向上查 6 层
                    if os.path.isdir(os.path.join(model_dir, 'Utility')):
                        if model_dir not in sys.path:
                            gnn4id_candidates.insert(0, model_dir)
                            break
                    parent = os.path.dirname(model_dir)
                    if parent == model_dir:
                        break
                    model_dir = parent
                for d in gnn4id_candidates:
                    if os.path.isdir(os.path.join(d, 'Utility')) and d not in sys.path:
                        sys.path.insert(0, d)
                        break

                # 安全加载：优先 weights_only=True，失败时给出警告并回退
                try:
                    loaded = torch.load(model_path, map_location=self.device, weights_only=True)
                except Exception:
                    print(f"[GNN4ID] 警告: 模型包含非张量数据，使用 weights_only=False 加载")
                    loaded = torch.load(model_path, map_location=self.device, weights_only=False)

                # 处理 checkpoint dict 格式（如 {"state_dict": ..., "meta": ...}）
                if isinstance(loaded, dict):
                    state_dict = loaded.get('state_dict', loaded.get('model_state_dict', loaded))
                    if not isinstance(state_dict, dict):
                        raise ValueError("无法从 checkpoint 中提取 state_dict")

                    # 构建 dummy 图用于模型实例化
                    from torch_geometric.data import HeteroData
                    dummy = HeteroData()
                    dummy['flow'].x = torch.zeros(1, 82)
                    dummy['packet'].x = torch.zeros(1, 1500)
                    dummy['flow', 'contain', 'packet'].edge_index = torch.zeros((2, 1), dtype=torch.int64)
                    dummy['packet', 'link', 'packet'].edge_index = torch.zeros((2, 0), dtype=torch.int64)
                    dummy['flow', 'contain', 'packet'].edge_attr = torch.zeros(1, 4)
                    dummy['packet', 'link', 'packet'].edge_attr = torch.zeros(0, 1)
                    dummy = T.ToUndirected()(dummy)

                    # 自动检测模型架构: HeteroGNN (SAGEConv) vs HeteroGNN_Edge (GATConv)
                    # GATConv 的 state_dict key 含 'att_src'/'att_dst'，SAGEConv 没有
                    from nemesys_gnn4id.gnn4id.model import HeteroGNN, HeteroGNN_Edge
                    is_edge_model = any('att_src' in k or 'att_dst' in k
                                        for k in state_dict.keys())
                    model_cls = HeteroGNN_Edge if is_edge_model else HeteroGNN
                    arch_name = 'HeteroGNN_Edge (GATConv)' if is_edge_model else 'HeteroGNN (SAGEConv)'

                    self.model = model_cls(dummy, GNN_DEFAULTS).to(self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"[GNN4ID] 检测到架构: {arch_name}, state_dict 加载成功")
                else:
                    self.model = loaded
                    print(f"[GNN4ID] 加载完整模型对象成功")

                self.model.eval()
                self.model_loaded = True
                print(f"[GNN4ID] 模型已加载: {model_path}")
            except Exception as e:
                print(f"[GNN4ID] 模型加载失败: {e}，使用模拟模式")
        else:
            print("[GNN4ID] 未指定模型文件，使用模拟模式")

    def extract_features(self, pcap_path, max_flows=1000):
        """
        从PCAP文件提取 flow+packet 级别特征

        Args:
            pcap_path: PCAP文件路径
            max_flows: 最大提取flow数

        Returns:
            pd.DataFrame: 包含flow和packet特征的DataFrame
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f'PCAP文件不存在: {pcap_path}')

        if FEATURE_BACKEND == 'scapy':
            _record_backend('scapy', 'forced by NEMESYS_FEATURE_BACKEND=scapy')
            return self._extract_with_scapy(pcap_path, max_flows)

        if NFSTREAM_AVAILABLE:
            try:
                df = self._extract_with_nfstream(pcap_path, max_flows)
                _record_backend('nfstream', 'ok')
                return df
            except Exception as e:
                if FEATURE_BACKEND == 'nfstream':
                    _record_backend('nfstream-failed', repr(e))
                    raise RuntimeError(
                        'nfstream extraction failed and NEMESYS_FEATURE_BACKEND=nfstream '
                        f'forbids the scapy fallback: {e!r}') from e
                print('!' * 72)
                print(f'[GNN4ID] !! nfstream extraction FAILED: {e!r}')
                print('[GNN4ID] !! FALLING BACK TO SCAPY. The two backends produce')
                print('[GNN4ID] !! DIFFERENT 82-dim features (~49% cell agreement).')
                print('[GNN4ID] !! Any result from this run MUST record the backend.')
                print('!' * 72)
                _record_backend('scapy', f'nfstream failed: {e!r}')
                return self._extract_with_scapy(pcap_path, max_flows)

        if FEATURE_BACKEND == 'nfstream':
            # Forced nfstream but it is NOT usable: the same contract as the
            # extraction-failure branch above applies -- NEMESYS_FEATURE_BACKEND=nfstream
            # forbids the scapy fallback, so degrading here would silently produce the
            # non-equivalent feature set the caller explicitly ruled out. Recorded
            # before raising so a caught exception cannot leave a stale record.
            _record_backend(
                'nfstream-failed',
                f'nfstream unavailable (importable={_NFSTREAM_IMPORTABLE}, '
                f'{_NPCAP_STATUS})')
            raise RuntimeError(
                'nfstream is unavailable (importable='
                f'{_NFSTREAM_IMPORTABLE}; {_NPCAP_STATUS}) and '
                'NEMESYS_FEATURE_BACKEND=nfstream forbids the scapy fallback. '
                'Install/repair nfstream (and Npcap), or set '
                'NEMESYS_FEATURE_BACKEND=auto|scapy explicitly.')

        _record_backend('scapy', 'nfstream unavailable')
        print('[GNN4ID] 使用 scapy 提取特征（降级模式）')
        return self._extract_with_scapy(pcap_path, max_flows)

    def _extract_with_nfstream(self, pcap_path, max_flows):
        """使用 nfstream 提取特征"""
        import tempfile, os
        # 在 NFStreamer 内部 import scapy 之前抑制其模块加载告警
        try:
            import logging
            from scapy.config import conf
            conf.verb = 0
            import scapy.error
            scapy.error.SCAPY_ROOT_LOGGER.setLevel(logging.CRITICAL + 10)
        except Exception:
            pass
        print(f"[GNN4ID] nfstream 提取特征: {pcap_path}")
        streamer = NFStreamer(
            source=pcap_path,
            accounting_mode=1,
            idle_timeout=120,
            statistical_analysis=True,
            n_dissections=0,
            udps=FlowPacketExtractor(limit=20)
        )

        # nfstream 6.5.x 在 Python 3.8 上 to_csv(path=None) 有 bug，
        # 写入临时文件再读回可绕过
        tmp_path = os.path.join(tempfile.gettempdir(), f'_nfstream_{os.getpid()}.csv')
        streamer.to_csv(path=tmp_path)
        df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)

        if df is None or len(df) == 0:
            print("[GNN4ID] 警告: 未从PCAP中提取到flow")
            return pd.DataFrame()

        # 限制flow数量
        if len(df) > max_flows:
            df = df.head(max_flows)

        print(f"[GNN4ID] 提取到 {len(df)} 条flow, {len(df.columns)} 列")

        # 后处理：按 FLOW_FEATURE_NAMES 顺序排列特征，确保 GNN 输入顺序正确
        self._add_flow_features_column(df)
        return df

    @staticmethod
    def _add_flow_features_column(df):
        """为 nfstream DataFrame 各行计算 82 维 flow_features 列，
        严格按 FLOW_FEATURE_NAMES / 训练 CSV 的特征顺序排列。"""
        import math

        def _get(row, name, default=0.0):
            val = row.get(name, default)
            if pd.isna(val):
                return default
            return float(val)

        flow_features_list = []
        for _, row in df.iterrows():
            # 解析 udps 列中的 packet 级数据（列表格式需要解析）
            def _parse_list(name):
                raw = row.get(f'udps.{name}', row.get(name, ''))
                if isinstance(raw, list):
                    return raw
                if isinstance(raw, str):
                    s = raw.strip('[]')
                    if not s:
                        return []
                    try:
                        return [int(x.strip().strip("'")) for x in s.split(',') if x.strip().strip("'")]
                    except ValueError:
                        try:
                            return [float(x.strip().strip("'")) for x in s.split(',') if x.strip().strip("'")]
                        except ValueError:
                            return s.split(',')
                if isinstance(raw, (int, float)):
                    return [float(raw)]
                return []

            pkt_sizes = _parse_list('payload_size')  # udps.payload_size
            deltas = _parse_list('delta_time')  # udps.delta_time
            syns = _parse_list('syn')  # udps.syn
            cwrs = _parse_list('cwr')
            eces = _parse_list('ece')
            urgs = _parse_list('urg')
            acks = _parse_list('ack')
            pshs = _parse_list('psh')
            rsts = _parse_list('rst')
            fins = _parse_list('fin')
            dirs = _parse_list('packet_direction')  # 0=src→dst, 1=dst→src

            def _stats(vals):
                if not vals:
                    return 0.0, 0.0, 0.0, 0.0
                mn = min(vals)
                mx = max(vals)
                avg = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((x - avg)**2 for x in vals) / (len(vals) - 1)
                    std = math.sqrt(var)
                else:
                    std = 0.0
                return mn, avg, std, mx

            # 按方向分离 packet sizes
            s2d_sizes = [pkt_sizes[i] for i in range(len(pkt_sizes)) if i < len(dirs) and dirs[i] == 0]
            d2s_sizes = [pkt_sizes[i] for i in range(len(pkt_sizes)) if i < len(dirs) and dirs[i] == 1]
            s2d_deltas = [deltas[i] for i in range(len(deltas)) if i < len(dirs) and dirs[i] == 0]
            d2s_deltas = [deltas[i] for i in range(len(deltas)) if i < len(dirs) and dirs[i] == 1]
            bi_deltas = deltas

            bi_min_ps, bi_mean_ps, bi_std_ps, bi_max_ps = _stats(pkt_sizes)
            s2d_min_ps, s2d_mean_ps, s2d_std_ps, s2d_max_ps = _stats(s2d_sizes)
            d2s_min_ps, d2s_mean_ps, d2s_std_ps, d2s_max_ps = _stats(d2s_sizes)

            bi_min_piat, bi_mean_piat, bi_std_piat, bi_max_piat = _stats(bi_deltas)
            s2d_min_piat, s2d_mean_piat, s2d_std_piat, s2d_max_piat = _stats(s2d_deltas)
            d2s_min_piat, d2s_mean_piat, d2s_std_piat, d2s_max_piat = _stats(d2s_deltas)

            # 计算 direction-specific flag count
            s2d_syn = sum(1 for i in range(min(len(syns), len(dirs))) if dirs[i] == 0 and syns[i])
            s2d_cwr = sum(1 for i in range(min(len(cwrs), len(dirs))) if dirs[i] == 0 and cwrs[i])
            s2d_ece = sum(1 for i in range(min(len(eces), len(dirs))) if dirs[i] == 0 and eces[i])
            s2d_urg = sum(1 for i in range(min(len(urgs), len(dirs))) if dirs[i] == 0 and urgs[i])
            s2d_ack = sum(1 for i in range(min(len(acks), len(dirs))) if dirs[i] == 0 and acks[i])
            s2d_psh = sum(1 for i in range(min(len(pshs), len(dirs))) if dirs[i] == 0 and pshs[i])
            s2d_rst = sum(1 for i in range(min(len(rsts), len(dirs))) if dirs[i] == 0 and rsts[i])
            s2d_fin = sum(1 for i in range(min(len(fins), len(dirs))) if dirs[i] == 0 and fins[i])

            d2s_syn = sum(1 for i in range(min(len(syns), len(dirs))) if dirs[i] == 1 and syns[i])
            d2s_cwr = sum(1 for i in range(min(len(cwrs), len(dirs))) if dirs[i] == 1 and cwrs[i])
            d2s_ece = sum(1 for i in range(min(len(eces), len(dirs))) if dirs[i] == 1 and eces[i])
            d2s_urg = sum(1 for i in range(min(len(urgs), len(dirs))) if dirs[i] == 1 and urgs[i])
            d2s_ack = sum(1 for i in range(min(len(acks), len(dirs))) if dirs[i] == 1 and acks[i])
            d2s_psh = sum(1 for i in range(min(len(pshs), len(dirs))) if dirs[i] == 1 and pshs[i])
            d2s_rst = sum(1 for i in range(min(len(rsts), len(dirs))) if dirs[i] == 1 and rsts[i])
            d2s_fin = sum(1 for i in range(min(len(fins), len(dirs))) if dirs[i] == 1 and fins[i])

            pkt_size_var = bi_std_ps / bi_mean_ps if bi_mean_ps > 0 else 0.0

            # 流级近似 rolling window 特征
            proto = int(row.get('protocol', 0))
            dport = int(row.get('dst_port', 0))
            is_udp = 1 if proto == 17 else 0
            is_tcp = 1 if proto == 6 else 0
            has_http = 1 if dport in [80, 443, 8080] else 0
            has_dns = 1 if dport == 53 or int(row.get('src_port', 0)) == 53 else 0
            has_vuln = 1 if dport in [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080] else 0
            bi_pkts = _get(row, 'bidirectional_packets')
            d2s_pkts = _get(row, 'dst2src_packets')

            rolling_features = [
                float(is_udp),                            # 47
                float(is_udp and dport == 53),             # 48
                float(is_tcp),                             # 49
                float(is_tcp),                             # 50
                float(s2d_ack / max(bi_pkts, 1)),          # 51
                float(d2s_ack / max(bi_pkts, 1)),          # 52
                float(s2d_fin / max(bi_pkts, 1)),          # 53
                float(d2s_fin / max(bi_pkts, 1)),          # 54
                float(s2d_rst / max(bi_pkts, 1)),          # 55
                float(d2s_rst / max(bi_pkts, 1)),          # 56
                float(s2d_psh / max(bi_pkts, 1)),          # 57
                float(d2s_psh / max(bi_pkts, 1)),          # 58
                float(s2d_syn / max(bi_pkts, 1)),          # 59
                float(d2s_syn / max(bi_pkts, 1)),          # 60
                1.0,                                       # 61 Unique ports (single flow: 1)
                0.0,                                       # 62 ICMP
                0.0,                                       # 63 ICMP dst
                float(has_http),                           # 64
                float(has_http),                           # 65
                1.0,                                       # 66 duration
                1.0,                                       # 67 duration sd
                float(has_dns),                            # 68
                float(has_dns),                            # 69
                float(has_dns),                            # 70
                float(has_dns),                            # 71
                float(has_vuln),                           # 72
                float(d2s_pkts / max(bi_pkts, 1)),         # 73
                1.0,                                       # 74
            ]

            # One-hot
            exp_id = int(row.get('expiration_id', 0))
            exp_0 = 1 if exp_id == 0 else 0
            exp_m1 = 1 if exp_id == -1 else 0
            proto_oh = [
                1 if proto == 1 else 0,
                1 if proto == 2 else 0,
                1 if proto == 6 else 0,
                1 if proto == 17 else 0,
                1 if proto == 58 else 0,
            ]

            # --- 组装 82 维特征向量（严格 FLOW_FEATURE_NAMES 顺序）---
            feat_82 = [
                # [0-5] 基础双向统计
                _get(row, 'src2dst_duration_ms'),          # 0
                _get(row, 'src2dst_packets'),              # 1
                _get(row, 'src2dst_bytes'),                # 2
                _get(row, 'dst2src_duration_ms'),          # 3
                _get(row, 'dst2src_packets'),              # 4
                _get(row, 'dst2src_bytes'),                # 5
                # [6-9] 双向包大小统计
                bi_min_ps, bi_mean_ps, bi_std_ps, bi_max_ps,
                # [10-13] src→dst 包大小统计
                s2d_min_ps, s2d_mean_ps, s2d_std_ps, s2d_max_ps,
                # [14-17] dst→src 包大小统计
                d2s_min_ps, d2s_mean_ps, d2s_std_ps, d2s_max_ps,
                # [18-21] 双向 PIAT
                bi_min_piat, bi_mean_piat, bi_std_piat, bi_max_piat,
                # [22-25] src→dst PIAT
                s2d_min_piat, s2d_mean_piat, s2d_std_piat, s2d_max_piat,
                # [26-29] dst→src PIAT
                d2s_min_piat, d2s_mean_piat, d2s_std_piat, d2s_max_piat,
                # [30-37] src→dst TCP flag 计数
                float(s2d_syn), float(s2d_cwr), float(s2d_ece), float(s2d_urg),
                float(s2d_ack), float(s2d_psh), float(s2d_rst), float(s2d_fin),
                # [38-45] dst→src TCP flag 计数
                float(d2s_syn), float(d2s_cwr), float(d2s_ece), float(d2s_urg),
                float(d2s_ack), float(d2s_psh), float(d2s_rst), float(d2s_fin),
                # [46] packet_size_variation
                pkt_size_var,
            ]
            feat_82.extend(rolling_features)     # 47-74
            feat_82.extend([float(exp_0), float(exp_m1)])  # 75-76
            feat_82.extend([float(x) for x in proto_oh])   # 77-81

            flow_features_list.append(feat_82)

        df['flow_features'] = flow_features_list

    def _extract_with_scapy(self, pcap_path, max_flows):
        """使用 scapy 提取特征，产出与训练 CSV 一致的 82 维特征向量"""
        from scapy.all import rdpcap, IP, TCP, UDP, ICMP

        print(f"[GNN4ID] scapy 读取PCAP: {pcap_path}")
        packets = rdpcap(pcap_path)
        print(f"[GNN4ID] 共 {len(packets)} 个数据包")

        # 按 5 元组聚合为 flow
        flows = {}
        for pkt in packets:
            if IP not in pkt:
                continue

            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto

            sport = 0
            dport = 0
            flags = 0
            pkt_len = len(pkt)
            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
                flags = int(pkt[TCP].flags)
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport

            # 双向 flow key：使用元组排序确保双向流归为同一条
            ep1 = (src_ip, sport)
            ep2 = (dst_ip, dport)
            if ep1 <= ep2:
                key = (src_ip, sport, dst_ip, dport, proto)
            else:
                key = (dst_ip, dport, src_ip, sport, proto)

            if key not in flows:
                flows[key] = {
                    'src_ip': src_ip, 'dst_ip': dst_ip,
                    'src_port': sport, 'dst_port': dport,
                    'protocol': proto,
                    # 时间戳
                    'first_seen': float(pkt.time),
                    'last_seen': float(pkt.time),
                    # 双向统计
                    'bi_packets': 0, 'bi_bytes': 0,
                    'bi_pkt_sizes': [], 'bi_delta_times': [],
                    # src→dst 统计
                    's2d_packets': 0, 's2d_bytes': 0,
                    's2d_pkt_sizes': [], 's2d_delta_times': [],
                    's2d_first_seen': float(pkt.time), 's2d_last_seen': float(pkt.time),
                    # dst→src 统计
                    'd2s_packets': 0, 'd2s_bytes': 0,
                    'd2s_pkt_sizes': [], 'd2s_delta_times': [],
                    'd2s_first_seen': float(pkt.time), 'd2s_last_seen': float(pkt.time),
                    # TCP flag 计数 (src→dst, dst→src 分开)
                    's2d_flags': [0]*8,  # syn,cwr,ece,urg,ack,psh,rst,fin
                    'd2s_flags': [0]*8,
                    # payload 和包级特征（仅存前20个包）
                    'payload_data': [], 'delta_time': [],
                    'packet_direction': [], 'ip_size': [],
                    'transport_size': [], 'payload_size': [],
                    'syn': [], 'cwr': [], 'ece': [], 'urg': [],
                    'ack': [], 'psh': [], 'rst': [], 'fin': [],
                }

            flow = flows[key]
            is_src = (src_ip == flow['src_ip'])

            # delta time — compute BEFORE updating last_seen
            if flow['bi_packets'] == 0:
                delta = 0.0
            else:
                delta = max(0, (float(pkt.time) - flow['last_seen']) * 1000)

            # Update counters
            flow['bi_packets'] += 1
            flow['bi_bytes'] += pkt_len
            flow['bi_pkt_sizes'].append(pkt_len)
            flow['bi_delta_times'].append(delta)
            flow['last_seen'] = float(pkt.time)

            if is_src:
                flow['s2d_packets'] += 1
                flow['s2d_bytes'] += pkt_len
                flow['s2d_pkt_sizes'].append(pkt_len)
                flow['s2d_delta_times'].append(delta)
                flow['s2d_last_seen'] = float(pkt.time)
                # TCP flags
                for i, mask in enumerate([0x02, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x01]):
                    if flags & mask:
                        flow['s2d_flags'][i] += 1
            else:
                flow['d2s_packets'] += 1
                flow['d2s_bytes'] += pkt_len
                flow['d2s_pkt_sizes'].append(pkt_len)
                flow['d2s_delta_times'].append(delta)
                flow['d2s_last_seen'] = float(pkt.time)
                for i, mask in enumerate([0x02, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x01]):
                    if flags & mask:
                        flow['d2s_flags'][i] += 1

            # Per-packet features (only store first 20)
            if flow['bi_packets'] <= 20:
                raw_payload = b''
                if TCP in pkt and pkt[TCP].payload:
                    raw_payload = bytes(pkt[TCP].payload)
                elif UDP in pkt and pkt[UDP].payload:
                    raw_payload = bytes(pkt[UDP].payload)
                elif pkt.payload:
                    raw_payload = bytes(pkt.payload)
                flow['payload_data'].append(raw_payload.hex() if raw_payload else '00')
                flow['payload_size'].append(len(raw_payload))
                flow['ip_size'].append(len(pkt[IP]))
                flow['transport_size'].append(pkt_len - len(pkt[IP]))
                flow['delta_time'].append(delta)
                flow['packet_direction'].append(0 if is_src else 1)
                flow['syn'].append(1 if flags & 0x02 else 0)
                flow['cwr'].append(1 if flags & 0x80 else 0)
                flow['ece'].append(1 if flags & 0x40 else 0)
                flow['urg'].append(1 if flags & 0x20 else 0)
                flow['ack'].append(1 if flags & 0x10 else 0)
                flow['psh'].append(1 if flags & 0x08 else 0)
                flow['rst'].append(1 if flags & 0x04 else 0)
                flow['fin'].append(1 if flags & 0x01 else 0)

        # 构建 DataFrame，每行产出 82 维特征 + payload 列
        rows = []
        for key, f in flows.items():
            if f['bi_packets'] < 1:
                continue

            import math

            def _stats(vals):
                """计算 min, mean, stddev, max"""
                if not vals:
                    return 0.0, 0.0, 0.0, 0.0
                mn = min(vals)
                mx = max(vals)
                avg = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((x - avg)**2 for x in vals) / (len(vals) - 1)
                    std = math.sqrt(var)
                else:
                    std = 0.0
                return mn, avg, std, mx

            bi_min_ps, bi_mean_ps, bi_std_ps, bi_max_ps = _stats(f['bi_pkt_sizes'])
            s2d_min_ps, s2d_mean_ps, s2d_std_ps, s2d_max_ps = _stats(f['s2d_pkt_sizes'])
            d2s_min_ps, d2s_mean_ps, d2s_std_ps, d2s_max_ps = _stats(f['d2s_pkt_sizes'])

            bi_min_piat, bi_mean_piat, bi_std_piat, bi_max_piat = _stats(f['bi_delta_times'])
            s2d_min_piat, s2d_mean_piat, s2d_std_piat, s2d_max_piat = _stats(f['s2d_delta_times'])
            d2s_min_piat, d2s_mean_piat, d2s_std_piat, d2s_max_piat = _stats(f['d2s_delta_times'])

            s2d_duration = max(0, (f['s2d_last_seen'] - f['s2d_first_seen']) * 1000) if f['s2d_packets'] > 1 else 0.0
            d2s_duration = max(0, (f['d2s_last_seen'] - f['d2s_first_seen']) * 1000) if f['d2s_packets'] > 1 else 0.0

            # packet_size_variation = stddev / mean (coefficient of variation)
            pkt_size_var = bi_std_ps / bi_mean_ps if bi_mean_ps > 0 else 0.0

            # One-hot: expiration_id 默认 0, protocol
            exp_0 = 1
            exp_m1 = 0
            proto = f['protocol']
            proto_oh = [0, 0, 0, 0, 0]  # proto_1, proto_2, proto_6, proto_17, proto_58
            proto_map = {1: 0, 2: 1, 6: 2, 17: 3, 58: 4}
            if proto in proto_map:
                proto_oh[proto_map[proto]] = 1

            # 滚动窗口特征：用流级统计近似（单流无法做跨流滚动）
            # 用流内可用信息填充，避免全零导致模型失效
            s2d_flags = f['s2d_flags']  # [syn,cwr,ece,urg,ack,psh,rst,fin]
            d2s_flags = f['d2s_flags']

            # 判断协议类型
            is_tcp = (proto == 6)
            is_udp = (proto == 17)
            is_icmp = (proto == 1)

            # 使用 flow 的端口（不是循环变量 sport/dport，那是最后一个包的值）
            flow_sport = f['src_port']
            flow_dport = f['dst_port']

            # HTTP 端口判断
            is_http = (flow_sport in (80, 8080) or flow_dport in (80, 8080))
            # DNS 端口判断
            is_dns = (flow_sport == 53 or flow_dport == 53)
            # 已知漏洞端口（简化列表）
            vuln_ports = {21, 23, 25, 135, 139, 445, 1433, 3306, 3389, 5900}
            has_vuln_port = (flow_sport in vuln_ports or flow_dport in vuln_ports)

            rolling_features = [
                float(f['bi_packets'] if is_udp else 0),    # 47: Rolling_UDP_Requests_SD
                float(f['d2s_packets'] if is_udp else 0),   # 48: Rolling_UDP_Requests_D
                float(f['bi_packets'] if is_tcp else 0),    # 49: Rolling_TCP_Requests_SD
                float(f['d2s_packets'] if is_tcp else 0),   # 50: Rolling_TCP_Requests_D
                float(s2d_flags[4] + d2s_flags[4]),         # 51: Rolling_ACK_Packets_SD (双向 ACK)
                float(d2s_flags[4]),                        # 52: Rolling_ACK_Packets_D
                float(s2d_flags[7] + d2s_flags[7]),         # 53: Rolling_FIN_Packets_SD
                float(d2s_flags[7]),                        # 54: Rolling_FIN_Packets_D
                float(s2d_flags[6] + d2s_flags[6]),         # 55: Rolling_rst_Packets_SD
                float(d2s_flags[6]),                        # 56: Rolling_rst_Packets_D
                float(s2d_flags[5] + d2s_flags[5]),         # 57: Rolling_psh_Packets_SD
                float(d2s_flags[5]),                        # 58: Rolling_psh_Packets_D
                float(s2d_flags[0] + d2s_flags[0]),         # 59: Rolling_SYN_Packets_SD
                float(d2s_flags[0]),                        # 60: Rolling_SYN_Packets_D
                1.0,                                        # 61: Unique_Ports_In_SD (单流=1)
                float(f['bi_packets'] if is_icmp else 0),   # 62: Rolling_ICMP_Requests_SD
                float(f['d2s_packets'] if is_icmp else 0),  # 63: Rolling_ICMP_Requests_D
                float(1 if is_http else 0),                 # 64: Rolling_http_port_SD
                float(1 if is_http and dport in (80, 8080) else 0),  # 65: Rolling_http_port_D
                d2s_duration,                               # 66: Rolling_Duration_D
                s2d_duration + d2s_duration,                # 67: Rolling_Duration_SD
                float(1 if is_dns else 0),                  # 68: Rolling_DNS_request_SD
                float(1 if is_dns and dport == 53 else 0),  # 69: Rolling_DNS_request_D
                float(1 if is_dns else 0),                  # 70: Rolling_DNS_request_SD2
                float(1 if is_dns and dport == 53 else 0),  # 71: Rolling_DNS_request_D2
                float(1 if has_vuln_port else 0),           # 72: Rolling_vulnerable_port
                float(f['d2s_packets']),                    # 73: Rolling_packets_destination
                float(f['bi_packets']),                     # 74: Rolling_bipackets_destination
            ]

            # 组装 82 维特征向量（严格按训练 CSV 列顺序）
            feat_82 = [
                s2d_duration,                    # 0: src2dst_duration_ms
                float(f['s2d_packets']),         # 1: src2dst_packets
                float(f['s2d_bytes']),           # 2: src2dst_bytes
                d2s_duration,                    # 3: dst2src_duration_ms
                float(f['d2s_packets']),         # 4: dst2src_packets
                float(f['d2s_bytes']),           # 5: dst2src_bytes
                bi_min_ps,                       # 6: bidirectional_min_ps
                bi_mean_ps,                      # 7: bidirectional_mean_ps
                bi_std_ps,                       # 8: bidirectional_stddev_ps
                bi_max_ps,                       # 9: bidirectional_max_ps
                s2d_min_ps,                      # 10: src2dst_min_ps
                s2d_mean_ps,                     # 11: src2dst_mean_ps
                s2d_std_ps,                      # 12: src2dst_stddev_ps
                s2d_max_ps,                      # 13: src2dst_max_ps
                d2s_min_ps,                      # 14: dst2src_min_ps
                d2s_mean_ps,                     # 15: dst2src_mean_ps
                d2s_std_ps,                      # 16: dst2src_stddev_ps
                d2s_max_ps,                      # 17: dst2src_max_ps
                bi_min_piat,                     # 18: bidirectional_min_piat_ms
                bi_mean_piat,                    # 19: bidirectional_mean_piat_ms
                bi_std_piat,                     # 20: bidirectional_stddev_piat_ms
                bi_max_piat,                     # 21: bidirectional_max_piat_ms
                s2d_min_piat,                    # 22: src2dst_min_piat_ms
                s2d_mean_piat,                   # 23: src2dst_mean_piat_ms
                s2d_std_piat,                    # 24: src2dst_stddev_piat_ms
                s2d_max_piat,                    # 25: src2dst_max_piat_ms
                d2s_min_piat,                    # 26: dst2src_min_piat_ms
                d2s_mean_piat,                   # 27: dst2src_mean_piat_ms
                d2s_std_piat,                    # 28: dst2src_stddev_piat_ms
                d2s_max_piat,                    # 29: dst2src_max_piat_ms
                float(f['s2d_flags'][0]),        # 30: src2dst_syn_packets
                float(f['s2d_flags'][1]),        # 31: src2dst_cwr_packets
                float(f['s2d_flags'][2]),        # 32: src2dst_ece_packets
                float(f['s2d_flags'][3]),        # 33: src2dst_urg_packets
                float(f['s2d_flags'][4]),        # 34: src2dst_ack_packets
                float(f['s2d_flags'][5]),        # 35: src2dst_psh_packets
                float(f['s2d_flags'][6]),        # 36: src2dst_rst_packets
                float(f['s2d_flags'][7]),        # 37: src2dst_fin_packets
                float(f['d2s_flags'][0]),        # 38: dst2src_syn_packets
                float(f['d2s_flags'][1]),        # 39: dst2src_cwr_packets
                float(f['d2s_flags'][2]),        # 40: dst2src_ece_packets
                float(f['d2s_flags'][3]),        # 41: dst2src_urg_packets
                float(f['d2s_flags'][4]),        # 42: dst2src_ack_packets
                float(f['d2s_flags'][5]),        # 43: dst2src_psh_packets
                float(f['d2s_flags'][6]),        # 44: dst2src_rst_packets
                float(f['d2s_flags'][7]),        # 45: dst2src_fin_packets
                pkt_size_var,                    # 46: packet_size_variation
            ]
            feat_82.extend(rolling_features)     # 47-74: rolling features (流级近似)
            feat_82.extend([float(exp_0), float(exp_m1)])  # 75-76: Exp_0, Exp_-1
            feat_82.extend([float(x) for x in proto_oh])   # 77-81: proto one-hot

            row = {
                'flow_features': feat_82,
                'src_ip': f['src_ip'], 'dst_ip': f['dst_ip'],
                'src_port': f['src_port'], 'dst_port': f['dst_port'],
                'protocol': f['protocol'],
                'bidirectional_packets': f['bi_packets'],
                'payload_data': str(f['payload_data']),
                'delta_time': str(f['delta_time']),
                'packet_direction': str(f['packet_direction']),
                'ip_size': str(f['ip_size']),
                'transport_size': str(f['transport_size']),
                'payload_size': str(f['payload_size']),
                'syn': str(f['syn']), 'cwr': str(f['cwr']),
                'ece': str(f['ece']), 'urg': str(f['urg']),
                'ack': str(f['ack']), 'psh': str(f['psh']),
                'rst': str(f['rst']), 'fin': str(f['fin']),
            }
            rows.append(row)

        if not rows:
            print("[GNN4ID] 警告: 未从PCAP中提取到有效flow")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if len(df) > max_flows:
            df = df.head(max_flows)

        print(f"[GNN4ID] scapy 提取到 {len(df)} 条flow")
        return df

    def build_graphs(self, df, expected_flow_dim=None):
        """
        将特征DataFrame转换为HeteroData图列表

        Args:
            df: extract_features 返回的DataFrame
            expected_flow_dim: 流节点特征维度。None则使用自然维度。

        Returns:
            list[HeteroData]: 图对象列表
        """
        graphs = []
        for idx, row in df.iterrows():
            try:
                graph = GraphBuilder.build_single_graph(row, expected_flow_dim=expected_flow_dim)
                graphs.append(graph)
            except Exception as e:
                print(f"[GNN4ID] 构建图 {idx} 失败: {e}")
                continue
        print(f"[GNN4ID] 构建了 {len(graphs)} 个图对象")
        return graphs

    def predict(self, graphs):
        """
        对图列表做分类推理

        Args:
            graphs: HeteroData 图列表

        Returns:
            list[dict]: 每个图的分类结果
            [{'class_id': int, 'class_name': str, 'confidence': float, 'level': str}, ...]
        """
        results = []
        for i, graph in enumerate(graphs):
            if self.model_loaded:
                result = self._predict_with_model(graph)
            else:
                result = self._predict_simulated(graph)
            result['flow_index'] = i
            results.append(result)
        return results

    def _predict_with_model(self, graph):
        """使用GNN模型做真实推理"""
        if not hasattr(self, '_model_error_printed'):
            self._model_error_printed = False
        try:
            from torch_geometric.data import Batch
            graph = graph.to(self.device)
            # 将单图包装为 batch（模型需要 batch.batch_dict）
            batch = Batch.from_data_list([graph])

            with torch.no_grad():
                # 尝试 HeteroGNN (无edge attr)
                try:
                    output = self.model(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch
                    )
                except TypeError:
                    # 尝试 HeteroGNN_Edge (有edge attr)
                    output = self.model(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch.edge_attr_dict,
                        batch
                    )

                probs = F.softmax(output, dim=1)
                conf, pred = probs.max(dim=1)
                class_id = pred.item()
                confidence = conf.item()
                # P(Benign) of this flow's softmax. Added for the ablation's continuous
                # score (gnn_prob_attack = 1 - prob_benign); every pre-existing key and
                # value below is unchanged, and nothing in the pipeline reads this back
                # into a decision.
                prob_benign = float(probs[0, 0].item())

            return {
                'class_id': class_id,
                'class_name': ATTACK_TYPES.get(class_id, {}).get('name', '未知'),
                'class_name_en': ATTACK_TYPES.get(class_id, {}).get('name_en', 'Unknown'),
                'confidence': confidence,
                'level': ATTACK_TYPES.get(class_id, {}).get('level', 'none'),
                'description': ATTACK_TYPES.get(class_id, {}).get('description', ''),
                'prob_benign': prob_benign,
            }
        except Exception as e:
            if not self._model_error_printed:
                print(f"[GNN4ID] 模型推理失败: {e}，后续使用模拟模式")
                self._model_error_printed = True
            return self._predict_simulated(graph)

    def _predict_simulated(self, graph):
        """模拟模式：基于流量特征的启发式规则做分类"""
        try:
            flow_feats = graph['flow'].x[0].cpu().numpy()
            num_packets = graph['packet'].x.shape[0]

            # Extract available features from flow vector
            total_bytes = float(flow_feats[2]) if len(flow_feats) > 2 else 0
            duration = float(flow_feats[0]) if len(flow_feats) > 0 else 0

            # Actual payload sizes from contain edge features (col index 3 = payload_size)
            contain_attr = graph['flow', 'contain', 'packet'].edge_attr
            if contain_attr is not None and contain_attr.shape[0] > 0:
                payload_sizes = contain_attr[:, 3].cpu().numpy()
                directions = contain_attr[:, 0].cpu().numpy()
                fwd_ratio = float(np.sum(directions == 0)) / max(len(directions), 1)
                size_mean = float(np.mean(payload_sizes))
                size_std = float(np.std(payload_sizes))
                non_zero_payloads = int(np.sum(payload_sizes > 0))
            else:
                payload_sizes = np.array([0])
                fwd_ratio = 1.0
                size_mean = 0
                size_std = 0
                non_zero_payloads = 0

        except Exception:
            num_packets = 1
            total_bytes = 0
            duration = 0
            size_std = 0
            size_mean = 0
            fwd_ratio = 1.0
            non_zero_payloads = 0

        # ---- Heuristic classification rules ----
        class_id = 0  # default: Benign
        conf = 0.70

        # Rule 1: Very short flows with few packets → likely scan/recon
        if num_packets <= 3 and duration < 5000:
            if size_mean < 100:
                class_id = 3  # Recon (SYN scan pattern)
                conf = 0.75
            else:
                class_id = 0  # Benign (short handshake)
                conf = 0.65

        # Rule 2: Many identical small packets → DoS/DDoS flood
        elif num_packets > 50 and size_std < 10 and size_mean < 200:
            if num_packets > 200:
                class_id = 6  # DDoS
                conf = min(0.90, 0.70 + num_packets / 1000)
            else:
                class_id = 5  # DoS
                conf = min(0.85, 0.65 + num_packets / 500)

        # Rule 3: Unidirectional flood (mostly one direction, many small packets)
        elif num_packets > 20 and (fwd_ratio > 0.95 or fwd_ratio < 0.05) and size_mean < 300:
            class_id = 5  # DoS
            conf = min(0.85, 0.60 + num_packets / 300)

        # Rule 4: Many small packets with no/low payload → port scan
        elif num_packets > 10 and non_zero_payloads < num_packets * 0.3:
            class_id = 3  # Recon
            conf = 0.70

        # Rule 5: High-entropy large payloads → possible web attack or exfiltration
        elif size_mean > 1000 and num_packets > 5:
            class_id = 1  # WebBased
            conf = 0.60

        # Rule 6: Brute force pattern: many short exchanges, moderate packet count
        elif num_packets > 15 and num_packets < 100 and 0.3 < fwd_ratio < 0.7 and size_mean < 500:
            # Could be brute force login attempts
            class_id = 7  # BruteForce
            conf = 0.55

        # Rule 7: Moderate traffic with balanced bidirectional → likely benign
        elif 0.3 < fwd_ratio < 0.7 and num_packets > 5:
            class_id = 0  # Benign
            conf = 0.80

        # Default: moderate confidence benign
        else:
            class_id = 0  # Benign
            conf = 0.65

        return {
            'class_id': class_id,
            'class_name': ATTACK_TYPES[class_id]['name'],
            'class_name_en': ATTACK_TYPES[class_id]['name_en'],
            'confidence': round(conf, 3),
            'level': ATTACK_TYPES[class_id]['level'],
            'description': ATTACK_TYPES[class_id]['description'],
        }

    def analyze_pcap(self, pcap_path, max_flows=1000):
        """
        一站式分析：PCAP → 特征提取 → 图构建 → 分类推理

        Args:
            pcap_path: PCAP文件路径
            max_flows: 最大flow数

        Returns:
            list[dict]: 分类结果列表
        """
        df = self.extract_features(pcap_path, max_flows)
        if df.empty:
            return []
        graphs = self.build_graphs(df, expected_flow_dim=82)
        return self.predict(graphs)

    def analyze_pcap_with_metadata(self, pcap_path, max_flows=1000):
        """
        一站式分析，同时返回flow元数据（5-tuple）用于下游关联

        Args:
            pcap_path: PCAP文件路径
            max_flows: 最大flow数

        Returns:
            tuple: (results, flow_metadata)
                results: list[dict] 分类结果
                flow_metadata: list[dict] 每条flow的 {src_ip, dst_ip, src_port, dst_port, protocol}
        """
        df = self.extract_features(pcap_path, max_flows)
        if df.empty:
            return [], []

        # Extract flow metadata before building graphs
        flow_metadata = []
        for _, row in df.iterrows():
            flow_metadata.append({
                'src_ip': str(row.get('src_ip', '')),
                'dst_ip': str(row.get('dst_ip', '')),
                'src_port': int(row.get('src_port', 0)),
                'dst_port': int(row.get('dst_port', 0)),
                'protocol': int(row.get('protocol', 0)),
            })

        # Always use 82-dim for model mode; for simulated, use 82 too (pre-computed)
        graphs = self.build_graphs(df, expected_flow_dim=82)
        results = self.predict(graphs)
        return results, flow_metadata

    def analyze_pcap_filtered(self, pcap_path, max_flows=1000, include_flow_keys=None):
        """
        Analyze PCAP but only run GNN inference for flows matching include_flow_keys.

        Flow metadata is returned for ALL flows (for downstream matching with clustering),
        but graph building and inference is skipped for non-included flows (flagged gnn_skipped=True).

        Args:
            pcap_path: PCAP file path
            max_flows: max flows to extract
            include_flow_keys: set of canonical flow tuples to include in GNN inference.
                               None means include all (same as analyze_pcap_with_metadata).

        Returns:
            tuple: (results, flow_metadata)
                results: list[dict] with gnn_skipped=True for excluded flows
                flow_metadata: list[dict] for ALL extracted flows
        """
        from nemesys_gnn4id.pipeline import canonical_flow_tuple

        df = self.extract_features(pcap_path, max_flows)
        if df.empty:
            return [], []

        # Extract flow metadata for ALL flows
        flow_metadata = []
        for _, row in df.iterrows():
            flow_metadata.append({
                'src_ip': str(row.get('src_ip', '')),
                'dst_ip': str(row.get('dst_ip', '')),
                'src_port': int(row.get('src_port', 0)),
                'dst_port': int(row.get('dst_port', 0)),
                'protocol': int(row.get('protocol', 0)),
            })

        # Build canonical keys for filtering
        if include_flow_keys is not None:
            filtered_indices = set()
            for i, meta in enumerate(flow_metadata):
                key = canonical_flow_tuple(
                    meta['src_ip'], meta['dst_ip'],
                    meta['src_port'], meta['dst_port'],
                    meta['protocol'],
                )
                if key in include_flow_keys:
                    filtered_indices.add(i)
        else:
            filtered_indices = set(range(len(df)))

        # Build graphs and run inference only for selected flows
        all_results = []
        for i in range(len(df)):
            if include_flow_keys is not None and i not in filtered_indices:
                all_results.append({
                    'flow_index': i,
                    'class_id': 0,
                    'class_name': '跳过(OOD)',
                    'class_name_en': 'Skipped(OOD)',
                    'confidence': 0.0,
                    'level': 'none',
                    'description': '域外流，跳过GNN推理',
                    'gnn_skipped': True,
                })
            else:
                row = df.iloc[i]
                try:
                    graph = GraphBuilder.build_single_graph(row, expected_flow_dim=82)
                    result = self.predict([graph])[0]
                    result['flow_index'] = i  # override: predict() always sets 0 for single-graph batch
                    result['gnn_skipped'] = False
                    all_results.append(result)
                except Exception as e:
                    all_results.append({
                        'flow_index': i,
                        'class_id': 0,
                        'class_name': '错误',
                        'class_name_en': 'Error',
                        'confidence': 0.0,
                        'level': 'none',
                        'description': f'GNN推理失败: {e}',
                        'gnn_skipped': True,
                    })

        return all_results, flow_metadata
