"""
配置文件：攻击类型映射、模型参数、NEMESYS参数、特征定义
"""

import os
from pathlib import Path


# ============================================================
# 82 维 flow 特征定义（与训练 CSV 列顺序一致）
# ============================================================
# 训练 CSV 有 97 列，去掉 14 个 udps.* 列和 Label 后剩 82 维
# 顺序必须与训练时完全一致，否则模型输出无意义

FLOW_FEATURE_NAMES = [
    # [0-5] 基础双向统计
    'src2dst_duration_ms',        # 0
    'src2dst_packets',            # 1
    'src2dst_bytes',              # 2
    'dst2src_duration_ms',        # 3
    'dst2src_packets',            # 4
    'dst2src_bytes',              # 5
    # [6-9] 双向包大小统计
    'bidirectional_min_ps',       # 6
    'bidirectional_mean_ps',      # 7
    'bidirectional_stddev_ps',    # 8
    'bidirectional_max_ps',       # 9
    # [10-13] src→dst 包大小统计
    'src2dst_min_ps',             # 10
    'src2dst_mean_ps',            # 11
    'src2dst_stddev_ps',          # 12
    'src2dst_max_ps',             # 13
    # [14-17] dst→src 包大小统计
    'dst2src_min_ps',             # 14
    'dst2src_mean_ps',            # 15
    'dst2src_stddev_ps',          # 16
    'dst2src_max_ps',             # 17
    # [18-21] 双向 PIAT (Packet Inter-Arrival Time) 统计
    'bidirectional_min_piat_ms',  # 18
    'bidirectional_mean_piat_ms', # 19
    'bidirectional_stddev_piat_ms', # 20
    'bidirectional_max_piat_ms',  # 21
    # [22-25] src→dst PIAT 统计
    'src2dst_min_piat_ms',        # 22
    'src2dst_mean_piat_ms',       # 23
    'src2dst_stddev_piat_ms',     # 24
    'src2dst_max_piat_ms',        # 25
    # [26-29] dst→src PIAT 统计
    'dst2src_min_piat_ms',        # 26
    'dst2src_mean_piat_ms',       # 27
    'dst2src_stddev_piat_ms',     # 28
    'dst2src_max_piat_ms',        # 29
    # [30-37] src→dst TCP flag 计数
    'src2dst_syn_packets',        # 30
    'src2dst_cwr_packets',        # 31
    'src2dst_ece_packets',        # 32
    'src2dst_urg_packets',        # 33
    'src2dst_ack_packets',        # 34
    'src2dst_psh_packets',        # 35
    'src2dst_rst_packets',        # 36
    'src2dst_fin_packets',        # 37
    # [38-45] dst→src TCP flag 计数
    'dst2src_syn_packets',        # 38
    'dst2src_cwr_packets',        # 39
    'dst2src_ece_packets',        # 40
    'dst2src_urg_packets',        # 41
    'dst2src_ack_packets',        # 42
    'dst2src_psh_packets',        # 43
    'dst2src_rst_packets',        # 44
    'dst2src_fin_packets',        # 45
    # [46] 包大小变异系数
    'packet_size_variation',      # 46
    # [47-74] 滚动窗口特征（单流无法计算，用流级近似）
    'Rolling_UDP_Requests_SourceDestination',  # 47
    'Rolling_UDP_Requests_Destination',        # 48
    'Rolling_TCP_Requests_SourceDestination',  # 49
    'Rolling_TCP_Requests_Destination',        # 50
    'Rolling_ACK_Packets_SourceDestination',   # 51
    'Rolling_ACK_Packets_Destination',         # 52
    'Rolling_FIN_Packets_SourceDestination',   # 53
    'Rolling_FIN_Packets_Destination',         # 54
    'Rolling_rst_Packets_SourceDestination',   # 55
    'Rolling_rst_Packets_Destination',         # 56
    'Rolling_psh_Packets_SourceDestination',   # 57
    'Rolling_psh_Packets_Destination',         # 58
    'Rolling_SYN_Packets_SourceDestination',   # 59
    'Rolling_SYN_Packets_Destination',         # 60
    'Unique_Ports_In_SourceDestinationIP',     # 61
    'Rolling_ICMP_Requests_SourceDestination', # 62
    'Rolling_ICMP_Requests_Destination',       # 63
    'Rolling_http_port_SourceDestination',     # 64
    'Rolling_http_port_Destination',           # 65
    'Rolling_Duration_Destination',            # 66
    'Rolling_Duration_SourceDestination',      # 67
    'Rolling_DNS_request_SourceDestination',   # 68
    'Rolling_DNS_request_Destination',         # 69
    'Rolling_DNS_request_SourceDestination2',  # 70
    'Rolling_DNS_request_Destination2',        # 71
    'Rolling_vulnerable_port',                 # 72
    'Rolling_packets_destination',             # 73
    'Rolling_bipackets_destination',           # 74
    # [75-81] One-hot 编码
    'Exp_0',                      # 75
    'Exp_-1',                     # 76
    'proto_1',                    # 77
    'proto_2',                    # 78
    'proto_6',                    # 79
    'proto_17',                   # 80
    'proto_58',                   # 81
]

# Protocol one-hot 编码映射
PROTO_ONEHOT = {1: 77, 2: 78, 6: 79, 17: 80, 58: 81}

# ============================================================
# 攻击类型映射 (复用 GNN4ID 的 8 类分类)
# ============================================================
ATTACK_TYPES = {
    0: {'name': '正常流量', 'name_en': 'Benign', 'level': 'none', 'description': '合法的网络流量'},
    1: {'name': 'Web攻击', 'name_en': 'WebBased', 'level': 'high', 'description': 'SQL注入、XSS等Web应用攻击'},
    2: {'name': '欺骗攻击', 'name_en': 'Spoofing', 'level': 'medium', 'description': 'ARP欺骗、IP欺骗等'},
    3: {'name': '侦察攻击', 'name_en': 'Recon', 'level': 'medium', 'description': '端口扫描、网络踩点'},
    4: {'name': 'Mirai', 'name_en': 'Mirai', 'level': 'high', 'description': 'IoT僵尸网络攻击'},
    5: {'name': 'DoS攻击', 'name_en': 'Dos', 'level': 'high', 'description': '拒绝服务攻击'},
    6: {'name': 'DDoS攻击', 'name_en': 'DDos', 'level': 'high', 'description': '分布式拒绝服务攻击'},
    7: {'name': '暴力破解', 'name_en': 'BruteForce', 'level': 'medium', 'description': '密码暴力猜测攻击'},
}

# GNN4ID 标签字典 (与训练时一致)
LABEL_DICT = {
    'Benign': 0,
    'WebBased': 1,
    'Spoofing': 2,
    'Recon': 3,
    'Mirai': 4,
    'Dos': 5,
    'DDos': 6,
    'BruteForce': 7,
}

# ============================================================
# GNN 模型参数
# ============================================================
GNN_DEFAULTS = {
    'hidden_size': 64,
    'attn_size': 32,
    'eps': 1.0,
}

# 默认模型路径
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL_CANDIDATES = [
    # models/model.pth (copied into project)
    REPO_ROOT / 'models' / 'model.pth',
]
DEFAULT_MODEL_PATH = next(
    (path for path in DEFAULT_MODEL_CANDIDATES if path.exists()),
    DEFAULT_MODEL_CANDIDATES[0],
)

DEFAULT_PROTO_MODEL_PATH = REPO_ROOT / 'field_proto_model.pth'

# ============================================================
# NEMESYS 参数
# ============================================================
NEMESYS_DEFAULTS = {
    'sigma': 0.6,
    'layer': 2,
    'relative_to_ip': True,
}

# ============================================================
# 分析参数
# ============================================================
# 低置信度警告阈值：confidence < 此值时标记为"低置信度"
# 不再影响正常/攻击分类决策，仅作为可信度提示
CONFIDENCE_THRESHOLD = 0.3

# 需要 NEMESYS 深度分析的攻击等级
DEEP_ANALYSIS_LEVELS = {'high', 'medium'}

# 风险等级中文映射 (共享常量)
RISK_LEVEL_ZH = {'high': '高危', 'medium': '中危', 'low': '低危', 'none': '安全'}

# ============================================================
# 未知协议攻击检测配置
# ============================================================
UNKNOWN_PROTO_CONFIG = {
    # 已知正常协议白名单
    'known_normal_protocols': {
        'HTTP', 'TLS', 'DNS', 'SSH', 'NTP', 'DHCP', 'SMTP', 'POP3', 'IMAP',
        'FTP', 'FTP-data', 'Telnet', 'MySQL', 'PostgreSQL', 'Redis',
        'MQTT', 'MQTTS', 'SNMP', 'SNMP-trap', 'RDP', 'VNC', 'mDNS',
        'SIP', 'SIPS', 'IRC', 'RTSP',
        'Modbus', 'DNP3', 'S7Comm', 'EtherNet/IP',  # 工控协议
    },
    # 可疑协议端口 (常被恶意软件使用)
    # 注意: 不要将 known_normal_protocols 中已有的协议端口加入此列表
    'suspicious_ports': {
        4444: 'Metasploit', 5555: 'Android Debug', 8443: 'Alt-HTTPS',
        9001: 'Tor', 9030: 'Tor', 12345: 'NetBus', 31337: 'BackOrifice',
        6697: 'IRC-TLS',
        1080: 'SOCKS', 2082: 'cPanel', 2083: 'cPanel-TLS',
        4443: 'Alt-HTTPS', 5901: 'VNC-1',
        6666: 'IRC-Alt', 7001: 'WebLogic',
        9200: 'Elasticsearch', 27017: 'MongoDB',
    },
    # 高熵阈值 (加密/压缩数据的特征)
    'high_entropy_threshold': 7.0,
    # 未知协议攻击判定规则阈值
    'attack_indicators': {
        'high_binary_ratio': 0.7,      # 字段中二进制类型占比
        'high_entropy': 7.0,           # payload 字节熵阈值
        'anomalous_payload_max': 1400, # 超大 payload 阈值 (bytes)
        'anomalous_payload_min': 4,    # 超小 payload 阈值 (bytes, 非空)
    },
    # 风险等级阈值
    'risk_thresholds': {
        'high': 3,    # 命中 3+ 个指标 → 高危
        'medium': 2,  # 命中 2 个指标 → 中危
        'low': 1,     # 命中 1 个指标 → 低危
    },
}

# ============================================================
# 新增：训练参数默认值
# ============================================================
TRAIN_DEFAULTS = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'weight_decay': 5e-4,
    'dropout': 0.3,
}

# ============================================================
# 新增：评估参数默认值
# ============================================================
EVAL_DEFAULTS = {
    'batch_size': 32,
    'proto_threshold': 0.5,
}

# ============================================================
# 新增：环境变量覆写
# ============================================================
_ENV_OVERRIDES = {
    'DEFAULT_MODEL_PATH': 'GNN4ID_MODEL_PATH',
    'DEFAULT_PROTO_MODEL_PATH': 'NEMESYS_PROTO_MODEL_PATH',
}

def from_env(key: str, default=None):
    """从环境变量读取配置，未设置时返回默认值"""
    env_var = _ENV_OVERRIDES.get(key)
    if env_var and env_var in os.environ:
        return os.environ[env_var]
    return default
