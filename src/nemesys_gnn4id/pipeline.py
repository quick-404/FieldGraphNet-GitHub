"""
主流水线：NEMESYS 协议逆向 → 域路由 → GNN4ID 攻击检测（域内）+ 结构聚类（域外）

架构流程：
1. NEMESYS BCDG 字段结构分析（对所有流量做 BCDG 分段）
2. FieldProtoGNN 字段图协议识别
3. 域路由器：将流分为域内（GNN 训练域）和域外
4. GNN4ID（域内过滤模式）：只对域内流做攻击检测
5. 结构聚类（域外）：只对域外流做结构指纹聚类
6. 融合决策：合并域内 GNN 结果 + 域外聚类结果
7. 时序异常检测 + 交叉验证
"""

import os
import time
import math
from datetime import datetime
from collections import Counter

from nemesys_gnn4id.config import ATTACK_TYPES, CONFIDENCE_THRESHOLD, DEEP_ANALYSIS_LEVELS, UNKNOWN_PROTO_CONFIG, RISK_LEVEL_ZH
from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer, UnknownProtocolDetector
from nemesys_gnn4id.nemesys.cluster import ClusterAnalyzer, is_in_gnn_domain, classify_flow_trust, route_flows_by_protocol


FUSION_ALERT_STATUSES = {
    'confirmed_gnn_attack',
    'suspicious_unknown_protocol_attack',
    'possible_missed_unknown_protocol_attack',
}

# Diagnostic switch for the per-flow ablation dump. DEFAULT OFF: with the variable
# unset (or anything but '1') this module behaves exactly as before -- no extra key is
# attached to any result dict and no extra work is done. It only ever ADDS a
# diagnostic dict; it never feeds back into a decision.
ABLATION_DUMP_ENV = 'NEMESYS_ABLATION_DUMP'

# The structural ("flag AND non-tight cohesion") branch of _cluster_is_high_risk.
# Duplicated here (not read from the method) on purpose: the diagnostic must be able
# to report this branch's incidence WITHOUT the decision logic being refactored.
_CLUSTER_STRUCTURAL_HIGH_RISK_FLAGS = {
    '未知协议', '孤立小簇', '高离群率', '高熵(加密/混淆)',
}


def ablation_dump_enabled():
    """True only when NEMESYS_ABLATION_DUMP=1. Default off (see ABLATION_DUMP_ENV)."""
    return os.environ.get(ABLATION_DUMP_ENV, '') == '1'


def canonical_flow_tuple(src, dst, sport, dport, proto):
    """Build a direction-insensitive flow key."""
    sport = int(sport or 0)
    dport = int(dport or 0)
    proto = int(proto or 0)
    left = (str(src), sport)
    right = (str(dst), dport)
    if left <= right:
        return left + right + (proto,)
    return right + left + (proto,)


def flow_meta_key(meta):
    """Return a canonical flow key for GNN4ID flow metadata."""
    return canonical_flow_tuple(
        meta.get('src_ip', ''),
        meta.get('dst_ip', ''),
        meta.get('src_port', 0),
        meta.get('dst_port', 0),
        meta.get('protocol', 0),
    )


def segment_flow_key(seg):
    """Return a canonical flow key for a NEMESYS segment dict, or None."""
    if not isinstance(seg, dict):
        return None
    if 'src' not in seg or 'dst' not in seg:
        return None
    return canonical_flow_tuple(
        seg.get('src', ''),
        seg.get('dst', ''),
        seg.get('sport', 0),
        seg.get('dport', 0),
        seg.get('proto', 0),
    )


def _segments_to_dicts(segments):
    """
    将 NEMESYS 完整 BCDG 模式返回的 List[List[MessageSegment]]
    转为 dict 格式，供聚类和交叉验证使用。
    如果已经是 dict 格式，直接返回不变。
    """
    if not segments:
        return segments
    if isinstance(segments[0], dict):
        return segments  # 已经是 dict 格式

    dicts = []
    for msg_segs in segments:
        if not isinstance(msg_segs, (list, tuple)) or not msg_segs:
            continue

        # 从 MessageSegment 提取字段信息
        field_types = []
        payload_bytes = b''
        for ms in msg_segs:
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

        # 提取消息级元数据（从第一个 segment 的 .message 属性）
        src = dst = ''
        sport = dport = proto = 0
        first_seg = msg_segs[0]
        msg_obj = getattr(first_seg, 'message', None)
        if msg_obj is not None:
            src = str(getattr(msg_obj, 'source', ''))
            dst = str(getattr(msg_obj, 'destination', ''))
            # 尝试从深层属性提取 IP/端口
            l3 = getattr(msg_obj, 'payload', None) or getattr(msg_obj, 'l3', None)
            if l3 is not None:
                proto = int(getattr(l3, 'proto', getattr(l3, 'protocol', 0)))
                src = str(getattr(l3, 'src', getattr(l3, 'source', src)))
                dst = str(getattr(l3, 'dst', getattr(l3, 'destination', dst)))
                sport = int(getattr(l3, 'sport', 0))
                dport = int(getattr(l3, 'dport', 0))

        payload_hex = payload_bytes.hex() if payload_bytes else '00'
        dicts.append({
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
        })
    return dicts


class TrafficPipeline:
    """
    网络流量分析流水线

    串联 GNN4ID（分类）和 NEMESYS（逆向）两个分析器，
    实现从原始PCAP到综合分析报告的完整流程。
    """

    def __init__(self, model_path=None, sigma=None, device=None,
                 proto_model_path=None, proto_threshold=None):
        print("=" * 60)
        print("  nemesys-gnn4id 流量分析流水线")
        print("=" * 60)

        self.gnn = GNN4IDAnalyzer(model_path=model_path, device=device)
        self.nemesys = NEMESYSAnalyzer(sigma=sigma)
        self.protocol_classifier = None
        self.protocol_classifier_error = None
        self.proto_model_path = proto_model_path

        if proto_model_path:
            try:
                from nemesys_gnn4id.proto_gnn.classifier import FieldProtocolClassifier
                self.protocol_classifier = FieldProtocolClassifier(
                    proto_model_path,
                    device=device,
                    confidence_threshold=proto_threshold,
                )
                print(f"[PIPELINE] 字段图协议模型已加载: {proto_model_path}")
            except Exception as e:
                self.protocol_classifier_error = str(e)
                print(f"[PIPELINE] 字段图协议模型加载失败: {e}")

        print("=" * 60)

    def analyze(self, pcap_path, max_flows=1000, deep_analysis=True, run_gnn=True):
        """
        完整分析流程 (NEMESYS-First)

        Args:
            pcap_path: PCAP文件路径
            max_flows: GNN4ID 最大处理flow数
            deep_analysis: 是否做NEMESYS深度分析
            run_gnn: 是否运行GNN4ID分析（False时只跑纯NEMESYS）

        Returns:
            dict: 综合分析结果
        """
        start_time = time.time()

        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP文件不存在: {pcap_path}")

        pcap_name = os.path.basename(pcap_path)
        print(f"\n[PIPELINE] 开始分析: {pcap_name}")
        print(f"[PIPELINE] 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ---- Step 1: NEMESYS BCDG (始终最先执行) ----
        nemesys_results = None
        clustering_results = None
        protocol_results = None
        if deep_analysis:
            print(f"\n{'='*40}")
            print("  Step 1: NEMESYS 协议逆向 (BCDG 字段分割)")
            print(f"{'='*40}")
            nemesys_results = self._deep_analyze(pcap_path, target_flows=None)

            # ---- Step 1.5: 字段图协议识别 ----
            if nemesys_results and 'segments' in nemesys_results:
                segs = nemesys_results['segments']
                max_msg = max_flows * 2 if max_flows else len(segs)
                protocol_results = self._infer_protocols(segs[:max_msg])
            elif self.protocol_classifier_error:
                protocol_results = {
                    'error': self.protocol_classifier_error,
                    'summary': {'error_count': 1},
                    'messages': [],
                    'flow_predictions': {},
                }

        # ---- Step 2: 域路由器 (根据协议识别结果分流) ----
        in_domain_keys = set()
        ood_keys = set()
        flow_protocols = {}
        if protocol_results and protocol_results.get('flow_predictions'):
            routing = route_flows_by_protocol(protocol_results['flow_predictions'])
            in_domain_keys = routing['in_domain_keys']
            ood_keys = routing['ood_keys']
            flow_protocols = routing['flow_protocols']
            print(f"\n{'='*40}")
            print("  Step 2: 域路由器")
            print(f"{'='*40}")
            print(f"  域内流: {len(in_domain_keys)}")
            print(f"  域外流 (OOD): {len(ood_keys)}")
        else:
            print(f"\n{'='*40}")
            print("  Step 2: 域路由器")
            print(f"{'='*40}")
            print(f"  无协议预测结果，所有流视为域外")

        # ---- Step 3: GNN4ID 攻击检测 (域内过滤模式) ----
        gnn_results = None
        flow_metadata = None
        if run_gnn:
            print(f"\n{'='*40}")
            print("  Step 3: GNN4ID 攻击检测 (域内过滤模式)")
            print(f"{'='*40}")
            # When routing info exists, always pass in_domain_keys for filtering
            # (even empty set = filter everything out). Only pass None when no
            # protocol routing was available (process all flows).
            has_proto_routing = bool(protocol_results and protocol_results.get('flow_predictions'))
            include_flow_keys = in_domain_keys if has_proto_routing else None
            gnn_results, flow_metadata = self.gnn.analyze_pcap_filtered(
                pcap_path, max_flows=max_flows,
                include_flow_keys=include_flow_keys,
            )
        else:
            print(f"\n{'='*40}")
            print("  跳过 GNN4ID，仅执行 NEMESYS 分析")
            print(f"{'='*40}")

        # ---- Step 4: 结构指纹聚类 (全量，为交叉验证提供信号) ----
        if nemesys_results and 'segments' in nemesys_results:
            print(f"\n{'='*40}")
            print("  Step 4: 协议结构指纹聚类")
            print(f"{'='*40}")

            cluster_segments = _segments_to_dicts(nemesys_results['segments'])
            cluster_analyzer = ClusterAnalyzer()
            clustering_results = cluster_analyzer.analyze(cluster_segments)

            cluster_summary = clustering_results.get('summary', {})
            print(f"  协议簇数: {cluster_summary.get('total_clusters', 0)}")
            print(f"  结构离群: {cluster_summary.get('total_outliers', 0)} 条")
            print(f"  风险等级: {cluster_summary.get('risk_level', 'none')}")

            for c in clustering_results.get('clusters', [])[:5]:
                indicators = c.get('indicators', [])
                flag_str = f" [{'|'.join(indicators)}]" if indicators else ""
                print(f"    簇{c['cluster_id']}: {c['size']}条, "
                      f"{c.get('inferred_name', '未知')}, "
                      f"紧密度={c['cohesion']}{flag_str}")

        # ---- Step 5: 分类统计与融合决策 ----
        classified = None
        domain_info = None
        if gnn_results:
            print(f"\n{'='*40}")
            print("  Step 5: 分类统计与融合决策")
            print(f"{'='*40}")

            # 标记域外流为不可信
            ood_count = 0
            ood_reclassified = 0
            for r in gnn_results:
                if r.get('gnn_skipped'):
                    r['ood'] = True
                    r['gnn_unreliable'] = True
                    ood_count += 1
                    if r['class_id'] != 0:
                        r['_gnn_original_class_id'] = r['class_id']
                        r['_gnn_original_class_name'] = r['class_name']
                        r['class_id'] = 0
                        r['class_name'] = '正常流量(OOD排除)'
                        r['class_name_en'] = 'Benign(OOD)'
                        ood_reclassified += 1

            classified = self._classify_flows(gnn_results)
            # 域过滤后重新计算分类
            classified = self._recompute_classification(gnn_results)

            # ---- 融合决策：为每条流计算 fusion_decision ----
            fusion_status_counts = Counter()
            for r in gnn_results:
                idx = r.get('flow_index', -1)
                if idx < 0 or not flow_metadata or idx >= len(flow_metadata):
                    continue
                meta = flow_metadata[idx]
                key = flow_meta_key(meta)

                # 确定协议名
                proto_name = flow_protocols.get(key, 'UNKNOWN_PROTO')
                in_domain = key in in_domain_keys

                # 构建 proto_pred
                from nemesys_gnn4id.nemesys.analyzer import PORT_PROTOCOL_MAP
                sport = meta.get('src_port', 0)
                dport = meta.get('dst_port', 0)
                if proto_name != 'UNKNOWN_PROTO':
                    proto_pred = {
                        'protocol': proto_name,
                        'confidence': None,
                        'is_known': True,
                        'source': 'field_graph',
                    }
                else:
                    fallback = PORT_PROTOCOL_MAP.get(sport) or PORT_PROTOCOL_MAP.get(dport)
                    proto_pred = {
                        'protocol': fallback or 'UNKNOWN_PROTO',
                        'confidence': None,
                        'is_known': fallback is not None,
                        'source': 'port_fallback',
                    }

                # 匹配到结构聚类簇
                cluster_info = None
                if clustering_results:
                    cl_match = self._match_flow_to_cluster(meta, clustering_results, nemesys_results)
                    if cl_match:
                        for c in clustering_results.get('clusters', []):
                            if c.get('cluster_id') == cl_match['cluster_id']:
                                cluster_info = c
                                break

                # 构建融合决策
                fusion = self._build_fusion_decision(r, proto_name, in_domain, cluster_info, proto_pred)
                r['fusion_decision'] = fusion
                r['protocol_prediction'] = proto_pred
                fusion_status_counts[fusion['status']] += 1

                # 诊断（仅当 NEMESYS_ABLATION_DUMP=1）：记录每个变体离散决策背后的连续量。
                # 只写 r['_diag']，不参与任何决策；开关关闭时行为与之前完全一致。
                if ablation_dump_enabled():
                    r['_diag'] = self._build_ablation_diag(
                        r, proto_name, in_domain, cluster_info, proto_pred, fusion)

            if fusion_status_counts:
                print('  融合决策:')
                for status, count in fusion_status_counts.most_common():
                    print(f'    {status}: {count} 条')

            domain_info = {
                'in_domain_keys': list(in_domain_keys),
                'ood_keys': list(ood_keys),
                'flow_protocols': flow_protocols,
                'in_domain_count': len(in_domain_keys),
                'ood_count': len(ood_keys),
                'ood_reclassified': ood_reclassified,
                'fusion_summary': dict(fusion_status_counts),
            }

            print(f"  域内流: {len(in_domain_keys)}")
            print(f"  域外流 (OOD): {len(ood_keys)}")
            if ood_reclassified > 0:
                print(f"  OOD分类覆写: {ood_reclassified} 条 → 降级为结构异常检测")

        # ---- Step 6: 时序异常检测 ----
        temporal = None
        if gnn_results:
            print(f"\n{'='*40}")
            print("  Step 6: 时序异常检测")
            print(f"{'='*40}")
            temporal = self._temporal_analysis(gnn_results, classified or {})

        # ---- Step 7: 交叉验证（含聚类信息） ----
        cross_validation = None
        if nemesys_results and 'summary' in nemesys_results:
            print(f"\n{'='*40}")
            print("  Step 7: GNN-NEMESYS-聚类 交叉验证")
            print(f"{'='*40}")
            cross_validation = self._cross_validate(
                classified, nemesys_results, flow_metadata,
                clustering_results=clustering_results,
                protocol_results=protocol_results,
            )

        # ---- 汇总结果 ----
        elapsed = time.time() - start_time
        result = {
            'pcap_file': pcap_name,
            'analysis_time': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed, 2),
            'gnn': {
                'total_flows': len(gnn_results) if gnn_results else 0,
                'results': gnn_results or [],
                'model_loaded': self.gnn.model_loaded if gnn_results else False,
            },
            'flow_metadata': flow_metadata if gnn_results else [],
            'classification': classified or {},
            'nemesys': nemesys_results,
            'protocol_inference': protocol_results,
            'clustering': clustering_results,
            'domain_info': domain_info,
            'temporal': temporal,
            'cross_validation': cross_validation,
        }

        self._print_summary(result)
        return result

    def _classify_flows(self, gnn_results):
        """将GNN分类结果按攻击/正常分离"""
        attack_flows = []
        normal_flows = []
        attack_stats = {}

        for r in gnn_results:
            class_id = r['class_id']
            class_name = r['class_name']
            level = r['level']
            confidence = r['confidence']

            # 低置信度标记（不影响分类，仅做提示）
            if confidence < CONFIDENCE_THRESHOLD:
                r['low_confidence'] = True

            if class_id == 0:
                normal_flows.append(r)
            else:
                attack_flows.append(r)
                if class_name not in attack_stats:
                    attack_stats[class_name] = {
                        'count': 0,
                        'level': level,
                        'max_confidence': 0,
                        'min_confidence': 1.0,
                        'avg_confidence': 0,
                        'flows': [],
                        'low_conf_count': 0,
                    }
                stats = attack_stats[class_name]
                stats['count'] += 1
                stats['max_confidence'] = max(stats['max_confidence'], confidence)
                stats['min_confidence'] = min(stats['min_confidence'], confidence)
                stats['flows'].append(r['flow_index'])
                if r.get('low_confidence'):
                    stats['low_conf_count'] += 1

        # Compute average confidence
        for name, stats in attack_stats.items():
            flow_confs = [r['confidence'] for r in attack_flows if r['class_name'] == name]
            stats['avg_confidence'] = round(sum(flow_confs) / max(len(flow_confs), 1), 3)

        classified = {
            'normal_count': len(normal_flows),
            'attack_count': len(attack_flows),
            'attack_flows': attack_flows,
            'normal_flows': normal_flows,
            'attack_stats': attack_stats,
        }

        print(f"  正常流量: {len(normal_flows)}")
        print(f"  攻击流量: {len(attack_flows)}")
        for name, stats in attack_stats.items():
            low_tip = f", 低置信度{stats['low_conf_count']}条" if stats.get('low_conf_count') else ""
            print(f"    - {name}: {stats['count']} 条 "
                  f"(置信度 {stats['min_confidence']:.3f}~{stats['max_confidence']:.3f}{low_tip})")

        return classified

    def _recompute_classification(self, gnn_results):
        """基于域过滤后的 GNN 结果重新生成分类统计。"""
        return self._classify_flows(gnn_results)

    def _extract_attack_flow_tuples(self, attack_flows, flow_metadata):
        """Extract 5-tuples of attack flows for NEMESYS targeting."""
        tuples = []
        for r in attack_flows:
            idx = r['flow_index']
            if idx < len(flow_metadata):
                meta = flow_metadata[idx]
                tuples.append((
                    meta.get('src_ip', ''),
                    meta.get('dst_ip', ''),
                    meta.get('src_port', 0),
                    meta.get('dst_port', 0),
                    meta.get('protocol', 0),
                ))
        return tuples if tuples else None

    def _extract_all_flow_tuples(self, flow_metadata):
        """Extract 5-tuples of ALL flows for NEMESYS comprehensive analysis."""
        tuples = []
        for meta in flow_metadata:
            tuples.append((
                meta.get('src_ip', ''),
                meta.get('dst_ip', ''),
                meta.get('src_port', 0),
                meta.get('dst_port', 0),
                meta.get('protocol', 0),
            ))
        return tuples if tuples else None

    def _infer_protocols(self, segments):
        """Use FieldProtoGNN to infer protocol names from NEMESYS field graphs."""
        if self.protocol_classifier is None:
            if self.protocol_classifier_error:
                print(f"  字段图协议识别跳过: {self.protocol_classifier_error}")
                return {
                    'error': self.protocol_classifier_error,
                    'summary': {'error_count': 1},
                    'messages': [],
                    'flow_predictions': {},
                }
            print("  字段图协议识别跳过: 未提供 --proto-model")
            return None

        print(f"\n{'='*40}")
        print("  Step 1.5: FieldProtoGNN 字段图协议识别")
        print(f"{'='*40}")

        result = self.protocol_classifier.predict_segments(segments)
        summary = result.get('summary', {})
        print(f"  消息数: {summary.get('total_messages', 0)}")
        print(f"  已知协议消息: {summary.get('known_messages', 0)}")
        print(f"  低置信/未知消息: {summary.get('unknown_messages', 0)}")
        if summary.get('error_count', 0):
            print(f"  推理错误消息: {summary.get('error_count', 0)}")
        if summary.get('protocol_distribution'):
            print("  字段图协议分布:")
            for proto, count in summary['protocol_distribution'].items():
                print(f"    {proto}: {count} 条")
        return result

    @staticmethod
    def _cluster_is_high_risk(cluster_info):
        """Return True when a structural cluster carries suspicious evidence."""
        if not cluster_info:
            return False
        indicators = cluster_info.get('indicators', []) or []
        if cluster_info.get('outlier_count', 0) > 0:
            return True
        high_risk_flags = {'未知协议', '孤立小簇', '高离群率', '高熵(加密/混淆)'}
        # 折中方案：离群率>15%，或至少1个高风险指标且簇结构不紧密
        outlier_count = cluster_info.get('outlier_count', 0)
        cluster_size = cluster_info.get('size', 1)
        outlier_ratio = outlier_count / max(cluster_size, 1)
        if outlier_ratio > 0.15:
            return True
        cohesion = cluster_info.get('cohesion', '')
        risk_flags = [f for f in indicators if f in high_risk_flags]
        if risk_flags and cohesion in ('loose', 'moderate'):
            return True
        return False

    @staticmethod
    def _build_nemesys_protocol_predictions(nemesys_results):
        """Aggregate raw NEMESYS protocol labels to flow-level votes."""
        if not nemesys_results:
            return {}

        by_flow = {}
        for seg in nemesys_results.get('segments', []) or []:
            key = segment_flow_key(seg)
            if key is None:
                continue
            by_flow.setdefault(key, Counter())[seg.get('protocol', 'UNKNOWN_PROTO')] += 1

        predictions = {}
        for key, votes in by_flow.items():
            protocol, _ = votes.most_common(1)[0]
            predictions[key] = {
                'protocol': protocol,
                'distribution': dict(votes),
                'message_count': sum(votes.values()),
            }
        return predictions

    def _should_nemesys_unknown_override(self, proto_name, proto_pred, nemesys_proto_pred, cluster_info):
        """Let NEMESYS unknown evidence override field-graph known labels when risk is high."""
        if not nemesys_proto_pred or nemesys_proto_pred.get('protocol') != 'UNKNOWN_PROTO':
            return False
        if proto_name == 'UNKNOWN_PROTO':
            return False

        distribution = nemesys_proto_pred.get('distribution', {})
        message_count = max(nemesys_proto_pred.get('message_count', 0), 1)
        unknown_ratio = distribution.get('UNKNOWN_PROTO', 0) / message_count
        confidence = proto_pred.get('confidence') if proto_pred else None
        low_confidence = confidence is None or confidence < 0.9

        return self._cluster_is_high_risk(cluster_info) or unknown_ratio >= 0.6 or low_confidence

    def _build_fusion_decision(self, gnn_result, proto_name, in_domain, cluster_info, proto_pred):
        """Combine GNN attack class, protocol graph inference and cluster evidence."""
        is_attack = gnn_result.get('class_id', 0) != 0
        known_proto = bool(proto_pred.get('is_known', proto_name != 'UNKNOWN_PROTO')) if proto_pred else False
        cluster_high = self._cluster_is_high_risk(cluster_info)
        confidence = proto_pred.get('confidence') if proto_pred else None
        cluster_id = cluster_info.get('cluster_id') if cluster_info else None

        if is_attack and in_domain:
            return {
                'status': 'confirmed_gnn_attack',
                'risk_level': gnn_result.get('level', 'high'),
                'trust_gnn': True,
                'preserve_attack_alert': True,
                'protocol': proto_name,
                'protocol_confidence': confidence,
                'cluster_id': cluster_id,
                'reason': '协议在GNN训练域内，保留GNN攻击分类',
            }

        if is_attack and not in_domain and (not known_proto) and cluster_high:
            return {
                'status': 'suspicious_unknown_protocol_attack',
                'risk_level': 'high',
                'trust_gnn': False,
                'preserve_attack_alert': True,
                'protocol': proto_name,
                'protocol_confidence': confidence,
                'cluster_id': cluster_id,
                'reason': '未知/低置信协议同时存在结构聚类异常，保留高风险告警',
            }

        if is_attack and not in_domain:
            return {
                'status': 'ood_gnn_possible_false_positive',
                'risk_level': 'medium',
                'trust_gnn': False,
                'preserve_attack_alert': False,
                'protocol': proto_name,
                'protocol_confidence': confidence,
                'cluster_id': cluster_id,
                'reason': '协议不在GNN训练域内且暂无结构异常，GNN攻击分类降级',
            }

        if (not is_attack) and (not in_domain) and (not known_proto) and cluster_high:
            return {
                'status': 'possible_missed_unknown_protocol_attack',
                'risk_level': 'high',
                'trust_gnn': False,
                'preserve_attack_alert': False,
                'protocol': proto_name,
                'protocol_confidence': confidence,
                'cluster_id': cluster_id,
                'reason': 'GNN判正常，但未知/域外协议存在结构异常，需人工复核',
            }

        if (not known_proto) and (not in_domain):
            return {
                'status': 'unknown_protocol_monitor',
                'risk_level': 'low',
                'trust_gnn': False,
                'preserve_attack_alert': False,
                'protocol': proto_name,
                'protocol_confidence': confidence,
                'cluster_id': cluster_id,
                'reason': '未知协议暂无攻击或结构异常证据，进入新协议簇跟踪',
            }

        return {
            'status': 'low_risk_known_protocol',
            'risk_level': 'none',
            'trust_gnn': in_domain,
            'preserve_attack_alert': False,
            'protocol': proto_name,
            'protocol_confidence': confidence,
            'cluster_id': cluster_id,
            'reason': '未发现攻击分类或结构异常',
        }

    def _build_ablation_diag(self, gnn_result, proto_name, in_domain, cluster_info,
                             proto_pred, fusion):
        """Per-flow diagnostic record for the ablation (NEMESYS_ABLATION_DUMP=1 only).

        Records the CONTINUOUS quantities behind each ablation variant's discrete
        decision, so the variants can be re-evaluated at a matched operating point
        without re-running the pipeline. Purely observational: this method is called
        after the decision is already made, its return value is stored under the
        private key ``_diag``, and nothing reads it back. The decision logic in
        ``_build_fusion_decision`` is untouched, and the stored ``status`` /
        ``is_attack`` / ``in_domain`` / ``known_proto`` are recomputed with the exact
        expressions that method uses (a mismatch is detectable by replay -- see
        experiments/ablation_controlled_fpr.py).

        cluster_score is built so that ``cluster_score > 0`` is EXACTLY
        ``_cluster_is_high_risk(cluster_info)``. That method is a disjunction of two
        independent pieces of evidence, so no raw single field stands in for it:

          (i)  the cluster contains a structural outlier
               (``outlier_count > 0``, i.e. outlier_ratio > 0; its second outlier
               branch, outlier_ratio > 0.15, can never decide because outlier_count > 0
               has already returned True), and
          (ii) the cluster carries a high-risk indicator flag AND is not tight
               (cohesion in ('loose', 'moderate'), i.e. frame-level mean_distance
               >= 0.15) -- this branch fires with outlier_count == 0.

        A disjunction of two non-negative evidence strengths is their maximum, so

          cluster_score = max(outlier_ratio, mean_distance if (ii) else 0.0)

        maps the pipeline's own boolean onto the single strict threshold
        ``cluster_score > 0``. The raw components are recorded separately
        (cluster_outlier_ratio / cluster_structural_score / cluster_mean_distance /
        cluster_cohesion / cluster_indicators / cluster_branch_*) so the composition is
        auditable and can be replaced by any other convention offline.

        Fields:
          gnn_prob_attack 1 - P(Benign) from the GNN softmax, or None when the flow
                          never reached the model (gnn_skipped OOD flow, or error)
          cluster_score   see above; 0.0 when no cluster matched. Computed UNROUNDED
                          from outlier_count/size (the cluster record itself stores a
                          3-decimal outlier_ratio); mean_distance is the stored
                          4-decimal value.
          in_domain       GNN training-domain membership (domain router)
          known_proto     protocol_prediction['is_known'] as used by the fusion rule
          is_attack       gnn_result['class_id'] != 0 (post OOD reclassification)
          status          the fusion status string (None when no decision was made)
          cluster_high    the pipeline's OWN _cluster_is_high_risk(cluster_info)
                          boolean -- the ground truth for replay verification
          cluster_branch_outlier / cluster_branch_structural
                          which of _cluster_is_high_risk's branches fired, so the
                          incidence of each can be audited independently of the score
        """
        prob_benign = gnn_result.get('prob_benign')
        cluster = cluster_info or None
        outlier_count = int(cluster.get('outlier_count', 0) or 0) if cluster else 0
        cluster_size = int(cluster.get('size', 1) or 1) if cluster else 0
        outlier_ratio = (outlier_count / max(cluster_size, 1)) if cluster else 0.0
        indicators = list(cluster.get('indicators', []) or []) if cluster else []
        cohesion = cluster.get('cohesion', '') if cluster else ''
        mean_distance = float(cluster.get('mean_distance', 0.0) or 0.0) if cluster else 0.0
        branch_structural = bool(
            [f for f in indicators if f in _CLUSTER_STRUCTURAL_HIGH_RISK_FLAGS]
            and cohesion in ('loose', 'moderate')
        )
        structural_score = mean_distance if branch_structural else 0.0
        cluster_score = max(outlier_ratio, structural_score)
        return {
            'gnn_prob_attack': None if prob_benign is None else 1.0 - float(prob_benign),
            'prob_benign': None if prob_benign is None else float(prob_benign),
            'cluster_score': float(cluster_score),
            'cluster_outlier_ratio': float(outlier_ratio),
            'cluster_structural_score': float(structural_score),
            'in_domain': bool(in_domain),
            'known_proto': bool(
                proto_pred.get('is_known', proto_name != 'UNKNOWN_PROTO')
            ) if proto_pred else False,
            'is_attack': bool(gnn_result.get('class_id', 0) != 0),
            'status': fusion.get('status'),
            'cluster_high': bool(self._cluster_is_high_risk(cluster_info)),
            'cluster_branch_outlier': bool(outlier_count > 0),
            'cluster_branch_structural': branch_structural,
            'cluster_matched': bool(cluster),
            'cluster_id': cluster.get('cluster_id') if cluster else None,
            'cluster_outlier_count': outlier_count if cluster else None,
            'cluster_size': cluster_size if cluster else None,
            'cluster_cohesion': cohesion or None,
            'cluster_mean_distance': mean_distance if cluster else None,
            'cluster_indicators': indicators,
            # inputs used by the ablation variants that are defined outside the fusion
            # rule (A1 uses the pre-OOD-override class, A2 uses the ood flag)
            'orig_class_id': gnn_result.get('_gnn_original_class_id',
                                            gnn_result.get('class_id', 0)),
            'class_id': gnn_result.get('class_id', 0),
            'ood': bool(gnn_result.get('ood', False)),
            'gnn_skipped': bool(gnn_result.get('gnn_skipped', False)),
        }

    @staticmethod
    def _match_flow_to_cluster(flow_meta, clustering_results, nemesys_results):
        """将 flow 匹配到聚类结果中的簇（按 IP+端口 对应 NEMESYS segment）。"""
        if not clustering_results or not nemesys_results:
            return None
        segments = nemesys_results.get('segments', [])
        labels = clustering_results.get('labels', [])
        if not segments or len(labels) != len(segments):
            return None

        target_key = flow_meta_key(flow_meta)

        votes = Counter()
        for i, seg in enumerate(segments):
            if segment_flow_key(seg) == target_key:
                lbl = int(labels[i])
                if lbl >= 0:
                    votes[lbl + 1] += 1
        if votes:
            best = votes.most_common(1)[0][0]
            return {'cluster_id': best, 'votes': dict(votes)}
        return None

    def _deep_analyze(self, pcap_path, target_flows=None):
        """对攻击流量做NEMESYS深度分析"""
        if target_flows:
            print(f"  对 {len(target_flows)} 条攻击flow做协议逆向...")
        else:
            print(f"  对整个PCAP做BCDG协议逆向...")

        try:
            nemesys_result = self.nemesys.analyze(pcap_path, target_flows=target_flows)
            summary = nemesys_result['summary']

            print(f"  分析消息数: {summary.get('total_messages', 0)}")
            print(f"  平均字段数: {summary.get('avg_fields', 0)}")
            print(f"  分析模式: {summary.get('mode', 'unknown')}")

            if 'protocol_distribution' in summary:
                print(f"  协议分布:")
                for proto, count in summary['protocol_distribution'].items():
                    print(f"    {proto}: {count} 条")

            return nemesys_result
        except Exception as e:
            print(f"  [ERROR] NEMESYS分析失败: {e}")
            return {'error': str(e)}

    def _temporal_analysis(self, gnn_results, classified):
        """
        时序异常检测：按时间窗口聚合 flow，检测突增模式。
        使用 _classify_flows 的统一分类结果。

        Returns:
            dict: 时序分析结果
        """
        if not gnn_results:
            return {'anomalies': [], 'windows': []}

        # 使用 _classify_flows 的结果，构建攻击 flow 索引集合（统一判断标准）
        attack_indices = set(r['flow_index'] for r in classified.get('attack_flows', []))
        normal_indices = set(r['flow_index'] for r in classified.get('normal_flows', []))

        # Count attacks per "window" of consecutive flows
        window_size = max(5, len(gnn_results) // 10)
        windows = []
        anomalies = []

        for i in range(0, len(gnn_results), window_size):
            window_flows = gnn_results[i:i + window_size]
            attack_count = sum(1 for r in window_flows if r['flow_index'] in attack_indices)
            attack_ratio = attack_count / max(len(window_flows), 1)

            # Class distribution in this window (attack types only)
            class_dist = Counter(r['class_name'] for r in window_flows if r['flow_index'] in attack_indices)

            window = {
                'start_flow': i,
                'end_flow': min(i + window_size, len(gnn_results)),
                'total': len(window_flows),
                'attacks': attack_count,
                'attack_ratio': round(attack_ratio, 3),
                'class_distribution': dict(class_dist),
            }
            windows.append(window)

            # Anomaly detection: sudden spike in attack ratio
            if attack_ratio > 0.5 and attack_count >= 3:
                anomalies.append({
                    'type': 'attack_burst',
                    'window': f"flow {i}-{min(i + window_size, len(gnn_results))}",
                    'attack_count': attack_count,
                    'attack_ratio': round(attack_ratio, 3),
                    'dominant_class': class_dist.most_common(1)[0][0] if class_dist else 'unknown',
                })

            # Detect class transitions (e.g., recon → exploit)
            if i > 0:
                prev_window = gnn_results[max(0, i - window_size):i]
                prev_classes = set(r['class_name'] for r in prev_window if r['flow_index'] in attack_indices)
                curr_classes = set(r['class_name'] for r in window_flows if r['flow_index'] in attack_indices)
                new_classes = curr_classes - prev_classes
                if new_classes and prev_classes:
                    anomalies.append({
                        'type': 'class_transition',
                        'window': f"flow {i}-{min(i + window_size, len(gnn_results))}",
                        'new_classes': list(new_classes),
                        'previous_classes': list(prev_classes),
                    })

        if anomalies:
            print(f"  检测到 {len(anomalies)} 个时序异常")
            for a in anomalies:
                if a['type'] == 'attack_burst':
                    print(f"    攻击突增: {a['window']}, {a['attack_count']} attacks ({a['attack_ratio']:.0%})")
                elif a['type'] == 'class_transition':
                    print(f"    攻击类型变化: {a['window']}, 新类型: {a['new_classes']}")
        else:
            print(f"  未检测到时序异常")

        return {
            'anomalies': anomalies,
            'windows': windows,
            'total_windows': len(windows),
        }

    def _detect_unknown_protocol_attacks(self, nemesys_results):
        """检测利用未知协议的网络攻击"""
        detector = UnknownProtocolDetector()
        result = detector.detect_unknown_protocol_attacks(nemesys_results.get('segments', []))

        total = result.get('total_unknown', 0)
        risk = result.get('risk_assessment', {})
        clusters = result.get('protocol_clusters', [])

        print(f"  未知协议流数: {total}")
        print(f"  风险等级: {risk.get('level', 'none')}")
        print(f"  协议聚类数: {len(clusters)}")

        for c in clusters[:3]:
            print(f"    聚类 {c['cluster_id']}: {c['description']} "
                  f"({c['flow_count']} 条流, 风险: {c['risk_level']})")

        return result

    def _cross_validate(self, classified, nemesys_results, flow_metadata,
                         clustering_results=None, protocol_results=None):
        """
        交叉验证 GNN、NEMESYS 和聚类结果。

        GNN 标记为攻击但 NEMESYS 发现正常协议 → 可能误报
        GNN 标记为正常但 NEMESYS 发现异常字段结构 → 可能漏报
        GNN 标记为正常但使用未知协议且有攻击指标 → 可能漏报
        聚类中发现的结构离群点 → 独立于 GNN 的可疑信号

        Returns:
            dict: 交叉验证结果
        """
        attack_flows = classified.get('attack_flows', [])
        normal_flows = classified.get('normal_flows', [])
        nemesys_summary = nemesys_results.get('summary', {})
        protocol_dist = nemesys_summary.get('protocol_distribution', {})

        confirmed = []      # GNN 攻击 + NEMESYS 确认异常
        suspicious = []     # GNN 攻击 + NEMESYS 发现正常协议
        potential_miss = [] # GNN 正常 + NEMESYS 发现异常结构/未知协议

        known_normal_protocols = UNKNOWN_PROTO_CONFIG['known_normal_protocols']
        suspicious_ports = UNKNOWN_PROTO_CONFIG['suspicious_ports']

        from nemesys_gnn4id.nemesys.analyzer import PORT_PROTOCOL_MAP
        protocol_predictions = (protocol_results or {}).get('flow_predictions', {})
        nemesys_protocol_predictions = self._build_nemesys_protocol_predictions(nemesys_results)

        def proto_for(meta):
            pred = protocol_predictions.get(flow_meta_key(meta))
            if pred:
                pred = dict(pred)
                proto_name = pred.get('protocol', 'UNKNOWN_PROTO')
            else:
                sport, dport = meta.get('src_port', 0), meta.get('dst_port', 0)
                proto_name = PORT_PROTOCOL_MAP.get(sport) or PORT_PROTOCOL_MAP.get(dport) or 'UNKNOWN_PROTO'
                pred = {'protocol': proto_name, 'source': 'port_fallback'}

            nemesys_pred = nemesys_protocol_predictions.get(flow_meta_key(meta))
            if nemesys_pred:
                pred['nemesys_protocol'] = nemesys_pred.get('protocol')
                pred['nemesys_protocol_distribution'] = nemesys_pred.get('distribution', {})

            cluster_info = None
            if clustering_results:
                cl_match = self._match_flow_to_cluster(meta, clustering_results, nemesys_results)
                if cl_match:
                    for c in clustering_results.get('clusters', []):
                        if c.get('cluster_id') == cl_match['cluster_id']:
                            cluster_info = c
                            break

            if self._should_nemesys_unknown_override(proto_name, pred, nemesys_pred, cluster_info):
                pred['field_graph_protocol'] = proto_name
                pred['protocol'] = 'UNKNOWN_PROTO'
                pred['is_known'] = False
                pred['source'] = 'nemesys_unknown_override'
                pred['nemesys_unknown_override'] = True
                proto_name = 'UNKNOWN_PROTO'
            return proto_name, pred

        for r in attack_flows:
            idx = r['flow_index']
            if idx < len(flow_metadata):
                meta = flow_metadata[idx]
                sport, dport = meta.get('src_port', 0), meta.get('dst_port', 0)

                proto_name, proto_pred = proto_for(meta)

                if proto_name in known_normal_protocols:
                    suspicious.append({
                        'flow_index': idx,
                        'gnn_class': r['class_name'],
                        'confidence': r['confidence'],
                        'protocol': proto_name,
                        'protocol_confidence': proto_pred.get('confidence'),
                        'reason': f'GNN标记为{r["class_name"]}但使用已知正常协议{proto_name}',
                    })
                else:
                    confirmed.append({
                        'flow_index': idx,
                        'gnn_class': r['class_name'],
                        'confidence': r['confidence'],
                        'protocol': proto_name,
                        'protocol_confidence': proto_pred.get('confidence'),
                    })

        # Check NEMESYS field structure anomalies
        field_types = nemesys_summary.get('field_type_distribution', {})
        total_fields = sum(field_types.values()) if field_types else 0
        binary_ratio = field_types.get('binary', 0) / max(total_fields, 1)

        # High binary ratio in attack flows suggests actual malicious payloads.
        # Skip flows already flagged as suspicious (known normal protocol) — they
        # have a benign explanation for binary payload (e.g. DNS responses).
        if binary_ratio > 0.6 and total_fields > 10:
            already_handled = {c['flow_index'] for c in confirmed} | {s['flow_index'] for s in suspicious}
            for r in attack_flows:
                if r['flow_index'] not in already_handled:
                    confirmed.append({
                        'flow_index': r['flow_index'],
                        'gnn_class': r['class_name'],
                        'confidence': r['confidence'],
                        'reason': 'NEMESYS确认高比例二进制payload',
                    })

        # ---- 检测潜在漏报：GNN 标记为正常但 NEMESYS 发现可疑特征 ----
        for r in normal_flows:
            idx = r['flow_index']
            if idx >= len(flow_metadata):
                continue
            meta = flow_metadata[idx]
            sport, dport = meta.get('src_port', 0), meta.get('dst_port', 0)

            # 1. 使用可疑端口
            if sport in suspicious_ports or dport in suspicious_ports:
                port = sport if sport in suspicious_ports else dport
                potential_miss.append({
                    'flow_index': idx,
                    'gnn_class': r['class_name'],
                    'confidence': r['confidence'],
                    'reason': f'GNN标记为正常但使用可疑端口 {port} ({suspicious_ports.get(port, "未知")})',
                    'indicator': 'suspicious_port',
                })
                continue

            # 2. 使用未知协议且非标准端口
            proto_name, proto_pred = proto_for(meta)
            if proto_name == 'UNKNOWN_PROTO' and sport not in PORT_PROTOCOL_MAP and dport not in PORT_PROTOCOL_MAP:
                potential_miss.append({
                    'flow_index': idx,
                    'gnn_class': r['class_name'],
                    'confidence': r['confidence'],
                    'protocol': proto_name,
                    'protocol_confidence': proto_pred.get('confidence'),
                    'reason': f'GNN标记为正常但使用未知协议 (端口 {sport}/{dport})',
                    'indicator': 'unknown_protocol',
                })

        # ---- 聚类结构离群验证 ----
        cluster_outlier_alerts = []
        if clustering_results:
            outlier_flags = clustering_results.get('outlier_flags', [])
            labels = clustering_results.get('labels', [])
            for i, is_outlier in enumerate(outlier_flags):
                if is_outlier and i < len(labels):
                    cluster_id = int(labels[i]) + 1 if int(labels[i]) >= 0 else 0
                    if i < len(flow_metadata):
                        meta = flow_metadata[i]
                        cluster_outlier_alerts.append({
                            'flow_index': i,
                            'cluster_id': cluster_id,
                            'src': meta.get('src_ip', ''),
                            'dst': meta.get('dst_ip', ''),
                            'sport': meta.get('src_port', 0),
                            'dport': meta.get('dst_port', 0),
                            'reason': f'结构指纹在簇{cluster_id}中偏离中心超过阈值',
                        })

        # Summary
        result = {
            'confirmed': len(confirmed),
            'suspicious': len(suspicious),
            'potential_miss': len(potential_miss),
            'cluster_outlier_alerts': len(cluster_outlier_alerts),
            'details': {
                'confirmed': confirmed[:10],
                'suspicious': suspicious[:10],
                'potential_miss': potential_miss[:10],
                'cluster_outliers': cluster_outlier_alerts[:10],
            },
        }

        print(f"  确认攻击: {len(confirmed)}")
        print(f"  可疑误报: {len(suspicious)}")
        print(f"  潜在漏报: {len(potential_miss)}")
        if cluster_outlier_alerts:
            print(f"  结构离群告警: {len(cluster_outlier_alerts)} (聚类独立检出)")

        for s in suspicious[:3]:
            print(f"    [可疑] flow {s['flow_index']}: {s['reason']}")

        for m in potential_miss[:3]:
            print(f"    [漏报] flow {m['flow_index']}: {m['reason']}")

        return result

    def _empty_result(self, pcap_name, start_time):
        """空结果"""
        return {
            'pcap_file': pcap_name,
            'analysis_time': datetime.now().isoformat(),
            'elapsed_seconds': round(time.time() - start_time, 2),
            'gnn': {'total_flows': 0, 'results': [], 'model_loaded': self.gnn.model_loaded},
            'classification': {'normal_count': 0, 'attack_count': 0,
                               'attack_flows': [], 'normal_flows': [], 'attack_stats': {}},
            'nemesys': None,
            'protocol_inference': None,
            'clustering': None,
            'domain_info': None,
            'temporal': {'anomalies': [], 'windows': [], 'total_windows': 0},
            'cross_validation': None,
            'metrics': None,
        }

    def evaluate(self, result, ground_truth_labels, domain_aware=True):
        """
        计算分类评估指标

        Args:
            result: analyze() 返回的结果字典
            ground_truth_labels: list[int]，每个 flow 的真实标签（0-7，与 ATTACK_TYPES 对应）
            domain_aware: 是否启用域感知评估（OOD流不计入GNN指标，单独统计）

        Returns:
            dict: 评估指标，含传统指标和域感知指标
        """
        gnn_results = result.get('gnn', {}).get('results', [])
        if not gnn_results:
            return None

        n = min(len(gnn_results), len(ground_truth_labels))
        y_true = ground_truth_labels[:n]
        y_pred_all = [r['class_id'] for r in gnn_results[:n]]
        # 原始GNN预测（覆写前），用于计算"若信任GNN"的误报率
        y_pred_raw = [
            r.get('_gnn_original_class_id', r['class_id'])
            for r in gnn_results[:n]
        ]
        y_pred_fusion_alert = [
            1 if (r.get('fusion_decision') or {}).get('status') in FUSION_ALERT_STATUSES else 0
            for r in gnn_results[:n]
        ]

        # 域感知：分离域内和域外流（使用 gnn 结果中的 ood 标记）
        ood_indices = set()
        if domain_aware:
            for r in gnn_results[:n]:
                if r.get('ood', False):
                    idx = r.get('flow_index', -1)
                    if 0 <= idx < n:
                        ood_indices.add(idx)

        # 域内流（用于标准GNN评估）
        in_domain_mask = [i for i in range(n) if i not in ood_indices]
        y_true_in = [y_true[i] for i in in_domain_mask]
        y_pred_in = [y_pred_all[i] for i in in_domain_mask]

        # 域外流（单独统计，用覆写前的原始GNN分类计算误报率）
        y_true_ood = [y_true[i] for i in ood_indices]
        y_pred_ood = [
            gnn_results[i].get('_gnn_original_class_id', y_pred_all[i])
            for i in ood_indices
        ]

        # 传统指标用原始GNN预测（未经OOD覆写），展示"若信任GNN"的完整性能
        traditional = self._compute_metrics(y_true, y_pred_raw, '全部流(原始GNN)')
        if traditional is not None:
            traditional['fusion_alert'] = self._compute_binary_alert_metrics(
                y_true, y_pred_fusion_alert, '融合高风险告警'
            )

        # 再计算域感知指标
        domain_aware_result = {
            'total_samples': n,
            'in_domain_samples': len(in_domain_mask),
            'ood_samples': len(ood_indices),
        }

        if in_domain_mask:
            domain_aware_result['in_domain'] = self._compute_metrics(
                y_true_in, y_pred_in, '域内流(GNN可信)')
        else:
            domain_aware_result['in_domain'] = None

        if ood_indices:
            domain_aware_result['ood'] = {
                'total': len(ood_indices),
                'description': '域外流不计入GNN评估指标，以聚类异常检测为准',
                'y_true': [int(y) for y in y_true_ood],
                'y_pred': [int(y) for y in y_pred_ood],
                # 计算如果信任GNN的错误率
                'gnn_would_be_fpr': round(
                    sum(1 for i in range(len(y_true_ood))
                        if y_true_ood[i] == 0 and y_pred_ood[i] != 0)
                    / max(len(y_true_ood), 1), 4
                ),
            }

        # 合并：对调用方保持兼容
        if traditional:
            traditional['domain_aware'] = domain_aware_result
        return traditional

    @staticmethod
    def _compute_binary_alert_metrics(y_true, y_pred_binary, label=''):
        """Compute binary attack-vs-normal metrics for alert-style predictions."""
        n = min(len(y_true), len(y_pred_binary))
        y_true_binary = [1 if y_true[i] > 0 else 0 for i in range(n)]
        y_pred_binary = [1 if y_pred_binary[i] > 0 else 0 for i in range(n)]

        tp = sum(1 for i in range(n) if y_true_binary[i] == 1 and y_pred_binary[i] == 1)
        fp = sum(1 for i in range(n) if y_true_binary[i] == 0 and y_pred_binary[i] == 1)
        fn = sum(1 for i in range(n) if y_true_binary[i] == 1 and y_pred_binary[i] == 0)
        tn = sum(1 for i in range(n) if y_true_binary[i] == 0 and y_pred_binary[i] == 0)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        fpr = fp / max(fp + tn, 1)

        return {
            'description': label or '攻击(1) vs 正常(0)',
            'total_samples': n,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'fpr': round(fpr, 4),
        }

    @staticmethod
    def _compute_metrics(y_true, y_pred, label=''):
        """计算标准分类指标。"""
        n = len(y_true)
        if n == 0:
            return None

        num_classes = len(ATTACK_TYPES)
        class_names = {i: ATTACK_TYPES[i]['name'] for i in range(num_classes)}

        # 计算混淆矩阵元素：TP, FP, FN, TN per class
        metrics_per_class = {}
        for c in range(num_classes):
            tp = sum(1 for i in range(n) if y_true[i] == c and y_pred[i] == c)
            fp = sum(1 for i in range(n) if y_true[i] != c and y_pred[i] == c)
            fn = sum(1 for i in range(n) if y_true[i] == c and y_pred[i] != c)
            tn = sum(1 for i in range(n) if y_true[i] != c and y_pred[i] != c)

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            fpr = fp / max(fp + tn, 1)

            metrics_per_class[c] = {
                'name': class_names[c],
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'fpr': round(fpr, 4),
                'support': tp + fn,  # 真实标签中该类的数量
            }

        # 整体指标：macro 平均（各类等权）和 weighted 平按样本数加权）
        valid_classes = [c for c in range(num_classes) if metrics_per_class[c]['support'] > 0]

        if valid_classes:
            macro_precision = sum(metrics_per_class[c]['precision'] for c in valid_classes) / len(valid_classes)
            macro_recall = sum(metrics_per_class[c]['recall'] for c in valid_classes) / len(valid_classes)
            macro_f1 = sum(metrics_per_class[c]['f1'] for c in valid_classes) / len(valid_classes)
            macro_fpr = sum(metrics_per_class[c]['fpr'] for c in valid_classes) / len(valid_classes)

            total_support = sum(metrics_per_class[c]['support'] for c in valid_classes)
            weighted_precision = sum(metrics_per_class[c]['precision'] * metrics_per_class[c]['support'] for c in valid_classes) / max(total_support, 1)
            weighted_recall = sum(metrics_per_class[c]['recall'] * metrics_per_class[c]['support'] for c in valid_classes) / max(total_support, 1)
            weighted_f1 = sum(metrics_per_class[c]['f1'] * metrics_per_class[c]['support'] for c in valid_classes) / max(total_support, 1)
            weighted_fpr = sum(metrics_per_class[c]['fpr'] * metrics_per_class[c]['support'] for c in valid_classes) / max(total_support, 1)
        else:
            macro_precision = macro_recall = macro_f1 = macro_fpr = 0.0
            weighted_precision = weighted_recall = weighted_f1 = weighted_fpr = 0.0

        # 二分类指标（攻击 vs 正常）
        y_true_binary = [1 if y > 0 else 0 for y in y_true]
        y_pred_binary = [1 if y > 0 else 0 for y in y_pred]
        tp_bin = sum(1 for i in range(n) if y_true_binary[i] == 1 and y_pred_binary[i] == 1)
        fp_bin = sum(1 for i in range(n) if y_true_binary[i] == 0 and y_pred_binary[i] == 1)
        fn_bin = sum(1 for i in range(n) if y_true_binary[i] == 1 and y_pred_binary[i] == 0)
        tn_bin = sum(1 for i in range(n) if y_true_binary[i] == 0 and y_pred_binary[i] == 0)

        binary_precision = tp_bin / max(tp_bin + fp_bin, 1)
        binary_recall = tp_bin / max(tp_bin + fn_bin, 1)
        binary_f1 = 2 * binary_precision * binary_recall / max(binary_precision + binary_recall, 1e-10)
        binary_fpr = fp_bin / max(fp_bin + tn_bin, 1)

        return {
            'total_samples': n,
            'per_class': metrics_per_class,
            'macro': {
                'precision': round(macro_precision, 4),
                'recall': round(macro_recall, 4),
                'f1': round(macro_f1, 4),
                'fpr': round(macro_fpr, 4),
            },
            'weighted': {
                'precision': round(weighted_precision, 4),
                'recall': round(weighted_recall, 4),
                'f1': round(weighted_f1, 4),
                'fpr': round(weighted_fpr, 4),
            },
            'binary': {
                'description': '攻击(1) vs 正常(0)',
                'tp': tp_bin, 'fp': fp_bin, 'fn': fn_bin, 'tn': tn_bin,
                'precision': round(binary_precision, 4),
                'recall': round(binary_recall, 4),
                'f1': round(binary_f1, 4),
                'fpr': round(binary_fpr, 4),
            },
        }

    def _print_summary(self, result):
        """打印分析摘要"""
        cls = result.get('classification', {}) or {}
        gnn = result.get('gnn', {})

        print(f"\n{'='*60}")
        print("  分析结果摘要")
        print(f"{'='*60}")
        print(f"  PCAP文件: {result.get('pcap_file', 'N/A')}")
        print(f"  耗时: {result.get('elapsed_seconds', 0)}s")
        print(f"  GNN模型: {'已加载' if gnn.get('model_loaded', False) else '模拟模式'}")
        print(f"  总flow数: {gnn.get('total_flows', 0)}")
        print(f"  正常流量: {cls.get('normal_count', 0)}")
        print(f"  攻击流量: {cls.get('attack_count', 0)}")

        domain_info = result.get('domain_info')
        if domain_info and domain_info.get('ood_reclassified', 0) > 0:
            print(f"  (其中 {domain_info['ood_reclassified']} 条OOD流已排除，GNN分类作废)")

        if cls.get('attack_stats'):
            print(f"\n  域内可信攻击详情 (GNN训练域内):")
            domain_only_attacks = {k: v for k, v in cls['attack_stats'].items()
                                   if not k.endswith('(OOD排除)')}
            if domain_only_attacks:
                for name, stats in domain_only_attacks.items():
                    level = stats.get('level', 'unknown')
                    level_zh = RISK_LEVEL_ZH.get(level, level)
                    print(f"    [{level_zh}] {name}: {stats.get('count', 0)} 条, "
                          f"置信度 {stats.get('avg_confidence', 0):.3f}")
            else:
                print(f"    无 — 所有GNN判定均来自域外流，已排除")

        nemesys = result.get('nemesys')
        if nemesys and 'summary' in nemesys:
            s = nemesys['summary']
            print(f"\n  NEMESYS协议逆向:")
            print(f"    消息数: {s.get('total_messages', 0)}")
            print(f"    平均字段数: {s.get('avg_fields', 0)}")

            if 'protocol_distribution' in s:
                print(f"    协议分布:")
                for proto, count in s['protocol_distribution'].items():
                    print(f"      {proto}: {count}")

        protocol = result.get('protocol_inference')
        if protocol and protocol.get('summary'):
            ps = protocol['summary']
            print(f"\n  字段图协议识别:")
            print(f"    已知协议消息: {ps.get('known_messages', 0)}")
            print(f"    未知/低置信消息: {ps.get('unknown_messages', 0)}")
            if ps.get('protocol_distribution'):
                print(f"    协议分布:")
                for proto, count in ps['protocol_distribution'].items():
                    print(f"      {proto}: {count}")

        # 域感知 + 结构聚类摘要
        domain_info = result.get('domain_info')
        if domain_info:
            print(f"\n  域感知分析:")
            print(f"    GNN训练域内: {domain_info.get('in_domain_count', 0)} 条")
            print(f"    域外(OOD): {domain_info.get('ood_count', 0)} 条")
            if domain_info.get('ood_reclassified', 0) > 0:
                print(f"    域外攻击降级: {domain_info['ood_reclassified']} 条 → 改为结构异常检测")

        clustering = result.get('clustering')
        if clustering and clustering.get('summary'):
            cs = clustering['summary']
            print(f"\n  协议结构聚类:")
            print(f"    协议簇: {cs.get('total_clusters', 0)}")
            print(f"    结构离群: {cs.get('total_outliers', 0)} 条")
            print(f"    风险等级: {cs.get('risk_level', 'none')}")
            if cs.get('description'):
                print(f"    {cs['description']}")
            for c in clustering.get('clusters', [])[:5]:
                indicators = c.get('indicators', [])
                flag_str = f" [{'|'.join(indicators)}]" if indicators else ""
                outlier_info = f", 离群{c.get('outlier_count', 0)}条" if c.get('outlier_count', 0) > 0 else ""
                print(f"    簇{c.get('cluster_id', 0)}: {c.get('inferred_name', '?')} "
                      f"({c.get('size', 0)}条{outlier_info}){flag_str}")

        print(f"{'='*60}\n")
