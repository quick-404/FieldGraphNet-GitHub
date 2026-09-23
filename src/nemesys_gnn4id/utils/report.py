"""
报告生成：将分析结果输出为文本/JSON报告
"""

import json
import os
from datetime import datetime

from nemesys_gnn4id.config import ATTACK_TYPES, RISK_LEVEL_ZH

# 模块级常量：中文序号
_CN_NUMBERS = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五',
               6: '六', 7: '七', 8: '八', 9: '九', 10: '十'}


def generate_report(result, output_path=None, fmt='auto'):
    """
    生成分析报告

    Args:
        result: pipeline.analyze() 返回的结果字典
        output_path: 报告输出路径。None则返回字符串。
        fmt: 'text', 'json', 或 'auto' (根据文件扩展名判断)

    Returns:
        str: 报告内容（如果output_path为None）
    """
    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        if fmt == 'auto':
            if ext == '.json':
                fmt = 'json'
            elif ext == '.md':
                fmt = 'md'
            else:
                fmt = 'text'

    if fmt == 'json':
        report = _generate_json(result)
    elif fmt == 'md':
        report = _generate_markdown(result)
    else:
        report = _generate_text(result)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[REPORT] 报告已保存: {output_path}")

    return report


def _generate_json(result):
    """Generate structured JSON report."""
    cls = result.get('classification', {})
    gnn = result.get('gnn', {})
    nemesys = result.get('nemesys')

    # Build attack summary
    attack_summary = []
    for name, stats in cls.get('attack_stats', {}).items():
        attack_summary.append({
            'type': name,
            'level': stats.get('level', 'unknown'),
            'count': stats.get('count', 0),
            'confidence_min': stats.get('min_confidence', 0),
            'confidence_max': stats.get('max_confidence', 0),
            'confidence_avg': stats.get('avg_confidence', 0),
            'low_conf_count': stats.get('low_conf_count', 0),
            'flow_indices': stats.get('flows', []),
        })

    # Build NEMESYS summary
    nemesys_summary = None
    if nemesys and 'summary' in nemesys:
        s = nemesys['summary']
        nemesys_summary = {
            'mode': s.get('mode', 'unknown'),
            'total_messages': s.get('total_messages', 0),
            'avg_fields': s.get('avg_fields', 0),
            'protocol_distribution': s.get('protocol_distribution', {}),
            'field_type_distribution': s.get('field_type_distribution', {}),
            'payload_analysis': s.get('payload_analysis', {}),
        }

    # Unknown protocol analysis (legacy)
    up = result.get('unknown_protocol')
    unknown_proto_summary = None
    if up and up.get('total_unknown', 0) > 0:
        unknown_proto_summary = {
            'total_unknown_flows': up.get('total_unknown', 0),
            'risk_assessment': up.get('risk_assessment', {}),
            'attack_indicators': up.get('attack_indicators', {}),
            'protocol_clusters': up.get('protocol_clusters', []),
            'flows': [
                {
                    'src': f.get('src', ''),
                    'dst': f.get('dst', ''),
                    'sport': f.get('sport', 0),
                    'dport': f.get('dport', 0),
                    'protocol': f.get('protocol', ''),
                    'risk_level': f.get('risk_level', 'none'),
                    'risk_score': f.get('risk_score', 0),
                    'indicators': f.get('indicators', {}),
                }
                for f in up.get('unknown_protocol_flows', [])[:20]
            ],
        }

    # Domain-aware analysis
    domain_info = result.get('domain_info')
    domain_summary = None
    if domain_info:
        domain_summary = {
            'in_domain_count': domain_info.get('in_domain_count', 0),
            'ood_count': domain_info.get('ood_count', 0),
            'ood_reclassified': domain_info.get('ood_reclassified', 0),
            'fusion_summary': domain_info.get('fusion_summary', {}),
            'flow_tags': domain_info.get('flow_tags', []),
        }

    protocol_inference = _serialize_protocol_inference(result.get('protocol_inference'))

    # Structural clustering
    clustering = result.get('clustering')
    cluster_summary = None
    if clustering:
        cs = clustering.get('summary', {})
        cluster_summary = {
            'total_clusters': cs.get('total_clusters', 0),
            'total_outliers': cs.get('total_outliers', 0),
            'risk_level': cs.get('risk_level', 'none'),
            'description': cs.get('description', ''),
            'clusters': [
                {
                    'cluster_id': c.get('cluster_id', 0),
                    'size': c.get('size', 0),
                    'inferred_name': c.get('inferred_name', '?'),
                    'avg_entropy': c.get('avg_entropy', 0),
                    'avg_payload_size': c.get('avg_payload_size', 0),
                    'avg_fields': c.get('avg_fields', 0),
                    'cohesion': c.get('cohesion', 'unknown'),
                    'outlier_count': c.get('outlier_count', 0),
                    'indicators': c.get('indicators', []),
                    'top_ports': c.get('top_ports', []),
                }
                for c in clustering.get('clusters', [])
            ],
        }

    # Temporal analysis
    temporal = result.get('temporal', {})
    temporal_summary = None
    if temporal:
        temporal_summary = {
            'total_windows': temporal.get('total_windows', 0),
            'anomalies': temporal.get('anomalies', []),
        }

    # Cross-validation
    cv = result.get('cross_validation')

    # Evaluation metrics
    metrics = result.get('metrics')

    report = {
        'meta': {
            'pcap_file': result.get('pcap_file', ''),
            'analysis_time': result.get('analysis_time', ''),
            'elapsed_seconds': result.get('elapsed_seconds', 0),
            'model_loaded': gnn.get('model_loaded', False),
        },
        'summary': {
            'total_flows': gnn.get('total_flows', 0),
            'normal_count': cls.get('normal_count', 0),
            'attack_count': cls.get('attack_count', 0),
            'attack_types': len(cls.get('attack_stats', {})),
        },
        'attacks': attack_summary,
        'nemesys': nemesys_summary,
        'protocol_inference': protocol_inference,
        'domain_info': domain_summary,
        'clustering': cluster_summary,
        'unknown_protocol': unknown_proto_summary,
        'temporal': temporal_summary,
        'cross_validation': cv,
        'metrics': metrics,
        'domain_aware_metrics': metrics.get('domain_aware') if metrics else None,
        'gnn_details': [
            {
                'flow_index': r.get('flow_index', i),
                'class_name': r.get('class_name', ''),
                'class_name_en': r.get('class_name_en', ''),
                'confidence': r.get('confidence', 0),
                'level': r.get('level', ''),
                'low_confidence': r.get('low_confidence', False),
                'protocol_prediction': r.get('protocol_prediction', {}),
                'fusion_status': r.get('fusion_decision', {}).get('status'),
                'fusion_risk_level': r.get('fusion_decision', {}).get('risk_level'),
            }
            for i, r in enumerate(gnn.get('results', []))
        ],
    }

    return json.dumps(report, indent=2, ensure_ascii=False, default=os.fspath)


def _serialize_protocol_inference(protocol_result, max_flows=50, max_messages=20):
    """Make protocol inference results JSON friendly."""
    if not protocol_result:
        return None

    flow_items = []
    for key, pred in (protocol_result.get('flow_predictions') or {}).items():
        flow_items.append({
            'flow_key': list(key) if isinstance(key, tuple) else key,
            'protocol': pred.get('protocol', 'UNKNOWN_PROTO'),
            'confidence': pred.get('confidence'),
            'is_known': pred.get('is_known', False),
            'message_count': pred.get('message_count', 0),
            'low_confidence_count': pred.get('low_confidence_count', 0),
            'nemesys_unknown_override_count': pred.get('nemesys_unknown_override_count', 0),
            'votes': pred.get('votes', {}),
            'model_votes': pred.get('model_votes', {}),
            'message_indices': pred.get('message_indices', []),
        })

    messages = []
    for item in (protocol_result.get('messages') or [])[:max_messages]:
        messages.append({
            'message_index': item.get('message_index'),
            'protocol': item.get('protocol', 'UNKNOWN_PROTO'),
            'raw_protocol': item.get('raw_protocol', 'UNKNOWN_PROTO'),
            'model_protocol': item.get('model_protocol', item.get('raw_protocol', 'UNKNOWN_PROTO')),
            'nemesys_protocol': item.get('nemesys_protocol'),
            'confidence': item.get('confidence', 0),
            'is_known': item.get('is_known', False),
            'is_low_confidence': item.get('is_low_confidence', True),
            'nemesys_unknown_override': item.get('nemesys_unknown_override', False),
            'error': item.get('error'),
        })

    return {
        'summary': protocol_result.get('summary', {}),
        'flow_predictions': flow_items[:max_flows],
        'messages': messages,
        'error': protocol_result.get('error'),
    }


def _generate_text(result):
    """Generate human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  nemesys-gnn4id 网络流量分析报告")
    lines.append("=" * 70)
    lines.append("")

    # 基本信息
    lines.append("一、基本信息")
    lines.append("-" * 40)
    lines.append(f"  PCAP文件:    {result.get('pcap_file', 'N/A')}")
    lines.append(f"  分析时间:    {result.get('analysis_time', 'N/A')}")
    lines.append(f"  耗时:        {result.get('elapsed_seconds', 0)}s")

    gnn = result.get('gnn', {})
    lines.append(f"  GNN模型:     {'已加载' if gnn.get('model_loaded') else '模拟模式'}")
    lines.append(f"  总flow数:    {gnn.get('total_flows', 0)}")
    lines.append("")

    # 分类统计
    cls = result.get('classification', {})
    domain_info = result.get('domain_info') or {}
    total = gnn.get('total_flows', 0)
    normal = cls.get('normal_count', 0)
    attack = cls.get('attack_count', 0)

    lines.append("二、流量分类统计（域感知修正后）")
    lines.append("-" * 40)
    lines.append(f"  正常流量:    {normal} ({normal/max(total,1)*100:.1f}%)")
    lines.append(f"  攻击流量:    {attack} ({attack/max(total,1)*100:.1f}%)")
    od_reclass = domain_info.get('ood_reclassified', 0)
    if od_reclass > 0:
        lines.append(f"  (其中 {od_reclass} 条域外流已排除，原始GNN分类作废)")
    lines.append("")

    # 攻击详情
    attack_stats = cls.get('attack_stats', {})
    if attack_stats:
        lines.append("三、攻击详情")
        lines.append("-" * 40)
        for name, stats in attack_stats.items():
            level = stats.get('level', 'unknown')
            level_zh = RISK_LEVEL_ZH.get(level, level)
            lines.append(f"  [{level_zh}] {name}")
            lines.append(f"    数量:       {stats.get('count', 0)} 条")
            low_n = stats.get('low_conf_count', 0)
            if low_n:
                lines.append(f"    低置信度:   {low_n} 条 (阈值 0.3)")
            lines.append(f"    置信度范围: {stats.get('min_confidence', 0):.3f} ~ {stats.get('max_confidence', 0):.3f}")
            lines.append(f"    平均置信度: {stats.get('avg_confidence', 0):.3f}")
            lines.append(f"    涉及flow:   {stats.get('flows', [])[:10]}{'...' if len(stats.get('flows', [])) > 10 else ''}")
            lines.append("")
    else:
        lines.append("三、攻击详情")
        lines.append("-" * 40)
        lines.append("  未检测到攻击流量")
        lines.append("")

    # NEMESYS 协议逆向
    nemesys = result.get('nemesys')
    if nemesys and 'summary' in nemesys:
        s = nemesys['summary']
        lines.append("四、NEMESYS 协议逆向分析")
        lines.append("-" * 40)
        lines.append(f"  分析模式:    {s.get('mode', 'N/A')}")
        lines.append(f"  消息数:      {s.get('total_messages', 0)}")
        lines.append(f"  平均字段数:  {s.get('avg_fields', 0)}")

        if 'total_payload_bytes' in s:
            lines.append(f"  总载荷字节:  {s.get('total_payload_bytes', 0)}")
            lines.append(f"  平均载荷大小: {s.get('avg_payload_size', 0)} bytes")

        if 'protocol_distribution' in s:
            lines.append(f"  协议分布:")
            for proto, count in s['protocol_distribution'].items():
                pct = count / max(s['total_messages'], 1) * 100
                lines.append(f"    {proto}: {count} 条 ({pct:.1f}%)")

        if 'field_type_distribution' in s and s['field_type_distribution']:
            lines.append(f"  BCDG字段类型分布:")
            for ft, count in s['field_type_distribution'].items():
                lines.append(f"    {ft}: {count} 个字段")

        if 'payload_analysis' in s and s['payload_analysis']:
            lines.append(f"  Payload特征分析:")
            for proto, analysis in s['payload_analysis'].items():
                lines.append(f"    {proto}:")
                lines.append(f"      平均大小: {analysis.get('avg_size', 0)} bytes")
                lines.append(f"      字节熵:   {analysis.get('entropy', 0)}")
                lines.append(f"      总字节:   {analysis.get('total_bytes', 0)}")

        if 'error' in nemesys:
            lines.append(f"  错误: {nemesys['error']}")

        lines.append("")
    elif nemesys and 'error' in nemesys:
        lines.append("四、NEMESYS 协议逆向分析")
        lines.append("-" * 40)
        lines.append(f"  分析失败: {nemesys['error']}")
        lines.append("")

    # 五、字段图协议识别
    protocol = result.get('protocol_inference')
    domain_info = result.get('domain_info')
    section_counter = 5
    if protocol and protocol.get('summary'):
        ps = protocol['summary']
        cn = _section_cn(section_counter)
        lines.append(f"{cn}、字段图协议识别")
        lines.append("-" * 40)
        if protocol.get('error'):
            lines.append(f"  推理失败: {protocol.get('error')}")
        lines.append(f"  消息数:        {ps.get('total_messages', 0)}")
        lines.append(f"  已知协议消息:  {ps.get('known_messages', 0)}")
        lines.append(f"  未知/低置信:   {ps.get('unknown_messages', 0)}")
        lines.append(f"  置信阈值:      {ps.get('confidence_threshold', 'N/A')}")
        if ps.get('protocol_distribution'):
            lines.append("  协议分布:")
            for proto, count in ps['protocol_distribution'].items():
                lines.append(f"    {proto}: {count} 条")
        lines.append("")
        section_counter += 1

    # 域感知分析
    if domain_info:
        cn = _section_cn(section_counter)
        lines.append(f"{cn}、域感知分析 (GNN训练域过滤)")
        lines.append("-" * 40)
        lines.append(f"  GNN训练域内流:   {domain_info.get('in_domain_count', 0)} 条 (可信任GNN分类)")
        lines.append(f"  域外流 (OOD):    {domain_info.get('ood_count', 0)} 条 (GNN结果不可靠)")
        od_reclass = domain_info.get('ood_reclassified', 0)
        if od_reclass > 0:
            lines.append(f"  域外攻击标记降级: {od_reclass} 条 → 改为结构异常检测")
        if domain_info.get('fusion_summary'):
            lines.append("  融合决策:")
            for status, count in domain_info['fusion_summary'].items():
                lines.append(f"    {status}: {count} 条")
        lines.append(f"  处理策略:        GNN仅对IoT域流量做8类细分，域外流量以结构聚类为准")
        lines.append("")
        section_counter += 1

    # 六、协议结构聚类
    clustering = result.get('clustering')
    if clustering:
        cs = clustering.get('summary', {})
        cn = _section_cn(section_counter)
        lines.append(f"{cn}、协议结构聚类分析")
        lines.append("-" * 40)
        lines.append(f"  协议簇总数:      {cs.get('total_clusters', 0)}")
        lines.append(f"  结构离群流数:    {cs.get('total_outliers', 0)} 条")
        lines.append(f"  整体风险:        {cs.get('risk_level', 'none')}")
        lines.append(f"  判断:            {cs.get('description', '')}")
        lines.append("")

        clusters = clustering.get('clusters', [])
        if clusters:
            lines.append(f"  各协议簇详情:")
            for c in clusters:
                outlier_s = f"离群{c['outlier_count']}条" if c.get('outlier_count', 0) > 0 else "无离群"
                indicators = c.get('indicators', [])
                flag_s = f" [{'|'.join(indicators)}]" if indicators else ""
                lines.append(f"    簇{c['cluster_id']}: {c.get('inferred_name', '?')} "
                             f"({c['size']}条, {outlier_s}, 紧密度={c['cohesion']}){flag_s}")
                lines.append(f"      平均熵: {c['avg_entropy']}, "
                             f"平均载荷: {c['avg_payload_size']}B, "
                             f"平均字段: {c['avg_fields']}")
                if c.get('top_ports'):
                    lines.append(f"      涉及端口: {c['top_ports']}")
            lines.append("")

        section_counter += 1

    # 时序异常检测
    temporal = result.get('temporal', {})
    if temporal and temporal.get('anomalies'):
        cn = _section_cn(section_counter)
        lines.append(f"{cn}、时序异常检测")
        lines.append("-" * 40)
        lines.append(f"  分析窗口数: {temporal.get('total_windows', 0)}")
        lines.append(f"  检测到异常: {len(temporal['anomalies'])} 个")
        for a in temporal['anomalies']:
            if a['type'] == 'attack_burst':
                lines.append(f"    [攻击突增] {a['window']}: {a['attack_count']} attacks ({a['attack_ratio']:.0%})")
            elif a['type'] == 'class_transition':
                lines.append(f"    [类型变化] {a['window']}: {a['new_classes']}")
        lines.append("")
        section_counter += 1

    # 交叉验证
    cv = result.get('cross_validation')
    if cv:
        cn = _section_cn(section_counter)
        lines.append(f"{cn}、GNN-NEMESYS 交叉验证")
        lines.append("-" * 40)
        lines.append(f"  确认攻击: {cv.get('confirmed', 0)}")
        lines.append(f"  可疑误报: {cv.get('suspicious', 0)}")
        lines.append(f"  潜在漏报: {cv.get('potential_miss', 0)}")
        details = cv.get('details', {})
        for s in details.get('suspicious', [])[:3]:
            lines.append(f"    [可疑] flow {s['flow_index']}: {s.get('reason', '')}")
        for m in details.get('potential_miss', [])[:3]:
            lines.append(f"    [漏报] flow {m['flow_index']}: {m.get('reason', '')}")
        lines.append("")
        section_counter += 1

    # 评估指标（如果有 ground truth）
    metrics = result.get('metrics')
    if metrics:
        cn = _section_cn(section_counter)
        lines.append(f"{cn}、分类评估指标")
        lines.append("-" * 40)
        lines.append(f"  评估样本数: {metrics.get('total_samples', 0)}")
        lines.append("")

        # 二分类指标（攻击 vs 正常）
        binary = metrics.get('binary', {})
        lines.append(f"  二分类指标 (攻击 vs 正常):")
        lines.append(f"    精确率 (Precision): {binary.get('precision', 0):.4f}")
        lines.append(f"    召回率 (Recall):    {binary.get('recall', 0):.4f}")
        lines.append(f"    F1 分数:            {binary.get('f1', 0):.4f}")
        lines.append(f"    误报率 (FPR):       {binary.get('fpr', 0):.4f}")
        lines.append(f"    混淆矩阵: TP={binary.get('tp', 0)} FP={binary.get('fp', 0)} "
                     f"FN={binary.get('fn', 0)} TN={binary.get('tn', 0)}")
        lines.append("")

        fusion_alert = metrics.get('fusion_alert', {})
        if fusion_alert:
            lines.append(f"  融合高风险告警指标:")
            lines.append(f"    精确率 (Precision): {fusion_alert.get('precision', 0):.4f}")
            lines.append(f"    召回率 (Recall):    {fusion_alert.get('recall', 0):.4f}")
            lines.append(f"    F1 分数:            {fusion_alert.get('f1', 0):.4f}")
            lines.append(f"    误报率 (FPR):       {fusion_alert.get('fpr', 0):.4f}")
            lines.append(f"    混淆矩阵: TP={fusion_alert.get('tp', 0)} FP={fusion_alert.get('fp', 0)} "
                         f"FN={fusion_alert.get('fn', 0)} TN={fusion_alert.get('tn', 0)}")
            lines.append("")

        # 整体指标
        macro = metrics.get('macro', {})
        weighted = metrics.get('weighted', {})
        lines.append(f"  整体指标 (多分类):")
        lines.append(f"    {'指标':<15} {'Macro':>10} {'Weighted':>10}")
        lines.append(f"    {'----':<15} {'-----':>10} {'--------':>10}")
        lines.append(f"    {'精确率':<15} {macro.get('precision', 0):>10.4f} {weighted.get('precision', 0):>10.4f}")
        lines.append(f"    {'召回率':<15} {macro.get('recall', 0):>10.4f} {weighted.get('recall', 0):>10.4f}")
        lines.append(f"    {'F1 分数':<15} {macro.get('f1', 0):>10.4f} {weighted.get('f1', 0):>10.4f}")
        lines.append(f"    {'误报率':<15} {macro.get('fpr', 0):>10.4f} {weighted.get('fpr', 0):>10.4f}")
        lines.append("")

        # 每个类别的指标
        per_class = metrics.get('per_class', {})
        if per_class:
            lines.append(f"  各类别指标:")
            lines.append(f"    {'类别':<12} {'精确率':>8} {'召回率':>8} {'F1':>8} {'FPR':>8} {'支持数':>8}")
            lines.append(f"    {'----':<12} {'------':>8} {'------':>8} {'--':>8} {'---':>8} {'------':>8}")
            for c_id in sorted(per_class.keys()):
                cm = per_class[c_id]
                lines.append(f"    {cm['name']:<12} {cm['precision']:>8.4f} {cm['recall']:>8.4f} "
                             f"{cm['f1']:>8.4f} {cm['fpr']:>8.4f} {cm['support']:>8}")
            lines.append("")

        # 域感知指标
        da = metrics.get('domain_aware')
        if da:
            lines.append(f"  域感知评估:")
            lines.append(f"    域内样本 (GNN可信): {da.get('in_domain_samples', 0)}")
            lines.append(f"    域外样本 (OOD排除): {da.get('ood_samples', 0)}")
            ood = da.get('ood')
            if ood:
                lines.append(f"    域外流若信任GNN的误报率: {ood.get('gnn_would_be_fpr', 0):.4f}")
                lines.append(f"    处理策略: {ood.get('description', '')}")
            in_domain = da.get('in_domain')
            if in_domain:
                in_bin = in_domain.get('binary', {})
                lines.append(f"    域内流二分类指标:")
                lines.append(f"      精确率: {in_bin.get('precision', 0):.4f}")
                lines.append(f"      召回率: {in_bin.get('recall', 0):.4f}")
                lines.append(f"      F1:     {in_bin.get('f1', 0):.4f}")
                lines.append(f"      FPR:    {in_bin.get('fpr', 0):.4f}")
            lines.append("")
        section_counter += 1

    # GNN 逐条结果（最多显示50条）
    gnn_results = gnn.get('results', [])
    if gnn_results:
        cn = _section_cn(section_counter)
        lines.append(f"{cn}、GNN 分类明细（前50条）")
        lines.append("-" * 40)
        lines.append(f"  {'序号':>6}  {'分类':>12}  {'置信度':>8}  {'等级':>8}")
        lines.append(f"  {'----':>6}  {'----':>12}  {'------':>8}  {'----':>8}")
        for r in gnn_results[:50]:
            tag = ' [低]' if r.get('low_confidence') else ''
            lines.append(f"  {r.get('flow_index', 0):>6}  {r.get('class_name', '?') + tag:<12}  "
                         f"{r.get('confidence', 0):>8.3f}  {r.get('level', '?'):>8}")
        if len(gnn_results) > 50:
            lines.append(f"  ... 共 {len(gnn_results)} 条，仅显示前50条")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append("=" * 70)

    return "\n".join(lines)


def _generate_markdown(result):
    """Generate Markdown report."""
    lines = []
    gnn = result.get('gnn', {})
    cls = result.get('classification', {})
    domain_info = result.get('domain_info', {})
    total = gnn.get('total_flows', 0)
    normal = cls.get('normal_count', 0)
    attack = cls.get('attack_count', 0)

    # 标题
    lines.append(f"# nemesys-gnn4id 流量分析报告")
    lines.append("")

    # 基本信息
    lines.append("## 一、基本信息")
    lines.append("")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| PCAP文件 | `{result.get('pcap_file', 'N/A')}` |")
    lines.append(f"| 分析时间 | {result.get('analysis_time', 'N/A')} |")
    lines.append(f"| 耗时 | {result.get('elapsed_seconds', 0)}s |")
    lines.append(f"| GNN模型 | {'已加载' if gnn.get('model_loaded') else '模拟模式'} |")
    lines.append(f"| 总flow数 | {total} |")
    lines.append("")

    # 分类统计
    lines.append("## 二、流量分类统计（域感知修正后）")
    lines.append("")
    od_reclass = domain_info.get('ood_reclassified', 0)
    lines.append(f"| 类别 | 数量 | 占比 |")
    lines.append(f"|------|------|------|")
    lines.append(f"| 正常流量 | {normal} | {normal/max(total,1)*100:.1f}% |")
    lines.append(f"| 攻击流量 | {attack} | {attack/max(total,1)*100:.1f}% |")
    if od_reclass > 0:
        lines.append("")
        lines.append(f"> **域感知过滤**: {od_reclass} 条域外流已排除，原始GNN分类作废")
    lines.append("")

    # 攻击详情
    attack_stats = cls.get('attack_stats', {})
    lines.append("## 三、攻击详情")
    lines.append("")
    if attack_stats:
        lines.append(f"| 等级 | 攻击类型 | 数量 | 置信度范围 | 平均置信度 | 低置信度 |")
        lines.append(f"|------|----------|------|-----------|-----------|---------|")
        for name, stats in attack_stats.items():
            level = stats.get('level', 'unknown')
            level_zh = RISK_LEVEL_ZH.get(level, level)
            low_n = stats.get('low_conf_count', 0)
            low_s = f"{low_n} 条" if low_n else "—"
            lines.append(f"| {level_zh} | {name} | {stats.get('count', 0)} | "
                         f"{stats.get('min_confidence', 0):.3f}~{stats.get('max_confidence', 0):.3f} | "
                         f"{stats.get('avg_confidence', 0):.3f} | {low_s} |")
    else:
        lines.append("未检测到攻击流量")
    lines.append("")

    # NEMESYS 协议逆向
    nemesys = result.get('nemesys')
    if nemesys and 'summary' in nemesys:
        s = nemesys['summary']
        lines.append("## 四、NEMESYS 协议逆向分析")
        lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 分析模式 | {s.get('mode', 'N/A')} |")
        lines.append(f"| 消息数 | {s.get('total_messages', 0)} |")
        lines.append(f"| 平均字段数 | {s.get('avg_fields', 0)} |")
        if 'total_payload_bytes' in s:
            lines.append(f"| 总载荷字节 | {s.get('total_payload_bytes', 0)} |")
            lines.append(f"| 平均载荷大小 | {s.get('avg_payload_size', 0)} bytes |")
        lines.append("")

        if 'protocol_distribution' in s:
            lines.append("### 协议分布")
            lines.append("")
            lines.append(f"| 协议 | 数量 | 占比 |")
            lines.append(f"|------|------|------|")
            for proto, count in s['protocol_distribution'].items():
                pct = count / max(s['total_messages'], 1) * 100
                lines.append(f"| {proto} | {count} | {pct:.1f}% |")
            lines.append("")

        if 'field_type_distribution' in s and s['field_type_distribution']:
            lines.append("### BCDG字段类型分布")
            lines.append("")
            lines.append(f"| 类型 | 数量 |")
            lines.append(f"|------|------|")
            for ft, count in s['field_type_distribution'].items():
                lines.append(f"| {ft} | {count} |")
            lines.append("")

        if 'payload_analysis' in s and s['payload_analysis']:
            lines.append("### Payload特征分析")
            lines.append("")
            lines.append(f"| 协议 | 平均大小 | 字节熵 | 总字节 |")
            lines.append(f"|------|----------|--------|--------|")
            for proto, analysis in s['payload_analysis'].items():
                lines.append(f"| {proto} | {analysis.get('avg_size', 0)}B | "
                             f"{analysis.get('entropy', 0)} | {analysis.get('total_bytes', 0)} |")
            lines.append("")

        if 'error' in nemesys:
            lines.append(f"> **错误**: {nemesys['error']}")
            lines.append("")
    elif nemesys and 'error' in nemesys:
        lines.append("## 四、NEMESYS 协议逆向分析")
        lines.append("")
        lines.append(f"分析失败: {nemesys['error']}")
        lines.append("")

    # 字段图协议识别
    protocol = result.get('protocol_inference')
    if protocol and protocol.get('summary'):
        ps = protocol['summary']
        lines.append("## 五、字段图协议识别")
        lines.append("")
        if protocol.get('error'):
            lines.append(f"> **推理失败**: {protocol.get('error')}")
            lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 消息数 | {ps.get('total_messages', 0)} |")
        lines.append(f"| 已知协议消息 | {ps.get('known_messages', 0)} |")
        lines.append(f"| 未知/低置信消息 | {ps.get('unknown_messages', 0)} |")
        lines.append(f"| 置信阈值 | {ps.get('confidence_threshold', 'N/A')} |")
        lines.append("")
        if ps.get('protocol_distribution'):
            lines.append("### 字段图协议分布")
            lines.append("")
            lines.append(f"| 协议 | 数量 |")
            lines.append(f"|------|------|")
            for proto, count in ps['protocol_distribution'].items():
                lines.append(f"| {proto} | {count} |")
            lines.append("")

    # 域感知分析
    if domain_info:
        lines.append("## 六、域感知分析 (GNN训练域过滤)")
        lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| GNN训练域内流 | {domain_info.get('in_domain_count', 0)} 条 (可信任GNN) |")
        lines.append(f"| 域外流 (OOD) | {domain_info.get('ood_count', 0)} 条 (GNN不可靠) |")
        od_reclass = domain_info.get('ood_reclassified', 0)
        if od_reclass > 0:
            lines.append(f"| 域外攻击降级 | {od_reclass} 条 → 改为结构异常检测 |")
        if domain_info.get('fusion_summary'):
            for status, count in domain_info['fusion_summary'].items():
                lines.append(f"| 融合决策 `{status}` | {count} 条 |")
        lines.append(f"| 处理策略 | GNN仅对IoT域流量做8类细分，域外流量以结构聚类为准 |")
        lines.append("")

    # 协议结构聚类
    clustering = result.get('clustering')
    if clustering:
        cs = clustering.get('summary', {})
        lines.append("## 七、协议结构聚类分析")
        lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 协议簇总数 | {cs.get('total_clusters', 0)} |")
        lines.append(f"| 结构离群流数 | {cs.get('total_outliers', 0)} 条 |")
        lines.append(f"| 整体风险 | `{cs.get('risk_level', 'none')}` |")
        lines.append(f"| 判断 | {cs.get('description', '')} |")
        lines.append("")

        clusters = clustering.get('clusters', [])
        if clusters:
            lines.append("### 各协议簇详情")
            lines.append("")
            lines.append(f"| 簇ID | 协议 | 数量 | 离群 | 紧密度 | 平均熵 | 平均载荷 | 平均字段 | 标记 |")
            lines.append(f"|------|------|------|------|--------|--------|----------|----------|------|")
            for c in clusters:
                outlier_s = f"{c['outlier_count']}条" if c.get('outlier_count', 0) > 0 else "无"
                indicators = c.get('indicators', [])
                flag_s = f"`{'|'.join(indicators)}`" if indicators else "—"
                lines.append(f"| {c['cluster_id']} | {c.get('inferred_name', '?')} | {c['size']} | "
                             f"{outlier_s} | {c['cohesion']} | {c['avg_entropy']} | "
                             f"{c['avg_payload_size']}B | {c['avg_fields']} | {flag_s} |")
                if c.get('top_ports'):
                    lines.append(f"| | 涉及端口 | {c['top_ports']} | | | | | | |")
            lines.append("")

    # 时序异常检测
    temporal = result.get('temporal', {})
    if temporal and temporal.get('anomalies'):
        lines.append("## 八、时序异常检测")
        lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 分析窗口数 | {temporal.get('total_windows', 0)} |")
        lines.append(f"| 检测到异常 | {len(temporal['anomalies'])} 个 |")
        lines.append("")
        for a in temporal['anomalies']:
            if a['type'] == 'attack_burst':
                lines.append(f"- **[攻击突增]** {a['window']}: {a['attack_count']} attacks ({a['attack_ratio']:.0%})")
            elif a['type'] == 'class_transition':
                lines.append(f"- **[类型变化]** {a['window']}: {a['new_classes']}")
        lines.append("")

    # 交叉验证
    cv = result.get('cross_validation')
    if cv:
        lines.append("## 九、GNN-NEMESYS 交叉验证")
        lines.append("")
        lines.append(f"| 项目 | 数量 |")
        lines.append(f"|------|------|")
        lines.append(f"| 确认攻击 | {cv.get('confirmed', 0)} |")
        lines.append(f"| 可疑误报 | {cv.get('suspicious', 0)} |")
        lines.append(f"| 潜在漏报 | {cv.get('potential_miss', 0)} |")
        lines.append("")

        details = cv.get('details', {})
        for s in details.get('suspicious', [])[:3]:
            lines.append(f"- [可疑] flow {s['flow_index']}: {s.get('reason', '')}")
        for m in details.get('potential_miss', [])[:3]:
            lines.append(f"- [漏报] flow {m['flow_index']}: {m.get('reason', '')}")
        lines.append("")

    # 评估指标
    metrics = result.get('metrics')
    if metrics:
        lines.append("## 十、分类评估指标")
        lines.append("")
        lines.append(f"**评估样本数**: {metrics.get('total_samples', 0)}")
        lines.append("")

        binary = metrics.get('binary', {})
        lines.append("### 二分类指标 (攻击 vs 正常)")
        lines.append("")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 精确率 (Precision) | {binary.get('precision', 0):.4f} |")
        lines.append(f"| 召回率 (Recall) | {binary.get('recall', 0):.4f} |")
        lines.append(f"| F1 分数 | {binary.get('f1', 0):.4f} |")
        lines.append(f"| 误报率 (FPR) | {binary.get('fpr', 0):.4f} |")
        lines.append(f"| 混淆矩阵 | TP={binary.get('tp', 0)} FP={binary.get('fp', 0)} FN={binary.get('fn', 0)} TN={binary.get('tn', 0)} |")
        lines.append("")

        fusion_alert = metrics.get('fusion_alert', {})
        if fusion_alert:
            lines.append("### 融合高风险告警指标")
            lines.append("")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|----|")
            lines.append(f"| 精确率 (Precision) | {fusion_alert.get('precision', 0):.4f} |")
            lines.append(f"| 召回率 (Recall) | {fusion_alert.get('recall', 0):.4f} |")
            lines.append(f"| F1 分数 | {fusion_alert.get('f1', 0):.4f} |")
            lines.append(f"| 误报率 (FPR) | {fusion_alert.get('fpr', 0):.4f} |")
            lines.append(f"| 混淆矩阵 | TP={fusion_alert.get('tp', 0)} FP={fusion_alert.get('fp', 0)} FN={fusion_alert.get('fn', 0)} TN={fusion_alert.get('tn', 0)} |")
            lines.append("")

        macro = metrics.get('macro', {})
        weighted = metrics.get('weighted', {})
        lines.append("### 整体指标 (多分类)")
        lines.append("")
        lines.append(f"| 指标 | Macro | Weighted |")
        lines.append(f"|------|-------|----------|")
        lines.append(f"| 精确率 | {macro.get('precision', 0):.4f} | {weighted.get('precision', 0):.4f} |")
        lines.append(f"| 召回率 | {macro.get('recall', 0):.4f} | {weighted.get('recall', 0):.4f} |")
        lines.append(f"| F1 分数 | {macro.get('f1', 0):.4f} | {weighted.get('f1', 0):.4f} |")
        lines.append(f"| 误报率 | {macro.get('fpr', 0):.4f} | {weighted.get('fpr', 0):.4f} |")
        lines.append("")

        per_class = metrics.get('per_class', {})
        if per_class:
            lines.append("### 各类别指标")
            lines.append("")
            lines.append(f"| 类别 | 精确率 | 召回率 | F1 | FPR | 支持数 |")
            lines.append(f"|------|--------|--------|-----|------|--------|")
            for c_id in sorted(per_class.keys()):
                cm = per_class[c_id]
                lines.append(f"| {cm['name']} | {cm['precision']:.4f} | {cm['recall']:.4f} | "
                             f"{cm['f1']:.4f} | {cm['fpr']:.4f} | {cm['support']} |")
            lines.append("")

        da = metrics.get('domain_aware')
        if da:
            lines.append("### 域感知评估")
            lines.append("")
            lines.append(f"| 项目 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| 域内样本 (GNN可信) | {da.get('in_domain_samples', 0)} |")
            lines.append(f"| 域外样本 (OOD排除) | {da.get('ood_samples', 0)} |")
            ood = da.get('ood')
            if ood:
                lines.append(f"| 域外流若信任GNN误报率 | {ood.get('gnn_would_be_fpr', 0):.4f} |")
                lines.append(f"| 处理策略 | {ood.get('description', '')} |")
            in_domain = da.get('in_domain')
            if in_domain:
                in_bin = in_domain.get('binary', {})
                lines.append(f"| 域内流精确率 | {in_bin.get('precision', 0):.4f} |")
                lines.append(f"| 域内流召回率 | {in_bin.get('recall', 0):.4f} |")
                lines.append(f"| 域内流F1 | {in_bin.get('f1', 0):.4f} |")
                lines.append(f"| 域内流FPR | {in_bin.get('fpr', 0):.4f} |")
            lines.append("")

    # GNN 逐条结果
    gnn_results = gnn.get('results', [])
    if gnn_results:
        lines.append("## 十一、GNN 分类明细（前50条）")
        lines.append("")
        lines.append(f"| 序号 | 分类 | 置信度 | 等级 |")
        lines.append(f"|------|------|--------|------|")
        for r in gnn_results[:50]:
            tag = ' [低]' if r.get('low_confidence') else ''
            lines.append(f"| {r.get('flow_index', 0)} | {r.get('class_name', '?')}{tag} | "
                         f"{r.get('confidence', 0):.3f} | {r.get('level', '?')} |")
        if len(gnn_results) > 50:
            lines.append(f"| ... | 共 {len(gnn_results)} 条，仅显示前50条 | | |")
        lines.append("")

    lines.append("---")
    lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    return "\n".join(lines)


def _section_cn(n):
    """将数字转为中文序号 (5→五, 6→六, 7→七, 8→八, 9→九, 10→十)"""
    return _CN_NUMBERS.get(n, str(n))
