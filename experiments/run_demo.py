"""NG4Id 系统完整流水线演示"""
import sys
sys.path.insert(0, 'src')
from nemesys_gnn4id.pipeline import TrafficPipeline

p = TrafficPipeline(device='cpu', proto_model_path='field_proto_model.pth')
result = p.analyze('data/data/http_100.pcap', max_flows=5)

print()
print('=' * 70)
print('  NG4Id 系统运行结果汇总')
print('=' * 70)
print()

n = result.get('nemesys', {}).get('summary', {})
print('[1] NEMESYS BCDG 字段分割')
print('  分析消息: %d 条  |  平均字段: %.2f 个/条' % (n.get('total_messages',0), n.get('avg_fields',0)))
print('  协议分布: %s' % n.get('protocol_distribution', {}))
print()

pr = result.get('protocol_inference', {}).get('summary', {})
print('[2] FieldProtoGNN 字段图协议识别')
print('  已知协议: %d 条  |  未知/低置信: %d 条' % (pr.get('known_messages',0), pr.get('unknown_messages',0)))
print('  协议分布: %s' % pr.get('protocol_distribution', {}))
print()

d = result.get('domain_info', {})
print('[3] 域路由器')
print('  域内流: %d  |  域外流: %d' % (d['in_domain_count'], d['ood_count']))
print()

print('[4] GNN4ID 攻击检测（域内过滤模式）')
for r in result['gnn']['results']:
    cn = r.get('class_name_en', r['class_name'])
    conf = r.get('confidence', 0)
    print('  流 %d: %s (置信度=%.3f, 已跳过=%s)' % (r['flow_index'], cn, conf, r.get('gnn_skipped')))
print()

print('[5] 结构指纹聚类')
cl = result.get('clustering', {}).get('summary', {})
print('  聚类数: %d  |  离群点: %d  |  风险等级: %s' % (cl.get('total_clusters',0), cl.get('total_outliers',0), cl.get('risk_level','none')))
print()

print('[6] 融合决策')
for status, count in d.get('fusion_summary', {}).items():
    print('  %s: %d 条' % (status, count))
print()

print('[7] 交叉验证')
cv = result.get('cross_validation', {})
if cv:
    print('  确认攻击: %d  |  可疑误报: %d  |  潜在漏报: %d' % (cv.get('confirmed',0), cv.get('suspicious',0), cv.get('potential_miss',0)))
    print('  结构聚类离群告警: %d' % cv.get('cluster_outlier_alerts',0))
print()
print('=' * 70)
print('  系统运行完成  |  耗时: %.1fs' % result['elapsed_seconds'])
print('=' * 70)
