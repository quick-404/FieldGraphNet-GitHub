"""生成 GNN+NEMESYS 集成方法说明 Word 文档"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()

# ── 全局样式 ──
style = doc.styles['Normal']
style.font.name = '微软雅黑'
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

for s in ['Heading 1', 'Heading 2', 'Heading 3']:
    hs = doc.styles[s]
    hs.font.name = '微软雅黑'
    hs.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    hs.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)

# ── 封面 ──
doc.add_paragraph()
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('GNN4ID + NEMESYS\n集成架构说明')
run.font.size = Pt(22)
run.font.bold = True
run.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)

doc.add_paragraph()
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run('基于图神经网络的入侵检测 × 协议逆向工程\n六步流水线集成方案')
run.font.size = Pt(12)
run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

doc.add_paragraph()
doc.add_paragraph()
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
info.add_run('2026-05-24').font.size = Pt(11)

doc.add_page_break()

# ── 目录页 ──
doc.add_heading('目录', level=1)
toc = [
    '1. 架构总览',
    '2. Step 1 — GNN4ID 流量分类',
    '3. Step 2 — 攻击/正常流量分离',
    '4. Step 3 — NEMESYS 协议逆向分析',
    '5. Step 4 — 时序异常检测',
    '6. Step 5 — 未知协议攻击检测',
    '7. Step 6 — GNN-NEMESYS 交叉验证',
    '8. 集成结合点总结',
    '9. 评估指标体系',
]
for t in toc:
    p = doc.add_paragraph(t)
    p.paragraph_format.space_after = Pt(4)

doc.add_page_break()

# ═══════════════════════════════════════
# 1. 架构总览
# ═══════════════════════════════════════
doc.add_heading('1. 架构总览', level=1)

doc.add_paragraph(
    'GNN4ID 负责流量分类（这是不是攻击），NEMESYS 负责协议逆向（这到底是什么协议）。'
    '两者串联为六步流水线，从原始 PCAP 到综合报告全自动完成。'
)

doc.add_paragraph(
    'PCAP → [Step 1: GNN分类] → [Step 2: 分离攻击/正常] → [Step 3: NEMESYS协议逆向]\n'
    '          │                          │\n'
    '          ↓                          ↓\n'
    '  [Step 4: 时序异常]        [Step 5: 未知协议攻击检测]\n'
    '          │                          │\n'
    '          └────→ [Step 6: 交叉验证] ←──┘\n'
    '                       │\n'
    '                       ↓\n'
    '                  综合报告'
)

doc.add_paragraph(
    '技术栈：nfstream(特征提取) + PyTorch Geometric(异构图) + HeteroGNN(SAGEConv) '
    '+ BCDG(字节级分段) + 协议指纹匹配 + 多维攻击指标检测'
)

# ═══════════════════════════════════════
# 2. Step 1
# ═══════════════════════════════════════
doc.add_heading('2. Step 1 — GNN4ID 流量分类', level=1)

doc.add_heading('2.1 特征提取（nfstream + 自定义插件）', level=2)
doc.add_paragraph(
    '使用 nfstream NFStreamer 从 PCAP 提取流级特征，搭配自定义 FlowPacketExtractor 插件'
    '捕获每条流中前 20 个数据包的 payload、方向、TCP 标志、包大小等逐包信息。'
)

doc.add_paragraph('输出：91 列原始 DataFrame（含 udps.* 逐包列表列）', style='List Bullet')
doc.add_paragraph('后处理 _add_flow_features_column()：从逐包数据计算 82 维特征向量', style='List Bullet')

doc.add_heading('2.2 82 维特征结构', level=2)

# 特征表格
table = doc.add_table(rows=9, cols=3, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table.rows[0].cells
hdr[0].text = '维度范围'
hdr[1].text = '特征组'
hdr[2].text = '内容说明'
for i, c in enumerate(hdr):
    for p in c.paragraphs:
        for r in p.runs:
            r.font.bold = True

rows_data = [
    ('[0-5]', '双向基础统计', 'duration, packets, bytes (src→dst / dst→src)'),
    ('[6-9]', '双向包大小统计', 'min/mean/stddev/max (bidirectional)'),
    ('[10-17]', '方向包大小统计', 'min/mean/stddev/max (src→dst / dst→src)'),
    ('[18-29]', 'PIAT 统计', 'Packet Inter-Arrival Time 的 min/mean/stddev/max'),
    ('[30-37]', 'src→dst TCP标志', 'SYN/CWR/ECE/URG/ACK/PSH/RST/FIN 计数'),
    ('[38-45]', 'dst→src TCP标志', '同上，按方向分离'),
    ('[47-74]', '滚动窗口特征', '协议类型、端口特征、DNS/HTTP标记等（流级近似）'),
    ('[75-81]', 'One-hot 编码', 'expiration_id + protocol(1/2/6/17/58)'),
]
for i, (rng, name, desc) in enumerate(rows_data):
    row = table.rows[i + 1].cells
    row[0].text = rng
    row[1].text = name
    row[2].text = desc

doc.add_paragraph()

doc.add_heading('2.3 异构图构建（GraphBuilder）', level=2)
doc.add_paragraph(
    '每条 flow 构建一个 HeteroData 图对象：\n'
    '• flow 节点（1 个）：82 维特征向量\n'
    '• packet 节点（N 个）：payload_size + delta_time + direction + TCP flags (12维)\n'
    '• contain 边：flow → packet（携带逐包特征作为 edge_attr）\n'
    '• sequential 边：packet[i] → packet[i+1]（携带间隔时间）'
)

doc.add_heading('2.4 GNN 推理', level=2)
doc.add_paragraph(
    '模型：HeteroGNN × SAGEConv（隐藏层 64 维，注意力 32 维）\n'
    '训练集：CIC-IoT-2023（IoT 入侵检测数据集）\n'
    '输出：8 类 softmax → (class_id, class_name, confidence)'
)

# 8类表格
table2 = doc.add_table(rows=9, cols=4, style='Light Grid Accent 1')
table2.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr2 = table2.rows[0].cells
hdr2[0].text = 'ID'
hdr2[1].text = '类别'
hdr2[2].text = '英文'
hdr2[3].text = '风险等级'
for i, c in enumerate(hdr2):
    for p in c.paragraphs:
        for r in p.runs:
            r.font.bold = True
classes = [
    ('0', '正常流量', 'Benign', '安全'),
    ('1', 'Web攻击', 'WebBased', '高危'),
    ('2', '欺骗攻击', 'Spoofing', '中危'),
    ('3', '侦察攻击', 'Recon', '中危'),
    ('4', 'Mirai', 'Mirai', '高危'),
    ('5', 'DoS攻击', 'Dos', '高危'),
    ('6', 'DDoS攻击', 'DDos', '高危'),
    ('7', '暴力破解', 'BruteForce', '中危'),
]
for i, (cid, cn, en, lv) in enumerate(classes):
    row = table2.rows[i + 1].cells
    row[0].text = cid
    row[1].text = cn
    row[2].text = en
    row[3].text = lv

# ═══════════════════════════════════════
# 3. Step 2
# ═══════════════════════════════════════
doc.add_heading('3. Step 2 — 攻击/正常流量分离', level=1)

doc.add_paragraph(
    '分类规则（_classify_flows）：\n'
    '• class_id == 0 → 正常流量\n'
    '• class_id != 0 → 攻击流量\n'
    '• confidence < 0.3 → 附加 low_confidence 标记（仅提示，不影响分类决策）\n\n'
    '输出：按攻击类型分组统计，含数量、置信度范围、平均置信度、低置信度计数。'
)

# ═══════════════════════════════════════
# 4. Step 3
# ═══════════════════════════════════════
doc.add_heading('4. Step 3 — NEMESYS 协议逆向分析', level=1)

doc.add_paragraph(
    '仅对 Step 2 标记的攻击流量做深度分析。若无攻击流则对整个 PCAP 做分析。'
)

doc.add_heading('4.1 BCDG 字节级分段', level=2)
doc.add_paragraph(
    'BCDG（Bit Congruence Delta Gauss）算法流程：\n'
    '1. 计算相邻字节的 bit congruence（8位中相同位的比例，0~1）\n'
    '2. 一阶差分（delta），捕获 congruence 突变点\n'
    '3. 高斯滤波平滑（scipy.gaussian_filter1d, σ=0.6），去噪\n'
    '4. 拐点检测（min-max 上升对），作为字段边界\n'
    '5. 输出字段列表：[(offset, length), ...]\n\n'
    '示例：DNS 查询 28 字节 → 分出 header(12B) + question(16B)，字段数 = 7'
)

doc.add_heading('4.2 字段类型推断', level=2)
doc.add_paragraph(
    '对每个 BCDG 字段分析内容类型：\n'
    '• 全可打印字符 → text\n'
    '• 含不可打印字节 → binary\n'
    '• 全零 → zero\n\n'
    '统计整条流的 binary/text/zero 比例，用于后续攻击指标和交叉验证。'
)

doc.add_heading('4.3 协议识别', level=2)
doc.add_paragraph('两层识别策略：')

doc.add_paragraph('第一层：端口查表（PORT_PROTOCOL_MAP）', style='List Bullet')
# 端口表
table3 = doc.add_table(rows=5, cols=4, style='Light Grid Accent 1')
table3.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr3 = table3.rows[0].cells
hdr3[0].text = '端口'
hdr3[1].text = '协议'
hdr3[2].text = '端口'
hdr3[3].text = '协议'
for p in hdr3:
    for r in p.paragraphs:
        for r2 in r.runs:
            r2.font.bold = True
ports = [
    ('53', 'DNS', '80/8080', 'HTTP'),
    ('67/68', 'DHCP', '443', 'TLS'),
    ('123', 'NTP', '22', 'SSH'),
    ('502', 'Modbus*', '3306', 'MySQL'),
]
for i, (p1, n1, p2, n2) in enumerate(ports):
    row = table3.rows[i + 1].cells
    row[0].text = p1
    row[1].text = n1
    row[2].text = p2
    row[3].text = n2

doc.add_paragraph()  # spacing
doc.add_paragraph('第二层：Payload 指纹匹配', style='List Bullet')
doc.add_paragraph(
    '• TLS：首字节 0x16 0x03 → Client/Server Hello\n'
    '• HTTP：首4字节匹配 GET/POST/PUT /HEAD/OPTN/DEL /PATC/CONN/TRAC\n'
    '• SSH：首4字节 b\'SSH-\'\n'
    '• DNS：第3-4字节为 0x00 0x00 且端口=53\n'
    '• VNC：首4字节 b\'RFB \'\n'
    '• FTP：首4字节 b\'220 \' 且端口=21\n'
    '• SOCKS4/5：版本字节 0x04/0x05\n'
    '• 均未命中 → UNKNOWN_PROTO'
)

doc.add_paragraph(
    '注意：Modbus(502)、DNP3(20000)、S7Comm(102) 等工控协议不在指纹库中，'
    '当前返回 UNKNOWN_PROTO。'
)

doc.add_heading('4.4 能力边界说明（重要）', level=2)

doc.add_paragraph(
    'NEMESYS 核心能力是 BCDG 字节级分段（找到字段边界），不是协议识别。'
    '协议名称（DNS/DHCP/NTP 等）是通过端口查表和 payload 指纹匹配获得的，'
    '与 BCDG 算法无关。'
)

table_cap = doc.add_table(rows=6, cols=3, style='Light Grid Accent 1')
table_cap.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr_cap = table_cap.rows[0].cells
hdr_cap[0].text = '能力'
hdr_cap[1].text = '谁负责'
hdr_cap[2].text = '实际情况'
for c in hdr_cap:
    for p in c.paragraphs:
        for r in p.runs:
            r.font.bold = True
cap_data = [
    ('流量是否为攻击', 'GNN (HeteroGNN)', '仅在训练见过的协议上有效\n(CIC-IoT-2023 IoT流量)'),
    ('字段边界在哪', 'NEMESYS (BCDG)', '通用算法，任何协议都能分段'),
    ('这是什么协议(名称)', '端口表 + Payload指纹', '覆盖约20种常见协议\n工控协议(Modbus/DNP3/S7Comm)无法识别'),
    ('未知协议有无攻击嫌疑', 'Step 5 启发式指标', '5个指标打分，非精确判断\n不识别协议本身'),
    ('攻击类型细分', 'GNN (HeteroGNN)', '8类分类，取决于训练集覆盖'),
]
for i, (cap, who, actual) in enumerate(cap_data):
    row = table_cap.rows[i + 1].cells
    row[0].text = cap
    row[1].text = who
    row[2].text = actual

doc.add_paragraph()

doc.add_paragraph(
    '关键认知：\n'
    '• NEMESYS ≠ 协议识别工具，它是协议结构分析工具（字段边界+类型）\n'
    '• GNN ≠ 未知协议检测器，它是已知攻击模式分类器（8类）\n'
    '• Step 5 是唯一面向"不认识的东西"的检测环节，但用的是旁路推断而非精确识别'
)

# ═══════════════════════════════════════
# 5. Step 4
# ═══════════════════════════════════════
doc.add_heading('5. Step 4 — 时序异常检测', level=1)

doc.add_paragraph(
    '使用 Step 2 的分类结果（attack_indices 集合），按时间窗聚合分析。\n\n'
    '窗口划分：window_size = max(5, total_flows / 10)，按 flow 顺序滑窗\n\n'
    '每窗统计：总流数、攻击流数、攻击比例、攻击类型分布\n\n'
    '异常判定：\n'
    '• 攻击突增：attack_ratio > 0.5 且 attack_count ≥ 3\n'
    '• 攻击类型转移：前后窗出现新的攻击类别（如 侦察→漏洞利用）'
)

# ═══════════════════════════════════════
# 6. Step 5
# ═══════════════════════════════════════
doc.add_heading('6. Step 5 — 未知协议攻击检测', level=1)

doc.add_paragraph(
    'UnknownProtocolDetector 专门处理 NEMESYS 识别为 UNKNOWN_PROTO 的流量，'
    '判断其是否具有攻击特征。'
)

doc.add_heading('6.1 五维攻击指标', level=2)
table4 = doc.add_table(rows=6, cols=3, style='Light Grid Accent 1')
table4.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr4 = table4.rows[0].cells
hdr4[0].text = '指标'
hdr4[1].text = '检测逻辑'
hdr4[2].text = '阈值'
for c in hdr4:
    for p in c.paragraphs:
        for r in p.runs:
            r.font.bold = True
indicators = [
    ('可疑端口', '端口在 suspicious_ports 表中', 'Metasploit(4444), BackOrifice(31337) 等'),
    ('非标准端口', '端口不在 PORT_PROTOCOL_MAP 中', '无匹配'),
    ('高熵载荷', 'Shannon 熵 ≥ 阈值', '7.0（加密/混淆特征）'),
    ('高二进制字段比', 'BCD段中 binary 类型占比 ≥ 阈值', '70%'),
    ('异常载荷大小', 'payload < 4 字节 或 > 1400 字节', '4 / 1400'),
]
for i, (name, logic, thresh) in enumerate(indicators):
    row = table4.rows[i + 1].cells
    row[0].text = name
    row[1].text = logic
    row[2].text = thresh

doc.add_paragraph()

doc.add_heading('6.2 风险等级判定', level=2)
doc.add_paragraph(
    '• 命中 3+ 个指标 → 高危\n'
    '• 命中 2 个指标 → 中危\n'
    '• 命中 1 个指标 → 低危\n'
    '• 命中 0 个指标 → 安全'
)

doc.add_heading('6.3 协议聚类', level=2)
doc.add_paragraph(
    '对未知协议流按 (端口组合, 字段数范围) 聚类，同类协议归为一组。\n'
    '输出每个聚类的：流数、平均熵、平均载荷大小、主要字段类型、风险等级。\n\n'
    '示例（Modbus 100 条流）：\n'
    '• 聚类1: 9.4字节小包, binary字段~3, 62条, 中危\n'
    '• 聚类3: 11.9字节小包, binary字段~4, 30条, 高危\n'
    '• 聚类4: 260字节中等包, binary字段~10, 6条, 高危'
)

# ═══════════════════════════════════════
# 7. Step 6
# ═══════════════════════════════════════
doc.add_heading('7. Step 6 — GNN-NEMESYS 交叉验证', level=1)

doc.add_paragraph(
    '用 NEMESYS 的协议识别结果反向校验 GNN 分类，输出三种结果：'
)

doc.add_heading('7.1 确认攻击（confirmed）', level=2)
doc.add_paragraph(
    'GNN 判为攻击，且 NEMESYS 识别协议不在已知正常协议白名单中。\n'
    '附加条件：若整体 BCDG 分析中 binary 字段占比 > 60%，且该流不在可疑列表中，'
    '追加到确认攻击（高二进制 payload 是恶意载荷的典型特征）。'
)

doc.add_heading('7.2 可疑误报（suspicious）', level=2)
doc.add_paragraph(
    'GNN 判为攻击，但 NEMESYS 识别出已知正常协议（DNS/DHCP/NTP/HTTP/TLS 等）。\n'
    '例如：GNN 将 DNS 流分类为 Web攻击，但 NEMESYS 明确识别出这是 DNS 协议 → 标记为可疑误报。'
)

doc.add_heading('7.3 潜在漏报（potential_miss）', level=2)
doc.add_paragraph(
    'GNN 判为正常，但具备以下特征之一：\n'
    '• 使用已知可疑端口（Metasploit 4444、BackOrifice 31337 等）\n'
    '• 使用未知协议且非标准端口\n'
    '→ 标记为潜在漏报，建议人工复核。'
)

# ═══════════════════════════════════════
# 8. 结合点总结
# ═══════════════════════════════════════
doc.add_heading('8. 集成结合点总结', level=1)

table5 = doc.add_table(rows=7, cols=4, style='Light Grid Accent 1')
table5.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr5 = table5.rows[0].cells
hdr5[0].text = '步骤'
hdr5[1].text = 'GNN 的作用'
hdr5[2].text = 'NEMESYS 的作用'
hdr5[3].text = '结合方式'
for c in hdr5:
    for p in c.paragraphs:
        for r in p.runs:
            r.font.bold = True
integration = [
    ('Step 1-2', '流量分类 + 攻击/正常分离', '—', 'GNN 主导'),
    ('Step 3', '—', 'BCDG分段 + 协议识别', '仅分析 GNN 标记的攻击流'),
    ('Step 4', '提供 classified 结果', '—', '用 Step 2 输出做攻击比例统计'),
    ('Step 5', '—', '提供分段结果 + 协议字段', '检测 GNN 没见过的可疑协议'),
    ('Step 6', '提供分类结果', '提供协议识别', '双向校验：协议验证分类，分类发现异常协议'),
    ('报告', '输出分类明细、指标', '输出协议分布、字段结构', '合并为统一 JSON/文本报告'),
]
for i, (step, gnn, nem, combine) in enumerate(integration):
    row = table5.rows[i + 1].cells
    row[0].text = step
    row[1].text = gnn
    row[2].text = nem
    row[3].text = combine

doc.add_paragraph()

doc.add_paragraph(
    '核心设计思路：GNN 管"这是不是攻击"，NEMESYS 管"这到底是什么协议"。\n'
    '两者通过交叉验证步骤相互校验，减少单一模型的误报和漏报。'
)

# ═══════════════════════════════════════
# 9. 评估指标
# ═══════════════════════════════════════
doc.add_heading('9. 评估指标体系', level=1)

doc.add_paragraph(
    'Pipeline 内置分类评估功能，需要提供 ground truth 标签文件（CSV）。'
)

doc.add_heading('9.1 指标计算', level=2)
doc.add_paragraph(
    '每个类别（8 类）计算：TP、FP、FN、TN → Precision、Recall、F1、FPR\n\n'
    '整体指标：\n'
    '• Macro 平均：各类等权平均\n'
    '• Weighted 平均：按各类样本数加权\n\n'
    '二分类指标（攻击 vs 正常）：\n'
    '• 将所有 class_id > 0 的预测合并为"攻击"类别\n'
    '• 计算 Precision / Recall / F1 / FPR'
)

doc.add_heading('9.2 使用方式', level=2)
doc.add_paragraph(
    'CLI：python cli.py analyze capture.pcap --labels labels.csv\n\n'
    '标签 CSV 格式：每行一个整数 0-7，对应 ATTACK_TYPES 的 class_id\n'
    '标签数量 ≥ 流程量，不足时自动截断'
)

doc.add_heading('9.3 当前模型评估结果（2026-05-24）', level=2)
doc.add_paragraph(
    '测试集：6 个 PCAP（DHCP/DNS/NTP/Modbus/DNP3/S7Comm，各 100 包，全部 Benign）\n'
    '共 121 条 flow，仅 DNS 有 1 条正确识别，整体准确率 0.83%。\n\n'
    '结论：GNN 模型（CIC-IoT-2023 训练）对非 IoT 协议（网络服务/工控）无判别力，'
    '需要协议白名单前置过滤或扩充训练数据。'
)

# ── 保存 ──
output_path = 'D:/网络流量分析项目/nemesys-gnn4id/GNN4ID-NEMESYS集成架构说明.docx'
doc.save(output_path)
print(f'文档已保存: {output_path}')
