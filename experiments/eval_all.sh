#!/bin/bash
# ==============================================================
# nemesys-gnn4id 完整评估脚本
# 对 6 个测试 PCAP 运行 GNN 推理 + 评估指标计算
# ==============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_DIR/data/data"
LABELS_DIR="$PROJECT_DIR/data/labels"
OUTPUT_DIR="$PROJECT_DIR/eval_results"
MODEL_PATH="$HOME/网络流量分析项目/GNN4ID/model.pth"

# 测试 PCAP 列表
PCAPS=("dhcp_100" "dns_100" "ntp_100" "modbus_100" "dnp3_100" "s7comm_100")

echo "============================================================"
echo "  nemesys-gnn4id 完整评估"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "模型路径: $MODEL_PATH"
echo "数据目录: $DATA_DIR"
echo "标签目录: $LABELS_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查模型
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查依赖
if ! python3 -c "import nfstream" 2>/dev/null; then
    echo "WARNING: nfstream 未安装，将使用 scapy 降级模式（特征不准确）"
fi

mkdir -p "$OUTPUT_DIR"

# 逐项评估
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
echo "# 评估结果汇总" > "$SUMMARY_FILE"
echo "# 时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "# 模型: $MODEL_PATH" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

TOTAL=0
PASS=0
FAIL=0

for name in "${PCAPS[@]}"; do
    PCAP="$DATA_DIR/${name}.pcap"
    LABEL="$LABELS_DIR/${name}.csv"
    OUTPUT_TXT="$OUTPUT_DIR/${name}_report.txt"
    OUTPUT_JSON="$OUTPUT_DIR/${name}_report.json"
    OUTPUT_MD="$OUTPUT_DIR/${name}_report.md"

    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] 分析: $name"
    echo "  PCAP: $PCAP"
    echo "  Labels: $LABEL"
    echo ""

    if python3 "$PROJECT_DIR/cli.py" analyze "$PCAP" \
        --model "$MODEL_PATH" \
        --labels "$LABEL" \
        --output "$OUTPUT_TXT" \
        --format text \
        --max-flows 200 \
        2>&1 | tail -20; then
        echo "  [OK] $name 分析完成"
        echo "$name: OK" >> "$SUMMARY_FILE"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name 分析失败"
        echo "$name: FAIL" >> "$SUMMARY_FILE"
        FAIL=$((FAIL + 1))
    fi

    # 同时输出 JSON
    python3 "$PROJECT_DIR/cli.py" analyze "$PCAP" \
        --model "$MODEL_PATH" \
        --labels "$LABEL" \
        --output "$OUTPUT_JSON" \
        --format json \
        --max-flows 200 \
        2>/dev/null && echo "  JSON报告: $OUTPUT_JSON"

    # 同时输出 Markdown
    python3 "$PROJECT_DIR/cli.py" analyze "$PCAP" \
        --model "$MODEL_PATH" \
        --labels "$LABEL" \
        --output "$OUTPUT_MD" \
        --format md \
        --max-flows 200 \
        2>/dev/null && echo "  Markdown报告: $OUTPUT_MD"

    TOTAL=$((TOTAL + 1))
    echo ""
done

echo "============================================================"
echo "  评估完成"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  总计: $TOTAL, 成功: $PASS, 失败: $FAIL"
echo "  结果目录: $OUTPUT_DIR"
echo "============================================================"

echo "" >> "$SUMMARY_FILE"
echo "# 结果: $PASS / $TOTAL 成功" >> "$SUMMARY_FILE"

# 汇总指标
echo ""
echo "--- 各PCAP评估指标汇总 ---"
for name in "${PCAPS[@]}"; do
    JSON="$OUTPUT_DIR/${name}_report.json"
    if [ -f "$JSON" ]; then
        echo ""
        echo "=== $name ==="
        python3 -c "
import json
with open('$JSON') as f:
    data = json.load(f)
metrics = data.get('metrics', {})
da = metrics.get('domain_aware', {}) if metrics else {}

# 域感知概览
ood_samples = da.get('ood_samples', 0) if da else 0
in_domain_samples = da.get('in_domain_samples', 0) if da else 0
if da:
    print(f\"  域内样本: {in_domain_samples}  域外(OOD): {ood_samples}\")

# 原始GNN指标（包含OOD流，仅供参考）
if metrics:
    binary = metrics.get('binary', {})
    print(f\"  原始GNN FPR(含OOD): {binary.get('fpr', 'N/A')}\")

# 域感知指标
in_domain = da.get('in_domain') if da else None
if in_domain:
    in_bin = in_domain.get('binary', {})
    print(f\"  域内GNN FPR(可信):  {in_bin.get('fpr', 'N/A')}\")
    print(f\"  域内GNN F1:         {in_bin.get('f1', 'N/A')}\")
elif in_domain_samples > 0:
    print(f\"  域内GNN: 计算异常\")
else:
    print(f\"  域内GNN: 无域内样本，跳过GNN评估\")

ood = da.get('ood') if da else None
if ood:
    print(f\"  域外流若信任GNN误报率: {ood.get('gnn_would_be_fpr', 'N/A')}\")
elif ood_samples > 0:
    print(f\"  域外流: {ood_samples} 条 (指标数据缺失)\")
else:
    print(f\"  域外流: 0 条\")
"
    fi
done
