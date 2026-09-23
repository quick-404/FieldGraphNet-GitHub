#!/bin/bash
# 消融实验批量运行脚本
# 用法: bash eval_ablation.sh

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo " FieldProtoGNN 消融实验"
echo "=========================================="

# Step 1: 训练 10 类模型 (GAT, drop-ports) — 若已有则跳过
if [ ! -f field_proto_model.pth ]; then
    echo "[1] 训练 10 类 FieldProtoGNN (drop-ports)..."
    python train_proto_classifier.py \
        --data-dir data/data \
        --output field_proto_model.pth \
        --drop-ports \
        --epochs 100
else
    echo "[1] field_proto_model.pth 已存在，跳过训练"
fi

# Step 2: 端口无关性验证
echo ""
echo "[2] 端口无关性验证..."
python test_port_invariance.py \
    --model field_proto_model.pth \
    --data-dir data/data

# Step 3: 完整消融实验
echo ""
echo "[3] 运行消融实验 (6 组配置)..."
python run_ablation.py \
    --data-dir data/data \
    --epochs 80 \
    --output ablation_results.json

echo ""
echo "=========================================="
echo " 完成!"
echo "  - 模型: field_proto_model.pth"
echo "  - 消融结果: ablation_results.json"
echo "=========================================="
