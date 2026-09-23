"""
nemesys-gnn4id 命令行入口

用法:
    python cli.py analyze <pcap_file> [选项]
    python cli.py info

示例:
    python cli.py analyze capture.pcap
    python cli.py analyze capture.pcap --model path/to/model.pth --sigma 0.8 --output report.txt
    python cli.py analyze capture.pcap --max-flows 500 --no-deep
    python cli.py analyze capture.pcap --no-gnn                    # 纯NEMESYS
    python cli.py analyze capture.pcap --labels labels.txt --cm cm.png   # 带混淆矩阵
"""

import argparse
import os
import sys

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nemesys_gnn4id.config import (
    DEFAULT_MODEL_CANDIDATES,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROTO_MODEL_PATH,
    NEMESYS_DEFAULTS,
)
from nemesys_gnn4id.pipeline import TrafficPipeline
from nemesys_gnn4id.utils.report import generate_report


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def cmd_analyze(args):
    """执行分析"""
    # 确定模型路径
    model_path = args.model
    if model_path is None and os.path.exists(DEFAULT_MODEL_PATH):
        model_path = DEFAULT_MODEL_PATH
        print(f"[CLI] 使用默认模型: {model_path}")

    proto_model_path = args.proto_model
    if proto_model_path is None and os.path.exists(DEFAULT_PROTO_MODEL_PATH):
        proto_model_path = DEFAULT_PROTO_MODEL_PATH
        print(f"[CLI] 使用默认字段图协议模型: {proto_model_path}")

    # 初始化流水线
    pipeline = TrafficPipeline(
        model_path=model_path,
        sigma=args.sigma,
        device=args.device,
        proto_model_path=proto_model_path,
        proto_threshold=args.proto_threshold,
    )

    # 执行分析
    result = pipeline.analyze(
        pcap_path=args.pcap,
        max_flows=args.max_flows,
        deep_analysis=not args.no_deep,
        run_gnn=not args.no_gnn,
    )

    # 如果提供了 ground truth 标签，计算评估指标
    if args.labels:
        labels = _load_labels(args.labels)
        if labels:
            result['metrics'] = pipeline.evaluate(result, labels)
            print(f"[CLI] 评估指标已计算 (样本数: {result['metrics']['total_samples']})")

            # 生成混淆矩阵热力图（直接从内存构造，不依赖报告文件）
            if args.cm:
                try:
                    from plot_confusion_matrix import plot_confusion_matrix, compute_confusion_matrix
                    gnn = result.get('gnn', {}).get('results', [])
                    y_true, y_pred = [], []
                    for i, r in enumerate(gnn):
                        if i < len(labels):
                            y_true.append(labels[i])
                            y_pred.append(r.get('class_id', 0))
                    if y_true:
                        cm_raw, cm_pct = compute_confusion_matrix(y_true, y_pred)
                        cm_path = args.cm
                        plot_confusion_matrix(cm_pct, cm_raw, cm_path,
                                              title=f'GNN4ID 混淆矩阵 - {os.path.basename(args.pcap)}')
                except Exception as e:
                    print(f"[CLI] 混淆矩阵生成失败: {e}")

    # 生成报告
    if args.output:
        generate_report(result, args.output, fmt=args.format)
    else:
        report = generate_report(result, fmt=args.format)
        print(report)


def _load_labels(labels_path):
    """
    加载 ground truth 标签文件

    支持格式：
    - CSV: 每行一个标签（整数 0-7）
    - TXT: 每行一个标签
    """
    import csv

    if not os.path.exists(labels_path):
        print(f"[CLI] 警告: 标签文件不存在: {labels_path}")
        return None

    labels = []
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            # 尝试 CSV 读取
            reader = csv.reader(f)
            for row in reader:
                if row:
                    try:
                        labels.append(int(row[0]))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[CLI] 警告: 读取标签文件失败: {e}")
        return None

    if not labels:
        print(f"[CLI] 警告: 标签文件为空")
        return None

    print(f"[CLI] 加载了 {len(labels)} 个标签")
    return labels


def cmd_info(args):
    """显示系统信息"""
    print("=" * 50)
    print("  nemesys-gnn4id 系统信息")
    print("=" * 50)

    # 检查依赖
    deps = {
        'torch': 'torch',
        'torch_geometric': 'torch_geometric',
        'nfstream': 'nfstream',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scapy': 'scapy',
    }

    print("\n  依赖检查:")
    for name, module in deps.items():
        try:
            m = __import__(module)
            ver = getattr(m, '__version__', 'installed')
            print(f"    [OK] {name}: {ver}")
            if name == 'torch':
                cuda = getattr(m, 'cuda', None)
                if cuda:
                    print(f"         CUDA可用: {cuda.is_available()}, CUDA版本: {getattr(m.version, 'cuda', None)}")
        except Exception as e:
            print(f"    [--] {name}: 不可用 ({e})")

    # 检查 nemere
    try:
        import nemere
        print(f"    [OK] nemere: installed")
    except Exception as e:
        print(f"    [--] nemere: 不可用 ({e}; NEMESYS将使用模拟/指纹模式)")

    # 检查模型
    print(f"\n  GNN4ID 模型候选路径:")
    for path in DEFAULT_MODEL_CANDIDATES:
        marker = "[OK]" if os.path.exists(path) else "[--]"
        selected = " (当前默认)" if path == DEFAULT_MODEL_PATH else ""
        print(f"    {marker} {path}{selected}")
    print(f"\n  默认 GNN4ID 模型: {DEFAULT_MODEL_PATH}")
    if os.path.exists(DEFAULT_MODEL_PATH):
        print(f"    [OK] 模型文件存在")
    else:
        print(f"    [--] 模型文件不存在 (将使用模拟模式)")

    print(f"\n  默认字段图协议模型: {DEFAULT_PROTO_MODEL_PATH}")
    if os.path.exists(DEFAULT_PROTO_MODEL_PATH):
        print(f"    [OK] 模型文件存在")
    else:
        print(f"    [--] 模型文件不存在 (可用 train_proto_classifier.py 训练)")

    # 检查内置协议 PCAP
    data_dir = os.path.join(REPO_ROOT, 'data', 'data')
    expected_pcaps = [
        'dhcp_100.pcap', 'dns_100.pcap', 'ntp_100.pcap',
        'modbus_100.pcap', 'dnp3_100.pcap', 's7comm_100.pcap',
    ]
    print(f"\n  内置协议PCAP目录: {data_dir}")
    for name in expected_pcaps:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            print(f"    [OK] {name}")
        else:
            print(f"    [--] {name} 缺失")

    print(f"\n{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='nemesys-gnn4id: GNN4ID + NEMESYS 集成流量分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py analyze capture.pcap
  python cli.py analyze capture.pcap --model model.pth --output report.txt
  python cli.py info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # analyze 子命令
    p_analyze = subparsers.add_parser('analyze', help='分析PCAP文件')
    p_analyze.add_argument('pcap', help='PCAP文件路径')
    p_analyze.add_argument('--model', '-m', default=None,
                           help='GNN模型路径 (.pth)。默认自动查找。')
    p_analyze.add_argument('--proto-model', default=None,
                           help='FieldProtoGNN协议图模型路径 (.pth)。默认自动使用仓库内 field_proto_model.pth。')
    p_analyze.add_argument('--proto-threshold', type=float, default=None,
                           help='字段图协议识别置信度阈值。默认使用模型checkpoint中的阈值。')
    p_analyze.add_argument('--sigma', '-s', type=float,
                           default=NEMESYS_DEFAULTS['sigma'],
                           help=f'BCDG sigma参数 (默认: {NEMESYS_DEFAULTS["sigma"]})')
    p_analyze.add_argument('--device', '-d', default=None,
                           choices=['cuda', 'cpu'],
                           help='计算设备 (默认: 自动选择)')
    p_analyze.add_argument('--max-flows', type=int, default=1000,
                           help='最大处理flow数 (默认: 1000)')
    p_analyze.add_argument('--no-deep', action='store_true',
                           help='跳过NEMESYS深度分析')
    p_analyze.add_argument('--no-gnn', action='store_true',
                           help='跳过GNN4ID，只跑NEMESYS分析')
    p_analyze.add_argument('--output', '-o', default=None,
                           help='报告输出路径。不指定则打印到终端。')
    p_analyze.add_argument('--format', '-f', default='auto',
                           choices=['text', 'json', 'md', 'auto'],
                           help='输出格式 (默认: auto，根据扩展名判断)')
    p_analyze.add_argument('--labels', '-l', default=None,
                           help='Ground truth 标签文件 (CSV/TXT，每行一个标签 0-7)。'
                                '提供后将计算精确率/召回率/F1/FPR 评估指标。')
    p_analyze.add_argument('--cm', default=None,
                           help='混淆矩阵热力图输出路径 (如 confusion_matrix.png)。需同时提供 --labels。')
    p_analyze.set_defaults(func=cmd_analyze)

    # info 子命令
    p_info = subparsers.add_parser('info', help='显示系统信息')
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
