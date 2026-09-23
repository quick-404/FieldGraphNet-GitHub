"""
Full cross-dataset evaluation on CICIDS2017 using editcap-split chunks.
"""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings('ignore')

# Add GNN4ID paths for Utility module
sys.path.insert(0, 'D:/网络流量分析项目/GNN4ID')
sys.path.insert(0, 'D:/网络流量分析项目/GNN4ID/Utility')
sys.path.insert(0, 'src')

CHUNK_DIR = 'data/data/17pcap_chunks'
EVAL_DIR = 'eval_results/fusion_comparison'
os.makedirs(EVAL_DIR, exist_ok=True)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer
from nemesys_gnn4id.pipeline import TrafficPipeline


def run_evaluation(file_prefix, label, max_chunks=10):
    """Run full evaluation on chunks from a 2017 pcap."""
    parser = TrafficPipeline(device='cpu', model_path='models/model.pth',
                              proto_model_path='field_proto_model_bcdg.pth')
    analyzer_gnn = GNN4IDAnalyzer(model_path='models/model.pth')

    # Collect features for sklearn
    all_features = []
    chunk_results = []

    chunks = sorted([f for f in os.listdir(CHUNK_DIR)
                     if f.startswith(file_prefix) and f.endswith('.pcap')])
    chunks = chunks[:max_chunks]

    print(f'\n=== {label}: {len(chunks)} chunks ===')

    for ci, chunk in enumerate(chunks):
        chunk_path = os.path.join(CHUNK_DIR, chunk)

        # Extract features for sklearn
        df = analyzer_gnn.extract_features(chunk_path, max_flows=50)
        if df.empty or 'flow_features' not in df.columns:
            continue
        for _, row in df.iterrows():
            feat = list(row['flow_features'])
            for z in range(77, min(82, len(feat))):
                feat[z] = 0.0
            all_features.append(feat)

        # Run full pipeline (BCDG + GNN + clustering + fusion)
        print(f'  Chunk {ci+1}/{len(chunks)}: {chunk}', end='', flush=True)
        result = parser.analyze(chunk_path, max_flows=50)
        gnn_results = result.get('gnn', {}).get('results', [])

        # Collect GNN predictions
        for r in gnn_results:
            chunk_results.append({
                'chunk': chunk,
                'class_id': r.get('class_id', 0),
                'class_name': r.get('class_name', ''),
                'confidence': r.get('confidence', 0),
                'ood': r.get('ood', False),
                'fusion_status': r.get('fusion_decision', {}).get('status', ''),
            })

        clustering = result.get('clustering', {}).get('summary', {})
        print(f' -> {len(gnn_results)} flows, {clustering.get("total_clusters", 0)} clusters')

    return all_features, chunk_results


def main():
    # Run on Wednesday and Friday samples
    wed_features, wed_results = run_evaluation('Wed_', 'CICIDS2017-Wednesday', max_chunks=5)
    print(f'\nWednesday: {len(wed_features)} total flows extracted')

    fri_features, fri_results = run_evaluation('Fri_', 'CICIDS2017-Friday', max_chunks=5)
    print(f'\nFriday: {len(fri_features)} total flows extracted')

    # Summary
    print(f'\n{"="*60}')
    print('  CROSS-DATASET EVALUATION COMPLETE')
    print(f'{"="*60}')
    print(f'  Wednesday flows: {len(wed_features)}')
    print(f'  Friday flows: {len(fri_features)}')
    print(f'  Wednesday results: {len(wed_results)}')
    print(f'  Friday results: {len(fri_results)}')

    # Save results
    out_path = os.path.join(EVAL_DIR, 'cross_dataset_full_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'n_wed_features': len(wed_features),
            'n_fri_features': len(fri_features),
            'wed_results': wed_results[:100],
            'fri_results': fri_results[:100],
        }, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
