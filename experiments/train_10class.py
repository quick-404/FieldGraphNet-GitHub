"""
Quick 10-class training with graph caching for speed.
"""
import os, sys, json, time, copy, random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from train_proto_classifier import *
from nemesys_gnn4id.proto_gnn.model import *

CACHE_FILE = 'field_proto_graphs_cache.pt'

def get_or_build_graphs(analyzer, data_dir, samples_per_class=100, drop_ports=True):
    if os.path.exists(CACHE_FILE):
        print(f'[CACHE] 加载缓存: {CACHE_FILE}')
        data = torch.load(CACHE_FILE, weights_only=False)
        return data['graphs'], data['labels'], data['stats']
    print(f'[BUILD] 构建图...')
    graphs, labels, stats = collect_all_graphs(
        analyzer, data_dir, samples_per_class=samples_per_class, drop_ports=drop_ports)
    torch.save({'graphs': graphs, 'labels': labels, 'stats': stats}, CACHE_FILE)
    print(f'[CACHE] 已保存: {CACHE_FILE}')
    return graphs, labels, stats

def main():
    device = 'cpu'
    analyzer = NEMESYSAnalyzer(sigma=0.6)
    analyzer.nemere_available = False

    graphs, labels, stats = get_or_build_graphs(analyzer, 'data/data', 100, drop_ports=True)
    print(f'图: {len(graphs)}, 标签: {len(labels)}')

    split = stratified_split(graphs, labels, seed=42)
    model = FieldProtoGNN(hidden_dim=64, latent_dim=32,
                          num_classes=len(PROTO_CLASSES), dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_f1 = -1
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    start = time.time()

    epochs = 80
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, split['train']['graphs'], split['train']['labels'], optimizer, device)
        val_metrics = evaluate(model, split['val']['graphs'], split['val']['labels'], device)
        val_f1 = val_metrics['macro']['f1']
        scheduler.step(val_metrics['loss'])

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        t = time.time() - t0
        print(f'Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f} acc={train_acc:.4f} | '
              f'val_acc={val_metrics["accuracy"]:.4f} val_f1={val_f1:.4f} | {t:.1f}s')

    elapsed = time.time() - start
    print(f'\n完成! {elapsed:.1f}s, best epoch={best_epoch}, val_f1={best_f1:.4f}')

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, split['test']['graphs'], split['test']['labels'], device)
    _print_metrics(test_metrics, '最终测试')

    # Save model
    save_checkpoint('field_proto_model.pth', model, None, None, test_metrics, best_epoch, 'FieldProtoGNN')
    print(f'[SAVE] field_proto_model.pth')

if __name__ == '__main__':
    main()
