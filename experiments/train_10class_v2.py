"""
Train 10-class FieldProtoGNN with cached graphs, per-epoch printing, and checkpoints.
"""
import os, sys, copy, time, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
import numpy as np
from nemesys_gnn4id.nemesys.field_graph_builder import PROTO_CLASSES, FIELD_FEAT_DIM, PROTO_FEAT_DIM
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE, train_epoch, evaluate

def stratified_split(graphs, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed); np.random.seed(seed)
    split = {'train': {'graphs': [], 'labels': []},
             'val': {'graphs': [], 'labels': []},
             'test': {'graphs': [], 'labels': []}}
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        random.shuffle(idx)
        n = len(idx)
        te = int(n * train_ratio)
        ve = te + int(n * val_ratio)
        for name, part in [('train', idx[:te]), ('val', idx[te:ve]), ('test', idx[ve:])]:
            for i in part:
                split[name]['graphs'].append(graphs[i])
                split[name]['labels'].append(labels[i])
    return split

def save_checkpoint(path, model, metrics, epoch, model_class='FieldProtoGNN'):
    ckpt = {
        'state_dict': model.state_dict(),
        'model_class': model_class,
        'hidden_dim': 64, 'latent_dim': 32,
        'num_classes': len(PROTO_CLASSES),
        'proto_classes': list(PROTO_CLASSES),
        'field_feat_dim': FIELD_FEAT_DIM,
        'proto_feat_dim': PROTO_FEAT_DIM,
        'confidence_threshold': 0.65,
        'drop_ports': True,
        'best_epoch': epoch,
        'final_metrics': metrics,
    }
    torch.save(ckpt, path)
    print(f'[CHECKPOINT] {path} (epoch {epoch})')

device = 'cpu'
print(f'Device: {device}')
print('Loading cached graphs...')
data = torch.load('field_proto_graphs_cache.pt', weights_only=False)
graphs, labels = data['graphs'], data['labels']
print(f'Graphs: {len(graphs)}, Labels: {len(labels)}')

split = stratified_split(graphs, labels, seed=42)
print(f'Train: {len(split["train"]["labels"])}, Val: {len(split["val"]["labels"])}, Test: {len(split["test"]["labels"])}')

model = FieldProtoGNN(hidden_dim=64, latent_dim=32, num_classes=len(PROTO_CLASSES), dropout=0.3).to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

best_f1 = -1.0
best_epoch = 0
best_state = copy.deepcopy(model.state_dict())
start = time.time()

epochs = 200
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
    dt = time.time() - t0
    print(f'Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f} acc={train_acc:.4f} | '
          f'val_acc={val_metrics["accuracy"]:.4f} val_f1={val_f1:.4f} | {dt:.1f}s', flush=True)

total_time = time.time() - start
print(f'\nTraining: {total_time:.1f}s, best val_f1={best_f1:.4f} at epoch={best_epoch}')

model.load_state_dict(best_state)
test = evaluate(model, split['test']['graphs'], split['test']['labels'], device)
print(f'Test acc={test["accuracy"]:.4f} macro_f1={test["macro"]["f1"]:.4f}')
for c in range(len(PROTO_CLASSES)):
    s = test['per_class'][c]
    print(f'  {PROTO_CLASSES[c]:8s}: P={s["precision"]:.3f} R={s["recall"]:.3f} F1={s["f1"]:.3f} '
          f'(tp={s["tp"]} fp={s["fp"]} fn={s["fn"]})')

save_checkpoint('field_proto_model.pth', model, test, best_epoch)
print('Done!')
