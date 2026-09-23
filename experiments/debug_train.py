"""
Debug 10-class training with lower LR and per-class monitoring.
"""
import sys, os, copy, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
import numpy as np
from nemesys_gnn4id.nemesys.field_graph_builder import PROTO_CLASSES
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, train_epoch, evaluate
from train_proto_classifier import stratified_split

device = 'cpu'
print(f'Device: {device}')

# Load cached graphs
print('Loading cached graphs...')
data = torch.load('field_proto_graphs_cache.pt', weights_only=False)
graphs, labels = data['graphs'], data['labels']
print(f'Graphs: {len(graphs)}, Labels: {len(labels)}')

split = stratified_split(graphs, labels, seed=42)
print(f'Train: {len(split["train"]["labels"])}, Val: {len(split["val"]["labels"])}, Test: {len(split["test"]["labels"])}')

model = FieldProtoGNN(hidden_dim=64, latent_dim=32, num_classes=len(PROTO_CLASSES), dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)

best_f1 = -1
best_epoch = 0
best_state = copy.deepcopy(model.state_dict())

for epoch in range(1, 201):
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
    if epoch % 10 == 0:
        t = time.time() - t0
        val_per_class = {PROTO_CLASSES[int(c)]: f'{m["f1"]:.3f}'
                         for c, m in val_metrics['per_class'].items()}
        print(f'Epoch {epoch:3d}: train_loss={train_loss:.4f} acc={train_acc:.4f} | '
              f'val_acc={val_metrics["accuracy"]:.4f} val_f1={val_f1:.4f} | {t:.1f}s')

print(f'\nBest val_f1={best_f1:.4f} at epoch={best_epoch}')
model.load_state_dict(best_state)
test = evaluate(model, split['test']['graphs'], split['test']['labels'], device)
print(f'\n=== Test Results ===')
print(f'Accuracy: {test["accuracy"]:.4f}')
print(f'Macro F1: {test["macro"]["f1"]:.4f}')
print()
for c in range(len(PROTO_CLASSES)):
    s = test['per_class'][c]
    print(f'  {PROTO_CLASSES[c]:8s}: P={s["precision"]:.3f} R={s["recall"]:.3f} F1={s["f1"]:.3f} '
          f'(tp={s["tp"]} fp={s["fp"]} fn={s["fn"]})')

# Save model checkpoint
from train_proto_classifier import save_checkpoint
save_checkpoint('field_proto_model.pth', model, None, None, test, best_epoch, 'FieldProtoGNN')
print('\n[SAVE] field_proto_model.pth')
