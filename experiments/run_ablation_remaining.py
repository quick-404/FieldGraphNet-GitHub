"""
Run remaining ablation experiments (with-ports + data efficiency).
Prints every epoch for progress tracking.
"""
import os, sys, copy, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
import numpy as np
from nemesys_gnn4id.nemesys.field_graph_builder import PROTO_CLASSES
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE, train_epoch, evaluate
from train_proto_classifier import stratified_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIGS = [
    # (name, model_type, cache_path)
    ('GAT+with-ports',   'gat',  'graph_cache/graphs_drop0_s100.pt'),
    ('MLP+with-ports',   'mlp',  'graph_cache/graphs_drop0_s100.pt'),
    ('SAGE+with-ports',  'sage', 'graph_cache/graphs_drop0_s100.pt'),
    ('GAT+drop-ports+50','gat',  'graph_cache/graphs_drop1_s50.pt'),
    ('GAT+drop-ports+25','gat',  'graph_cache/graphs_drop1_s25.pt'),
]

def make_model(typ, nc):
    if typ == 'gat':
        return FieldProtoGNN(hidden_dim=64, latent_dim=32, num_classes=nc, dropout=0.3)
    elif typ == 'sage':
        return FieldProtoGNN_SAGE(hidden_dim=64, latent_dim=32, num_classes=nc, dropout=0.3)
    elif typ == 'mlp':
        import torch.nn as nn
        from nemesys_gnn4id.nemesys.field_graph_builder import FIELD_FEAT_DIM, PROTO_FEAT_DIM
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(PROTO_FEAT_DIM + FIELD_FEAT_DIM * 30, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(32, nc),
                )
            def forward(self, x_dict, edge_index_dict=None):
                p = x_dict['proto']
                f = x_dict['field']
                nf = f.size(0)
                if nf < 30:
                    f = torch.cat([f, torch.zeros(30-nf, FIELD_FEAT_DIM, device=f.device)], dim=0)
                else:
                    f = f[:30]
                return self.net(torch.cat([p, f.view(1, -1)], dim=1))
        return SimpleMLP()

print(f'Device: {device}  Configs: {len(CONFIGS)}', flush=True)

for name, typ, cache_path in CONFIGS:
    print(f'\n{"="*60}', flush=True)
    print(f'  {name}', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Loading {cache_path}...', flush=True)

    try:
        data = torch.load(cache_path, weights_only=False)
    except FileNotFoundError:
        print(f'  [ERROR] Cache not found: {cache_path}', flush=True)
        # Build the cache by running run_ablation_fast.py's build
        print(f'  [INFO] Cache will be built on first run. Please run run_ablation_fast.py first.', flush=True)
        continue

    G, L = data['graphs'], data['labels']
    split = stratified_split(G, L, seed=42)
    print(f'  Train: {len(split["train"]["labels"])}, Val: {len(split["val"]["labels"])}, Test: {len(split["test"]["labels"])}', flush=True)

    model = make_model(typ, len(PROTO_CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    best_f1, best_ep, best_state = -1, 0, copy.deepcopy(model.state_dict())

    t_start = time.time()
    for ep in range(1, 101):
        t0 = time.time()
        tl, ta = train_epoch(model, split['train']['graphs'], split['train']['labels'], opt, device)
        vm = evaluate(model, split['val']['graphs'], split['val']['labels'], device)
        vf1 = vm['macro']['f1']
        sched.step(vm['loss'])
        if vf1 > best_f1:
            best_f1 = vf1; best_ep = ep; best_state = copy.deepcopy(model.state_dict())
        print(f'  E{ep:3d}/100: loss={tl:.4f} acc={ta:.4f} | val_acc={vm["accuracy"]:.4f} f1={vf1:.4f} | {time.time()-t0:.1f}s', flush=True)

    model.load_state_dict(best_state)
    test = evaluate(model, split['test']['graphs'], split['test']['labels'], device)
    elapsed = time.time() - t_start
    print(f'\n  >>> {name}: test_acc={test["accuracy"]:.4f} macro_f1={test["macro"]["f1"]:.4f} ({elapsed:.1f}s)', flush=True)
    for c in range(len(PROTO_CLASSES)):
        s = test['per_class'][c]
        print(f'    {PROTO_CLASSES[c]:8s}: P={s["precision"]:.3f} R={s["recall"]:.3f} F1={s["f1"]:.3f} (tp={s["tp"]})', flush=True)

print('\nDone!', flush=True)
