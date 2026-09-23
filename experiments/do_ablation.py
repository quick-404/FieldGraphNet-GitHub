"""
Run ablation experiments with per-epoch printing.
Usage: python _do_ablation.py > ablation_log.txt 2>&1
"""
import sys, os, time, copy, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
import numpy as np
from nemesys_gnn4id.nemesys.field_graph_builder import PROTO_CLASSES
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE, train_epoch, evaluate
from train_proto_classifier import stratified_split

device = 'cpu'
run_name = sys.argv[1] if len(sys.argv) > 1 else 'all'

CONFIGS = []

if run_name in ('all', 'sage'):
    CONFIGS.append(('SAGE+drop-ports', 'sage', 'graph_cache/graphs_drop1_s100.pt'))
if run_name in ('all', 'mlp'):
    CONFIGS.append(('MLP+drop-ports', 'mlp', 'graph_cache/graphs_drop1_s100.pt'))

def make_model(typ, nc):
    if typ == 'sage':
        return FieldProtoGNN_SAGE(hidden_dim=64, latent_dim=32, num_classes=nc, dropout=0.3)
    elif typ == 'mlp':
        # Simple MLP baseline
        from nemesys_gnn4id.nemesys.field_graph_builder import FIELD_FEAT_DIM, PROTO_FEAT_DIM
        import torch.nn as nn
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

print(f'Device: {device}  Configs: {len(CONFIGS)}')

for name, typ, cache_path in CONFIGS:
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')
    print(f'Loading {cache_path}...')
    data = torch.load(cache_path, weights_only=False)
    G, L = data['graphs'], data['labels']
    split = stratified_split(G, L, seed=42)
    print(f'  Train: {len(split["train"]["labels"])}, Val: {len(split["val"]["labels"])}, Test: {len(split["test"]["labels"])}')

    model = make_model(typ, len(PROTO_CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    best_f1, best_ep, best_state = -1, 0, copy.deepcopy(model.state_dict())

    for ep in range(1, 101):
        tl, ta = train_epoch(model, split['train']['graphs'], split['train']['labels'], opt, device)
        vm = evaluate(model, split['val']['graphs'], split['val']['labels'], device)
        vf1 = vm['macro']['f1']
        sched.step(vm['loss'])
        if vf1 > best_f1:
            best_f1 = vf1; best_ep = ep; best_state = copy.deepcopy(model.state_dict())
        print(f'  E{ep:3d}/100: loss={tl:.4f} acc={ta:.4f} | val_acc={vm["accuracy"]:.4f} f1={vf1:.4f}', flush=True)

    model.load_state_dict(best_state)
    test = evaluate(model, split['test']['graphs'], split['test']['labels'], device)
    print(f'\n  >>> {name}: test_acc={test["accuracy"]:.4f} macro_f1={test["macro"]["f1"]:.4f}')
    for c in range(len(PROTO_CLASSES)):
        s = test['per_class'][c]
        print(f'    {PROTO_CLASSES[c]:8s}: P={s["precision"]:.3f} R={s["recall"]:.3f} F1={s["f1"]:.3f} (tp={s["tp"]})')

print('\nDone!')
