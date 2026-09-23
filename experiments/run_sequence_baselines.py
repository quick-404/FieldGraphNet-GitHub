"""Sequence baselines (LSTM/GRU/Transformer) for per-protocol OOD comparison.

Temporal-window sequence models: each flow is classified from a window of the
preceding W-1 flows' 82-dim drop-ports features, i.e. the input is a length-W
sequence of flow feature vectors. Trained on in-domain chunks (BenignTraffic +
DDoS-HTTP_Flood) as a binary attack/benign classifier, tested per OOD protocol,
evaluated per-flow (identical evaluation unit to the sklearn baselines).

Outputs recall/f1 per protocol into per_protocol_baselines.json (merged).
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn

CHUNK_DIR = 'data/data/23pcap_chunks'
MODEL_PATH = 'models/model.pth'
EVAL_DIR = 'eval_results/fusion_comparison'
WINDOW = 5
EPOCHS = 60
BATCH = 32
SEED = 42


def compute_metrics(y_true, y_pred):
    n = len(y_true)
    tp = sum(1 for i in range(n) if y_true[i] > 0 and y_pred[i] > 0)
    fp = sum(1 for i in range(n) if y_true[i] == 0 and y_pred[i] > 0)
    fn = sum(1 for i in range(n) if y_true[i] > 0 and y_pred[i] == 0)
    tn = sum(1 for i in range(n) if y_true[i] == 0 and y_pred[i] == 0)
    return {'recall': round(tp / max(tp + fn, 1), 4),
            'fpr': round(fp / max(fp + tn, 1), 4),
            'f1': round(2 * tp / max(2 * tp + fp + fn, 1), 4)}


class SeqBaseline(nn.Module):
    def __init__(self, kind, in_dim=82, hidden=32, window=WINDOW, nclass=2):
        super().__init__()
        self.kind = kind
        self.window = window
        if kind == 'LSTM':
            self.net = nn.LSTM(in_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, nclass)
        elif kind == 'GRU':
            self.net = nn.GRU(in_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, nclass)
        elif kind == 'Transformer':
            self.embed = nn.Linear(in_dim, hidden)
            self.pos = nn.Parameter(torch.zeros(1, window, hidden))
            enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=2, dim_feedforward=64,
                                             batch_first=True, dropout=0.1)
            self.net = nn.TransformerEncoder(enc, num_layers=1)
            self.head = nn.Linear(hidden, nclass)
        else:
            raise ValueError(kind)

    def forward(self, x):  # x: (B, W, 82)
        if self.kind == 'Transformer':
            h = self.embed(x) + self.pos
            h = self.net(h)
        else:
            h, _ = self.net(x)
        return self.head(h[:, -1, :])  # last time step


def build_windows(flows_by_chunk, window):
    """Flatten per-chunk flow sequences into (window, label, feat_seq) per flow."""
    X, y = [], []
    for chunk_feats, chunk_label in flows_by_chunk:
        n = len(chunk_feats)
        for i in range(window - 1, n):
            X.append(chunk_feats[i - window + 1: i + 1])
            y.append(chunk_label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(os.path.join(CHUNK_DIR, '_chunk_index.json')) as f:
        chunks = json.load(f)

    from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer
    analyzer = GNN4IDAnalyzer(model_path=MODEL_PATH)

    # In-domain training / OOD testing split (same as run_baselines.py)
    in_domain = ['BenignTraffic.pcap', 'DDoS-HTTP_Flood-.pcap']
    ood_pcaps = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']

    # Collect per-chunk ordered flow features, keeping temporal order within a chunk
    labeled = []
    for chunk_name, (attack_label, source_pcap) in chunks.items():
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            continue
        df = analyzer.extract_features(chunk_path, max_flows=50)
        if df.empty or 'flow_features' not in df.columns:
            continue
        feats = []
        for _, row in df.iterrows():
            feat = list(row['flow_features'])
            for z in range(77, min(82, len(feat))):
                feat[z] = 0.0  # drop ports
            feats.append(feat)
        if len(feats) >= WINDOW:
            labeled.append((source_pcap, feats, attack_label))

    print(f'Chunks with >= {WINDOW} flows: {len(labeled)}')

    # Train/test split
    train_data = [(f, l) for src, f, l in labeled if src in in_domain]
    X_tr, y_tr = build_windows(train_data, WINDOW)
    y_bin = (y_tr > 0).astype(np.int64)
    print(f'Train windows: {len(X_tr)} (attack={int(y_bin.sum())}, benign={int(len(y_bin)-y_bin.sum())})')

    results = {}
    for kind in ['LSTM', 'GRU', 'Transformer']:
        model = SeqBaseline(kind)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        model.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(X_tr))
            tot = 0.0
            for b in range(0, len(X_tr), BATCH):
                idx = perm[b:b + BATCH]
                xb = torch.from_numpy(X_tr[idx])
                yb = torch.from_numpy(y_bin[idx])
                opt.zero_grad()
                out = model(xb)
                loss = lossf(out, yb)
                loss.backward()
                opt.step()
                tot += loss.item()
            if (ep + 1) % 20 == 0:
                print(f'  {kind} epoch {ep+1}: loss {tot/ (len(X_tr)//BATCH+1):.4f}')

        model.eval()
        print(f'\n--- {kind} ---')
        per_proto = {}
        with torch.no_grad():
            # Aggregate predictions across all chunks of each OOD protocol
            for proto in [p.replace('.pcap', '') for p in ood_pcaps]:
                src = proto + '.pcap'
                agg_y, agg_pred = [], []
                for s, feats, label in labeled:
                    if s != src:
                        continue
                    X_te, y_te = build_windows([(feats, label)], WINDOW)
                    yb_te = (y_te > 0).astype(np.int64)
                    preds = []
                    for b in range(0, len(X_te), BATCH):
                        xb = torch.from_numpy(X_te[b:b + BATCH])
                        out = model(xb)
                        preds.extend(torch.argmax(out, dim=1).tolist())
                    agg_y.extend(yb_te.tolist())
                    agg_pred.extend(preds)
                if not agg_y:
                    continue
                m = compute_metrics(agg_y, agg_pred)
                per_proto[proto] = m
                print(f'  {proto:22s} Recall={m["recall"]:.4f} FPR={m["fpr"]:.4f} F1={m["f1"]:.4f} '
                      f'(windows={len(agg_y)})')
        results[kind] = per_proto

    # Merge into per_protocol_baselines.json
    save_path = os.path.join(EVAL_DIR, 'per_protocol_baselines.json')
    if os.path.exists(save_path):
        with open(save_path, encoding='utf-8') as f:
            saved = json.load(f)
    else:
        saved = {}
    for kind, per_proto in results.items():
        for proto, m in per_proto.items():
            saved.setdefault(proto, {})[kind] = m
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)
    print(f'\nMerged into {save_path}')


if __name__ == '__main__':
    main()
