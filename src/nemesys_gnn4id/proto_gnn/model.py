"""
字段图协议分类 GNN：从 NEMESYS BCDG 字段图预测协议类别

模型结构类似 HeteroGNN_Edge (GATConv)，适配 proto/field 节点类型。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, HeteroConv, Linear
from torch_geometric.data import Batch

from nemesys_gnn4id.nemesys.field_graph_builder import FIELD_FEAT_DIM, PROTO_FEAT_DIM, PROTO_CLASSES


class FieldProtoGNN(nn.Module):
    """
    从 proto+field 异构图做协议多分类。

    编码器:
        proto  node: PROTO_FEAT_DIM → hidden → latent    (GATConv/SAGEConv)
        field  node: FIELD_FEAT_DIM → hidden → latent

    解码器:
        proto node embedding → Linear → 6-class softmax
    """

    def __init__(self, hidden_dim=64, latent_dim=32, num_classes=6,
                 use_attention=True, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.dropout = dropout

        ConvLayer = GATConv if use_attention else SAGEConv

        # 第一层卷积
        self.conv1 = HeteroConv({
            ('proto', 'contain', 'field'): ConvLayer(
                (PROTO_FEAT_DIM, FIELD_FEAT_DIM), hidden_dim, add_self_loops=False),
            ('field', 'rev_contain', 'proto'): ConvLayer(
                (FIELD_FEAT_DIM, PROTO_FEAT_DIM), hidden_dim, add_self_loops=False),
            ('field', 'link', 'field'): ConvLayer(
                FIELD_FEAT_DIM, hidden_dim, add_self_loops=False),
            ('field', 'rev_link', 'field'): ConvLayer(
                FIELD_FEAT_DIM, hidden_dim, add_self_loops=False),
        })

        # 第二层卷积
        self.conv2 = HeteroConv({
            ('proto', 'contain', 'field'): ConvLayer(
                (hidden_dim, hidden_dim), latent_dim, add_self_loops=False),
            ('field', 'rev_contain', 'proto'): ConvLayer(
                (hidden_dim, hidden_dim), latent_dim, add_self_loops=False),
            ('field', 'link', 'field'): ConvLayer(
                hidden_dim, latent_dim, add_self_loops=False),
            ('field', 'rev_link', 'field'): ConvLayer(
                hidden_dim, latent_dim, add_self_loops=False),
        })

        # 分类头
        self.classifier = nn.Sequential(
            Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(latent_dim, num_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        # 第一层
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                  for key, x in x_dict.items()}

        # 第二层
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # proto 节点 embedding → 分类
        proto_emb = x_dict['proto']
        logits = self.classifier(proto_emb)
        return logits


class FieldProtoGNN_SAGE(nn.Module):
    """
    简化版：纯 SAGEConv，更稳定。
    """

    def __init__(self, hidden_dim=64, latent_dim=32, num_classes=6, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.conv1 = HeteroConv({
            ('proto', 'contain', 'field'): SAGEConv(
                (PROTO_FEAT_DIM, FIELD_FEAT_DIM), hidden_dim),
            ('field', 'rev_contain', 'proto'): SAGEConv(
                (FIELD_FEAT_DIM, PROTO_FEAT_DIM), hidden_dim),
            ('field', 'link', 'field'): SAGEConv(
                FIELD_FEAT_DIM, hidden_dim),
            ('field', 'rev_link', 'field'): SAGEConv(
                FIELD_FEAT_DIM, hidden_dim),
        })

        self.conv2 = HeteroConv({
            ('proto', 'contain', 'field'): SAGEConv(
                (hidden_dim, hidden_dim), latent_dim),
            ('field', 'rev_contain', 'proto'): SAGEConv(
                (hidden_dim, hidden_dim), latent_dim),
            ('field', 'link', 'field'): SAGEConv(
                hidden_dim, latent_dim),
            ('field', 'rev_link', 'field'): SAGEConv(
                hidden_dim, latent_dim),
        })

        self.classifier = nn.Sequential(
            Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(latent_dim, num_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                  for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        proto_emb = x_dict['proto']
        logits = self.classifier(proto_emb)
        return logits


class FieldProtoGNN_V2(nn.Module):
    """V2: residual + LayerNorm + message pooling + multi-head GAT.

    Fixes v1 weaknesses observed in eval: no residual/norm (deep stacking
    degrades), single-head GAT, and no message-level pooling (field info
    compressed by only 2 conv layers).
    """

    def __init__(self, hidden_dim=64, latent_dim=32, num_classes=6,
                 dropout=0.3, heads=4, field_feat_dim=None, proto_feat_dim=None):
        super().__init__()
        from nemesys_gnn4id.nemesys.field_graph_builder import FIELD_FEAT_DIM, PROTO_FEAT_DIM
        self.field_dim = field_feat_dim or FIELD_FEAT_DIM
        self.proto_dim = proto_feat_dim or PROTO_FEAT_DIM
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.heads = heads
        if heads < 1 or hidden_dim % heads != 0 or latent_dim % heads != 0:
            raise ValueError(
                f'FieldProtoGNN_V2: heads ({heads}) must be >= 1 and divide hidden_dim ({hidden_dim}) '
                f'and latent_dim ({latent_dim}) evenly (concat output must match residual dims)')

        # input projections for residual compatibility
        self.proto_proj = nn.Linear(self.proto_dim, hidden_dim)
        self.field_proj = nn.Linear(self.field_dim, hidden_dim)

        # layer 1 (multi-head GAT, per edge type)
        self.conv1 = HeteroConv({
            ('proto', 'contain', 'field'): GATConv(
                (self.proto_dim, self.field_dim), hidden_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
            ('field', 'rev_contain', 'proto'): GATConv(
                (self.field_dim, self.proto_dim), hidden_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
            ('field', 'link', 'field'): GATConv(
                self.field_dim, hidden_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
            ('field', 'rev_link', 'field'): GATConv(
                self.field_dim, hidden_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
        })
        self.norm1_p = nn.LayerNorm(hidden_dim)
        self.norm1_f = nn.LayerNorm(hidden_dim)

        # layer 2
        self.conv2 = HeteroConv({
            ('proto', 'contain', 'field'): GATConv(
                (hidden_dim, hidden_dim), latent_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
            ('field', 'rev_contain', 'proto'): GATConv(
                (hidden_dim, hidden_dim), latent_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
            ('field', 'link', 'field'): GATConv(
                hidden_dim, latent_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
            ('field', 'rev_link', 'field'): GATConv(
                hidden_dim, latent_dim // heads, heads=heads,
                add_self_loops=False, concat=True),
        })
        self.norm2_p = nn.LayerNorm(latent_dim)
        self.norm2_f = nn.LayerNorm(latent_dim)

        # message-level pooling: field mean+max -> concat to proto latent
        self.pool_proj = nn.Linear(latent_dim * 2, latent_dim)

        # classifier head: proto latent + pooled field info
        self.classifier = nn.Sequential(
            Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(latent_dim, num_classes),
        )

    def forward(self, x_dict, edge_index_dict, batch=None):
        x_p, x_f = x_dict['proto'], x_dict['field']

        # layer 1 with residual + norm (single conv pass per node type)
        h1 = self.conv1(x_dict, edge_index_dict)
        h1_p = self.norm1_p(F.relu(h1['proto']) + self.proto_proj(x_p))
        h1_f = self.norm1_f(F.relu(h1['field']) + self.field_proj(x_f))
        h1_p = F.dropout(h1_p, p=self.dropout, training=self.training)
        h1_f = F.dropout(h1_f, p=self.dropout, training=self.training)

        # layer 2
        d2 = {'proto': h1_p, 'field': h1_f}
        h2 = self.conv2(d2, edge_index_dict)
        h2_p = self.norm2_p(F.relu(h2['proto']))
        h2_f = self.norm2_f(F.relu(h2['field']))

        # message-level pooling: per-graph mean+max over field nodes
        if batch is None:
            field_mean = h2_f.mean(dim=0, keepdim=True)
            field_max = h2_f.max(dim=0, keepdim=True).values
        else:
            from torch_geometric.utils import scatter
            field_mean = scatter(h2_f, batch, dim=0, reduce='mean')
            field_max = scatter(h2_f, batch, dim=0, reduce='max')
        pooled = self.pool_proj(torch.cat([field_mean, field_max], dim=-1))

        # proto embedding + pooled field context
        proto_emb = torch.cat([h2_p, pooled], dim=-1)
        logits = self.classifier(proto_emb)
        return logits


def train_epoch(model, graphs, labels, optimizer, device, batch_size=32, class_weight=None, seed=None):
    """Train one epoch over graphs in mini-batches (supports class weights).

    Mini-batches are drawn in shuffled order: graphs arrive ordered per
    protocol, so unshuffled sequential batches would each contain a single
    class. seed=None -> fresh (non-deterministic) shuffle every epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    order = list(range(len(graphs)))
    rng = np.random.RandomState(seed)
    rng.shuffle(order)
    for start in range(0, len(graphs), batch_size):
        idx = order[start:start + batch_size]
        batch_graphs = [graphs[j] for j in idx]
        batch_labels = [labels[j] for j in idx]
        batch = Batch.from_data_list(batch_graphs).to(device)
        label_t = torch.tensor(batch_labels, dtype=torch.int64).to(device)
        optimizer.zero_grad()
        # V2 需要 field 节点 batch 索引做 per-graph 池化；v1 模型不接受 batch kwarg
        if isinstance(model, FieldProtoGNN_V2):
            fb = batch['field'].batch
            logits = model(batch.x_dict, batch.edge_index_dict, batch=fb)
        else:
            logits = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(logits, label_t, weight=class_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_graphs)
        pred = logits.argmax(dim=1)
        correct += (pred == label_t).sum().item()
        n += len(batch_graphs)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, graphs, labels, device, batch_size=32):
    """评估模型。"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)
        label_t = torch.tensor(batch_labels, dtype=torch.int64).to(device)
        if isinstance(model, FieldProtoGNN_V2):
            fb = batch['field'].batch
            logits = model(batch.x_dict, batch.edge_index_dict, batch=fb)
        else:
            logits = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(logits, label_t)
        total_loss += loss.item() * len(batch_graphs)
        preds = logits.argmax(dim=1).tolist()
        all_preds.extend(preds)
        all_labels.extend(batch_labels)

    # 计算指标
    from collections import Counter
    n = len(all_labels)
    correct = sum(1 for i in range(n) if all_preds[i] == all_labels[i])
    accuracy = correct / max(n, 1)

    # 每类指标
    num_classes = max(max(all_labels), max(all_preds)) + 1
    per_class = {}
    for c in range(num_classes):
        tp = sum(1 for i in range(n) if all_preds[i] == c and all_labels[i] == c)
        fp = sum(1 for i in range(n) if all_preds[i] == c and all_labels[i] != c)
        fn = sum(1 for i in range(n) if all_preds[i] != c and all_labels[i] == c)
        tn = sum(1 for i in range(n) if all_preds[i] != c and all_labels[i] != c)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        fpr = fp / max(fp + tn, 1)
        per_class[c] = {
            'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'support': tp + fn,
        }

    # 平均指标
    macro_p = sum(per_class[c]['precision'] for c in range(num_classes)) / num_classes
    macro_r = sum(per_class[c]['recall'] for c in range(num_classes)) / num_classes
    macro_f1 = sum(per_class[c]['f1'] for c in range(num_classes)) / num_classes
    macro_fpr = sum(per_class[c]['fpr'] for c in range(num_classes)) / num_classes

    return {
        'accuracy': accuracy,
        'loss': total_loss / max(n, 1),
        'per_class': per_class,
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1, 'fpr': macro_fpr},
        'preds': all_preds,
        'labels': all_labels,
    }
