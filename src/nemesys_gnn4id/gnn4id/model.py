"""Compatibility model definitions for original GNN4ID checkpoints.

The upstream GNN4ID training code saves checkpoints that reference
``Utility.Model.HeteroGNN``. Keeping these class names available lets old
``torch.save(model, path)`` artifacts load inside this integrated project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


class HeteroGNN(torch.nn.Module):
    """Original GNN4ID SAGEConv heterogenous graph classifier."""

    def __init__(self, hetero_graph, args, aggr="mean"):
        super().__init__()
        self.aggr = aggr
        self.hidden_size = args['hidden_size']
        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        self.convs1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 64)
            for edge_type in hetero_graph.metadata()[1]
        })
        self.convs2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 64)
            for edge_type in hetero_graph.metadata()[1]
        })

        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.bns2[node_type] = nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

        self.graph_prediction = nn.Linear(128, 64)
        self.graph_prediction_1 = nn.Linear(64, 16)
        self.graph_prediction_2 = nn.Linear(16, 8)

    def forward(self, node_feature, edge_index, batch):
        x = self.convs1(node_feature, edge_index)
        x = {key: self.bns1[key](value) for key, value in x.items()}
        x = {key: self.relus1[key](value) for key, value in x.items()}

        x = self.convs2(x, edge_index)
        x = {key: self.bns2[key](value) for key, value in x.items()}
        x = {key: self.relus2[key](value) for key, value in x.items()}

        graph_emb = {
            key: pyg_nn.global_mean_pool(x[key], batch.batch_dict[key])
            for key in batch.node_types
        }
        graph_emb = torch.cat([graph_emb[t] for t in graph_emb.keys()], dim=1)
        graph_pred = self.graph_prediction(graph_emb)
        graph_pred = self.graph_prediction_1(graph_pred)
        graph_pred = self.graph_prediction_2(graph_pred)
        return F.log_softmax(graph_pred, dim=1)

    def loss(self, preds, label):
        return F.nll_loss(preds, label)


class HeteroGNN_Edge(torch.nn.Module):
    """Original GNN4ID GATConv classifier with edge attributes."""

    def __init__(self, hetero_graph, args, aggr="mean"):
        super().__init__()
        self.aggr = aggr
        self.hidden_size = args['hidden_size']
        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        self.convs1 = HeteroConv({
            edge_type: GATConv((-1, -1), 64, edge_dim=-1, add_self_loops=False)
            for edge_type in hetero_graph.metadata()[1]
        })
        self.convs2 = HeteroConv({
            edge_type: GATConv((-1, -1), 64, edge_dim=-1, add_self_loops=False)
            for edge_type in hetero_graph.metadata()[1]
        })

        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.bns2[node_type] = nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

        self.graph_prediction = nn.Linear(128, 64)
        self.graph_prediction_1 = nn.Linear(64, 16)
        self.graph_prediction_2 = nn.Linear(16, 8)

    def forward(self, node_feature, edge_index, edge_attr, batch):
        x = self.convs1(node_feature, edge_index, edge_attr)
        x = {key: self.bns1[key](value) for key, value in x.items()}
        x = {key: self.relus1[key](value) for key, value in x.items()}

        x = self.convs2(x, edge_index, edge_attr)
        x = {key: self.bns2[key](value) for key, value in x.items()}
        x = {key: self.relus2[key](value) for key, value in x.items()}

        graph_emb = {
            key: pyg_nn.global_mean_pool(x[key], batch.batch_dict[key])
            for key in batch.node_types
        }
        graph_emb = torch.cat([graph_emb[t] for t in graph_emb.keys()], dim=1)
        graph_pred = self.graph_prediction(graph_emb)
        graph_pred = self.graph_prediction_1(graph_pred)
        graph_pred = self.graph_prediction_2(graph_pred)
        return F.log_softmax(graph_pred, dim=1)

    def loss(self, preds, label):
        return F.nll_loss(preds, label)

