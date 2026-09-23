"""Test FieldProtoGNN model forward pass"""
import torch
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE
from nemesys_gnn4id.nemesys.field_graph_builder import FIELD_FEAT_DIM, PROTO_FEAT_DIM


def _make_test_data():
    """Create minimal x_dict and edge_index_dict for a 2-graph batch."""
    x_dict = {
        "proto": torch.randn(2, PROTO_FEAT_DIM),
        "field": torch.randn(6, FIELD_FEAT_DIM),
    }
    edge_index_dict = {
        ("proto", "contain", "field"): torch.tensor([[0, 0, 0], [0, 1, 2]]),
        ("field", "rev_contain", "proto"): torch.tensor([[0, 1, 2], [0, 0, 0]]),
        ("field", "link", "field"): torch.tensor([[0, 1], [1, 2]]),
        ("field", "rev_link", "field"): torch.tensor([[1, 0], [0, 1]]),
    }
    return x_dict, edge_index_dict


class TestFieldProtoGNN:
    def test_gat_forward_shape(self):
        model = FieldProtoGNN(hidden_dim=16, num_classes=10)
        x_dict, edge_index_dict = _make_test_data()
        out = model(x_dict, edge_index_dict)
        assert out.shape == (2, 10)

    def test_gat_forward_not_nan(self):
        model = FieldProtoGNN(hidden_dim=16, num_classes=10)
        x_dict, edge_index_dict = _make_test_data()
        out = model(x_dict, edge_index_dict)
        assert not torch.isnan(out).any()


class TestFieldProtoGNN_SAGE:
    def test_sage_forward_shape(self):
        model = FieldProtoGNN_SAGE(hidden_dim=16, num_classes=10)
        x_dict, edge_index_dict = _make_test_data()
        out = model(x_dict, edge_index_dict)
        assert out.shape == (2, 10)

    def test_sage_forward_not_nan(self):
        model = FieldProtoGNN_SAGE(hidden_dim=16, num_classes=10)
        x_dict, edge_index_dict = _make_test_data()
        out = model(x_dict, edge_index_dict)
        assert not torch.isnan(out).any()
