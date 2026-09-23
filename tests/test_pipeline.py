"""Test TrafficPipeline initialization (with mocks)"""
from unittest.mock import patch, MagicMock
from nemesys_gnn4id.pipeline import TrafficPipeline


class TestTrafficPipeline:
    @patch("nemesys_gnn4id.pipeline.GNN4IDAnalyzer")
    @patch("nemesys_gnn4id.pipeline.NEMESYSAnalyzer")
    def test_init_defaults(self, MockNEMESYS, MockGNN):
        MockGNN.return_value = MagicMock()
        MockNEMESYS.return_value = MagicMock()
        pipeline = TrafficPipeline()
        assert pipeline is not None

    @patch("nemesys_gnn4id.pipeline.GNN4IDAnalyzer")
    @patch("nemesys_gnn4id.pipeline.NEMESYSAnalyzer")
    def test_init_with_custom_model_path(self, MockNEMESYS, MockGNN):
        MockGNN.return_value = MagicMock()
        MockNEMESYS.return_value = MagicMock()
        pipeline = TrafficPipeline(model_path="/fake/model.pth")
        assert pipeline is not None
