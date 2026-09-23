"""Test cluster analyzer"""
from nemesys_gnn4id.nemesys.cluster import is_in_gnn_domain


class TestIsInGnnDomain:
    def test_known_protocol_name(self):
        assert is_in_gnn_domain("HTTP", 0, 0) is True
        assert is_in_gnn_domain("MQTT", 0, 0) is True

    def test_known_port(self):
        assert is_in_gnn_domain("TCP", 80, 443) is True
        assert is_in_gnn_domain("TCP", 443, 1024) is True

    def test_unknown_protocol_and_port(self):
        assert is_in_gnn_domain("UNKNOWN", 9999, 8888) is False

    def test_known_protocol_trumps_unknown_port(self):
        assert is_in_gnn_domain("SSH", 9999, 8888) is True
