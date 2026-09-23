"""Test field graph construction"""
from nemesys_gnn4id.nemesys.field_graph_builder import (
    FIELD_FEAT_DIM, PROTO_FEAT_DIM, PROTO_CLASSES, build_proto_graph,
)


class TestConstants:
    def test_proto_classes_count(self):
        # Pinned to the exact vocabulary rather than to a bare count. The corpus was
        # extended from 10 to 12 classes (TLS and mDNS added) when FieldProtoGNN V2 was
        # trained and the model now emits 12 logits, but this assertion stayed at 10 and
        # left the whole suite red. Comparing the full list also catches silent
        # reordering, which a count check would miss.
        assert PROTO_CLASSES == [
            'DHCP', 'DNS', 'NTP', 'Modbus', 'DNP3', 'S7Comm',
            'MQTT', 'HTTP', 'SSH', 'FTP', 'TLS', 'mDNS',
        ]

    def test_known_protocols(self):
        assert "DNS" in PROTO_CLASSES
        assert "HTTP" in PROTO_CLASSES
        assert "MQTT" in PROTO_CLASSES
        assert "Modbus" in PROTO_CLASSES

    def test_feature_dimensions_positive(self):
        # Exact values, not merely positivity. FIELD_FEAT_DIM is 12 base coordinates plus
        # a six-way field-type one-hot; PROTO_FEAT_DIM is 16 base coordinates plus two
        # field-type ratios. A bare "> 0" is what let the stale "14 base" comment in
        # field_graph_builder.py go unnoticed.
        assert FIELD_FEAT_DIM == 18
        assert PROTO_FEAT_DIM == 18


class TestBuildProtoGraph:
    def test_basic_graph_structure(self):
        segments = [
            {"offset": 0, "type": "binary", "value": [0, 1, 2], "size": 3},
            {"offset": 3, "type": "text", "value": [0x41, 0x42], "size": 2},
        ]
        graph = build_proto_graph(segments, label=0)
        assert graph is not None
        assert "proto" in graph.node_types
        assert "field" in graph.node_types
