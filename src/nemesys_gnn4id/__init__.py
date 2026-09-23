"""
nemesys-gnn4id: GNN4ID + NEMESYS 集成分析系统

GNN4ID: GNN-based network attack detection (8-class)
NEMESYS: BCDG-based protocol reverse engineering
FieldProtoGNN: Field graph based protocol identification
"""

__version__ = "1.0.0"

from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer

try:
    from nemesys_gnn4id.pipeline import TrafficPipeline
except ImportError:
    TrafficPipeline = None

try:
    from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer, UnknownProtocolDetector
except ImportError:
    NEMESYSAnalyzer = None
    UnknownProtocolDetector = None

try:
    from nemesys_gnn4id.nemesys.cluster import ClusterAnalyzer
except ImportError:
    ClusterAnalyzer = None

try:
    from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE
except ImportError:
    FieldProtoGNN = None
    FieldProtoGNN_SAGE = None

try:
    from nemesys_gnn4id.proto_gnn.classifier import FieldProtocolClassifier
except ImportError:
    FieldProtocolClassifier = None

__all__ = [
    "GNN4IDAnalyzer",
    "TrafficPipeline",
    "NEMESYSAnalyzer",
    "UnknownProtocolDetector",
    "ClusterAnalyzer",
    "FieldProtoGNN",
    "FieldProtoGNN_SAGE",
    "FieldProtocolClassifier",
]
