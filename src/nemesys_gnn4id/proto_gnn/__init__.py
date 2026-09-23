try:
    from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE
except ImportError:
    FieldProtoGNN = None
    FieldProtoGNN_SAGE = None
try:
    from nemesys_gnn4id.proto_gnn.classifier import FieldProtocolClassifier
except ImportError:
    FieldProtocolClassifier = None
__all__ = ["FieldProtoGNN", "FieldProtoGNN_SAGE", "FieldProtocolClassifier"]
