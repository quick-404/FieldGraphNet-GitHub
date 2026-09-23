"""
Field graph protocol inference for NEMESYS segmentation results.
"""

import os
from collections import Counter, defaultdict

import torch
from torch_geometric.data import Batch

from nemesys_gnn4id.nemesys.field_graph_builder import FIELD_FEAT_DIM, PROTO_CLASSES, PROTO_FEAT_DIM, build_proto_graph
from nemesys_gnn4id.proto_gnn.model import FieldProtoGNN, FieldProtoGNN_SAGE, FieldProtoGNN_V2


DEFAULT_CONFIDENCE_THRESHOLD = 0.65
UNKNOWN_PROTO = 'UNKNOWN_PROTO'


def canonical_flow_tuple(src, dst, sport, dport, proto):
    """Build a direction-insensitive flow key."""
    sport = int(sport or 0)
    dport = int(dport or 0)
    proto = int(proto or 0)
    left = (str(src), sport)
    right = (str(dst), dport)
    if left <= right:
        return left + right + (proto,)
    return right + left + (proto,)


def segment_flow_key(seg):
    """Return a canonical flow key for a NEMESYS segment dict, or None."""
    if not isinstance(seg, dict):
        return None
    if 'src' not in seg or 'dst' not in seg:
        return None
    return canonical_flow_tuple(
        seg.get('src', ''),
        seg.get('dst', ''),
        seg.get('sport', 0),
        seg.get('dport', 0),
        seg.get('proto', 0),
    )


def flow_meta_key(meta):
    """Return a canonical flow key for GNN4ID flow metadata."""
    return canonical_flow_tuple(
        meta.get('src_ip', ''),
        meta.get('dst_ip', ''),
        meta.get('src_port', 0),
        meta.get('dst_port', 0),
        meta.get('protocol', 0),
    )


def _segments_from_dict(seg):
    """Convert a simulated NEMESYS segment dict to field segments with bytes."""
    field_types = seg.get('field_types', [])
    payload_hex = seg.get('payload_hex', '')
    raw = b''
    if payload_hex and payload_hex != '00':
        try:
            raw = bytes.fromhex(payload_hex)
        except (ValueError, TypeError):
            raw = b''

    msg_segments = []
    for ft in field_types:
        off = int(ft.get('offset', 0))
        ln = int(ft.get('length', 0))
        field_bytes = ft.get('bytes')
        if not isinstance(field_bytes, (bytes, bytearray)):
            field_bytes = raw[off:off + ln] if raw else b''
        msg_segments.append({
            'offset': off,
            'length': ln,
            'type': ft.get('type', 'unknown'),
            'bytes': bytes(field_bytes),
        })

    if not msg_segments and raw:
        msg_segments.append({
            'offset': 0,
            'length': len(raw),
            'type': 'binary',
            'bytes': raw,
        })
    return msg_segments


def _message_to_graph_input(msg):
    """Return msg_segments, sport, dport, ip_proto for a NEMESYS message."""
    sport = dport = ip_proto = 0
    msg_segments = msg

    if isinstance(msg, dict):
        sport = int(msg.get('sport', 0))
        dport = int(msg.get('dport', 0))
        ip_proto = int(msg.get('proto', 0))
        msg_segments = _segments_from_dict(msg)
    elif isinstance(msg, list) and msg:
        msg_obj = getattr(msg[0], 'message', None)
        if msg_obj:
            sport = int(getattr(msg_obj, 'sport', 0))
            dport = int(getattr(msg_obj, 'dport', 0))
            ip_proto = int(getattr(msg_obj, 'proto', 0))

    return msg_segments, sport, dport, ip_proto


def _nemesys_protocol_label(msg):
    """Return the protocol label produced by NEMESYS/fingerprint analysis."""
    if isinstance(msg, dict):
        return msg.get('protocol')
    return None


def _graph_summary(graph, msg_segments, sport, dport, ip_proto, graph_sport=None, graph_dport=None):
    return {
        'field_count': int(graph['field'].x.shape[0]) if 'field' in graph.node_types else len(msg_segments),
        'proto_nodes': int(graph['proto'].x.shape[0]) if 'proto' in graph.node_types else 1,
        'sport': int(sport),
        'dport': int(dport),
        'graph_sport': int(sport if graph_sport is None else graph_sport),
        'graph_dport': int(dport if graph_dport is None else graph_dport),
        'ip_proto': int(ip_proto),
        'edge_types': [str(edge_type) for edge_type in graph.edge_types],
    }


class FieldProtocolClassifier:
    """Load a FieldProtoGNN checkpoint and classify NEMESYS field graphs."""

    def __init__(self, model_path, device=None, confidence_threshold=None):
        if not model_path:
            raise ValueError('model_path is required')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'协议图模型不存在: {model_path}')

        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            self.checkpoint = torch.load(model_path, map_location=self.device)

        ckpt_field_dim = self.checkpoint.get('field_feat_dim')
        ckpt_proto_dim = self.checkpoint.get('proto_feat_dim')
        if ckpt_field_dim != FIELD_FEAT_DIM or ckpt_proto_dim != PROTO_FEAT_DIM:
            raise ValueError(
                f'协议图模型特征维度不匹配: checkpoint=({ckpt_field_dim},{ckpt_proto_dim}), '
                f'code=({FIELD_FEAT_DIM},{PROTO_FEAT_DIM})'
            )

        self.proto_classes = list(self.checkpoint.get('proto_classes', PROTO_CLASSES))
        self.drop_ports = bool(self.checkpoint.get('drop_ports', False))
        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(self.checkpoint.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD))
        )

        model_class = self.checkpoint.get('model_class', 'FieldProtoGNN')
        hidden = int(self.checkpoint.get('hidden_dim', 64))
        latent = int(self.checkpoint.get('latent_dim', 32))
        n_cls = int(self.checkpoint.get('num_classes', len(self.proto_classes)))
        if model_class == 'FieldProtoGNN_SAGE':
            self.model = FieldProtoGNN_SAGE(
                hidden_dim=hidden, latent_dim=latent, num_classes=n_cls, dropout=0.3).to(self.device)
        elif model_class == 'FieldProtoGNN_V2':
            heads = int(self.checkpoint.get('v2_heads') or 4)
            self.model = FieldProtoGNN_V2(
                hidden_dim=hidden, latent_dim=latent, num_classes=n_cls, dropout=0.3,
                heads=heads,
                field_feat_dim=int(self.checkpoint.get('field_feat_dim')),
                proto_feat_dim=int(self.checkpoint.get('proto_feat_dim')),
            ).to(self.device)
        elif model_class == 'FieldProtoGNN':
            self.model = FieldProtoGNN(
                hidden_dim=hidden, latent_dim=latent, num_classes=n_cls, dropout=0.3).to(self.device)
        else:
            raise ValueError(f'未知模型类型: {model_class}')
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def predict_message(self, msg, index=None):
        """Classify one NEMESYS message/segment."""
        try:
            msg_segments, sport, dport, ip_proto = _message_to_graph_input(msg)
            graph_sport = 0 if self.drop_ports else sport
            graph_dport = 0 if self.drop_ports else dport
            graph = build_proto_graph(msg_segments, sport=graph_sport, dport=graph_dport, proto=ip_proto)
            batch = Batch.from_data_list([graph]).to(self.device)
            logits = self.model(batch.x_dict, batch.edge_index_dict)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())
            raw_protocol = self.proto_classes[pred_id] if pred_id < len(self.proto_classes) else UNKNOWN_PROTO
            is_low = confidence < self.confidence_threshold
            protocol = raw_protocol if not is_low else UNKNOWN_PROTO
            nemesys_protocol = _nemesys_protocol_label(msg)
            nemesys_unknown_override = nemesys_protocol == UNKNOWN_PROTO
            if nemesys_unknown_override:
                protocol = UNKNOWN_PROTO
            flow_key = segment_flow_key(msg)

            return {
                'message_index': index,
                'protocol': protocol,
                'raw_protocol': raw_protocol,
                'model_protocol': raw_protocol,
                'nemesys_protocol': nemesys_protocol,
                'class_id': pred_id,
                'confidence': round(confidence, 4),
                'is_known': (not is_low) and (not nemesys_unknown_override),
                'is_low_confidence': is_low,
                'nemesys_unknown_override': nemesys_unknown_override,
                'confidence_threshold': self.confidence_threshold,
                'drop_ports': self.drop_ports,
                'flow_key': flow_key,
                'graph_summary': _graph_summary(
                    graph, msg_segments, sport, dport, ip_proto,
                    graph_sport=graph_sport, graph_dport=graph_dport,
                ),
            }
        except Exception as e:
            return {
                'message_index': index,
                'protocol': UNKNOWN_PROTO,
                'raw_protocol': UNKNOWN_PROTO,
                'model_protocol': UNKNOWN_PROTO,
                'nemesys_protocol': _nemesys_protocol_label(msg),
                'class_id': None,
                'confidence': 0.0,
                'is_known': False,
                'is_low_confidence': True,
                'nemesys_unknown_override': False,
                'confidence_threshold': self.confidence_threshold,
                'drop_ports': self.drop_ports,
                'flow_key': segment_flow_key(msg),
                'graph_summary': {},
                'error': str(e),
            }

    def predict_segments(self, segments):
        """Classify all NEMESYS segments and return message and flow aggregates."""
        messages = [self.predict_message(seg, index=i) for i, seg in enumerate(segments or [])]
        by_flow = defaultdict(list)
        for item in messages:
            if item.get('flow_key') is not None:
                by_flow[item['flow_key']].append(item)

        flow_predictions = {}
        for key, items in by_flow.items():
            known_items = [it for it in items if it.get('is_known')]
            vote_source = known_items if known_items else items
            votes = Counter(it.get('protocol', UNKNOWN_PROTO) for it in vote_source)
            model_votes = Counter(it.get('model_protocol', it.get('raw_protocol', UNKNOWN_PROTO)) for it in items)
            protocol = votes.most_common(1)[0][0] if votes else UNKNOWN_PROTO
            avg_conf = sum(it.get('confidence', 0.0) for it in vote_source) / max(len(vote_source), 1)
            low_count = sum(1 for it in items if it.get('is_low_confidence'))
            nemesys_unknown_count = sum(1 for it in items if it.get('nemesys_unknown_override'))
            flow_predictions[key] = {
                'protocol': protocol,
                'confidence': round(avg_conf, 4),
                'is_known': protocol != UNKNOWN_PROTO,
                'message_count': len(items),
                'low_confidence_count': low_count,
                'nemesys_unknown_override_count': nemesys_unknown_count,
                'votes': dict(votes),
                'model_votes': dict(model_votes),
                'message_indices': [it.get('message_index') for it in items],
            }

        summary = {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'drop_ports': self.drop_ports,
            'total_messages': len(messages),
            'known_messages': sum(1 for it in messages if it.get('is_known')),
            'unknown_messages': sum(1 for it in messages if not it.get('is_known')),
            'nemesys_unknown_overrides': sum(1 for it in messages if it.get('nemesys_unknown_override')),
            'protocol_distribution': dict(Counter(it.get('protocol', UNKNOWN_PROTO) for it in messages)),
            'model_protocol_distribution': dict(
                Counter(it.get('model_protocol', it.get('raw_protocol', UNKNOWN_PROTO)) for it in messages)
            ),
            'error_count': sum(1 for it in messages if it.get('error')),
        }

        return {
            'summary': summary,
            'messages': messages,
            'flow_predictions': flow_predictions,
        }
