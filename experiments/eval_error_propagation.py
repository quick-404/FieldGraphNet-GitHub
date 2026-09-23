# -*- coding: utf-8 -*-
"""Error propagation: FieldProtoGNN identification errors -> domain router -> fusion.

Measures (1) the deployed V2 model's per-protocol identification recall on real
pcaps (HTTP in-domain, DNS/NTP OOD), (2) router behavior: how often an
identification error flips the in/OOD routing decision, and (3) the resulting
impact on fusion decisions (a misrouted flow enters the wrong detection arm).

The deployed router (`nemesys_gnn4id.nemesys.cluster.route_flows_by_protocol`,
consumed by `pipeline.py`) decides on the *post-threshold* protocol name only
(`is_in_gnn_domain(proto, 0, 0)`): low-confidence predictions (threshold 0.65)
and NEMESYS-unknown overrides become UNKNOWN_PROTO and are routed to the
structural/OOD arm.  A "routing flip" is a message whose actual router decision
differs from the decision a perfect identification would have produced.

Usage:
    python experiments/eval_error_propagation.py

Output:
    eval_results/fusion_comparison/error_propagation.json
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(ROOT, 'eval_results', 'fusion_comparison', 'error_propagation.json')
MODEL_PATH = os.path.join(ROOT, 'models', 'field_proto_model_v2.pth')
METRICS_PATH = os.path.join(ROOT, 'models', 'field_proto_model_v2.metrics.json')
MAX_SEGMENTS = 100

# (ground-truth label, pcap path)
PCAPS = [
    ('HTTP', os.path.join(ROOT, 'data', 'data', 'http_100.pcap')),
    ('DNS', os.path.join(ROOT, 'data', 'data', 'dns_100.pcap')),
    ('NTP', os.path.join(ROOT, 'data', 'data', 'ntp_100.pcap')),
]


def router_decision(protocol, sport=0, dport=0):
    """Return 'in-domain' or 'OOD' exactly as the deployed pipeline routes."""
    from nemesys_gnn4id.nemesys.cluster import is_in_gnn_domain
    return 'in-domain' if is_in_gnn_domain(protocol, sport, dport) else 'OOD'


def main():
    from nemesys_gnn4id.proto_gnn.classifier import FieldProtocolClassifier, UNKNOWN_PROTO
    from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
    from nemesys_gnn4id.nemesys.cluster import route_flows_by_protocol

    clf = FieldProtocolClassifier(MODEL_PATH)
    analyzer = NEMESYSAnalyzer(sigma=0.6)

    out = {'_meta': {
        'task': 'W3-2 error propagation: identification -> router -> fusion',
        'model_path': MODEL_PATH,
        'confidence_threshold': clf.confidence_threshold,
        'max_segments_per_pcap': MAX_SEGMENTS,
        'router_rule': 'is_in_gnn_domain(protocol, sport=0, dport=0) '
                       '(post-threshold protocol; UNKNOWN_PROTO -> OOD)',
    }}

    for label, pcap in PCAPS:
        if not os.path.exists(pcap):
            out[label] = {'error': 'missing {}'.format(pcap)}
            print(label, 'ERROR missing', pcap)
            continue

        res = analyzer.analyze(pcap)
        segs = (res.get('segments') or [])[:MAX_SEGMENTS]
        preds = clf.predict_segments(segs)
        msgs = preds.get('messages', []) if isinstance(preds, dict) else preds
        n = len(msgs)
        if n == 0:
            out[label] = {'n': 0, 'note': 'no messages'}
            print(label, json.dumps(out[label], ensure_ascii=False))
            continue

        correct = sum(1 for m in msgs if m.get('raw_protocol') == label)
        unknown = sum(1 for m in msgs if m.get('protocol') == UNKNOWN_PROTO)
        known = sum(1 for m in msgs if m.get('is_known'))
        # low-confidence predictions (would become UNKNOWN_PROTO at the router)
        low_conf = sum(1 for m in msgs if m.get('is_low_confidence'))
        nemesys_unknown_ovr = sum(1 for m in msgs if m.get('nemesys_unknown_override'))

        true_domain = router_decision(label)  # what a perfect ID would route to
        true_ood = true_domain == 'OOD'

        flips = []
        routed_ood = 0
        routed_ood_raw = 0
        for m in msgs:
            actual = router_decision(m.get('protocol', UNKNOWN_PROTO))
            raw_actual = router_decision(m.get('raw_protocol', UNKNOWN_PROTO))
            if actual == 'OOD':
                routed_ood += 1
            if raw_actual == 'OOD':
                routed_ood_raw += 1
            if actual != true_domain:
                flips.append({
                    'message_index': m.get('message_index'),
                    'raw_protocol': m.get('raw_protocol'),
                    'protocol': m.get('protocol'),
                    'confidence': m.get('confidence'),
                    'is_low_confidence': m.get('is_low_confidence'),
                    'nemesys_protocol': m.get('nemesys_protocol'),
                    'error': m.get('error'),
                })

        # identification errors (raw label differs from ground truth)
        id_errors = []
        for m in msgs:
            if m.get('raw_protocol') != label:
                id_errors.append({
                    'message_index': m.get('message_index'),
                    'true': label,
                    'raw_protocol': m.get('raw_protocol'),
                    'protocol': m.get('protocol'),
                    'confidence': m.get('confidence'),
                    'is_low_confidence': m.get('is_low_confidence'),
                })

        # flow-level routing exactly as the pipeline consumes it
        flow_routing = route_flows_by_protocol(preds.get('flow_predictions', {}))

        out[label] = {
            'n': n,
            'pcap': os.path.basename(pcap),
            'identification_recall': round(correct / n, 4),
            'unknown_rate': round(unknown / n, 4),
            'known_rate': round(known / n, 4),
            'low_confidence_rate': round(low_conf / n, 4),
            'nemesys_unknown_override_rate': round(nemesys_unknown_ovr / n, 4),
            'predicted_protocols': {p: c for p, c in Counter(
                m.get('raw_protocol') for m in msgs).most_common(5)},
            'true_domain': true_domain,
            # router behavior: what the deployed router (post-threshold) does
            'routed_ood_frac': round(routed_ood / n, 4),
            # routing if the raw (pre-threshold) identification were used
            'routed_ood_frac_raw': round(routed_ood_raw / n, 4),
            'routing_flips': len(flips),
            'routing_flip_rate': round(len(flips) / n, 4),
            'flip_details': flips,
            'identification_errors': len(id_errors),
            'identification_error_details': id_errors,
            'flow_routing': {
                'in_domain_flows': len(flow_routing['in_domain_keys']),
                'ood_flows': len(flow_routing['ood_keys']),
                'flow_protocols': {
                    str(k): v for k, v in flow_routing['flow_protocols'].items()
                },
            },
            'router_error_impact': (
                'routed to structural arm (OOD)' if true_ood else
                'stays in GNN arm (in-domain)'
            ),
        }
        print(label, json.dumps(
            {k: v for k, v in out[label].items() if k not in
             ('flip_details', 'identification_error_details', 'flow_routing')},
            ensure_ascii=False))

    # per-class identification metrics from the deployed model (checkpoint order)
    if os.path.exists(METRICS_PATH):
        m = json.load(open(METRICS_PATH, encoding='utf-8'))
        per = m.get('final_metrics', {}).get('per_class', {})
        names = clf.proto_classes
        out['_model_per_class'] = {
            names[int(k)] if int(k) < len(names) else k: v
            for k, v in sorted(per.items(), key=lambda kv: int(kv[0]))
        }
        out['_model_accuracy'] = m.get('final_metrics', {}).get('accuracy')
        out['_model_macro'] = m.get('final_metrics', {}).get('macro')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
