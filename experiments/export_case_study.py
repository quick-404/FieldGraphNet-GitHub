"""Export per-flow case-study evidence (BCDG fields + cluster + fusion) for the paper.

Runs the NEMESYS-first pipeline on a chunk, finds flows whose fusion status is
`possible_missed_unknown_protocol_attack` (GNN says benign, but structural
clustering flags a high-risk unknown-protocol anomaly), and dumps the raw
field-level evidence needed for the paper's green/purple box:

  * BCDG field segmentation of the flow's messages (offset, length, type, bytes)
  * cluster statistics (size, entropy, cohesion, mean distance, indicators)
  * per-segment distance to the cluster centroid
  * GNN class / confidence / OOD flag and the full fusion decision

Usage:
  python experiments/export_case_study.py [--chunk CHUNK] [--max-flows N] [--out PATH]
"""
import sys, os, json, argparse

sys.path.insert(0, 'src')

from nemesys_gnn4id.pipeline import TrafficPipeline, flow_meta_key, segment_flow_key

MODEL_PATH = 'models/model.pth'
PROTO_MODEL = 'field_proto_model_bcdg.pth'
CHUNK_DIR = 'data/data/23pcap_chunks'
DEFAULT_OUT = 'eval_results/fusion_comparison/case_study_evidence.json'

TARGET_STATUS = 'possible_missed_unknown_protocol_attack'


def hexstr(b):
    return bytes(b).hex() if isinstance(b, (bytes, bytearray, list)) else ''


def flow_segment_keys(segments):
    """Map canonical flow key -> list of (segment_index, segment)."""
    out = {}
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        k = segment_flow_key(seg)
        if k is not None:
            out.setdefault(k, []).append((i, seg))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', default='DNS_Spoofing_chunk1.pcap')
    parser.add_argument('--max-flows', type=int, default=50)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--status', default=TARGET_STATUS,
                        help='fusion status to export (default: possible_missed)')
    args = parser.parse_args()

    chunk_path = os.path.join(CHUNK_DIR, args.chunk)
    if not os.path.exists(chunk_path):
        print(f'ERROR: {chunk_path} not found')
        sys.exit(1)

    print(f'Pipeline on {chunk_path} (max_flows={args.max_flows}) ...')
    p = TrafficPipeline(device='cpu', model_path=MODEL_PATH,
                        proto_model_path=PROTO_MODEL)
    result = p.analyze(chunk_path, max_flows=args.max_flows)

    gnn_results = result.get('gnn', {}).get('results', [])
    nemesys = result.get('nemesys', {})
    segments = nemesys.get('segments', []) if nemesys else []
    clustering = result.get('clustering', {})
    flow_metadata = result.get('flow_metadata', [])

    if not segments:
        print('WARNING: no NEMESYS segments in result; cannot build field evidence')
    if not flow_metadata and 'metadata' in result:
        flow_metadata = result.get('metadata', [])

    # Build per-flow segment lookup
    seg_by_flow = flow_segment_keys(segments)
    labels = clustering.get('labels', [])
    distances = clustering.get('distances', [])
    clusters = {c['cluster_id']: c for c in clustering.get('clusters', [])}

    matched = []
    for idx, r in enumerate(gnn_results):
        fusion = r.get('fusion_decision', {})
        if fusion.get('status') != args.status:
            continue

        # Locate the flow metadata
        fi = r.get('flow_index', idx)
        meta = flow_metadata[fi] if fi < len(flow_metadata) else None
        key = flow_meta_key(meta) if meta else None
        segs = seg_by_flow.get(key, [])

        # Per-segment cluster evidence
        seg_evidence = []
        cluster_id = fusion.get('cluster_id')
        for si, seg in segs:
            lbl = int(labels[si]) if si < len(labels) else -1
            dist = float(distances[si]) if si < len(distances) else None
            fields = []
            for ft in seg.get('field_types', []):
                fields.append({
                    'offset': ft.get('offset'),
                    'length': ft.get('length'),
                    'type': ft.get('type'),
                    'bytes_hex': hexstr(ft.get('bytes', b'')),
                })
            seg_evidence.append({
                'segment_index': si,
                'protocol': seg.get('protocol'),
                'payload_size': seg.get('payload_size'),
                'payload_hex': seg.get('payload_hex', '')[:120],
                'bcdg_field_count': seg.get('bcdg_segments'),
                'cluster_label': lbl + 1 if lbl >= 0 else None,
                'cluster_distance': round(dist, 4) if dist is not None else None,
                'fields': fields,
            })

        cluster_desc = clusters.get(cluster_id)
        matched.append({
            'flow_index': idx,
            'flow_meta': {
                'src_ip': meta.get('src_ip') if meta else None,
                'dst_ip': meta.get('dst_ip') if meta else None,
                'src_port': meta.get('src_port') if meta else None,
                'dst_port': meta.get('dst_port') if meta else None,
                'protocol': meta.get('protocol') if meta else None,
            },
            'gnn': {
                'class_id': r.get('class_id'),
                'class_name': r.get('class_name'),
                'class_name_en': r.get('class_name_en'),
                'confidence': r.get('confidence'),
                'ood': r.get('ood'),
                'gnn_skipped': r.get('gnn_skipped'),
                'gnn_unreliable': r.get('gnn_unreliable'),
                'original_class_id': r.get('_gnn_original_class_id'),
                'original_class_name': r.get('_gnn_original_class_name'),
            },
            'protocol_prediction': r.get('protocol_prediction'),
            'fusion_decision': fusion,
            'cluster': cluster_desc,
            'segments': seg_evidence,
        })

    if not matched:
        print(f'No flows with status={args.status} in this chunk '
              f'(chunk may need different max_flows or another chunk).')
        print('Statuses present:')
        from collections import Counter
        c = Counter(f.get('fusion_decision', {}).get('status', 'none')
                    for f in gnn_results)
        for st, n in c.most_common():
            print(f'  {st}: {n}')
        sys.exit(1)

    print(f'\nMatched {len(matched)} flow(s) with status={args.status}')
    for m in matched:
        print(f"  Flow {m['flow_index']}: gnn={m['gnn']['class_name_en']} "
              f"conf={m['gnn']['confidence']} ood={m['gnn']['ood']} "
              f"cluster={m['fusion_decision'].get('cluster_id')} "
              f"risk={m['fusion_decision'].get('risk_level')} "
              f"n_segments={len(m['segments'])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'chunk': args.chunk, 'max_flows': args.max_flows,
                   'status': args.status, 'flows': matched},
                  f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
