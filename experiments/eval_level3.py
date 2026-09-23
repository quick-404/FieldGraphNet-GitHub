# -*- coding: utf-8 -*-
"""Level-3 protocol (TLS, S7Comm) structural-space probe.

Measures how Level-3 protocols (encrypted/highly dynamic) behave in the 24-dim
structural fingerprint space vs Level-1/2 protocols: fingerprint spread
(intra-protocol dispersion) and distance to the nearest other-protocol centroid
(cross-protocol separation). The structural detector's discriminative power
depends on how regular/recoverable a protocol's message structure is — Level-3
is expected to be the hardest (previously "n/a" / excluded from the review).

NOTE on the real API: incremental_unknown_protocol.extract_fingerprints()
covers only http/mqtt/ssh/ftp/tls + OOD attack sources, so NTP/DHCP/DNS/Modbus/
S7Comm are NOT in that cache. This probe therefore re-runs the same pipeline
(NEMESYSAnalyzer.analyze + extract_structural_fingerprint) over the six
per-protocol pcaps (data/data/*_100.pcap — the same sources the V2 field-proto
model was trained on) and caches the fingerprints for fast re-runs.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.nemesys.cluster import extract_structural_fingerprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data', 'data')
OUT = os.path.join(ROOT, 'eval_results', 'fusion_comparison', 'level3.json')
FPS_CACHE = '/tmp/level3_fps.json'

LEVELS = {
    1: ['NTP', 'DHCP'],
    2: ['DNS', 'Modbus'],
    3: ['TLS', 'S7Comm'],
}

PCAP_STEMS = {
    'NTP': 'ntp_100',
    'DHCP': 'dhcp_100',
    'DNS': 'dns_100',
    'Modbus': 'modbus_100',
    'TLS': 'tls_100',
    'S7Comm': 's7comm_100',
}

# V2 field-proto model identification recall per probed protocol
# (models/field_proto_model_v2.metrics.json, final_metrics.per_class).
V2_RECALL = {
    'NTP': 1.0,
    'DHCP': 1.0,
    'DNS': 1.0,
    'Modbus': 0.9333,
    'TLS': 1.0,
    'S7Comm': 1.0,
}


def extract_fingerprints():
    """Return {proto: list of 24-dim fingerprints} for the six probe protocols."""
    if os.path.exists(FPS_CACHE):
        with open(FPS_CACHE, encoding='utf-8') as f:
            raw = json.load(f)
        return {k: [np.array(x, dtype=np.float32) for x in v]
                for k, v in raw.items()}

    analyzer = NEMESYSAnalyzer()
    out = {}
    for proto in [p for lvl in (1, 2, 3) for p in LEVELS[lvl]]:
        pcap = os.path.join(DATA_DIR, PCAP_STEMS[proto] + '.pcap')
        if not os.path.exists(pcap):
            print(f'[SKIP] {pcap} 不存在')
            out[proto] = []
            continue
        result = analyzer.analyze(pcap)
        out[proto] = [extract_structural_fingerprint(s)
                      for s in result.get('segments', []) if isinstance(s, dict)]
        print(f'  {proto:7s}: {len(out[proto])} fingerprints from {PCAP_STEMS[proto]}.pcap')
    with open(FPS_CACHE, 'w', encoding='utf-8') as f:
        json.dump({k: [x.tolist() for x in v] for k, v in out.items()}, f)
    return out


def main():
    fps = extract_fingerprints()  # {proto: [fp24]}
    out = {}
    centroids = {}

    for proto in [p for lvl in (1, 2, 3) for p in LEVELS[lvl]]:
        arr = fps.get(proto, [])
        if not arr:
            out[proto] = {'level': next(l for l, ps in LEVELS.items() if proto in ps),
                          'n': 0, 'note': 'no fingerprints'}
            continue
        arr = np.array(arr)
        centroid = arr.mean(axis=0)
        centroids[proto] = centroid
        # intra-protocol dispersion: mean distance of messages to own centroid
        dists = np.linalg.norm(arr - centroid, axis=1)
        out[proto] = {
            'level': next(l for l, ps in LEVELS.items() if proto in ps),
            'n': int(len(arr)),
            'mean_dist_to_centroid': round(float(dists.mean()), 4),
            'std_dist': round(float(dists.std()), 4),
            'fingerprint_norm_mean': round(float(np.linalg.norm(arr, axis=1).mean()), 4),
            'v2_identification_recall': V2_RECALL.get(proto),
        }
        print(proto, out[proto])

    # cross-protocol separation: mean distance of each proto's messages to the
    # nearest centroid among the OTHER probed protocols (leave-one-out).
    for proto in list(centroids):
        others = np.array([c for p, c in centroids.items() if p != proto])
        arr = np.array(fps[proto])
        if len(others) == 0:
            out[proto]['mean_dist_to_nearest_other'] = None
            continue
        d = np.linalg.norm(arr[:, None, :] - others[None, :, :], axis=2).min(axis=1)
        out[proto]['mean_dist_to_nearest_other'] = round(float(d.mean()), 4)
        print(f'  {proto:7s} nearest-other-centroid dist: {out[proto]["mean_dist_to_nearest_other"]}')

    # level aggregates: mean intra-protocol dispersion across member protocols
    summary = {}
    for lvl, protos in LEVELS.items():
        ds = [out[p]['mean_dist_to_centroid'] for p in protos
              if out.get(p, {}).get('n', 0) > 0]
        summary[f'level{lvl}'] = {
            'protos': protos,
            'mean_dispersion': round(float(np.mean(ds)), 4) if ds else None,
        }
    out['_summary'] = {
        'levels': summary,
        'level3_dispersion_gt_level1': bool(
            summary['level3']['mean_dispersion'] is not None
            and summary['level1']['mean_dispersion'] is not None
            and summary['level3']['mean_dispersion'] > summary['level1']['mean_dispersion']),
        'note': ('Level-3 (TLS/S7Comm) expected to show the highest intra-protocol '
                 'dispersion; verify with the per-protocol numbers above.'),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('saved', OUT)
    print('summary:', json.dumps(out['_summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
