# -*- coding: utf-8 -*-
"""Canonical flow-key alignment between 82-dim flow features and 24-dim structural
fingerprints.

WHY THIS EXISTS
---------------
`extract_features_by_source()` (statistical_fusion_eval.py) caps at
`max_flows=50` **per chunk**, while `flow_structural_fingerprints()`
(unknown_protocol_detection.py) is **uncapped** (one entry per distinct flow).

Measured on the 23pcap chunks (2026-08-18):

    source                     feature rows   structural fingerprints
    BenignTraffic.pcap                    500                     3320
    DDoS-HTTP_Flood-.pcap                 500                     4398
    DNS_Spoofing.pcap                     500                     3093
    DictionaryBruteForce.pcap             500                     2269
    Mirai-udpplain.pcap                   233                      133
    Recon-PortScan.pcap                   500                    15403
    SqlInjection.pcap                     500                     4592

So the two containers do not share an index space, and the historical pattern

    benign_struct = list(struct.get(src, {}).values())
    benign_struct[i]          # <-- paired with benign_feats[i]

silently pairs flow *i*'s feature vector with a **different** flow's fingerprint.
The mismatch fails in the silent direction (no IndexError) because the fingerprint
list is the longer one. Consequences:

  * single-arm metrics keep correct *class* labels (the benign/attack blocks are
    class-pure), so historical numbers are not fabricated -- but the fitted
    centroids come from a different set of flows, and the attack test set is a
    different subset (Mirai silently used 133 of 233 flows);
  * genuine row-level *fusion* of a structural score with a flow-statistical
    score is meaningless, because the two scores describe unrelated flows.

THE FIX
-------
Build both arrays from the same canonical flow key
(`nemesys_gnn4id.pipeline.canonical_flow_tuple`, direction-insensitive). Every
feature row's key is looked up in the fingerprint map, and rows without a
fingerprint are dropped from both arrays so the two stay index-aligned.

Verified 100% coverage on all seven sources (500/500 benign, 233/233 Mirai, ...).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer
from nemesys_gnn4id.pipeline import canonical_flow_tuple

CHUNK_DIR = 'data/data/23pcap_chunks'
MODEL_PATH = 'models/model.pth'
DEFAULT_MAX_FLOWS = 50  # must match extract_features_by_source() for comparability


def flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """Canonical, direction-insensitive key in the same form as the fingerprint cache.

    `flow_structural_fingerprints()` stores its keys as `tuple(k.split('|'))`, i.e. a
    tuple of strings, so we reproduce exactly that form.
    """
    return tuple(str(x) for x in canonical_flow_tuple(
        src_ip, dst_ip, src_port, dst_port, proto))


def extract_features_with_keys(chunk_dir=CHUNK_DIR, model_path=MODEL_PATH,
                               max_flows=DEFAULT_MAX_FLOWS):
    """Return {source: [(flow_key, 82-dim feature), ...]}.

    The feature vector is built exactly as in
    `statistical_fusion_eval.extract_features_by_source()` (ports 77-81 zeroed),
    so values remain comparable with previously reported numbers.
    """
    with open(os.path.join(chunk_dir, '_chunk_index.json'), encoding='utf-8') as f:
        chunks = json.load(f)
    analyzer = GNN4IDAnalyzer(model_path=model_path)
    out = {}
    for chunk_name, (_label, src) in chunks.items():
        chunk_path = os.path.join(chunk_dir, chunk_name)
        if not os.path.exists(chunk_path):
            continue
        df = analyzer.extract_features(chunk_path, max_flows=max_flows)
        if df.empty or 'flow_features' not in df.columns:
            continue
        for _, row in df.iterrows():
            feat = list(row['flow_features'])
            for z in range(77, min(82, len(feat))):
                feat[z] = 0.0  # drop ports, as in extract_features_by_source
            key = flow_key(str(row.get('src_ip', '')), str(row.get('dst_ip', '')),
                           int(row.get('src_port', 0) or 0),
                           int(row.get('dst_port', 0) or 0),
                           int(row.get('protocol', 0) or 0))
            out.setdefault(src, []).append((key, feat))
    return out


def align_source(feature_rows, struct_flows, src, strict=True):
    """Align one source's feature rows with its structural fingerprints by flow key.

    feature_rows : [(flow_key, 82-dim feature), ...]
    struct_flows : {flow_key: 24-dim fingerprint}

    Returns (feats, fingerprints) as two equal-length lists, dropping feature rows
    that have no fingerprint. With strict=True, raises if anything was dropped, so
    a silent misalignment can never be reintroduced downstream.
    """
    feats, fps, missing = [], [], []
    for key, feat in feature_rows:
        fp = struct_flows.get(key)
        if fp is None:
            missing.append(key)
            continue
        feats.append(feat)
        fps.append(np.asarray(fp, dtype=np.float32))
    if missing and strict:
        raise AssertionError(
            f'{src}: {len(missing)}/{len(feature_rows)} feature rows have no '
            f'structural fingerprint (e.g. {missing[:2]}). Refusing to continue: '
            f'positional alignment would silently pair unrelated flows.')
    return feats, fps


def load_aligned(chunk_dir=CHUNK_DIR, model_path=MODEL_PATH,
                 max_flows=DEFAULT_MAX_FLOWS, strict=True, verbose=True):
    """Load BOTH representations for every source, aligned by canonical flow key.

    Returns (feat, struct) where
        feat[src]   = [82-dim feature, ...]
        struct[src] = [24-dim fingerprint, ...]
    and feat[src][i] and struct[src][i] describe the SAME flow for every i.
    Only sources present in the fingerprint data are returned.
    """
    from unknown_protocol_detection import flow_structural_fingerprints

    struct_raw = flow_structural_fingerprints()
    feat_rows = extract_features_with_keys(chunk_dir, model_path, max_flows)

    feat, struct = {}, {}
    for src, rows in feat_rows.items():
        flows = struct_raw.get(src)
        if not flows:
            continue
        f, s = align_source(rows, flows, src, strict=strict)
        feat[src], struct[src] = f, s
        if verbose:
            print(f'[ALIGN] {src:<28} features={len(f):>4} '
                  f'fingerprints={len(s):>4} (same flows, key-matched)')
    return feat, struct
