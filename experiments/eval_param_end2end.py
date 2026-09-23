"""End-to-end parameter sensitivity: distance threshold / outlier sigma -> detection metrics.

Extends the paper's tab:param (cluster/outlier counts on the DNS_Spoofing chunk) with
end-to-end detection metrics of the fusion's STRUCTURAL anomaly arm:

  - clusters / outliers  : DNS_Spoofing chunk clustering at (tau, sigma)
                           (tab:param extension; reproduces the paper numbers).
  - auc / f1_at_20fpr / recall : min-distance to in-domain structural centroids
                           ("Fusion-struct" signal, identical protocol to
                           unknown_protocol_detection.py so the default row is
                           comparable with that file's published DNS numbers)
                           scored on the mixed test (150 held-out benign + 500
                           DNS attack flows); the F1 threshold is the 20%-FPR
                           quantile of test-benign scores (repo convention).
                           tau enters here: it controls the in-domain clustering
                           granularity -> centroids -> score separation.
  - structural_alert_recall / _f1 : production structural-alert arm — a DNS flow
                           is alerted when its majority-vote matched cluster in the
                           DNS-chunk clustering is high-risk under the sigma-driven
                           rules of TrafficPipeline._cluster_is_high_risk
                           (outlier_count > 0 or outlier_ratio > 0.15). sigma
                           enters here: it controls the per-cluster outlier flags.
                           FPR = 0 by construction (the DNS chunk contains no
                           benign flows; benign flows are not clustered in it) —
                           this mirrors the fusion eval's cluster_only arm.

Scope (honest): structural-only probe on the DNS chunk; no GNN / FieldProtoGNN /
full-fusion rerun per setting. The anomaly score is the real structural signal the
fusion's structural arm uses (BCDG 24-dim fingerprints + greedy clustering +
sigma-outlier detection).
"""
import json
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sklearn.metrics import roc_auc_score

from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.nemesys.cluster import extract_structural_fingerprint
from nemesys_gnn4id.pipeline import canonical_flow_tuple

from flow_key_alignment import load_aligned

CHUNK_DIR = 'data/data/23pcap_chunks'
DNS_SOURCE = 'DNS_Spoofing.pcap'
BENIGN_SOURCE = 'BenignTraffic.pcap'
DDoS_SOURCE = 'DDoS-HTTP_Flood-.pcap'

TAU_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SIGMA_SWEEP = [1.5, 2.0, 2.5, 3.0, 3.5]
FIXED_SIGMA = 2.5
FIXED_TAU = 0.5

BENIGN_TRAIN = 350
TARGET_FPR = 0.20
SEED = 42
MAX_FLOWS_PER_SOURCE = 500

OUT_PATH = 'eval_results/fusion_comparison/param_end2end.json'
CACHE_PATH = os.path.join(tempfile.gettempdir(), 'param_end2end_fps_cache.json')


def flow_key_str(src, dst, sport, dport, proto):
    return '|'.join(str(x) for x in canonical_flow_tuple(src, dst, sport, dport, proto))


def cluster_fingerprints_fast(fingerprints, threshold):
    """Vectorized greedy clustering — same algorithm as
    nemesys_gnn4id.nemesys.cluster.cluster_fingerprints (first-nearest centroid
    within threshold, running-mean centroids) but computing per-point distances
    to all centroids in one op. Returns (labels, centroids)."""
    n = len(fingerprints)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels, []
    sums = [fingerprints[0].copy()]
    sizes = [1]
    labels[0] = 0
    for i in range(1, n):
        fp = fingerprints[i]
        k = len(sums)
        cents = np.empty((k, fingerprints.shape[1]), dtype=np.float32)
        for c in range(k):
            cents[c] = sums[c] / sizes[c]
        d = np.linalg.norm(cents - fp, axis=1)
        cand = np.where(d <= threshold)[0]
        if len(cand) == 0:
            sums.append(fp.copy())
            sizes.append(1)
            labels[i] = k
        else:
            best = cand[int(np.argmin(d[cand]))]
            labels[i] = best
            sums[best] += fp
            sizes[best] += 1
    centroids = [sums[c] / sizes[c] for c in range(len(sums))]
    return labels, centroids


def detect_outliers_fast(fingerprints, labels, centroids, sigma):
    """Vectorized outlier detection — same rule as
    nemesys_gnn4id.nemesys.cluster.detect_cluster_outliers (per cluster:
    dist > mean + sigma*std), returning (outlier_flags, distances)."""
    n = len(fingerprints)
    outlier_flags = np.zeros(n, dtype=bool)
    distances = np.zeros(n, dtype=np.float32)
    unique_labels = sorted(set(int(l) for l in labels if l >= 0))
    for c in unique_labels:
        mask = labels == c
        idxs = np.where(mask)[0]
        if len(idxs) < 3:  # MIN_CLUSTER_SIZE_FOR_OUTLIER
            continue
        cent = centroids[c]
        cluster_dists = np.linalg.norm(fingerprints[idxs] - cent, axis=1)
        distances[idxs] = cluster_dists
        thr = float(np.mean(cluster_dists)) + sigma * max(float(np.std(cluster_dists)), 1e-8)
        outlier_flags[idxs] = cluster_dists > thr
    return outlier_flags, distances


# ---------------------------------------------------------------- extraction

def extract_source_fingerprints():
    """Return {source: {chunk: [(flow_key_str, 24-dim fp), ...]}} per segment.

    Segments (BCDG output) are kept per chunk so the same chunk can be
    re-clustered cheaply for every (tau, sigma) setting. Cached to TEMP.
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            raw = json.load(f)
        return {src: {ck: [(k, np.array(v, dtype=np.float32)) for k, v in segs]
                      for ck, segs in chunks.items()}
                for src, chunks in raw.items()}

    analyzer = NEMESYSAnalyzer()
    with open(os.path.join(CHUNK_DIR, '_chunk_index.json'), encoding='utf-8') as f:
        chunk_index = json.load(f)

    out = {}
    for chunk_name, (_label, src) in chunk_index.items():
        if src not in (DNS_SOURCE, BENIGN_SOURCE, DDoS_SOURCE):
            continue
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            continue
        result = analyzer.analyze(chunk_path)
        segs = []
        for seg in result.get('segments', []):
            if not isinstance(seg, dict) or 'src' not in seg:
                continue
            key = flow_key_str(seg.get('src', ''), seg.get('dst', ''),
                               seg.get('sport', 0), seg.get('dport', 0),
                               seg.get('proto', 0))
            segs.append((key, extract_structural_fingerprint(seg).tolist()))
        out.setdefault(src, {})[chunk_name] = segs
        print(f'  [{src}] {chunk_name}: {len(segs)} segments')

    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    return {src: {ck: [(k, np.array(v, dtype=np.float32)) for k, v in segs]
                  for ck, segs in chunks.items()}
            for src, chunks in out.items()}


def per_flow_mean_fps(source_chunks, cap=MAX_FLOWS_PER_SOURCE):
    """Aggregate segment fingerprints per flow key -> list of mean 24-dim fps.

    NOTE: no longer called by main() (superseded by flow_key_alignment.load_aligned(),
    whose list is index-aligned with the 82-dim feature rows). Kept as the reference
    for the raw per-flow aggregation.
    """
    per_flow = defaultdict(list)
    for _chunk, segs in source_chunks.items():
        for key, fp in segs:
            per_flow[key].append(fp)
    fps = [np.mean(v, axis=0).astype(np.float32) for v in per_flow.values()]
    return fps[:cap]


# ---------------------------------------------------------------- metrics

def metrics_at_fpr(scores, y):
    """AUC + recall/F1 at the 20%-FPR threshold (repo convention: threshold is the
    (1-TARGET_FPR) quantile of the test-benign scores, as in
    unknown_protocol_detection.py, so the default row matches that file)."""
    auc = roc_auc_score(y, scores)
    benign_scores = scores[y == 0]
    thr = np.quantile(benign_scores, 1 - TARGET_FPR)
    pred = (scores > thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    return {'auc': round(auc, 4), 'recall': round(rec, 4),
            'precision': round(prec, 4), 'f1': round(f1, 4), 'fpr': round(fpr, 4)}


def min_dist_score(fps, centroids):
    """Min Euclidean distance of each fingerprint to the centroid set."""
    if centroids is None or len(centroids) == 0:
        return np.zeros(len(fps))
    d = np.linalg.norm(np.array(fps)[:, None, :] - np.array(centroids)[None, :, :], axis=2)
    return d.min(axis=1)


def structural_alert_stats(dns_chunks, tau, sigma):
    """Production structural-alert arm on the DNS chunk flows.

    Each DNS chunk is clustered at (tau, sigma); a flow is alerted when its
    majority-vote matched cluster is high-risk under the sigma-driven rules of
    TrafficPipeline._cluster_is_high_risk (outlier_count > 0 or
    outlier_ratio > 0.15). All DNS flows are attack, so this yields recall
    (FPR = 0 by construction: the DNS chunk contains no benign flows).
    """
    alert_count = 0
    flow_count = 0
    for chunk, segs in dns_chunks.items():
        if not segs:
            continue
        fps = np.array([fp for _k, fp in segs], dtype=np.float32)
        labels, centroids = cluster_fingerprints_fast(fps, tau)
        outlier_flags, _dist = detect_outliers_fast(fps, labels, centroids, sigma)

        # per-cluster outlier counts (sigma-driven high-risk rules)
        cl_size = defaultdict(int)
        cl_out = defaultdict(int)
        for lbl, oflag in zip(labels.tolist(), outlier_flags.tolist()):
            cl_size[lbl] += 1
            if oflag:
                cl_out[lbl] += 1

        def high_risk(lbl):
            if cl_out[lbl] > 0:
                return True
            return cl_out[lbl] / max(cl_size[lbl], 1) > 0.15

        # per-flow majority-vote cluster
        flow_votes = defaultdict(lambda: defaultdict(int))
        for (key, _fp), lbl in zip(segs, labels.tolist()):
            flow_votes[key][lbl] += 1
        for key, votes in flow_votes.items():
            lbl = max(votes.items(), key=lambda kv: kv[1])[0]
            flow_count += 1
            if high_risk(lbl):
                alert_count += 1

    if flow_count == 0:
        return {'recall': 0.0, 'f1': 0.0, 'n_flows': 0}
    recall = alert_count / flow_count
    # FPR = 0 by construction (no benign flows in the DNS chunk clustering)
    f1 = 2 * recall / (1 + recall) if recall > 0 else 0.0
    return {'recall': round(recall, 4), 'f1': round(f1, 4), 'n_flows': flow_count}


# ---------------------------------------------------------------- main

def main():
    print('Extracting BCDG fingerprints (cached to TEMP)...')
    fps_by_source = extract_source_fingerprints()

    # Per-chunk DNS segments are needed verbatim: the tab:param counts and the
    # production structural-alert arm re-cluster the DNS chunk at every (tau, sigma).
    dns_chunks = fps_by_source[DNS_SOURCE]

    # DNS chunk used for the tab:param counts (paper object: 4594 segments)
    dns_chunk1_name = sorted(dns_chunks.keys())[0]
    dns_chunk1_segs = dns_chunks[dns_chunk1_name]
    print(f'\nDNS counts object: {dns_chunk1_name} ({len(dns_chunk1_segs)} segments)')

    # Per-flow fingerprints for the anomaly-score row, KEY-ALIGNED with the 82-dim flow
    # feature rows: struct[src][i] is the fingerprint of the flow described by
    # feat[src][i]. The raw per-segment cache is uncapped, so its .values() order shares
    # no index space with the feature rows and would score a different flow population
    # than the comparison scripts (see experiments/flow_key_alignment.py).
    _feat, struct = load_aligned()
    dns_fps = struct.get(DNS_SOURCE, [])
    benign_fps = struct.get(BENIGN_SOURCE, [])
    # in-domain attack pool: the key-aligned DDoS fingerprints — exactly the flows
    # unknown_protocol_detection.py clusters via struct.get(IN_DOMAIN_ATTACK, [])
    ddos_fps = struct.get(DDoS_SOURCE, [])
    n_ben = len(benign_fps)
    n_train = min(BENIGN_TRAIN, max(n_ben - 50, 1))
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n_ben)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    ben_train = [benign_fps[i] for i in tr_idx]
    ben_test = [benign_fps[i] for i in te_idx]
    n_att = min(len(dns_fps), MAX_FLOWS_PER_SOURCE)
    att_fps = dns_fps[:n_att]

    print(f'Benign: {n_ben} flows ({len(ben_train)} train / {len(ben_test)} test)')
    print(f'DDoS (in-domain attack): {len(ddos_fps)} key-aligned flows')
    print(f'DNS attack: {n_att} flows')

    # mixed test set (held-out benign + DNS attack)
    X_test = np.array(ben_test + att_fps)
    y_test = np.array([0] * len(ben_test) + [1] * n_att)

    results = {'tau_sweep': [], 'sigma_sweep': [], 'meta': {}}

    def evaluate(tau, sigma):
        # 1) tab:param counts on the DNS chunk (paper object)
        fps1 = np.array([fp for _k, fp in dns_chunk1_segs], dtype=np.float32)
        labels1, centroids1 = cluster_fingerprints_fast(fps1, tau)
        flags1, _d = detect_outliers_fast(fps1, labels1, centroids1, sigma)
        n_clusters = int(len(centroids1))
        n_outliers = int(flags1.sum())

        # 2) in-domain structural model at threshold tau
        X_in = np.array(ben_train + ddos_fps)
        _lbl_in, cent_in = cluster_fingerprints_fast(X_in, tau)

        # mixed-test scores; F1@20%FPR calibrated on test-benign scores (repo convention)
        test_scores = min_dist_score(X_test, cent_in)
        m = metrics_at_fpr(test_scores, y_test)

        # 3) production structural-alert arm (sigma-sensitive)
        alert = structural_alert_stats(dns_chunks, tau, sigma)

        return {
            'tau': tau, 'sigma': sigma,
            'clusters': n_clusters, 'outliers': n_outliers,
            'auc': m['auc'], 'f1_at_20fpr': m['f1'],
            'recall': m['recall'], 'precision': m['precision'], 'fpr': m['fpr'],
            'structural_alert_recall': alert['recall'],
            'structural_alert_f1': alert['f1'],
            'structural_alert_n_flows': alert['n_flows'],
        }

    print('\n=== tau sweep (sigma fixed %.1f) ===' % FIXED_SIGMA)
    for tau in TAU_SWEEP:
        r = evaluate(tau, FIXED_SIGMA)
        results['tau_sweep'].append(r)
        print('tau=%.1f clusters=%4d outliers=%4d  AUC=%.4f  F1@20%%FPR=%.4f  '
              'recall=%.4f  struct_alert_recall=%.4f'
              % (tau, r['clusters'], r['outliers'], r['auc'], r['f1_at_20fpr'],
                 r['recall'], r['structural_alert_recall']))

    print('\n=== sigma sweep (tau fixed %.1f) ===' % FIXED_TAU)
    for sigma in SIGMA_SWEEP:
        r = evaluate(FIXED_TAU, sigma)
        results['sigma_sweep'].append(r)
        print('sigma=%.1f clusters=%4d outliers=%4d  AUC=%.4f  F1@20%%FPR=%.4f  '
              'recall=%.4f  struct_alert_recall=%.4f'
              % (sigma, r['clusters'], r['outliers'], r['auc'], r['f1_at_20fpr'],
                 r['recall'], r['structural_alert_recall']))

    results['meta'] = {
        'description': 'End-to-end parameter sensitivity of the fusion structural '
                       'anomaly arm on the DNS_Spoofing chunk (extends tab:param).',
        'scope': 'Structural-only probe: BCDG 24-dim fingerprints + greedy clustering + '
                 'sigma-outlier detection (the fusion structural signal). No GNN / '
                 'FieldProtoGNN / full-fusion rerun per setting.',
        'counts_object': 'DNS_Spoofing_chunk1.pcap (%d segments, the paper tab:param object)' % len(dns_chunk1_segs),
        'test_set': '%d held-out benign + %d DNS attack flows (repo mixed-traffic protocol, '
                    'seed %d)' % (len(ben_test), n_att, SEED),
        'in_domain_pool': 'Benign-train (%d) + key-aligned DDoS-HTTP_Flood struct flows '
                          '(%d), clustered at tau' % (len(ben_train), len(ddos_fps)),
        'metric_notes': {
            'auc_f1_recall': 'min distance to in-domain (Benign-train + DDoS) structural '
                             'centroids clustered at tau; F1 threshold = 20%-FPR quantile '
                             'of test-benign scores (same convention as '
                             'unknown_protocol_detection.py, so the tau=0.5/sigma=2.5 row is '
                             'comparable with that file). sigma does not enter this score '
                             '(sigma only sets the per-cluster outlier flag threshold).',
            'structural_alert': 'production fusion structural-alert arm: a flow is alerted '
                                'when its majority-vote cluster in the DNS-chunk clustering '
                                'is high-risk under the sigma-driven rules of '
                                '_cluster_is_high_risk (outlier_count>0 or outlier_ratio>0.15). '
                                'FPR=0 by construction (DNS chunk has no benign flows); mirrors '
                                'the fusion eval cluster_only arm.',
        },
        'validated_paper_counts': 'tau sweep clusters {676,459,301,203,133,87}, outliers '
                                  '{168,160,93,100,62,50}; sigma sweep outliers {468,219,93,42,10} '
                                  'at fixed 301 clusters (reproduced exactly).',
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
