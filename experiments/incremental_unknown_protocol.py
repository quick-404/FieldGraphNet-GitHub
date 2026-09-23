# experiments/incremental_unknown_protocol.py
"""Incremental unknown-protocol discovery with pseudo-labeling (advisor idea).

Batch 1 (10K messages): known/unknown split by structural similarity to the
GNN4ID training-domain protocol model; unknown messages greedy-clustered and
cohesive clusters pseudo-labeled unknown1/2/3. Batch 2 (10K) is handled in
Task 2 (re-cluster vs incremental).
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.nemesys.cluster import extract_structural_fingerprint, cluster_fingerprints

DATA_DIR = 'data/data'
CHUNK_DIR = 'data/data/23pcap_chunks'
FPS_CACHE = '/tmp/incremental_fps.json'

KNOWN_PROTOCOLS = ['http', 'mqtt', 'ssh', 'ftp', 'tls']
KNOWN_PCAPS = ['http_100.pcap', 'iot_mqtt.pcap', 'ssh_100.pcap',
               'ftp_100.pcap', 'tls_100.pcap']
OOD_SOURCES = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
               'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
BATCH_SIZE = 10000
KNOWN_THRESHOLD = 0.5
PSEUDO_MIN_SIZE = 50
SEED = 42


def extract_fingerprints():
    """Return {label: list of 24-dim fingerprints} for all sources (cached)."""
    if os.path.exists(FPS_CACHE):
        with open(FPS_CACHE, encoding='utf-8') as f:
            raw = json.load(f)
        return {k: [np.array(x, dtype=np.float32) for x in v]
                for k, v in raw.items()}

    analyzer = NEMESYSAnalyzer()
    out = {}
    for stem, proto in zip(KNOWN_PCAPS, KNOWN_PROTOCOLS):
        pcap = os.path.join(DATA_DIR, stem)
        if not os.path.exists(pcap):
            continue
        result = analyzer.analyze(pcap)
        out[proto] = [extract_structural_fingerprint(s) for s in result.get('segments', [])
                      if isinstance(s, dict)]
    with open(os.path.join(CHUNK_DIR, '_chunk_index.json'), encoding='utf-8') as f:
        chunks = json.load(f)
    for src in OOD_SOURCES:
        proto = src.replace('.pcap', '')
        fps = []
        for chunk_name, (_label, s) in chunks.items():
            if s != src:
                continue
            chunk_path = os.path.join(CHUNK_DIR, chunk_name)
            if not os.path.exists(chunk_path):
                continue
            result = analyzer.analyze(chunk_path)
            fps.extend(extract_structural_fingerprint(x) for x in result.get('segments', [])
                       if isinstance(x, dict))
            if len(fps) >= BATCH_SIZE:
                break
        out[proto] = fps[:BATCH_SIZE]
    with open(FPS_CACHE, 'w', encoding='utf-8') as f:
        json.dump({k: [x.tolist() for x in v] for k, v in out.items()}, f)
    return out


def build_known_model(known_fps):
    """Greedy-cluster known-protocol fingerprints -> centroids."""
    all_fps = [fp for fps in known_fps.values() for fp in fps]
    _labels, centroids = cluster_fingerprints(np.array(all_fps), threshold=KNOWN_THRESHOLD)
    return np.array(centroids)


def split_known_unknown(fps, known_centroids):
    """Return (known_list, unknown_list, known_flags) by distance to known centroids."""
    known, unknown, flags = [], [], []
    for fp in fps:
        if len(known_centroids) == 0:
            unknown.append(fp); flags.append(0); continue
        d = np.linalg.norm(known_centroids - fp, axis=1).min()
        if d <= KNOWN_THRESHOLD:
            known.append(fp); flags.append(1)
        else:
            unknown.append(fp); flags.append(0)
    return known, unknown, flags


def pseudo_label(clusters, min_size=PSEUDO_MIN_SIZE):
    """Label clusters >= min_size as unknown1/2/... (by descending size)."""
    big = sorted([(i, len(c)) for i, c in clusters.items() if len(c) >= min_size],
                 key=lambda x: -x[1])
    return {cid: f'unknown{rank}' for rank, (cid, _n) in enumerate(big, start=1)}


def cluster_mapping_accuracy(clusters, cluster_true):
    """Per-cluster dominant-true-protocol accuracy."""
    correct = 0
    total = 0
    per_cluster = {}
    for cid, fps in clusters.items():
        proto_counts = Counter(cluster_true[cid])
        dominant, n_dom = proto_counts.most_common(1)[0]
        correct += n_dom
        total += len(fps)
        per_cluster[int(cid)] = {'size': len(fps), 'dominant_protocol': dominant,
                                 'n_dominant': n_dom}
    return round(correct / max(total, 1), 4), per_cluster


def make_batch(pools, rng, per_source):
    """Sample `per_source` messages from each source pool, return (fps, true_labels)."""
    fps, labels = [], []
    for src_name, pool in pools:
        n = min(per_source, len(pool))
        idx = rng.choice(len(pool), n, replace=False)
        fps += [pool[i] for i in idx]
        labels += [src_name] * n
    order = rng.permutation(len(fps))
    fps = [fps[i] for i in order]
    labels = [labels[i] for i in order]
    return fps, labels


def main():
    fps_by_proto = extract_fingerprints()
    known_fps = {p: fps_by_proto[p] for p in KNOWN_PROTOCOLS if p in fps_by_proto}
    known_centroids = build_known_model(known_fps)
    print(f'Known model: {len(known_centroids)} centroids from '
          f'{sum(len(v) for v in known_fps.values())} known-protocol messages')

    rng = np.random.RandomState(SEED)
    # pools: 'known' (concatenated HTTP/MQTT/SSH/FTP/TLS) + 5 OOD sources
    known_pool = [fp for fps in known_fps.values() for fp in fps]
    pools = [('known', known_pool)] + [(p, fps_by_proto[p])
                                       for p in fps_by_proto if p not in KNOWN_PROTOCOLS]
    per_source = BATCH_SIZE // len(pools)
    batch1_fps, batch1_true = make_batch(pools, rng, per_source)
    print(f'Batch 1: {len(batch1_fps)} msgs '
          f'(known pool={len(known_pool)}, per_source={per_source})')

    known_m, unknown_m, flags = split_known_unknown(batch1_fps, known_centroids)
    unknown_true = [batch1_true[i] for i in range(len(batch1_fps)) if flags[i] == 0]
    print(f'  -> known={len(known_m)} unknown={len(unknown_m)}')

    labels, centroids = cluster_fingerprints(np.array(unknown_m), threshold=KNOWN_THRESHOLD)
    clusters = {}
    cluster_true = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(unknown_m[i])
        cluster_true.setdefault(int(lbl), []).append(unknown_true[i])
    plabels = pseudo_label(clusters)
    print(f'  unknown clusters: {len(clusters)}, pseudo-labeled: {len(plabels)}')
    for cid, name in sorted(plabels.items(), key=lambda x: x[1]):
        dom = Counter(cluster_true[cid]).most_common(1)[0][0]
        print(f'    {name}: {len(clusters[cid])} msgs, dominant true proto={dom}')

    mapping_acc, per_cluster = cluster_mapping_accuracy(clusters, cluster_true)
    print(f'  unknown-cluster mapping accuracy: {mapping_acc:.4f}')

    # ---- Batch 2: another BATCH_SIZE messages ----
    batch1_unknown_m, batch1_unknown_true = unknown_m, unknown_true
    batch1_clusters, batch1_true = clusters, cluster_true
    batch1_centroids = list(centroids)  # per-cluster mean vectors from greedy clustering

    batch2_fps, batch2_true = make_batch(pools, rng, per_source)
    b2_known_m, b2_unknown_m, b2_flags = split_known_unknown(batch2_fps, known_centroids)
    b2_unknown_true = [batch2_true[i] for i in range(len(batch2_fps)) if b2_flags[i] == 0]
    print(f'\nBatch 2: {len(batch2_fps)} msgs '
          f'-> known={len(b2_known_m)} unknown={len(b2_unknown_m)}')

    # Form 1: re-cluster ALL unknown messages from both batches from scratch
    all_unknown = batch1_unknown_m + b2_unknown_m
    all_unknown_true = batch1_unknown_true + b2_unknown_true
    f1_labels, _f1_cent = cluster_fingerprints(np.array(all_unknown), threshold=KNOWN_THRESHOLD)
    f1_clusters, f1_true = {}, {}
    for i, lbl in enumerate(f1_labels):
        f1_clusters.setdefault(int(lbl), []).append(all_unknown[i])
        f1_true.setdefault(int(lbl), []).append(all_unknown_true[i])
    f1_plabels = pseudo_label(f1_clusters)
    f1_acc, f1_per = cluster_mapping_accuracy(f1_clusters, f1_true)

    # Form 2: incremental — add batch2 unknown to batch1 clusters (greedy update)
    inc_clusters = {cid: list(c) for cid, c in batch1_clusters.items()}
    inc_true = {cid: list(t) for cid, t in batch1_true.items()}
    inc_centroids = list(batch1_centroids)
    for fp, true_lbl in zip(b2_unknown_m, b2_unknown_true):
        best, best_d = -1, float('inf')
        for cidx, c in enumerate(inc_centroids):
            d = np.linalg.norm(fp - c)
            if d <= KNOWN_THRESHOLD and d < best_d:
                best_d, best = d, cidx
        if best >= 0:
            inc_clusters[best].append(fp)
            inc_true[best].append(true_lbl)
            n = len(inc_clusters[best])
            inc_centroids[best] = (inc_centroids[best] * (n - 1) + fp) / n
        else:
            new_id = len(inc_clusters)
            inc_clusters[new_id] = [fp]
            inc_true[new_id] = [true_lbl]
            inc_centroids.append(fp)
    inc_plabels = pseudo_label(inc_clusters)
    inc_acc, inc_per = cluster_mapping_accuracy(inc_clusters, inc_true)

    print('=== Batch 2 comparison ===')
    print(f'Form 1 (re-cluster all): n_clusters={len(f1_clusters)} '
          f'pseudo_labels={len(f1_plabels)} mapping={f1_acc:.4f}')
    print(f'Form 2 (incremental):    n_clusters={len(inc_clusters)} '
          f'pseudo_labels={len(inc_plabels)} mapping={inc_acc:.4f}')
    # does incremental discover the same new-protocol labels as re-cluster?
    f1_label_ids = set(plabels.keys())
    inc_label_ids = set(inc_plabels.keys())
    new_f1 = f1_label_ids - set(batch1_clusters.keys())
    new_inc = inc_label_ids - set(batch1_clusters.keys())
    print(f'  new labels beyond batch1: re-cluster={len(new_f1)}, incremental={len(new_inc)}')

    results = {
        'batch1': {
            'total': len(batch1_fps),
            'known_pool': len(known_pool),
            'known_split': len(known_m),
            'unknown': len(unknown_m),
            'n_clusters': len(clusters),
            'pseudo_labels': {name: len(clusters[cid]) for cid, name in plabels.items()},
            'mapping_accuracy': mapping_acc,
            'cluster_dominant': {str(cid): d for cid, d in per_cluster.items()},
        },
        'batch2': {
            'total': len(batch2_fps),
            'known_split': len(b2_known_m),
            'unknown': len(b2_unknown_m),
            'form1': {
                'n_clusters': len(f1_clusters),
                'n_pseudo_labels': len(f1_plabels),
                'pseudo_labels': {name: len(f1_clusters[cid])
                                  for cid, name in f1_plabels.items()},
                'mapping_accuracy': f1_acc,
                'cluster_dominant': {str(cid): d for cid, d in f1_per.items()},
            },
            'form2': {
                'n_clusters': len(inc_clusters),
                'n_pseudo_labels': len(inc_plabels),
                'pseudo_labels': {name: len(inc_clusters[cid])
                                  for cid, name in inc_plabels.items()},
                'mapping_accuracy': inc_acc,
                'cluster_dominant': {str(cid): d for cid, d in inc_per.items()},
            },
            'comparison': {
                'new_labels_recluster': len(new_f1),
                'new_labels_incremental': len(new_inc),
                'incremental_matches_recluster': sorted(
                    [str(f1_plabels[cid]) for cid in new_f1]) == sorted(
                    [str(inc_plabels[cid]) for cid in new_inc]),
            },
        },
    }
    os.makedirs('eval_results/fusion_comparison', exist_ok=True)
    with open('eval_results/fusion_comparison/incremental_unknown_protocol.json', 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Saved: eval_results/fusion_comparison/incremental_unknown_protocol.json')


if __name__ == '__main__':
    main()
